"""SQLAlchemy-backed ActivityRepository for the append-mostly event stream.

Implements heartbeat merge (ADR-002, ADR-007):
  When a new window_snapshot event arrives for the same user with the
  same app_name as the preceding window_snapshot, and the timestamp
  difference is within ``pulsetime_s``, the existing row's duration_s
  is atomically extended rather than inserting a new row. The same
  merge applies to consecutive idle_change events (overnight idle would
  otherwise insert one row per collector tick, inflating the table).

Table schema matches the Alembic migration (0001_create_core_tables).
All timestamps are stored as ISO8601 text (timezone-aware UTC).
Data payload (WindowSnapshot) is stored as JSON text in data_json.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.config import get_settings
from mindflow.domain.events import ActivityEvent, WindowSnapshot
from mindflow.infrastructure.schema import activity_events as activity_events

# Backward-compat alias: tests and importers used to do
# ``activity_events.metadata.create_all``. The table now lives in the shared
# schema module (architecture plan D); expose the same metadata here so no
# consumer needs to change.
metadata = activity_events.metadata

# Event types eligible for heartbeat merge (manual_tag never merges).
_MERGEABLE_EVENT_TYPES: frozenset[str] = frozenset({"window_snapshot", "idle_change"})

# Keyset page size for query_range — bounds the per-round-trip DB buffer
# on large ranges (e.g. multi-week exports) without changing the return value.
_QUERY_PAGE_SIZE: int = 5000

# ── Repository ────────────────────────────────────────────────────────


class SQLAlchemyActivityRepository:
    """Activity event repository backed by SQLAlchemy Core + async SQLite.

    Satisfies the ``ActivityRepository`` protocol via structural typing
    (no explicit subclassing required). See ``repositories/base.py``.

    Args:
        session_factory: Async session maker bound to the application engine.
        pulsetime_s: Heartbeat merge window in seconds. Defaults to
            ``settings.heartbeat_pulsetime_s``.
    """

    # Static assertion: SQLAlchemyActivityRepository satisfies the
    # ActivityRepository protocol. Uncomment to verify at import time:
    # from mindflow.infrastructure.repositories.base import ActivityRepository
    # _: type[ActivityRepository] = SQLAlchemyActivityRepository

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        pulsetime_s: int | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._pulsetime_s = pulsetime_s or get_settings().heartbeat_pulsetime_s

    # ── Public API ────────────────────────────────────────────────────

    async def append_event(self, event: ActivityEvent) -> None:
        """Persist an activity event with heartbeat merge.

        If the last mergeable event of the same ``event_type`` for this
        user shares the same app_name and falls within ``pulsetime_s``,
        the existing row's duration_s is extended atomically. Otherwise
        a new row is inserted. Both window_snapshot and idle_change events
        merge (against their own kind); manual_tag never merges.

        The entire operation runs inside a single transaction.
        """
        async with self._session_factory() as session, session.begin():
            last = await self._last_mergeable_event(
                session, event.user_id, event.event_type
            )

            if last is not None and self._should_merge(last, event):
                await session.execute(
                    sa.update(activity_events)
                    .where(activity_events.c.id == last.id)
                    .values(
                        duration_s=activity_events.c.duration_s + event.duration_s
                    )
                )
                return

            await session.execute(
                activity_events.insert().values(
                    id=event.id,
                    user_id=event.user_id,
                    timestamp=event.timestamp_utc.isoformat(),
                    duration_s=event.duration_s,
                    data_json=json.dumps(event.data.to_dict()),
                    event_type=event.event_type,
                )
            )

    async def query_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
        *,
        limit: int | None = None,
        offset: int | None = None,
        descending: bool = False,
        cursor: tuple[str, str] | None = None,
    ) -> list[ActivityEvent]:
        """Return events for *user_id* in [*start*, *end*], ordered by time.

        Paginated reads use one SQL query.  A ``cursor`` applies keyset
        pagination on ``(timestamp, id)`` and takes precedence over OFFSET.
        Full-range reads consume bounded keyset chunks internally.
        """
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        if limit is not None or offset is not None or cursor is not None:
            return await self._query_range_paginated(
                user_id,
                start_iso,
                end_iso,
                limit=limit,
                offset=offset,
                descending=descending,
                cursor=cursor,
            )

        events: list[ActivityEvent] = []
        async for chunk in self.iter_range_chunks(user_id, start, end):
            events.extend(chunk)
        return events

    async def _query_range_paginated(
        self,
        user_id: int,
        start_iso: str,
        end_iso: str,
        *,
        limit: int | None,
        offset: int | None,
        descending: bool,
        cursor: tuple[str, str] | None,
    ) -> list[ActivityEvent]:
        """Single-query OFFSET/LIMIT or keyset fetch for paginated reads."""
        order = (
            activity_events.c.timestamp.desc(),
            activity_events.c.id.desc(),
        ) if descending else (
            activity_events.c.timestamp.asc(),
            activity_events.c.id.asc(),
        )

        stmt = sa.select(activity_events).where(
            activity_events.c.user_id == user_id,
            activity_events.c.timestamp >= start_iso,
            activity_events.c.timestamp <= end_iso,
        )
        if cursor is not None:
            cursor_value = sa.tuple_(cursor[0], cursor[1])  # type: ignore[arg-type]
            row_value = sa.tuple_(activity_events.c.timestamp, activity_events.c.id)
            stmt = stmt.where(
                row_value < cursor_value if descending else row_value > cursor_value
            )
        elif offset is not None:
            stmt = stmt.offset(offset)
        stmt = stmt.order_by(*order)
        if limit is not None:
            stmt = stmt.limit(limit)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [_row_to_event(row) for row in result.fetchall()]

    async def iter_range_chunks(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
        *,
        chunk_size: int = _QUERY_PAGE_SIZE,
    ) -> AsyncIterator[list[ActivityEvent]]:
        """Yield ascending event chunks without materialising the full range."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        start_iso = start.isoformat()
        end_iso = end.isoformat()
        cursor: tuple[str, str] | None = None

        while True:
            stmt = sa.select(activity_events).where(
                activity_events.c.user_id == user_id,
                activity_events.c.timestamp >= start_iso,
                activity_events.c.timestamp <= end_iso,
            )
            if cursor is not None:
                stmt = stmt.where(
                    sa.tuple_(activity_events.c.timestamp, activity_events.c.id)
                    > sa.tuple_(cursor[0], cursor[1])  # type: ignore[arg-type]
                )
            stmt = stmt.order_by(
                activity_events.c.timestamp.asc(),
                activity_events.c.id.asc(),
            ).limit(chunk_size)

            # Release the SQLite connection before yielding to a potentially
            # slow client so exports do not hold long-lived read transactions.
            async with self._session_factory() as session:
                rows = (await session.execute(stmt)).fetchall()
            if not rows:
                break
            yield [_row_to_event(row) for row in rows]
            if len(rows) < chunk_size:
                break
            cursor = (rows[-1].timestamp, rows[-1].id)

    async def query_overlapping_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> list[ActivityEvent]:
        events = await self.query_range(user_id, start, end)
        previous = await self.last_event_before(user_id, start)
        if previous is not None:
            previous_end = previous.timestamp_utc + timedelta(
                seconds=max(0.0, previous.duration_s)
            )
            if previous_end > start:
                events.insert(0, previous)

        clipped: list[ActivityEvent] = []
        for event in events:
            event_end = event.timestamp_utc + timedelta(
                seconds=max(0.0, event.duration_s)
            )
            overlap_start = max(event.timestamp_utc, start)
            overlap_end = min(event_end, end)
            if overlap_end <= overlap_start:
                continue
            clipped_snapshot = replace(event.data, timestamp_utc=overlap_start)
            clipped.append(
                replace(
                    event,
                    timestamp_utc=overlap_start,
                    duration_s=(overlap_end - overlap_start).total_seconds(),
                    data=clipped_snapshot,
                )
            )
        return clipped

    async def count_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> int:
        """Return the total number of events for *user_id* in [*start*, *end*].

        Args:
            user_id: User identifier.
            start: Inclusive start of the time range (timezone-aware UTC).
            end: Inclusive end of the time range (timezone-aware UTC).

        Returns:
            The count of matching events.
        """
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        stmt = sa.select(sa.func.count()).select_from(activity_events).where(
            activity_events.c.user_id == user_id,
            activity_events.c.timestamp >= start_iso,
            activity_events.c.timestamp <= end_iso,
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one()

    async def last_event_before(
        self,
        user_id: int,
        timestamp: datetime,
    ) -> ActivityEvent | None:
        stmt = (
            sa.select(activity_events)
            .where(
                activity_events.c.user_id == user_id,
                activity_events.c.timestamp < timestamp.isoformat(),
            )
            .order_by(activity_events.c.timestamp.desc(), activity_events.c.id.desc())
            .limit(1)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()
            return _row_to_event(row) if row is not None else None

    async def last_event(self, user_id: int) -> ActivityEvent | None:
        """Return the most recent event for *user_id*, or None.

        Args:
            user_id: User identifier.

        Returns:
            The latest ActivityEvent by timestamp, or None if no events exist.
        """
        stmt = (
            sa.select(activity_events)
            .where(activity_events.c.user_id == user_id)
            .order_by(activity_events.c.timestamp.desc())
            .limit(1)
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()

        return _row_to_event(row) if row is not None else None

    async def get_activity_summary(
        self, user_id: int,
    ) -> dict[str, Any]:
        """Return aggregate activity-event stats in a single query.

        Returns dict with keys: ``total_events``, ``oldest_timestamp``,
        ``newest_timestamp``, ``coverage_days``.  All timestamp values are
        ISO-8601 strings; ``None`` when no events exist.
        """
        stmt = sa.select(
            sa.func.count(),
            sa.func.min(activity_events.c.timestamp),
            sa.func.max(activity_events.c.timestamp),
        ).where(activity_events.c.user_id == user_id)

        async with self._session_factory() as session:
            row = (await session.execute(stmt)).fetchone()

        if row is None or row[0] == 0:
            return {
                "total_events": 0,
                "oldest_timestamp": None,
                "newest_timestamp": None,
                "coverage_days": 0,
            }

        total: int = int(row[0])
        oldest: str | None = row[1]
        newest: str | None = row[2]
        coverage = 0
        if oldest is not None and newest is not None:
            try:
                od = datetime.fromisoformat(oldest)
                nd = datetime.fromisoformat(newest)
                coverage = (nd.date() - od.date()).days + 1
            except (ValueError, TypeError):
                coverage = 0

        return {
            "total_events": total,
            "oldest_timestamp": oldest,
            "newest_timestamp": newest,
            "coverage_days": coverage,
        }

    async def compact_history(self, user_id: int = 1) -> dict[str, int]:
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.select(activity_events)
                .where(activity_events.c.user_id == user_id)
                .order_by(activity_events.c.timestamp.asc())
            )
            rows = result.fetchall()
            before = len(rows)
            if before < 2:
                return {"before": before, "after": before, "removed": 0}

            segments: list[dict[str, Any]] = []
            for row in rows:
                data = json.loads(row.data_json)
                timestamp = datetime.fromisoformat(row.timestamp)
                duration_s = max(0.0, float(row.duration_s))
                context = (
                    row.event_type,
                    data.get("app_name", ""),
                    data.get("process_name", ""),
                    data.get("window_title", ""),
                    bool(data.get("is_idle", False)),
                )
                if segments:
                    previous = segments[-1]
                    previous_end = previous["timestamp"] + timedelta(
                        seconds=previous["duration_s"]
                    )
                    gap_s = (timestamp - previous_end).total_seconds()
                    if (
                        previous["context"] == context
                        and -self._pulsetime_s <= gap_s <= self._pulsetime_s
                    ):
                        previous["duration_s"] += duration_s
                        previous["delete_ids"].append(row.id)
                        continue
                segments.append({
                    "id": row.id,
                    "timestamp": timestamp,
                    "duration_s": duration_s,
                    "context": context,
                    "delete_ids": [],
                })

            for segment in segments:
                await session.execute(
                    sa.update(activity_events)
                    .where(activity_events.c.id == segment["id"])
                    .values(duration_s=segment["duration_s"])
                )
                if segment["delete_ids"]:
                    await session.execute(
                        sa.delete(activity_events).where(
                            activity_events.c.id.in_(segment["delete_ids"])
                        )
                    )

            after = len(segments)
            return {"before": before, "after": after, "removed": before - after}


    # ── Internal helpers ──────────────────────────────────────────────

    async def _last_mergeable_event(
        self,
        session: AsyncSession,
        user_id: int,
        event_type: str,
    ) -> sa.Row[Any] | None:
        """Find the most recent event of *event_type* for *user_id*.

        Merge candidates are looked up per event_type so that, e.g., an
        idle_change between two window_snapshots does not become the merge
        target for the next window_snapshot (and vice versa).

        Args:
            session: Active SQLAlchemy session.
            user_id: User identifier.
            event_type: The incoming event's type (window_snapshot / idle_change).

        Returns:
            The latest row of that event_type, or None if none exist.
        """
        stmt = (
            sa.select(activity_events)
            .where(
                activity_events.c.user_id == user_id,
                activity_events.c.event_type == event_type,
            )
            .order_by(activity_events.c.timestamp.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.fetchone()

    def _should_merge(self, last_row: sa.Row[Any], event: ActivityEvent) -> bool:
        """Determine whether *event* should merge into *last_row*.

        Merge conditions (all must hold):
          1. New event is a mergeable type (window_snapshot or idle_change;
             manual_tag never merges).
          2. Same ``event_type`` as ``last_row`` (window/idle don't cross-merge).
          3. Same ``app_name`` in the snapshot data.
          4. Timestamp difference < ``pulsetime_s``.
        """
        if event.event_type not in _MERGEABLE_EVENT_TYPES:
            return False

        if last_row.event_type != event.event_type:
            return False

        context_matches = (
            (last_row.app_name or "") == event.data.app_name
            and (last_row.process_name or "") == event.data.process_name
            and (last_row.window_title or "") == event.data.window_title
            and bool(last_row.is_idle) == event.data.is_idle
        )
        if not context_matches:
            return False

        try:
            last_ts = datetime.fromisoformat(last_row.timestamp)
            last_end = last_ts + timedelta(seconds=max(0.0, float(last_row.duration_s)))
        except (TypeError, ValueError, AttributeError):
            return False

        gap_s = (event.timestamp_utc - last_end).total_seconds()
        return -self._pulsetime_s <= gap_s <= self._pulsetime_s

    def __repr__(self) -> str:
        return f"<SQLAlchemyActivityRepository pulsetime={self._pulsetime_s}s>"


# ── Serialisation helpers ─────────────────────────────────────────────


def _row_to_event(row: sa.Row[Any]) -> ActivityEvent:
    """Convert a database row (``activity_events``) to an ``ActivityEvent``."""
    data_dict = json.loads(row.data_json)
    return ActivityEvent(
        id=row.id,
        user_id=row.user_id,
        timestamp_utc=datetime.fromisoformat(row.timestamp),
        duration_s=row.duration_s,
        event_type=row.event_type,
        data=WindowSnapshot.from_dict(data_dict),
    )
