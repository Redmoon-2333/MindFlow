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
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.config import get_settings
from mindflow.domain.events import ActivityEvent, WindowSnapshot

# Event types eligible for heartbeat merge (manual_tag never merges).
_MERGEABLE_EVENT_TYPES: frozenset[str] = frozenset({"window_snapshot", "idle_change"})

# Keyset page size for query_range — bounds the per-round-trip DB buffer
# on large ranges (e.g. multi-week exports) without changing the return value.
_QUERY_PAGE_SIZE: int = 5000

# ── Table definition (matches migration 0001_create_core_tables) ─────

metadata = sa.MetaData()

activity_events = sa.Table(
    "activity_events",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("timestamp", sa.Text(), nullable=False),
    sa.Column("duration_s", sa.Float(), nullable=False, server_default=sa.text("0.0")),
    sa.Column("data_json", sa.Text(), nullable=False),
    sa.Column(
        "event_type",
        sa.Text(),
        nullable=False,
        server_default=sa.text("'window_snapshot'"),
    ),
    sa.Column(
        "created_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
)


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
    ) -> list[ActivityEvent]:
        """Return events for *user_id* in [*start*, *end*], ordered by time.

        Fetched internally in keyset-paginated chunks of ``_QUERY_PAGE_SIZE``
        rows so a single large range (e.g. a 30-day export) never buffers the
        whole result set in one round-trip. The full list is still assembled
        and returned — this only bounds the per-round-trip DB buffer, not the
        final in-memory size. Ordering is ``(timestamp, id)`` ascending; ``id``
        (UUIDv7) breaks timestamp ties so paging is deterministic (no skipped
        or duplicated rows across page boundaries).

        Args:
            user_id: User identifier.
            start: Inclusive start of the time range (timezone-aware UTC).
            end: Inclusive end of the time range (timezone-aware UTC).

        Returns:
            A list of ActivityEvents sorted by timestamp ascending.
        """
        start_iso = start.isoformat()
        end_iso = end.isoformat()

        events: list[ActivityEvent] = []
        cursor_ts: str | None = None
        cursor_id: str | None = None

        async with self._session_factory() as session:
            while True:
                stmt = sa.select(activity_events).where(
                    activity_events.c.user_id == user_id,
                    activity_events.c.timestamp >= start_iso,
                    activity_events.c.timestamp <= end_iso,
                )
                if cursor_ts is not None:
                    # Keyset predicate: (timestamp, id) > (cursor_ts, cursor_id).
                    stmt = stmt.where(
                        sa.tuple_(
                            activity_events.c.timestamp, activity_events.c.id
                        )
                        > sa.tuple_(cursor_ts, cursor_id)
                    )
                stmt = stmt.order_by(
                    activity_events.c.timestamp.asc(),
                    activity_events.c.id.asc(),
                ).limit(_QUERY_PAGE_SIZE)

                result = await session.execute(stmt)
                rows = result.fetchall()
                if not rows:
                    break

                events.extend(_row_to_event(row) for row in rows)

                if len(rows) < _QUERY_PAGE_SIZE:
                    break
                cursor_ts = rows[-1].timestamp
                cursor_id = rows[-1].id

        return events

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

        try:
            last_data = json.loads(last_row.data_json)
        except (json.JSONDecodeError, AttributeError):
            return False

        last_app = last_data.get("app_name", "")

        if last_app != event.data.app_name:
            return False

        try:
            last_ts = datetime.fromisoformat(last_row.timestamp)
        except (ValueError, AttributeError):
            return False

        diff = (event.timestamp_utc - last_ts).total_seconds()
        return diff < self._pulsetime_s

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
