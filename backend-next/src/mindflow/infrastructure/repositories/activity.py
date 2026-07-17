"""SQLAlchemy-backed ActivityRepository for the append-mostly event stream.

Implements heartbeat merge (ADR-002, ADR-007):
  When a new window_snapshot event arrives for the same user with the
  same app_name as the preceding window_snapshot, and the timestamp
  difference is within ``pulsetime_s``, the existing row's duration_s
  is atomically extended rather than inserting a new row.

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

    Args:
        session_factory: Async session maker bound to the application engine.
        pulsetime_s: Heartbeat merge window in seconds. Defaults to
            ``settings.heartbeat_pulsetime_s``.
    """

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

        If the last window_snapshot for this user shares the same
        app_name and falls within ``pulsetime_s``, the existing row's
        duration_s is extended atomically. Otherwise a new row is inserted.

        The entire operation runs inside a single transaction.
        """
        async with self._session_factory() as session, session.begin():
            last = await self._last_window_snapshot(session, event.user_id)

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

        Args:
            user_id: User identifier.
            start: Inclusive start of the time range (timezone-aware UTC).
            end: Inclusive end of the time range (timezone-aware UTC).

        Returns:
            A list of ActivityEvents sorted by timestamp ascending.
        """
        stmt = (
            sa.select(activity_events)
            .where(
                activity_events.c.user_id == user_id,
                activity_events.c.timestamp >= start.isoformat(),
                activity_events.c.timestamp <= end.isoformat(),
            )
            .order_by(activity_events.c.timestamp.asc())
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [_row_to_event(row) for row in result.fetchall()]

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

    async def _last_window_snapshot(
        self,
        session: AsyncSession,
        user_id: int,
    ) -> sa.Row[Any] | None:
        """Find the most recent window_snapshot for *user_id*.

        Args:
            session: Active SQLAlchemy session.
            user_id: User identifier.

        Returns:
            The latest window_snapshot row, or None if none exist.
        """
        stmt = (
            sa.select(activity_events)
            .where(
                activity_events.c.user_id == user_id,
                activity_events.c.event_type == "window_snapshot",
            )
            .order_by(activity_events.c.timestamp.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.fetchone()

    def _should_merge(self, last_row: sa.Row[Any], event: ActivityEvent) -> bool:
        """Determine whether *event* should merge into *last_row*.

        Merge conditions (all must hold):
          1. New event is ``window_snapshot`` type (idle/manual don't merge).
          2. Same ``app_name`` in the snapshot data.
          3. Timestamp difference < ``pulsetime_s``.
        """
        if event.event_type != "window_snapshot":
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
