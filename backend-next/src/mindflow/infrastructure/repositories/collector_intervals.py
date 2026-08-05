"""SQLAlchemy-backed CollectorInterval repository.

Persists one row per CollectorService run: ``open()`` records the start,
``close()`` records terminal facts on that exact row by ``id``, and
``list_by_user*`` return user-scoped history ordered by ``started_at``.

All timestamps are stored as ISO8601 text (timezone-aware UTC) — matching
the intervention / scheduled-job repositories.

Table schema matches the Alembic migration 0021_create_collector_intervals:

  collector_intervals:
    id           TEXT PK (UUIDv7)
    user_id      INTEGER NOT NULL
    started_at   TEXT NOT NULL (ISO8601 UTC)
    ended_at     TEXT (nullable, ISO8601 UTC)
    reason       TEXT (nullable)
    manual_stop  INTEGER NOT NULL (0/1 flag)
    failure      INTEGER NOT NULL (0/1 flag)
    sleep        INTEGER NOT NULL (0/1 flag)
    last_error   TEXT (nullable)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id
from mindflow.infrastructure.schema import collector_intervals
from mindflow.ports import CollectorIntervalRecord

_FLAG_ON = 1
_FLAG_OFF = 0


class CollectorIntervalsRepository:
    """Runtime collector-interval lifecycle store.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    # ── Public API ────────────────────────────────────────────────────

    async def open(
        self,
        user_id: int,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord:
        """Open a new collector interval — inserts exactly one row.

        Args:
            user_id: User identifier the interval belongs to.
            reason: Optional reason attached at open time.
            manual_stop / failure / sleep: Lifecycle flags at open time.
            now: Override the ``started_at`` timestamp (for testing).

        Returns:
            The inserted interval record with ``ended_at=None``.
        """
        interval_id = new_id()
        started_at = (now or datetime.now(UTC)).isoformat()
        row = {
            "id": interval_id,
            "user_id": user_id,
            "started_at": started_at,
            "ended_at": None,
            "reason": reason,
            "manual_stop": _FLAG_ON if manual_stop else _FLAG_OFF,
            "failure": _FLAG_ON if failure else _FLAG_OFF,
            "sleep": _FLAG_ON if sleep else _FLAG_OFF,
            "last_error": None,
        }
        async with self._session_factory() as session, session.begin():
            await session.execute(collector_intervals.insert().values(**row))
        return CollectorIntervalRecord(
            id=interval_id,
            user_id=user_id,
            started_at=started_at,
            ended_at=None,
            reason=reason,
            manual_stop=manual_stop,
            failure=failure,
            sleep=sleep,
            last_error=None,
        )

    async def close(
        self,
        interval_id: str,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord | None:
        """Close an open collector interval by ``interval_id``.

        Idempotent: only an interval with ``ended_at IS NULL`` is updated,
        so a second close for the same id returns the existing record
        without rewriting its terminal facts.

        Args:
            interval_id: The interval id returned by ``open()``.
            reason / manual_stop / failure / sleep / last_error: Terminal
                facts recorded on the first successful close.
            now: Override the ``ended_at`` timestamp (for testing).

        Returns:
            The interval record (closed on first call, unchanged on later
            calls), or ``None`` if no interval with that id exists.
        """
        terminal = {
            "ended_at": (now or datetime.now(UTC)).isoformat(),
            "reason": reason,
            "manual_stop": _FLAG_ON if manual_stop else _FLAG_OFF,
            "failure": _FLAG_ON if failure else _FLAG_OFF,
            "sleep": _FLAG_ON if sleep else _FLAG_OFF,
            "last_error": last_error,
        }
        async with self._session_factory() as session, session.begin():
            update = (
                sa.update(collector_intervals)
                .where(
                    collector_intervals.c.id == interval_id,
                    collector_intervals.c.ended_at.is_(None),
                )
                .values(**terminal)
                .returning(*collector_intervals.c)
            )
            result = await session.execute(update)
            row = result.fetchone()
            if row is not None:
                return _row_to_record(row)
            # No row matched the guarded update: either the id is unknown
            # or the interval was already closed. Re-read to distinguish —
            # an already-closed interval is returned unchanged.
            existing = await session.execute(
                sa.select(collector_intervals).where(
                    collector_intervals.c.id == interval_id
                )
            )
            row = existing.fetchone()
            return _row_to_record(row) if row is not None else None

    async def list_by_user(
        self, user_id: int, *, limit: int = 100
    ) -> list[CollectorIntervalRecord]:
        """Return the most-recent ``limit`` intervals for *user_id*.

        Ordered by ``started_at`` descending (newest first).
        """
        stmt = (
            sa.select(collector_intervals)
            .where(collector_intervals.c.user_id == user_id)
            .order_by(collector_intervals.c.started_at.desc())
            .limit(limit)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [_row_to_record(row) for row in result.fetchall()]

    async def list_by_user_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> list[CollectorIntervalRecord]:
        """Return intervals for *user_id* in [*start*, *end*].

        Args:
            user_id: User identifier.
            start: Inclusive start datetime (timezone-aware UTC).
            end: Inclusive end datetime (timezone-aware UTC).

        Returns:
            A list of interval records sorted by ``started_at`` ascending.
        """
        stmt = (
            sa.select(collector_intervals)
            .where(
                collector_intervals.c.user_id == user_id,
                collector_intervals.c.started_at >= start.isoformat(),
                collector_intervals.c.started_at <= end.isoformat(),
            )
            .order_by(collector_intervals.c.started_at.asc())
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [_row_to_record(row) for row in result.fetchall()]


# ── Serialisation helpers ─────────────────────────────────────────────


def _row_to_record(row: Any) -> CollectorIntervalRecord:
    """Convert an ``sa.Row`` from a select/returning to an interval record."""
    mapping = row._mapping
    return CollectorIntervalRecord(
        id=mapping["id"],
        user_id=mapping["user_id"],
        started_at=mapping["started_at"],
        ended_at=mapping["ended_at"],
        reason=mapping["reason"],
        manual_stop=bool(mapping["manual_stop"]),
        failure=bool(mapping["failure"]),
        sleep=bool(mapping["sleep"]),
        last_error=mapping["last_error"],
    )
