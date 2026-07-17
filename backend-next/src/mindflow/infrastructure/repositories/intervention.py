"""SQLAlchemy-backed InterventionLog repository.

Stores and queries intervention history for throttle decisions and
effectiveness analysis (Wave 7).

Table schema matches the Alembic migration (0001_create_core_tables):

  intervention_logs:
    id                  TEXT PK (UUIDv7)
    user_id             INTEGER NOT NULL
    triggered_at        TEXT NOT NULL (ISO8601 UTC)
    intervention_type   TEXT NOT NULL
    cbt_technique       TEXT (nullable)
    context_json        TEXT (nullable, JSON blob)
    user_response       TEXT (nullable: "accepted"|"ignored"|"dismissed")
    response_latency_s  REAL (nullable)
    created_at          TEXT NOT NULL (ISO8601 UTC)

All timestamps are stored as ISO8601 text (timezone-aware UTC).
"""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, Protocol

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id


class Clock(Protocol):
    """Minimal clock protocol for injectable time (reused by throttle)."""

    def now(self) -> datetime: ...


class UTCCLock:
    """Production clock — returns datetime.now(UTC)."""

    def now(self) -> datetime:
        return datetime.now(UTC)

# ── Table definition (matches migration 0001_create_core_tables) ─────

metadata = sa.MetaData()

intervention_logs = sa.Table(
    "intervention_logs",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("triggered_at", sa.Text(), nullable=False),
    sa.Column("intervention_type", sa.Text(), nullable=False),
    sa.Column("cbt_technique", sa.Text(), nullable=True),
    sa.Column("context_json", sa.Text(), nullable=True),
    sa.Column("user_response", sa.Text(), nullable=True),
    sa.Column("response_latency_s", sa.Float(), nullable=True),
    sa.Column(
        "created_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
)

ResponseType = Literal["accepted", "ignored", "dismissed"]


class InterventionLogRepository:
    """Intervention history, backed by SQLAlchemy Core + async SQLite.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        clock: Clock | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._clock = clock or UTCCLock()

    # ── Public API ────────────────────────────────────────────────────

    async def log_triggered(
        self,
        user_id: int,
        intervention_type: str,
        cbt_technique: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        intervention_id: str | None = None,
        triggered_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Record an intervention trigger event.

        Args:
            user_id: User identifier.
            intervention_type: One of the four intervention types.
            cbt_technique: Optional CBT technique that informed this intervention.
            context: Optional JSON-serialisable context (e.g. current assessment data).
            intervention_id: Override the auto-generated ID (for testing).
            triggered_at: Override the timestamp (for testing).

        Returns:
            The inserted row as a dict.
        """
        row_id = intervention_id or new_id()
        ts = triggered_at or self._clock.now()

        row = {
            "id": row_id,
            "user_id": user_id,
            "triggered_at": ts.isoformat(),
            "intervention_type": intervention_type,
            "cbt_technique": cbt_technique,
            "context_json": json.dumps(context, ensure_ascii=False) if context else None,
            "user_response": None,
            "response_latency_s": None,
        }

        async with self._session_factory() as session, session.begin():
            await session.execute(intervention_logs.insert().values(**row))

        result = {**row, "created_at": ts.isoformat()}
        # Parse context_json back to dict for API consistency
        ctx = result.get("context_json")
        if isinstance(ctx, str):
            with suppress(json.JSONDecodeError, TypeError):
                result["context_json"] = json.loads(ctx)
        return result

    async def update_response(
        self,
        intervention_id: str,
        user_response: ResponseType,
        latency_s: float = 0.0,
    ) -> dict[str, Any] | None:
        """Update the user's response to a previously triggered intervention.

        Args:
            intervention_id: The intervention's UUIDv7 string.
            user_response: One of "accepted", "ignored", "dismissed".
            latency_s: Seconds between trigger and response.

        Returns:
            The updated row dict, or None if the intervention wasn't found.
        """
        stmt = (
            sa.update(intervention_logs)
            .where(intervention_logs.c.id == intervention_id)
            .values(user_response=user_response, response_latency_s=latency_s)
        )

        async with self._session_factory() as session, session.begin():
            result = await session.execute(stmt)
            # CursorResult.rowcount is the number of rows matched
            rowcount: int = result.rowcount
            if rowcount == 0:
                return None

            # Fetch the updated row
            select_stmt = sa.select(intervention_logs).where(
                intervention_logs.c.id == intervention_id
            )
            fetch = await session.execute(select_stmt)
            row = fetch.fetchone()

        return _row_to_dict(row) if row is not None else None

    async def count_today(self, user_id: int) -> int:
        """Return the number of interventions triggered today for *user_id*."""
        today_start = self._clock.now().replace(hour=0, minute=0, second=0, microsecond=0)

        stmt = (
            sa.select(sa.func.count())
            .select_from(intervention_logs)
            .where(
                intervention_logs.c.user_id == user_id,
                intervention_logs.c.triggered_at >= today_start.isoformat(),
            )
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            count: int = result.scalar() or 0
            return count

    async def count_today_by_type(self, user_id: int, intervention_type: str) -> int:
        """Return count of today's interventions of a specific type."""
        today_start = self._clock.now().replace(hour=0, minute=0, second=0, microsecond=0)

        stmt = (
            sa.select(sa.func.count())
            .select_from(intervention_logs)
            .where(
                intervention_logs.c.user_id == user_id,
                intervention_logs.c.intervention_type == intervention_type,
                intervention_logs.c.triggered_at >= today_start.isoformat(),
            )
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            count: int = result.scalar() or 0
            return count

    async def ignore_rate_7d(self, user_id: int) -> float:
        """Compute the 7-day ignore rate for *user_id*.

        Returns:
            Fraction of interventions in the last 7 days that were IGNORED
            (not yet responded to). Returns 0.0 if there are no interventions
            in the window.
        """
        cutoff = self._clock.now() - timedelta(days=7)

        stmt = sa.select(
            sa.func.count().label("total"),
            sa.func.sum(
                sa.case(
                    (intervention_logs.c.user_response == "ignored", 1),
                    else_=0,
                )
            ).label("ignored"),
        ).where(
            intervention_logs.c.user_id == user_id,
            intervention_logs.c.triggered_at >= cutoff.isoformat(),
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()
            if row is None:
                return 0.0
            total: int = row.total or 0
            if total == 0:
                return 0.0
            ignored: int = row.ignored or 0
            return ignored / total

    async def query_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        """Return intervention logs in [*start*, *end*], ordered by triggered_at.

        Args:
            user_id: User identifier.
            start: Inclusive start datetime (timezone-aware UTC).
            end: Inclusive end datetime (timezone-aware UTC).

        Returns:
            A list of intervention log dicts sorted by triggered_at ascending.
        """
        stmt = (
            sa.select(intervention_logs)
            .where(
                intervention_logs.c.user_id == user_id,
                intervention_logs.c.triggered_at >= start.isoformat(),
                intervention_logs.c.triggered_at <= end.isoformat(),
            )
            .order_by(intervention_logs.c.triggered_at.asc())
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.fetchall()]

    async def get_by_id(self, intervention_id: str) -> dict[str, Any] | None:
        """Return a single intervention log by ID, or None."""
        stmt = sa.select(intervention_logs).where(intervention_logs.c.id == intervention_id)

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()

        return _row_to_dict(row) if row is not None else None

    async def query_range_by_date(
        self,
        user_id: int,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        """Return intervention logs in [*start_date*, *end_date*] (date range).

        Args:
            user_id: User identifier.
            start_date: Inclusive start date.
            end_date: Inclusive end date.

        Returns:
            A list of intervention log dicts sorted by triggered_at ascending.
        """
        start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC)
        end_dt = (
            datetime(end_date.year, end_date.month, end_date.day, tzinfo=UTC)
            + timedelta(days=1)
            - timedelta(seconds=1)
        )
        return await self.query_range(user_id, start_dt, end_dt)

    def __repr__(self) -> str:
        return "<InterventionLogRepository>"


# ── Serialisation helpers ─────────────────────────────────────────────


def _row_to_dict(row: sa.Row[Any]) -> dict[str, Any]:
    """Convert a database row (``intervention_logs``) to a plain dict."""
    context = None
    if row.context_json:
        try:
            context = json.loads(row.context_json)
        except (json.JSONDecodeError, TypeError):
            context = None

    return {
        "id": row.id,
        "user_id": row.user_id,
        "triggered_at": row.triggered_at,
        "intervention_type": row.intervention_type,
        "cbt_technique": row.cbt_technique,
        "context_json": context,
        "user_response": row.user_response,
        "response_latency_s": row.response_latency_s,
        "created_at": row.created_at,
    }
