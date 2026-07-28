"""SQLAlchemy-backed InterventionLog repository.

Stores and queries intervention history for throttle decisions and
effectiveness analysis (Wave 7).

Table schema matches the Alembic migrations (0001-0005):

  intervention_logs:
    id                  TEXT PK (UUIDv7)
    user_id             INTEGER NOT NULL
    triggered_at        TEXT NOT NULL (ISO8601 UTC)
    intervention_type   TEXT NOT NULL
    cbt_technique       TEXT (nullable)
    context_json        TEXT (nullable, JSON blob)
    user_response       TEXT (nullable: "accepted"|"ignored"|"dismissed")
    response_latency_s  REAL (nullable)
    feedback_rating     TEXT (nullable: "helpful"|"neutral"|"annoying")
    feedback_comment    TEXT (nullable)
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
from mindflow.domain.intervention import ThrottleStats
from mindflow.infrastructure.schema import intervention_logs


class Clock(Protocol):
    """Minimal clock protocol for injectable time (reused by throttle)."""

    def now(self) -> datetime: ...


class UTCCLock:
    """Production clock — returns datetime.now(UTC)."""

    def now(self) -> datetime:
        return datetime.now(UTC)

ResponseType = Literal["accepted", "ignored", "dismissed"]
FeedbackRating = Literal["helpful", "neutral", "annoying"]


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
            .returning(*intervention_logs.c)
        )

        async with self._session_factory() as session, session.begin():
            result = await session.execute(stmt)
            row = result.fetchone()

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

    async def update_feedback(
        self,
        intervention_id: str,
        rating: FeedbackRating,
        comment: str | None = None,
    ) -> dict[str, Any] | None:
        """Update the user's feedback on an intervention.

        Args:
            intervention_id: The intervention's UUIDv7 string.
            rating: One of "helpful", "neutral", "annoying".
            comment: Optional free-text comment.

        Returns:
            The updated row dict, or None if the intervention wasn't found.
        """
        stmt = (
            sa.update(intervention_logs)
            .where(intervention_logs.c.id == intervention_id)
            .values(feedback_rating=rating, feedback_comment=comment)
            .returning(*intervention_logs.c)
        )

        async with self._session_factory() as session, session.begin():
            result = await session.execute(stmt)
            row = result.fetchone()

        return _row_to_dict(row) if row is not None else None

    async def annoying_count_7d_by_type(self, user_id: int, intervention_type: str) -> int:
        """Count "annoying" feedback ratings for a type in the last 7 days.

        Used by the throttle to reduce frequency for disliked intervention types.
        """
        cutoff = self._clock.now() - timedelta(days=7)

        stmt = sa.select(sa.func.count()).select_from(intervention_logs).where(
            intervention_logs.c.user_id == user_id,
            intervention_logs.c.intervention_type == intervention_type,
            intervention_logs.c.feedback_rating == "annoying",
            intervention_logs.c.triggered_at >= cutoff.isoformat(),
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            count: int = result.scalar() or 0
            return count

    async def get_throttle_stats(
        self,
        user_id: int,
        intervention_type: str,
        *,
        now: datetime,
        today_start: datetime,
        cutoff_7d: datetime,
        cooldown_lower_bound: datetime,
    ) -> ThrottleStats:
        """Single-query aggregate for throttle decisions (P2-1 optimisation).

        All time boundaries are provided by the caller so that *now* is the
        single authoritative snapshot.  The method itself does not access
        the clock — boundaries are fully explicit.

        The outer WHERE clause uses ``query_start = min(cutoff_7d,
        cooldown_lower_bound, today_start)`` so that **every** row relevant
        to **any** sub-aggregate is scanned.  Each aggregate (ignore rate,
        today counts, annoying count, cooldown last-triggered) then applies
        its own exact boundary via ``CASE WHEN``, preserving the same
        per-method semantics as the original five separate queries.

        Args:
            user_id: User identifier.
            intervention_type: The requested intervention type.
            now: Current instant (cooldown upper bound, inclusive via
                ``<= now`` matching old ``query_range``).
            today_start: Start of calendar day (UTC).
            cutoff_7d: Lower bound for the 7-day window
                (ignore rate, annoying count).
            cooldown_lower_bound: Lower bound for the cooldown window
                (matches old ``query_range(start=…)`` call).

        Returns:
            ``ThrottleStats`` — all counts default to 0; ``last_triggered_at``
            is ``None`` when no row exists within the cooldown window.
        """
        # Earliest instant needed by any sub-aggregate.  When cooldown_h is
        # larger than 3.5 days, cooldown_lower_bound can be older than the
        # 7-day window — the outer scan must include those rows so the
        # cooldown CASE can see them.
        query_start = min(today_start, cutoff_7d, cooldown_lower_bound)

        c = intervention_logs.c

        stmt = (
            sa.select(
                # ── 7-day window (denominator + numerator) ─────────────
                sa.func.sum(
                    sa.case(
                        (c.triggered_at >= cutoff_7d.isoformat(), 1),
                        else_=0,
                    ),
                ).label("total_7d"),
                sa.func.sum(
                    sa.case(
                        (
                            sa.and_(
                                c.triggered_at >= cutoff_7d.isoformat(),
                                c.user_response == "ignored",
                            ),
                            1,
                        ),
                        else_=0,
                    ),
                ).label("ignored_7d"),
                # ── Today counts ───────────────────────────────────────
                sa.func.sum(
                    sa.case(
                        (c.triggered_at >= today_start.isoformat(), 1),
                        else_=0,
                    ),
                ).label("today_count"),
                sa.func.sum(
                    sa.case(
                        (
                            sa.and_(
                                c.triggered_at >= today_start.isoformat(),
                                c.intervention_type == intervention_type,
                            ),
                            1,
                        ),
                        else_=0,
                    ),
                ).label("today_count_by_type"),
                # ── Annoying count (7-day window + type + rating) ──────
                sa.func.sum(
                    sa.case(
                        (
                            sa.and_(
                                c.triggered_at >= cutoff_7d.isoformat(),
                                c.intervention_type == intervention_type,
                                c.feedback_rating == "annoying",
                            ),
                            1,
                        ),
                        else_=0,
                    ),
                ).label("annoying_count_7d_by_type"),
                # ── Cooldown last-triggered (exact old query_range set) ─
                sa.func.max(
                    sa.case(
                        (
                            sa.and_(
                                c.triggered_at >= cooldown_lower_bound.isoformat(),
                                c.triggered_at <= now.isoformat(),
                            ),
                            c.triggered_at,
                        ),
                        else_=None,
                    ),
                ).label("last_triggered_at"),
            )
            .select_from(intervention_logs)
            .where(
                c.user_id == user_id,
                c.triggered_at >= query_start.isoformat(),
            )
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()

        if row is None:
            return ThrottleStats(
                ignore_rate=0.0,
                today_count=0,
                last_triggered_at=None,
                annoying_count_by_type=0,
                today_count_by_type=0,
            )

        total_7d: int = row.total_7d or 0
        ignored_7d: int = row.ignored_7d or 0
        ignore_rate = (ignored_7d / total_7d) if total_7d > 0 else 0.0

        return ThrottleStats(
            ignore_rate=ignore_rate,
            today_count=row.today_count or 0,
            last_triggered_at=row.last_triggered_at,
            annoying_count_by_type=row.annoying_count_7d_by_type or 0,
            today_count_by_type=row.today_count_by_type or 0,
        )

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
        "feedback_rating": row.feedback_rating,
        "feedback_comment": row.feedback_comment,
        "created_at": row.created_at,
    }
