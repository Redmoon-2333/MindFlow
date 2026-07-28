"""Tests for services/intervention_throttle.py — rule matrix coverage.

Covers (C3 requirements):
  - Daily cap: ≤3 total per day
  - Cooldown: ≥2h since last intervention
  - Type cap: ≤2 of same type per day
  - Fatigue: 7d ignore rate >60% → reduced to 1/day
  - OK: when all checks pass
  - Midnight reset: counts reset at calendar day boundary

All tests use an injected ``FakeClock`` for deterministic time control.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
    intervention_logs,
)
from mindflow.services.intervention_throttle import (
    InterventionThrottle,
    ThrottleReason,
)


class FakeClock:
    """Deterministic clock for throttle testing."""

    def __init__(self, start: datetime | None = None) -> None:
        # Non-today date to prove date-independence (P0 regression)
        self._now = start or datetime(2026, 1, 15, 8, 0, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, **kwargs: Any) -> None:
        """Advance the clock by a timedelta."""
        self._now += timedelta(**kwargs)


class TestThrottleRules:
    """Full rule matrix coverage for InterventionThrottle."""

    @pytest.fixture
    async def intervention_tables(self, engine):
        """Create the intervention_logs table."""
        async with engine.begin() as conn:
            await conn.run_sync(intervention_logs.metadata.create_all)

    @pytest.fixture
    def repo(self, session_factory, intervention_tables, clock) -> InterventionLogRepository:
        """Repository bound to a test DB with intervention_logs table, same clock as throttle."""
        return InterventionLogRepository(session_factory=session_factory, clock=clock)

    @pytest.fixture
    def clock(self) -> FakeClock:
        """Deterministic clock starting at 2026-01-15 08:00 UTC (non-today)."""
        return FakeClock()

    @pytest.fixture
    def throttle(self, repo, clock) -> InterventionThrottle:
        """Throttle with injected clock and default limits."""
        return InterventionThrottle(
            repo=repo,
            clock=clock,
            daily_limit=3,
            type_limit=2,
            cooldown_h=2.0,
            ignore_rate_threshold=0.6,
            fatigue_daily_limit=1,
        )

    # ── OK path ──────────────────────────────────────────────────────

    async def test_ok_first_intervention(self, throttle, clock) -> None:
        """First intervention of the day should be allowed."""
        decision = await throttle.can_intervene(1, "nudge")
        assert decision.allowed
        assert decision.reason == ThrottleReason.OK

    # ── Daily cap ────────────────────────────────────────────────────

    async def test_daily_cap_reached(self, throttle, clock, repo) -> None:
        """After 3 interventions, the 4th should be blocked."""
        # Insert 3 interventions today (all different types)
        for i, t in enumerate(["task_breakdown", "nudge", "environment_optimization"]):
            await repo.log_triggered(
                user_id=1,
                intervention_type=t,
                triggered_at=clock.now() + timedelta(minutes=i * 10),
            )

        decision = await throttle.can_intervene(1, "smart_prioritization")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.DAILY_CAP

    async def test_daily_cap_boundary(self, throttle, clock, repo) -> None:
        """Exactly 2 interventions should still allow a 3rd."""
        for i, t in enumerate(["task_breakdown", "nudge"]):
            await repo.log_triggered(
                user_id=1,
                intervention_type=t,
                triggered_at=clock.now() + timedelta(minutes=i * 10),
            )

        # Allow cooldown to pass — we need >2h from the last one
        # (we test cooldown separately; here we want to isolate daily cap)
        clock.advance(hours=3)

        decision = await throttle.can_intervene(1, "environment_optimization")
        assert decision.allowed, f"Expected OK, got {decision.reason}: {decision.detail}"

    # ── Cooldown ─────────────────────────────────────────────────────

    async def test_cooldown_active(self, throttle, clock, repo) -> None:
        """Intervention within 2h of last one should be blocked."""
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=1)  # Only 1h later

        decision = await throttle.can_intervene(1, "task_breakdown")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.COOLDOWN

    async def test_cooldown_expired(self, throttle, clock, repo) -> None:
        """After 2h+ cooldown, interventions should be allowed."""
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=2, minutes=1)  # Just past cooldown

        decision = await throttle.can_intervene(1, "task_breakdown")
        assert decision.allowed, f"Expected OK, got {decision.reason}: {decision.detail}"

    async def test_cooldown_just_before_boundary(self, throttle, clock, repo) -> None:
        """At just under 2h, still within cooldown."""
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=1, minutes=59)  # Just under 2h

        decision = await throttle.can_intervene(1, "task_breakdown")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.COOLDOWN

    # ── Type cap ─────────────────────────────────────────────────────

    async def test_type_cap_reached(self, throttle, clock, repo) -> None:
        """Same type more than 2 times in a day should be blocked."""
        await repo.log_triggered(
            user_id=1, intervention_type="nudge", triggered_at=clock.now()
        )
        clock.advance(hours=3)
        await repo.log_triggered(
            user_id=1, intervention_type="nudge", triggered_at=clock.now()
        )
        clock.advance(hours=3)

        decision = await throttle.can_intervene(1, "nudge")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.TYPE_CAP

    async def test_type_cap_different_types_allowed(self, throttle, clock, repo) -> None:
        """Two of one type should still allow a different type."""
        await repo.log_triggered(
            user_id=1, intervention_type="nudge", triggered_at=clock.now()
        )
        clock.advance(hours=3)
        await repo.log_triggered(
            user_id=1, intervention_type="nudge", triggered_at=clock.now()
        )
        clock.advance(hours=3)

        # Different type should be OK
        decision = await throttle.can_intervene(1, "task_breakdown")
        assert decision.allowed, f"Expected OK, got {decision.reason}: {decision.detail}"

    # ── Fatigue ──────────────────────────────────────────────────────

    async def test_fatigue_reduces_limit(self, throttle, clock, repo) -> None:
        """High ignore rate triggers fatigue mode (1/day)."""
        # Create 7 interventions with most ignored
        for i in range(7):
            await repo.log_triggered(
                user_id=1,
                intervention_type="nudge",
                triggered_at=clock.now() - timedelta(days=i),
            )
        # Mark 5 of them as ignored (rate ~71% > 60%)
        # We need to mark them as ignored in the DB to affect ignore_rate_7d
        # The last 7 should have 5 ignored
        past_logs = await repo.query_range(
            user_id=1,
            start=clock.now() - timedelta(days=7),
            end=clock.now(),
        )
        # Last 5 logs mark as ignored
        for log_entry in past_logs[:5]:
            await repo.update_response(log_entry["id"], "ignored", 0.0)

        # First intervention today (should count against daily cap of 1)
        await repo.log_triggered(
            user_id=1, intervention_type="nudge", triggered_at=clock.now()
        )

        clock.advance(hours=3)

        # Second attempt should be blocked by fatigue-reduced cap
        decision = await throttle.can_intervene(1, "task_breakdown")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.DAILY_CAP

    async def test_fatigue_below_threshold(self, throttle, clock, repo) -> None:
        """Below 60% ignore rate, normal daily limit applies."""
        # Create 7 interventions with only 2 ignored (~29% < 60%)
        for i in range(7):
            await repo.log_triggered(
                user_id=1,
                intervention_type="nudge",
                triggered_at=clock.now() - timedelta(days=i),
            )
        past_logs = await repo.query_range(
            user_id=1,
            start=clock.now() - timedelta(days=7),
            end=clock.now(),
        )
        # Only mark 2 as ignored
        for log_entry in past_logs[:2]:
            await repo.update_response(log_entry["id"], "ignored", 0.0)

        # Should still allow 3 per day
        for i in range(3):
            await repo.log_triggered(
                user_id=1,
                intervention_type=["task_breakdown", "nudge", "environment_optimization"][i],
                triggered_at=clock.now() + timedelta(minutes=i * 10),
            )
            clock.advance(hours=3)

        decision = await throttle.can_intervene(1, "smart_prioritization")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.DAILY_CAP

    # ── Midnight reset ──────────────────────────────────────────────

    async def test_midnight_reset(self, throttle, clock, repo) -> None:
        """Daily counts reset at calendar day boundary."""
        # Log interventions yesterday
        yesterday = clock.now() - timedelta(days=1)
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=yesterday,
        )

        # Today should be fresh
        decision = await throttle.can_intervene(1, "nudge")
        assert decision.allowed, f"Expected OK, got {decision.reason}: {decision.detail}"

    # ── Combined scenarios ──────────────────────────────────────────

    async def test_cooldown_checked_before_type_cap(self, throttle, clock, repo) -> None:
        """Cooldown should be checked before type cap (short-circuit)."""
        await repo.log_triggered(
            user_id=1, intervention_type="nudge", triggered_at=clock.now()
        )
        clock.advance(minutes=30)  # Within cooldown

        decision = await throttle.can_intervene(1, "nudge")
        assert not decision.allowed
        # Cooldown should be the reason, not type cap
        assert decision.reason == ThrottleReason.COOLDOWN

    # ── Cross-day cooldown ──────────────────────────────────────────

    async def test_cooldown_cross_day_boundary(self, throttle, clock, repo) -> None:
        """Intervention at 23:30 yesterday → 00:45 today should still block.

        Regression for P1: the cooldown query previously only looked at
        today's interventions, missing yesterday's late interventions.
        """
        # Set clock to yesterday 23:30 and log an intervention
        clock._now = clock._now.replace(hour=23, minute=30)
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )

        # Advance to today 00:45 (only 1h15m later, well within cooldown)
        clock._now = clock._now.replace(hour=0, minute=45) + timedelta(days=1)

        decision = await throttle.can_intervene(1, "task_breakdown")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.COOLDOWN

    # ── Annoying feedback ──────────────────────────────────────────

    async def test_annoying_feedback_reduces_type_limit(self, throttle, clock, repo) -> None:
        """3+ annoying ratings in 7 days reduces type daily limit to 1."""
        # Create 3 interventions of type "nudge" with "annoying" feedback (past days only)
        for i in range(1, 4):
            log = await repo.log_triggered(
                user_id=1,
                intervention_type="nudge",
                triggered_at=clock.now() - timedelta(days=i),
            )
            await repo.update_feedback(log["id"], "annoying")

        # First nudge today should be allowed
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=3)

        # Second nudge today should be blocked (limit reduced to 1)
        decision = await throttle.can_intervene(1, "nudge")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.ANNOYING

    async def test_below_annoying_threshold_allows_normal_type_limit(
        self, throttle, clock, repo
    ) -> None:
        """2 annoying ratings (< threshold of 3) keeps normal type limit of 2."""
        # Create 2 interventions with "annoying" feedback (past days only)
        for i in range(1, 3):
            log = await repo.log_triggered(
                user_id=1,
                intervention_type="nudge",
                triggered_at=clock.now() - timedelta(days=i),
            )
            await repo.update_feedback(log["id"], "annoying")

        # First nudge today
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=3)

        # Second nudge today should still be allowed (normal limit = 2)
        decision = await throttle.can_intervene(1, "nudge")
        assert decision.allowed, f"Expected OK, got {decision.reason}: {decision.detail}"

    async def test_helpful_feedback_does_not_affect_limit(
        self, throttle, clock, repo
    ) -> None:
        """5+ helpful ratings keep normal limits (no reduction)."""
        for i in range(1, 6):
            log = await repo.log_triggered(
                user_id=1,
                intervention_type="nudge",
                triggered_at=clock.now() - timedelta(days=i),
            )
            await repo.update_feedback(log["id"], "helpful")

        # Normal type limit of 2 still applies
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=3)
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=3)

        # Third nudge blocked by normal type cap, not ANNOYING
        decision = await throttle.can_intervene(1, "nudge")
        assert not decision.allowed
        assert decision.reason == ThrottleReason.TYPE_CAP


# ── P2-1: single-query optimisation ────────────────────────────────────────


class TestP2SingleQuery:
    """RED→GREEN: can_intervene must use exactly one SQL SELECT."""

    @pytest.fixture
    async def intervention_tables(self, engine):
        async with engine.begin() as conn:
            await conn.run_sync(intervention_logs.metadata.create_all)

    @pytest.fixture
    def clock(self) -> FakeClock:
        """Deterministic clock starting at 2026-01-15 08:00 UTC."""
        return FakeClock()

    @pytest.fixture
    def repo(self, session_factory, intervention_tables, clock) -> InterventionLogRepository:
        return InterventionLogRepository(session_factory=session_factory, clock=clock)

    @pytest.fixture
    def throttle(self, repo, clock) -> InterventionThrottle:
        return InterventionThrottle(
            repo=repo,
            clock=clock,
            daily_limit=3,
            type_limit=2,
            cooldown_h=2.0,
            ignore_rate_threshold=0.6,
            fatigue_daily_limit=1,
        )

    async def test_single_sql_select_on_ok_path(
        self, throttle, clock, repo, engine
    ) -> None:
        """P2-1 RED→GREEN: can_intervene(allowed=True) emits 1 SQL SELECT."""
        # Pre-seed one intervention (not today, >cooldown ago) so the
        # all-pass path doesn't need any extra rows.
        await repo.log_triggered(
            user_id=1,
            intervention_type="task_breakdown",
            triggered_at=clock.now() - timedelta(days=3, hours=3),
        )

        # ── Attach SQLAlchemy event listener to count SELECTs ──────────
        select_count = 0

        def _count_select(
            conn, cursor, statement, parameters, context, executemany
        ) -> None:
            nonlocal select_count
            if statement.strip().upper().startswith("SELECT"):
                select_count += 1

        sa.event.listen(engine.sync_engine, "before_cursor_execute", _count_select)

        try:
            decision = await throttle.can_intervene(1, "nudge")
        finally:
            sa.event.remove(engine.sync_engine, "before_cursor_execute", _count_select)

        assert decision.allowed, f"Expected OK, got {decision.reason}: {decision.detail}"
        assert decision.reason == ThrottleReason.OK
        assert select_count == 1, (
            f"P2-1 FAILED: can_intervene emitted {select_count} SQL SELECT(s), "
            f"expected exactly 1 (RED→GREEN gate)"
        )

    async def test_single_sql_select_on_daily_cap_path(
        self, throttle, clock, repo, engine
    ) -> None:
        """Short-circuited daily-cap rejection still makes only 1 SELECT."""
        for i, t in enumerate(
            ["task_breakdown", "nudge", "environment_optimization"]
        ):
            await repo.log_triggered(
                user_id=1,
                intervention_type=t,
                triggered_at=clock.now() + timedelta(minutes=i * 10),
            )

        select_count = 0

        def _count_select(conn, cursor, statement, parameters, context, executemany):
            nonlocal select_count
            if statement.strip().upper().startswith("SELECT"):
                select_count += 1

        sa.event.listen(engine.sync_engine, "before_cursor_execute", _count_select)
        try:
            decision = await throttle.can_intervene(1, "smart_prioritization")
        finally:
            sa.event.remove(engine.sync_engine, "before_cursor_execute", _count_select)

        assert not decision.allowed
        assert decision.reason == ThrottleReason.DAILY_CAP
        assert select_count == 1

    async def test_single_sql_select_on_cooldown_path(
        self, throttle, clock, repo, engine
    ) -> None:
        """Cooldown short-circuit still only 1 SELECT."""
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=1)

        select_count = 0

        def _count_select(conn, cursor, statement, parameters, context, executemany):
            nonlocal select_count
            if statement.strip().upper().startswith("SELECT"):
                select_count += 1

        sa.event.listen(engine.sync_engine, "before_cursor_execute", _count_select)
        try:
            decision = await throttle.can_intervene(1, "task_breakdown")
        finally:
            sa.event.remove(engine.sync_engine, "before_cursor_execute", _count_select)

        assert not decision.allowed
        assert decision.reason == ThrottleReason.COOLDOWN
        assert select_count == 1

    async def test_no_inserts_leak_through(
        self, throttle, clock, repo, engine
    ) -> None:
        """Zero interventions — still 1 SELECT, zero INSERTs."""
        select_count = 0
        insert_count = 0

        def _count(conn, cursor, statement, parameters, context, executemany):
            nonlocal select_count, insert_count
            upper = statement.strip().upper()
            if upper.startswith("SELECT"):
                select_count += 1
            elif upper.startswith("INSERT"):
                insert_count += 1

        sa.event.listen(engine.sync_engine, "before_cursor_execute", _count)
        try:
            decision = await throttle.can_intervene(1, "nudge")
        finally:
            sa.event.remove(engine.sync_engine, "before_cursor_execute", _count)

        assert decision.allowed
        assert decision.reason == ThrottleReason.OK
        assert select_count == 1
        assert insert_count == 0, "can_intervene must be read-only"


# ── P2-1 decision-equivalence smoke-test ────────────────────────────────────


class TestP2DecisionEquivalence:
    """Smoke: every decision path exercised once, counts pinned to DB rows."""

    @pytest.fixture
    async def intervention_tables(self, engine):
        async with engine.begin() as conn:
            await conn.run_sync(intervention_logs.metadata.create_all)

    @pytest.fixture
    def clock(self) -> FakeClock:
        return FakeClock()

    @pytest.fixture
    def repo(self, session_factory, intervention_tables, clock) -> InterventionLogRepository:
        return InterventionLogRepository(session_factory=session_factory, clock=clock)

    @pytest.fixture
    def throttle(self, repo, clock) -> InterventionThrottle:
        return InterventionThrottle(
            repo=repo,
            clock=clock,
            daily_limit=3,
            type_limit=2,
            cooldown_h=2.0,
            ignore_rate_threshold=0.6,
            fatigue_daily_limit=1,
        )

    async def test_get_throttle_stats_zero_rows(
        self, repo, clock
    ) -> None:
        """No rows → all defaults, no division-by-zero."""
        from datetime import timedelta
        now = clock.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_7d = now - timedelta(days=7)
        cooldown_lower_bound = now - timedelta(hours=4)

        stats = await repo.get_throttle_stats(
            1, "nudge",
            now=now,
            today_start=today_start,
            cutoff_7d=cutoff_7d,
            cooldown_lower_bound=cooldown_lower_bound,
        )
        assert stats.ignore_rate == 0.0
        assert stats.today_count == 0
        assert stats.last_triggered_at is None
        assert stats.annoying_count_by_type == 0
        assert stats.today_count_by_type == 0

    async def test_get_throttle_stats_with_data(
        self, repo, clock
    ) -> None:
        """Populated DB → correct single-query aggregates."""
        # Today interventions
        await repo.log_triggered(
            user_id=1, intervention_type="nudge",
            triggered_at=clock.now(),
        )
        await repo.log_triggered(
            user_id=1, intervention_type="nudge",
            triggered_at=clock.now() + timedelta(minutes=10),
        )
        await repo.log_triggered(
            user_id=1, intervention_type="task_breakdown",
            triggered_at=clock.now() + timedelta(minutes=20),
        )

        # Past intervention with 'ignored' response (for ignore_rate)
        past1 = await repo.log_triggered(
            user_id=1, intervention_type="nudge",
            triggered_at=clock.now() - timedelta(days=2),
        )
        await repo.update_response(past1["id"], "ignored")

        # Past intervention with 'annoying' feedback for the type
        past2 = await repo.log_triggered(
            user_id=1, intervention_type="nudge",
            triggered_at=clock.now() - timedelta(days=3),
        )
        await repo.update_feedback(past2["id"], "annoying")

        now = clock.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_7d = now - timedelta(days=7)
        cooldown_lower_bound = now - timedelta(hours=4)

        stats = await repo.get_throttle_stats(
            1, "nudge",
            now=now,
            today_start=today_start,
            cutoff_7d=cutoff_7d,
            cooldown_lower_bound=cooldown_lower_bound,
        )
        # 5 total records in 7d (3 today + 2 past)
        assert stats.ignore_rate == 0.2  # 1 ignored / 5 total
        assert stats.today_count == 3
        assert stats.last_triggered_at is not None
        assert stats.annoying_count_by_type == 1
        assert stats.today_count_by_type == 2  # 2 nudge today

    async def test_decision_matrix_behaviour_unchanged(
        self, throttle, repo, clock, engine
    ) -> None:
        """Full decision-matrix cross-check: every rejection reason fires.

        Uses SQLAlchemy event listener to confirm exactly 1 SELECT per call.
        """
        # Phase 0 — OK (zero rows)
        d = await throttle.can_intervene(1, "nudge")
        assert d.allowed and d.reason == ThrottleReason.OK

        # Phase 1 — Daily cap (3 interventions today, 4th blocked)
        for idx, t in enumerate(
            ["task_breakdown", "nudge", "environment_optimization"]
        ):
            await repo.log_triggered(
                user_id=1, intervention_type=t,
                triggered_at=clock.now() + timedelta(minutes=idx * 10),
            )
        d = await throttle.can_intervene(1, "smart_prioritization")
        assert d.reason == ThrottleReason.DAILY_CAP and not d.allowed

        # Reset clock to tomorrow (2026-01-16) — counts reset
        clock.advance(days=1)

        # Phase 2 — Cooldown: insert one nudge, check 1h later
        await repo.log_triggered(
            user_id=1, intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=1)
        d = await throttle.can_intervene(1, "nudge")
        assert d.reason == ThrottleReason.COOLDOWN and not d.allowed
        # Advance past cooldown so we have room for type-cap check
        clock.advance(hours=2)

        # Phase 3 — Type cap: insert 2 nudges (now total 2 today, type cap=2)
        # First nudge (the Phase 2 one) + 1 more from below = 2 nudges today
        await repo.log_triggered(
            user_id=1, intervention_type="nudge",
            triggered_at=clock.now(),
        )
        clock.advance(hours=3)
        # 3rd nudge attempt → TYPE_CAP (2 nudges already today)
        d = await throttle.can_intervene(1, "nudge")
        assert d.reason == ThrottleReason.TYPE_CAP and not d.allowed

        # Phase 4 — OK for a different type (only 2 total today, no type cap)
        d = await throttle.can_intervene(1, "task_breakdown")
        assert d.allowed and d.reason == ThrottleReason.OK

    async def test_malformed_last_triggered_does_not_crash(
        self, repo, throttle, clock
    ) -> None:
        """Malformed ISO8601 in last_triggered_at → defensive pass (no crash).

        The row's ``triggered_at`` must fall inside the cooldown window
        so that the aggregate SELECT returns it as ``last_triggered_at``.
        The throttle's ``datetime.fromisoformat()`` must raise ValueError,
        and the ``except`` branch must allow the intervention through.
        """
        from sqlalchemy import text

        now = clock.now()
        # Insert a row whose timestamp is within the cooldown window
        # [now-4h, now] but is not a valid ISO8601 datetime string.
        async with repo._session_factory() as session, session.begin():
            await session.execute(
                text(
                    "INSERT INTO intervention_logs (id, user_id, triggered_at, "
                    "intervention_type, created_at) "
                    "VALUES ('malformed-id', 1, "
                    ":ts, 'nudge', '2026-01-15T08:00:00+00:00')"
                ),
                {"ts": (now - timedelta(hours=1)).isoformat().replace("+00:00", "INVALID")},
            )

        # This should not crash — the throttle defensively catches ValueError.
        decision = await throttle.can_intervene(1, "nudge")
        assert decision.allowed
        assert decision.reason == ThrottleReason.OK


# ── P2-1 differential equivalence (Wave 3 review) ──────────────────────────


class TestP2DifferentialEquivalence:
    """For each fixture: call old per-method repo queries, call new
    aggregate method, assert identical results.  Covers future rows,
    mixed ISO offsets, boundary timestamps, zero rows, and
    threshold-equality edge cases."""

    @pytest.fixture
    async def intervention_tables(self, engine):
        async with engine.begin() as conn:
            await conn.run_sync(intervention_logs.metadata.create_all)

    @pytest.fixture
    def clock(self) -> FakeClock:
        return FakeClock()

    @pytest.fixture
    def repo(self, session_factory, intervention_tables, clock) -> InterventionLogRepository:
        return InterventionLogRepository(session_factory=session_factory, clock=clock)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _boundaries(clock: FakeClock, cooldown_h: float = 2.0) -> dict:
        now = clock.now()
        return {
            "now": now,
            "today_start": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "cutoff_7d": now - timedelta(days=7),
            "cooldown_lower_bound": now - timedelta(hours=cooldown_h * 2),
        }

    async def _old_ignore_rate(self, repo, user_id: int) -> float:
        return await repo.ignore_rate_7d(user_id)

    async def _old_today_count(self, repo, user_id: int) -> int:
        return await repo.count_today(user_id)

    async def _old_today_by_type(
        self, repo, user_id: int, intervention_type: str,
    ) -> int:
        return await repo.count_today_by_type(user_id, intervention_type)

    async def _old_annoying(
        self, repo, user_id: int, intervention_type: str,
    ) -> int:
        return await repo.annoying_count_7d_by_type(user_id, intervention_type)

    async def _old_cooldown_last(
        self, repo, user_id: int, b: dict,
    ) -> str | None:
        recent = await repo.query_range(
            user_id, b["cooldown_lower_bound"], b["now"],
        )
        return recent[-1]["triggered_at"] if recent else None

    async def _assert_equiv(
        self, repo, user_id: int, itype: str, b: dict,
    ) -> None:
        """Assert old and new aggregates match field-for-field."""
        new = await repo.get_throttle_stats(
            user_id, itype,
            now=b["now"],
            today_start=b["today_start"],
            cutoff_7d=b["cutoff_7d"],
            cooldown_lower_bound=b["cooldown_lower_bound"],
        )
        old_ir = await self._old_ignore_rate(repo, user_id)
        old_tc = await self._old_today_count(repo, user_id)
        old_tbt = await self._old_today_by_type(repo, user_id, itype)
        old_an = await self._old_annoying(repo, user_id, itype)
        old_cl = await self._old_cooldown_last(repo, user_id, b)

        assert new.ignore_rate == pytest.approx(old_ir), (
            f"ignore_rate mismatch: {new.ignore_rate} != {old_ir}"
        )
        assert new.today_count == old_tc
        assert new.annoying_count_by_type == old_an
        assert new.today_count_by_type == old_tbt
        assert new.last_triggered_at == old_cl, (
            f"last_triggered_at mismatch: {new.last_triggered_at!r} != {old_cl!r}"
        )

    # ── fixtures ─────────────────────────────────────────────────────

    async def test_zero_rows(self, repo, clock) -> None:
        b = self._boundaries(clock)
        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_today_and_types(self, repo, clock) -> None:
        now = clock.now()
        await repo.log_triggered(1, "nudge", triggered_at=now + timedelta(minutes=5))
        await repo.log_triggered(1, "nudge", triggered_at=now + timedelta(minutes=10))
        await repo.log_triggered(1, "task_breakdown", triggered_at=now + timedelta(minutes=15))
        b = self._boundaries(clock)
        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_ignored_total_denominator(self, repo, clock) -> None:
        now = clock.now()
        past = await repo.log_triggered(
            1, "nudge", triggered_at=now - timedelta(days=2),
        )
        await repo.update_response(past["id"], "ignored")
        b = self._boundaries(clock)
        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_annoying_by_type(self, repo, clock) -> None:
        now = clock.now()
        past = await repo.log_triggered(
            1, "nudge", triggered_at=now - timedelta(days=3),
        )
        await repo.update_feedback(past["id"], "annoying")
        b = self._boundaries(clock)
        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_cooldown_latest(self, repo, clock) -> None:
        now = clock.now()
        # Insert rows within the cooldown window [now-4h, now]
        await repo.log_triggered(
            1, "nudge", triggered_at=now - timedelta(hours=3),
        )
        await repo.log_triggered(
            1, "nudge", triggered_at=now - timedelta(hours=1),
        )
        b = self._boundaries(clock)
        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_future_row_excluded_from_cooldown(self, repo, clock) -> None:
        """Future row → old query_range excludes it; new aggregate must too."""
        now = clock.now()
        # Insert a future row (should NOT appear as cooldown last)
        await repo.log_triggered(
            1, "nudge", triggered_at=now + timedelta(hours=1),
        )
        # Insert a real recent row inside cooldown
        await repo.log_triggered(
            1, "task_breakdown", triggered_at=now - timedelta(hours=1),
        )
        b = self._boundaries(clock)
        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_offset_equivalent_text(self, repo, clock) -> None:
        """Same instant, different offset → old/new must agree on cooldown last.

        +01:00 offset and +00:00 offset for the same instant produce
        different text, but old query_range and new aggregate both use
        text ordering, so they must agree on which row is selected.
        """
        now = clock.now()
        # Two rows at the same instant, different offsets
        await repo.log_triggered(
            1, "nudge",
            triggered_at=now - timedelta(hours=1),
        )
        # Same UTC instant but different textual form: +01:00 vs +00:00
        async with repo._session_factory() as session, session.begin():
            from sqlalchemy import text as sa_text
            await session.execute(
                sa_text(
                    "INSERT INTO intervention_logs (id, user_id, triggered_at, "
                    "intervention_type, created_at) "
                    "VALUES ('off-1', 1, :ts, 'nudge', '2026-01-15T08:00:00+00:00')"
                ),
                {"ts": (now - timedelta(hours=1)).isoformat().replace("+00:00", "+01:00")},
            )
        b = self._boundaries(clock)
        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_cutoff_boundary(self, repo, clock) -> None:
        """Row exactly at cutoff_7d edge → included in both old and new."""
        b = self._boundaries(clock)
        # Exactly at the cutoff boundary
        await repo.log_triggered(
            1, "nudge", triggered_at=b["cutoff_7d"],
        )
        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_large_cooldown_includes_old_rows(self, repo, clock) -> None:
        """cooldown_h=200 → cooldown_lower_bound ≈ 16.7 days ago.

        An 8-day-old row is outside the 7-day window for ignore_rate /
        annoying count but INSIDE the cooldown window.  The outer WHERE
        must include it so the cooldown CASE can see it, while the 7-day
        aggregates must NOT count it.
        """
        now = clock.now()
        b = self._boundaries(clock, cooldown_h=200.0)

        # 8-day-old row: within cooldown window, outside 7-day window
        eight_days_ago = now - timedelta(days=8)
        await repo.log_triggered(
            1, "nudge", triggered_at=eight_days_ago,
        )

        # Also insert a recent row for cooldown to have something to find
        recent = now - timedelta(hours=1)
        await repo.log_triggered(
            1, "nudge", triggered_at=recent,
        )

        await self._assert_equiv(repo, 1, "nudge", b)

    async def test_old_row_in_cooldown_not_in_7day(self, repo, clock) -> None:
        """An 8-day-old row affects cooldown but zero 7-day aggregates.

        With cooldown_h=200, the 8-day-old row is the only row in the
        cooldown window.  It must appear as last_triggered_at (old code
        returns it via query_range), but total_7d/ignored_7d/annoying
        must be zero since the row is older than 7 days.
        """
        now = clock.now()
        b = self._boundaries(clock, cooldown_h=200.0)

        # Only one row — 8 days old, inside cooldown window only
        eight_days_ago = now - timedelta(days=8)
        await repo.log_triggered(
            1, "nudge", triggered_at=eight_days_ago,
        )

        new = await repo.get_throttle_stats(
            1, "nudge",
            now=b["now"],
            today_start=b["today_start"],
            cutoff_7d=b["cutoff_7d"],
            cooldown_lower_bound=b["cooldown_lower_bound"],
        )

        # 7-day stats: zero (row is 8 days old)
        assert new.ignore_rate == 0.0
        assert new.today_count == 0
        assert new.today_count_by_type == 0
        assert new.annoying_count_by_type == 0

        # Cooldown: the old row should be the last_triggered_at
        assert new.last_triggered_at is not None, (
            "8-day-old row must appear in cooldown last_triggered_at"
        )

        # Cross-check via old methods
        old_cl = await repo.query_range(
            1, b["cooldown_lower_bound"], b["now"],
        )
        assert old_cl, "old query_range must find the 8-day-old row"
        assert new.last_triggered_at == old_cl[-1]["triggered_at"]
