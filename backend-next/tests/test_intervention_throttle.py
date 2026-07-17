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
        self._now = start or datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)

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
    def repo(self, session_factory, intervention_tables) -> InterventionLogRepository:
        """Repository bound to a test DB with intervention_logs table."""
        return InterventionLogRepository(session_factory=session_factory)

    @pytest.fixture
    def clock(self) -> FakeClock:
        """Deterministic clock starting at 2026-07-17 08:00 UTC."""
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
