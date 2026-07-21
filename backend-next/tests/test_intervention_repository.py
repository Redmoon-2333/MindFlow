"""Tests for infrastructure/repositories/intervention.py.

Covers:
  - log_triggered: basic insert with and without context
  - update_response: accepted/ignored/dismissed
  - update_response: not found returns None
  - count_today: daily count
  - count_today_by_type: per-type daily count
  - ignore_rate_7d: edge cases (zero, all ignored, partial)
  - query_range: time-bounded query
  - get_by_id: lookup by intervention ID
  - Clock-injected repo is date-independent (P0 regression)
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
)


class FakeClock:
    """Deterministic clock for testing date independence."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 15, 8, 0, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._now

    def advance(self, **kwargs: Any) -> None:
        self._now += timedelta(**kwargs)


@pytest.fixture
async def intervention_tables(engine):
    """Create the intervention_logs table."""
    async with engine.begin() as conn:
        await conn.run_sync(intervention_logs.metadata.create_all)


class TestInterventionLogRepository:
    """CRUD and query tests for InterventionLogRepository."""

    @pytest.fixture
    def repo(self, session_factory, intervention_tables) -> InterventionLogRepository:
        return InterventionLogRepository(session_factory=session_factory)

    async def test_log_triggered_basic(self, repo) -> None:
        """Basic intervention log insertion."""
        result = await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
        )
        assert result["user_id"] == 1
        assert result["intervention_type"] == "nudge"
        assert result["cbt_technique"] is None
        assert result["context_json"] is None
        assert result["user_response"] is None

    async def test_log_triggered_with_context(self, repo) -> None:
        """Insert with CBT technique and context dict."""
        result = await repo.log_triggered(
            user_id=1,
            intervention_type="task_breakdown",
            cbt_technique="goal_setting",
            context={"source": "rule_engine", "confidence": 0.8},
        )
        assert result["cbt_technique"] == "goal_setting"
        assert result["context_json"] == {"source": "rule_engine", "confidence": 0.8}

    async def test_log_triggered_with_id(self, repo) -> None:
        """Custom intervention ID is persisted."""
        result = await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            intervention_id="custom-id-001",
        )
        assert result["id"] == "custom-id-001"

    async def test_update_response_accepted(self, repo) -> None:
        """Update response to accepted."""
        log = await repo.log_triggered(user_id=1, intervention_type="nudge")
        updated = await repo.update_response(log["id"], "accepted", latency_s=5.0)
        assert updated is not None
        assert updated["user_response"] == "accepted"
        assert updated["response_latency_s"] == 5.0

    async def test_update_response_ignored(self, repo) -> None:
        """Update response to ignored."""
        log = await repo.log_triggered(user_id=1, intervention_type="nudge")
        updated = await repo.update_response(log["id"], "ignored")
        assert updated is not None
        assert updated["user_response"] == "ignored"

    async def test_update_response_dismissed(self, repo) -> None:
        """Update response to dismissed."""
        log = await repo.log_triggered(user_id=1, intervention_type="nudge")
        updated = await repo.update_response(log["id"], "dismissed")
        assert updated is not None
        assert updated["user_response"] == "dismissed"

    async def test_update_response_not_found(self, repo) -> None:
        """Update on non-existent ID returns None."""
        updated = await repo.update_response("non-existent-id", "accepted")
        assert updated is None

    async def test_count_today_zero(self, repo) -> None:
        """No interventions today → count is 0."""
        count = await repo.count_today(1)
        assert count == 0

    async def test_count_today(self, repo) -> None:
        """Count today's interventions."""
        await repo.log_triggered(user_id=1, intervention_type="nudge")
        await repo.log_triggered(user_id=1, intervention_type="task_breakdown")
        count = await repo.count_today(1)
        assert count == 2

    async def test_count_today_other_user(self, repo) -> None:
        """Count is per-user."""
        await repo.log_triggered(user_id=2, intervention_type="nudge")
        count = await repo.count_today(1)
        assert count == 0

    async def test_count_today_by_type(self, repo) -> None:
        """Count today's interventions of a specific type."""
        await repo.log_triggered(user_id=1, intervention_type="nudge")
        await repo.log_triggered(user_id=1, intervention_type="nudge")
        await repo.log_triggered(user_id=1, intervention_type="task_breakdown")

        nudge_count = await repo.count_today_by_type(1, "nudge")
        assert nudge_count == 2

        tb_count = await repo.count_today_by_type(1, "task_breakdown")
        assert tb_count == 1

        missing_count = await repo.count_today_by_type(1, "smart_prioritization")
        assert missing_count == 0

    async def test_ignore_rate_7d_no_data(self, repo) -> None:
        """Zero interventions → ignore rate is 0.0."""
        rate = await repo.ignore_rate_7d(1)
        assert rate == 0.0

    async def test_ignore_rate_7d_all_ignored(self, repo) -> None:
        """All interventions ignored → rate is 1.0."""
        log1 = await repo.log_triggered(user_id=1, intervention_type="nudge")
        log2 = await repo.log_triggered(user_id=1, intervention_type="task_breakdown")
        await repo.update_response(log1["id"], "ignored")
        await repo.update_response(log2["id"], "ignored")
        rate = await repo.ignore_rate_7d(1)
        assert rate == 1.0

    async def test_ignore_rate_7d_partial(self, repo) -> None:
        """1 ignored out of 3 → rate is 1/3."""
        logs = []
        for t in ["nudge", "task_breakdown", "environment_optimization"]:
            log = await repo.log_triggered(user_id=1, intervention_type=t)
            logs.append(log)
        await repo.update_response(logs[0]["id"], "ignored")
        await repo.update_response(logs[1]["id"], "accepted")
        await repo.update_response(logs[2]["id"], "accepted")
        rate = await repo.ignore_rate_7d(1)
        assert rate == pytest.approx(1.0 / 3.0)

    async def test_query_range(self, repo) -> None:
        """Query within time range returns matching logs."""
        now = datetime.now(UTC)
        await repo.log_triggered(
            user_id=1, intervention_type="nudge", triggered_at=now - timedelta(hours=2)
        )
        await repo.log_triggered(user_id=1, intervention_type="task_breakdown", triggered_at=now)
        await repo.log_triggered(
            user_id=1, intervention_type="nudge", triggered_at=now + timedelta(hours=2)
        )

        results = await repo.query_range(
            user_id=1,
            start=now - timedelta(hours=1),
            end=now + timedelta(hours=1),
        )
        assert len(results) == 1
        assert results[0]["intervention_type"] == "task_breakdown"

    async def test_get_by_id_found(self, repo) -> None:
        """Lookup by existing ID returns the log."""
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            intervention_id="find-me-001",
        )
        found = await repo.get_by_id("find-me-001")
        assert found is not None
        assert found["id"] == "find-me-001"
        assert found["intervention_type"] == "nudge"

    async def test_get_by_id_not_found(self, repo) -> None:
        """Lookup by non-existent ID returns None."""
        found = await repo.get_by_id("non-existent")
        assert found is None

    async def test_query_range_by_date(self, repo) -> None:
        """Date-based range query."""
        from datetime import date
        now = datetime.now(UTC)
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=now - timedelta(days=2),
        )
        await repo.log_triggered(
            user_id=1,
            intervention_type="task_breakdown",
            triggered_at=now,
        )

        yesterday = date.today() - timedelta(days=1)
        tomorrow = date.today() + timedelta(days=1)

        results = await repo.query_range_by_date(1, yesterday, tomorrow)
        assert len(results) == 1
        assert results[0]["intervention_type"] == "task_breakdown"


class TestClockInjection:
    """P0 regression: clock-injected repo is date-independent.

    Uses a FakeClock set to a non-today date (2026-01-15) so tests pass
    regardless of the real system date.
    """

    @pytest.fixture
    async def intervention_tables(self, engine):
        async with engine.begin() as conn:
            await conn.run_sync(intervention_logs.metadata.create_all)

    @pytest.fixture
    def clock(self) -> FakeClock:
        """Clock at a fixed non-today date."""
        return FakeClock()

    @pytest.fixture
    def repo(self, session_factory, intervention_tables, clock) -> InterventionLogRepository:
        """Repo injected with the same FakeClock."""
        return InterventionLogRepository(session_factory=session_factory, clock=clock)

    @pytest.fixture
    def throttle(self, repo, clock) -> InterventionThrottle:
        """Throttle sharing the same FakeClock."""
        return InterventionThrottle(
            repo=repo,
            clock=clock,
            daily_limit=3,
            type_limit=2,
            cooldown_h=2.0,
            ignore_rate_threshold=0.6,
            fatigue_daily_limit=1,
        )

    async def test_log_and_count_use_clock(self, repo, clock) -> None:
        """log_triggered and count_today both use the injected clock."""
        await repo.log_triggered(user_id=1, intervention_type="nudge")
        count = await repo.count_today(1)
        assert count == 1

        # The log should have the clock's timestamp
        logs = await repo.query_range(
            1,
            start=clock.now() - timedelta(seconds=1),
            end=clock.now() + timedelta(seconds=1),
        )
        assert len(logs) == 1
        assert logs[0]["intervention_type"] == "nudge"

    async def test_count_today_by_type_uses_clock(self, repo) -> None:
        """count_today_by_type respects the injected clock."""
        await repo.log_triggered(user_id=1, intervention_type="nudge")
        await repo.log_triggered(user_id=1, intervention_type="task_breakdown")
        await repo.log_triggered(user_id=1, intervention_type="nudge")

        assert await repo.count_today_by_type(1, "nudge") == 2
        assert await repo.count_today_by_type(1, "task_breakdown") == 1
        assert await repo.count_today_by_type(1, "smart_prioritization") == 0

    async def test_ignore_rate_7d_uses_clock(self, repo) -> None:
        """ignore_rate_7d uses the injected clock for its cutoff."""
        log1 = await repo.log_triggered(user_id=1, intervention_type="nudge")
        log2 = await repo.log_triggered(user_id=1, intervention_type="task_breakdown")
        await repo.update_response(log1["id"], "ignored")
        await repo.update_response(log2["id"], "accepted")
        rate = await repo.ignore_rate_7d(1)
        assert rate == pytest.approx(0.5)

    async def test_repo_and_throttle_share_clock(self, repo, throttle, clock) -> None:
        """Same FakeClock across repo + throttle — date-independent pass."""
        # Log a "today" intervention (clock says 2026-01-15)
        await repo.log_triggered(user_id=1, intervention_type="nudge")
        # Advance past cooldown so cooldown doesn't interfere
        clock.advance(hours=3)
        decision = await throttle.can_intervene(1, "nudge")
        assert decision.allowed

    async def test_midnight_reset_with_clock(self, repo, throttle, clock) -> None:
        """Midnight reset works at any clock date."""
        yesterday = clock.now() - timedelta(days=1)
        await repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=yesterday,
        )
        decision = await throttle.can_intervene(1, "nudge")
        assert decision.allowed, f"Expected OK after reset, got {decision.reason}"
