"""Tests for services/effectiveness_service.py — C5 minimal evaluation.

Covers:
  - compare_windows: before/after with sufficient data
  - compare_windows: no data returns empty report
  - compare_windows: non-existent intervention
  - weekly_effectiveness: aggregation
  - weekly_effectiveness: no data returns zeroed report
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mindflow.domain.events import make_event
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
    intervention_logs,
)
from mindflow.services.effectiveness_service import EffectivenessService

_TS = datetime(2026, 7, 17, 10, 0, 0, tzinfo=UTC)


class TestEffectivenessService:
    """Effectiveness evaluation tests."""

    @pytest.fixture
    async def seeded_repos(self, engine, session_factory):
        """Create tables and seed data."""
        async with engine.begin() as conn:
            await conn.run_sync(activity_events.metadata.create_all)
            await conn.run_sync(intervention_logs.metadata.create_all)

        activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
        intervention_repo = InterventionLogRepository(session_factory=session_factory)
        return activity_repo, intervention_repo

    @pytest.fixture
    def service(self, seeded_repos) -> EffectivenessService:
        activity_repo, intervention_repo = seeded_repos
        return EffectivenessService(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
        )

    async def _seed_events(
        self,
        repo: SQLAlchemyActivityRepository,
        user_id: int,
        center: datetime,
        count_before: int = 20,
        count_after: int = 20,
    ) -> None:
        """Seed events before and after *center*."""
        for i in range(count_before):
            ev = make_event(
                user_id=user_id,
                timestamp_utc=center - timedelta(minutes=30) + timedelta(seconds=i * 30),
                duration_s=30.0,
                process_name="Code.exe",
            )
            await repo.append_event(ev)

        for i in range(count_after):
            ev = make_event(
                user_id=user_id,
                timestamp_utc=center + timedelta(seconds=i * 30),
                duration_s=30.0,
                process_name="Code.exe",
            )
            await repo.append_event(ev)

    async def test_compare_windows_with_data(
        self, service, seeded_repos
    ) -> None:
        """Before/after with sufficient data returns metrics."""
        activity_repo, intervention_repo = seeded_repos

        # Seed intervention
        await intervention_repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=_TS,
            intervention_id="eval-001",
        )

        # Seed events
        await self._seed_events(activity_repo, 1, _TS)

        report = await service.compare_windows("eval-001")
        assert report.has_data is True
        assert report.before["focus_score"] > 0
        assert report.after["focus_score"] > 0
        # Both before and after windows should have same-app focus, so
        # focus_score should be similar
        assert "focus_score" in report.deltas
        assert "switch_rate" in report.deltas
        assert "distraction_ratio" in report.deltas

    async def test_compare_windows_no_data(
        self, service, seeded_repos
    ) -> None:
        """Insufficient data returns empty report."""
        _, intervention_repo = seeded_repos

        await intervention_repo.log_triggered(
            user_id=1,
            intervention_type="nudge",
            triggered_at=_TS,
            intervention_id="eval-002",
        )

        report = await service.compare_windows("eval-002")
        assert report.has_data is False
        assert report.deltas["focus_score"] == 0.0

    async def test_compare_windows_not_found(self, service) -> None:
        """Non-existent intervention returns empty report."""
        report = await service.compare_windows("non-existent")
        assert report.has_data is False
        assert report.intervention_id == "non-existent"

    async def test_weekly_effectiveness_with_data(
        self, service, seeded_repos
    ) -> None:
        """Weekly aggregation with intervention data."""
        activity_repo, intervention_repo = seeded_repos

        # Seed multiple interventions
        for i in range(3):
            ts = _TS - timedelta(days=i)
            log = await intervention_repo.log_triggered(
                user_id=1,
                intervention_type="nudge",
                triggered_at=ts,
            )
            await self._seed_events(activity_repo, 1, ts, count_before=10, count_after=10)
            await intervention_repo.update_response(log["id"], "accepted", 5.0)

        result = await service.weekly_effectiveness(user_id=1)
        assert result["total_interventions"] >= 3
        assert result["with_feedback"] >= 3
        assert result["acceptance_rate"] == 1.0  # all accepted
        # Some average deltas should be present (even if zero)
        assert "avg_focus_delta" in result

    async def test_weekly_effectiveness_no_data(self, service) -> None:
        """No interventions → zeroed report."""
        result = await service.weekly_effectiveness(user_id=1)
        assert result["total_interventions"] == 0
        assert result["with_feedback"] == 0
        assert result["acceptance_rate"] == 0.0
        assert result["avg_focus_delta"] == 0.0
