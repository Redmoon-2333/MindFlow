"""Tests for scheduler (build_scheduler).

Covers:
  - 5 jobs registered (4 cron + 1 interval) with correct configuration
  - Graceful handling of missing services
  - Auto-intervention job: interval config and time-of-day guard logic
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from apscheduler.triggers.interval import IntervalTrigger

from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType
from mindflow.services.scheduler import (
    _auto_intervention_check,
    build_scheduler,
)


def _make_assessment(
    confidence: float, top_type: ProcrastinationType = ProcrastinationType.TASK_AVERSION
) -> MagicMock:
    """Create a mock ProcrastinationAssessment with a specific confidence."""
    a = MagicMock()
    a.types = (top_type,)
    a.confidence = {top_type: confidence}
    a.recommended_technique = CBTTechnique.GRADED_EXPOSURE
    a.rationale = "test"
    a.source = "rule_engine"
    return a


class TestBuildScheduler:
    """Scheduler configuration tests."""

    async def test_registers_5_jobs_when_all_provided(self) -> None:
        """With all services provided, 5 jobs should be registered."""
        analysis = MagicMock()
        report = MagicMock()
        maintenance = MagicMock()
        intervention = MagicMock()
        activity_repo = MagicMock()

        scheduler = build_scheduler(
            analysis_service=analysis,
            report_service=report,
            maintenance_service=maintenance,
            intervention_service=intervention,
            activity_repository=activity_repo,
        )

        jobs = scheduler.get_jobs()
        assert len(jobs) == 5

        job_ids = {j.id for j in jobs}
        assert "identify_sessions" in job_ids
        assert "daily_report" in job_ids
        assert "event_cleanup" in job_ids
        assert "daily_backup" in job_ids
        assert "auto_intervention_check" in job_ids

    async def test_auto_intervention_check_is_interval_30min(self) -> None:
        """auto_intervention_check should be an interval job at 30 minutes."""
        scheduler = build_scheduler(
            intervention_service=MagicMock(),
            activity_repository=MagicMock(),
        )
        job = scheduler.get_job("auto_intervention_check")
        assert job is not None

        trigger = job.trigger
        assert isinstance(trigger, IntervalTrigger)
        assert trigger.interval.total_seconds() == 1800  # 30 min

    async def test_registers_4_jobs_without_intervention(self) -> None:
        """Without intervention service, only 4 jobs should be registered."""
        analysis = MagicMock()
        report = MagicMock()
        maintenance = MagicMock()

        scheduler = build_scheduler(
            analysis_service=analysis,
            report_service=report,
            maintenance_service=maintenance,
        )

        jobs = scheduler.get_jobs()
        assert len(jobs) == 4
        job_ids = {j.id for j in jobs}
        assert "auto_intervention_check" not in job_ids

    async def test_identify_sessions_cron(self) -> None:
        """identify_sessions should run at 23:59 daily."""
        scheduler = build_scheduler(
            analysis_service=MagicMock(),
        )
        job = scheduler.get_job("identify_sessions")
        assert job is not None
        trigger = job.trigger
        assert str(trigger.fields[5]) == "23"  # hour
        assert str(trigger.fields[6]) == "59"  # minute

    async def test_event_cleanup_cron(self) -> None:
        """event_cleanup should run at 03:00 daily with retention_days kwarg."""
        scheduler = build_scheduler(
            maintenance_service=MagicMock(),
        )
        job = scheduler.get_job("event_cleanup")
        assert job is not None
        assert "retention_days" in job.kwargs

    async def test_missing_service_skips_jobs(self) -> None:
        """Without services, corresponding jobs should be skipped."""
        scheduler = build_scheduler()
        jobs = scheduler.get_jobs()
        assert len(jobs) == 0

    async def test_scheduler_timezone_is_utc(self) -> None:
        """Scheduler timezone should be UTC."""
        scheduler = build_scheduler()
        assert scheduler.timezone == UTC

    async def test_report_job_registered(self) -> None:
        """daily_report should run at 00:01."""
        scheduler = build_scheduler(
            report_service=MagicMock(),
        )
        job = scheduler.get_job("daily_report")
        assert job is not None

    async def test_backup_job_registered(self) -> None:
        """daily_backup should have maintenance_service as dependency."""
        scheduler = build_scheduler(
            maintenance_service=MagicMock(),
        )
        job = scheduler.get_job("daily_backup")
        assert job is not None

    async def test_shutdown_does_not_raise(self) -> None:
        """shutdown(wait=False) should not raise."""
        scheduler = build_scheduler()
        scheduler.start()  # Initialise event loop reference
        scheduler.shutdown(wait=False)


class TestAutoInterventionCheck:
    """_auto_intervention_check logic tests.

    Covers time-of-day guard, empty events guard, all-idle guard,
    confidence guard, and successful dispatch.
    """

    async def test_skips_outside_working_hours(self) -> None:
        """Before 08:00 or after 23:00 should skip silently."""
        mock_repo = AsyncMock()
        mock_svc = MagicMock()

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 7, 59, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(mock_repo, mock_svc)

            # Should not query events
            mock_repo.query_range.assert_not_called()
            mock_svc.maybe_intervene.assert_not_called()

    async def test_skips_when_no_events(self) -> None:
        """No events in lookback window should skip silently."""
        mock_repo = AsyncMock()
        mock_repo.query_range = AsyncMock(return_value=[])
        mock_svc = MagicMock()

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(mock_repo, mock_svc)

            mock_repo.query_range.assert_awaited_once()
            mock_svc.maybe_intervene.assert_not_called()

    async def test_skips_when_all_idle(self) -> None:
        """All idle events should skip silently."""
        from mindflow.domain.events import make_event

        mock_repo = AsyncMock()
        idle_events = [
            make_event(
                user_id=1,
                timestamp_utc=datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC),
                is_idle=True,
            )
        ]
        mock_repo.query_range = AsyncMock(return_value=idle_events)
        mock_svc = MagicMock()

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(mock_repo, mock_svc)

            mock_repo.query_range.assert_awaited_once()
            mock_svc.maybe_intervene.assert_not_called()

    async def test_skips_on_low_confidence(self) -> None:
        """Low confidence assessment should skip."""
        from mindflow.domain.events import make_event

        mock_repo = AsyncMock()
        events = [
            make_event(
                user_id=1,
                timestamp_utc=datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC),
                duration_s=10.0,
                process_name="Code.exe",
                app_name="Code.exe",
            )
        ]
        mock_repo.query_range = AsyncMock(return_value=events)

        # Mock intervention_service
        mock_svc = MagicMock()

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(mock_repo, mock_svc)

            # Should have checked but not intervened (low confidence from
            # single event with no significant pattern)
            mock_repo.query_range.assert_awaited_once()

    async def test_dispatches_intervention_on_high_confidence(self) -> None:
        """High confidence assessment should call maybe_intervene."""
        from mindflow.domain.events import make_event

        mock_repo = AsyncMock()
        # Create enough events to trigger impulsivity detection
        base = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
        events = []
        for i in range(25):  # >12 switches/h implied by 25 events in 30 min
            events.append(
                make_event(
                    user_id=1,
                    timestamp_utc=base + __import__("datetime").timedelta(seconds=i * 60),
                    duration_s=30.0,
                    process_name=f"App_{i % 5}.exe",
                    app_name=f"App_{i % 5}.exe",
                )
            )
        mock_repo.query_range = AsyncMock(return_value=events)

        # Mock intervention_service to return success
        mock_result = MagicMock()
        mock_result.skipped = False
        mock_svc = MagicMock()
        mock_svc.maybe_intervene = AsyncMock(return_value=mock_result)

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(mock_repo, mock_svc)

            mock_repo.query_range.assert_awaited_once()
            mock_svc.maybe_intervene.assert_awaited_once()


class TestAutoInterventionCheckThreeTier:
    """Three-tier routing (G005) tests for _auto_intervention_check.

    All tests use a mock RuleEngine for deterministic confidence control.
    The real rule-engine → confidence mapping is tested in TestAutoInterventionCheck.

    Covers:
      - Autonomy disabled → skip
      - < 0.5 confidence → skip (tier 0)
      - 0.5 <= confidence < 0.75 → direct maybe_intervene (tier 1)
      - >= 0.75 with panel → panel escalation (tier 2)
      - >= 0.75 but panel fails → fallback to rule engine
      - >= 0.75 but autonomy disabled → skip before panel
    """

    async def test_skips_when_autonomy_disabled(self) -> None:
        """Autonomy service reports disabled → skip."""
        mock_repo = AsyncMock()
        mock_svc = MagicMock()
        mock_autonomy = MagicMock()
        mock_autonomy.is_enabled = AsyncMock(return_value=False)

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(
                mock_repo, mock_svc, autonomy_service=mock_autonomy
            )

            mock_repo.query_range.assert_not_called()
            mock_svc.maybe_intervene.assert_not_called()

    async def _run_with_mock_rule(
        self, confidence: float, panel_service: object = None, autonomy_service: object = None
    ) -> MagicMock:
        """Run _auto_intervention_check with a mock rule engine at *confidence*.

        Returns the mock intervention_service for assertion.
        """
        from mindflow.domain.events import make_event

        base = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
        events = [
            make_event(
                user_id=1,
                timestamp_utc=base,
                duration_s=10.0,
                process_name="Code.exe",
                app_name="Code.exe",
            ),
        ]
        mock_repo = AsyncMock()
        mock_repo.query_range = AsyncMock(return_value=events)

        mock_engine = MagicMock()
        mock_engine.assess.return_value = _make_assessment(confidence=confidence)

        mock_result = MagicMock()
        mock_result.skipped = False
        mock_svc = MagicMock()
        mock_svc.maybe_intervene = AsyncMock(return_value=mock_result)

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(
                mock_repo,
                mock_svc,
                rule_engine=mock_engine,
                panel_service=panel_service,  # type: ignore[arg-type]
                autonomy_service=autonomy_service,  # type: ignore[arg-type]
            )

        return mock_svc

    async def test_skip_on_low_confidence(self) -> None:
        """Mock engine returns confidence 0.3 < 0.5 → skip (tier 0)."""
        svc = await self._run_with_mock_rule(confidence=0.3)
        svc.maybe_intervene.assert_not_called()

    async def test_mid_confidence_direct_intervention(self) -> None:
        """Mock engine returns confidence 0.6 → direct maybe_intervene (tier 1)."""
        svc = await self._run_with_mock_rule(confidence=0.6)
        svc.maybe_intervene.assert_awaited_once()

    async def test_high_confidence_panel_escalation(self) -> None:
        """Mock engine returns confidence 0.85 → escalate to panel (tier 2)."""
        from mindflow.agents.types import PanelVerdict

        verdict = PanelVerdict(
            types=(ProcrastinationType.IMPULSIVITY,),
            confidence={ProcrastinationType.IMPULSIVITY: 0.85},
            recommended_technique=CBTTechnique.STIMULUS_CONTROL,
            rationale="测试裁决",
            dissent=(),
            transcript=(),
            escalated=False,
            call_count=1,
            source="panel",
        )
        mock_panel = MagicMock()
        mock_panel.run_daily_panel = AsyncMock(return_value=verdict)

        from mindflow.services.scheduler import _DAILY_PANEL_RUN_DATES

        _DAILY_PANEL_RUN_DATES.clear()

        svc = await self._run_with_mock_rule(confidence=0.85, panel_service=mock_panel)

        mock_panel.run_daily_panel.assert_awaited_once()
        svc.maybe_intervene.assert_awaited_once()
        _DAILY_PANEL_RUN_DATES.discard("2026-07-17")

    async def test_panel_failure_fallback_to_rule(self) -> None:
        """Panel raises PanelUnavailableError → fallback to rule (tier 2 fallback)."""
        mock_panel = MagicMock()
        from mindflow.agents.types import PanelUnavailableError

        mock_panel.run_daily_panel = AsyncMock(
            side_effect=PanelUnavailableError(reason="Test failure")
        )

        from mindflow.services.scheduler import _DAILY_PANEL_RUN_DATES

        _DAILY_PANEL_RUN_DATES.discard("2026-07-17")

        svc = await self._run_with_mock_rule(confidence=0.85, panel_service=mock_panel)

        mock_panel.run_daily_panel.assert_awaited_once()
        svc.maybe_intervene.assert_awaited_once()

    async def test_autonomy_disabled_before_panel(self) -> None:
        """Autonomy disabled → skip even with high confidence."""
        mock_repo = AsyncMock()
        mock_svc = MagicMock()
        mock_autonomy = MagicMock()
        mock_autonomy.is_enabled = AsyncMock(return_value=False)
        mock_panel = MagicMock()

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(
                mock_repo,
                mock_svc,
                panel_service=mock_panel,
                autonomy_service=mock_autonomy,
            )

            mock_repo.query_range.assert_not_called()
            mock_panel.run_daily_panel.assert_not_called()
            mock_svc.maybe_intervene.assert_not_called()


class TestDailyPanelRunClaim:
    """C4: the daily-panel run is claimed atomically before the await, so the
    23:30 cron and the 30-min check cannot both fire the panel on the same day.
    """

    @staticmethod
    def _events() -> list[object]:
        from mindflow.domain.events import make_event

        base = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
        return [
            make_event(
                user_id=1,
                timestamp_utc=base,
                duration_s=10.0,
                process_name="Code.exe",
                app_name="Code.exe",
            ),
        ]

    async def test_concurrent_checks_trigger_panel_once(self) -> None:
        """Two concurrent high-confidence checks → panel runs exactly once."""
        import asyncio

        from mindflow.agents.types import PanelVerdict
        from mindflow.services.scheduler import _DAILY_PANEL_RUN_DATES

        _DAILY_PANEL_RUN_DATES.discard("2026-07-17")

        verdict = PanelVerdict(
            types=(ProcrastinationType.IMPULSIVITY,),
            confidence={ProcrastinationType.IMPULSIVITY: 0.85},
            recommended_technique=CBTTechnique.STIMULUS_CONTROL,
            rationale="并发测试裁决",
            dissent=(),
            transcript=(),
            escalated=False,
            call_count=1,
            source="panel",
        )

        # Panel blocks until released, forcing the two tasks to interleave so
        # the second one reaches the claim while the first is still "running".
        gate = asyncio.Event()
        call_count = {"n": 0}

        async def _blocking_panel(**_: object) -> PanelVerdict:
            call_count["n"] += 1
            await gate.wait()
            return verdict

        mock_panel = MagicMock()
        mock_panel.run_daily_panel = AsyncMock(side_effect=_blocking_panel)

        mock_engine = MagicMock()
        mock_engine.assess.return_value = _make_assessment(confidence=0.85)

        def _make_check() -> object:
            mock_repo = AsyncMock()
            mock_repo.query_range = AsyncMock(return_value=self._events())
            mock_result = MagicMock()
            mock_result.skipped = False
            mock_svc = MagicMock()
            mock_svc.maybe_intervene = AsyncMock(return_value=mock_result)
            return _auto_intervention_check(
                mock_repo,
                mock_svc,
                rule_engine=mock_engine,
                panel_service=mock_panel,  # type: ignore[arg-type]
            )

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            t1 = asyncio.create_task(_make_check())
            t2 = asyncio.create_task(_make_check())
            # Let both tasks reach the claim/await point, then release the panel.
            await asyncio.sleep(0)
            gate.set()
            await asyncio.gather(t1, t2)

        # Only one task won the claim, so the panel ran exactly once.
        assert call_count["n"] == 1
        _DAILY_PANEL_RUN_DATES.discard("2026-07-17")

    async def test_failed_panel_releases_claim_for_retry(self) -> None:
        """When the panel fails, the claim is released so a later tick can retry."""
        from mindflow.agents.types import PanelUnavailableError
        from mindflow.services.scheduler import _DAILY_PANEL_RUN_DATES

        _DAILY_PANEL_RUN_DATES.discard("2026-07-17")

        mock_panel = MagicMock()
        mock_panel.run_daily_panel = AsyncMock(
            side_effect=PanelUnavailableError(reason="down")
        )
        mock_engine = MagicMock()
        mock_engine.assess.return_value = _make_assessment(confidence=0.85)

        mock_repo = AsyncMock()
        mock_repo.query_range = AsyncMock(return_value=self._events())
        mock_svc = MagicMock()
        mock_svc.maybe_intervene = AsyncMock(return_value=MagicMock(skipped=False))

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(
                mock_repo,
                mock_svc,
                rule_engine=mock_engine,
                panel_service=mock_panel,  # type: ignore[arg-type]
            )

        # Claim released → date is NOT stuck as "already run".
        assert "2026-07-17" not in _DAILY_PANEL_RUN_DATES
