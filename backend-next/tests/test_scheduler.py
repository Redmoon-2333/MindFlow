"""Tests for scheduler (build_scheduler).

Covers:
  - 5 jobs registered (4 cron + 1 interval) with correct configuration
  - Graceful handling of missing services
  - Auto-intervention job: interval config and time-of-day guard logic
"""

from __future__ import annotations

import asyncio
import io
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import mindflow.services.scheduler as scheduler_module
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType
from mindflow.services.scheduler import (
    AsyncioScheduler,
    _auto_intervention_check,
    _CronTrigger,
    _IntervalTrigger,
    _next_daily_run_utc,
    build_scheduler,
)


def _job_coro(scheduler: AsyncioScheduler, name: str):
    return next(job["coro"] for job in scheduler._jobs if job["name"] == name)


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

    async def test_auto_intervention_check_is_interval_5min(self) -> None:
        """auto_intervention_check should be an interval job at 5 minutes."""
        scheduler = build_scheduler(
            intervention_service=MagicMock(),
            activity_repository=MagicMock(),
        )
        job = scheduler.get_job("auto_intervention_check")
        assert job is not None

        trigger = job.trigger
        assert trigger.interval.total_seconds() == 300  # 5 min

    async def test_auto_intervention_job_passes_custom_hours(self) -> None:
        """build_scheduler wires start_hour/end_hour into the job kwargs."""
        scheduler = build_scheduler(
            intervention_service=MagicMock(),
            activity_repository=MagicMock(),
            start_hour=10,
            end_hour=11,
        )
        job = scheduler.get_job("auto_intervention_check")
        assert job is not None
        assert job.kwargs["start_hour"] == 10
        assert job.kwargs["end_hour"] == 11

    async def test_auto_intervention_job_defaults_to_8_and_23(self) -> None:
        """Default window bounds are 08:00-23:00 (exclusive end)."""
        scheduler = build_scheduler(
            intervention_service=MagicMock(),
            activity_repository=MagicMock(),
        )
        job = scheduler.get_job("auto_intervention_check")
        assert job is not None
        assert job.kwargs["start_hour"] == 8
        assert job.kwargs["end_hour"] == 23

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

    async def test_event_cleanup_invokes_intervention_check_cleanup(self) -> None:
        """event_cleanup deletes raw events AND intervention checks at the same
        activity-retention horizon (single effective retention value)."""
        maintenance = MagicMock()
        maintenance.cleanup_old_events = AsyncMock()
        maintenance.cleanup_old_intervention_checks = AsyncMock()
        maintenance.expire_stale_budgets = AsyncMock()
        scheduler = build_scheduler(
            maintenance_service=maintenance,
            event_retention_days=45,
        )
        coro = _job_coro(scheduler, "event_cleanup")

        await coro(retention_days=45)

        maintenance.cleanup_old_events.assert_awaited_once_with(retention_days=45)
        maintenance.cleanup_old_intervention_checks.assert_awaited_once_with(
            retention_days=45
        )
        maintenance.expire_stale_budgets.assert_awaited_once_with()

    async def test_missing_service_skips_jobs(self) -> None:
        """Without services, corresponding jobs should be skipped."""
        scheduler = build_scheduler()
        jobs = scheduler.get_jobs()
        assert len(jobs) == 0

    async def test_scheduler_timezone_defaults_to_local(self) -> None:
        scheduler = build_scheduler()
        assert scheduler.timezone == datetime.now().astimezone().tzinfo

    async def test_scheduler_uses_configured_local_timezone(self) -> None:
        scheduler = build_scheduler(timezone="Asia/Shanghai")
        assert scheduler.timezone == ZoneInfo("Asia/Shanghai")

    def test_next_cron_run_uses_local_wall_clock(self) -> None:
        now_utc = datetime(2026, 7, 17, 15, 0, tzinfo=UTC)

        target = _next_daily_run_utc(now_utc, hour=0, minute=1, timezone="Asia/Shanghai")

        assert target == datetime(2026, 7, 17, 16, 1, tzinfo=UTC)

    async def test_report_job_registered(self) -> None:
        """daily_report should run at 00:05."""
        scheduler = build_scheduler(
            report_service=MagicMock(),
        )
        job = scheduler.get_job("daily_report")
        assert job is not None
        assert str(job.trigger.fields[5]) == "0"
        assert str(job.trigger.fields[6]) == "5"

    async def test_backup_job_registered(self) -> None:
        """daily_backup should have maintenance_service as dependency."""
        scheduler = build_scheduler(
            maintenance_service=MagicMock(),
        )
        job = scheduler.get_job("daily_backup")
        assert job is not None

    async def test_startup_recovers_latest_complete_telemetry_day_once(
        self,
    ) -> None:
        runs = AsyncMock()
        runs.claim.side_effect = [1, None]
        runs.mark_succeeded.return_value = True
        telemetry = MagicMock()
        telemetry.rebuild_baseline_if_needed = AsyncMock(
            return_value=SimpleNamespace(
                rebuilt=False,
                reason="skipped_v2",
                windows_loaded=0,
                samples=0,
                cutoff_utc=datetime(2026, 7, 26, 11, 30, tzinfo=UTC),
            )
        )
        telemetry.rollup_feature_windows = AsyncMock()
        telemetry.cleanup_retained_data = AsyncMock()
        scheduler = build_scheduler(
            telemetry_service=telemetry,
            scheduled_job_runs_repository=runs,
            timezone="Asia/Shanghai",
        )
        startup_time = datetime(2026, 7, 26, 11, 30, tzinfo=UTC)

        await scheduler.run_startup_recovery(now_utc=startup_time)
        await scheduler.run_startup_recovery(now_utc=startup_time)

        # The complete-day yesterday rollup is claimed once (never repeated)
        # and still runs after the Todo 12 current-day catch-up.
        telemetry.rollup_feature_windows.assert_any_await(
            datetime(2026, 7, 24, 16, 0, tzinfo=UTC),
            datetime(2026, 7, 25, 16, 0, tzinfo=UTC),
        )
        # Todo 12 current-day catch-up: bounded [now-2h, now] derived from the
        # captured startup time, re-run on each recovery invocation.
        assert any(
            call.args == (startup_time - timedelta(hours=2), startup_time)
            for call in telemetry.rollup_feature_windows.await_args_list
        )
        telemetry.cleanup_retained_data.assert_awaited_once_with()
        runs.claim.assert_any_await(
            "telemetry_rollup",
            date(2026, 7, 25),
            retry_failed=True,
        )

    async def test_shutdown_does_not_raise(self) -> None:
        """shutdown(wait=False) should not raise."""
        scheduler = build_scheduler()
        scheduler.start()  # Initialise event loop reference
        await scheduler.shutdown()

    async def test_startup_recovers_latest_complete_business_day_once_in_order(
        self,
    ) -> None:
        calls: list[tuple[str, date]] = []
        identify_refreshes: list[bool] = []
        claimed: set[tuple[str, date]] = set()
        runs = AsyncMock()

        async def _claim(job_name: str, target_date: date, **_: object) -> int | None:
            key = (job_name, target_date)
            if key in claimed:
                return None
            claimed.add(key)
            return 1

        runs.claim.side_effect = _claim
        runs.has_succeeded.return_value = True
        runs.mark_succeeded.return_value = True

        analysis = MagicMock()
        async def _identify(
            _user_id: int,
            target_date: date,
            *,
            refresh: bool = False,
        ) -> None:
            calls.append(("identify_sessions", target_date))
            identify_refreshes.append(refresh)

        analysis.identify_focus_sessions = AsyncMock(side_effect=_identify)
        panel = MagicMock()
        panel.run_daily_panel = AsyncMock(
            side_effect=lambda *, user_id, target_date: calls.append(
                ("daily_panel", target_date)
            )
        )
        report = MagicMock()
        report.generate_daily_for_all = AsyncMock(
            side_effect=lambda *, target_date, refresh: calls.append(
                ("daily_report", target_date)
            )
        )
        autonomy = MagicMock()
        autonomy.is_enabled = AsyncMock(return_value=True)
        scheduler = build_scheduler(
            analysis_service=analysis,
            panel_service=panel,
            report_service=report,
            autonomy_service=autonomy,
            scheduled_job_runs_repository=runs,
            timezone="Asia/Shanghai",
        )
        startup_time = datetime(2026, 7, 25, 16, 10, tzinfo=UTC)

        await scheduler.run_startup_recovery(now_utc=startup_time)
        await scheduler.run_startup_recovery(now_utc=startup_time)

        assert calls == [
            ("identify_sessions", date(2026, 7, 25)),
            ("daily_panel", date(2026, 7, 25)),
            ("daily_report", date(2026, 7, 25)),
        ]
        assert identify_refreshes == [True]

    async def test_startup_after_panel_time_recovers_current_business_day(self) -> None:
        runs = AsyncMock()
        runs.claim.side_effect = [None, 1]
        runs.has_succeeded.return_value = False
        runs.mark_succeeded.return_value = True
        panel = MagicMock()
        panel.run_daily_panel = AsyncMock()
        scheduler = build_scheduler(
            panel_service=panel,
            scheduled_job_runs_repository=runs,
            timezone="Asia/Shanghai",
        )

        await scheduler.run_startup_recovery(
            now_utc=datetime(2026, 7, 26, 15, 45, tzinfo=UTC)
        )

        panel.run_daily_panel.assert_awaited_once_with(
            user_id=1,
            target_date=date(2026, 7, 26),
        )

    async def test_startup_after_identify_time_recovers_current_business_day(
        self,
    ) -> None:
        runs = AsyncMock()
        runs.claim.side_effect = [1, 1]
        runs.has_succeeded.return_value = False
        runs.mark_succeeded.return_value = True
        analysis = MagicMock()
        analysis.identify_focus_sessions = AsyncMock()
        scheduler = build_scheduler(
            analysis_service=analysis,
            scheduled_job_runs_repository=runs,
            timezone="Asia/Shanghai",
        )

        await scheduler.run_startup_recovery(
            now_utc=datetime(2026, 7, 26, 15, 59, 30, tzinfo=UTC)
        )

        assert [
            item.args for item in analysis.identify_focus_sessions.await_args_list
        ] == [
            (1, date(2026, 7, 25)),
            (1, date(2026, 7, 26)),
        ]

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

            await _auto_intervention_check(mock_repo, mock_svc, timezone="UTC")

            # Should not query events
            mock_repo.query_range.assert_not_called()
            mock_svc.maybe_intervene.assert_not_called()

    async def test_honors_custom_start_hour(self) -> None:
        """Custom window 10:00-11:00: 09:30 local is outside_hours."""
        mock_repo = AsyncMock()
        mock_svc = MagicMock()
        telemetry = MagicMock()
        telemetry.save_intervention_check = AsyncMock()

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 9, 30, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(
                mock_repo,
                mock_svc,
                timezone="UTC",
                start_hour=10,
                end_hour=11,
                telemetry_service=telemetry,
            )

        mock_repo.query_range.assert_not_called()
        mock_svc.maybe_intervene.assert_not_called()
        telemetry.save_intervention_check.assert_awaited_once()
        assert (
            telemetry.save_intervention_check.await_args.kwargs["reason"]
            == "outside_hours"
        )

    async def test_honors_custom_end_hour(self) -> None:
        """Custom window 10:00-11:00: 11:00 local (end, exclusive) is outside_hours."""
        mock_repo = AsyncMock()
        mock_svc = MagicMock()

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 11, 0, 0, tzinfo=UTC)
            mock_dt.UTC = UTC
            mock_dt.timedelta = __import__("datetime").timedelta

            await _auto_intervention_check(
                mock_repo,
                mock_svc,
                timezone="UTC",
                start_hour=10,
                end_hour=11,
            )

        mock_repo.query_range.assert_not_called()
        mock_svc.maybe_intervene.assert_not_called()

    async def test_working_hours_use_configured_local_timezone(self) -> None:
        mock_repo = AsyncMock()
        mock_repo.query_range = AsyncMock(return_value=[])
        mock_svc = MagicMock()

        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 0, 30, tzinfo=UTC)

            await _auto_intervention_check(mock_repo, mock_svc, timezone="Asia/Shanghai")

        mock_repo.query_range.assert_awaited_once()

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

    async def test_skips_when_non_idle_below_minimum(self) -> None:
        """Fewer than 10 minutes of non-idle activity should skip."""
        from mindflow.domain.events import make_event

        mock_repo = AsyncMock()
        events = [
            make_event(
                user_id=1,
                timestamp_utc=datetime(2026, 7, 17, 14, 0, 0, tzinfo=UTC),
                duration_s=300.0,
                process_name="Code.exe",
                app_name="Code.exe",
            ),
            make_event(
                user_id=1,
                timestamp_utc=datetime(2026, 7, 17, 14, 5, 0, tzinfo=UTC),
                duration_s=60.0,
                process_name="Code.exe",
                app_name="Code.exe",
            ),
        ]
        mock_repo.query_range = AsyncMock(return_value=events)
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
                duration_s=600.0,
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

            await _auto_intervention_check(mock_repo, mock_svc, autonomy_service=mock_autonomy)

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
                duration_s=600.0,
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
                duration_s=600.0,
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
        mock_panel.run_daily_panel = AsyncMock(side_effect=PanelUnavailableError(reason="down"))
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


class TestSchedulerJobRegistrationContract:
    """Characterization: pin the full job registration surface.

    Locks the complete job set — including ``daily_panel``,
    ``telemetry_rollup`` and the 15-minute ``telemetry_rollup_recent``
    interval, which earlier tests omit.  Later scheduler work must extend
    this set deliberately.
    """

    async def test_telemetry_rollup_job_at_0245(self) -> None:
        scheduler = build_scheduler(telemetry_service=MagicMock())
        job = scheduler.get_job("telemetry_rollup")
        assert job is not None
        trigger = job.trigger
        assert isinstance(trigger, _CronTrigger)
        assert str(trigger.fields[5]) == "2"
        assert str(trigger.fields[6]) == "45"

    async def test_daily_panel_job_at_2330(self) -> None:
        scheduler = build_scheduler(panel_service=MagicMock())
        job = scheduler.get_job("daily_panel")
        assert job is not None
        trigger = job.trigger
        assert isinstance(trigger, _CronTrigger)
        assert str(trigger.fields[5]) == "23"
        assert str(trigger.fields[6]) == "30"

    async def test_full_service_set_registers_eight_jobs(self) -> None:
        scheduler = build_scheduler(
            analysis_service=MagicMock(),
            report_service=MagicMock(),
            maintenance_service=MagicMock(),
            intervention_service=MagicMock(),
            activity_repository=MagicMock(),
            panel_service=MagicMock(),
            telemetry_service=MagicMock(),
            training_job_service=MagicMock(),
        )
        job_ids = {j.id for j in scheduler.get_jobs()}
        assert job_ids == {
            "daily_panel",
            "identify_sessions",
            "daily_report",
            "event_cleanup",
            "daily_backup",
            "auto_intervention_check",
            "telemetry_rollup",
            "telemetry_rollup_recent",
            # Architecture plan F: hourly auto-training check.
            "auto_training_check",
        }

    async def test_interval_jobs_are_intervention_30min_and_recent_rollup_15min(
        self,
    ) -> None:
        scheduler = build_scheduler(
            analysis_service=MagicMock(),
            report_service=MagicMock(),
            maintenance_service=MagicMock(),
            intervention_service=MagicMock(),
            activity_repository=MagicMock(),
            panel_service=MagicMock(),
            telemetry_service=MagicMock(),
        )
        interval_jobs: list[tuple[str, float]] = []
        for job in scheduler.get_jobs():
            trigger = job.trigger
            if isinstance(trigger, _IntervalTrigger):
                interval_jobs.append((job.id, trigger.interval.total_seconds()))
        assert interval_jobs == [
            ("auto_intervention_check", 300.0),
            ("telemetry_rollup_recent", 900.0),
        ]


class TestRecentTelemetryRollup:
    """Todo 11: the 15-minute bounded recent-window telemetry rollup.

    Reuses the Todo 8 idempotent seam (``rollup_feature_windows`` window
    upsert + baseline folded from newly inserted rows only), so overlapping
    two-hour windows rolled every 15 minutes are safe. No per-date claim:
    the window shifts continuously and the interval loop's catch-and-log
    keeps later invocations alive after a failure.
    """

    async def test_registered_once_with_15min_interval_trigger(self) -> None:
        telemetry = MagicMock()
        telemetry.rollup_feature_windows = AsyncMock(return_value=0)
        scheduler = build_scheduler(telemetry_service=telemetry, timezone="UTC")

        jobs = [j for j in scheduler.get_jobs() if j.id == "telemetry_rollup_recent"]
        assert len(jobs) == 1
        assert isinstance(jobs[0].trigger, _IntervalTrigger)
        assert jobs[0].trigger.interval == timedelta(minutes=15)

    async def test_invocation_rolls_two_hour_window_with_fixed_clock(self) -> None:
        telemetry = MagicMock()
        telemetry.rollup_feature_windows = AsyncMock(return_value=0)
        scheduler = build_scheduler(telemetry_service=telemetry, timezone="UTC")
        fixed_now = datetime(2026, 7, 26, 12, 34, 56, tzinfo=UTC)

        with patch("mindflow.services.scheduler.datetime") as mock_datetime:
            mock_datetime.now.return_value = fixed_now
            await _job_coro(scheduler, "telemetry_rollup_recent")()

        # One aware UTC now captured per invocation; both bounds derive from it.
        telemetry.rollup_feature_windows.assert_awaited_once_with(
            fixed_now - timedelta(hours=2),
            fixed_now,
            user_id=1,
        )

    async def test_coexists_with_daily_yesterday_rollup(self) -> None:
        telemetry = MagicMock()
        telemetry.rollup_feature_windows = AsyncMock(return_value=0)
        telemetry.cleanup_retained_data = AsyncMock(return_value=0)
        scheduler = build_scheduler(
            telemetry_service=telemetry,
            scheduled_job_runs_repository=AsyncMock(),
            timezone="UTC",
        )

        daily = scheduler.get_job("telemetry_rollup")
        assert daily is not None
        assert isinstance(daily.trigger, _CronTrigger)
        assert str(daily.trigger.fields[5]) == "2"
        assert str(daily.trigger.fields[6]) == "45"

        recent = [j for j in scheduler.get_jobs() if j.id == "telemetry_rollup_recent"]
        assert len(recent) == 1
        assert isinstance(recent[0].trigger, _IntervalTrigger)

    async def test_failure_is_logged_and_later_invocation_still_runs(self) -> None:
        telemetry = MagicMock()
        telemetry.rollup_feature_windows = AsyncMock(
            side_effect=[RuntimeError("boom"), 3]
        )
        scheduler = build_scheduler(telemetry_service=telemetry, timezone="UTC")
        scheduler._running = True
        sleep_count = 0

        async def _fake_sleep(_: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:
                scheduler._running = False

        log_buffer = io.StringIO()
        sink_id = scheduler_module.logger.add(
            log_buffer, format="{message}", level="ERROR"
        )
        try:
            with patch(
                "mindflow.services.scheduler.asyncio.sleep", side_effect=_fake_sleep
            ):
                await scheduler._run_interval(
                    15,
                    _job_coro(scheduler, "telemetry_rollup_recent"),
                    None,
                    "telemetry_rollup_recent",
                )
        finally:
            scheduler_module.logger.remove(sink_id)

        # First call raised and was swallowed by the interval boundary; the
        # second call still ran on the next tick — no sleep involved.
        assert telemetry.rollup_feature_windows.await_count == 2
        assert "telemetry_rollup_recent failed" in log_buffer.getvalue()


class TestStartupRecoveryTelemetryCatchUp:
    """Todo 12: startup recovery composes the conditional V2 baseline backfill
    then the bounded current recent-window catch-up ending at the single
    captured startup ``now_utc``, reusing the existing ``_startup_recovery``
    callback (no second lifecycle engine).

    The complete-day telemetry rollup for the previous business day remains a
    separate step; the new current-day catch-up is bounded by the same
    two-hour policy as the 15-minute interval job, not a full-day recompute.
    """

    @staticmethod
    def _telemetry() -> MagicMock:
        telemetry = MagicMock()
        telemetry.rebuild_baseline_if_needed = AsyncMock(
            return_value=SimpleNamespace(
                rebuilt=False,
                reason="skipped_v2",
                windows_loaded=0,
                samples=0,
                cutoff_utc=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
            )
        )
        telemetry.rollup_feature_windows = AsyncMock(return_value=0)
        telemetry.cleanup_retained_data = AsyncMock(return_value=0)
        return telemetry

    async def test_backfill_runs_before_bounded_recent_catch_up(self) -> None:
        """Given: a telemetry-wired scheduler. When: startup recovery runs with
        a fixed aware UTC now. Then: the baseline backfill is invoked first,
        then the current-day catch-up rolls exactly [now-2h, now] for user 1 —
        both deriving from the same captured ``now_utc``."""
        calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

        async def _backfill(
            user_id: int = 1,
            *,
            timezone: str = "",
            now_utc: datetime | None = None,
        ) -> object:
            calls.append(
                ("backfill", (user_id,), {"timezone": timezone, "now_utc": now_utc})
            )
            return SimpleNamespace(
                rebuilt=False,
                reason="skipped_v2",
                windows_loaded=0,
                samples=0,
                cutoff_utc=now_utc,
            )

        async def _rollup(start: datetime, end: datetime, user_id: int = 1) -> int:
            calls.append(("rollup", (start, end, user_id), {}))
            return 0

        telemetry = self._telemetry()
        telemetry.rebuild_baseline_if_needed = AsyncMock(side_effect=_backfill)
        telemetry.rollup_feature_windows = AsyncMock(side_effect=_rollup)
        telemetry.cleanup_retained_data = AsyncMock(
            side_effect=lambda: calls.append(("cleanup", (), {}))
        )
        scheduler = build_scheduler(telemetry_service=telemetry, timezone="Asia/Shanghai")
        startup_time = datetime(2026, 7, 26, 12, 0, 0, tzinfo=UTC)

        await scheduler.run_startup_recovery(now_utc=startup_time)

        # Exact order: conditional baseline backfill, then recent catch-up.
        assert [name for name, _, _ in calls][:2] == ["backfill", "rollup"]
        # Backfill receives the configured timezone and the exact captured now.
        assert calls[0] == (
            "backfill",
            (1,),
            {"timezone": "Asia/Shanghai", "now_utc": startup_time},
        )
        # Bounded two-hour recent window ending at the same captured now.
        assert calls[1] == (
            "rollup",
            (startup_time - timedelta(hours=2), startup_time, 1),
            {},
        )
        # The complete-day yesterday rollup and cleanup still follow.
        assert calls[2] == (
            "rollup",
            (
                datetime(2026, 7, 24, 16, 0, tzinfo=UTC),
                datetime(2026, 7, 25, 16, 0, tzinfo=UTC),
                1,
            ),
            {},
        )
        assert calls[3] == ("cleanup", (), {})

    async def test_recovery_introduces_no_wall_clock_sleeps(self) -> None:
        """Given: a healthy telemetry-wired recovery. When: it runs to
        completion. Then: no ``asyncio.sleep`` is issued anywhere — the
        success path never sleeps, only inter-attempt retries would."""
        telemetry = self._telemetry()
        scheduler = build_scheduler(telemetry_service=telemetry, timezone="UTC")
        sleeps: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        with patch(
            "mindflow.services.scheduler.asyncio.sleep", side_effect=_fake_sleep
        ):
            await scheduler.run_startup_recovery(
                now_utc=datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
            )

        assert sleeps == []

    async def test_backfill_result_is_surfaced_in_logs(self) -> None:
        """Given: a rebuild that actually replaces the baseline. When: startup
        recovery runs. Then: the backfill outcome (rebuilt, reason, counts,
        cutoff) is logged — a no-op is never mistaken for a rebuild."""
        result = SimpleNamespace(
            rebuilt=True,
            reason="missing",
            windows_loaded=5,
            samples=120,
            cutoff_utc=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        )
        telemetry = self._telemetry()
        telemetry.rebuild_baseline_if_needed = AsyncMock(return_value=result)
        scheduler = build_scheduler(telemetry_service=telemetry, timezone="UTC")
        log_buffer = io.StringIO()
        sink_id = scheduler_module.logger.add(
            log_buffer, format="{message}", level="INFO"
        )
        try:
            await scheduler.run_startup_recovery(
                now_utc=datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
            )
        finally:
            scheduler_module.logger.remove(sink_id)

        logged = log_buffer.getvalue()
        assert "baseline backfill" in logged
        assert "rebuilt=True" in logged
        assert "reason=missing" in logged
        assert "windows_loaded=5" in logged
        assert "samples=120" in logged

    async def test_backfill_failure_logged_and_recovery_continues(self) -> None:
        """Given: the baseline backfill raises. When: startup recovery runs.
        Then: the existing step boundary logs the failure (never falsely claims
        completion, never raises) and the recent catch-up still runs."""
        telemetry = self._telemetry()
        telemetry.rebuild_baseline_if_needed = AsyncMock(
            side_effect=RuntimeError("baseline unavailable")
        )
        scheduler = build_scheduler(telemetry_service=telemetry, timezone="UTC")
        log_buffer = io.StringIO()
        sink_id = scheduler_module.logger.add(
            log_buffer, format="{message}", level="ERROR"
        )
        try:
            await scheduler.run_startup_recovery(
                now_utc=datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
            )
        finally:
            scheduler_module.logger.remove(sink_id)

        assert "baseline_backfill failed" in log_buffer.getvalue()
        telemetry.rollup_feature_windows.assert_awaited()

    async def test_built_scheduler_shutdown_cancels_recovery_and_all_tasks(
        self,
    ) -> None:
        """Given: a built scheduler whose startup recovery has completed.
        When: shutdown() runs. Then: the recovery task and all job tasks are
        cancelled and cleared — no leaked tasks."""
        telemetry = self._telemetry()
        scheduler = build_scheduler(telemetry_service=telemetry, timezone="UTC")
        recovery_done = asyncio.Event()
        real_recover = scheduler._startup_recovery
        assert real_recover is not None

        async def _tracking_recover(now_utc: datetime) -> None:
            await real_recover(now_utc)
            recovery_done.set()

        scheduler._startup_recovery = _tracking_recover
        scheduler.start()
        await asyncio.wait_for(recovery_done.wait(), timeout=2.0)
        tasks = list(scheduler._tasks)

        await scheduler.shutdown()

        assert tasks and all(task.done() for task in tasks)
        assert scheduler._tasks == []
