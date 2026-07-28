"""Scheduler shutdown and persistent claim integration tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mindflow.services.scheduler as scheduler_module
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
    procrastination_analyses,
)
from mindflow.infrastructure.repositories.scheduled_jobs import (
    ScheduledJobRunsRepository,
    scheduled_job_runs,
)
from mindflow.services.panel_service import PanelService
from mindflow.services.scheduler import AsyncioScheduler, _run_claimed_job, build_scheduler


def _job_coro(scheduler: AsyncioScheduler, name: str):
    return next(job["coro"] for job in scheduler._jobs if job["name"] == name)


async def test_shutdown_cancels_and_awaits_tasks() -> None:
    scheduler = AsyncioScheduler(timezone="UTC")
    scheduler.interval_minutes(1, AsyncMock(), name="waiter")
    scheduler.start()
    tasks = list(scheduler._tasks)
    await scheduler.shutdown()
    assert tasks and all(task.done() for task in tasks)
    assert scheduler._tasks == []


async def test_shutdown_cancels_and_awaits_startup_recovery() -> None:
    scheduler = AsyncioScheduler(timezone="UTC")
    recovery_started = asyncio.Event()
    recovery_finalized = asyncio.Event()

    async def _recover(_: datetime) -> None:
        recovery_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            recovery_finalized.set()

    scheduler._startup_recovery = _recover
    scheduler.start()
    await asyncio.wait_for(recovery_started.wait(), timeout=1.0)
    tasks = list(scheduler._tasks)

    await scheduler.shutdown()

    assert recovery_finalized.is_set()
    assert tasks and all(task.done() for task in tasks)
    assert scheduler._tasks == []


async def test_daily_cron_catches_up_once_after_scheduled_time() -> None:
    scheduler = AsyncioScheduler(timezone="UTC")
    scheduler._running = True
    job = AsyncMock(side_effect=lambda: setattr(scheduler, "_running", False))

    with patch("mindflow.services.scheduler.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 7, 26, 4, 1, tzinfo=UTC)
        await scheduler._run_daily_cron(4, 0, job, None, "daily_backup", catch_up=True)

    job.assert_awaited_once_with()


async def test_daily_cron_does_not_catch_up_before_scheduled_time() -> None:
    scheduler = AsyncioScheduler(timezone="UTC")
    scheduler._running = True
    job = AsyncMock()

    async def _stop_on_sleep(_: float) -> None:
        scheduler._running = False

    with (
        patch("mindflow.services.scheduler.datetime") as mock_datetime,
        patch("mindflow.services.scheduler.asyncio.sleep", side_effect=_stop_on_sleep),
    ):
        mock_datetime.now.return_value = datetime(2026, 7, 26, 3, 59, tzinfo=UTC)
        await scheduler._run_daily_cron(4, 0, job, None, "daily_backup", catch_up=True)

    job.assert_not_awaited()


async def test_persistent_daily_jobs_are_registered_for_catch_up() -> None:
    runs = AsyncMock()
    panel = MagicMock()
    panel.run_daily_panel = AsyncMock()
    report = MagicMock()
    report.generate_daily_for_all = AsyncMock()
    maintenance = MagicMock()
    maintenance.run_daily_backup = AsyncMock()
    maintenance.cleanup_old_events = AsyncMock()

    scheduler = build_scheduler(
        panel_service=panel,
        report_service=report,
        maintenance_service=maintenance,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )

    catch_up_by_name = {job["name"]: job["catch_up"] for job in scheduler._jobs}
    assert catch_up_by_name["daily_panel"] is False
    assert catch_up_by_name["daily_report"] is False
    assert catch_up_by_name["daily_backup"] is True
    assert catch_up_by_name["event_cleanup"] is False


async def test_daily_report_and_backup_use_retryable_persistent_claims() -> None:
    runs = AsyncMock()
    runs.claim.return_value = 1
    runs.mark_succeeded.return_value = True
    report = MagicMock()
    report.generate_daily_for_all = AsyncMock()
    maintenance = MagicMock()
    maintenance.run_daily_backup = AsyncMock()
    scheduler = build_scheduler(
        report_service=report,
        maintenance_service=maintenance,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )
    with patch("mindflow.services.scheduler.business_today", return_value=date(2026, 7, 26)):
        await _job_coro(scheduler, "daily_report")()
        await _job_coro(scheduler, "daily_backup")()
    assert [call.args[0] for call in runs.claim.await_args_list] == ["daily_report", "daily_backup"]
    assert all(call.kwargs == {"retry_failed": True} for call in runs.claim.await_args_list)
    assert runs.claim.await_args_list[0].args[1] == date(2026, 7, 25)
    report.generate_daily_for_all.assert_awaited_once_with(
        target_date=date(2026, 7, 25),
        refresh=True,
    )
    assert runs.mark_succeeded.await_count == 2
    assert all(call.kwargs == {"attempt_count": 1} for call in runs.mark_succeeded.await_args_list)


async def test_failed_backup_result_marks_claim_failed() -> None:
    runs = AsyncMock()
    runs.claim.return_value = 1
    runs.mark_failed.return_value = True
    maintenance = MagicMock()
    maintenance.run_daily_backup = AsyncMock(return_value=False)
    scheduler = build_scheduler(
        maintenance_service=maintenance,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )

    with (
        patch("mindflow.services.scheduler.business_today", return_value=date(2026, 7, 26)),
        pytest.raises(RuntimeError, match="Daily backup failed"),
    ):
        await _job_coro(scheduler, "daily_backup")()

    runs.mark_failed.assert_awaited_once_with(
        "daily_backup",
        date(2026, 7, 26),
        attempt_count=1,
        error="Daily backup failed",
    )
    runs.mark_succeeded.assert_not_awaited()


async def test_daily_report_ensures_identify_succeeds_first() -> None:
    calls: list[str] = []
    runs = AsyncMock()
    runs.claim.side_effect = [1, 1]
    runs.mark_succeeded.return_value = True
    analysis = MagicMock()
    analysis.identify_focus_sessions = AsyncMock(
        side_effect=lambda *_, **__: calls.append("identify")
    )
    report = MagicMock()
    report.generate_daily_for_all = AsyncMock(
        side_effect=lambda **_: calls.append("report")
    )
    scheduler = build_scheduler(
        analysis_service=analysis,
        report_service=report,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )

    with patch("mindflow.services.scheduler.business_today", return_value=date(2026, 7, 26)):
        await _job_coro(scheduler, "daily_report")()

    assert calls == ["identify", "report"]
    assert [call.args[:2] for call in runs.claim.await_args_list] == [
        ("identify_sessions_final", date(2026, 7, 25)),
        ("daily_report", date(2026, 7, 25)),
    ]
    report.generate_daily_for_all.assert_awaited_once_with(
        target_date=date(2026, 7, 25),
        refresh=True,
    )


async def test_final_report_refreshes_after_provisional_identification() -> None:
    claimed: set[tuple[str, date]] = set()
    runs = AsyncMock()

    async def _claim(job_name: str, target_date: date, **_: object) -> int | None:
        key = (job_name, target_date)
        if key in claimed:
            return None
        claimed.add(key)
        return 1

    runs.claim.side_effect = _claim
    runs.has_succeeded.side_effect = lambda name, target: (name, target) in claimed
    runs.mark_succeeded.return_value = True
    analysis = MagicMock()
    analysis.identify_focus_sessions = AsyncMock()
    report = MagicMock()
    report.generate_daily_for_all = AsyncMock()
    scheduler = build_scheduler(
        analysis_service=analysis,
        report_service=report,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )

    with patch(
        "mindflow.services.scheduler.business_today",
        side_effect=[date(2026, 7, 25), date(2026, 7, 26)],
    ):
        await _job_coro(scheduler, "identify_sessions")()
        await _job_coro(scheduler, "daily_report")()

    assert analysis.identify_focus_sessions.await_count == 2
    assert [
        call.args for call in analysis.identify_focus_sessions.await_args_list
    ] == [
        (1, date(2026, 7, 25)),
        (1, date(2026, 7, 25)),
    ]
    assert all(
        call.kwargs == {"refresh": True}
        for call in analysis.identify_focus_sessions.await_args_list
    )
    assert ("identify_sessions", date(2026, 7, 25)) in claimed
    assert ("identify_sessions_final", date(2026, 7, 25)) in claimed


async def test_successful_persistent_job_is_not_repeated() -> None:
    runs = AsyncMock()
    runs.claim.side_effect = [1, None]
    runs.mark_succeeded.return_value = True
    report = MagicMock()
    report.generate_daily_for_all = AsyncMock()
    scheduler = build_scheduler(
        report_service=report, scheduled_job_runs_repository=runs, timezone="UTC"
    )

    with patch("mindflow.services.scheduler.business_today", return_value=date(2026, 7, 26)):
        await _job_coro(scheduler, "daily_report")()
        await _job_coro(scheduler, "daily_report")()

    report.generate_daily_for_all.assert_awaited_once_with(
        target_date=date(2026, 7, 25),
        refresh=True,
    )
    runs.mark_succeeded.assert_awaited_once_with(
        "daily_report", date(2026, 7, 25), attempt_count=1
    )


async def test_failed_claimed_job_is_recorded_for_retry() -> None:
    runs = AsyncMock()
    runs.claim.return_value = 3
    runs.mark_failed.return_value = True
    report = MagicMock()
    report.generate_daily_for_all = AsyncMock(side_effect=RuntimeError("boom"))
    scheduler = build_scheduler(
        report_service=report, scheduled_job_runs_repository=runs, timezone="UTC"
    )
    with (
        patch("mindflow.services.scheduler.business_today", return_value=date(2026, 7, 26)),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await _job_coro(scheduler, "daily_report")()
    runs.mark_failed.assert_awaited_once_with(
        "daily_report", date(2026, 7, 25), attempt_count=3, error="boom"
    )


async def test_cancelled_claimed_job_is_marked_cancelled_and_propagated() -> None:
    runs = AsyncMock()
    runs.claim.return_value = 2
    runs.mark_cancelled.return_value = True
    job = AsyncMock(side_effect=asyncio.CancelledError)

    with pytest.raises(asyncio.CancelledError):
        await _run_claimed_job(
            runs,
            "daily_backup",
            date(2026, 7, 26),
            job,
            retry_failed=True,
        )

    runs.mark_cancelled.assert_awaited_once_with(
        "daily_backup", date(2026, 7, 26), attempt_count=2
    )
    runs.mark_failed.assert_not_awaited()
    runs.mark_succeeded.assert_not_awaited()


async def test_cancellation_during_success_commit_marks_claim_cancelled() -> None:
    runs = AsyncMock()
    runs.claim.return_value = 1
    runs.mark_cancelled.return_value = True
    commit_started = asyncio.Event()

    async def _mark_succeeded(*_args: object, **_kwargs: object) -> bool:
        commit_started.set()
        await asyncio.Event().wait()
        return True

    runs.mark_succeeded.side_effect = _mark_succeeded
    task = asyncio.create_task(
        _run_claimed_job(
            runs,
            "daily_report",
            date(2026, 7, 26),
            AsyncMock(return_value=None),
            retry_failed=True,
        )
    )
    await asyncio.wait_for(commit_started.wait(), timeout=1.0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    runs.mark_cancelled.assert_awaited_once_with(
        "daily_report",
        date(2026, 7, 26),
        attempt_count=1,
    )


async def test_ownership_loss_cancels_job_and_discards_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = AsyncMock()
    runs.claim.return_value = 1
    runs.heartbeat.return_value = False
    cancelled = asyncio.Event()

    async def _job() -> str:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "stale"

    monkeypatch.setattr(
        scheduler_module,
        "_SCHEDULED_JOB_HEARTBEAT_INTERVAL_SECONDS",
        0.0,
    )

    claimed, result = await _run_claimed_job(
        runs,
        "daily_report",
        date(2026, 7, 26),
        _job,
        retry_failed=True,
    )

    assert claimed is False
    assert result is None
    assert cancelled.is_set()
    runs.mark_succeeded.assert_not_awaited()
    runs.mark_failed.assert_not_awaited()


async def test_heartbeat_error_cancels_and_awaits_business_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = AsyncMock()
    runs.claim.return_value = 1
    runs.heartbeat.side_effect = RuntimeError("heartbeat failed")
    runs.mark_failed.return_value = True
    finalized = asyncio.Event()
    tasks_before = set(asyncio.all_tasks())

    async def _job() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    monkeypatch.setattr(
        scheduler_module,
        "_SCHEDULED_JOB_HEARTBEAT_INTERVAL_SECONDS",
        0.0,
    )

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        await _run_claimed_job(
            runs,
            "daily_report",
            date(2026, 7, 26),
            _job,
            retry_failed=True,
        )

    assert finalized.is_set()
    leaked_tasks = {
        task
        for task in asyncio.all_tasks()
        if task not in tasks_before and not task.done()
    }
    assert leaked_tasks == set()


async def test_cancellation_during_failed_commit_marks_claim_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = AsyncMock()
    runs.claim.return_value = 1
    runs.heartbeat.side_effect = RuntimeError("heartbeat failed")
    runs.mark_cancelled.return_value = True
    failed_commit_started = asyncio.Event()

    async def _mark_failed(*_args: object, **_kwargs: object) -> bool:
        failed_commit_started.set()
        await asyncio.Event().wait()
        return True

    runs.mark_failed.side_effect = _mark_failed

    async def _job() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        scheduler_module,
        "_SCHEDULED_JOB_HEARTBEAT_INTERVAL_SECONDS",
        0.0,
    )
    task = asyncio.create_task(
        _run_claimed_job(
            runs,
            "daily_report",
            date(2026, 7, 26),
            _job,
            retry_failed=True,
        )
    )
    await asyncio.wait_for(failed_commit_started.wait(), timeout=1.0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    runs.mark_cancelled.assert_awaited_once_with(
        "daily_report",
        date(2026, 7, 26),
        attempt_count=1,
    )


async def test_repeated_recovery_with_real_claim_does_not_repeat_panel(
    engine,
    session_factory,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(scheduled_job_runs.metadata.create_all)
        await connection.run_sync(procrastination_analyses.metadata.create_all)
    runs = ScheduledJobRunsRepository(session_factory)
    analysis_repository = SQLAlchemyProcrastinationAnalysisRepository(session_factory)
    panel_service = PanelService.__new__(PanelService)
    panel_service._builder = AsyncMock()
    panel_service._orchestrator = AsyncMock()
    panel_service._llm_service = AsyncMock()
    panel_service._analysis_repository = analysis_repository
    panel_service._timezone = "UTC"

    from mindflow.agents.types import PanelVerdict, TranscriptEntry
    from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType

    expected = PanelVerdict(
        types=(ProcrastinationType.IMPULSIVITY,),
        confidence={ProcrastinationType.IMPULSIVITY: 0.85},
        recommended_technique=CBTTechnique.STIMULUS_CONTROL,
        rationale="测试会诊结果",
        dissent=(),
        transcript=(
            TranscriptEntry(role="数据分析师", content="模式分析完成", round=0),
        ),
        escalated=False,
        call_count=6,
        source="panel",
    )
    panel_service._orchestrator.run.return_value = expected
    scheduler = build_scheduler(
        panel_service=panel_service,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )
    startup_time = datetime(2026, 7, 26, 0, 10, tzinfo=UTC)

    await scheduler.run_startup_recovery(now_utc=startup_time)
    await scheduler.run_startup_recovery(now_utc=startup_time)

    panel_service._orchestrator.run.assert_awaited_once()
    assert await panel_service.get_stored_verdict(1, date(2026, 7, 25)) == expected


async def test_startup_recovery_and_cron_share_real_panel_claim(
    engine,
    session_factory,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(scheduled_job_runs.metadata.create_all)
    runs = ScheduledJobRunsRepository(session_factory)
    panel_service = MagicMock()
    target_dates: list[date] = []

    async def _run_panel(*, user_id: int, target_date: date) -> object:
        assert user_id == 1
        target_dates.append(target_date)
        await asyncio.sleep(0.01)
        return object()

    panel_service.run_daily_panel = AsyncMock(side_effect=_run_panel)
    scheduler = build_scheduler(
        panel_service=panel_service,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )
    startup_time = datetime(2026, 7, 26, 23, 31, tzinfo=UTC)

    await asyncio.gather(
        scheduler.run_startup_recovery(now_utc=startup_time),
        _job_coro(scheduler, "daily_panel")(),
    )

    assert target_dates.count(date(2026, 7, 25)) == 1
    assert target_dates.count(date(2026, 7, 26)) == 1


async def test_startup_recovery_continues_panel_after_identify_failure() -> None:
    runs = AsyncMock()
    runs.claim.return_value = 1
    runs.mark_failed.return_value = True
    analysis = MagicMock()
    analysis.identify_focus_sessions = AsyncMock(side_effect=RuntimeError("identify failed"))
    panel = MagicMock()
    panel.run_daily_panel = AsyncMock(return_value=object())
    report = MagicMock()
    report.generate_daily_for_all = AsyncMock()
    scheduler = build_scheduler(
        analysis_service=analysis,
        panel_service=panel,
        report_service=report,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )

    await scheduler.run_startup_recovery(
        now_utc=datetime(2026, 7, 26, 23, 31, tzinfo=UTC)
    )

    assert [call.kwargs["target_date"] for call in panel.run_daily_panel.await_args_list] == [
        date(2026, 7, 25),
        date(2026, 7, 26),
    ]
    report.generate_daily_for_all.assert_not_awaited()


async def test_startup_recovery_continues_current_panel_after_report_failure() -> None:
    runs = AsyncMock()
    runs.claim.return_value = 1
    runs.mark_failed.return_value = True
    panel = MagicMock()
    panel.run_daily_panel = AsyncMock(return_value=object())
    report = MagicMock()
    report.generate_daily_for_all = AsyncMock(side_effect=RuntimeError("report failed"))
    scheduler = build_scheduler(
        panel_service=panel,
        report_service=report,
        scheduled_job_runs_repository=runs,
        timezone="UTC",
    )

    await scheduler.run_startup_recovery(
        now_utc=datetime(2026, 7, 26, 23, 31, tzinfo=UTC)
    )

    assert [call.kwargs["target_date"] for call in panel.run_daily_panel.await_args_list] == [
        date(2026, 7, 25),
        date(2026, 7, 26),
    ]
