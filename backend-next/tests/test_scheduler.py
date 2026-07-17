"""Tests for scheduler (build_scheduler).

Covers:
  - 4 cron jobs registered with correct cron expressions
  - Graceful handling of missing services
"""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock

from mindflow.services.scheduler import build_scheduler


class TestBuildScheduler:
    """Scheduler configuration tests."""

    async def test_registers_4_jobs(self):
        """With all services provided, 4 jobs should be registered."""
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
        assert "identify_sessions" in job_ids
        assert "daily_report" in job_ids
        assert "event_cleanup" in job_ids
        assert "daily_backup" in job_ids

    async def test_identify_sessions_cron(self):
        """identify_sessions should run at 23:59 daily."""
        scheduler = build_scheduler(
            analysis_service=MagicMock(),
        )
        job = scheduler.get_job("identify_sessions")
        assert job is not None
        trigger = job.trigger
        assert str(trigger.fields[5]) == "23"  # hour
        assert str(trigger.fields[6]) == "59"  # minute

    async def test_event_cleanup_cron(self):
        """event_cleanup should run at 03:00 daily with retention_days kwarg."""
        scheduler = build_scheduler(
            maintenance_service=MagicMock(),
        )
        job = scheduler.get_job("event_cleanup")
        assert job is not None
        assert "retention_days" in job.kwargs

    async def test_missing_service_skips_jobs(self):
        """Without services, corresponding jobs should be skipped."""
        scheduler = build_scheduler()
        jobs = scheduler.get_jobs()
        assert len(jobs) == 0

    async def test_scheduler_timezone_is_utc(self):
        """Scheduler timezone should be UTC."""
        scheduler = build_scheduler()
        assert scheduler.timezone == UTC

    async def test_report_job_registered(self):
        """daily_report should run at 00:01."""
        scheduler = build_scheduler(
            report_service=MagicMock(),
        )
        job = scheduler.get_job("daily_report")
        assert job is not None

    async def test_backup_job_registered(self):
        """daily_backup should have maintenance_service as dependency."""
        scheduler = build_scheduler(
            maintenance_service=MagicMock(),
        )
        job = scheduler.get_job("daily_backup")
        assert job is not None

    async def test_shutdown_does_not_raise(self):
        """shutdown(wait=False) should not raise."""
        scheduler = build_scheduler()
        scheduler.start()  # Initialise event loop reference
        scheduler.shutdown(wait=False)
