"""APScheduler-based cron job configuration for maintenance tasks.

Per ADR-007, APScheduler is used exclusively for cron-style scheduling
(fixed time-of-day jobs) — not for the high-frequency collector tick loop,
which runs as a bare ``asyncio.create_task`` inside CollectorService.

Registered cron jobs (all times are UTC):
  - 23:59  — ``identify_all_today``: run daily session identification.
  - 00:01  — ``daily_report``: generate today's daily report.
  - 03:00  — ``event_cleanup``: delete raw events past retention policy.
  - 04:00  — ``daily_backup``: crash-consistent VACUUM INTO snapshot.

Jobs are idempotent — if a target date already has sessions or reports,
the service skips recomputation.
"""

from __future__ import annotations

from datetime import UTC

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from mindflow.services.analysis_service import AnalysisService
from mindflow.services.maintenance_service import MaintenanceService
from mindflow.services.report_service import ReportService


def build_scheduler(
    analysis_service: AnalysisService | None = None,
    report_service: ReportService | None = None,
    maintenance_service: MaintenanceService | None = None,
    event_retention_days: int = 30,
) -> AsyncIOScheduler:
    """Create and configure an ``AsyncIOScheduler`` with Wave 5 cron jobs.

    Args:
        analysis_service: Service for session identification
            (required for the 23:59 job).
        report_service: Service for daily report generation
            (required for the 00:01 job).
        maintenance_service: Service for event cleanup and backup
            (required for the 03:00 and 04:00 jobs).
        event_retention_days: Retention period for raw events in days.
            Passed to the cleanup job.

    Returns:
        A configured ``AsyncIOScheduler`` instance.  Caller is responsible
        for calling ``scheduler.start()`` after creation and
        ``scheduler.shutdown()`` during application shutdown.
    """
    scheduler = AsyncIOScheduler(timezone=UTC)

    # ── 23:59 — Session identification ────────────────────────────────
    if analysis_service is not None:
        scheduler.add_job(
            analysis_service.identify_all_today,
            trigger="cron",
            hour=23,
            minute=59,
            id="identify_sessions",
            replace_existing=True,
            name="Daily focus session identification",
        )
        logger.debug("Scheduler: registered identify_sessions at T23:59")
    else:
        logger.warning("Scheduler: analysis_service not provided, skipping identify_sessions")

    # ── 00:01 — Daily report ──────────────────────────────────────────
    if report_service is not None:
        scheduler.add_job(
            report_service.generate_daily_for_all,
            trigger="cron",
            hour=0,
            minute=1,
            id="daily_report",
            replace_existing=True,
            name="Daily report generation",
        )
        logger.debug("Scheduler: registered daily_report at T00:01")
    else:
        logger.warning("Scheduler: report_service not provided, skipping daily_report")

    # ── 03:00 — Event cleanup ─────────────────────────────────────────
    if maintenance_service is not None:
        scheduler.add_job(
            maintenance_service.cleanup_old_events,
            trigger="cron",
            hour=3,
            minute=0,
            id="event_cleanup",
            replace_existing=True,
            name="Raw event cleanup",
            kwargs={"retention_days": event_retention_days},
        )
        logger.debug("Scheduler: registered event_cleanup at T03:00")
    else:
        logger.warning("Scheduler: maintenance_service not provided, skipping event_cleanup")

    # ── 04:00 — Daily backup ──────────────────────────────────────────
    if maintenance_service is not None:
        scheduler.add_job(
            maintenance_service.run_daily_backup,
            trigger="cron",
            hour=4,
            minute=0,
            id="daily_backup",
            replace_existing=True,
            name="Daily database backup",
        )
        logger.debug("Scheduler: registered daily_backup at T04:00")
    else:
        logger.warning("Scheduler: maintenance_service not provided, skipping daily_backup")

    logger.info(
        "Scheduler built with cron jobs: identify_sessions, daily_report, "
        "event_cleanup, daily_backup"
    )
    return scheduler
