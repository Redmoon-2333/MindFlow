"""APScheduler-based cron job configuration for maintenance tasks.

Per ADR-007, APScheduler is used exclusively for cron-style scheduling
(fixed time-of-day jobs) — not for the high-frequency collector tick loop,
which runs as a bare ``asyncio.create_task`` inside CollectorService.

Registered cron jobs (all times are UTC):
  - 23:59  — ``identify_sessions``: run daily session identification.
  - 00:01  — ``daily_report``: generate today's daily report.
  - 03:00  — ``event_cleanup``: delete raw events past retention policy.
  - 04:00  — ``daily_backup``: crash-consistent VACUUM INTO snapshot.

Registered interval job:
  - every 30 min — ``auto_intervention_check``: assess recent behavior
    and intervene if significant procrastination detected (08:00-23:00).
    (Wave 8b, Wave 7 residual)

Jobs are idempotent — if a target date already has sessions or reports,
the service skips recomputation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from loguru import logger

from mindflow.domain.events import ActivityEvent
from mindflow.domain.procrastination import RuleEngine
from mindflow.infrastructure.llm.summary import build_behavior_summary
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.intervention_service import InterventionService
from mindflow.services.maintenance_service import MaintenanceService
from mindflow.services.report_service import ReportService

# Minimum confidence threshold for auto-intervention trigger.
# Assessments below this threshold are considered "no significant pattern"
# and are silently skipped (saves computation and avoids false positives).
_AUTO_INTERVENTION_MIN_CONFIDENCE: float = 0.5


async def _auto_intervention_check(
    activity_repo: SQLAlchemyActivityRepository,
    intervention_service: InterventionService,
    rule_engine: RuleEngine | None = None,
    user_id: int = 1,
    window_min: int = 30,
) -> None:
    """Assess recent behavior and intervene if significant procrastination detected.

    Guard conditions (silent skip):
      1. Outside 08:00-23:00 local-time-equivalent window.
      2. No events in the look-back window.
      3. All events are idle (user away from computer).
      4. RuleEngine assessment confidence < 0.5 (no significant pattern).

    When triggered, calls ``intervention_service.maybe_intervene()`` which
    applies its own throttle guard — this job does not bypass throttling.

    This function never raises (all errors are logged and swallowed).

    Args:
        activity_repo: Repository for querying recent activity events.
        intervention_service: Service to dispatch interventions.
        rule_engine: RuleEngine instance (created fresh if None).
        user_id: User identifier (default 1 for single-user mode).
        window_min: Look-back window in minutes (default 30).
    """
    engine = rule_engine or RuleEngine()
    now = datetime.now(UTC)

    # ── Time-of-day guard: only 08:00-23:00 ─────────────────────────
    hour = now.hour
    if hour < 8 or hour >= 23:
        logger.debug("Auto-intervention: outside working hours ({:02d}:00), skipping", hour)
        return

    # ── Fetch recent events ─────────────────────────────────────────
    try:
        start = now - timedelta(minutes=window_min)
        events: list[ActivityEvent] = await activity_repo.query_range(
            user_id, start, now
        )
    except Exception as exc:
        logger.error("Auto-intervention: failed to query events: {}", exc)
        return

    # ── Guard: no events or all idle ────────────────────────────────
    if not events:
        logger.debug("Auto-intervention: no events in last {}min, skipping", window_min)
        return

    if all(ev.data.is_idle for ev in events):
        logger.debug("Auto-intervention: all events idle, skipping")
        return

    # ── Build behavior summary ──────────────────────────────────────
    try:
        summary = build_behavior_summary(events)
    except ValueError:
        logger.debug("Auto-intervention: cannot build summary from empty events")
        return
    except Exception as exc:
        logger.error("Auto-intervention: failed to build summary: {}", exc)
        return

    # ── Assess with RuleEngine (L3, no LLM cost) ────────────────────
    try:
        assessment = engine.assess(summary)
    except Exception as exc:
        logger.error("Auto-intervention: rule engine assessment failed: {}", exc)
        return

    # ── Confidence guard ────────────────────────────────────────────
    if not assessment.types:
        logger.debug("Auto-intervention: no types detected, skipping")
        return

    top_type = assessment.types[0]
    top_confidence = assessment.confidence.get(top_type, 0.0)
    if top_confidence < _AUTO_INTERVENTION_MIN_CONFIDENCE:
        logger.debug(
            "Auto-intervention: confidence {:.2f} < {:.2f}, skipping",
            top_confidence,
            _AUTO_INTERVENTION_MIN_CONFIDENCE,
        )
        return

    # ── Dispatch intervention ───────────────────────────────────────
    try:
        result = await intervention_service.maybe_intervene(
            assessment=assessment,
            recent_events=events,
            user_id=user_id,
        )
        if result.skipped:
            logger.info(
                "Auto-intervention: skipped ({}) — {}",
                result.skip_reason,
                result.throttle_decision or "",
            )
        else:
            logger.info(
                "Auto-intervention: dispatched {} to user {}",
                result.intervention.id if result.intervention else "?",
                user_id,
            )
    except Exception as exc:
        logger.error("Auto-intervention: dispatch failed: {}", exc)


def build_scheduler(
    analysis_service: AnalysisService | None = None,
    report_service: ReportService | None = None,
    maintenance_service: MaintenanceService | None = None,
    intervention_service: InterventionService | None = None,
    activity_repository: SQLAlchemyActivityRepository | None = None,
    event_retention_days: int = 30,
) -> AsyncIOScheduler:
    """Create and configure an ``AsyncIOScheduler`` with cron + interval jobs.

    Args:
        analysis_service: Service for session identification
            (required for the 23:59 job).
        report_service: Service for daily report generation
            (required for the 00:01 job).
        maintenance_service: Service for event cleanup and backup
            (required for the 03:00 and 04:00 jobs).
        intervention_service: Service for auto-intervention dispatch
            (required for the interval job).
        activity_repository: Repository for querying recent activity events
            (required for the auto-intervention job).
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

    # ── Every 30 min — Auto intervention check (Wave 8b) ──────────────
    if intervention_service is not None and activity_repository is not None:
        scheduler.add_job(
            _auto_intervention_check,
            trigger="interval",
            minutes=30,
            id="auto_intervention_check",
            replace_existing=True,
            name="Auto intervention check (every 30 min)",
            kwargs={
                "activity_repo": activity_repository,
                "intervention_service": intervention_service,
            },
        )
        logger.debug("Scheduler: registered auto_intervention_check (interval=30min)")
    else:
        logger.warning(
            "Scheduler: intervention_service or activity_repository not provided, "
            "skipping auto_intervention_check"
        )

    logger.info(
        "Scheduler built with jobs: identify_sessions, daily_report, "
        "event_cleanup, daily_backup, auto_intervention_check"
    )
    return scheduler
