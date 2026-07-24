"""Pure-asyncio scheduler for maintenance tasks.

Replaces APScheduler to avoid Windows CTRL_BREAK_EVENT issues
(APScheduler's AsyncIOScheduler triggers console events on Windows
that uvicorn >=0.41 captures as shutdown signals).

Registered jobs (all times are UTC):
  - 23:30  — ``daily_panel``: run expert panel deliberation.
  - 23:59  — ``identify_sessions``: run daily session identification.
  - 00:01  — ``daily_report``: generate today's daily report.
  - 03:00  — ``event_cleanup``: delete raw events past retention policy.
  - 04:00  — ``daily_backup``: crash-consistent VACUUM INTO snapshot.
  - every 30 min — ``auto_intervention_check``: assess recent behavior
    and intervene if significant procrastination detected (08:00-23:00).

Jobs are idempotent — if a target date already has sessions or reports,
the service skips recomputation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from apscheduler.triggers.interval import IntervalTrigger as _APSchedulerIntervalTrigger
from loguru import logger

from mindflow.agents.types import (
    PanelBudgetExceededError,
    PanelUnavailableError,
)
from mindflow.domain.events import ActivityEvent
from mindflow.domain.procrastination import ProcrastinationAssessment, RuleEngine
from mindflow.infrastructure.llm.summary import build_behavior_summary
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.autonomy_service import AutonomyService
from mindflow.services.intervention_service import InterventionService
from mindflow.services.maintenance_service import MaintenanceService
from mindflow.services.panel_service import PanelService
from mindflow.services.report_service import ReportService
from mindflow.time_utils import utc_today

# Minimum confidence threshold for auto-intervention trigger.
# Assessments below this threshold are considered "no significant pattern"
# and are silently skipped (saves computation and avoids false positives).
_AUTO_INTERVENTION_MIN_CONFIDENCE: float = 0.5

# Confidence threshold for escalating to the expert panel for a more
# precise intervention (see G005 three-tier routing).
_AUTO_INTERVENTION_PANEL_CONFIDENCE: float = 0.75

# Track which dates the daily panel has already been triggered (either by the
# 23:30 cron or the 30-min auto-intervention check).  Guarded by
# ``_DAILY_PANEL_LOCK`` and claimed BEFORE awaiting the panel so the two jobs
# cannot both observe "not run yet" and double-fire the expensive panel
# (review C4 — shared-mutable race).  The set + module scope are kept so a run
# is deduped for the whole calendar day across both callers.
_DAILY_PANEL_RUN_DATES: set[str] = set()
_DAILY_PANEL_LOCK = asyncio.Lock()


class _CronTrigger:
    """APScheduler-compatible cron trigger for test compatibility."""
    def __init__(self, hour: int, minute: int):
        self.fields: list[str | None] = [None] * 7
        self.fields[5] = str(hour)
        self.fields[6] = str(minute)


class _IntervalTrigger(_APSchedulerIntervalTrigger):  # type: ignore[misc]
    """APScheduler-compatible interval trigger.  Extends the real type so
    isinstance checks pass in scheduler tests."""
    def __init__(self, minutes: int):
        super().__init__(minutes=minutes)


@dataclass
class _JobInfo:
    """APScheduler-compatible job info for test compatibility."""
    id: str
    trigger: _CronTrigger | _IntervalTrigger
    kwargs: dict[str, Any] = field(default_factory=dict)


class AsyncioScheduler:
    """Minimal pure-asyncio scheduler with cron + interval support.

    Replaces APScheduler's AsyncIOScheduler to avoid Windows CTRL_BREAK_EVENT
    issues.  Jobs are registered during construction, then launched as asyncio
    tasks when ``start()`` is called.
    """

    def __init__(self) -> None:
        self._jobs: list[dict[str, Any]] = []
        self._job_infos: list[_JobInfo] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self.timezone = UTC

    @property
    def timezone(self) -> Any:
        return self._timezone

    @timezone.setter
    def timezone(self, value: Any) -> None:
        self._timezone = value

    def get_jobs(self) -> list[_JobInfo]:
        """APScheduler-compatible: return all registered jobs."""
        return list(self._job_infos)

    def get_job(self, job_id: str) -> _JobInfo | None:
        """APScheduler-compatible: return job by id."""
        for j in self._job_infos:
            if j.id == job_id:
                return j
        return None

    def daily_cron(
        self,
        hour: int,
        minute: int,
        coro: Callable[..., Awaitable[Any]],
        kwargs: dict[str, Any] | None = None,
        name: str = "",
    ) -> AsyncioScheduler:
        """Schedule *coro* to run daily at UTC *hour*:*minute*."""
        self._jobs.append({
            "type": "cron",
            "hour": hour,
            "minute": minute,
            "coro": coro,
            "kwargs": kwargs,
            "name": name,
        })
        self._job_infos.append(_JobInfo(
            id=name,
            trigger=_CronTrigger(hour=hour, minute=minute),
            kwargs=kwargs or {},
        ))
        logger.debug("Scheduler: registered {} at T{:02d}:{:02d}", name, hour, minute)
        return self

    def interval_minutes(
        self,
        minutes: int,
        coro: Callable[..., Awaitable[Any]],
        kwargs: dict[str, Any] | None = None,
        name: str = "",
    ) -> AsyncioScheduler:
        """Schedule *coro* to run every *minutes* minutes."""
        self._jobs.append({
            "type": "interval",
            "minutes": minutes,
            "coro": coro,
            "kwargs": kwargs,
            "name": name,
        })
        self._job_infos.append(_JobInfo(
            id=name,
            trigger=_IntervalTrigger(minutes=minutes),
            kwargs=kwargs or {},
        ))
        logger.debug("Scheduler: registered {} (interval={}min)", name, minutes)
        return self

    def start(self) -> None:
        """Launch all registered jobs as background asyncio tasks."""
        self._running = True
        for job in self._jobs:
            if job["type"] == "cron":
                t = asyncio.create_task(
                    self._run_daily_cron(
                        job["hour"], job["minute"],
                        job["coro"], job["kwargs"], job["name"],
                    )
                )
            else:
                t = asyncio.create_task(
                    self._run_interval(
                        job["minutes"],
                        job["coro"], job["kwargs"], job["name"],
                    )
                )
            self._tasks.append(t)
            logger.info("AsyncioScheduler: created task for '{}'", job["name"])
        logger.info("AsyncioScheduler started with {} job(s)", len(self._tasks))

    def shutdown(self, wait: bool = False) -> None:
        """Cancel all background tasks."""
        self._running = False
        for t in self._tasks:
            t.cancel()
        logger.debug("AsyncioScheduler shut down")

    async def _run_daily_cron(
        self,
        hour: int,
        minute: int,
        coro: Callable[..., Awaitable[Any]],
        kwargs: dict[str, Any] | None,
        name: str,
    ) -> None:
        while self._running:
            now = datetime.now(UTC)
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            wait_s = (target - now).total_seconds()
            try:
                await asyncio.sleep(wait_s)
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            try:
                if kwargs:
                    await coro(**kwargs)
                else:
                    await coro()
            except Exception:
                logger.exception("Scheduler job {} failed", name)

    async def _run_interval(
        self,
        minutes: int,
        coro: Callable[..., Awaitable[Any]],
        kwargs: dict[str, Any] | None,
        name: str,
    ) -> None:
        while self._running:
            try:
                await asyncio.sleep(max(minutes * 60, 60))
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            try:
                if kwargs:
                    await coro(**kwargs)
                else:
                    await coro()
            except Exception:
                logger.exception("Scheduler job {} failed", name)



async def _claim_daily_panel_run(date_str: str) -> bool:
    """Atomically claim the daily-panel run for *date_str*.

    Returns True if the caller won the claim (must run the panel), False if it
    was already claimed today.  The claim is recorded BEFORE the panel runs so
    a concurrent tick sees it immediately; callers must release via
    ``_release_daily_panel_run`` if the run then fails, so a later attempt can
    retry.
    """
    async with _DAILY_PANEL_LOCK:
        if date_str in _DAILY_PANEL_RUN_DATES:
            return False
        _DAILY_PANEL_RUN_DATES.add(date_str)
        return True


async def _release_daily_panel_run(date_str: str) -> None:
    """Release a previously-claimed daily-panel run (on failure) so it can retry."""
    async with _DAILY_PANEL_LOCK:
        _DAILY_PANEL_RUN_DATES.discard(date_str)


async def _auto_intervention_check(
    activity_repo: SQLAlchemyActivityRepository,
    intervention_service: InterventionService,
    rule_engine: RuleEngine | None = None,
    panel_service: PanelService | None = None,
    autonomy_service: AutonomyService | None = None,
    telemetry_service: Any | None = None,
    user_id: int = 1,
    window_min: int = 30,
    min_confidence: float = _AUTO_INTERVENTION_MIN_CONFIDENCE,
    panel_confidence: float = _AUTO_INTERVENTION_PANEL_CONFIDENCE,
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
        panel_service: Panel service for L1 expert consultation
            (optional — may be None if PanelService is unavailable).
        autonomy_service: Autonomy switch service
            (optional — if None, skips the autonomy check).
        user_id: User identifier (default 1 for single-user mode).
        window_min: Look-back window in minutes (default 30).
    """
    engine = rule_engine or RuleEngine()
    now = datetime.now(UTC)

    # ── Autonomy guard: user-level kill switch (§7 safety boundary) ──
    if autonomy_service is not None:
        try:
            if not await autonomy_service.is_enabled(user_id):
                logger.debug("Auto-intervention: autonomy disabled, skipping")
                return
        except Exception as exc:
            logger.error("Auto-intervention: autonomy check failed: {}", exc)
            return

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
    if top_confidence < min_confidence:
        logger.debug(
            "Auto-intervention: confidence {:.2f} < {:.2f}, skipping",
            top_confidence,
            min_confidence,
        )
        return

    # ── Three-tier routing (G005) ──────────────────────────────────
    #   Tier 1 (mid confidence): direct rule-engine intervention
    #   Tier 2 (high confidence): try expert panel for precise attribution
    #
    # Tier 2 is attempted when confidence >= 0.75 AND the panel has not
    # yet been triggered today.  If the panel fails, we fall back to Tier 1
    # with the original rule-engine assessment.

    assessment_for_dispatch = assessment
    panel_attempted = False

    if top_confidence >= panel_confidence and panel_service is not None:
        today_str = now.strftime("%Y-%m-%d")
        # Claim the run BEFORE awaiting so a concurrent daily-panel cron tick
        # cannot also fire the panel (review C4 race). If we lose the claim,
        # another caller already ran (or is running) today's panel — skip.
        claimed = await _claim_daily_panel_run(today_str)
        if claimed:
            logger.info(
                "Auto-intervention: confidence {:.2f} >= {:.2f}, "
                "escalating to expert panel",
                top_confidence,
                panel_confidence,
            )
            panel_attempted = True
            try:
                verdict = await panel_service.run_daily_panel(
                    user_id=user_id,
                    target_date=utc_today(),
                )

                # Convert PanelVerdict → ProcrastinationAssessment for
                # downstream intervention dispatch (more precise attribution).
                panel_assessment = ProcrastinationAssessment(
                    types=verdict.types,
                    confidence=dict(verdict.confidence),
                    recommended_technique=verdict.recommended_technique,
                    rationale=verdict.rationale,
                    source="rule_engine",  # mimic rule_engine for dispatch
                )
                assessment_for_dispatch = panel_assessment

                logger.info(
                    "Panel verdict applied: types={}, technique={}",
                    [str(t) for t in verdict.types],
                    verdict.recommended_technique,
                )
            except (PanelUnavailableError, PanelBudgetExceededError) as exc:
                logger.warning(
                    "Panel escalation failed ({}), falling back to rule engine",
                    exc,
                )
                # Release the claim so a later tick can retry today's panel.
                await _release_daily_panel_run(today_str)
                # Fall through: keep assessment_for_dispatch = assessment
            except Exception as exc:
                logger.error("Panel escalation unexpected error: {}", exc)
                await _release_daily_panel_run(today_str)

    if panel_attempted and assessment_for_dispatch is assessment:
        logger.info(
            "Auto-intervention: panel failed, falling back to rule-based dispatch"
        )

    # ── Dispatch intervention ───────────────────────────────────────
    try:
        result = await intervention_service.maybe_intervene(
            assessment=assessment_for_dispatch,
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
    panel_service: Any | None = None,
    autonomy_service: AutonomyService | None = None,
    telemetry_service: Any | None = None,
    event_retention_days: int = 30,
    min_confidence: float = _AUTO_INTERVENTION_MIN_CONFIDENCE,
    panel_confidence: float = _AUTO_INTERVENTION_PANEL_CONFIDENCE,
) -> AsyncioScheduler:
    """Create a pure-asyncio scheduler with cron + interval jobs.

    Uses ``AsyncioScheduler`` (pure asyncio, no Windows issues) instead of
    APScheduler's ``AsyncIOScheduler`` which triggers CTRL_BREAK_EVENT on
    Windows.
    """
    scheduler = AsyncioScheduler()

    # ── 23:30 — Expert panel deliberation ────────────────────────────
    if panel_service is not None:
        async def _run_daily_panel() -> None:
            try:
                if (
                    autonomy_service is not None
                    and not await autonomy_service.is_enabled(user_id=1)
                ):
                    logger.debug("Daily panel: autonomy disabled, skipping")
                    return
                today_str = utc_today().strftime("%Y-%m-%d")
                if not await _claim_daily_panel_run(today_str):
                    logger.debug("Daily panel: already run today ({}), skipping", today_str)
                    return
                try:
                    await panel_service.run_daily_panel(user_id=1, target_date=utc_today())
                except Exception:
                    await _release_daily_panel_run(today_str)
                    raise
            except Exception:
                logger.exception("Daily panel job failed")

        scheduler.daily_cron(hour=23, minute=30, coro=_run_daily_panel,
                             name="daily_panel")

    # ── 23:59 — Session identification ────────────────────────────────
    if analysis_service is not None:
        scheduler.daily_cron(hour=23, minute=59,
                             coro=analysis_service.identify_all_today,
                             name="identify_sessions")

    # ── 00:01 — Daily report ──────────────────────────────────────────
    if report_service is not None:
        scheduler.daily_cron(hour=0, minute=1,
                             coro=report_service.generate_daily_for_all,
                             name="daily_report")

    # ── 03:00 — Event cleanup ─────────────────────────────────────────
    if maintenance_service is not None:
        scheduler.daily_cron(hour=3, minute=0,
                             coro=maintenance_service.cleanup_old_events,
                             kwargs={"retention_days": event_retention_days},
                             name="event_cleanup")

    # ── 04:00 — Daily backup ──────────────────────────────────────────
    if maintenance_service is not None:
        scheduler.daily_cron(hour=4, minute=0,
                             coro=maintenance_service.run_daily_backup,
                             name="daily_backup")

    # ── Every 30 min — Auto intervention check ──────────────────────────
    if intervention_service is not None and activity_repository is not None:
        scheduler.interval_minutes(minutes=30,
                                   coro=_auto_intervention_check,
                                   kwargs={
                                       "activity_repo": activity_repository,
                                       "intervention_service": intervention_service,
                                       "panel_service": panel_service,
                                       "autonomy_service": autonomy_service,
                                       "min_confidence": min_confidence,
                                       "panel_confidence": panel_confidence,
                                   },
                                   name="auto_intervention_check")

    if telemetry_service is not None:
        async def _rollup_telemetry() -> None:
            end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            start = end - timedelta(days=1)
            await telemetry_service.rollup_feature_windows(start, end)
            await telemetry_service.cleanup_retained_data()

        scheduler.daily_cron(
            hour=2,
            minute=45,
            coro=_rollup_telemetry,
            name="telemetry_rollup",
        )


    logger.info(
        "AsyncioScheduler built with jobs: {}",
        [j["name"] for j in scheduler._jobs],

    )
    return scheduler
