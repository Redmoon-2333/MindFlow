"""Pure-asyncio scheduler for maintenance tasks.

Replaces APScheduler to avoid Windows CTRL_BREAK_EVENT issues
(APScheduler's AsyncIOScheduler triggers console events on Windows
that uvicorn >=0.41 captures as shutdown signals).

Registered jobs (cron and working hours use the configured local timezone):
  - 23:30  — ``daily_panel``: run expert panel deliberation.
  - 23:59  — ``identify_sessions``: run daily session identification.
  - 00:05  — ``daily_report``: generate the previous business day's report.
  - 03:00  — ``event_cleanup``: delete raw events past retention policy.
  - 04:00  — ``daily_backup``: crash-consistent VACUUM INTO snapshot.
  - every 30 min — ``auto_intervention_check``: assess recent behavior
    and intervene if significant procrastination detected (08:00-23:00).
  - every 15 min — ``telemetry_rollup_recent``: roll up the trailing
    two-hour window of feature windows (idempotent — overlaps are safe).

Jobs are idempotent — if a target date already has sessions or reports,
the service skips recomputation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo
from typing import Any, TypeVar

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
from mindflow.ports import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisWorkflowPort,
    ScheduledJobRunsPort,
)
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.autonomy_service import AutonomyService
from mindflow.services.intervention_service import InterventionService
from mindflow.services.maintenance_service import MaintenanceService
from mindflow.services.panel_service import PanelService
from mindflow.services.report_service import ReportService
from mindflow.time_utils import (
    TimezoneLike,
    business_day_bounds_utc,
    business_today,
    resolve_timezone,
)

# Minimum confidence threshold for auto-intervention trigger.
# Assessments below this threshold are considered "no significant pattern"
# and are silently skipped (saves computation and avoids false positives).
_AUTO_INTERVENTION_MIN_CONFIDENCE: float = 0.5

# Confidence threshold for escalating to the expert panel for a more
# precise intervention (see G005 three-tier routing).
_AUTO_INTERVENTION_PANEL_CONFIDENCE: float = 0.75

# Auto-intervention local-time window bounds. ``_AUTO_INTERVENTION_END_HOUR``
# is exclusive — an intervention is only dispatched when
# ``start_hour <= local_hour < end_hour``. Both are configurable via settings.
_AUTO_INTERVENTION_START_HOUR: int = 8
_AUTO_INTERVENTION_END_HOUR: int = 23

# Auto-intervention check cadence (minutes). Throttle still limits dispatch.
_AUTO_INTERVENTION_INTERVAL_MINUTES: int = 5

# Minimum non-idle activity time (in minutes) before intervention
# can be considered. Avoids triggering on brief computer use.
_MIN_NON_IDLE_MINUTES: float = 10.0

_SCHEDULED_JOB_HEARTBEAT_INTERVAL_SECONDS = 10 * 60.0
_IDENTIFY_COMPLETION_POLL_SECONDS = 1.0
_STARTUP_RECOVERY_RETRIES = 1
# Startup recovery intentionally covers only the latest completed business day
# so a long offline period cannot trigger an unexpected burst of LLM spend.
_STARTUP_RECOVERY_COMPLETE_DAYS = 1

# Todo 11: bounded recent-window rollup. The interval bounds are one fixed
# ``now`` captured per invocation so [now-2h, now] is a single window, and the
# 15-minute cadence re-rolls a shifting two-hour range whose overlaps are safe
# because ``rollup_feature_windows`` upserts windows idempotently and folds the
# baseline only from newly inserted rows (Todo 8 seam).
_RECENT_ROLLUP_INTERVAL_MINUTES = 15
_RECENT_ROLLUP_WINDOW_HOURS = 2

# Compatibility fallback for direct unit use without the persistent repository.
_DAILY_PANEL_RUN_DATES: set[str] = set()
_DAILY_PANEL_LOCK = asyncio.Lock()


async def _claim_daily_panel_run(date_str: str) -> bool:
    async with _DAILY_PANEL_LOCK:
        if date_str in _DAILY_PANEL_RUN_DATES:
            return False
        _DAILY_PANEL_RUN_DATES.add(date_str)
        return True


async def _release_daily_panel_run(date_str: str) -> None:
    async with _DAILY_PANEL_LOCK:
        _DAILY_PANEL_RUN_DATES.discard(date_str)


def _next_daily_run_utc(
    now_utc: datetime,
    *,
    hour: int,
    minute: int,
    timezone: TimezoneLike = "local",
) -> datetime:
    """Return the next local wall-clock cron occurrence as an aware UTC datetime."""
    if now_utc.tzinfo is None:
        msg = "now_utc must be timezone-aware"
        raise ValueError(msg)

    local_timezone = resolve_timezone(timezone)
    now_local = now_utc.astimezone(local_timezone)
    target_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target_local <= now_local:
        next_date = target_local.date() + timedelta(days=1)
        target_local = target_local.replace(
            year=next_date.year, month=next_date.month, day=next_date.day
        )
    return target_local.astimezone(UTC)


class _CronTrigger:
    """APScheduler-compatible cron trigger for test compatibility."""

    def __init__(self, hour: int, minute: int):
        self.fields: list[str | None] = [None] * 7
        self.fields[5] = str(hour)
        self.fields[6] = str(minute)


class _IntervalTrigger:
    """Small interval descriptor exposed by ``get_jobs()``."""

    def __init__(self, minutes: int) -> None:
        self.interval = timedelta(minutes=minutes)


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

    def __init__(self, timezone: TimezoneLike = "local") -> None:
        self._jobs: list[dict[str, Any]] = []
        self._job_infos: list[_JobInfo] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._startup_recovery: Callable[[datetime], Awaitable[None]] | None = None
        self.timezone = resolve_timezone(timezone)
        self.last_heartbeat_at: datetime | None = None

    def _touch_heartbeat(self) -> None:
        """Record that a scheduler job ran; exposed via /health."""
        self.last_heartbeat_at = datetime.now(UTC)

    @property
    def timezone(self) -> tzinfo:
        return self._timezone

    @timezone.setter
    def timezone(self, value: tzinfo) -> None:
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
        *,
        catch_up: bool = False,
    ) -> AsyncioScheduler:
        """Schedule *coro* daily at local *hour*:*minute*."""
        self._jobs.append(
            {
                "type": "cron",
                "hour": hour,
                "minute": minute,
                "coro": coro,
                "kwargs": kwargs,
                "name": name,
                "catch_up": catch_up,
            }
        )
        self._job_infos.append(
            _JobInfo(
                id=name,
                trigger=_CronTrigger(hour=hour, minute=minute),
                kwargs=kwargs or {},
            )
        )
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
        self._jobs.append(
            {
                "type": "interval",
                "minutes": minutes,
                "coro": coro,
                "kwargs": kwargs,
                "name": name,
            }
        )
        self._job_infos.append(
            _JobInfo(
                id=name,
                trigger=_IntervalTrigger(minutes=minutes),
                kwargs=kwargs or {},
            )
        )
        logger.debug("Scheduler: registered {} (interval={}min)", name, minutes)
        return self

    def start(self) -> None:
        """Launch all registered jobs as background asyncio tasks."""
        self._running = True
        for job in self._jobs:
            if job["type"] == "cron":
                t = asyncio.create_task(
                    self._run_daily_cron(
                        job["hour"],
                        job["minute"],
                        job["coro"],
                        job["kwargs"],
                        job["name"],
                        catch_up=job["catch_up"],
                    )
                )
            else:
                t = asyncio.create_task(
                    self._run_interval(
                        job["minutes"],
                        job["coro"],
                        job["kwargs"],
                        job["name"],
                    )
                )
            self._tasks.append(t)
            logger.info("AsyncioScheduler: created task for '{}'", job["name"])
        if self._startup_recovery is not None:
            recovery_task = asyncio.create_task(
                self.run_startup_recovery(),
                name="startup_recovery",
            )
            self._tasks.append(recovery_task)
            logger.info("AsyncioScheduler: created startup recovery task")
        logger.info("AsyncioScheduler started with {} job(s)", len(self._tasks))

    async def run_startup_recovery(self, *, now_utc: datetime | None = None) -> None:
        if self._startup_recovery is None:
            return
        now = now_utc or datetime.now(UTC)
        if now.tzinfo is None:
            msg = "now_utc must be timezone-aware"
            raise ValueError(msg)
        await self._startup_recovery(now)

    async def shutdown(self) -> None:
        """Cancel and await all background tasks before returning."""
        self._running = False
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        logger.debug("AsyncioScheduler shut down")

    async def _run_daily_cron(
        self,
        hour: int,
        minute: int,
        coro: Callable[..., Awaitable[Any]],
        kwargs: dict[str, Any] | None,
        name: str,
        *,
        catch_up: bool = False,
    ) -> None:
        if catch_up:
            now_local = datetime.now(UTC).astimezone(self.timezone)
            if (now_local.hour, now_local.minute) >= (hour, minute):
                try:
                    if kwargs:
                        await coro(**kwargs)
                    else:
                        await coro()
                except Exception:
                    logger.exception("Scheduler catch-up job {} failed", name)
                finally:
                    self._touch_heartbeat()
        while self._running:
            now = datetime.now(UTC)
            target = _next_daily_run_utc(now, hour=hour, minute=minute, timezone=self.timezone)
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
            finally:
                self._touch_heartbeat()

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
            finally:
                self._touch_heartbeat()


_JobResult = TypeVar("_JobResult")


async def _heartbeat_claim(
    repository: ScheduledJobRunsPort,
    job_name: str,
    local_date: date,
    attempt_count: int,
) -> None:
    while True:
        await asyncio.sleep(_SCHEDULED_JOB_HEARTBEAT_INTERVAL_SECONDS)
        if not await repository.heartbeat(
            job_name,
            local_date,
            attempt_count=attempt_count,
        ):
            return


async def _await_terminal_claim_update(update: Awaitable[bool]) -> bool:
    """Finish a claim-state write even if shutdown repeats cancellation."""
    update_task = asyncio.ensure_future(update)
    while True:
        try:
            return await asyncio.shield(update_task)
        except asyncio.CancelledError:
            if update_task.done():
                return update_task.result()


async def _mark_claim_cancelled(
    repository: ScheduledJobRunsPort,
    job_name: str,
    local_date: date,
    attempt_count: int,
) -> None:
    try:
        await _await_terminal_claim_update(
            repository.mark_cancelled(
                job_name,
                local_date,
                attempt_count=attempt_count,
            )
        )
    except Exception:
        logger.exception(
            "Failed to mark cancelled claim {} for {} attempt {}",
            job_name,
            local_date,
            attempt_count,
        )


async def _run_claimed_job(
    repository: ScheduledJobRunsPort | None,
    job_name: str,
    local_date: date,
    job: Callable[[], Awaitable[_JobResult]],
    *,
    retry_failed: bool = False,
) -> tuple[bool, _JobResult | None]:
    fallback_panel_claim = repository is None and job_name == "daily_panel"
    attempt_count: int | None = None
    if repository is not None:
        attempt_count = await repository.claim(
            job_name, local_date, retry_failed=retry_failed
        )
        if attempt_count is None:
            return False, None
    elif fallback_panel_claim and not await _claim_daily_panel_run(local_date.isoformat()):
        return False, None
    job_task: asyncio.Future[_JobResult] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    try:
        if repository is None or attempt_count is None:
            result = await job()
        else:
            job_task = asyncio.ensure_future(job())
            heartbeat_task = asyncio.create_task(
                _heartbeat_claim(
                    repository,
                    job_name,
                    local_date,
                    attempt_count,
                )
            )
            done, _ = await asyncio.wait(
                (job_task, heartbeat_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                await heartbeat_task
                job_task.cancel()
                await asyncio.gather(job_task, return_exceptions=True)
                return False, None
            result = await job_task
        if repository is not None and attempt_count is not None:
            still_owned = await repository.mark_succeeded(
                job_name, local_date, attempt_count=attempt_count
            )
            if not still_owned:
                return False, None
        return True, result
    except asyncio.CancelledError:
        for task in (job_task, heartbeat_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (job_task, heartbeat_task) if task is not None),
            return_exceptions=True,
        )
        if repository is not None and attempt_count is not None:
            await _mark_claim_cancelled(
                repository,
                job_name,
                local_date,
                attempt_count,
            )
        elif fallback_panel_claim:
            await _release_daily_panel_run(local_date.isoformat())
        raise
    except Exception as exc:
        for task in (job_task, heartbeat_task):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (job_task, heartbeat_task) if task is not None),
            return_exceptions=True,
        )
        if repository is not None and attempt_count is not None:
            try:
                await repository.mark_failed(
                    job_name,
                    local_date,
                    attempt_count=attempt_count,
                    error=str(exc),
                )
            except asyncio.CancelledError:
                await _mark_claim_cancelled(
                    repository,
                    job_name,
                    local_date,
                    attempt_count,
                )
                raise
        elif fallback_panel_claim:
            await _release_daily_panel_run(local_date.isoformat())
        raise
    finally:
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)


async def _auto_intervention_check(
    activity_repo: SQLAlchemyActivityRepository,
    intervention_service: InterventionService,
    rule_engine: RuleEngine | None = None,
    panel_service: PanelService | None = None,
    workflow_port: AnalysisWorkflowPort | None = None,
    autonomy_service: AutonomyService | None = None,
    telemetry_service: Any | None = None,
    scheduled_job_runs_repository: ScheduledJobRunsPort | None = None,
    user_id: int = 1,
    window_min: int = 45,
    min_confidence: float = _AUTO_INTERVENTION_MIN_CONFIDENCE,
    panel_confidence: float = _AUTO_INTERVENTION_PANEL_CONFIDENCE,
    timezone: TimezoneLike = "local",
    start_hour: int = _AUTO_INTERVENTION_START_HOUR,
    end_hour: int = _AUTO_INTERVENTION_END_HOUR,
) -> None:
    """Assess recent behavior and intervene if significant procrastination detected.

    Guard conditions (silent skip):
      1. Outside the configurable local-time intervention window
         (default 08:00-23:00, exclusive end — see ``start_hour``/``end_hour``).
      2. No events in the look-back window.
      3. All events are idle (user away from computer).
      4. Non-idle activity < 10 min (insufficient data for pattern).
      5. RuleEngine assessment confidence < threshold (no significant pattern).

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
        start_hour: Intervention window start hour, local 24h (default 8).
        end_hour: Intervention window end hour, exclusive, local 24h (default 23).
    """
    engine = rule_engine or RuleEngine()
    now = datetime.now(UTC)
    now_local = now.astimezone(resolve_timezone(timezone))

    # ML is a secondary signal: it can veto a rule-based reminder, but it
    # never blocks the rule path when data/model are unavailable.
    ml_status = "unavailable"
    ml_probability: float | None = None
    if telemetry_service is not None:
        try:
            prediction = await telemetry_service.predict_latest_focus(user_id=user_id)
            if isinstance(prediction, dict):
                ml_status = prediction.get("status", "unknown") or "unknown"
                ml_probability = prediction.get("focus_probability")
        except Exception as exc:
            logger.debug("Auto-intervention: ML prediction failed: {}", exc)
            ml_status = "error"

    async def _record(
        reason: str,
        *,
        confidence: float | None = None,
        intervention_type: str | None = None,
        throttle_reason: str | None = None,
        source: str = "rule_engine",
    ) -> None:
        if telemetry_service is None:
            return
        try:
            await telemetry_service.save_intervention_check(
                user_id=user_id,
                checked_at=datetime.now(UTC).isoformat(),
                reason=reason,
                confidence=confidence,
                intervention_type=intervention_type,
                throttle_reason=throttle_reason,
                source=source,
                ml_status=ml_status,
            )
        except Exception as exc:
            logger.debug("Auto-intervention: failed to record check: {}", exc)

    # ── Autonomy guard: user-level kill switch (§7 safety boundary) ──
    if autonomy_service is not None:
        try:
            if not await autonomy_service.is_enabled(user_id):
                await _record("autonomy_disabled")
                logger.debug("Auto-intervention: autonomy disabled, skipping")
                return
        except Exception as exc:
            logger.error("Auto-intervention: autonomy check failed: {}", exc)
            await _record("autonomy_error")
            return

    # ── Time-of-day guard: configurable local window (default 08:00-23:00) ─
    hour = now_local.hour
    if hour < start_hour or hour >= end_hour:
        await _record("outside_hours")
        logger.debug(
            "Auto-intervention: outside working hours "
            "({:02d}:00 not in [{:02d}:00, {:02d}:00)), skipping",
            hour,
            start_hour,
            end_hour,
        )
        return

    # ── Fetch recent events ─────────────────────────────────────────
    try:
        start = now - timedelta(minutes=window_min)
        events: list[ActivityEvent] = await activity_repo.query_range(user_id, start, now)
    except Exception as exc:
        logger.error("Auto-intervention: failed to query events: {}", exc)
        await _record("event_query_error")
        return

    # ── Guard: no events or all idle ────────────────────────────────
    if not events:
        await _record("no_events")
        logger.debug("Auto-intervention: no events in last {}min, skipping", window_min)
        return

    if all(ev.data.is_idle for ev in events):
        await _record("all_idle")
        logger.debug("Auto-intervention: all events idle, skipping")
        return

    # ── Guard: insufficient non-idle activity ──────────────────────────
    # Avoids triggering on brief computer use — the user must have been
    # actively working for at least ``_MIN_NON_IDLE_MINUTES`` in the window.
    non_idle_s = sum(max(0.0, ev.duration_s) for ev in events if not ev.data.is_idle)
    if non_idle_s < _MIN_NON_IDLE_MINUTES * 60:
        await _record("insufficient_non_idle")
        logger.debug(
            "Auto-intervention: non-idle activity {:.0f}s < {:.0f}min, skipping",
            non_idle_s,
            _MIN_NON_IDLE_MINUTES,
        )
        return

    # ── Build behavior summary ──────────────────────────────────────
    try:
        summary = build_behavior_summary(events)
    except ValueError:
        await _record("summary_error")
        logger.debug("Auto-intervention: cannot build summary from empty events")
        return
    except Exception as exc:
        logger.error("Auto-intervention: failed to build summary: {}", exc)
        await _record("summary_error")
        return

    # ── Assess with RuleEngine (L3, no LLM cost) ────────────────────
    try:
        assessment = engine.assess(summary)
    except Exception as exc:
        logger.error("Auto-intervention: rule engine assessment failed: {}", exc)
        await _record("assessment_error")
        return

    # ── Confidence guard ────────────────────────────────────────────
    if not assessment.types:
        await _record("no_types")
        logger.debug("Auto-intervention: no types detected, skipping")
        return

    top_type = assessment.types[0]
    top_confidence = assessment.confidence.get(top_type, 0.0)
    if top_confidence < min_confidence:
        await _record("low_confidence", confidence=top_confidence)
        logger.debug(
            "Auto-intervention: confidence {:.2f} < {:.2f}, skipping",
            top_confidence,
            min_confidence,
        )
        return

    # ML veto: if a ready model says the user is focused, do not interrupt.
    if ml_status == "ready" and ml_probability is not None and ml_probability >= 0.5:
        await _record("ml_disagrees", confidence=top_confidence)
        logger.debug(
            "Auto-intervention: ML probability {:.2f} >= 0.5, skipping", ml_probability
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

    if top_confidence >= panel_confidence and (
        panel_service is not None or workflow_port is not None
    ):
        target_date = business_today(timezone, now_utc=now)

        # Claim the run BEFORE awaiting so a concurrent daily-panel cron tick
        # cannot also fire the panel (review C4 race). If we lose the claim,
        # another caller already ran (or is running) today's panel — skip.
        async def _run_panel() -> Any:
            if workflow_port is not None:
                result: AnalysisResult = await workflow_port.run_analysis(
                    AnalysisRequest(
                        user_id=user_id,
                        target_date=target_date,
                        force=False,
                        origin="auto_intervention",
                    )
                )
                return result.verdict
            assert panel_service is not None
            return await panel_service.run_daily_panel(
                user_id=user_id, target_date=target_date
            )

        try:
            claimed, verdict = await _run_claimed_job(
                scheduled_job_runs_repository,
                "daily_panel",
                target_date,
                _run_panel,
                retry_failed=True,
            )
            if claimed and verdict is not None:
                logger.info(
                    "Auto-intervention: confidence {:.2f} >= {:.2f}, escalating to expert panel",
                    top_confidence,
                    panel_confidence,
                )
                panel_attempted = True
                assessment_for_dispatch = ProcrastinationAssessment(
                    types=verdict.types,
                    confidence=dict(verdict.confidence),
                    recommended_technique=verdict.recommended_technique,
                    rationale=verdict.rationale,
                    source="rule_engine",
                )
        except (PanelUnavailableError, PanelBudgetExceededError) as exc:
            logger.warning("Panel escalation failed ({}), falling back to rule engine", exc)
        except Exception as exc:
            logger.error("Panel escalation unexpected error: {}", exc)

    if panel_attempted and assessment_for_dispatch is assessment:
        logger.info("Auto-intervention: panel failed, falling back to rule-based dispatch")

    # ── Dispatch intervention ───────────────────────────────────────
    try:
        result = await intervention_service.maybe_intervene(
            assessment=assessment_for_dispatch,
            recent_events=events,
            user_id=user_id,
        )
        if result.skipped:
            await _record(
                "skipped",
                confidence=top_confidence,
                intervention_type=
                str(result.intervention.intervention_type)
                if result.intervention
                else None,
                throttle_reason=
                str(result.throttle_decision.reason)
                if result.throttle_decision
                else None,
            )
            logger.info(
                "Auto-intervention: skipped ({}) — {}",
                result.skip_reason,
                result.throttle_decision or "",
            )
        else:
            await _record(
                "dispatched",
                confidence=top_confidence,
                intervention_type=
                str(result.intervention.intervention_type)
                if result.intervention
                else None,
                source="panel" if panel_attempted else "rule_engine",
            )
            logger.info(
                "Auto-intervention: dispatched {} to user {}",
                result.intervention.id if result.intervention else "?",
                user_id,
            )
    except Exception as exc:
        logger.error("Auto-intervention: dispatch failed: {}", exc)
        await _record("dispatch_error", confidence=top_confidence)


def build_scheduler(
    analysis_service: AnalysisService | None = None,
    report_service: ReportService | None = None,
    maintenance_service: MaintenanceService | None = None,
    intervention_service: InterventionService | None = None,
    activity_repository: SQLAlchemyActivityRepository | None = None,
    panel_service: Any | None = None,
    autonomy_service: AutonomyService | None = None,
    telemetry_service: Any | None = None,
    training_job_service: Any | None = None,
    scheduled_job_runs_repository: ScheduledJobRunsPort | None = None,
    workflow_port: AnalysisWorkflowPort | None = None,
    event_retention_days: int = 30,
    min_confidence: float = _AUTO_INTERVENTION_MIN_CONFIDENCE,
    panel_confidence: float = _AUTO_INTERVENTION_PANEL_CONFIDENCE,
    start_hour: int = _AUTO_INTERVENTION_START_HOUR,
    end_hour: int = _AUTO_INTERVENTION_END_HOUR,
    timezone: TimezoneLike = "local",
) -> AsyncioScheduler:
    """Create a pure-asyncio scheduler with cron + interval jobs.

    Uses ``AsyncioScheduler`` (pure asyncio, no Windows issues) instead of
    APScheduler's ``AsyncIOScheduler`` which triggers CTRL_BREAK_EVENT on
    Windows.
    """
    scheduler = AsyncioScheduler(timezone=timezone)

    async def _run_identify_for_date(
        target_date: date,
        *,
        refresh: bool = False,
        claim_name: str = "identify_sessions",
    ) -> bool:
        if analysis_service is None:
            return False

        async def _identify() -> Any:
            return await analysis_service.identify_focus_sessions(
                1,
                target_date,
                refresh=refresh,
            )

        claimed, _ = await _run_claimed_job(
            scheduled_job_runs_repository,
            claim_name,
            target_date,
            _identify,
            retry_failed=True,
        )
        return claimed

    async def _ensure_identified_for_date(
        target_date: date,
        *,
        refresh: bool = False,
        claim_name: str = "identify_sessions",
    ) -> None:
        if analysis_service is None:
            return
        while True:
            if await _run_identify_for_date(
                target_date,
                refresh=refresh,
                claim_name=claim_name,
            ):
                return
            if (
                scheduled_job_runs_repository is not None
                and await scheduled_job_runs_repository.has_succeeded(
                    claim_name,
                    target_date,
                )
            ):
                return
            await asyncio.sleep(_IDENTIFY_COMPLETION_POLL_SECONDS)

    async def _run_panel_for_date(target_date: date) -> bool:
        if panel_service is None and workflow_port is None:
            return False
        if autonomy_service is not None and not await autonomy_service.is_enabled(user_id=1):
            logger.debug("Daily panel: autonomy disabled, skipping")
            return False

        async def _run_panel() -> Any:
            if workflow_port is not None:
                result: AnalysisResult = await workflow_port.run_analysis(
                    AnalysisRequest(
                        user_id=1,
                        target_date=target_date,
                        force=False,
                        origin="scheduler",
                    )
                )
                return result.verdict
            assert panel_service is not None
            return await panel_service.run_daily_panel(user_id=1, target_date=target_date)

        claimed, _ = await _run_claimed_job(
            scheduled_job_runs_repository,
            "daily_panel",
            target_date,
            _run_panel,
            retry_failed=True,
        )
        return claimed

    async def _run_report_for_date(target_date: date) -> bool:
        if report_service is None:
            return False

        async def _generate_report() -> Any:
            return await report_service.generate_daily_for_all(
                target_date=target_date,
                refresh=True,
            )

        claimed, _ = await _run_claimed_job(
            scheduled_job_runs_repository,
            "daily_report",
            target_date,
            _generate_report,
            retry_failed=True,
        )
        return claimed

    async def _run_telemetry_for_date(target_date: date) -> bool:
        if telemetry_service is None:
            return False

        async def _rollup() -> None:
            start, end_inclusive = business_day_bounds_utc(target_date, timezone)
            await telemetry_service.rollup_feature_windows(
                start,
                end_inclusive + timedelta(microseconds=1),
            )
            await telemetry_service.cleanup_retained_data()

        claimed, _ = await _run_claimed_job(
            scheduled_job_runs_repository,
            "telemetry_rollup",
            target_date,
            _rollup,
            retry_failed=True,
        )
        return claimed

    # ── 23:30 — Expert panel deliberation ────────────────────────────
    if panel_service is not None or workflow_port is not None:

        async def _run_daily_panel() -> None:
            try:
                target_date = business_today(timezone)
                claimed = await _run_panel_for_date(target_date)
                if not claimed:
                    logger.debug("Daily panel: already claimed for {}, skipping", target_date)
            except Exception:
                logger.exception("Daily panel job failed")

        scheduler.daily_cron(
            hour=23,
            minute=30,
            coro=_run_daily_panel,
            name="daily_panel",
        )

    # ── 23:59 — Session identification ────────────────────────────────
    if analysis_service is not None:

        async def _run_daily_identify() -> None:
            target_date = business_today(timezone)
            await _run_identify_for_date(target_date, refresh=True)

        scheduler.daily_cron(
            hour=23,
            minute=59,
            coro=_run_daily_identify,
            name="identify_sessions",
        )

    # ── 00:05 — Daily report ──────────────────────────────────────────
    if report_service is not None:

        async def _run_daily_report() -> None:
            target_date = business_today(timezone) - timedelta(days=1)
            await _ensure_identified_for_date(
                target_date,
                refresh=True,
                claim_name="identify_sessions_final",
            )
            await _run_report_for_date(target_date)

        scheduler.daily_cron(
            hour=0,
            minute=5,
            coro=_run_daily_report,
            name="daily_report",
        )

    async def _run_startup_recovery(now_utc: datetime) -> None:
        now_local = now_utc.astimezone(scheduler.timezone)
        complete_date = now_local.date() - timedelta(days=_STARTUP_RECOVERY_COMPLETE_DAYS)

        async def _run_recovery_step(
            name: str,
            action: Callable[[], Awaitable[Any]],
            *,
            retries: int = 0,
        ) -> bool:
            for attempt in range(retries + 1):
                try:
                    await action()
                    return True
                except Exception:
                    logger.exception(
                        "Startup recovery step {} failed (attempt {}/{})",
                        name,
                        attempt + 1,
                        retries + 1,
                    )
                    if attempt < retries:
                        await asyncio.sleep(0)
            return False

        identify_ready = analysis_service is None
        if analysis_service is not None:
            identify_ready = await _run_recovery_step(
                f"identify_sessions:{complete_date}",
                lambda: _ensure_identified_for_date(
                    complete_date,
                    refresh=True,
                    claim_name="identify_sessions_final",
                ),
                retries=_STARTUP_RECOVERY_RETRIES,
            )
        if panel_service is not None or workflow_port is not None:
            await _run_recovery_step(
                f"daily_panel:{complete_date}",
                lambda: _run_panel_for_date(complete_date),
            )
        if report_service is not None:
            if identify_ready:
                await _run_recovery_step(
                    f"daily_report:{complete_date}",
                    lambda: _run_report_for_date(complete_date),
                    retries=_STARTUP_RECOVERY_RETRIES,
                )
            else:
                logger.warning(
                    "Startup recovery skipped daily_report for {} because "
                    "identify_sessions did not complete",
                    complete_date,
                )
        if telemetry_service is not None:
            # ── Todo 12: conditional V2 baseline backfill, then the bounded
            # current recent-window catch-up ending at the single captured
            # startup ``now_utc``, then the complete-day yesterday rollup.
            #
            # The backfill runs FIRST so an existing window history can seed
            # the baseline before any rollup creates a partial one; the recent
            # catch-up then folds only newly inserted windows through the
            # Todo 8 idempotent seam. The catch-up bound reuses the two-hour
            # recent-rollup policy, not an unbounded full-day recompute.
            # A rerun after interruption skips the backfill (V2 baseline
            # exists) and re-upserts the same windows, so one V2 baseline and
            # stable counts persist. Both new steps use the existing step
            # boundary (log-and-continue, never falsely claiming completion).

            async def _backfill_baseline() -> None:
                result = await telemetry_service.rebuild_baseline_if_needed(
                    user_id=1,
                    timezone=timezone,
                    now_utc=now_utc,
                )
                logger.info(
                    "Startup recovery baseline backfill: rebuilt={} reason={} "
                    "windows_loaded={} samples={} cutoff_utc={}",
                    result.rebuilt,
                    result.reason,
                    result.windows_loaded,
                    result.samples,
                    result.cutoff_utc,
                )

            await _run_recovery_step(
                "baseline_backfill",
                _backfill_baseline,
                retries=_STARTUP_RECOVERY_RETRIES,
            )
            await _run_recovery_step(
                "telemetry_rollup_recent:startup",
                lambda: telemetry_service.rollup_feature_windows(
                    now_utc - timedelta(hours=_RECENT_ROLLUP_WINDOW_HOURS),
                    now_utc,
                    user_id=1,
                ),
                retries=_STARTUP_RECOVERY_RETRIES,
            )
            await _run_recovery_step(
                f"telemetry_rollup:{complete_date}",
                lambda: _run_telemetry_for_date(complete_date),
                retries=_STARTUP_RECOVERY_RETRIES,
            )

        current_date = now_local.date()
        if (panel_service is not None or workflow_port is not None) and (now_local.hour, now_local.minute) >= (23, 30):
            await _run_recovery_step(
                f"daily_panel:{current_date}",
                lambda: _run_panel_for_date(current_date),
            )
        if analysis_service is not None and (now_local.hour, now_local.minute) >= (23, 59):
            await _run_recovery_step(
                f"identify_sessions:{current_date}",
                lambda: _run_identify_for_date(current_date, refresh=True),
                retries=_STARTUP_RECOVERY_RETRIES,
            )

    scheduler._startup_recovery = _run_startup_recovery

    # ── 03:00 — Event cleanup ─────────────────────────────────────────
    if maintenance_service is not None:

        async def _run_event_cleanup(retention_days: int) -> None:
            # One effective activity-retention horizon drives both raw
            # activity events and intervention_checks (preference
            # ``activity_retention_days`` wins; the env ``event_retention_days``
            # passed here is only the startup default when no preference exists).
            await maintenance_service.cleanup_old_events(
                retention_days=retention_days
            )
            await maintenance_service.cleanup_old_intervention_checks(
                retention_days=retention_days
            )
            # Budget reservations are short-lived leases.  Run expiry in the
            # same daily maintenance window so a crashed workflow cannot keep
            # its idempotency key occupied indefinitely.
            await maintenance_service.expire_stale_budgets()

        scheduler.daily_cron(
            hour=3,
            minute=0,
            coro=_run_event_cleanup,
            kwargs={"retention_days": event_retention_days},
            name="event_cleanup",
        )

    # ── 04:00 — Daily backup ──────────────────────────────────────────
    if maintenance_service is not None:

        async def _run_daily_backup() -> None:
            target_date = business_today(timezone)

            async def _backup() -> None:
                if not await maintenance_service.run_daily_backup():
                    raise RuntimeError("Daily backup failed")

            await _run_claimed_job(
                scheduled_job_runs_repository,
                "daily_backup",
                target_date,
                _backup,
                retry_failed=True,
            )

        scheduler.daily_cron(
            hour=4,
            minute=0,
            coro=_run_daily_backup,
            name="daily_backup",
            catch_up=scheduled_job_runs_repository is not None,
        )

    # ── Every 5 min — Auto intervention check ───────────────────────────
    if intervention_service is not None and activity_repository is not None:
        scheduler.interval_minutes(
            minutes=_AUTO_INTERVENTION_INTERVAL_MINUTES,
            coro=_auto_intervention_check,
            kwargs={
                "activity_repo": activity_repository,
                "intervention_service": intervention_service,
                "panel_service": panel_service,
                "workflow_port": workflow_port,
                "autonomy_service": autonomy_service,
                "scheduled_job_runs_repository": scheduled_job_runs_repository,
                "min_confidence": min_confidence,
                "panel_confidence": panel_confidence,
                "timezone": timezone,
                "start_hour": start_hour,
                "end_hour": end_hour,
            },
            name="auto_intervention_check",
        )

    if telemetry_service is not None:

        async def _rollup_telemetry() -> None:
            yesterday = business_today(timezone) - timedelta(days=1)
            await _run_telemetry_for_date(yesterday)

        scheduler.daily_cron(
            hour=2,
            minute=45,
            coro=_rollup_telemetry,
            name="telemetry_rollup",
        )

        # ── Every 15 min — Recent-window telemetry rollup ────────────────
        # Rolls the trailing [now-2h, now] window through the Todo 8
        # idempotent seam, so overlapping re-rolls just re-upsert the same
        # windows and the baseline folds only newly inserted rows. No
        # per-date claim (the window shifts continuously); the interval
        # boundary's catch-and-log keeps later invocations alive.
        async def _rollup_recent_telemetry() -> None:
            now = datetime.now(UTC)
            await telemetry_service.rollup_feature_windows(
                now - timedelta(hours=_RECENT_ROLLUP_WINDOW_HOURS),
                now,
                user_id=1,
            )

        scheduler.interval_minutes(
            minutes=_RECENT_ROLLUP_INTERVAL_MINUTES,
            coro=_rollup_recent_telemetry,
            name="telemetry_rollup_recent",
        )

    # ── Hourly — Auto incremental training check (architecture plan F) ──
    # The model improves itself when new feedback accumulates; the job runs
    # in shadow mode and never auto-activates.
    if training_job_service is not None:

        async def _auto_train_check() -> None:
            try:
                started = await training_job_service.auto_train_if_due()
                if started:
                    logger.info("Auto-training started (new feedback accumulated)")
            except Exception as exc:
                logger.error("Auto-training check failed: {}", exc)

        scheduler.interval_minutes(
            minutes=60,
            coro=_auto_train_check,
            name="auto_training_check",
        )

    logger.info(
        "AsyncioScheduler built with jobs: {}",
        [j["name"] for j in scheduler._jobs],
    )
    return scheduler
