"""Intervention service — generates, throttles, and dispatches interventions.

This is the central orchestrator for Wave 7.  It is called by:
  - The scheduler (automated, throttled)
  - The manual trigger endpoint (bypasses throttle, respects rate-limit)

Flow:
  1. Deep-work guard: if current focus_score > 80, return zero-intervention
  2. Throttle check (automated only)
  3. Select intervention type from assessment types
  4. Look up CBT technique from the top procrastination type
  5. Render message from intensity-based templates
  6. Persist intervention log
  7. Broadcast via WebSocket (``intervention`` frame type)
  8. Desktop notification (best-effort, never raises)

Design:
  - ``maybe_intervene()`` never raises — errors are logged and returned
    as structured ``InterventionResult``.
  - LLM enhancement slot via ``enhance_with_llm`` parameter (default False).
  - ``record_response()`` updates the log with user feedback.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from loguru import logger

from mindflow.domain.events import ActivityEvent
from mindflow.domain.features import focus_score
from mindflow.domain.ids import new_id
from mindflow.domain.intervention import (
    INTENSITY_TEMPLATES,
    INTERVENTION_TYPE_LABELS,
    Intervention,
    InterventionIntensity,
    InterventionType,
)
from mindflow.domain.procrastination import (
    CBTTechnique,
    ProcrastinationAssessment,
    ProcrastinationType,
)
from mindflow.infrastructure.notification import NotificationService
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.services.intervention_throttle import (
    InterventionThrottle,
    ThrottleDecision,
    ThrottleReason,
)

# ── Type → detail/suggestion templates (Chinese, NF-S7 compliant) ───────

_TYPE_TEMPLATES: dict[str, dict[str, str]] = {
    "task_breakdown": {
        "detail": "面临的任务较大，可能感到难以着手",
        "suggestion": "将任务拆解为 3-5 个小步骤，每次完成一个小目标",
    },
    "nudge": {
        "detail": "似乎有些分心或延迟启动",
        "suggestion": "设定一个 5 分钟计时器，先开始一小步",
    },
    "environment_optimization": {
        "detail": "工作环境中存在较多干扰源",
        "suggestion": "关闭无关标签页，将手机调至勿扰模式",
    },
    "smart_prioritization": {
        "detail": "同时处理多个任务，注意力可能分散",
        "suggestion": "按优先级排序，先完成最重要的一个任务",
    },
}

_MAX_TYPES: int = 3

# ProcrastinationType → InterventionType mapping
_TYPE_MAP: dict[ProcrastinationType, InterventionType] = {
    ProcrastinationType.TASK_AVERSION: "task_breakdown",
    ProcrastinationType.IMPULSIVITY: "environment_optimization",
    ProcrastinationType.DECISIONAL: "nudge",
    ProcrastinationType.PERFECTIONISM: "smart_prioritization",
    ProcrastinationType.EMOTIONAL_REGULATION: "nudge",
}

# InterventionType → CBT technique override (when assessment has no technique)
_INTERVENTION_CBT_MAP: dict[InterventionType, str] = {
    "task_breakdown": str(CBTTechnique.GOAL_SETTING),
    "nudge": str(CBTTechnique.BEHAVIORAL_EXPERIMENT),
    "environment_optimization": str(CBTTechnique.STIMULUS_CONTROL),
    "smart_prioritization": str(CBTTechnique.GOAL_SETTING),
}


class InterventionResult:
    """Structured result from ``maybe_intervene()``.

    Attributes:
        intervention: The created Intervention, or None if skipped/throttled.
        skipped: True if the intervention was skipped (deep work / throttle / no type).
        skip_reason: Human-readable explanation for the skip.
        throttle_decision: The throttle decision, for debugging.
    """

    def __init__(
        self,
        intervention: Intervention | None = None,
        skipped: bool = False,
        skip_reason: str = "",
        throttle_decision: ThrottleDecision | None = None,
    ) -> None:
        self.intervention = intervention
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.throttle_decision = throttle_decision


def _deep_work_guard(
    events: list[ActivityEvent],
    threshold: float = 80.0,
) -> bool:
    """Return True if the user appears to be in deep work (>threshold)."""
    if not events:
        return False
    score = focus_score(events)
    return score > threshold


def _select_intervention_type(
    assessment: ProcrastinationAssessment,
) -> InterventionType | None:
    """Select the best intervention type from an assessment.

    Uses the top-confidence procrastination type to determine the
    intervention category.  Returns None if no significant pattern
    is detected.
    """
    if not assessment.types:
        return None
    if (
        assessment.recommended_technique is None
        and assessment.confidence.get(assessment.types[0], 0) < 0.2
    ):
        return None
    ptype = assessment.types[0]
    return _TYPE_MAP.get(ptype)


def _render_message(
    intervention_type: InterventionType,
    intensity: InterventionIntensity,
    cbt_technique: str | None = None,
) -> tuple[str, str]:
    """Render notification title and body from templates.

    Args:
        intervention_type: Type of intervention.
        intensity: Tone/intensity level.
        cbt_technique: Optional CBT technique to include.

    Returns:
        A (title, body) tuple.
    """
    type_label = INTERVENTION_TYPE_LABELS.get(intervention_type, intervention_type)
    tmpl = _TYPE_TEMPLATES.get(intervention_type, _TYPE_TEMPLATES["nudge"])
    detail = tmpl["detail"]
    suggestion = tmpl["suggestion"]

    if cbt_technique:
        suggestion = f"{suggestion}（可尝试 {cbt_technique} 方法）"

    title_tmpl, body_tmpl = INTENSITY_TEMPLATES[intensity]

    title = title_tmpl.format(type_label=type_label)
    body = body_tmpl.format(detail=detail, suggestion=suggestion)

    return title, body


class InterventionService:
    """Central intervention orchestrator.

    Args:
        intervention_repo: Intervention log repository.
        throttle: Intervention throttle (includes its own repo reference).
        notifier: Desktop notification service.
        activity_repo: Activity repository for deep-work check.
        broadcast_fn: Async callable for WebSocket broadcast.
            Signature: ``async broadcast(message: dict) -> int``.
    """

    def __init__(  # noqa: PLR0913 — service wiring
        self,
        intervention_repo: InterventionLogRepository,
        throttle: InterventionThrottle,
        notifier: NotificationService,
        activity_repo: object | None = None,
        broadcast_fn: Callable[..., Awaitable[int]] | None = None,
    ) -> None:
        self._repo = intervention_repo
        self._throttle = throttle
        self._notifier = notifier
        self._activity_repo = activity_repo
        self._broadcast_fn = broadcast_fn

    # ── Public API ────────────────────────────────────────────────────

    async def maybe_intervene(  # noqa: PLR0913 — many params is intentional
        self,
        assessment: ProcrastinationAssessment,
        intensity: InterventionIntensity = InterventionIntensity.STANDARD,
        *,
        bypass_throttle: bool = False,
        enhance_with_llm: bool = False,
        recent_events: list[ActivityEvent] | None = None,
        user_id: int = 1,
    ) -> InterventionResult:
        """Evaluate and potentially dispatch an intervention.

        Args:
            assessment: The procrastination assessment to act on.
            intensity: Intervention tone intensity.
            bypass_throttle: If True, skip throttle check
                (for manual trigger).
            enhance_with_llm: Reserved for future LLM enhancement
                (currently ignored — always False).
            recent_events: Recent activity events for deep-work detection.
            user_id: User identifier.

        Returns:
            An ``InterventionResult`` describing what happened.
        """
        # ── 1. Select intervention type ────────────────────────────────
        intervention_type = _select_intervention_type(assessment)
        if intervention_type is None:
            return InterventionResult(
                skipped=True,
                skip_reason="未检测到显著的拖延模式，无需干预",
            )

        # ── 2. Deep-work guard ────────────────────────────────────────
        if recent_events is not None and _deep_work_guard(recent_events):
            logger.debug("Deep work detected — skipping intervention")
            return InterventionResult(
                skipped=True,
                skip_reason="当前处于深度专注状态 (focus_score>80)，零打扰",
            )

        # ── 3. Throttle check ─────────────────────────────────────────
        if not bypass_throttle:
            decision = await self._throttle.can_intervene(user_id, intervention_type)
            if not decision.allowed:
                logger.debug("Intervention throttled: {}", decision.reason)
                return InterventionResult(
                    skipped=True,
                    skip_reason=decision.detail,
                    throttle_decision=decision,
                )
        else:
            decision = ThrottleDecision(ThrottleReason.OK, detail="手动触发，绕过节流")

        # ── 4. Determine CBT technique ────────────────────────────────
        cbt_technique: str | None = None
        if assessment.recommended_technique:
            cbt_technique = str(assessment.recommended_technique)
        else:
            cbt_technique = _INTERVENTION_CBT_MAP.get(intervention_type)

        # ── 5. Render message ─────────────────────────────────────────
        title, message = _render_message(intervention_type, intensity, cbt_technique)

        # ── 6. Create domain object ───────────────────────────────────
        now = datetime.now(UTC)
        intervention = Intervention(
            id=new_id(),
            user_id=user_id,
            intervention_type=intervention_type,
            cbt_technique=cbt_technique,
            title=title,
            message=message,
            dismissible=True,
            created_at=now,
        )

        # ── 7. Persist ────────────────────────────────────────────────
        try:
            context = {
                "procrastination_types": [str(t) for t in assessment.types],
                "confidence": {str(t): round(c, 3) for t, c in assessment.confidence.items()},
                "intensity": str(intensity),
                "bypass_throttle": bypass_throttle,
            }
            await self._repo.log_triggered(
                user_id=user_id,
                intervention_type=intervention_type,
                cbt_technique=cbt_technique,
                context=context,
                intervention_id=intervention.id,
                triggered_at=now,
            )
        except Exception as exc:
            logger.error("Failed to persist intervention log: {}", exc)
            # Continue — don't fail the user experience for a log write

        # ── 8. Broadcast via WebSocket ────────────────────────────────
        await self._broadcast_intervention(intervention)

        # ── 9. Desktop notification ────────────────────────────────────
        await self._notifier.send(
            title=intervention.title,
            body=intervention.message,
            urgency="normal",
        )

        return InterventionResult(intervention=intervention)

    async def record_response(
        self,
        intervention_id: str,
        response: str,
        latency_s: float = 0.0,
    ) -> dict[str, Any] | None:
        """Record a user's response to an intervention.

        Args:
            intervention_id: The intervention's UUID.
            response: One of "accepted", "ignored", "dismissed".
            latency_s: Seconds between trigger and response.

        Returns:
            The updated log dict, or None if the intervention wasn't found.
        """
        from mindflow.infrastructure.repositories.intervention import ResponseType

        try:
            result = await self._repo.update_response(
                intervention_id,
                cast("ResponseType", response),
                latency_s,
            )
            if result is None:
                logger.warning("Intervention {} not found for response", intervention_id)
            return result
        except Exception as exc:
            logger.error("Failed to record intervention response: {}", exc)
            return None

    async def get_history(
        self,
        user_id: int,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Return intervention history for the past N days.

        Args:
            user_id: User identifier.
            days: Number of days of history to return.

        Returns:
            A list of intervention log dicts.
        """
        now = datetime.now(UTC)
        start = now - timedelta(days=days)
        return await self._repo.query_range(user_id, start, now)

    # ── Internal helpers ──────────────────────────────────────────────

    async def _broadcast_intervention(self, intervention: Intervention) -> None:
        """Broadcast an intervention frame via WebSocket (best-effort)."""
        if self._broadcast_fn is None:
            return
        try:
            message = {
                "type": "intervention",
                "payload": {
                    "id": intervention.id,
                    "intervention_type": intervention.intervention_type,
                    "title": intervention.title,
                    "message": intervention.message,
                    "dismissible": intervention.dismissible,
                    "cbt_technique": intervention.cbt_technique,
                },
            }
            await self._broadcast_fn(message)
        except Exception as exc:
            logger.warning("WebSocket broadcast failed for intervention: {}", exc)
