"""Intervention service — generates, throttles, and dispatches interventions.

This is the central orchestrator for Wave 7.  It is called by:
  - The scheduler (automated, throttled)
  - The manual trigger endpoint (bypasses throttle, respects rate-limit)

Flow:
  1. Deep-work guard: if current focus_score > 80, return zero-intervention
  2. Throttle check (automated only)
  3. Select intervention type from assessment types
  4. Look up CBT technique from the top procrastination type
  5. Render message — AI-generated (primary) or template fallback
  6. Persist intervention log
  7. Broadcast via WebSocket (``intervention`` frame type)
  8. Desktop notification (best-effort, never raises)

Design:
  - ``maybe_intervene()`` never raises — errors are logged and returned
    as structured ``InterventionResult``.
  - LLM enhancement is attempted first; template fallback ensures messages
    are always generated even when LLM is unavailable or fails.
  - ``record_response()`` updates the log with user feedback.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx
from loguru import logger

from mindflow.domain.events import ActivityEvent
from mindflow.domain.features import focus_score
from mindflow.domain.ids import new_id
from mindflow.domain.intervention import (
    CBT_TECHNIQUE_LABELS_ZH,
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
from mindflow.infrastructure.llm.summary import serialize_summary
from mindflow.infrastructure.notification import NotificationService
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.services.intervention_throttle import (
    InterventionThrottle,
    ThrottleDecision,
    ThrottleReason,
)

# ── Type -> detail/suggestion templates (Chinese, NF-S7 compliant) ───────

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

# ProcrastinationType -> InterventionType mapping
_TYPE_MAP: dict[ProcrastinationType, InterventionType] = {
    ProcrastinationType.TASK_AVERSION: "task_breakdown",
    ProcrastinationType.IMPULSIVITY: "environment_optimization",
    ProcrastinationType.DECISIONAL: "nudge",
    ProcrastinationType.PERFECTIONISM: "smart_prioritization",
    ProcrastinationType.EMOTIONAL_REGULATION: "nudge",
}

# InterventionType -> CBT technique override (when assessment has no technique)
_INTERVENTION_CBT_MAP: dict[InterventionType, str] = {
    "task_breakdown": str(CBTTechnique.GOAL_SETTING),
    "nudge": str(CBTTechnique.BEHAVIORAL_EXPERIMENT),
    "environment_optimization": str(CBTTechnique.STIMULUS_CONTROL),
    "smart_prioritization": str(CBTTechnique.GOAL_SETTING),
}

# ── LLM intervention message generation ──────────────────────────────────

_LLM_SYSTEM_PROMPT: str = (
    "你是一个友善的专注力教练。根据用户近期的行为数据，生成一条简短的干预提醒。\n"
    "要求：\n"
    "- 语气温暖但不啰嗦，像朋友提醒而非说教\n"
    "- 基于具体数据给出建议，不要泛泛而谈\n"
    "- 不要使用\"诊断\"、\"治疗\"、\"患者\"、\"处方\"等医疗用语\n"
    "- 不要解释你的分析过程，直接给出提醒内容\n\n"
    "输出 JSON 格式：\n"
    '  "title": 提醒标题(6字以内)\n'
    '  "message": 提醒正文(100字以内，包含具体建议)\n'
    '  "urgency": 紧急程度("low"/"medium"/"high")\n'
    "只输出 JSON，不要其他内容。"
)

_LLM_TIMEOUT_S: float = 10.0


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


def _render_template_message(
    intervention_type: InterventionType,
    intensity: InterventionIntensity,
    cbt_technique: str | None = None,
) -> tuple[str, str]:
    """Render notification title and body from templates (fallback).

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
        cbt_label = CBT_TECHNIQUE_LABELS_ZH.get(cbt_technique, cbt_technique)
        suggestion = f"{suggestion}（可尝试 {cbt_label} 方法）"

    title_tmpl, body_tmpl = INTENSITY_TEMPLATES[intensity]

    title = title_tmpl.format(type_label=type_label)
    body = body_tmpl.format(detail=detail, suggestion=suggestion)

    return title, body


async def _generate_llm_message(
    llm_client: httpx.AsyncClient,
    model: str,
    summary_json: str,
    intervention_type: str,
    intensity: str,
    cbt_technique: str | None = None,
) -> tuple[str, str] | None:
    """Generate intervention message via LLM (primary path).

    Feeds recent behavior data directly to the LLM and asks for a
    short, warm intervention message. No chain-of-thought — just
    data in, message out.

    Returns:
        (title, message) on success, None on any failure.
    """
    context_parts = [
        f"行为数据: {summary_json}",
        f"干预类型: {INTERVENTION_TYPE_LABELS.get(intervention_type, intervention_type)}",
        f"提醒强度: {intensity}",
    ]
    if cbt_technique:
        cbt_label = CBT_TECHNIQUE_LABELS_ZH.get(cbt_technique, cbt_technique)
        context_parts.append(f"建议方法: {cbt_label}")

    user_content = "\n".join(context_parts)

    try:
        response = await llm_client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.7,
                "max_tokens": 200,
            },
            timeout=_LLM_TIMEOUT_S,
        )

        if response.status_code != 200:
            logger.warning("LLM intervention message: HTTP {}", response.status_code)
            return None

        body = response.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            logger.warning("LLM intervention message: empty response")
            return None

        # Strip markdown code fences if present (some models wrap JSON in ```...```)
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json or ```) and last line (```)
            content = "\n".join(lines[1:-1]).strip()

        parsed = json.loads(content)
        title = str(parsed.get("title", "")).strip()
        message = str(parsed.get("message", "")).strip()

        if not title or not message:
            logger.warning("LLM intervention message: missing title or message")
            return None

        # Enforce length limits
        if len(title) > 15:
            title = title[:15]
        if len(message) > 200:
            message = message[:197] + "..."

        logger.debug("LLM generated intervention message: title={!r}", title)
        return title, message

    except httpx.TimeoutException:
        logger.warning("LLM intervention message: timeout ({}s)", _LLM_TIMEOUT_S)
        return None
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.warning("LLM intervention message: parse error: {}", exc)
        return None
    except Exception as exc:
        logger.warning("LLM intervention message: unexpected error: {}", exc)
        return None


def _enrich_history_item(row: dict[str, Any]) -> dict[str, Any]:
    """Enrich a history row with display-ready title and message.

    - New-style records (``context_json.title`` present) → promote stored text.
    - Legacy records (no stored text) → deterministic Chinese fallback from
      ``_render_template_message()`` using the recorded intensity when
      available, otherwise ``InterventionIntensity.STANDARD``.
    """
    ctx = row.get("context_json") or {}
    title = ctx.get("title")
    message = ctx.get("message")

    if title and message:
        # New record with stored prompt text — promote to top-level
        row["title"] = title
        row["message"] = message
        return row

    # Legacy record — derive fallback
    intervention_type = str(row.get("intervention_type", "nudge"))
    intensity_raw = ctx.get("intensity") if isinstance(ctx, dict) else None
    try:
        intensity = (
            InterventionIntensity(intensity_raw)
            if intensity_raw
            else InterventionIntensity.STANDARD
        )
    except ValueError:
        intensity = InterventionIntensity.STANDARD

    cbt_technique = row.get("cbt_technique")
    fallback_title, fallback_message = _render_template_message(
        cast("InterventionType", intervention_type), intensity, cbt_technique
    )
    row["title"] = fallback_title
    row["message"] = fallback_message
    return row


class InterventionService:
    """Central intervention orchestrator.

    Args:
        intervention_repo: Intervention log repository.
        throttle: Intervention throttle (includes its own repo reference).
        notifier: Desktop notification service.
        activity_repo: Activity repository for deep-work check.
        broadcast_fn: Async callable for WebSocket broadcast.
            Signature: ``async broadcast(message: dict) -> int``.
        llm_client: Optional httpx.AsyncClient for LLM message generation.
            When provided, intervention messages are AI-generated.
        llm_model: The LLM model name to use for message generation.
    """

    def __init__(  # noqa: PLR0913 — service wiring
        self,
        intervention_repo: InterventionLogRepository,
        throttle: InterventionThrottle,
        notifier: NotificationService,
        activity_repo: object | None = None,
        broadcast_fn: Callable[..., Awaitable[int]] | None = None,
        llm_client: httpx.AsyncClient | None = None,
        llm_model: str = "deepseek-chat",
        auth_token: str | None = None,
    ) -> None:
        self._repo = intervention_repo
        self._throttle = throttle
        self._notifier = notifier
        self._activity_repo = activity_repo
        self._broadcast_fn = broadcast_fn
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._auth_token = auth_token

    # ── Public API ────────────────────────────────────────────────────

    async def maybe_intervene(  # noqa: PLR0913 — many params is intentional
        self,
        assessment: ProcrastinationAssessment,
        intensity: InterventionIntensity = InterventionIntensity.STANDARD,
        *,
        bypass_throttle: bool = False,
        bypass_deep_work_guard: bool = False,
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
            bypass_deep_work_guard: If True, keep recent events as message
                context but do not suppress an explicitly requested reminder.
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
        if (
            not bypass_deep_work_guard
            and recent_events is not None
            and _deep_work_guard(recent_events)
        ):
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

        # ── 5. Generate message (AI primary, template fallback) ────────
        title: str
        message: str

        if self._llm_client is not None:
            # Build summary JSON from recent events for LLM context
            summary_json = "{}"
            if recent_events:
                try:
                    from mindflow.infrastructure.llm.summary import build_behavior_summary

                    summary = build_behavior_summary(recent_events)
                    summary_json = serialize_summary(summary)
                except Exception as exc:
                    logger.debug("Could not build summary for LLM: {}", exc)

            llm_result = await _generate_llm_message(
                llm_client=self._llm_client,
                model=self._llm_model,
                summary_json=summary_json,
                intervention_type=intervention_type,
                intensity=str(intensity.value),
                cbt_technique=cbt_technique,
            )

            if llm_result is not None:
                title, message = llm_result
            else:
                logger.debug("LLM message generation failed, using template fallback")
                title, message = _render_template_message(
                    intervention_type, intensity, cbt_technique
                )
        else:
            title, message = _render_template_message(
                intervention_type, intensity, cbt_technique
            )

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
                "message_source": "llm" if self._llm_client is not None else "template",
                "title": title,
                "message": message,
            }
            await self._repo.log_triggered(
                user_id=user_id,
                intervention_type=intervention_type,
                cbt_technique=cbt_technique,
                context=context,
                intervention_id=intervention.id,
                triggered_at=now,
                title=title,
                message=message,
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
            intervention_id=intervention.id,
            auth_token=self._auth_token,
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

    async def record_feedback(
        self,
        intervention_id: str,
        rating: str,
        comment: str | None = None,
    ) -> dict[str, Any] | None:
        """Record user feedback on intervention helpfulness.

        Args:
            intervention_id: The intervention's UUID.
            rating: One of "helpful", "neutral", "annoying".
            comment: Optional free-text comment.

        Returns:
            The updated log dict, or None if the intervention wasn't found.
        """
        from mindflow.infrastructure.repositories.intervention import FeedbackRating

        try:
            result = await self._repo.update_feedback(
                intervention_id,
                cast("FeedbackRating", rating),
                comment,
            )
            if result is None:
                logger.warning("Intervention {} not found for feedback", intervention_id)
            return result
        except Exception as exc:
            logger.error("Failed to record intervention feedback: {}", exc)
            return None

    async def get_history(
        self,
        user_id: int,
        days: int = 7,
    ) -> list[dict[str, Any]]:
        """Return intervention history for the past N days.

        Each item is enriched with ``title`` and ``message``:
        - New records carry the concrete generated text stored in
          ``context_json.title`` / ``context_json.message``.
        - Legacy records without stored prompt text receive a
          deterministic Chinese fallback derived from the existing
          intervention type, intensity, and CBT technique.

        Args:
            user_id: User identifier.
            days: Number of days of history to return.

        Returns:
            A list of enriched intervention log dicts.
        """
        now = datetime.now(UTC)
        start = now - timedelta(days=days)
        rows = await self._repo.query_range(user_id, start, now)
        return [_enrich_history_item(row) for row in rows]

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
