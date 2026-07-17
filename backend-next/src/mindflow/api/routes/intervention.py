"""API routes for Wave 7 intervention engine.

Endpoints:
  - POST /api/v1/intervention/trigger  — Manual trigger (bypasses throttle)
  - POST /api/v1/intervention/{id}/response — Record user response
  - GET  /api/v1/intervention/history  — Intervention history

Manual trigger bypasses the throttle but still counts toward rate limits
for future automated checks.  It is intended for scenarios where the user
explicitly requests feedback (e.g. via the frontend intervention panel).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Path, Query  # noqa: B008
from loguru import logger

from mindflow.api.deps import get_intervention_service
from mindflow.api.errors import _not_found
from mindflow.domain.intervention import InterventionIntensity
from mindflow.domain.procrastination import (
    BehaviorSummary,
    RuleEngine,
)
from mindflow.services.intervention_service import InterventionService

router = APIRouter(tags=["intervention"])

_DEFAULT_INTENSITY = InterventionIntensity.STANDARD


@router.post("/intervention/trigger")
async def trigger_intervention(
    intensity: str | None = None,
    intervention_svc: InterventionService = Depends(get_intervention_service),  # noqa: B008
) -> dict[str, Any]:
    """Manually trigger an intervention (bypasses throttle).

    This uses a lightweight rule-engine assessment of the *current*
    default behavior summary to determine the appropriate intervention
    type.  In production, the assessment is pre-computed by the
    attribution pipeline; this endpoint creates one on the fly for
    on-demand triggers.

    Args:
        intensity: Optional override for intervention intensity.
            One of "gentle", "standard", "strict".

    Returns:
        The intervention result.
    """
    # Resolve intensity
    if intensity:
        try:
            resolved_intensity = InterventionIntensity(intensity)
        except ValueError:
            resolved_intensity = _DEFAULT_INTENSITY
    else:
        resolved_intensity = _DEFAULT_INTENSITY

    # Build a minimal assessment from a default summary.
    # This is a reasonable estimate for on-demand triggers; the
    # full attribution pipeline feeds the automated path.
    rule_engine = RuleEngine()
    # A neutral summary — the rule engine will produce a low-confidence
    # assessment which the intervention service handles gracefully.
    summary = BehaviorSummary(
        intended_task=None,
        duration_min=60.0,
        actual_focus_min=20.0,
        context_switches_per_hour=15.0,
        longest_focus_block_s=180.0,
        social_media_ratio=0.3,
        start_delay_min=15.0,
        keyword_flags=frozenset(),
        baseline_deviation=None,
    )
    assessment = rule_engine.assess(summary)

    result = await intervention_svc.maybe_intervene(
        assessment=assessment,
        intensity=resolved_intensity,
        bypass_throttle=True,
        recent_events=None,
    )

    if result.skipped:
        return {
            "intervention": None,
            "skipped": True,
            "skip_reason": result.skip_reason,
        }

    if result.intervention is None:
        return {
            "intervention": None,
            "skipped": True,
            "skip_reason": "未能生成干预",
        }

    logger.info("Manual intervention triggered: {}", result.intervention.id)
    return {
        "intervention": {
            "id": result.intervention.id,
            "intervention_type": result.intervention.intervention_type,
            "title": result.intervention.title,
            "message": result.intervention.message,
            "dismissible": result.intervention.dismissible,
            "created_at": result.intervention.created_at.isoformat(),
        },
        "skipped": False,
    }


@router.post("/intervention/{intervention_id}/response")
async def respond_to_intervention(
    intervention_id: str = Path(..., description="Intervention UUID"),  # noqa: B008
    response: str = Query(..., description="Response: accepted/ignored/dismissed"),  # noqa: B008
    latency_s: float = Query(0.0, description="Response latency in seconds"),  # noqa: B008
    intervention_svc: InterventionService = Depends(get_intervention_service),  # noqa: B008
) -> dict[str, Any]:
    """Record a user's response to an intervention.

    Args:
        intervention_id: The intervention's UUID.
        response: One of "accepted", "ignored", "dismissed".
        latency_s: Time in seconds between trigger and response.

    Returns:
        A confirmation dict, or 404 if the intervention isn't found.
    """
    valid_responses = {"accepted", "ignored", "dismissed"}
    if response not in valid_responses:
        return {"error": f"无效的响应值。可用值: {', '.join(sorted(valid_responses))}"}

    result = await intervention_svc.record_response(intervention_id, response, latency_s)
    if result is None:
        raise _not_found(f"干预记录 {intervention_id}")

    logger.debug("Intervention {} response: {} (latency={}s)", intervention_id, response, latency_s)
    return {
        "status": "ok",
        "intervention_id": intervention_id,
        "user_response": response,
    }


@router.get("/intervention/history")
async def get_intervention_history(
    days: int = Query(7, ge=1, le=90, description="Days of history to return"),  # noqa: B008
    intervention_svc: InterventionService = Depends(get_intervention_service),  # noqa: B008
) -> dict[str, Any]:
    """Return intervention history for the past N days.

    Args:
        days: Number of days of history (1-90).

    Returns:
        A dict with ``interventions`` list and ``count``.
    """
    history = await intervention_svc.get_history(user_id=1, days=days)
    return {
        "interventions": history,
        "count": len(history),
    }
