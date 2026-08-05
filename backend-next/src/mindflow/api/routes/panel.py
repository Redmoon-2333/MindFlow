"""API route for expert panel operations.

POST /api/v1/panel/today — Trigger daily expert panel, return PanelVerdict JSON.
GET  /api/v1/panel       — Read the most recent stored panel result (no LLM run).

Response shape aligns with ``PanelVerdict``::

    {
      "types": ["impulsivity"],
      "confidence": {"impulsivity": 0.82},
      "technique": "stimulus_control",
      "rationale": "Chinese explanation",
      "dissent": [],
      "transcript": [{"role": "...", "content": "...", "round": 0}],
      "escalated": false,
      "call_count": 6,
      "degraded": false,
      "meta": { "degraded": false, "source": "panel", "cached": false }
    }

When the expert panel is unavailable and falls through to a lower tier,
``meta.degraded`` is ``true`` and ``source`` identifies ``"single_expert"``,
``"ollama"``, or ``"rule_engine"``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from fastapi import APIRouter, Depends, Request  # noqa: B008
from loguru import logger
from pydantic import BaseModel, Field

from mindflow.api.deps import get_panel_service, get_workflow_port
from mindflow.api.errors import ProblemDetail
from mindflow.api.schemas import PanelResponse
from mindflow.errors import NoActivityDataError
from mindflow.ports import AnalysisRequest, AnalysisWorkflowPort
from mindflow.services.panel_service import PanelService
from mindflow.time_utils import business_today

router = APIRouter(tags=["panel"])


class PanelTodayRequest(BaseModel):
    """Optional body for POST /panel/today."""

    force: bool = Field(default=False, description="Force re-analysis even if cached.")
    retry_if_degraded: bool = Field(
        default=False,
        description="Retry DeepSeek when the stored result is degraded.",
    )


def _verdict_to_response(verdict: Any) -> PanelResponse:
    """Convert a PanelVerdict to a serializable dict.

    Handles both ``PanelVerdict`` dataclass instances and anything with
    matching attributes. Returns a dict shaped for the API response with
    ``meta.degraded``.
    """
    is_degraded = verdict.source != "panel"

    # Convert types and confidence ProcrastinationType/StrEnum -> str
    types_str = [str(t) for t in getattr(verdict, "types", [])]
    confidence_str: dict[str, float] = {}
    for k, v in getattr(verdict, "confidence", {}).items():
        confidence_str[str(k)] = float(v)

    # Serialize transcript
    transcript_raw = getattr(verdict, "transcript", ())
    transcript_list: list[dict[str, Any]] = [
        {
            "role": getattr(entry, "role", ""),
            "content": getattr(entry, "content", ""),
            "round": getattr(entry, "round", 0),
        }
        for entry in transcript_raw
    ]

    source = getattr(verdict, "source", None)
    cached = bool(getattr(verdict, "cached", False))
    insufficient_data = bool(getattr(verdict, "insufficient_data", False))
    uncertainty = getattr(verdict, "uncertainty", None)
    evidence_gaps = list(getattr(verdict, "evidence_gaps", ()))
    degradation_path = list(getattr(verdict, "degradation_path", ()))
    retry_after = getattr(verdict, "retry_after_s", None)

    return PanelResponse.model_validate({
        "types": types_str,
        "confidence": confidence_str,
        "technique": (
            str(verdict.recommended_technique)
            if getattr(verdict, "recommended_technique", None)
            else None
        ),
        "rationale": getattr(verdict, "rationale", ""),
        "dissent": list(getattr(verdict, "dissent", ())),
        "transcript": transcript_list,
        "escalated": getattr(verdict, "escalated", False),
        "call_count": getattr(verdict, "call_count", 0),
        "degraded": is_degraded,
        "source": source,
        "cached": cached,
        "insufficient_data": insufficient_data,
        "uncertainty": uncertainty,
        "evidence_gaps": evidence_gaps,
        "meta": {
            "degraded": is_degraded,
            "source": source,
            "cached": cached,
            "degradation_path": degradation_path,
            "retry_after": retry_after,
            "insufficient_data": insufficient_data,
            "uncertainty": uncertainty,
            "evidence_gaps": evidence_gaps,
        },
    })


@router.post("/panel/today", response_model=PanelResponse)
async def post_panel_today(
    request: Request,
    panel_service: PanelService = Depends(get_panel_service),  # noqa: B008
    workflow_port: AnalysisWorkflowPort | None = Depends(get_workflow_port),  # noqa: B008
    payload: PanelTodayRequest | None = None,
) -> PanelResponse:
    """Trigger a daily expert panel for today.

    Runs the full multi-expert panel (analyst -> attribution x3 -> moderator -> critic).
    Falls through to single-expert LLM service on panel unavailability, with
    ``meta.degraded=true``.

    When the shared ``AnalysisWorkflowPort`` is available, the analysis is
    delegated through it with ``origin="api"``, converging all entry points
    (scheduler, API, chat, auto-intervention) through a single port instance.

    Returns:
        A ``PanelVerdict`` JSON response.
    """
    settings = getattr(request.app.state, "settings", None)
    today = business_today(getattr(settings, "timezone", "local"))
    req = payload or PanelTodayRequest()

    existing = await panel_service.get_stored_verdict(user_id=1, target_date=today)
    should_retry = existing is not None and (
        req.force or (req.retry_if_degraded and existing.source != "panel")
    )

    # Idempotency guard: return the stored result unless the caller explicitly
    # asks for a retry after a previous degraded fallback.
    if existing is not None and not should_retry:
        logger.info("Panel already exists for {}, returning stored result", today)
        return _verdict_to_response(replace(existing, cached=True))

    logger.info("Triggering daily panel for user 1 on {}", today)

    try:
        if workflow_port is not None:
            result = await workflow_port.run_analysis(
                AnalysisRequest(
                    user_id=1,
                    target_date=today,
                    analysis_kind="daily_panel",
                    force=should_retry,
                    retry_if_degraded=req.retry_if_degraded,
                    origin="api",
                )
            )
            verdict = result.verdict
        else:
            verdict = await panel_service.run_daily_panel(
                user_id=1,
                target_date=today,
                force=should_retry,
                retry_if_degraded=req.retry_if_degraded,
            )
    except (ProblemDetail, NoActivityDataError):
        # Let RFC 9457 errors and the no-activity domain error propagate to
        # their registered handlers instead of collapsing to a 500.
        raise
    except Exception:
        logger.exception("Panel service failed unexpectedly for user 1 on {}", today)
        if should_retry and existing is not None:
            return _verdict_to_response(replace(existing, cached=True, retry_after_s=300))
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    if should_retry and existing is not None and verdict.source != "panel":
        logger.warning("Retry still degraded for user 1 on {}; keeping old result", today)
        return _verdict_to_response(replace(existing, cached=True, retry_after_s=300))

    return _verdict_to_response(verdict)


@router.get("/panel", response_model=PanelResponse)
async def get_panel_result(
    request: Request,
    panel_service: PanelService = Depends(get_panel_service),  # noqa: B008
) -> PanelResponse:
    """Retrieve the most recent stored panel result (read-only, idempotent).

    A GET must not trigger the expensive 6-12-call expert panel (review C3).
    This reads the last persisted attribution for today (written by
    ``POST /panel/today`` or the daily cron) and returns it, or 404 if none
    has been produced yet.

    Returns:
        A ``PanelVerdict`` JSON response matching the POST shape, or 404.
    """
    settings = getattr(request.app.state, "settings", None)
    today = business_today(getattr(settings, "timezone", "local"))
    logger.debug("GET /panel — reading stored panel result for user 1 on {}", today)

    try:
        verdict = await panel_service.get_stored_verdict(user_id=1, target_date=today)
    except ProblemDetail:
        raise
    except Exception:
        logger.exception("Failed to read stored panel result for user 1 on {}", today)
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    if verdict is None:
        from mindflow.api.errors import _not_found

        raise _not_found("今日尚无面板分析结果，请先触发 POST /api/v1/panel/today")

    return _verdict_to_response(replace(verdict, cached=True))
