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
      "meta": { "degraded": false }
    }

When the expert panel is unavailable and falls through to a lower tier,
``meta.degraded`` is ``true`` and ``source`` identifies ``"single_expert"``,
``"ollama"``, or ``"rule_engine"``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request  # noqa: B008
from loguru import logger

from mindflow.api.deps import get_panel_service, get_workflow_port
from mindflow.api.errors import ProblemDetail
from mindflow.api.schemas import PanelResponse
from mindflow.errors import NoActivityDataError
from mindflow.ports import AnalysisRequest, AnalysisWorkflowPort
from mindflow.services.panel_service import PanelService
from mindflow.time_utils import business_today

router = APIRouter(tags=["panel"])


def _verdict_to_response(verdict: Any) -> PanelResponse:
    """Convert a PanelVerdict to a serializable dict.

    Handles both ``PanelVerdict`` dataclass instances and anything with
    matching attributes. Returns a dict shaped for the API response with
    ``meta.degraded``.
    """
    is_degraded = verdict.source != "panel"

    # Convert types and confidence ProcrastinationType/StrEnum → str
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
        "meta": {"degraded": is_degraded},
    })


@router.post("/panel/today", response_model=PanelResponse)
async def post_panel_today(
    request: Request,
    panel_service: PanelService = Depends(get_panel_service),  # noqa: B008
    workflow_port: AnalysisWorkflowPort | None = Depends(get_workflow_port),  # noqa: B008
) -> PanelResponse:
    """Trigger a daily expert panel for today.

    Runs the full multi-expert panel (analyst → attribution ×3 → moderator → critic).
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

    # Idempotency guard: if today's panel already ran (via cron or earlier
    # POST), return the stored result instead of incurring duplicate LLM cost.
    # Without this check, a concurrent 23:30 cron and manual POST would both
    # execute the expensive 6-12 call panel (Codex review finding).
    existing = await panel_service.get_stored_verdict(user_id=1, target_date=today)
    if existing is not None:
        logger.info("Panel already exists for {}, returning stored result", today)
        return _verdict_to_response(existing)

    logger.info("Triggering daily panel for user 1 on {}", today)

    try:
        if workflow_port is not None:
            result = await workflow_port.run_analysis(
                AnalysisRequest(
                    user_id=1,
                    target_date=today,
                    force=False,
                    origin="api",
                )
            )
            verdict = result.verdict
        else:
            verdict = await panel_service.run_daily_panel(user_id=1, target_date=today)
    except (ProblemDetail, NoActivityDataError):
        # The single-expert fallback (llm_service.analyze) raises
        # NoActivityDataError when no events exist — let it reach the
        # registered 404 handler instead of collapsing to a 500 (E4).
        raise
    except Exception:
        logger.exception("Panel service failed unexpectedly for user 1 on {}", today)
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    return _verdict_to_response(verdict)


@router.get("/panel", response_model=PanelResponse)
async def get_panel_result(
    request: Request,
    panel_service: PanelService = Depends(get_panel_service),  # noqa: B008
) -> PanelResponse:
    """Retrieve the most recent stored panel result (read-only, idempotent).

    A GET must not trigger the expensive 6-12-call expert panel (review C3 —
    that would cost money and violate REST idempotency). This reads the last
    persisted attribution for today (written by ``POST /panel/today`` or the
    daily cron) and returns it, or 404 if none has been produced yet.

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

    return _verdict_to_response(verdict)
