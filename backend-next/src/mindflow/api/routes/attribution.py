"""API route for LLM-based procrastination attribution.

POST /api/v1/analytics/attribution

Runs the three-tier attribution pipeline (DeepSeek → Ollama → RuleEngine).
Results are cached per (user_id, date); use ``force`` to bypass cache.

Response shape::

    {
      "assessment": { ... },
      "source": "deepseek" | "ollama" | "rule_engine",
      "cached": false,
      "meta": { "degraded": false }
    }

When every LLM tier fails and the rule engine handles the request,
HTTP 200 is returned with ``meta.degraded: true`` — the request itself
completed successfully (Architecture §3.3, ADR-003).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Request  # noqa: B008
from pydantic import BaseModel, Field, field_validator

from mindflow.api.deps import get_llm_service, get_workflow_port
from mindflow.api.errors import ProblemDetail
from mindflow.errors import NoActivityDataError
from mindflow.ports import AnalysisRequest, AnalysisWorkflowPort
from mindflow.services.llm_service import LLMService
from mindflow.time_utils import business_today

router = APIRouter(tags=["analytics"])


class AttributionRequest(BaseModel):
    """Optional request body for POST /analytics/attribution."""

    date: str | None = Field(
        default=None, description="Date in YYYY-MM-DD format. Defaults to today."
    )
    force: bool = Field(default=False, description="Force re-analysis even if cached.")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str | None) -> str | None:
        """Reject malformed or impossible ISO/calendar dates at the boundary.

        Without this, an unparseable ``date`` reaches ``date.fromisoformat`` in
        the handler and escapes as HTTP 500 via the generic Exception handler.
        Raising ``ValueError`` here makes FastAPI surface the established RFC
        9457 ``validation-error`` (422) instead.
        """
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"无效的日期格式，应为 YYYY-MM-DD: {v!r}") from exc
        return v


@router.post("/analytics/attribution")
async def post_attribution(
    request: Request,
    payload: AttributionRequest | None = None,
    llm_service: LLMService = Depends(get_llm_service),  # noqa: B008
    workflow_port: AnalysisWorkflowPort | None = Depends(get_workflow_port),  # noqa: B008
) -> dict[str, Any]:
    """Run (or retrieve cached) procrastination attribution for a date.

    When the shared ``AnalysisWorkflowPort`` is available, delegates through
    it with ``origin="api"``.  Otherwise falls back to ``llm_service.analyze``.

    Args:
        payload: Optional JSON body. ``date`` defaults to today, ``force`` to False.
        llm_service: Injected LLM service instance (fallback path).
        workflow_port: Shared analysis workflow port (primary path).

    Returns:
        Assessment data with ``source``, ``cached``, and ``meta.degraded``.

    Raises:
        404 ProblemDetail: No activity events exist for the requested date.
    """
    req = AttributionRequest.model_validate(payload or {})
    settings = getattr(request.app.state, "settings", None)
    target_date = (
        date.fromisoformat(req.date)
        if req.date
        else business_today(getattr(settings, "timezone", "local"))
    )

    try:
        if workflow_port is not None:
            result = await workflow_port.run_analysis(
                AnalysisRequest(
                    user_id=1,
                    target_date=target_date,
                    analysis_kind="daily_attribution",
                    force=req.force,
                    origin="api",
                )
            )
            verdict = result.verdict
            types_list = [t.value for t in verdict.types]
            conf_map = {k.value: v for k, v in verdict.confidence.items()}
            result = {
                "assessment": {
                    "procrastination_types": types_list,
                    "type_confidence": conf_map,
                    "cbt_technique": (
                        verdict.recommended_technique.value
                        if verdict.recommended_technique is not None
                        else None
                    ),
                    "response_text": verdict.rationale,
                },
                "source": verdict.source,
                "cached": verdict.cached,
                "meta": {
                    "degraded": verdict.source != "panel",
                },
                # Frontend aliases (Analytics.tsx reads these)
                "confidence": max(conf_map.values()) if conf_map else None,
                "procrastination_type": types_list[0] if types_list else None,
                "cbt_technique": (
                    verdict.recommended_technique.value
                    if verdict.recommended_technique is not None
                    else None
                ),
                "evidence": verdict.rationale,
            }
            return result
        outcome = await llm_service.analyze(
            user_id=1,
            target_date=target_date,
            force=req.force,
        )
    except (ProblemDetail, NoActivityDataError):
        # Let RFC 9457 errors and the no-activity domain error propagate to
        # their registered handlers (NoActivityDataError → 404). Only truly
        # unexpected failures collapse to a generic 500 below.
        raise
    except Exception:
        from mindflow.api.errors import _internal_error

        raise _internal_error() from None

    assessment = outcome.assessment or {}
    types_list = assessment.get("procrastination_types", [])
    conf_map = assessment.get("type_confidence", {})
    return {
        "assessment": assessment,
        "source": outcome.source,
        "cached": outcome.cached,
        "meta": {
            "degraded": outcome.degraded,
        },
        # Frontend aliases
        "confidence": max(conf_map.values()) if conf_map else None,
        "procrastination_type": types_list[0] if types_list else None,
        "cbt_technique": assessment.get("cbt_technique"),
        "evidence": assessment.get("response_text"),
    }
