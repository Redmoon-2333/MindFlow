"""API routes for focus session data.

Endpoints:
  - GET /focus (today's sessions and report)
  - GET /focus/trend (session trend over N days)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request  # noqa: B008
from loguru import logger
from pydantic import BaseModel, Field

from mindflow.api.deps import (
    get_analysis_service,
    get_focus_repo,
    get_telemetry_service,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.telemetry_service import TelemetryService
from mindflow.time_utils import business_today

router = APIRouter(tags=["focus"])


@router.get("/focus")
async def get_today_focus(
    request: Request,
    date_param: date | None = Query(  # noqa: B008
        None, alias="date", description="Target date (YYYY-MM-DD, default today)"
    ),
    analysis: AnalysisService = Depends(get_analysis_service),  # noqa: B008
    focus_repo: SQLAlchemyFocusSessionRepository = Depends(get_focus_repo),  # noqa: B008
    telemetry_service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, Any]:
    """Return today's focus sessions (auto-generates if missing)."""
    settings = getattr(request.app.state, "settings", None)
    target = date_param or business_today(getattr(settings, "timezone", "local"))

    # Ensure sessions exist
    sessions = await focus_repo.get_by_date(1, target)
    if not sessions:
        logger.debug("No sessions for {}, running identification", target)
        sessions = await analysis.identify_focus_sessions(1, target)

    # Fetch feedback for all sessions in one query
    session_ids = [s["id"] for s in sessions]
    feedback_map = await telemetry_service.get_feedback_for_sessions(session_ids)

    return {
        "date": target.isoformat(),
        "sessions": [
            {
                "id": s["id"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "session_type": s["session_type"],
                "dominant_app": s["dominant_app"],
                "focus_score": s["focus_score"],
                "switch_count": s["switch_count"],
                **(feedback_map.get(s["id"]) or {}),
            }
            for s in sessions
        ],
        "session_count": len(sessions),
    }


@router.get("/focus/trend")
async def get_focus_trend(
    request: Request,
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
    focus_repo: SQLAlchemyFocusSessionRepository = Depends(get_focus_repo),  # noqa: B008
) -> dict[str, Any]:
    """Return focus session trends over the last *days* days."""
    settings = getattr(request.app.state, "settings", None)
    today = business_today(getattr(settings, "timezone", "local"))
    start = today - timedelta(days=days - 1)

    sessions = await focus_repo.query_range(1, start, today)

    # Group by date
    by_date: dict[str, dict[str, Any]] = {}
    for s in sessions:
        d = s["date"]
        if d not in by_date:
            by_date[d] = {
                "date": d,
                "focus_min": 0.0,
                "distraction_min": 0.0,
                "session_count": 0,
                "avg_score": 0.0,
            }
        try:
            start_ts = datetime.fromisoformat(s["start_time"])
            end_ts = datetime.fromisoformat(s["end_time"])
            duration_min = (end_ts - start_ts).total_seconds() / 60.0
        except (ValueError, KeyError):
            duration_min = 0.0

        by_date[d]["session_count"] += 1
        if s.get("session_type") == "focus":
            by_date[d]["focus_min"] += duration_min
        elif s.get("session_type") == "distraction":
            by_date[d]["distraction_min"] += duration_min

    daily_trend = sorted(by_date.values(), key=lambda x: x["date"])

    return {
        "days": days,
        "start_date": start.isoformat(),
        "end_date": today.isoformat(),
        "daily": daily_trend,
        "total_sessions": len(sessions),
    }


class FocusFeedbackBody(BaseModel):
    label: Literal["focus", "distracted", "mixed"]
    score: int = Field(ge=1, le=5)
    task_type: str | None = Field(default=None, max_length=64)


@router.post("/focus/{session_id}/feedback")
async def save_focus_feedback(
    session_id: str,
    body: FocusFeedbackBody,
    telemetry_service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, Any]:
    return await telemetry_service.save_focus_feedback(
        session_id=session_id,
        label=body.label,
        score=body.score,
        task_type=body.task_type,
    )
