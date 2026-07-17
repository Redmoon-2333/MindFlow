"""API routes for focus session data.

Endpoints:
  - GET /focus (today's sessions and report)
  - GET /focus/trend (session trend over N days)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query  # noqa: B008
from loguru import logger

from mindflow.api.deps import (
    get_analysis_service,
    get_focus_repo,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.services.analysis_service import AnalysisService

router = APIRouter(tags=["focus"])


@router.get("/focus")
async def get_today_focus(
    date_param: date | None = Query(  # noqa: B008
        None, alias="date", description="Target date (YYYY-MM-DD, default today)"
    ),
    analysis: AnalysisService = Depends(get_analysis_service),  # noqa: B008
    focus_repo: SQLAlchemyFocusSessionRepository = Depends(get_focus_repo),  # noqa: B008
) -> dict[str, Any]:
    """Return today's focus sessions (auto-generates if missing)."""
    target = date_param or date.today()

    # Ensure sessions exist
    sessions = await focus_repo.get_by_date(1, target)
    if not sessions:
        logger.debug("No sessions for {}, running identification", target)
        sessions = await analysis.identify_focus_sessions(1, target)

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
            }
            for s in sessions
        ],
        "session_count": len(sessions),
    }


@router.get("/focus/trend")
async def get_focus_trend(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to look back"),
    focus_repo: SQLAlchemyFocusSessionRepository = Depends(get_focus_repo),  # noqa: B008
) -> dict[str, Any]:
    """Return focus session trends over the last *days* days."""
    today = date.today()
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
