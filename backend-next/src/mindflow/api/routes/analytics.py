"""API routes for analytics and behavioural insights.

Endpoints:
  - GET /analytics/patterns (distraction pattern analysis)
  - GET /analytics/baseline (current baseline summary, placeholder for Wave 6)
  - GET /analytics/profile (behavioural profile)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query  # noqa: B008

from mindflow.api.deps import get_analysis_service
from mindflow.api.errors import _not_found
from mindflow.services.analysis_service import AnalysisService

router = APIRouter(tags=["analytics"])


@router.get("/analytics/patterns")
async def get_patterns(
    days: int = Query(default=14, ge=1, le=90, description="Analysis window in days"),
    analysis: AnalysisService = Depends(get_analysis_service),  # noqa: B008
) -> dict[str, Any]:
    """Return distraction pattern analysis (high-switch periods, trigger apps, heatmap).

    Analyses the last *days* days of focus sessions.
    """
    patterns = await analysis.detect_patterns(1, days=days)

    if patterns["total_sessions"] == 0:
        raise _not_found("分析数据（暂无专注会话）")

    return patterns


@router.get("/analytics/baseline")
async def get_baseline() -> dict[str, Any]:
    """Return baseline information (placeholder, Wave 6)."""
    # Baseline Analysis will be powered by domain/baseline.py in Wave 6.
    # For now, return a stub that describes when baseline is ready.
    return {
        "status": "pending",
        "message": "基线模型将在收集足够数据后自动建立（最近约50个事件）",
        "note": "Wave 6 will integrate BaselineModel from domain/baseline.py",
    }


@router.get("/analytics/profile")
async def get_profile(
    days: int = Query(default=30, ge=1, le=365, description="Profile window in days"),
    analysis: AnalysisService = Depends(get_analysis_service),  # noqa: B008
) -> dict[str, Any]:
    """Return a behavioural profile for the current user.

    Combines event-stream analysis with focus session data to compute:
      - Peak focus hours
      - Top productive applications
      - Average focus block length
      - Distraction trigger applications
    """
    profile = await analysis.behavioral_profile(1, days=days)

    if profile["total_events_analysed"] == 0:
        raise _not_found("行为画像数据（暂无活动事件）")

    return profile
