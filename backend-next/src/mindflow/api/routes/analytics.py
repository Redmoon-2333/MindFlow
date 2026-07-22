"""API routes for analytics and behavioural insights.

Endpoints:
  - GET /analytics/patterns (distraction pattern analysis)
  - GET /analytics/baseline (current baseline summary, placeholder for Wave 6)
  - GET /analytics/profile (behavioural profile)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query  # noqa: B008

from mindflow.api.deps import get_analysis_service, get_baseline_repo, get_model_manager
from mindflow.api.errors import _not_found
from mindflow.infrastructure.repositories.baseline import BaselineRepository
from mindflow.services.analysis_service import AnalysisService
from mindflow.train.models.manager import ModelManager

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
async def get_baseline(
    baseline_repo: BaselineRepository = Depends(get_baseline_repo),  # noqa: B008
) -> dict[str, Any]:
    """Return the current user's personal behavior baseline."""
    baseline = await baseline_repo.get_latest(user_id=1)

    if baseline is None:
        raise _not_found("基线模型（暂无训练数据）")

    return {
        "user_id": baseline.user_id,
        "created_at": baseline.created_at.isoformat(),
        "updated_at": baseline.updated_at.isoformat(),
        "total_days": baseline.total_days,
        "total_samples": baseline.total_samples(),
        "features": baseline.FEATURE_COLS,
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


@router.get("/analytics/model-status")
async def get_model_status(
    model_manager: ModelManager | None = Depends(get_model_manager),  # noqa: B008
) -> dict[str, Any]:
    """Return ML model loading status and version information.

    Reports whether scikit-learn models (classifier, clustering, HMM) are
    loaded and available for runtime inference.  When models are loaded,
    includes the version tag and available versions for rollback.
    """
    if model_manager is None:
        return {
            "loaded": False,
            "mode": "rule_engine_only",
            "message": "ML models not available, running with rule engine only",
        }

    return {
        "loaded": True,
        "mode": "ml_enriched",
        "version": model_manager.current_version_tag,
        "available_versions": model_manager.list_versions(),
        "message": "ML models loaded, behaviour analysis enriched with predictions",
    }
