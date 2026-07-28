"""API routes for analytics and behavioural insights.

Endpoints:
  - GET /analytics/patterns (distraction pattern analysis)
  - GET /analytics/baseline (current baseline summary, placeholder for Wave 6)
  - GET /analytics/profile (behavioural profile)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request  # noqa: B008

from mindflow.api.deps import get_analysis_service, get_baseline_repo
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
    request: Request,
) -> dict[str, Any]:
    """Return ML model loading status and version information.

    Reports whether V2 feature-schema models (classifier, clustering, HMM)
    are loaded and available for runtime inference.
    """
    v2_model_manager = getattr(request.app.state, "v2_model_manager", None)
    v2_training_mode = getattr(
        request.app.state,
        "v2_training_mode",
        "rule_engine_only",
    )
    if v2_model_manager is not None:
        readiness = v2_model_manager.readiness_status()
        is_ready = bool(readiness["ready"])
        return {
            "loaded": True,
            "ready": is_ready,
            "mode": "ready" if is_ready else "rule_engine_only",
            "feature_schema_version": 2,
            "v2_mode": "ready" if is_ready else v2_training_mode,
            "version": v2_model_manager.current_version_tag,
            "available_versions": v2_model_manager.list_versions(),
            "reasons": readiness["reasons"],
            "message": (
                "Feature schema v2 model loaded and ready for inference"
                if is_ready
                else "Feature schema v2 artifacts failed readiness checks"
            ),
        }

    # V1 model path removed — all ML now goes through V2
    return {
        "loaded": False,
        "ready": False,
        "mode": v2_training_mode,
        "v2_mode": v2_training_mode,
        "reasons": ["v2_models_not_loaded"],
        "message": "V2 ML models not available, running with rule engine only",
    }
