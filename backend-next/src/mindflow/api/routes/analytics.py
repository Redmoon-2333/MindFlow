"""API routes for analytics and behavioural insights.

Endpoints:
  - GET /analytics/patterns (distraction pattern analysis)
  - GET /analytics/baseline (current baseline summary, placeholder for Wave 6)
  - GET /analytics/profile (behavioural profile)
  - GET /analytics/model-status (ML model loading and readiness)
  - GET /analytics/training-readiness (training data availability + gate checks)
  - POST /analytics/training-jobs (start a new training job)
  - GET /analytics/training-jobs/{job_id} (job lifecycle status + report)
  - POST /analytics/training-jobs/{job_id}/cancel (cancel a pending/running job)
"""

from __future__ import annotations

from typing import Any
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION

from fastapi import APIRouter, Depends, Query, Request  # noqa: B008
from fastapi import status as http_status

from mindflow.api.deps import get_analysis_service, get_baseline_repo
from mindflow.api.errors import ProblemDetail, _not_found
from mindflow.api.schemas import (
    BaselineSummary,
    CreateTrainingJobResponse,
    TrainingJobResponse,
    TrainingReadinessResponse,
)
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.baseline import BaselineRepository
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.training_job_service import (
    CancelRejectedError,
    ConcurrencyError,
    TrainingJobService,
)
from mindflow.services.training_readiness_service import TrainingReadinessService

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


@router.get("/analytics/baseline", response_model=BaselineSummary)
async def get_baseline(
    baseline_repo: BaselineRepository = Depends(get_baseline_repo),  # noqa: B008
) -> BaselineSummary:
    """Return the current user's personal behavior baseline (V2 vocabulary).

    The typed response carries the canonical V2 means
    (``mean_app_switch_count`` / ``mean_active_seconds_ratio`` /
    ``mean_idle_ratio``) plus the compatibility aliases
    ``switch_frequency == mean_app_switch_count`` and
    ``productivity_ratio == mean_active_seconds_ratio``. ``features`` is the
    exact 24-name V2 vocabulary. When no baseline exists the repository is
    empty and this endpoint stays 404.
    """
    baseline = await baseline_repo.get_latest(user_id=1)

    if baseline is None:
        raise _not_found("基线模型（暂无训练数据）")

    mean_app_switch_count = baseline.overall_mean("app_switch_count")
    mean_active_seconds_ratio = baseline.overall_mean("active_seconds_ratio")
    return BaselineSummary(
        user_id=baseline.user_id,
        created_at=baseline.created_at.isoformat(),
        updated_at=baseline.updated_at.isoformat(),
        total_days=baseline.total_days,
        total_samples=baseline.total_samples(),
        features=baseline.FEATURE_COLS,
        mean_app_switch_count=mean_app_switch_count,
        mean_active_seconds_ratio=mean_active_seconds_ratio,
        mean_idle_ratio=baseline.overall_mean("idle_ratio"),
        switch_frequency=mean_app_switch_count,
        productivity_ratio=mean_active_seconds_ratio,
    )


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

    # Frontend aliases (Analytics.tsx reads these field names)
    if "peak_focus" not in profile:
        hours = profile.get("peak_focus_hours", [])
        if hours:
            top = hours[0]
            profile["peak_focus"] = "{}:00 (avg {:.0f})".format(top["hour"], top["avg_score"])
        else:
            profile["peak_focus"] = None
    if "productivity_apps" not in profile:
        profile["productivity_apps"] = [
            a["app"] for a in profile.get("top_apps", [])[:5]
        ]
    if "trigger_apps" not in profile:
        profile["trigger_apps"] = [
            t["app"] for t in profile.get("distraction_triggers", [])[:5]
        ]
    if "details" not in profile:
        profile["details"] = {
            "avg_focus_block_min": profile.get("avg_focus_block_min"),
            "total_events": profile.get("total_events_analysed"),
            "distraction_ratio": round(
                sum(t["count"] for t in profile.get("distraction_triggers", []))
                / max(profile.get("total_events_analysed", 1), 1),
                3,
            ),
        }
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
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
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


@router.get(
    "/analytics/training-readiness",
    response_model=TrainingReadinessResponse,
)
async def get_training_readiness(
    request: Request,
) -> TrainingReadinessResponse:
    """Assess whether enough data exists to train a V2 feature-schema model.

    Matches feature windows to explicit feedback via time overlap using
    the same semantics as train/v2.py:prepare_v2_training_data. Reports
    raw activity events, V2 windows (including matched eligibility),
    feedback label distribution, trainability, evaluability, baseline
    readiness, the seven V2 gate checks, blocker codes, and the
    active/latest training job status.
    """
    telemetry_repo: TelemetryRepository | None = getattr(
        request.app.state, "telemetry_repository", None,
    )
    focus_repo: SQLAlchemyFocusSessionRepository | None = getattr(
        request.app.state, "focus_repository", None,
    )
    activity_repo: SQLAlchemyActivityRepository | None = getattr(
        request.app.state, "activity_repository", None,
    )
    baseline_repo: BaselineRepository | None = getattr(
        request.app.state, "baseline_repository", None,
    )
    v2_training_mode: str = getattr(
        request.app.state, "v2_training_mode", "rule_engine_only",
    )
    job_service: TrainingJobService | None = getattr(
        request.app.state, "training_job_service", None,
    )

    if None in (telemetry_repo, focus_repo, activity_repo, baseline_repo):
        raise _not_found("训练就绪评估服务（repository 未初始化）")

    assert telemetry_repo is not None
    assert focus_repo is not None
    assert activity_repo is not None
    assert baseline_repo is not None

    service = TrainingReadinessService(
        telemetry_repo=telemetry_repo,
        focus_repo=focus_repo,
        activity_repo=activity_repo,
        baseline_repo=baseline_repo,
        v2_training_mode=v2_training_mode,
    )
    result = await service.compute()

    # Inject current job status from the training job service.
    if job_service is not None:
        result.current_training_job = job_service.current_job

    return result


# ── Training job endpoints ──────────────────────────────────────────────────


@router.post(
    "/analytics/training-jobs",
    response_model=CreateTrainingJobResponse,
    status_code=http_status.HTTP_202_ACCEPTED,
)
async def create_training_job(
    request: Request,
) -> CreateTrainingJobResponse:
    """Start a V2 model training job.

    Returns 202 with job id when the readiness check passes (trainable=True).
    Returns 409 if another job is already active.
    Returns 412 if training data is insufficient.
    """
    job_service: TrainingJobService | None = getattr(
        request.app.state, "training_job_service", None,
    )
    if job_service is None:
        raise _not_found("训练任务服务（未初始化）")

    # ── Readiness gate ─────────────────────────────────────────────────
    # Reuse the same data-loading seam as the readiness endpoint.
    telemetry_repo: TelemetryRepository | None = getattr(
        request.app.state, "telemetry_repository", None,
    )
    focus_repo: SQLAlchemyFocusSessionRepository | None = getattr(
        request.app.state, "focus_repository", None,
    )
    activity_repo: SQLAlchemyActivityRepository | None = getattr(
        request.app.state, "activity_repository", None,
    )
    baseline_repo: BaselineRepository | None = getattr(
        request.app.state, "baseline_repository", None,
    )
    v2_training_mode: str = getattr(
        request.app.state, "v2_training_mode", "rule_engine_only",
    )

    if None in (telemetry_repo, focus_repo, activity_repo, baseline_repo):
        raise _not_found("训练任务服务（repository 未初始化）")

    assert telemetry_repo is not None
    assert focus_repo is not None
    assert activity_repo is not None
    assert baseline_repo is not None

    readiness = TrainingReadinessService(
        telemetry_repo=telemetry_repo,
        focus_repo=focus_repo,
        activity_repo=activity_repo,
        baseline_repo=baseline_repo,
        v2_training_mode=v2_training_mode,
    )
    assessment = await readiness.compute()

    if not assessment.trainable:
        raise ProblemDetail(
            type_slug="training-not-ready",
            title="Training Not Ready",
            status=412,
            detail="训练数据不足，无法启动训练任务",
            extra={
                "trainable": False,
                "blockers": [
                    {"code": b.code, "message": b.message}
                    for b in assessment.blockers
                ],
            },
        )

    try:
        response = await job_service.start_job(app_state=request.app.state)
    except ConcurrencyError as exc:
        raise ProblemDetail(
            type_slug="training-job-active",
            title="Training Job Already Active",
            status=409,
            detail=str(exc),
        ) from exc

    return CreateTrainingJobResponse(
        job_id=response.job_id,
        status="pending",
    )


@router.get(
    "/analytics/training-jobs/{job_id}",
    response_model=TrainingJobResponse,
)
async def get_training_job(
    job_id: str,
    request: Request,
) -> TrainingJobResponse:
    """Return the lifecycle status and report for a training job."""
    job_service: TrainingJobService | None = getattr(
        request.app.state, "training_job_service", None,
    )
    if job_service is None:
        raise _not_found("训练任务服务（未初始化）")

    job = job_service.get_job(job_id)
    if job is None:
        raise _not_found(f"训练任务（{job_id}）")

    return job


@router.post(
    "/analytics/training-jobs/{job_id}/cancel",
    response_model=TrainingJobResponse,
)
async def cancel_training_job(
    job_id: str,
    request: Request,
) -> TrainingJobResponse:
    """Cancel a pending or preparing training job.

    Returns the job status.  Once training has started, cancellation is
    rejected (409) because the thread may already write activated artifacts.
    """
    job_service: TrainingJobService | None = getattr(
        request.app.state, "training_job_service", None,
    )
    if job_service is None:
        raise _not_found("训练任务服务（未初始化）")

    try:
        result = await job_service.cancel_job(job_id)
    except CancelRejectedError as exc:
        raise ProblemDetail(
            type_slug="training-cancel-rejected",
            title="Cancel Rejected",
            status=409,
            detail=str(exc),
        ) from exc

    if result is None:
        raise _not_found(f"训练任务（{job_id}）")

    return result
