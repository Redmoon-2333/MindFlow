"""Telemetry API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from mindflow.api.deps import (
    get_blocklist_repo,
    get_feedback_request_store,
    get_telemetry_service,
)
from mindflow.api.errors import ProblemDetail
from mindflow.api.schemas import (
    FeedbackAnswerRequest,
    FeedbackAnswerResponse,
    FeedbackRequestModel,
    FocusPredictionResponse,
    PendingFeedbackRequestsResponse,
    TelemetryDeleteResponse,
)
from mindflow.infrastructure.repositories.blocklist import SQLAlchemyBlocklistRepository
from mindflow.services.feedback_request_service import FeedbackRequestStore
from mindflow.services.telemetry_service import TelemetryClearResult, TelemetryService

router = APIRouter(tags=["telemetry"])


class TelemetryPreferencesPatch(BaseModel):
    input_telemetry_enabled: bool | None = None
    browser_tracking_enabled: bool | None = None
    interaction_retention_days: int | None = Field(default=None, ge=1, le=30)
    activity_retention_days: int | None = Field(default=None, ge=7, le=90)


class BrowserPairRequest(BaseModel):
    # 8-char unambiguous alphabet codes (see TelemetryService.create_pairing_code);
    # keep a small tolerance for legacy 6-digit codes during migration.
    code: str = Field(min_length=6, max_length=8)


class BrowserHeartbeatRequest(BaseModel):
    timestamp_utc: datetime
    duration_s: float = Field(ge=1, le=60)
    browser_name: str = Field(min_length=1, max_length=32)
    domain: str = Field(min_length=1, max_length=2048)
    audible: bool = False
    incognito: bool = False


@router.get("/telemetry/status")
async def get_telemetry_status(
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, Any]:
    return await service.get_status()


@router.get("/telemetry/focus-prediction", response_model=FocusPredictionResponse)
async def get_focus_prediction(
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, Any]:
    return await service.predict_latest_focus()


@router.patch("/telemetry/preferences")
async def patch_telemetry_preferences(
    body: TelemetryPreferencesPatch,
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, Any]:
    return await service.patch_preferences(body.model_dump(exclude_none=True))


@router.delete(
    "/telemetry/data",
    response_model=TelemetryDeleteResponse,
    response_model_exclude_defaults=True,
)
async def delete_telemetry_data(
    scope: Literal["interaction", "browser", "feedback", "all"],
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> TelemetryDeleteResponse:
    result = await service.clear_data(scope)
    return TelemetryDeleteResponse(
        deleted=int(result),
        partial=isinstance(result, TelemetryClearResult) and result.partial,
        failures=(list(result.failures) if isinstance(result, TelemetryClearResult) else []),
    )


@router.post("/telemetry/browser/pairing-code")
async def create_browser_pairing_code(
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, Any]:
    return await service.create_pairing_code()


@router.post("/telemetry/browser/pair")
async def pair_browser(
    body: BrowserPairRequest,
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, str]:
    token = await service.pair_browser(body.code)
    if token is None:
        raise ProblemDetail(
            type_slug="invalid-pairing-code",
            title="Invalid Pairing Code",
            status=401,
            detail="配对码无效或已过期",
        )
    return {"token": token}


@router.post("/telemetry/browser/heartbeat")
async def browser_heartbeat(
    body: BrowserHeartbeatRequest,
    x_browser_token: str = Header(default=""),
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, Any]:
    result = await service.save_authenticated_browser_heartbeat(
        x_browser_token,
        **body.model_dump(),
    )
    if result is None:
        raise ProblemDetail(
            type_slug="browser-auth-required",
            title="Browser Authentication Required",
            status=401,
            detail="浏览器令牌无效或已撤销",
        )
    return result


@router.get("/telemetry/browser/blocklist")
async def browser_blocklist(
    x_browser_token: str = Header(default=""),
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
    repo: SQLAlchemyBlocklistRepository = Depends(get_blocklist_repo),  # noqa: B008
) -> dict[str, Any]:
    """Return the enabled blocklist for the paired browser extension.

    Authenticated by the browser pairing token (same credential as the
    heartbeat endpoint), then mapped to declarativeNetRequest dynamic
    rules by the extension's service worker.
    """
    if not await service.verify_browser_token(x_browser_token):
        raise ProblemDetail(
            type_slug="browser-auth-required",
            title="Browser Authentication Required",
            status=401,
            detail="浏览器令牌无效或已撤销",
        )
    domains = await repo.list_enabled(user_id=1)
    return {"domains": domains, "version": 1}


# ── Proactive-learning feedback requests (contract §5) ───────────────────


@router.get(
    "/telemetry/feedback-requests/pending",
    response_model=PendingFeedbackRequestsResponse,
)
async def get_pending_feedback_requests(
    store: FeedbackRequestStore = Depends(get_feedback_request_store),  # noqa: B008
) -> PendingFeedbackRequestsResponse:
    """The outstanding prompt, for clients that missed the WS push.

    At most one item: the store refuses to stack prompts, and an expired one
    is never returned — the user can no longer answer it.
    """
    items = [
        FeedbackRequestModel(**request.to_payload())
        for request in await store.pending()
    ]
    return PendingFeedbackRequestsResponse(items=items)


@router.post(
    "/telemetry/feedback-requests/{request_id}/answer",
    response_model=FeedbackAnswerResponse,
)
async def answer_feedback_request(
    request_id: str,
    body: FeedbackAnswerRequest,
    store: FeedbackRequestStore = Depends(get_feedback_request_store),  # noqa: B008
) -> FeedbackAnswerResponse:
    """Record the user's label for the window the request covered.

    ``skip`` records nothing and answers ``status="skipped"``; a request older
    than the TTL answers ``status="expired"`` rather than labelling behaviour
    the user can no longer remember. An unknown id is 404 — the client is
    polling a request that was already answered or never existed.
    """
    result = await store.answer(request_id, body.label)
    if result is None:
        raise ProblemDetail(
            type_slug="feedback-request-not-found",
            title="Feedback Request Not Found",
            status=404,
            detail="反馈请求不存在、已作答或已过期",
        )
    return FeedbackAnswerResponse(
        request_id=result.request_id,
        labeled_windows=result.labeled_windows,
        status=result.status,
    )
