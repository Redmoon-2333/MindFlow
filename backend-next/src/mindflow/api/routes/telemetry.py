"""Telemetry API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from mindflow.api.deps import get_telemetry_service
from mindflow.api.errors import ProblemDetail
from mindflow.api.schemas import FocusPredictionResponse
from mindflow.services.telemetry_service import TelemetryService

router = APIRouter(tags=["telemetry"])


class TelemetryPreferencesPatch(BaseModel):
    input_telemetry_enabled: bool | None = None
    browser_tracking_enabled: bool | None = None
    interaction_retention_days: int | None = Field(default=None, ge=1, le=30)
    activity_retention_days: int | None = Field(default=None, ge=7, le=90)


class BrowserPairRequest(BaseModel):
    code: str = Field(min_length=6, max_length=6)


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


@router.delete("/telemetry/data")
async def delete_telemetry_data(
    scope: Literal["interaction", "browser", "feedback", "all"],
    service: TelemetryService = Depends(get_telemetry_service),  # noqa: B008
) -> dict[str, int]:
    return {"deleted": await service.clear_data(scope)}


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
