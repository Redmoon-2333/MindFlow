"""API routes for autonomous agent control (G005).

Endpoints:
  GET    /api/v1/autonomy      — Current autonomy status
  POST   /api/v1/autonomy/pause  — Pause for N hours
  POST   /api/v1/autonomy/resume — Resume immediately

All responses include ``{"enabled": bool, "paused_until": str|null}``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends  # noqa: B008
from loguru import logger
from pydantic import BaseModel, Field

from mindflow.api.deps import get_autonomy_service
from mindflow.services.autonomy_service import AutonomyService

router = APIRouter(tags=["autonomy"])


class PauseRequest(BaseModel):
    """Request body for POST /api/v1/autonomy/pause."""

    hours: float = Field(default=1.0, ge=0.5, description="Pause duration in hours (min 0.5)")


class AutonomyStatus(BaseModel):
    """Response model for autonomy status."""

    enabled: bool
    paused_until: str | None = None


@router.get("/autonomy")
async def get_autonomy_status(
    autonomy_service: AutonomyService = Depends(get_autonomy_service),  # noqa: B008
) -> AutonomyStatus:
    """Return the current autonomy agent status.

    Checks ``autonomy.enabled`` and ``autonomy.paused_until`` from
    user preferences.
    """
    status = await autonomy_service.get_status()
    logger.debug(
        "Autonomy status: enabled={}, paused_until={}",
        status["enabled"],
        status["paused_until"],
    )
    return AutonomyStatus(**status)  # type: ignore[arg-type]


@router.post("/autonomy/pause")
async def pause_autonomy(
    body: PauseRequest,
    autonomy_service: AutonomyService = Depends(get_autonomy_service),  # noqa: B008
) -> AutonomyStatus:
    """Pause the autonomous agent for *hours*.

    The pause is persisted in user preferences and checked by all
    autonomous agent entry points (scheduler, daily-panel job).
    """
    await autonomy_service.pause(hours=body.hours)
    logger.info("Autonomy paused for {} hours", body.hours)

    status = await autonomy_service.get_status()
    return AutonomyStatus(**status)  # type: ignore[arg-type]


@router.post("/autonomy/resume")
async def resume_autonomy(
    autonomy_service: AutonomyService = Depends(get_autonomy_service),  # noqa: B008
) -> AutonomyStatus:
    """Resume the autonomous agent immediately.

    Clears the ``paused_until`` timestamp.  Does not change the
    ``autonomy.enabled`` master switch.
    """
    await autonomy_service.resume()
    logger.info("Autonomy resumed by user")

    status = await autonomy_service.get_status()
    return AutonomyStatus(**status)  # type: ignore[arg-type]
