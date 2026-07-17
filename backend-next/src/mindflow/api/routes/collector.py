"""Collector management endpoints — /api/v1/collector.

Provides:
  - GET /collector: Current collector status
  - POST /collector: Start the collector
  - POST /collector/stop: Stop the collector

Each endpoint checks the collector service state and returns appropriate
responses. If the collector is not available (e.g. no platform support),
endpoints return a 503 error.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from mindflow.api.deps import get_collector_service
from mindflow.api.errors import ProblemDetail
from mindflow.services.collector_service import CollectorService

router = APIRouter(tags=["collector"])


@router.get("/collector")
async def get_collector_status(
    collector_service: CollectorService | None = Depends(get_collector_service),  # noqa: B008
) -> dict[str, Any]:
    """Return the current collector status.

    Returns ``running``, ``stopped``, or ``degraded`` status.
    If the collector service is not initialized, returns 503.
    """
    if collector_service is None:
        raise ProblemDetail(
            type_slug="collector-not-running",
            title="Collector Not Running",
            status=503,
            detail="数据采集器未运行，请先启动采集器",
        )

    return {"status": collector_service.status}


@router.post("/collector")
async def start_collector(
    collector_service: CollectorService | None = Depends(get_collector_service),  # noqa: B008
) -> dict[str, Any]:
    """Start the data collector."""
    if collector_service is None:
        raise ProblemDetail(
            type_slug="collector-not-running",
            title="Collector Not Running",
            status=503,
            detail="运行状态不可用：采集器服务未初始化",
        )

    if collector_service.status == "running":
        return {"status": "running", "message": "采集器已在运行中"}

    await collector_service.start()
    return {"status": "running", "message": "采集器已启动"}


@router.post("/collector/stop")
async def stop_collector(
    collector_service: CollectorService | None = Depends(get_collector_service),  # noqa: B008
) -> dict[str, Any]:
    """Stop the data collector gracefully."""
    if collector_service is None:
        raise ProblemDetail(
            type_slug="collector-not-running",
            title="Collector Not Running",
            status=503,
            detail="运行状态不可用：采集器服务未初始化",
        )

    await collector_service.stop()
    return {"status": "stopped", "message": "采集器已停止"}
