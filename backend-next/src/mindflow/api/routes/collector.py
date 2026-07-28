"""Collector management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from mindflow.api.deps import get_collector_service
from mindflow.api.errors import ProblemDetail
from mindflow.api.schemas import CollectorStatusResponse
from mindflow.services.collector_service import CollectorService

router = APIRouter(tags=["collector"])


def _unavailable(detail: str) -> ProblemDetail:
    return ProblemDetail(
        type_slug="collector-not-running",
        title="Collector Not Running",
        status=503,
        detail=detail,
    )


@router.get("/collector", response_model=CollectorStatusResponse)
async def get_collector_status(
    collector_service: CollectorService | None = Depends(get_collector_service),  # noqa: B008
) -> CollectorStatusResponse:
    if collector_service is None:
        raise _unavailable("数据采集器未运行，请先启动采集器")
    status = collector_service.status
    return CollectorStatusResponse(status=status, running=status == "running")


@router.post("/collector", response_model=CollectorStatusResponse)
async def start_collector(
    collector_service: CollectorService | None = Depends(get_collector_service),  # noqa: B008
) -> CollectorStatusResponse:
    if collector_service is None:
        raise _unavailable("运行状态不可用：采集器服务未初始化")
    if collector_service.status == "running":
        return CollectorStatusResponse(
            status="running", running=True, message="采集器已在运行中"
        )
    await collector_service.start()
    return CollectorStatusResponse(status="running", running=True, message="采集器已启动")


@router.post("/collector/stop", response_model=CollectorStatusResponse)
async def stop_collector(
    collector_service: CollectorService | None = Depends(get_collector_service),  # noqa: B008
) -> CollectorStatusResponse:
    if collector_service is None:
        raise _unavailable("运行状态不可用：采集器服务未初始化")
    await collector_service.stop()
    return CollectorStatusResponse(status="stopped", running=False, message="采集器已停止")
