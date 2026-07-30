"""Liveness, readiness, and backward-compatible health endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncEngine

from mindflow import __version__
from mindflow.api.deps import (
    get_collector_service,
    get_engine,
    get_migration_status,
)
from mindflow.services.collector_service import CollectorService

router = APIRouter(tags=["health"])


async def _database_connected(engine: AsyncEngine | None) -> bool:
    if engine is None:
        return False
    try:
        async with engine.connect():
            return True
    except Exception:
        return False


def _base_payload() -> dict[str, Any]:
    return {
        "version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/health/live")
async def liveness_check() -> dict[str, Any]:
    """Report process liveness without touching startup dependencies."""
    return {"status": "alive", **_base_payload()}


@router.get("/health/ready")
async def readiness_check(
    request: Request,
    engine: AsyncEngine = Depends(get_engine),  # noqa: B008
    migration_status: bool = Depends(get_migration_status),  # noqa: B008
) -> JSONResponse:
    """Report readiness; failed migration, DB, or integrity checks return 503."""
    db_connected = await _database_connected(engine)
    integrity_ok = bool(getattr(request.app.state, "db_integrity_ok", False))

    # Checkpoint store availability (diagnostics / workflow audit).
    checkpointer = getattr(request.app.state, "checkpointer", None)
    checkpoint_available = checkpointer is not None and getattr(
        checkpointer, "_closed", True
    ) is False

    # Run store availability — verifies the workflow_runs table is queryable.
    run_store_available = False
    if db_connected:
        try:
            from mindflow.infrastructure.repositories.workflow_runs import (
                WorkflowRunsRepository,
            )

            repo = WorkflowRunsRepository(
                session_factory=request.app.state.session_factory
            )
            # Lightweight probe: list with limit=0 to check table existence
            await repo.list_runs(limit=0, offset=0)
            run_store_available = True
        except Exception:
            run_store_available = False

    ready = migration_status and db_connected and integrity_ok
    payload = {
        "status": "ready" if ready else "not_ready",
        **_base_payload(),
        "database": {
            "status": "ok" if db_connected else "error",
            "connected": db_connected,
            "integrity_ok": integrity_ok,
        },
        "migration": {"applied": migration_status},
        "checkpoint_store": "available" if checkpoint_available else "unavailable",
        "run_store": "available" if run_store_available else "unavailable",
    }
    return JSONResponse(content=payload, status_code=200 if ready else 503)


@router.get("/health")
async def health_check(
    request: Request,
    collector_service: CollectorService | None = Depends(get_collector_service),  # noqa: B008
    engine: AsyncEngine = Depends(get_engine),  # noqa: B008
    migration_status: bool = Depends(get_migration_status),  # noqa: B008
) -> dict[str, Any]:
    """Return the legacy component payload and always keep HTTP 200 compatibility."""
    db_connected = await _database_connected(engine)
    integrity_ok = bool(getattr(request.app.state, "db_integrity_ok", False))

    # ML status from PredictionService if available
    ml_status: dict[str, Any] | None = None
    prediction_service = getattr(request.app.state, "prediction_service", None)
    if prediction_service is not None:
        try:
            ml_status = await prediction_service.check_health()
        except Exception:
            ml_status = {"status": "error", "model_version": None, "feature_schema_version": 2}
    else:
        ml_status = {
            "status": "no_service",
            "v2_training_mode": getattr(request.app.state, "v2_training_mode", None),
        }

    # Checkpoint / run store availability (same probes as readiness).
    checkpointer = getattr(request.app.state, "checkpointer", None)
    checkpoint_available = checkpointer is not None and getattr(
        checkpointer, "_closed", True
    ) is False

    run_store_available = False
    if db_connected:
        try:
            from mindflow.infrastructure.repositories.workflow_runs import (
                WorkflowRunsRepository,
            )

            repo = WorkflowRunsRepository(
                session_factory=request.app.state.session_factory
            )
            await repo.list_runs(limit=0, offset=0)
            run_store_available = True
        except Exception:
            run_store_available = False

    return {
        "status": "ok",
        **_base_payload(),
        "collector": {
            "status": collector_service.status if collector_service else "unavailable",
        },
        "database": {
            "status": "ok" if db_connected else "error",
            "connected": db_connected,
            "integrity_ok": integrity_ok,
        },
        "migration": {"applied": migration_status},
        "ml": ml_status,
        "checkpoint_store": "available" if checkpoint_available else "unavailable",
        "run_store": "available" if run_store_available else "unavailable",
    }
