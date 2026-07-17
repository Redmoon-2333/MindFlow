"""Health check endpoint — /api/v1/health.

Returns the overall health status of the application, including:
  - Collector status (running, stopped, degraded)
  - Database connectivity (ok, error)
  - Migration status
  - Application version

This endpoint is exempt from authentication (§4.6 of requirements).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncEngine

from mindflow import __version__
from mindflow.api.deps import (
    get_collector_service,
    get_engine,
    get_migration_status,
)
from mindflow.services.collector_service import CollectorService

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check(
    collector_service: CollectorService | None = Depends(get_collector_service),  # noqa: B008
    engine: AsyncEngine = Depends(get_engine),  # noqa: B008
    migration_status: bool = Depends(get_migration_status),  # noqa: B008
) -> dict[str, Any]:
    """Return application health information.

    Returns collector status, database health, migration status, and version.
    Always succeeds — status codes in the body indicate component health.
    """
    db_ok = False
    try:
        async with engine.connect():
            db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "collector": {
            "status": collector_service.status if collector_service else "unavailable",
        },
        "database": {
            "status": "ok" if db_ok else "error",
            "connected": db_ok,
        },
        "migration": {
            "applied": migration_status,
        },
    }
