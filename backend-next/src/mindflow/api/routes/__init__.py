"""Route definitions for MindFlow API v1.

Exposes a ``register_routes(app)`` function that mounts all endpoint groups.
"""

from __future__ import annotations

from fastapi import FastAPI

from mindflow.api.routes.activities import router as activities_router
from mindflow.api.routes.analytics import router as analytics_router
from mindflow.api.routes.attribution import router as attribution_router
from mindflow.api.routes.collector import router as collector_router
from mindflow.api.routes.export import router as export_router
from mindflow.api.routes.focus import router as focus_router
from mindflow.api.routes.health import router as health_router
from mindflow.api.routes.intervention import router as intervention_router
from mindflow.api.routes.panel import router as panel_router
from mindflow.api.routes.preferences import router as preferences_router
from mindflow.api.routes.reports import router as reports_router


def register_routes(app: FastAPI) -> None:
    """Mount all API route groups on the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(collector_router, prefix="/api/v1")
    app.include_router(activities_router, prefix="/api/v1")
    app.include_router(export_router, prefix="/api/v1")
    app.include_router(preferences_router, prefix="/api/v1")
    app.include_router(focus_router, prefix="/api/v1")
    app.include_router(reports_router, prefix="/api/v1")
    app.include_router(analytics_router, prefix="/api/v1")
    app.include_router(attribution_router, prefix="/api/v1")
    app.include_router(intervention_router, prefix="/api/v1")
    app.include_router(panel_router, prefix="/api/v1")
