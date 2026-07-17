"""FastAPI application factory — ``create_app(settings) -> FastAPI``.

Wires together:
  - Lifespan: migration → integrity check → token loading → CollectorService
  - Middleware: logging → host → auth → rate-limit (per §3.5 order)
  - Routes: health, collector, activities, preferences
  - WebSocket: /api/v1/ws
  - Exception handlers: RFC 9457 ProblemDetail (8 error codes)
  - Security headers: X-MindFlow-Version, X-Content-Type-Options

No global singletons — all shared state lives on ``app.state``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import platformdirs
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from mindflow import __version__
from mindflow.api.errors import register_exception_handlers
from mindflow.api.middleware import (
    AuthMiddleware,
    HostValidationMiddleware,
    RateLimitMiddleware,
    StructuredLoggingMiddleware,
)
from mindflow.api.routes import register_routes
from mindflow.api.websocket import router as websocket_router
from mindflow.config import Settings
from mindflow.infrastructure.collectors.base import EventCollector, create_collector
from mindflow.infrastructure.database import (
    create_engine,
    create_session_factory,
    integrity_check,
)
from mindflow.infrastructure.migrations import run_migrations
from mindflow.infrastructure.notification import create_notifier
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.preferences import (
    PreferencesRepository,
)
from mindflow.infrastructure.security.token_manager import load_or_create_token
from mindflow.logging_config import setup_logging
from mindflow.services.collector_service import CollectorService

# ── Lifespan ────────────────────────────────────────────────────────────────


async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup initialisation, shutdown cleanup.

    Startup sequence:
      1. Run Alembic migrations (graceful on failure)
      2. Integrity check (attempt VACUUM recovery on failure)
      3. Load/create auth token
      4. Create CollectorService (not started yet — caller must start)
      5. Inject engine, session factory, repos into app.state

    Shutdown sequence (reverse order):
      1. Stop collector
      2. Flush remaining events
      3. Dispose engine
    """
    # ── Extract settings ─────────────────────────────────────────────
    settings: Settings = app.state.settings
    data_dir = Path(platformdirs.user_data_dir("mindflow", ensure_exists=True))
    token_path = data_dir / "token"

    # ── Database engine ───────────────────────────────────────────────
    engine = create_engine(settings.db_url)
    session_factory = create_session_factory(engine)

    # ── 1. Migrations ─────────────────────────────────────────────────
    migration_applied = await run_migrations(settings.db_url)
    if not migration_applied:
        logger.warning(
            "Database migration failed — running with existing schema "
            "(health endpoint will report migration_failed)"
        )

    # ── 2. Integrity check ────────────────────────────────────────────
    db_ok = await integrity_check(engine)
    if not db_ok:
        logger.critical("Database integrity check failed after recovery attempt")
    else:
        logger.info("Database integrity check passed")

    # ── 3. Auth token ─────────────────────────────────────────────────
    system_token = load_or_create_token(token_path)
    logger.debug("Auth token loaded from {}", token_path)

    # ── 4. Repositories ───────────────────────────────────────────────
    activity_repository = SQLAlchemyActivityRepository(
        session_factory=session_factory,
        pulsetime_s=settings.heartbeat_pulsetime_s,
    )
    preferences_repository = PreferencesRepository(
        session_factory=session_factory,
    )

    # ── 5. Collector ──────────────────────────────────────────────────
    collector: EventCollector | None = None
    collector_service: CollectorService | None = None
    try:
        collector = create_collector()
        collector_service = CollectorService(
            collector=collector,
            repository=activity_repository,
            interval_s=float(settings.collect_interval_s),
        )
        logger.info("CollectorService created (not started)")
    except Exception as exc:
        logger.warning("Failed to create collector: {}", exc)

    # ── 6. Notifier ───────────────────────────────────────────────────
    notifier = create_notifier()

    # ── Inject into app.state ─────────────────────────────────────────
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.activity_repository = activity_repository
    app.state.preferences_repository = preferences_repository
    app.state.collector_service = collector_service
    app.state.system_token = system_token
    app.state.migration_applied = migration_applied
    app.state.notifier = notifier

    logger.info("MindFlow v{} startup complete", __version__)

    yield  # ── Application runs here ──

    # ── Graceful shutdown ─────────────────────────────────────────────
    logger.info("Shutting down MindFlow...")

    # 1. Stop collector
    if collector_service is not None:
        try:
            await asyncio.wait_for(collector_service.stop(), timeout=3.0)
        except TimeoutError:
            logger.warning("Collector stop timed out, forcing")
        except Exception as exc:
            logger.warning("Collector stop error: {}", exc)

    # 2. Dispose engine
    try:
        await asyncio.wait_for(engine.dispose(), timeout=3.0)
    except TimeoutError:
        logger.warning("Engine dispose timed out")
    except Exception as exc:
        logger.warning("Engine dispose error: {}", exc)

    logger.info("MindFlow shutdown complete")


# ── App factory ────────────────────────────────────────────────────────────


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure a MindFlow FastAPI application instance.

    Args:
        settings: Application settings. If None, loads from defaults.

    Returns:
        A fully configured FastAPI application ready to serve.
    """
    if settings is None:
        from mindflow.config import get_settings

        settings = get_settings()

    # Configure logging
    setup_logging(settings)

    app = FastAPI(
        title="MindFlow API",
        description="Local-first intelligent focus assistant",
        version=__version__,
        lifespan=_lifespan,  # type: ignore[arg-type]
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Store settings for lifespan access
    app.state.settings = settings

    # ── Exception handlers (wraps everything) ─────────────────────────
    register_exception_handlers(app)

    # ── Middleware (order per §3.5, outermost first) ──────────────────

    # 1. StructuredLoggingMiddleware (request_id + timing)
    app.add_middleware(StructuredLoggingMiddleware)

    # 2. HostValidationMiddleware (localhost only)
    app.add_middleware(HostValidationMiddleware)

    # 3. AuthMiddleware (token check, exempt /health and /docs)
    # Token is read from app.state.system_token at request time,
    # so it doesn't need to be set during construction.
    app.add_middleware(AuthMiddleware)

    # 4. RateLimitMiddleware (token bucket)
    app.add_middleware(RateLimitMiddleware)

    # 5. CORSMiddleware (localhost origins only)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ─────────────────────────────────────────────────────────
    register_routes(app)

    # ── WebSocket ──────────────────────────────────────────────────────
    app.include_router(websocket_router, prefix="/api/v1")

    # ── Startup security headers (via middleware) ──────────────────────

    @app.middleware("http")
    async def add_security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Add security headers to every response."""
        response = await call_next(request)
        response.headers["X-MindFlow-Version"] = __version__
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    logger.info("MindFlow app created (v{})", __version__)
    return app
