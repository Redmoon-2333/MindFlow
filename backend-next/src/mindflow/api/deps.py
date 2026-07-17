"""FastAPI dependency injection for route handlers.

All shared dependencies are defined as callable ``Depends()`` functions
that extract instances from ``app.state`` (set during app creation in
``app.py``).

Route modules import ``Depends`` and the individual ``get_*`` functions
to declare dependencies inline:
  ``async def handler(repo = Depends(get_activity_repo)):``

No global singletons — every dependency is injected through FastAPI's
dependency resolution, making testing straightforward via override.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.preferences import (
    PreferencesRepository,
)
from mindflow.services.collector_service import CollectorService


def get_collector_service(request: Request) -> CollectorService | None:
    """Return the CollectorService instance from app.state."""
    return cast(CollectorService | None, getattr(request.app.state, "collector_service", None))


def get_engine(request: Request) -> AsyncEngine:
    """Return the AsyncEngine from app.state."""
    return cast(AsyncEngine, request.app.state.engine)


def get_session_factory(request: Request) -> async_sessionmaker[AsyncSession]:
    """Return the async_sessionmaker from app.state."""
    return cast(async_sessionmaker[AsyncSession], request.app.state.session_factory)


def get_migration_status(request: Request) -> bool:
    """Return migration status from app.state."""
    return cast(bool, getattr(request.app.state, "migration_applied", False))


def get_activity_repo(request: Request) -> SQLAlchemyActivityRepository:
    """Return the ActivityRepository from app.state."""
    return cast(SQLAlchemyActivityRepository, request.app.state.activity_repository)


def get_preferences_repo(request: Request) -> PreferencesRepository:
    """Return the PreferencesRepository from app.state."""
    return cast(PreferencesRepository, request.app.state.preferences_repository)


def get_system_token(request: Request) -> str:
    """Return the system token from app.state."""
    return cast(str, request.app.state.system_token)
