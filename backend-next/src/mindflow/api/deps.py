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

from mindflow.infrastructure.notification import (  # noqa: F401
    NotificationService,
)
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.infrastructure.repositories.preferences import (
    PreferencesRepository,
)
from mindflow.infrastructure.repositories.report import (
    SQLAlchemyDailyReportRepository,
)
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.autonomy_service import AutonomyService
from mindflow.services.chat_service import ChatService
from mindflow.services.collector_service import CollectorService
from mindflow.services.llm_service import LLMService
from mindflow.services.maintenance_service import MaintenanceService
from mindflow.services.panel_service import PanelService
from mindflow.services.report_service import ReportService


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


def get_focus_repo(request: Request) -> SQLAlchemyFocusSessionRepository:
    """Return the FocusSessionRepository from app.state."""
    return cast(SQLAlchemyFocusSessionRepository, request.app.state.focus_repository)


def get_report_repo(request: Request) -> SQLAlchemyDailyReportRepository:
    """Return the DailyReportRepository from app.state."""
    return cast(SQLAlchemyDailyReportRepository, request.app.state.report_repository)


def get_analysis_service(request: Request) -> AnalysisService:
    """Return the AnalysisService from app.state."""
    return cast(AnalysisService, request.app.state.analysis_service)


def get_report_service(request: Request) -> ReportService:
    """Return the ReportService from app.state."""
    return cast(ReportService, request.app.state.report_service)


def get_maintenance_service(request: Request) -> MaintenanceService:
    """Return the MaintenanceService from app.state."""
    return cast(MaintenanceService, request.app.state.maintenance_service)


def get_notifier(request: Request) -> NotificationService:
    """Return the NotificationService from app.state."""
    return cast(NotificationService, request.app.state.notifier)


def get_llm_service(request: Request) -> LLMService:
    """Return the LLMService from app.state."""
    return cast(LLMService, request.app.state.llm_service)


def get_panel_service(request: Request) -> PanelService:
    """Return the PanelService from app.state."""
    return cast(PanelService, request.app.state.panel_service)


def get_intervention_repo(request: Request) -> InterventionLogRepository:
    """Return the InterventionLogRepository from app.state."""
    return cast(
        InterventionLogRepository,
        request.app.state.intervention_repository,
    )


def get_intervention_service(request: Request) -> object:
    """Return the InterventionService from app.state."""
    return getattr(request.app.state, "intervention_service", None)


def get_effectiveness_service(request: Request) -> object:
    """Return the EffectivenessService from app.state."""
    return getattr(request.app.state, "effectiveness_service", None)


def get_chat_service(request: Request) -> ChatService:
    """Return the ChatService from app.state."""
    return cast(ChatService, request.app.state.chat_service)


def get_autonomy_service(request: Request) -> AutonomyService:
    """Return the AutonomyService from app.state."""
    return cast(AutonomyService, request.app.state.autonomy_service)
