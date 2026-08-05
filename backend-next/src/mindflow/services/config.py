"""Service-layer configuration dataclasses.

Provides typed ``*Config`` dataclasses for services whose constructors
have grown many positional parameters. Each config can be passed as a
single object, reducing constructor churn and making dependency injection
at the wiring layer (``api/dependencies.py`` or ``mindflow/main.py``)
more readable.

Usage::

    cfg = PanelServiceConfig(
        activity_repo=activity_repo,
        intervention_repo=intervention_repo,
        session_factory=session_factory,
        llm_service=llm_service,
        analysis_repository=analysis_repository,
    )
    service = PanelService(cfg)

Or pass config as a keyword arg alongside existing positional params
for backward compatibility::

    service = PanelService(
        activity_repo, intervention_repo, ..., config=cfg
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PanelServiceConfig:
    """Configuration for :class:`mindflow.services.panel_service.PanelService`.

    Attributes match the ``PanelService.__init__`` signature. Use ``None``
    to defer to the service's own default behaviour.
    """

    activity_repo: Any  # SQLAlchemyActivityRepository
    intervention_repo: Any  # InterventionLogRepository
    session_factory: Any  # async_sessionmaker[AsyncSession]
    llm_service: Any  # LLMService
    analysis_repository: Any  # SQLAlchemyProcrastinationAnalysisRepository
    effectiveness_service: Any | None = None  # EffectivenessService
    timezone: str = "local"
    evidence_builder: Any | None = None  # EvidenceBundleBuilder


@dataclass
class ChatServiceConfig:
    """Configuration for :class:`mindflow.services.chat_service.ChatService`.

    Attributes match the ``ChatService.__init__`` signature. Use ``None``
    to defer to the service's own default behaviour.

    Note: ``panel_service`` is a positional (required) parameter in the
    original constructor despite its optional type annotation; it is
    therefore required in this config as well.
    """

    session_factory: Any  # async_sessionmaker[AsyncSession]
    crisis_detector: Any  # CrisisDetector
    llm_gateway: Any  # DeepSeekGateway
    analysis_repo: Any  # SQLAlchemyProcrastinationAnalysisRepository
    panel_service: Any  # PanelService | None (positional, may be None)
    intervention_repo: Any  # InterventionLogRepository
    evidence_builder: Any  # EvidenceBundleBuilder
    chat_repo: Any | None = None  # ChatRepository
    max_history_rounds: int | None = None
    agent: Any | None = None
    model: Any | None = None  # BaseChatModel
    timezone: Any | None = None  # TimezoneLike


@dataclass
class InterventionServiceConfig:
    """Configuration for intervention-related services.

    Reserved for future use when ``InterventionService`` constructor
    needs consolidation.
    """

    intervention_repo: Any  # InterventionLogRepository
    effectiveness_service: Any | None = None  # EffectivenessService
    timezone: str = "local"
