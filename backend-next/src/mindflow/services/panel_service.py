"""Expert panel integration service — G003 wiring layer.

Connects the G001 EvidenceBundleBuilder → G002 PanelOrchestrator → existing
LLMService fallback chain for the daily expert panel workflow.

Degradation chain (07-agent-upgrade-design.md §5):
  L1: Expert panel (PanelOrchestrator)
  L2: Single-expert (existing LLMService.analyze)
  L3+: Handled by LLMService's own degradation (Ollama → RuleEngine)
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.agents.orchestrator import PanelOrchestrator
from mindflow.agents.types import (
    PanelBudgetExceededError,
    PanelUnavailableError,
    PanelVerdict,
)
from mindflow.domain.procrastination import ProcrastinationType
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.services.evidence_service import EvidenceBundleBuilder
from mindflow.services.llm_service import LLMService


class PanelService:
    """Service that wires the expert panel into the daily analysis workflow.

    Args:
        activity_repo: Repository for activity event data.
        intervention_repo: Repository for intervention history.
        session_factory: SQLAlchemy session factory.
        orchestrator: The expert panel orchestrator.
        llm_service: LLM service for fallback (single-expert mode).
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
        session_factory: async_sessionmaker[AsyncSession],
        orchestrator: PanelOrchestrator,
        llm_service: LLMService,
    ) -> None:
        self._builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
        )
        self._orchestrator = orchestrator
        self._llm_service = llm_service

    async def run_daily_panel(self, user_id: int, target_date: date) -> PanelVerdict:
        """Run the daily expert panel (or degrade gracefully).

        Attempts the full multi-expert panel. If the panel is unavailable
        (e.g. insufficient valid expert opinions), falls through to the
        existing single-expert LLM service.

        Args:
            user_id: The user to analyse.
            target_date: The date to analyse.

        Returns:
            A ``PanelVerdict`` — either from the full panel (source="panel")
            or from the fallback (source="single_expert").
        """
        # ── Build evidence bundle ──────────────────────────────────────────────
        window_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)
        window_end = window_start + timedelta(days=1)

        bundle = await self._builder.build(user_id, window_start, window_end)

        # ── Attempt expert panel ───────────────────────────────────────────────
        try:
            verdict = await self._orchestrator.run(bundle)
            logger.info(
                "Panel succeeded for user {} on {} ({} calls, escalated={})",
                user_id,
                target_date,
                verdict.call_count,
                verdict.escalated,
            )
            return verdict
        except PanelUnavailableError as exc:
            logger.warning(
                "Panel unavailable, falling back to single-expert analysis: {}",
                exc,
            )
        except PanelBudgetExceededError as exc:
            logger.warning(
                "Panel budget exceeded, falling back to single-expert analysis: {}",
                exc,
            )

        # ── Fallback to single-expert LLM service ──────────────────────────────
        logger.info("Panel unavailable, falling back to single-expert analysis")
        outcome = await self._llm_service.analyze(
            user_id=user_id,
            target_date=target_date,
            force=True,
        )

        return self._outcome_to_verdict(outcome)

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _outcome_to_verdict(outcome: Any) -> PanelVerdict:
        """Convert an ``AttributionOutcome`` to a ``PanelVerdict``.

        The outcome has an ``assessment`` dict with keys like
        ``procrastination_types``, ``type_confidence``, ``cbt_technique``,
        ``response_text``, etc.
        """
        import contextlib

        assessment: dict[str, Any] = outcome.assessment

        # Parse types
        types_raw: list[str] = assessment.get("procrastination_types", [])
        parsed_types: list[ProcrastinationType] = []
        for t in types_raw:
            with contextlib.suppress(ValueError):
                parsed_types.append(ProcrastinationType(t))

        # Parse confidence
        conf_raw: dict[str, object] = assessment.get("type_confidence", {})
        confidence: dict[ProcrastinationType, float] = {}
        for k, v in conf_raw.items():
            with contextlib.suppress(ValueError):
                pt = ProcrastinationType(k)
                if isinstance(v, (int, float)):
                    confidence[pt] = float(v)

        # Fill in missing confidence
        for pt in parsed_types:
            if pt not in confidence:
                confidence[pt] = 0.5

        # Parse technique
        from mindflow.domain.procrastination import CBTTechnique

        technique_raw = assessment.get("cbt_technique")
        technique: CBTTechnique | None = None
        if technique_raw is not None:
            with contextlib.suppress(ValueError):
                technique = CBTTechnique(str(technique_raw))

        rationale = str(assessment.get("response_text", assessment.get("rationale", "")))

        return PanelVerdict(
            types=tuple(parsed_types),
            confidence=confidence,
            recommended_technique=technique,
            rationale=rationale,
            dissent=(),
            transcript=(),
            escalated=False,
            call_count=0,
            source="single_expert",
        )
