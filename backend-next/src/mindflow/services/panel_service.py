"""Expert panel integration service — G003 wiring layer.

Connects the G001 EvidenceBundleBuilder → G002 PanelOrchestrator → existing
LLMService fallback chain for the daily expert panel workflow.

Degradation chain (07-agent-upgrade-design.md §5):
  L1: Expert panel (PanelOrchestrator)
  L2: Single-expert (existing LLMService.analyze)
  L3+: Handled by LLMService's own degradation (Ollama → RuleEngine)
"""

from __future__ import annotations

from dataclasses import dataclass
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
from mindflow.services.effectiveness_service import EffectivenessService
from mindflow.services.evidence_service import EvidenceBundleBuilder
from mindflow.services.llm_service import LLMService


@dataclass(frozen=True)
class _StoredOutcome:
    """Minimal adapter wrapping a stored analysis dict for ``_outcome_to_verdict``.

    ``_outcome_to_verdict`` only reads ``.assessment``; the persisted analysis
    dict already matches that shape, so this avoids duplicating the mapping.
    """

    assessment: dict[str, Any]


class PanelService:
    """Service that wires the expert panel into the daily analysis workflow.

    Args:
        activity_repo: Repository for activity event data.
        intervention_repo: Repository for intervention history.
        session_factory: SQLAlchemy session factory.
        orchestrator: The expert panel orchestrator.
        llm_service: LLM service for fallback (single-expert mode).
        effectiveness_service: Effectiveness service for enriching intervention
            records with outcome data (G005 learning loop — optional).
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
        session_factory: async_sessionmaker[AsyncSession],
        orchestrator: PanelOrchestrator,
        llm_service: LLMService,
        effectiveness_service: EffectivenessService | None = None,
    ) -> None:
        self._builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            effectiveness_service=effectiveness_service,
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

    async def get_stored_verdict(self, user_id: int, target_date: date) -> PanelVerdict | None:
        """Return the most recent stored analysis as a verdict, or None.

        Read-only: unlike ``run_daily_panel`` this triggers NO LLM calls. It
        serves the last persisted attribution (written by ``run_daily_panel``'s
        fallback path or the daily cron) so a GET stays idempotent and free
        (review C3 — a GET must not run the 6-12-call panel).

        Args:
            user_id: The user to look up.
            target_date: The date to look up.

        Returns:
            A ``PanelVerdict`` reconstructed from the stored analysis, or
            None if nothing has been analysed for that date yet.
        """
        cached = await self._llm_service._analysis_repo.get_by_date(  # noqa: SLF001
            user_id, target_date
        )
        if cached is None:
            return None

        # Reuse the outcome→verdict mapping; wrap the stored dict in the minimal
        # shape _outcome_to_verdict expects (an object with an ``assessment``).
        outcome = _StoredOutcome(assessment=cached)
        return self._outcome_to_verdict(outcome)

    async def aclose(self) -> None:
        """Close the underlying LLM gateway HTTP client.

        Cleanup hook for application shutdown (review P2 connection leak).
        """
        import contextlib

        with contextlib.suppress(Exception):
            await self._orchestrator._gateway.close()  # noqa: SLF001

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
