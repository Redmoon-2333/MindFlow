"""Expert panel integration service — G003 wiring layer.

Connects the G001 EvidenceBundleBuilder → G002 PanelOrchestrator → existing
LLMService fallback chain for the daily expert panel workflow.

Degradation chain (07-agent-upgrade-design.md §5):
  L1: Expert panel (PanelOrchestrator)
  L2: Single-expert (existing LLMService.analyze)
  L3+: Handled by LLMService's own degradation (Ollama → RuleEngine)
"""

from __future__ import annotations

from datetime import date
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.agents.orchestrator import PanelOrchestrator
from mindflow.agents.types import (
    PanelBudgetExceededError,
    PanelUnavailableError,
    PanelVerdict,
    TranscriptEntry,
)
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.ports import AnalysisRequest, AnalysisWorkflowPort, OriginType
from mindflow.services.effectiveness_service import EffectivenessService
from mindflow.services.evidence_service import EvidenceBundleBuilder
from mindflow.services.llm_service import LLMService
from mindflow.time_utils import TimezoneLike, business_day_bounds_utc

# ═══════════════════════════════════════════════════════════════════════════════
# Unified verdict conversion
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_dict_to_panel_verdict(
    assessment: dict[str, Any],
    *,
    escalated: bool | None = None,
    transcript: tuple[TranscriptEntry, ...] | None = None,
    call_count: int | None = None,
    source: str | None = None,
) -> PanelVerdict:
    """Convert an assessment dict (LLM output or storage) to a ``PanelVerdict``.

    The input dict can use either the LLM-output field names
    (``types``, ``confidence``, ``recommended_technique``) or the storage
    field names (``procrastination_types``, ``type_confidence``,
    ``cbt_technique``).  The function tries LLM names first, then falls
    back to storage names for each field.

    Metadata fields (``escalated``, ``transcript``, ``call_count``,
    ``source``) are taken from explicit keyword arguments when provided;
    otherwise they are extracted from ``panel_transcript`` (or top-level
    ``source``) in the dict.

    This is the single source of truth for ``PanelVerdict`` construction
    from raw dicts.  Both :meth:`PanelService._analysis_to_verdict` and
    :func:`mindflow.agents.orchestrator._verdict_dict_to_panel_verdict`
    delegate to this function.

    Args:
        assessment: Dict with assessment data.
        escalated: Override for ``escalated``.
        transcript: Override for ``transcript``.
        call_count: Override for ``call_count``.
        source: Override for ``source`` (``"panel"``, ``"single_expert"``,
            ``"ollama"``, or ``"rule_engine"``).

    Returns:
        A ``PanelVerdict`` populated from the assessment data.
    """
    # ── Parse types (try LLM key first, then storage key) ───────────────
    types_raw: list[str] | Any = assessment.get("types")
    if not isinstance(types_raw, list):
        types_raw = assessment.get("procrastination_types", [])

    parsed_types: list[ProcrastinationType] = []
    for t in types_raw:
        try:
            parsed_types.append(ProcrastinationType(t))
        except ValueError:
            logger.warning("Unknown procrastination type in verdict: {!r}", t)

    if not parsed_types:
        # Fallback — should not happen with a well-behaved moderator
        parsed_types = [ProcrastinationType.TASK_AVERSION]

    # ── Parse confidence (try LLM key first, then storage key) ──────────
    confidence_raw = assessment.get("confidence")
    if not isinstance(confidence_raw, dict):
        confidence_raw = assessment.get("type_confidence", {})

    confidence: dict[ProcrastinationType, float] = {}
    for k, v in confidence_raw.items():
        try:
            pt = ProcrastinationType(k)
            if isinstance(v, (int, float)):
                confidence[pt] = float(v)
        except ValueError:
            pass

    # Fill in any missing types with a default confidence
    for pt in parsed_types:
        if pt not in confidence:
            confidence[pt] = 0.5

    # ── Parse technique (try LLM key first, then storage key) ───────────
    technique_raw = assessment.get("recommended_technique")
    if technique_raw is None:
        technique_raw = assessment.get("cbt_technique")

    technique: CBTTechnique | None = None
    if technique_raw is not None:
        try:
            technique = CBTTechnique(str(technique_raw))
        except ValueError:
            logger.warning("Unknown CBT technique in verdict: {!r}", technique_raw)

    # ── Parse rationale ─────────────────────────────────────────────────
    rationale = str(assessment.get("rationale", assessment.get("response_text", "")))

    # ── Parse dissent (top-level first, then panel_transcript) ──────────
    dissent_raw: list[str] | Any = assessment.get("dissent")
    if not isinstance(dissent_raw, list):
        panel_data = assessment.get("panel_transcript", {})
        if isinstance(panel_data, dict):
            dissent_raw = panel_data.get("dissent", [])
        else:
            dissent_raw = []

    if not isinstance(dissent_raw, list):
        dissent_raw = []
    dissent = tuple(str(d) for d in dissent_raw)

    # ── Parse transcript ────────────────────────────────────────────────
    if transcript is not None:
        final_transcript = transcript
    else:
        panel_data = assessment.get("panel_transcript", {})
        if isinstance(panel_data, dict):
            raw_tr = panel_data.get("transcript", [])
            if isinstance(raw_tr, list):
                entries: list[TranscriptEntry] = []
                for entry in raw_tr:
                    if not isinstance(entry, dict):
                        continue
                    role = entry.get("role")
                    content = entry.get("content")
                    round_number = entry.get("round")
                    if (
                        isinstance(role, str)
                        and isinstance(content, str)
                        and isinstance(round_number, int)
                        and not isinstance(round_number, bool)
                    ):
                        entries.append(
                            TranscriptEntry(role=role, content=content, round=round_number)
                        )
                final_transcript = tuple(entries)
            else:
                final_transcript = ()
        else:
            final_transcript = ()

    # ── Parse escalated ─────────────────────────────────────────────────
    if escalated is not None:
        final_escalated = escalated
    else:
        panel_data = assessment.get("panel_transcript", {})
        final_escalated = (
            panel_data.get("escalated") is True if isinstance(panel_data, dict) else False
        )

    # ── Parse call_count ────────────────────────────────────────────────
    if call_count is not None:
        final_call_count = call_count
    else:
        panel_data = assessment.get("panel_transcript", {})
        raw_cc = panel_data.get("call_count", 0) if isinstance(panel_data, dict) else 0
        final_call_count = raw_cc if isinstance(raw_cc, int) and not isinstance(raw_cc, bool) else 0

    # ── Parse source ────────────────────────────────────────────────────
    if source is not None:
        final_source = source  # type: ignore[assignment]
    else:
        source_raw = assessment.get("source")
        if source_raw == "panel" or source_raw == "single_expert":
            final_source = source_raw
        elif source_raw == "ollama":
            final_source = "ollama"
        elif source_raw == "rule_engine":
            final_source = "rule_engine"
        else:
            final_source = "single_expert"

    return PanelVerdict(
        types=tuple(parsed_types),
        confidence=confidence,
        recommended_technique=technique,
        rationale=rationale,
        dissent=dissent,
        transcript=final_transcript,
        escalated=final_escalated,
        call_count=final_call_count,
        source=final_source,
    )


class PanelService:
    """Service that wires the expert panel into the daily analysis workflow.

    Args:
        activity_repo: Repository for activity event data.
        intervention_repo: Repository for intervention history.
        session_factory: SQLAlchemy session factory.
        orchestrator: The expert panel orchestrator.
        llm_service: LLM service for fallback (single-expert mode).
        analysis_repository: Repository that persists panel and fallback verdicts.
        effectiveness_service: Effectiveness service for enriching intervention
            records with outcome data (G005 learning loop — optional).
        workflow_port: Optional framework-neutral workflow port. When set,
            :meth:`run_daily_panel` delegates through this port instead of
            executing the panel directly.
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
        session_factory: async_sessionmaker[AsyncSession],
        orchestrator: PanelOrchestrator,
        llm_service: LLMService,
        analysis_repository: SQLAlchemyProcrastinationAnalysisRepository,
        effectiveness_service: EffectivenessService | None = None,
        timezone: TimezoneLike = "local",
        evidence_builder: EvidenceBundleBuilder | None = None,
        workflow_port: AnalysisWorkflowPort | None = None,
    ) -> None:
        # Accept an injected shared EvidenceBundleBuilder, or create one as fallback
        self._builder = evidence_builder or EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            effectiveness_service=effectiveness_service,
        )
        self._orchestrator = orchestrator
        self._llm_service = llm_service
        self._analysis_repository = analysis_repository
        self._timezone = timezone
        self._workflow_port = workflow_port

    async def run_daily_panel(
        self, user_id: int, target_date: date, *, origin: OriginType = "scheduler"
    ) -> PanelVerdict:
        """Run the daily expert panel (or degrade gracefully).

        **Compatibility adapter**: When ``workflow_port`` is injected, this
        method delegates through :class:`~mindflow.ports.AnalysisWorkflowPort`
        instead of executing the panel directly. The port contract keeps the
        call-site signature unchanged while allowing the workflow engine to be
        swapped (LangGraph, state machine, or test fake) without touching
        route handlers.

        Without a workflow port, the existing behaviour is preserved:
        attempts the full multi-expert panel. If the panel is unavailable
        (e.g. insufficient valid expert opinions), falls through to the
        existing single-expert LLM service.

        Args:
            user_id: The user to analyse.
            target_date: The date to analyse.
            origin: Which entry point triggered this run
                (``"scheduler"``, ``"api"``, ``"chat"``, or
                ``"auto_intervention"``).  Only used when delegating
                through the workflow port; ignored in the inline path.

        Returns:
            A ``PanelVerdict`` — either from the full panel (source="panel")
            or from the deepest successful fallback tier
            (source="single_expert", "ollama", or "rule_engine").
        """
        # ── Framework-neutral delegation path ──────────────────────────────────
        workflow_port = getattr(self, "_workflow_port", None)
        if workflow_port is not None:
            request = AnalysisRequest(
                user_id=user_id,
                target_date=target_date,
                force=False,
                origin=origin,
            )
            result = await workflow_port.run_analysis(request)
            return result.verdict

        # ── Existing inline orchestration path ─────────────────────────────────
        # Build evidence bundle ──────────────────────────────────────────────
        window_start, window_end = business_day_bounds_utc(
            target_date,
            self._timezone,
        )

        bundle = await self._builder.build(user_id, window_start, window_end)

        # ── Attempt expert panel ───────────────────────────────────────────────
        try:
            verdict = await self._orchestrator.run(bundle)
            await self._analysis_repository.upsert(
                user_id=user_id,
                target_date=target_date,
                procrastination_types=[item.value for item in verdict.types],
                type_confidence={
                    item.value: float(confidence)
                    for item, confidence in verdict.confidence.items()
                },
                cognitive_distortions=[],
                cbt_technique=(
                    verdict.recommended_technique.value
                    if verdict.recommended_technique is not None
                    else None
                ),
                response_text=verdict.rationale,
                llm_model="deepseek-chat",
                analysis_kind="daily_panel",
                source=verdict.source,
                panel_transcript={
                    "transcript": [
                        {
                            "role": entry.role,
                            "content": entry.content,
                            "round": entry.round,
                        }
                        for entry in verdict.transcript
                    ],
                    "dissent": list(verdict.dissent),
                    "escalated": verdict.escalated,
                    "call_count": verdict.call_count,
                },
            )
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
        # Pass analysis_kind="daily_panel" so the fallback result is stored AS
        # the daily panel (with the correct degradation source), not as a
        # separate daily_attribution row.  This is critical for the chat
        # agent's get_panel_verdict tool to find the result.
        logger.info("Panel unavailable, falling back to single-expert analysis")
        outcome = await self._llm_service.analyze(
            user_id=user_id,
            target_date=target_date,
            force=True,
            analysis_kind="daily_panel",
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
        cached = await self._analysis_repository.get_by_date(
            user_id, target_date, analysis_kind="daily_panel",
        )
        if cached is None:
            return None

        return self._analysis_to_verdict(cached)

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
        assessment = dict(outcome.assessment)
        outcome_source = getattr(outcome, "source", None)
        if outcome_source is not None:
            assessment["source"] = outcome_source

        return PanelService._analysis_to_verdict(assessment)

    @staticmethod
    def _analysis_to_verdict(assessment: dict[str, Any]) -> PanelVerdict:
        """Convert persisted analysis data to a ``PanelVerdict``.

        Delegates to the shared :func:`analysis_dict_to_panel_verdict`,
        passing no overrides so all metadata is extracted from the dict's
        ``panel_transcript`` and ``source`` keys.
        """
        return analysis_dict_to_panel_verdict(assessment)
