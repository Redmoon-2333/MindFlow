"""Panel deliberation graph — LangGraph StateGraph with Send fan-out.

Replaces the nested closures in ``PanelOrchestrator._build_compiled_graph``
with module-level testable callables, explicit validation nodes, reducer-based
fan-in, and conditional routers.

Graph topology (fast path, ~6 calls):
  analyst → parse_val → citation_val → forbidden_val
    → [Send fanout: attribution_call × 3 (parallel)]
    → conflict_detection
    → _panel_routing(min_valid+conflict) → [moderator|rebuttal|END]
    → rebuttal → parse_val → citation_val → forbidden_val → moderator
    → moderator → human_review_interrupt → critic
    → critic_verdict → [approved→END | retry→moderator | exhausted→END]

Design constraints:
  - Every node is a module-level async callable (testable in isolation).
  - ``Send`` provides parallel attribution fan-out (replaces asyncio.gather).
  - Validation nodes are MANDATORY graph steps (not optional tools).
  - Reducer-based fan-in guarantees order-independent opinion accumulation.
  - All parsing/prompt helpers imported lazily to avoid circular imports.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

if TYPE_CHECKING:
    # Private langgraph typing alias; used only as a return-type annotation so
    # it is never evaluated at runtime (``from __future__ import annotations``).
    from langgraph.graph._node import StateNode

from mindflow.agents.conflict import ConflictReport, detect_conflict
from mindflow.agents.disagreement import (
    DisagreementSummary,
    RebuttalDelta,
    analyze_disagreement,
    compute_rebuttal_delta,
)
from mindflow.agents.experts import (
    ANALYST,
    ATTRIBUTION_EXPERTS,
    CRITIC,
    MODERATOR,
    ExpertDef,
)
from mindflow.agents.llm_gateway import (
    GatewayAPIError,
    GatewayNotConfiguredError,
    PanelLLMGateway,
)
from mindflow.agents.types import (
    CriticResult,
    ExpertOpinion,
    PanelBudgetExceededError,
    PanelUnavailableError,
    TranscriptEntry,
    _contains_forbidden_words,
)
from mindflow.graph.reducers import append_opinion, append_transcript

# ═══════════════════════════════════════════════════════════════════════════════
# Budget helpers — standalone (was on PanelOrchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_CALLS: int = 12
"""Total LLM-call cap per panel run (legacy name kept for tests)."""

# Phase-aware budgets (architecture plan G/1.3): each expert role has its
# own allowance so the moderator/critic debate cannot starve the analyst
# round. The per-role budget is enforced in addition to the global
# _MAX_CALLS cap.
_PHASE_BUDGETS: dict[str, int] = {
    "analyst": 1,
    "attribution": 3,
    "moderator": 3,
    "critic": 2,
    "rebuttal": 1,
}


@dataclass
class _PanelRunContext:
    """Mutable state owned by exactly one panel invocation.

    Mirrors ``PanelOrchestrator._PanelRunContext`` so the new graph
    nodes can use the same concurrency-safe budget tracking.
    """

    call_count: int = 0
    transcript: list[TranscriptEntry] = field(default_factory=list)
    budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Per-role LLM-call usage for phase budgets (architecture plan G/1.3).
    phase_usage: dict[str, int] = field(default_factory=dict)


# Context variable to carry the mutable per-invocation runtime through the
# LangGraph StateGraph without including it in the checkpointable state.
# Set by PanelOrchestrator._run_graph before ainvoke; read by all graph nodes.
_PANEL_RUNTIME: contextvars.ContextVar[_PanelRunContext] = contextvars.ContextVar("pg_runtime")


async def _call_with_budget(
    runtime: Any,  # object from orchestrator to avoid circular import
    gateway: PanelLLMGateway,
    expert: ExpertDef,
    user_message: str,
) -> str:
    """Atomic budget check then gateway call.

    Enforces both the global cap and the per-phase allowance keyed by the
    expert role (architecture plan G/1.3)."""
    async with runtime.budget_lock:
        runtime.call_count += 1
        if runtime.call_count > _MAX_CALLS:
            raise PanelBudgetExceededError(call_count=runtime.call_count)
        phase_budget = _PHASE_BUDGETS.get(expert.role, 0)
        phase_used = runtime.phase_usage.get(expert.role, 0)
        if phase_budget > 0 and phase_used >= phase_budget:
            logger.warning(
                "Phase budget exhausted for {} ({} / {})",
                expert.role, phase_used, phase_budget,
            )
            raise PanelBudgetExceededError(call_count=runtime.call_count)
        runtime.phase_usage[expert.role] = phase_used + 1
    return await gateway.complete(
        system=expert.system_prompt,
        user=user_message,
        model=expert.model,
    )


async def _safe_call_with_budget(
    runtime: Any,  # object from orchestrator to avoid circular import
    gateway: PanelLLMGateway,
    expert: ExpertDef,
    user_message: str,
) -> str:
    """Like ``_call_with_budget`` but returns empty string on failure."""
    try:
        return await _call_with_budget(runtime, gateway, expert, user_message)
    except PanelBudgetExceededError:
        raise
    except Exception as exc:
        logger.error("Parallel call to {} failed: {}", expert.role, exc)
        return ""


# Backoff before re-running an all-empty expert batch (a transient provider
# blip usually takes the whole parallel batch down at once).
_BATCH_RETRY_DELAY_S = 3.0


async def _fanout_raw_with_batch_retry(
    task_factories: list[Callable[[], Awaitable[str]]],
    *,
    batch_label: str,
) -> list[str]:
    """Run a parallel expert batch, retrying the whole batch once when empty.

    ``_safe_call_with_budget`` swallows transport/API failures into an empty
    string, so an all-empty batch is the signature of a transient connectivity
    failure (e.g. DeepSeek connection blip) rather than a content/prompt issue.
    In that case retry the entire batch once after a short backoff.  The retry
    is naturally bounded: every attempt bills against ``runtime.call_count``
    and the panel's global call budget, so it cannot loop forever.

    Args:
        task_factories: Zero-argument async factories (must be re-runnable).
        batch_label: Human label for logging (e.g. ``"attribution"``).

    Returns:
        The (possibly retried) raw responses, in task order.
    """
    if not task_factories:
        return []
    results = list(await asyncio.gather(*(f() for f in task_factories)))
    if all(not r for r in results):
        logger.warning(
            "{} all-empty responses (transient failure?); retrying batch once",
            batch_label,
        )
        await asyncio.sleep(_BATCH_RETRY_DELAY_S)
        results = list(await asyncio.gather(*(f() for f in task_factories)))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Reducer adapters for LangGraph annotated channels
# ═══════════════════════════════════════════════════════════════════════════════


def _reduce_attribution_opinions(
    existing: tuple[ExpertOpinion, ...] | None,
    update: ExpertOpinion | tuple[ExpertOpinion, ...],
) -> tuple[ExpertOpinion, ...]:
    """LangGraph-compatible adapter for ``append_opinion``.

    Called by the graph engine when nodes return partial
    ``{"attribution_opinions": opinion}`` updates.  Handles both
    single-opinion updates (from individual Send branches) and
    tuple-of-opinions updates (from LangGraph channel merges).
    """
    if isinstance(update, tuple):
        # LangGraph channel merge — apply each element individually
        result: tuple[ExpertOpinion, ...] = existing or ()
        for op in update:
            result = append_opinion(result, op)
        return result
    return append_opinion(existing, update)


def _reduce_transcript(
    existing: tuple[TranscriptEntry, ...] | None,
    update: TranscriptEntry | tuple[TranscriptEntry, ...],
) -> tuple[TranscriptEntry, ...]:
    """LangGraph-compatible adapter for ``append_transcript``."""
    if isinstance(update, tuple):
        result: tuple[TranscriptEntry, ...] = existing or ()
        for entry in update:
            result = append_transcript(result, entry)
        return result
    return append_transcript(existing, update)


# ═══════════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════════


class PanelGraphState(TypedDict, total=False):
    """State flowing through the LangGraph deliberation graph.

    Fields marked ``Annotated[..., reducer]`` are accumulated via
    the reducer function when multiple nodes write to them (parallel
    fan-in).  Other fields use last-write-wins semantics.
    """

    # ── Input fields (set by PanelOrchestrator._run_graph) ────────────
    bundle_json: str
    valid_metrics: tuple[str, ...]
    # ── Accumulated via reducers (parallel fan-in) ────────────────────
    attribution_opinions: Annotated[
        tuple[ExpertOpinion, ...],
        _reduce_attribution_opinions,
    ]
    transcript: Annotated[
        tuple[TranscriptEntry, ...],
        _reduce_transcript,
    ]

    # ── Single-writer fields ─────────────────────────────────────────
    analyst_opinion: ExpertOpinion | None
    conflict_report: ConflictReport | None
    escalated: bool
    moderator_verdict: dict[str, Any] | None
    critic_result: CriticResult | None
    critic_retries: int
    moderator_redo_count: int
    call_count: int
    disagreement_summary: DisagreementSummary | None
    rebuttal_delta: RebuttalDelta | None

    # ── Fan-out identification (set by Send, read by target node) ────
    _expert_index: int


# ═══════════════════════════════════════════════════════════════════════════════
# Node factories (module-level, testable)
# ═══════════════════════════════════════════════════════════════════════════════


def make_analyst_node(
    gateway: PanelLLMGateway,
) -> StateNode[PanelGraphState, None]:
    """Create the analyst call node (round 0).

    Calls the analyst LLM, parses the output, and performs inline
    forbidden-word + citation validation (behaviour parity).
    """

    async def analyst_node(state: PanelGraphState) -> dict[str, Any]:
        # Lazy import to avoid circular dependency with orchestrator
        from mindflow.agents.orchestrator import (  # noqa: PLC0415
            _PANEL_RUNTIME,
            _opinion_summary,
            _parse_analyst_opinion,
            validate_citations,
        )

        runtime = _PANEL_RUNTIME.get()
        logger.info("Panel round 0: Analyst")
        raw = await _call_with_budget(
            runtime, gateway, ANALYST, state["bundle_json"],
        )
        analyst = _parse_analyst_opinion(raw, ANALYST)

        # Citation validation (prevent hallucinated refs)
        bogus = validate_citations(analyst, state["valid_metrics"])
        if bogus:
            logger.warning("Hallucinated citations {} in analyst — marking", bogus)
            analyst = ExpertOpinion(
                role=ANALYST.role,
                perspective=ANALYST.perspective,
                attribution_types=(),
                confidence={},
                evidence_citations=(),
                argument="",
                raw_json=raw,
                skipped=True,
            )

        runtime.transcript.append(
            TranscriptEntry(
                role=ANALYST.role,
                content=_opinion_summary(analyst),
                round=0,
            ),
        )

        return {
            "analyst_opinion": analyst,
            "transcript": TranscriptEntry(
                role=ANALYST.role,
                content=_opinion_summary(analyst),
                round=0,
            ),
            "call_count": runtime.call_count,
        }

    return analyst_node


def make_attribution_node(
    gateway: PanelLLMGateway,
) -> StateNode[PanelGraphState, None]:
    """Create the multi-expert attribution node (round 1).

    Calls all three attribution experts in parallel via asyncio.gather
    (behaviour parity with the original orchestrator).

    Each expert result is returned as a single ``ExpertOpinion``,
    accumulated via the ``_reduce_attribution_opinions`` reducer.
    """

    async def attribution_node(state: PanelGraphState) -> dict[str, Any]:
        from mindflow.agents.orchestrator import (  # noqa: PLC0415
            _PANEL_RUNTIME,
            _opinion_summary,
            _parse_expert_opinion,
        )

        runtime = _PANEL_RUNTIME.get()
        logger.info("Panel round 1: Attribution experts (parallel)")

        async def _call_raw(exp: ExpertDef) -> str:
            return await _safe_call_with_budget(
                runtime, gateway, exp, state["bundle_json"],
            )

        def _mk_att_task(exp: ExpertDef) -> Callable[[], Awaitable[str]]:
            async def _task() -> str:
                return await _call_raw(exp)
            return _task

        # Batch retry: a transient provider blip takes all parallel calls down
        # at once (all raw responses empty) — retry the whole batch once.
        raws = await _fanout_raw_with_batch_retry(
            [_mk_att_task(exp) for exp in ATTRIBUTION_EXPERTS],
            batch_label="attribution",
        )

        async def _parse_one(exp: ExpertDef, raw: str) -> ExpertOpinion:
            op = _parse_expert_opinion(
                raw, exp, valid_metrics=state["valid_metrics"],
            )
            # Retry once if skipped due to forbidden words
            if op.skipped and _contains_forbidden_words(raw):
                logger.warning(
                    "{} triggered forbidden words, retrying once", exp.role,
                )
                retry_msg = (
                    "你的上一条回复包含禁用词汇（诊断、治疗、患者、处方）。"
                    "请用中文重新输出，严格遵守禁用词规则。"
                )
                raw2 = await _safe_call_with_budget(runtime, gateway, exp, retry_msg)
                op2 = _parse_expert_opinion(
                    raw2, exp, valid_metrics=state["valid_metrics"],
                )
                if not op2.skipped:
                    return op2
                logger.warning(
                    "{} retry still failed, using original", exp.role,
                )
            return op

        opinions = [
            await _parse_one(exp, raw)
            for exp, raw in zip(ATTRIBUTION_EXPERTS, raws, strict=True)
        ]

        for op in opinions:
            runtime.transcript.append(
                TranscriptEntry(
                    role=op.role,
                    content=_opinion_summary(op),
                    round=1,
                ),
            )

        return {
            "attribution_opinions": tuple(opinions),
            "transcript": TranscriptEntry(
                role="attribution",
                content=f"归因完成：{len([o for o in opinions if not o.skipped])}份有效",
                round=1,
            ),
            "call_count": runtime.call_count,
        }

    return attribution_node


async def parse_validation_node(state: PanelGraphState) -> dict[str, Any]:
    """Validate that all opinions have valid JSON (not skipped due to parse failure).

    MANDATORY graph step.  Double-checks what call nodes already do inline.
    Does NOT change behavior — skipped opinions stay skipped.
    """
    verified: int = 0
    skipped_count: int = 0

    for op in state.get("attribution_opinions", ()):
        if not op.skipped and op.argument:
            verified += 1
        elif op.skipped:
            skipped_count += 1
            logger.debug("parse_validation: {} skipped (no valid JSON)", op.role)

    analyst = state.get("analyst_opinion")
    if analyst is not None:
        if not analyst.skipped and analyst.argument:
            verified += 1
        elif analyst.skipped:
            skipped_count += 1

    logger.info(
        "parse_validation: {} valid / {} skipped opinions",
        verified,
        skipped_count,
    )
    return {}


async def citation_validation_node(state: PanelGraphState) -> dict[str, Any]:
    """Code-level citation validation for all opinions in state.

    MANDATORY graph step.
    """
    from mindflow.agents.orchestrator import validate_citations  # noqa: PLC0415

    valid_metrics: tuple[str, ...] = state["valid_metrics"]
    opinions: tuple[ExpertOpinion, ...] = state.get("attribution_opinions", ())
    new_opinions: list[ExpertOpinion] = []
    bogus_count: int = 0

    for op in opinions:
        if op.skipped:
            new_opinions.append(op)
            continue

        bogus = validate_citations(op, valid_metrics)
        if bogus:
            logger.warning(
                "citation_validation: {} has bogus citations {} — marking skipped",
                op.role,
                bogus,
            )
            new_opinions.append(
                ExpertOpinion(
                    role=op.role,
                    perspective=op.perspective,
                    attribution_types=(),
                    confidence={},
                    evidence_citations=(),
                    argument="",
                    raw_json=op.raw_json,
                    skipped=True,
                ),
            )
            bogus_count += 1
        else:
            new_opinions.append(op)

    if bogus_count > 0:
        return {"attribution_opinions": tuple(new_opinions)}
    return {}


async def forbidden_word_validation_node(state: PanelGraphState) -> dict[str, Any]:
    """Forbidden-word gate — MANDATORY graph step.

    Checks every opinion's argument for forbidden medical terms.
    Marks offending opinions as skipped.
    """
    opinions: tuple[ExpertOpinion, ...] = state.get("attribution_opinions", ())
    new_opinions: list[ExpertOpinion] = []
    violations: int = 0

    for op in opinions:
        if op.skipped:
            new_opinions.append(op)
            continue

        forbidden = _contains_forbidden_words(op.argument)
        if forbidden:
            logger.warning(
                "forbidden_word_validation: {!r} in {} opinion — marking skipped",
                forbidden,
                op.role,
            )
            new_opinions.append(
                ExpertOpinion(
                    role=op.role,
                    perspective=op.perspective,
                    attribution_types=(),
                    confidence={},
                    evidence_citations=(),
                    argument="",
                    raw_json=op.raw_json,
                    skipped=True,
                ),
            )
            violations += 1
        else:
            new_opinions.append(op)

    if violations > 0:
        return {"attribution_opinions": tuple(new_opinions)}
    return {}


async def conflict_detection_node(state: PanelGraphState) -> dict[str, Any]:
    """Pure-function conflict detection + disagreement analytics (no LLM call)."""
    logger.info("Conflict detection")
    opinions: tuple[ExpertOpinion, ...] = state["attribution_opinions"]
    conflict = detect_conflict(opinions)
    escalated = conflict.has_conflict

    if escalated:
        logger.info("Conflict detected: {}", conflict.details)
    else:
        logger.info("No conflict among attribution experts")

    ds = analyze_disagreement(
        opinions,
        conflict.details,
        conflict.max_confidence_gap,
        rebuttal_delta=None,
    )
    logger.info(
        "Disagreement analytics: agreement={:.3f}, stability={}",
        ds.agreement_strength,
        ds.stability,
    )

    return {
        "conflict_report": conflict,
        "escalated": escalated,
        "disagreement_summary": ds,
    }


def make_rebuttal_node(
    gateway: PanelLLMGateway,
) -> StateNode[PanelGraphState, None]:
    """Create the rebuttal round node (round 2a).

    All three attribution experts rebut each other in parallel
    (internal asyncio.gather for behaviour parity).
    """

    async def rebuttal_node(state: PanelGraphState) -> dict[str, Any]:
        from mindflow.agents.orchestrator import (  # noqa: PLC0415
            _PANEL_RUNTIME,
            _build_rebuttal_prompt,
            _opinion_summary,
            _parse_expert_opinion,
        )

        runtime = _PANEL_RUNTIME.get()
        logger.info("Panel round 2a: Attribution rebuttal (parallel)")
        opinions: tuple[ExpertOpinion, ...] = state["attribution_opinions"]

        prompts = [
            (
                ATTRIBUTION_EXPERTS[i],
                _build_rebuttal_prompt(
                    state["bundle_json"], opinions, i,
                ),
            )
            for i in range(len(ATTRIBUTION_EXPERTS))
        ]
        def _mk_rebuttal_task(
            exp: ExpertDef, msg: str,
        ) -> Callable[[], Awaitable[str]]:
            async def _task() -> str:
                return await _safe_call_with_budget(runtime, gateway, exp, msg)
            return _task

        responses = await _fanout_raw_with_batch_retry(
            [_mk_rebuttal_task(exp, msg) for exp, msg in prompts],
            batch_label="rebuttal",
        )
        new_opinions: list[ExpertOpinion] = []
        for raw, exp in zip(responses, ATTRIBUTION_EXPERTS, strict=True):
            op = _parse_expert_opinion(
                raw, exp, valid_metrics=state["valid_metrics"],
            )
            # Retry once on forbidden words
            if op.skipped and _contains_forbidden_words(raw):
                logger.warning(
                    "{} rebuttal triggered forbidden words, retrying", exp.role,
                )
                retry_msg = (
                    "你的上一条回复包含禁用词汇（诊断、治疗、患者、处方）。"
                    "请用中文重新输出，严格遵守禁用词规则并回到推理内容。"
                )
                raw2 = await _safe_call_with_budget(runtime, gateway, exp, retry_msg)
                op2 = _parse_expert_opinion(
                    raw2, exp, valid_metrics=state["valid_metrics"],
                )
                if not op2.skipped:
                    op = op2
            new_opinions.append(op)

        for op in new_opinions:
            runtime.transcript.append(
                TranscriptEntry(
                    role=op.role,
                    content=_opinion_summary(op),
                    round=2,
                ),
            )

        non_skipped = [o for o in new_opinions if not o.skipped]
        if len(non_skipped) < 2:
            raise PanelUnavailableError(
                reason=f"辩论后仅{len(non_skipped)}份归因意见有效",
                call_count=runtime.call_count,
            )

        delta = compute_rebuttal_delta(opinions, new_opinions)
        logger.info(
            "Rebuttal delta: agreement {:.3f}→{:.3f}, delta={:+.3f}, converged={}",
            delta.before_agreement,
            delta.after_agreement,
            delta.agreement_delta,
            delta.converged,
        )

        return {
            "attribution_opinions": tuple(new_opinions),
            "transcript": TranscriptEntry(
                role="rebuttal",
                content=f"辩论完成，一致性变化{delta.agreement_delta:+.3f}",
                round=2,
            ),
            "call_count": runtime.call_count,
            "rebuttal_delta": delta,
        }

    return rebuttal_node


def make_moderator_node(
    gateway: PanelLLMGateway,
) -> StateNode[PanelGraphState, None]:
    """Create the moderator synthesis node (round 2b/3/4).

    Supports both first-pass and redo (when critic rejected).
    Tracks ``moderator_redo_count`` for exhaust detection.
    """

    async def moderator_node(state: PanelGraphState) -> dict[str, Any]:
        from mindflow.agents.orchestrator import (  # noqa: PLC0415
            _PANEL_RUNTIME,
            _build_moderator_redo_prompt,
            _build_moderator_user_prompt,
            _parse_verdict,
            _verdict_summary,
        )

        runtime = _PANEL_RUNTIME.get()
        is_redo: bool = state.get("moderator_redo_count", 0) > 0
        analyst: ExpertOpinion | None = state.get("analyst_opinion")
        conflict: ConflictReport | None = state.get("conflict_report")
        assert analyst is not None
        assert conflict is not None

        if is_redo:
            round_num = 4
            prompt = _build_moderator_redo_prompt(
                state["bundle_json"],
                analyst,
                state["attribution_opinions"],
                conflict,
                cast(CriticResult, state.get("critic_result")).issues,
            )
        else:
            round_num = 2 if not state.get("escalated", False) else 3
            prompt = _build_moderator_user_prompt(
                state["bundle_json"],
                analyst,
                state["attribution_opinions"],
                conflict,
                state.get("disagreement_summary"),
            )

        logger.info(
            "Panel round {}: Moderator (redo_count={})",
            round_num,
            state.get("moderator_redo_count", 0),
        )
        raw = await _call_with_budget(runtime, gateway, MODERATOR, prompt)
        verdict = _parse_verdict(raw)
        if verdict is None:
            raise PanelUnavailableError(
                reason="主持人输出解析失败",
                call_count=runtime.call_count,
            )

        runtime.transcript.append(
            TranscriptEntry(
                role=MODERATOR.role,
                content=_verdict_summary(verdict),
                round=round_num,
            ),
        )

        return {
            "moderator_verdict": verdict,
            "transcript": TranscriptEntry(
                role=MODERATOR.role,
                content=_verdict_summary(verdict),
                round=round_num,
            ),
            "call_count": runtime.call_count,
        }

    return moderator_node


def make_critic_node(
    gateway: PanelLLMGateway,
) -> StateNode[PanelGraphState, None]:
    """Create the critic validation node (round 3/4/5)."""

    async def critic_node(state: PanelGraphState) -> dict[str, Any]:
        from mindflow.agents.orchestrator import (  # noqa: PLC0415
            _PANEL_RUNTIME,
            _build_critic_user_prompt,
            _critic_summary,
            _parse_critic,
        )
        from mindflow.services.panel_service import (  # noqa: PLC0415
            analysis_dict_to_panel_verdict,
        )

        runtime = _PANEL_RUNTIME.get()
        base_round = 2 if not state.get("escalated", False) else 3
        round_num = base_round + 1 + state.get("moderator_redo_count", 0)

        logger.info("Panel round {}: Critic", round_num)

        pending_verdict = analysis_dict_to_panel_verdict(
            cast(dict[str, Any], state.get("moderator_verdict")),
            escalated=state.get("escalated", False),
            transcript=tuple(runtime.transcript),
            call_count=runtime.call_count,
            source="panel",
        )

        all_opinions: list[ExpertOpinion] = [
            cast(ExpertOpinion, state.get("analyst_opinion")),
            *state.get("attribution_opinions", ()),
        ]
        prompt = _build_critic_user_prompt(
            state["bundle_json"],
            pending_verdict,
            all_opinions,
            state["valid_metrics"],
        )
        raw = await _call_with_budget(runtime, gateway, CRITIC, prompt)
        result = _parse_critic(raw)

        runtime.transcript.append(
            TranscriptEntry(
                role=CRITIC.role,
                content=_critic_summary(result),
                round=round_num,
            ),
        )

        updates: dict[str, Any] = {
            "critic_result": result,
            "transcript": TranscriptEntry(
                role=CRITIC.role,
                content=_critic_summary(result),
                round=round_num,
            ),
            "call_count": runtime.call_count,
        }
        if not result.approved:
            updates["moderator_redo_count"] = (
                state.get("moderator_redo_count", 0) + 1
            )
            updates["critic_retries"] = (
                state.get("critic_retries", 0) + 1
            )
        return updates

    return critic_node


# ═══════════════════════════════════════════════════════════════════════════════
# Fan-out
# ═══════════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════════
# Routers
# ═══════════════════════════════════════════════════════════════════════════════


def minimum_valid_opinion_router(state: PanelGraphState) -> str:
    """Route: >= 2 valid opinions → valid, else → unavailable."""
    opinions: tuple[ExpertOpinion, ...] = state.get("attribution_opinions", ())
    non_skipped = [o for o in opinions if not o.skipped]
    if len(non_skipped) < 2:
        logger.warning(
            "minimum_valid_opinion_router: only {} valid opinions (need ≥2)",
            len(non_skipped),
        )
        return "unavailable"
    return "valid"


def conflict_router(state: PanelGraphState) -> str:
    """Route: escalated → rebuttal, else → moderator."""
    return "rebuttal" if state.get("escalated", False) else "moderator"


def critic_verdict(state: PanelGraphState) -> str:
    """Route: approved→END, rejected+redo<2→retry, exhausted→END."""
    if cast(CriticResult, state.get("critic_result")).approved:
        return "approved"
    if state.get("moderator_redo_count", 0) < 2:
        return "retry"
    return "exhausted"


# ═══════════════════════════════════════════════════════════════════════════════
# Human-review interrupt (optional, disabled by default)
# ═══════════════════════════════════════════════════════════════════════════════


async def verdict_schema_validation_node(state: PanelGraphState) -> dict[str, Any]:
    """Deterministically validate moderator JSON before the critic LLM call."""
    from mindflow.agents.orchestrator import validate_verdict_schema  # noqa: PLC0415

    verdict = state.get("moderator_verdict")
    if verdict is None:
        return {}
    issues = validate_verdict_schema(verdict)
    if issues:
        raise PanelUnavailableError(
            reason="主持人输出 schema 校验失败：" + "; ".join(issues),
            call_count=state.get("call_count", 0),
        )
    return {}


async def human_review_interrupt_node(state: PanelGraphState) -> dict[str, Any]:
    """Optional human review gate — disabled by default."""
    from langgraph.types import interrupt as lg_interrupt  # noqa: PLC0415

    from mindflow.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    if not settings.human_review_enabled:
        return {}  # no-op when disabled

    verdict = state.get("moderator_verdict")
    if verdict is None:
        return {}

    confidence_vals: list[float] = list(verdict.get("confidence", {}).values())
    min_conf = min(confidence_vals) if confidence_vals else 1.0

    ds = state.get("disagreement_summary")
    disagreement_strength = ds.agreement_strength if ds is not None else 1.0

    if min_conf < settings.human_review_confidence_threshold or (
        ds is not None
        and (1.0 - disagreement_strength) > settings.human_review_disagreement_threshold
    ):
        logger.info(
            "Human review gate triggered: min_conf={:.2f}, disagreement={:.2f}",
            min_conf,
            1.0 - disagreement_strength if ds else 0.0,
        )
        review = lg_interrupt({
            "verdict": verdict,
            "min_confidence": min_conf,
            "disagreement": 1.0 - disagreement_strength if ds else 0.0,
        })
        if isinstance(review, dict) and review.get("action") == "reject":
            logger.info("Human reviewer rejected verdict — raising unavailable")
            raise PanelUnavailableError(
                reason="人工审核驳回裁决",
                call_count=state.get("call_count", 0),
            )

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════


def _opinion_trace(opinion: Any) -> dict[str, Any]:
    """Convert an ExpertOpinion into a JSON-safe trace entry."""
    return {
        "role": getattr(opinion, "role", ""),
        "perspective": getattr(opinion, "perspective", ""),
        "attribution_types": list(getattr(opinion, "attribution_types", ())),
        "confidence": dict(getattr(opinion, "confidence", {})),
        "evidence_citations": list(getattr(opinion, "evidence_citations", ())),
        "argument": getattr(opinion, "argument", ""),
        "raw_json": getattr(opinion, "raw_json", None),
        "skipped": bool(getattr(opinion, "skipped", False)),
    }

class PanelGraph:
    """Build and hold a compiled LangGraph StateGraph for panel deliberation.

    The graph structure is static across all invocations; node closures
    capture only ``gateway``.  Per-call data flows through the state
    channels, so the compiled graph is safe to reuse concurrently.

    Args:
        gateway: The LLM gateway for calling experts.
    """

    def __init__(self, gateway: PanelLLMGateway) -> None:
        self._gateway = gateway
        self._compiled: CompiledStateGraph[Any, Any, Any, Any] | None = None

    def build(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Build and compile the LangGraph StateGraph.

        Returns:
            A compiled LangGraph graph ready for ``ainvoke``.
        """
        graph = StateGraph(PanelGraphState)

        # ── Node factories ──────────────────────────────────────────
        graph.add_node("analyst", make_analyst_node(self._gateway))
        graph.add_node("parse_validation", parse_validation_node)
        graph.add_node("citation_validation", citation_validation_node)
        graph.add_node("forbidden_word_validation", forbidden_word_validation_node)
        graph.add_node("attribution", make_attribution_node(self._gateway))
        graph.add_node("conflict_detection", conflict_detection_node)
        graph.add_node("rebuttal", make_rebuttal_node(self._gateway))
        graph.add_node("moderator", make_moderator_node(self._gateway))
        graph.add_node("verdict_schema_validation", verdict_schema_validation_node)
        graph.add_node("human_review_interrupt", human_review_interrupt_node)
        graph.add_node("critic", make_critic_node(self._gateway))

        # ── Wiring ─────────────────────────────────────────────────
        graph.set_entry_point("analyst")

        # Analyst → validation chain
        graph.add_edge("analyst", "parse_validation")
        graph.add_edge("parse_validation", "citation_validation")
        graph.add_edge("citation_validation", "forbidden_word_validation")

        # Post-validation routing: on the main (first) pass go to
        # attribution; on the rebuttal re-validation pass go directly
        # to moderator (ADR: escalated + rebuttal_delta distinguishes
        # the two passes — these are set by conflict_detection and
        # rebuttal before re-entering the validation chain).
        def _post_validation_router(state: PanelGraphState) -> str:
            if state.get("escalated", False) and state.get("rebuttal_delta") is not None:
                return "moderator"
            return "attribution"

        graph.add_conditional_edges(
            "forbidden_word_validation",
            _post_validation_router,
            {"attribution": "attribution", "moderator": "moderator"},
        )

        # Attribution → conflict detection
        graph.add_edge("attribution", "conflict_detection")

        # Combined routing: minimum-valid-opinion gate + conflict router
        def _panel_routing(state: PanelGraphState) -> str:
            slack = minimum_valid_opinion_router(state)
            if slack == "unavailable":
                return "unavailable"
            return conflict_router(state)

        graph.add_conditional_edges(
            "conflict_detection",
            _panel_routing,
            {
                "unavailable": END,
                "rebuttal": "rebuttal",
                "moderator": "moderator",
            },
        )

        # Rebuttal → validation chain → moderator
        graph.add_edge("rebuttal", "parse_validation")

        # Moderator → human_review_interrupt → critic
        graph.add_edge("moderator", "verdict_schema_validation")
        graph.add_edge("verdict_schema_validation", "human_review_interrupt")
        graph.add_edge("human_review_interrupt", "critic")

        # Critic routing (retry→moderator on rejection, exhaust at redo_count≥2)
        graph.add_conditional_edges(
            "critic",
            critic_verdict,
            {
                "approved": END,
                "retry": "moderator",
                "exhausted": END,
            },
        )

        # Checkpointer only needed when human_review_enabled is True.
        # When disabled (default), compile without one to avoid serializing
        # non-checkpointable state (e.g., asyncio.Lock in _PanelRunContext).
        from mindflow.config import get_settings

        checkpointer = MemorySaver() if get_settings().human_review_enabled else None
        self._compiled = graph.compile(checkpointer=checkpointer)
        return self._compiled

    async def ainvoke(
        self,
        state: PanelGraphState,
    ) -> dict[str, Any]:
        """Invoke the panel deliberation graph with per-call runtime isolation.

        Creates a ``_PanelRunContext``, sets the ``_PANEL_RUNTIME``
        ContextVar so graph nodes can track budget and transcript
        without including mutable state in the checkpointable graph state,
        then invokes the compiled LangGraph graph.

        The ContextVar is reset in a ``finally`` block so concurrent
        invocations cannot leak runtime between calls.

        Args:
            state: The initial ``PanelGraphState`` for this invocation.

        Returns:
            The final graph state dict, augmented with ``call_count``
            and ``transcript`` from the per-call runtime context.

        Raises:
            PanelUnavailableError: From graph nodes (e.g. insufficient
                valid opinions, moderator parse failure) OR translated
                from gateway unavailability (GatewayNotConfiguredError,
                GatewayAPIError) with causal chaining.
            PanelBudgetExceededError: Propagated from graph nodes when
                the 12-call budget is exceeded.
        """
        # Lazy import to avoid circular dependency: graph nodes already
        # import _PANEL_RUNTIME from orchestrator, so we set/reset the
        # same ContextVar here.
        from mindflow.agents.orchestrator import (  # noqa: PLC0415
            _PANEL_RUNTIME as _ORCH_RUNTIME,
        )
        from mindflow.agents.orchestrator import (  # noqa: PLC0415
            _PanelRunContext as _OrchRunContext,
        )

        runtime = _OrchRunContext()
        token = _ORCH_RUNTIME.set(runtime)
        try:
            result = await self.compiled.ainvoke(state)
        except GatewayNotConfiguredError as exc:
            raise PanelUnavailableError(
                reason=str(exc),
                call_count=runtime.call_count,
            ) from exc
        except GatewayAPIError as exc:
            raise PanelUnavailableError(
                reason=str(exc),
                call_count=runtime.call_count,
            ) from exc
        finally:
            _ORCH_RUNTIME.reset(token)

        # Augment the graph result with the per-call context values.
        # Graph nodes write call_count/transcript into state channels,
        # but the runtime holds the authoritative cumulative counters.
        if isinstance(result, dict):
            result["call_count"] = runtime.call_count
            result["transcript"] = tuple(runtime.transcript)
            trace: list[dict[str, Any]] = []
            analyst = result.get("analyst_opinion")
            if analyst is not None:
                trace.append({"node": "analyst", "type": "opinion", **(_opinion_trace(analyst))})
            for opinion in result.get("attribution_opinions", ()):
                trace.append(
                    {"node": "attribution", "type": "opinion", **(_opinion_trace(opinion))}
                )
            if result.get("moderator_verdict") is not None:
                trace.append(
                    {"node": "moderator", "type": "verdict", "payload": result["moderator_verdict"]}
                )
            if result.get("critic_result") is not None:
                critic = result["critic_result"]
                trace.append(
                    {
                        "node": "critic",
                        "type": "critic",
                        "approved": bool(critic.approved),
                        "issues": list(critic.issues),
                    }
                )
            result["trace"] = trace

        return result

    @property
    def compiled(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Return the compiled graph, building it on first access (lazy)."""
        if self._compiled is None:
            self._compiled = self.build()
        return self._compiled
