"""Serializable graph state contracts for MindFlow LangGraph orchestration.

Defines three state types that are framework-neutral — they carry zero
LangGraph/LangChain imports so they can be constructed and serialized
without a graph engine:

  - ``AnalysisState``: Top-level analysis orchestration state, tracking
    degradation path and evidence data.
  - ``PanelState``: Deliberation state for the multi-expert panel sub-graph
    with order-independent opinion accumulation.
  - ``ChatState``: Conversation state for the chat agent sub-graph
    with tool message accumulation and crisis gating.

Design constraints:
  - Frozen dataclasses throughout (matching ``domain/`` and ``agents/types.py``).
  - Every field is JSON/checkpointer-serializable: ``int``, ``float``, ``str``,
    ``bool``, ``None``, ``tuple``, ``frozenset``, ``dict[str, object]``, or
    stable domain value objects (``ExpertOpinion``, ``PanelVerdict``, etc.).
  - No ``asyncio.Lock``, model clients, repositories, ``ContextVar``, or
    exception objects in state fields.
  - Each state carries ``graph_version: int`` for migration awareness.
"""

from __future__ import annotations

from dataclasses import dataclass

from mindflow.agents.conflict import ConflictReport
from mindflow.agents.disagreement import DisagreementSummary, RebuttalDelta
from mindflow.agents.types import (
    CriticResult,
    ExpertOpinion,
    PanelSource,
    PanelVerdict,
    TranscriptEntry,
)

# ═══════════════════════════════════════════════════════════════════════════════
# AnalysisState — top-level orchestration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AnalysisState:
    """Top-level state flowing through the analysis orchestration graph.

    Tracks which degradation tier produced the result, the evidence data
    driving the analysis, and the panel sub-state for deliberation.

    Attributes:
        panel: The current panel deliberation state, or None before panel runs.
        evidence_data: Serialized evidence bundle as a plain dict, or None.
        crisis_flag: True if a crisis (self-harm, etc.) was detected.
        degradation_path: Ordered list of tiers attempted, e.g.
            ``["panel", "single_expert"]``.
        source: The tier that ultimately produced the verdict.
        graph_version: Schema version for state migration awareness.
    """

    panel: PanelState | None = None
    evidence_data: dict[str, object] | None = None
    crisis_flag: bool = False
    degradation_path: tuple[str, ...] = ()
    source: PanelSource = "rule_engine"
    graph_version: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# PanelState — multi-expert deliberation
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PanelState:
    """Serializable deliberation state for the expert panel sub-graph.

    All fields are immutable tuples or frozen value objects so the graph
    engine can checkpoint after every node transition.

    ``expert_opinions`` is accumulated via the ``append_opinion`` reducer
    which guarantees deterministic ordering by role for parallel fan-in.

    Attributes:
        expert_opinions: Accumulated opinions from analyst and attribution
            experts, ordered deterministically by role.
        moderator_verdict: The final verdict, or None before moderator runs.
        critic_result: The critic's validation result, or None.
        conflict_report: Conflict detection result, or None.
        disagreement_summary: Structured disagreement analytics, or None.
        rebuttal_delta: Convergence metrics after rebuttal round, or None.
        transcript: Full deliberation transcript entries.
        escalated: True if conflict escalation was triggered.
        call_count: Total LLM API calls made.
        graph_version: Schema version for state migration awareness.
    """

    expert_opinions: tuple[ExpertOpinion, ...] = ()
    moderator_verdict: PanelVerdict | None = None
    critic_result: CriticResult | None = None
    conflict_report: ConflictReport | None = None
    disagreement_summary: DisagreementSummary | None = None
    rebuttal_delta: RebuttalDelta | None = None
    transcript: tuple[TranscriptEntry, ...] = ()
    escalated: bool = False
    call_count: int = 0
    graph_version: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# ChatState — conversation with tool loop
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChatState:
    """Serializable state for the chat conversation sub-graph.

    Messages are stored as plain dicts (``{"role": str, "content": str}``)
    so the entire state round-trips through ``json.dumps(asdict(state))``.

    Tool messages (calls and results) are accumulated via
    ``accumulate_tool_messages``; errors use ``accumulate_errors``.

    Attributes:
        messages: Conversation messages as role/content dicts.
        tool_messages: Accumulated tool-call and tool-result records, each
            as ``{"type": "call"|"result", "name": str, "content": str}``.
        errors: Unique error records keyed by error message, as
            ``{"key": str, "message": str}``.
        crisis_gate: True if pre-LLM crisis detection triggered.
        retry_count: Number of retry loops (forbidden word, tool error).
        graph_version: Schema version for state migration awareness.
    """

    messages: tuple[dict[str, object], ...] = ()
    tool_messages: tuple[dict[str, str], ...] = ()
    errors: tuple[dict[str, str], ...] = ()
    crisis_gate: bool = False
    retry_count: int = 0
    graph_version: int = 1
