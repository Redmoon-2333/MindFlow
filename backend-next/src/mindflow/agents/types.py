"""Expert panel data types for multi-expert deliberation kernel.

Defines the core data contracts for the L1 expert panel (07-agent-upgrade-design.md §4):

  - ``ExpertOpinion``: Output from an individual expert (analyst or attribution).
  - ``PanelVerdict``: Final verdict from the moderator, aligned with
    ``ProcrastinationAssessment`` shape for seamless integration with downstream
    intervention pipelines.
  - ``TranscriptEntry``: Single message in the deliberation transcript.
  - ``PanelBudgetExceeded`` / ``PanelUnavailableError``: Exception types for
    the degradation chain.

Design decisions:
  - Frozen dataclasses throughout (following domain/evidence.py and domain/procrastination.py).
  - Zero framework dependencies (pure stdlib only — no pydantic in domain/agents layer).
  - ``source`` field on ``PanelVerdict`` tracks which tier produced the result,
    enabling the four-layer degradation chain: panel → single_expert → ollama → rule_engine.
  - All free-text fields are Chinese. ``rationale`` and ``argument`` never contain
    "诊断", "治疗", "患者", "处方" (NF-S7 contract — enforced by critic and parsers).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType

# ── Forbidden words (NF-S7) — reused from infrastructure/llm/schemas.py ────────

FORBIDDEN_WORDS: frozenset[str] = frozenset({
    "诊断",
    "治疗",
    "患者",
    "处方",
})

# ── Source type for the four-layer degradation chain ───────────────────────────

PanelSource = Literal["panel", "single_expert", "ollama", "rule_engine"]


# ── Core data types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TranscriptEntry:
    """A single message in the panel deliberation transcript.

    Attributes:
        role: The expert role who produced this message.
        content: Summary of the message content (first ~200 chars).
        round: Which round of deliberation this entry belongs to
            (0=analyst, 1=attribution, 2=moderator/rebuttal, 3=critic, …).
    """

    role: str
    content: str
    round: int


@dataclass(frozen=True)
class ExpertOpinion:
    """Structured output from a single expert (analyst or attribution expert).

    Attributes:
        role: Expert role name in Chinese (e.g. "数据分析师", "CBT归因专家").
        perspective: Theoretical perspective label (e.g. "认知行为理论视角").
        attribution_types: 1-3 procrastination types identified by this expert,
            as lowercase strings matching ProcrastinationType values.
        confidence: Per-type confidence scores in [0, 1].
        evidence_citations: Metric names cited in the argument, for critic validation.
        argument: Full Chinese argument text. Each empirical claim must be
            followed by "[证据: metric名]".
        raw_json: The raw JSON string returned by the LLM, retained for
            debugging and transcript logging.
        skipped: True if this expert was skipped due to a parsing failure
            or forbidden-word rejection.
    """

    role: str
    perspective: str
    attribution_types: tuple[str, ...]
    confidence: Mapping[str, float]
    evidence_citations: tuple[str, ...]
    argument: str
    raw_json: str | None = None
    skipped: bool = False


@dataclass(frozen=True)
class PanelVerdict:
    """Final verdict from the expert panel deliberation.

    Fields ``types``, ``confidence``, ``recommended_technique``, and ``rationale``
    are deliberately aligned with ``ProcrastinationAssessment`` so downstream
    consumers (intervention engine, throttler, etc.) can treat both interchangeably.

    Attributes:
        types: 1-3 ProcrastinationType values sorted by confidence descending.
        confidence: Per-type confidence in [0, 1].
        recommended_technique: Primary CBT technique for the top type.
        rationale: Chinese explanation from the moderator.
        dissent: Minority opinions that were overruled, as Chinese text strings.
        transcript: Full deliberation transcript for UI display.
        escalated: True if conflict escalation was triggered.
        call_count: Total LLM API calls made during this deliberation.
        source: Which tier of the degradation chain produced this verdict.
    """

    types: tuple[ProcrastinationType, ...]
    confidence: Mapping[ProcrastinationType, float]
    recommended_technique: CBTTechnique | None
    rationale: str
    dissent: tuple[str, ...]
    transcript: tuple[TranscriptEntry, ...]
    escalated: bool
    call_count: int
    source: PanelSource


# ── Exception types (degradation chain) ────────────────────────────────────────


class PanelBudgetExceededError(RuntimeError):
    """Raised when the panel would exceed the maximum allowed LLM calls (12).

    This is a hard safety guard (07-agent-upgrade-design.md §4):
    "辩论≤1轮, 打回≤1次 → 最坏 12 次调用/会诊"
    """

    def __init__(self, call_count: int = 0) -> None:
        super().__init__(
            f"Panel budget exceeded: {call_count} calls would exceed the maximum of 12"
        )
        self.call_count = call_count


class PanelUnavailableError(RuntimeError):
    """Raised when the expert panel cannot produce a verdict.

    This signals the caller (G003 wiring layer) to fall through to the next
    degradation tier: single-expert LLM service (existing llm_service.py),
    which itself has its own L2 (Ollama) and L3 (RuleEngine) fallbacks.

    Attributes:
        reason: Human-readable Chinese explanation of why the panel failed.
        call_count: How many LLM calls were made before failure.
    """

    def __init__(self, reason: str, call_count: int = 0) -> None:
        super().__init__(f"Panel unavailable after {call_count} calls: {reason}")
        self.reason = reason
        self.call_count = call_count


# ── Critic validation result ───────────────────────────────────────────────────


@dataclass(frozen=True)
class CriticResult:
    """Output from the critic expert after validating a moderator's verdict.

    Attributes:
        approved: True if the verdict passes all checks.
        issues: List of specific issues found (empty when approved).
    """

    approved: bool
    issues: tuple[str, ...]
