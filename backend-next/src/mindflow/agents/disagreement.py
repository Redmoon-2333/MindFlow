"""Structured disagreement analytics for multi-expert panel deliberation.

Extends ``conflict.py`` with metrics inspired by "Disagreement as Data: Reasoning
Trace Analytics in Multi-Agent Systems" (Borchers et al., ACM 2026) and "Adaptive
Stability Detection in Multi-Agent LLM Debate" (Hu et al., NeurIPS 2026).

Adds four dimensions beyond binary conflict detection:
  1. **Disagreement type classification** — type-mismatch / confidence-gap / evidence-divergence
  2. **Agreement strength** — Jaccard similarity of cited evidence + weighted type overlap
  3. **Stability tracking** — whether opinions converge after rebuttal rounds
  4. **Rebuttal delta** — what changed (confidence shifts, type additions/drops) after debate
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mindflow.agents.types import ExpertOpinion

# ── Disagreement type enum ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class DisagreementTypes:
    """Classified disagreement dimensions."""

    type_mismatch: bool = False
    confidence_gap: bool = False
    evidence_divergence: bool = False
    theoretical_disagreement: bool = False


# ── Agreement strength metrics ───────────────────────────────────────────────────


def _jaccard_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Jaccard similarity between two sets represented as tuples."""
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _evidence_overlap_score(opinions: Sequence[ExpertOpinion]) -> float:
    """Compute mean pairwise Jaccard similarity of evidence citations."""
    active = [o for o in opinions if not o.skipped]
    if len(active) < 2:
        return 1.0

    scores: list[float] = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            scores.append(
                _jaccard_similarity(
                    active[i].evidence_citations,
                    active[j].evidence_citations,
                )
            )

    return sum(scores) / len(scores) if scores else 1.0


def _type_overlap_score(opinions: Sequence[ExpertOpinion]) -> float:
    """Weighted type overlap: mean pairwise Jaccard of attribution types."""
    active = [o for o in opinions if not o.skipped]
    if len(active) < 2:
        return 1.0

    scores: list[float] = []
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            scores.append(
                _jaccard_similarity(
                    active[i].attribution_types,
                    active[j].attribution_types,
                )
            )

    return sum(scores) / len(scores) if scores else 1.0


def _confidence_overlap_score(opinions: Sequence[ExpertOpinion]) -> float:
    """Compute weighted confidence agreement: for shared types, 1-gap; for unshared, 0."""
    active = [o for o in opinions if not o.skipped]
    if len(active) < 2:
        return 1.0

    # Collect all unique types
    all_types: set[str] = set()
    for o in active:
        all_types.update(o.confidence.keys())
    if not all_types:
        return 1.0

    type_scores: list[float] = []
    for t in sorted(all_types):
        confs = [o.confidence.get(t, 0.0) for o in active]
        max_gap = max(confs) - min(confs)
        type_scores.append(1.0 - max_gap)

    return sum(type_scores) / len(type_scores) if type_scores else 1.0


def compute_agreement_strength(opinions: Sequence[ExpertOpinion]) -> float:
    """Compute a composite agreement strength score in [0, 1].

    Weights: 40% type overlap + 30% evidence overlap + 30% confidence overlap.
    Higher values indicate stronger consensus among experts.

    Args:
        opinions: Attribution expert opinions (2-3 experts).

    Returns:
        Agreement strength score in [0, 1].
    """
    type_score = _type_overlap_score(opinions)
    evidence_score = _evidence_overlap_score(opinions)
    confidence_score = _confidence_overlap_score(opinions)

    return round(
        type_score * 0.40 + evidence_score * 0.30 + confidence_score * 0.30,
        4,
    )


# ── Disagreement classification ──────────────────────────────────────────────────


def classify_disagreement(
    opinions: Sequence[ExpertOpinion],
    max_confidence_gap: float,
) -> DisagreementTypes:
    """Classify the type(s) of disagreement among experts.

    Args:
        opinions: Attribution expert opinions.
        max_confidence_gap: Pre-computed max confidence gap across any shared type.

    Returns:
        ``DisagreementTypes`` indicating which disagreement dimensions are present.
    """
    active = [o for o in opinions if not o.skipped]
    if len(active) < 2:
        return DisagreementTypes()

    # Criterion 1: Type mismatch
    top_sets = {o.attribution_types[0] for o in active if o.attribution_types}
    type_mismatch = len(top_sets) > 1

    # Criterion 2: Confidence gap
    confidence_gap = max_confidence_gap > 0.3

    # Criterion 3: Evidence divergence — low citation overlap
    evidence_divergence = _evidence_overlap_score(active) < 0.3

    # Criterion 4: Theoretical disagreement — different experts cite
    # fundamentally different metrics, indicating different frameworks
    all_citations: set[str] = set()
    for o in active:
        all_citations.update(o.evidence_citations)
    theoretical_disagreement = len(all_citations) > 6 and evidence_divergence

    return DisagreementTypes(
        type_mismatch=type_mismatch,
        confidence_gap=confidence_gap,
        evidence_divergence=evidence_divergence,
        theoretical_disagreement=theoretical_disagreement,
    )


# ── Rebuttal delta tracking ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class RebuttalDelta:
    """Tracks what changed between pre-rebuttal and post-rebuttal opinions.

    Attributes:
        before_agreement: Agreement strength before rebuttal.
        after_agreement: Agreement strength after rebuttal.
        agreement_delta: Positive = convergence, negative = divergence.
        confidence_shifts: Per-expert mean confidence change.
        type_changes: List of (expert, added_types, dropped_types) tuples.
        converged: True if agreement improved by >= 0.1.
    """

    before_agreement: float
    after_agreement: float
    agreement_delta: float
    confidence_shifts: tuple[float, ...]
    type_changes: tuple[str, ...]
    converged: bool


def compute_rebuttal_delta(
    before: Sequence[ExpertOpinion],
    after: Sequence[ExpertOpinion],
) -> RebuttalDelta:
    """Compute the delta between pre- and post-rebuttal opinions.

    Measures whether multi-agent debate led to convergence (desirable)
    or divergence/entrenchment (needs attention).

    Args:
        before: Expert opinions from round 1 (initial attribution).
        after: Expert opinions from round 2a (post-rebuttal).

    Returns:
        ``RebuttalDelta`` with quantitative convergence metrics.
    """
    before_active = [o for o in before if not o.skipped]
    after_active = [o for o in after if not o.skipped]

    before_agreement = compute_agreement_strength(before_active)
    after_agreement = compute_agreement_strength(after_active)

    # Confidence shift per expert
    shifts: list[float] = []
    for b_op, a_op in zip(before, after, strict=False):
        if b_op.skipped or a_op.skipped:
            continue
        shared = set(b_op.confidence.keys()) & set(a_op.confidence.keys())
        if shared:
            shift = sum(
                abs(a_op.confidence[t] - b_op.confidence.get(t, 0.0))
                for t in shared
            )
            shifts.append(round(shift / len(shared), 4))

    # Type changes
    type_parts: list[str] = []
    for i, (b_op, a_op) in enumerate(zip(before, after, strict=False)):
        b_set = set(b_op.attribution_types)
        a_set = set(a_op.attribution_types)
        added = a_set - b_set
        dropped = b_set - a_set
        if added or dropped:
            parts = []
            if added:
                parts.append(f"+{', '.join(added)}")
            if dropped:
                parts.append(f"-{', '.join(dropped)}")
            type_parts.append(f"Expert{i + 1}: {'; '.join(parts)}")

    delta_value = round(after_agreement - before_agreement, 4)
    return RebuttalDelta(
        before_agreement=before_agreement,
        after_agreement=after_agreement,
        agreement_delta=delta_value,
        confidence_shifts=tuple(shifts),
        type_changes=tuple(type_parts),
        converged=delta_value >= 0.1,
    )


# ── Disagreement summary (human-readable + structured) ───────────────────────────


@dataclass(frozen=True)
class DisagreementSummary:
    """Full disagreement analytics report for a panel deliberation.

    Attributes:
        agreement_strength: Composite score in [0, 1].
        disagreement_types: Classified disagreement dimensions.
        evidence_overlap: Jaccard similarity of Cited evidence.
        type_consensus: Whether all experts agree on top-1 type.
        stability: "stable", "converged", "entrenched", or "not_applicable".
        recommendations: Human-readable Chinese analysis of disagreement patterns.
    """

    agreement_strength: float
    disagreement_types: DisagreementTypes
    evidence_overlap: float
    type_consensus: bool
    stability: str
    recommendations: str = ""
    delta: RebuttalDelta | None = None


def analyze_disagreement(
    opinions: Sequence[ExpertOpinion],
    conflict_details: str,
    max_confidence_gap: float,
    rebuttal_delta: RebuttalDelta | None = None,
) -> DisagreementSummary:
    """Produce a full disagreement analytics report.

    Args:
        opinions: Attribution expert opinions (post-rebuttal if escalated).
        conflict_details: Human-readable conflict description from
            ``ConflictReport.details``.
        max_confidence_gap: From ``ConflictReport.max_confidence_gap``.
        rebuttal_delta: If escalation happened, the rebuttal delta.

    Returns:
        ``DisagreementSummary`` for logging and transcript enrichment.
    """
    strength = compute_agreement_strength(opinions)
    types = classify_disagreement(opinions, max_confidence_gap)
    evidence_overlap = _evidence_overlap_score(opinions)

    # Type consensus
    active = [o for o in opinions if not o.skipped]
    top_sets = {o.attribution_types[0] for o in active if o.attribution_types}
    type_consensus = len(top_sets) <= 1

    # Stability
    if rebuttal_delta is None:
        stability = "stable" if type_consensus else "contested"
    elif rebuttal_delta.converged:
        stability = "converged"
    elif rebuttal_delta.agreement_delta < -0.05:
        stability = "entrenched"
    else:
        stability = "stable"

    # Recommendations
    rec_lines: list[str] = []
    if types.theoretical_disagreement:
        rec_lines.append(
            "专家引用不同证据体系，可能存在理论框架层面的分歧——建议增加"
            "跨视角证据桥接，或由主持人明确指定分析维度优先级"
        )
    if types.confidence_gap:
        rec_lines.append(
            "同类型置信度差距较大——建议主持人在裁决时优先采纳有强证据"
            "支持的判断，并记录低置信度观点为 dissent"
        )
    if types.evidence_divergence:
        rec_lines.append(
            "专家引用的证据指标重叠度低——检查是否因数据分析师报告覆盖不完全"
            "导致某些维度信息不足"
        )
    if rebuttal_delta and not rebuttal_delta.converged:
        rec_lines.append(
            f"辩论后一致度变化{rebuttal_delta.agreement_delta:+.3f}，"
            "专家观点未收敛——建议接受分歧并明确标记为 dissent"
        )
    if not any(vars(types).values()):
        rec_lines.append("专家意见高度一致，归因结论可信度高")

    return DisagreementSummary(
        agreement_strength=strength,
        disagreement_types=types,
        evidence_overlap=round(evidence_overlap, 4),
        type_consensus=type_consensus,
        stability=stability,
        recommendations="；".join(rec_lines),
        delta=rebuttal_delta,
    )
