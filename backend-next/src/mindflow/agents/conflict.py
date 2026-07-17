"""Conflict detection for expert panel deliberation (07-agent-upgrade-design.md §4).

A pure-function module that detects disagreements among attribution experts.
Zero LLM calls, zero IO — purely comparing structured ``ExpertOpinion`` data.

Conflict criteria (from §4 "冲突升级"):
  1. **Top-1 type mismatch**: The highest-confidence procrastination type
     differs across experts.
  2. **Same-type confidence gap > 0.3**: Two experts agree on the top type
     but their confidence differs by more than 0.3.

Either criterion triggers escalation (a rebuttal round).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mindflow.agents.types import ExpertOpinion


@dataclass(frozen=True)
class ConflictReport:
    """Result of conflict detection among attribution experts.

    Attributes:
        has_conflict: True if either criterion is met.
        top_types: The top-1 type from each non-skipped expert, in order.
        max_confidence_gap: The maximum confidence gap for any shared type
            across any pair of experts.
        details: Human-readable Chinese explanation of the conflict(s).
    """

    has_conflict: bool
    top_types: tuple[str | None, ...]
    max_confidence_gap: float
    details: str


def _get_top_type(opinion: ExpertOpinion) -> str | None:
    """Return the top-1 attribution type from an opinion, or None if empty."""
    if not opinion.attribution_types or opinion.skipped:
        return None
    return opinion.attribution_types[0]


def _max_confidence_gap(opinions: Sequence[ExpertOpinion]) -> float:
    """Compute the maximum confidence gap for any shared type across any pair.

    For every procrastination type that appears in two or more opinions,
    compute the max difference across any pair. Return the largest such gap.
    """
    # Collect confidence per type per opinion index
    type_values: dict[str, list[float]] = {}

    for opinion in opinions:
        if opinion.skipped:
            continue
        for type_name, conf in opinion.confidence.items():
            if type_name not in type_values:
                type_values[type_name] = []
            type_values[type_name].append(conf)

    max_gap = 0.0
    for _type_name, values in type_values.items():
        if len(values) >= 2:
            gap = max(values) - min(values)
            if gap > max_gap:
                max_gap = gap

    return max_gap


def detect_conflict(opinions: Sequence[ExpertOpinion]) -> ConflictReport:
    """Detect conflicts among attribution expert opinions.

    Args:
        opinions: The opinions from 2-3 attribution experts.

    Returns:
        A ``ConflictReport`` with the detection result.

    Raises:
        ValueError: If fewer than 2 non-skipped opinions are provided.
    """
    non_skipped = [o for o in opinions if not o.skipped]

    if len(non_skipped) < 2:
        # Not enough opinions to detect conflict — no escalation possible
        return ConflictReport(
            has_conflict=False,
            top_types=tuple(_get_top_type(o) for o in opinions),
            max_confidence_gap=0.0,
            details="不足以检测冲突（有效意见不足2份）",
        )

    # Criterion 1: Top-1 type mismatch
    top_types = tuple(_get_top_type(o) for o in non_skipped)
    unique_top_types = {t for t in top_types if t is not None}
    top_type_mismatch = len(unique_top_types) > 1

    # Criterion 2: Same-type confidence gap > 0.3
    # Round to 6 decimal places to avoid IEEE 754 artifacts
    # (e.g. 0.80 - 0.50 = 0.30000000000000004)
    gap = round(_max_confidence_gap(non_skipped), 6)
    confidence_gap_exceeded = gap > 0.3

    has_conflict = top_type_mismatch or confidence_gap_exceeded

    # Build details
    details_parts: list[str] = []
    if top_type_mismatch:
        types_str = ", ".join(str(t) for t in unique_top_types if t is not None)
        details_parts.append(f"专家之间主要拖延类型不一致：{types_str}")
    if confidence_gap_exceeded:
        details_parts.append(f"同类型置信度差距超过0.3（最大差距={gap:.2f}）")

    details = "；".join(details_parts) if details_parts else "专家意见一致，无冲突"

    return ConflictReport(
        has_conflict=has_conflict,
        top_types=tuple(_get_top_type(o) for o in opinions),
        max_confidence_gap=gap,
        details=details,
    )
