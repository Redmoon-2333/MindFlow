"""EvidenceFact — normalized evidence catalog for LLM expert citation.

Creates a stable, canonical ID namespace between ``EvidenceBundle`` and
LLM expert prompts. Every citeable fact has exactly one ID, regardless of
which JSON field or ``EvidenceItem`` it originates from.

Usage::

    from mindflow.domain.evidence_facts import build_evidence_catalog
    catalog = build_evidence_catalog(bundle)
    # catalog[0].id == "focus.score"
    # catalog[0].value == 16.3
    # catalog[0].label_zh == "专注度评分"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mindflow.domain.evidence import EvidenceBundle

JSONValue = str | int | float | bool | list[str] | None


@dataclass(frozen=True)
class EvidenceFact:
    """A single citeable evidence fact with a stable canonical ID.

    Attributes:
        id: Canonical, domain-oriented identifier (e.g. ``"focus.score"``).
            ASCII-only so it works consistently inside Chinese prompts.
        value: The fact's value.
        label_zh: Chinese label for display and prompt context.
        source_refs: Tuple of source field paths for debugging/UI traceability.
        confidence: Optional confidence in [0, 1].
    """

    id: str
    value: JSONValue
    label_zh: str
    source_refs: tuple[str, ...] = ()
    confidence: float | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Catalog builder
# ═══════════════════════════════════════════════════════════════════════════════


def _severity_zh(severity: str) -> str:
    _m = {"info": "正常", "mild": "轻度", "moderate": "中度", "severe": "严重"}
    return _m.get(severity, severity)


def build_evidence_catalog(bundle: EvidenceBundle) -> tuple[EvidenceFact, ...]:
    """Build an evidence fact catalog from an ``EvidenceBundle``.

    Each fact has a stable canonical ``id`` that is independent of the
    underlying JSON field name or ``EvidenceItem.metric`` value.

    Returns:
        A tuple of ``EvidenceFact`` sorted by severity (severe first).
    """
    facts: list[EvidenceFact] = []
    summary = bundle.behavior_summary

    # ── 1. Evidence items (direct ML/production metrics) ─────────────────
    for item in bundle.items:
        # Canonical ID: use focus.* prefix for consistency
        canonical_prefix = "focus."
        if item.metric.startswith("ml_"):
            canonical_prefix = "ml."
        fid = canonical_prefix + item.metric
        facts.append(EvidenceFact(
            id=fid,
            value=item.value if isinstance(item.value, (int, float, str)) else str(item.value),
            label_zh=item.human_readable,
            source_refs=(f"evidence.{item.metric}",),
            confidence=item.confidence,
        ))

    # ── 2. Behavior summary fields ──────────────────────────────────────
    if summary.duration_min > 0:
        facts.append(EvidenceFact(
            id="summary.duration_min",
            value=summary.duration_min,
            label_zh=f"分析窗口总时长 {summary.duration_min:.0f} 分钟",
            source_refs=("behavior_summary.duration_min",),
        ))
        facts.append(EvidenceFact(
            id="summary.actual_focus_min",
            value=summary.actual_focus_min,
            label_zh=f"实际专注时长 {summary.actual_focus_min:.0f} 分钟",
            source_refs=("behavior_summary.actual_focus_min",),
        ))
        facts.append(EvidenceFact(
            id="summary.context_switches_per_hour",
            value=summary.context_switches_per_hour,
            label_zh=f"上下文切换频率 {summary.context_switches_per_hour:.1f} 次/小时",
            source_refs=("behavior_summary.context_switches_per_hour",),
        ))
        facts.append(EvidenceFact(
            id="summary.longest_focus_block_seconds",
            value=summary.longest_focus_block_s,
            label_zh=f"最长连续专注 {summary.longest_focus_block_s:.0f} 秒",
            source_refs=("behavior_summary.longest_focus_block_s",),
        ))
        facts.append(EvidenceFact(
            id="summary.social_media_ratio",
            value=summary.social_media_ratio,
            label_zh=f"社交媒体占比 {summary.social_media_ratio:.0%}",
            source_refs=("behavior_summary.social_media_ratio",),
        ))
        facts.append(EvidenceFact(
            id="summary.start_delay_min",
            value=summary.start_delay_min,
            label_zh=f"启动延迟 {summary.start_delay_min:.0f} 分钟",
            source_refs=("behavior_summary.start_delay_min",),
        ))
        if summary.baseline_deviation is not None:
            facts.append(EvidenceFact(
                id="summary.baseline_deviation",
                value=summary.baseline_deviation,
                label_zh=f"行为基线偏差 {summary.baseline_deviation:.2f}σ",
                source_refs=("behavior_summary.baseline_deviation",),
            ))

    # ── 3. Intervention history (latest response) ───────────────────────
    if bundle.intervention_history:
        latest = max(
            (r for r in bundle.intervention_history if r.triggered_at.tzinfo is not None),
            key=lambda r: r.triggered_at,
            default=bundle.intervention_history[-1],
        )
        facts.append(EvidenceFact(
            id="intervention.latest_type",
            value=latest.intervention_type,
            label_zh=f"最近干预类型: {latest.intervention_type}",
            source_refs=("intervention_history.latest.type",),
        ))
        facts.append(EvidenceFact(
            id="intervention.latest_response",
            value=latest.user_response or "未回应",
            label_zh=f"最近干预响应: {latest.user_response or '未回应'}",
            source_refs=("intervention_history.latest.user_response",),
        ))
        facts.append(EvidenceFact(
            id="intervention.latest_effect",
            value=latest.effect_note,
            label_zh=f"最近干预效果: {latest.effect_note}",
            source_refs=("intervention_history.latest.effect_note",),
        ))

    # ── 4. Novelty flags (stable aggregate ID, not per-flag) ────────────
    if bundle.novelty_flags:
        facts.append(EvidenceFact(
            id="novelty.flags",
            value=list(bundle.novelty_flags),
            label_zh=f"新奇行为模式: {'; '.join(bundle.novelty_flags)}",
            source_refs=("novelty_flags",),
        ))

    # Sort: severe first, then by label_zh
    severity_rank = {"severe": 0, "moderate": 1, "mild": 2, "info": 3}

    def _sort_key(f: EvidenceFact) -> tuple[int, str]:
        sev = "info"
        for item in bundle.items:
            if (f.id.endswith(item.metric)
                    or f"focus.{item.metric}" == f.id
                    or f"ml.{item.metric}" == f.id):
                sev = item.severity
                break
        return (severity_rank.get(sev, 3), f.label_zh)

    facts.sort(key=_sort_key)
    return tuple(facts)


def evidence_catalog_ids(catalog: tuple[EvidenceFact, ...]) -> frozenset[str]:
    """Return the set of all valid evidence IDs for citation validation."""
    return frozenset(fact.id for fact in catalog)
