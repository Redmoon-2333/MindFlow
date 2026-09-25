"""Panel fast path — decide *before* spending LLM calls (optimisation plan 2.2).

The panel is the most expensive part of an analysis run.  Most windows do not
need it: when the evidence is complete and high quality, the deterministic rule
engine is confident, and nothing conflicts, a panel deliberation cannot add much
beyond cost.  Conversely, when the evidence coverage is insufficient there is
nothing for a panel to reason about, so the honest answer is
``insufficient_data`` — again without LLM calls.

This module is deliberately a **pure decision function**.  It never calls a
model, never writes state, and makes no safety judgement of its own: every route
it returns is executed through the existing chain
(``single_expert → ollama → rule_engine``), which already applies the crisis
gate, the forbidden-word guard, the Pydantic schema checks and the
citation/evidence validation.  The fast path changes *how many* calls are made,
never *which* validations run.

Feature flag: :attr:`mindflow.config.Settings.panel_fast_path_enabled`
(default ``False``).  The plan requires an offline replay showing ≥95%
key-conclusion agreement with the full panel and no increase in safety
violations before the flag may be switched on; see
``scripts/experiment_fast_path.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

#: Evidence categories a standard analysis expects to hear about.  Coverage is
#: the share of these that the bundle actually carries.
EXPECTED_EVIDENCE_SOURCES: tuple[str, ...] = (
    "focus",
    "switch",
    "social",
    "delay",
)

#: Metric-name keywords per expected evidence category.
_SOURCE_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "focus": ("focus", "actual_focus", "longest_block", "longest_segment", "attention"),
    "switch": ("switch", "context_switch", "transition"),
    "social": ("social", "entertainment", "audible", "browser_ratio", "top_domain"),
    "delay": ("delay", "start_delay", "idle", "deviation"),
}

FastPathRoute = Literal["full_panel", "single_expert", "rule_engine", "insufficient_data"]


@dataclass(frozen=True)
class FastPathConfig:
    """Thresholds that gate the fast path.

    The defaults are the initial (unvalidated) proposal from the plan: they are
    deliberately conservative and must be re-derived from the offline replay
    before the feature flag is switched on.

    Attributes:
        min_evidence_coverage: Share of :data:`EXPECTED_EVIDENCE_SOURCES` that
            must be present before any non-panel route is considered.
        min_evidence_quality: Mean confidence of the non-``info`` evidence items.
        min_rule_confidence: Rule-engine confidence for its top type.
        min_confidence_margin: Required lead of the top type over the runner-up;
            a narrow margin means the deterministic rule is guessing.
        prefer_single_expert: Route to the single-expert LLM tier instead of the
            rule engine when the fast path applies.  Off by default: the plan
            allows "rule result or single expert", and the rule engine is free.
    """

    min_evidence_coverage: float = 0.6
    min_evidence_quality: float = 0.6
    min_rule_confidence: float = 0.75
    min_confidence_margin: float = 0.2
    prefer_single_expert: bool = False
    expected_sources: tuple[str, ...] = EXPECTED_EVIDENCE_SOURCES


DEFAULT_FAST_PATH_CONFIG = FastPathConfig()


@dataclass(frozen=True)
class FastPathSignals:
    """Everything the decision needs, computed from the bundle and the rules.

    Attributes:
        rule_confidence: Confidence of the rule engine's top type.
        confidence_margin: Top confidence minus runner-up (0.0 when只有一个类型).
        evidence_coverage: Share of expected evidence categories present.
        evidence_quality: Mean confidence of anomalous/deviating evidence items.
        rule_conflict: The rule engine itself is torn between types.
        history_disagreement: The user's recent explicit feedback contradicts the
            rule engine's current conclusion.
        missing_sources: Expected evidence categories that are absent.
    """

    rule_confidence: float
    confidence_margin: float
    evidence_coverage: float
    evidence_quality: float
    rule_conflict: bool = False
    history_disagreement: bool = False
    missing_sources: tuple[str, ...] = field(default_factory=tuple)
    #: Collectors that were disabled or unavailable for this window, taken from
    #: the window quality record.  A *missing collector* is different from
    #: "this window happens to have no metric of that category": only the former
    #: means the analysis cannot be trusted at all.
    collector_gaps: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FastPathDecision:
    """The chosen route plus the machine-readable reason for the trace."""

    route: FastPathRoute
    reason: str

    @property
    def runs_full_panel(self) -> bool:
        return self.route == "full_panel"


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _metric_names(bundle: Any) -> list[str]:
    """Metric names of the bundle items, tolerating test doubles."""
    items = getattr(bundle, "items", ()) or ()
    names: list[str] = []
    for item in items:
        metric = getattr(item, "metric", None)
        if metric:
            names.append(str(metric))
    return names


def evidence_sources_present(
    metric_names: Iterable[str],
    expected: Sequence[str] = EXPECTED_EVIDENCE_SOURCES,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return ``(present_categories, missing_categories)`` for *metric_names*."""
    lowered = [name.lower() for name in metric_names]
    present: set[str] = set()
    for category in expected:
        keywords = _SOURCE_KEYWORDS.get(category, (category,))
        if any(keyword in name for name in lowered for keyword in keywords):
            present.add(category)
    missing = tuple(category for category in expected if category not in present)
    return frozenset(present), missing


def evidence_quality(bundle: Any) -> float:
    """Mean confidence of the anomalous/deviating items (0.0 when there are none).

    Stable, ``info``-severity items carry no signal about whether the *observed*
    window is trustworthy, so they do not participate.
    """
    items = getattr(bundle, "items", ()) or ()
    confidences = [
        _as_float(getattr(item, "confidence", 0.0))
        for item in items
        if str(getattr(item, "severity", "info")) != "info"
    ]
    if not confidences:
        return 0.0
    return sum(confidences) / len(confidences)


def _rule_conflict(confidence: Mapping[str, float], margin_threshold: float) -> bool:
    """True when the rule engine's own top two types are within the margin."""
    values = sorted((_as_float(v) for v in confidence.values()), reverse=True)
    if len(values) < 2:
        return False
    return (values[0] - values[1]) < margin_threshold


def signals_from_assessment(
    bundle: Any,
    assessment: Mapping[str, Any] | None,
    *,
    history_disagreement: bool = False,
    config: FastPathConfig = DEFAULT_FAST_PATH_CONFIG,
) -> FastPathSignals:
    """Build :class:`FastPathSignals` from an evidence bundle + rule assessment.

    Args:
        bundle: The ``EvidenceBundle`` for the analysis window.
        assessment: The rule engine's assessment dict (``type_confidence`` or
            ``confidence`` mapping).  ``None`` is treated as "no opinion", which
            forces the full panel.
        history_disagreement: True when recent explicit feedback contradicts the
            rule engine's conclusion for this window.
        config: Threshold source (used for the conflict margin).
    """
    confidence_raw: Mapping[str, Any] = {}
    if isinstance(assessment, Mapping):
        raw = assessment.get("type_confidence") or assessment.get("confidence") or {}
        confidence_raw = raw if isinstance(raw, Mapping) else {}
    confidence = {str(k): _as_float(v) for k, v in dict(confidence_raw).items()}
    values = sorted(confidence.values(), reverse=True)
    top = values[0] if values else 0.0
    margin = (values[0] - values[1]) if len(values) >= 2 else top

    metric_names = _metric_names(bundle)
    present, missing = evidence_sources_present(metric_names, config.expected_sources)
    coverage = len(present) / len(config.expected_sources) if config.expected_sources else 1.0

    return FastPathSignals(
        rule_confidence=top,
        confidence_margin=margin,
        evidence_coverage=coverage,
        evidence_quality=evidence_quality(bundle),
        rule_conflict=_rule_conflict(confidence, config.min_confidence_margin),
        history_disagreement=history_disagreement,
        missing_sources=missing,
    )


def signals_from_payload(
    bundle_json: str,
    assessment: Mapping[str, Any] | None,
    *,
    history_disagreement: bool = False,
    config: FastPathConfig = DEFAULT_FAST_PATH_CONFIG,
) -> FastPathSignals:
    """Build signals from the *serialized* bundle the panel would receive.

    The graph state carries ``bundle_json`` (already compact-serialized for the
    LLM) rather than the bundle object, so this variant reads the same payload
    the panel would have seen: full evidence rows, the collapsed
    ``stable_summary`` metrics (still present, just summarised) and the
    ``evidence_catalog`` ids.  A malformed payload degrades to "no evidence",
    which routes to ``insufficient_data`` rather than to a confident guess.
    """
    metric_names: list[str] = []
    confidences: list[float] = []

    parsed: dict[str, Any] = {}
    try:
        import json

        loaded = json.loads(bundle_json or "{}")
        if isinstance(loaded, dict):
            parsed = loaded
    except (ValueError, TypeError):
        parsed = {}

    for row in parsed.get("evidence", ()) or ():
        if not isinstance(row, Mapping):
            continue
        metric = row.get("metric")
        if metric:
            metric_names.append(str(metric))
        if str(row.get("severity", "info")) != "info":
            confidences.append(_as_float(row.get("confidence", 0.0)))

    stable = parsed.get("stable_summary") or {}
    if isinstance(stable, Mapping):
        metrics = stable.get("metrics")
        if isinstance(metrics, Mapping):
            metric_names.extend(str(name) for name in metrics)

    for entry in parsed.get("evidence_catalog", ()) or ():
        # Catalog rows are positional ``[id, label, type]`` triples; dict rows
        # are accepted too so an older/legacy payload still parses.
        if isinstance(entry, Mapping):
            metric_id = entry.get("id")
        elif isinstance(entry, (list, tuple)) and entry:
            metric_id = entry[0]
        else:
            metric_id = None
        if metric_id:
            metric_names.append(str(metric_id))

    # Collector-level gaps (from the per-window quality record) are decisive:
    # they mean a data source was absent, not that this window was quiet.
    collector_gaps: list[str] = []
    raw_gaps = parsed.get("collector_gaps")
    if isinstance(raw_gaps, (list, tuple)):
        collector_gaps.extend(str(gap) for gap in raw_gaps if str(gap).strip())
    quality = parsed.get("window_quality")
    if isinstance(quality, Mapping):
        quality_gaps = quality.get("gaps")
        if isinstance(quality_gaps, (list, tuple)):
            collector_gaps.extend(str(gap) for gap in quality_gaps if str(gap).strip())

    confidence_raw: Mapping[str, Any] = {}
    if isinstance(assessment, Mapping):
        raw = assessment.get("type_confidence") or assessment.get("confidence") or {}
        confidence_raw = raw if isinstance(raw, Mapping) else {}
    confidence = {
        str(k): _as_float(v) for k, v in dict(confidence_raw).items()
    }
    values = sorted(confidence.values(), reverse=True)
    top = values[0] if values else 0.0
    margin = (values[0] - values[1]) if len(values) >= 2 else top

    present, missing = evidence_sources_present(metric_names, config.expected_sources)
    coverage = (
        len(present) / len(config.expected_sources)
        if config.expected_sources else 1.0
    )

    return FastPathSignals(
        rule_confidence=top,
        confidence_margin=margin,
        evidence_coverage=coverage,
        evidence_quality=(sum(confidences) / len(confidences)) if confidences else 0.0,
        rule_conflict=_rule_conflict(confidence, config.min_confidence_margin),
        history_disagreement=history_disagreement,
        missing_sources=missing,
        collector_gaps=tuple(dict.fromkeys(collector_gaps)),
    )


def decide_panel_route(
    signals: FastPathSignals,
    config: FastPathConfig = DEFAULT_FAST_PATH_CONFIG,
) -> FastPathDecision:
    """Choose the analysis route for *signals*.

    Order of precedence (each rule is documented so the trace explains itself):

    1. **insufficient evidence** → ``insufficient_data`` (no LLM calls).  This
       fires when a collector was missing/disabled (``collector_gaps``) or the
       expected-source coverage is below ``min_evidence_coverage``: a panel
       cannot reason about data that was never collected.  A single absent
       category is *not* enough on its own — windows legitimately differ in what
       they contain, which is why ``missing_sources`` is reported but only the
       collector-level signal is decisive.
    2. **anything that needs judgement** → ``full_panel``: rule-engine conflict,
       a narrow confidence margin, low rule confidence, low evidence quality, or
       recent explicit user feedback that contradicts the rules.  These are
       exactly the cases the plan requires the full panel for.
    3. **high-quality, confident, conflict-free** → ``rule_engine`` (or
       ``single_expert`` when ``prefer_single_expert``): the deterministic answer
       is already trustworthy, so the panel is skipped.
    """
    if signals.collector_gaps or signals.evidence_coverage < config.min_evidence_coverage:
        gaps = "、".join(signals.collector_gaps) or "整体覆盖不足"
        missing = "、".join(signals.missing_sources)
        detail = f"缺失采集源：{gaps}" if signals.collector_gaps else f"缺失：{missing or gaps}"
        return FastPathDecision(
            route="insufficient_data",
            reason=(
                f"证据覆盖不足（coverage={signals.evidence_coverage:.2f}，{detail}），"
                "直接返回 insufficient_data，不消耗 LLM 调用"
            ),
        )

    if signals.rule_conflict:
        return FastPathDecision(
            route="full_panel",
            reason=(
                f"规则引擎内部冲突（领先幅度 {signals.confidence_margin:.2f} < "
                f"{config.min_confidence_margin:.2f}），需要完整 Panel"
            ),
        )
    if signals.rule_confidence < config.min_rule_confidence:
        return FastPathDecision(
            route="full_panel",
            reason=(
                f"规则引擎置信度偏低（{signals.rule_confidence:.2f} < "
                f"{config.min_rule_confidence:.2f}），需要完整 Panel"
            ),
        )
    if signals.evidence_quality < config.min_evidence_quality:
        return FastPathDecision(
            route="full_panel",
            reason=(
                f"证据质量偏低（{signals.evidence_quality:.2f} < "
                f"{config.min_evidence_quality:.2f}），需要完整 Panel"
            ),
        )
    if signals.history_disagreement:
        return FastPathDecision(
            route="full_panel",
            reason="用户历史显式反馈与当前规则结论不一致，需要完整 Panel",
        )

    route: FastPathRoute = "single_expert" if config.prefer_single_expert else "rule_engine"
    return FastPathDecision(
        route=route,
        reason=(
            f"证据完整且质量高（coverage={signals.evidence_coverage:.2f}，"
            f"quality={signals.evidence_quality:.2f}），规则引擎置信度高"
            f"（{signals.rule_confidence:.2f}）且无冲突，走快速路径"
        ),
    )


__all__ = [
    "DEFAULT_FAST_PATH_CONFIG",
    "EXPECTED_EVIDENCE_SOURCES",
    "FastPathConfig",
    "FastPathDecision",
    "FastPathRoute",
    "FastPathSignals",
    "decide_panel_route",
    "evidence_quality",
    "evidence_sources_present",
    "signals_from_assessment",
    "signals_from_payload",
]
