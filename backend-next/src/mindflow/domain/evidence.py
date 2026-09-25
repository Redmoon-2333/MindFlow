"""EvidenceBundle — the evidence contract between ML sensing and LLM reasoning.

This is the most critical interface in the multi-agent upgrade (07-agent-upgrade-design.md §3).
All LLM expert opinions must cite evidence from this bundle; the critic validates
those citations against ``metric_names()``.

Design decisions:
  - Frozen dataclasses (following domain/events.py, domain/procrastination.py).
  - Zero framework dependencies — pure stdlib only.
  - ``to_prompt_json()`` produces a compact, Chinese-first serialization with
    NO window titles or file paths (privacy: NF-S3a).
  - ``metric_names()`` returns a frozenset for O(1) critic lookups.

Severity is the ML-level judgment (not clinical). Four levels:
  - info:     Normal / baseline-in-building / no action needed.
  - mild:     Noticeable but not urgent.
  - moderate: Clearly anomalous, warrants attention.
  - severe:   Extreme outlier, likely requires intervention.

Prompt payload (phase 2.4 compression)
--------------------------------------
``to_prompt_json()`` defaults to a *compressed* payload (``compressed=True``):

  window             analysis window (ISO timestamps)
  evidence           full rows, ONLY for actionable items — anomalous
                     (severity != "info"), significantly baseline-deviating,
                     or related to the most recent intervention
  stable_summary     counts/means describing the collapsed stable items;
                     ``metrics`` holds ``{metric: {"mean": x, "count": n}}``
  evidence_catalog   ``{id, label, type}`` only — the single canonical
                     citation namespace; carries NO values/baselines/
                     quality/severity (those live once, in ``evidence``).
                     One catalog entry per citeable id, so the citation
                     namespace is exactly as large as before compression.
  behavior_summary   aggregated behavioral metrics
  intervention_history / novelty_flags

Kept rows are *full* rows: an ``info`` item that is retained because it
deviates from the baseline (or is intervention-related) also carries its
``value``/``baseline``.  Without them the reason it was kept would be
invisible, and the item is absent from ``stable_summary`` as well.

Uncompressed (``compressed=False``) reproduces the legacy shape: every item
enumerated in ``evidence`` (with ``value``/``baseline`` only for non-``info``
items, the pre-2.4 severity gate the A/B comparison must stay faithful to) and
an ``evidence_catalog`` whose entries additionally repeat ``value`` under the
legacy ``label_zh`` key.  It is available for A/B comparison in tests and
experiments only — production callers use the default.

Privacy invariant (NF-S3a): neither shape ever emits window titles, file
paths, or raw event text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from mindflow.domain.evidence_facts import build_evidence_catalog
from mindflow.domain.procrastination import BehaviorSummary

Severity = Literal["info", "mild", "moderate", "severe"]

_VALID_SEVERITIES: frozenset[str] = frozenset({"info", "mild", "moderate", "severe"})


@dataclass(frozen=True)
class InterventionRecord:
    """A single intervention event with the user's response.

    Attributes:
        intervention_type: One of the four intervention types (nudge, task_breakdown, …).
        triggered_at: When the intervention was fired (timezone-aware UTC).
        user_response: The user's action, or None if unresponded.
        effect_note: Chinese human-readable description of the outcome.
    """

    intervention_type: str
    triggered_at: datetime
    user_response: str | None
    effect_note: str


@dataclass(frozen=True)
class EvidenceItem:
    """A single piece of evidence produced by the ML sensing layer.

    Attributes:
        metric: Machine-readable identifier (e.g. "focus_score", "switch_rate",
            "behavior_deviation"). Used by the critic for citation validation.
        value: The observed value (float for numeric metrics, str for categorical).
        baseline: The expected value from the user's personal baseline, or None
            when no baseline is available yet.
        severity: ML-level judgment — one of "info", "mild", "moderate", "severe".
        confidence: How confident the ML layer is in this item, in [0, 1].
        source: Which subsystem produced this item (e.g. "feature_computation",
            "welford_baseline", "hmm").
        human_readable: Chinese text for LLM and UI consumption. NEVER contains
            window titles or file paths (NF-S3a).
    """

    metric: str
    value: float | str
    baseline: float | None
    severity: Severity
    confidence: float
    source: str
    human_readable: str

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            valid = ", ".join(sorted(_VALID_SEVERITIES))
            raise ValueError(
                f"Invalid severity: {self.severity!r}. Must be one of: {valid}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass(frozen=True)
class EvidenceBundle:
    """The complete evidence package presented to the LLM expert panel.

    Attributes:
        user_id: The user being analysed.
        window: The (start, end) time window of this analysis.
        items: All evidence items from the ML sensing layer.
        behavior_summary: Aggregated behavioral metrics (reused from domain/procrastination.py).
        intervention_history: Recent intervention records for context.
        novelty_flags: Detected novel behaviour patterns (Phase A: simple heuristic).
        events: Activity events used to build this bundle, retained for workflow
            contracts that distinguish an empty window from degraded analysis.
    """

    user_id: int
    window: tuple[datetime, datetime]
    items: tuple[EvidenceItem, ...]
    behavior_summary: BehaviorSummary
    intervention_history: tuple[InterventionRecord, ...]
    novelty_flags: tuple[str, ...]
    events: tuple[object, ...] = ()


# ═══════════════════════════════════════════════════════════════════════════════
# Serialisation helpers
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# Compression policy (phase 2.4)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EvidenceCompressionPolicy:
    """Policy controlling which ``EvidenceItem`` rows stay in the prompt.

    An item is **kept in full** when ANY of the following holds:

      - ``severity != "info"`` — anomalous items always matter.
      - it deviates from the user's baseline by at least
        ``baseline_deviation_ratio`` (relative), or its absolute value
        differs from the baseline by ``max(absolute_floor, ratio·|baseline|)``
        — this keeps significant deviations even for small baselines.
      - its metric is *intervention-related*: the metric name contains a token
        (≥ 2 characters) of one of the newest ``intervention_window``
        intervention types (e.g. ``task_breakdown`` → ``task``/``breakdown``,
        so a ``task_*`` metric is kept in full).  Of the four production types
        (``task_breakdown``, ``nudge``, ``environment_optimization``,
        ``smart_prioritization``) a generic one such as ``nudge`` shares a
        token with none of the metric names currently produced, so it keeps
        nothing extra — the check is literal token containment, not semantics.

    Everything else is **stable** and is represented only by
    ``stable_summary`` (counts + means) instead of a full row.

    Attributes:
        baseline_deviation_ratio: Relative baseline deviation above which an
            item counts as "significantly deviating" (0.15 = 15%).
        baseline_deviation_absolute_floor: Absolute floor for that check, so
            items whose baseline is ~0 still get compared sensibly.
        intervention_window: How many of the most recent intervention records
            count as "recent" when deciding intervention-relatedness.
        stable_metric_limit: Safety valve — when more than this many stable
            metrics would be collapsed, only the ``stable_metric_limit``
            closest to action are summarised and the rest are kept in full.
            ``None`` disables the valve (production default: summarise all).
    """

    baseline_deviation_ratio: float = 0.15
    baseline_deviation_absolute_floor: float = 0.05
    intervention_window: int = 1
    stable_metric_limit: int | None = None


DEFAULT_COMPRESSION_POLICY = EvidenceCompressionPolicy()

# Metric-name keywords that document which metrics an intervention type is
# expected to target.  The relatedness check at runtime is literal token
# containment against the newest intervention type (see
# ``_recent_intervention_keywords`` / ``_is_intervention_related``); this tuple
# is the vocabulary of those metric tokens and is not consulted directly.
_INTERVENTION_KEYWORDS: tuple[str, ...] = (
    "focus",
    "switch",
    "block",
    "break",
    "delay",
    "social",
)


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _deviation_is_significant(item: EvidenceItem, policy: EvidenceCompressionPolicy) -> bool:
    """True when *item* deviates significantly from its personal baseline."""
    if item.baseline is None or not _is_numeric(item.value):
        return False
    observed = float(item.value)
    baseline = float(item.baseline)
    threshold = max(
        policy.baseline_deviation_absolute_floor,
        policy.baseline_deviation_ratio * abs(baseline),
    )
    return abs(observed - baseline) >= threshold


def _recent_intervention_keywords(
    bundle: EvidenceBundle,
    policy: EvidenceCompressionPolicy,
) -> frozenset[str]:
    """Tokens of the newest ``policy.intervention_window`` interventions."""
    if not bundle.intervention_history or policy.intervention_window <= 0:
        return frozenset()
    window = max(1, policy.intervention_window)
    try:
        recent = sorted(
            bundle.intervention_history,
            key=lambda rec: rec.triggered_at,
        )[-window:]
    except TypeError:  # naive/aware datetime mix — fall back to insertion order
        recent = list(bundle.intervention_history)[-window:]
    keywords: set[str] = set()
    for record in recent:
        keywords.update(
            token
            for token in record.intervention_type.lower().replace("-", "_").split("_")
            if len(token) >= 2
        )
    return frozenset(keywords)


def _is_intervention_related(metric: str, keywords: frozenset[str]) -> bool:
    """True when the metric name mentions a recent-intervention keyword."""
    lowered = metric.lower()
    return any(keyword in lowered for keyword in keywords)


def is_actionable_item(
    item: EvidenceItem,
    bundle: EvidenceBundle,
    policy: EvidenceCompressionPolicy = DEFAULT_COMPRESSION_POLICY,
    intervention_keywords: frozenset[str] | None = None,
) -> bool:
    """Whether *item* must be sent in full (anomalous / deviating / recent)."""
    if item.severity != "info":
        return True
    if _deviation_is_significant(item, policy):
        return True
    keywords = (
        intervention_keywords
        if intervention_keywords is not None
        else _recent_intervention_keywords(bundle, policy)
    )
    return _is_intervention_related(item.metric, keywords)


def _evidence_row(item: EvidenceItem, *, include_info_values: bool = False) -> dict[str, Any]:
    """One full evidence row.

    ``include_info_values`` is used by the compressed payload: an ``info`` item
    kept because it deviates from the baseline (or is intervention-related)
    must carry its ``value``/``baseline`` — the row is the only place they can
    appear, since the item is not part of ``stable_summary``.  The legacy
    payload keeps the severity gate so its shape stays byte-comparable with the
    pre-2.4 serialization.
    """
    entry: dict[str, Any] = {
        "metric": item.metric,
        "severity": item.severity,
        "confidence": item.confidence,
        "human_readable": item.human_readable,
    }
    if item.severity != "info" or include_info_values:
        entry["value"] = _json_safe(item.value)
        if item.baseline is not None:
            entry["baseline"] = _json_safe(item.baseline)
    return entry


def _stable_summary(
    stable: list[EvidenceItem],
    total_metrics: int,
) -> dict[str, Any]:
    """Counts + per-metric means for the collapsed (stable) items."""
    grouped: dict[str, list[EvidenceItem]] = {}
    for item in stable:
        grouped.setdefault(item.metric, []).append(item)

    metrics: dict[str, dict[str, Any]] = {}
    for metric, items in grouped.items():
        numeric = [float(i.value) for i in items if _is_numeric(i.value)]
        entry: dict[str, Any] = {"count": len(items)}
        if numeric:
            entry["mean"] = round(sum(numeric) / len(numeric), 4)
        else:
            entry["value"] = _json_safe(items[0].value)
        metrics[metric] = entry

    return {
        "collapsed_items": len(stable),
        "collapsed_metrics": len(grouped),
        "total_metrics": total_metrics,
        "metrics": metrics,
    }


def _legacy_prompt_payload(bundle: EvidenceBundle) -> dict[str, Any]:
    """Reproduce the pre-2.4 payload shape (A/B comparison only)."""
    evidence_list = [_evidence_row(item) for item in bundle.items]
    catalog = build_evidence_catalog(bundle)
    return {
        "window": {
            "start": bundle.window[0].isoformat(),
            "end": bundle.window[1].isoformat(),
        },
        "evidence": evidence_list,
        "behavior_summary": _behavior_summary_json(bundle.behavior_summary),
        "intervention_history": _intervention_history_json(bundle),
        "novelty_flags": list(bundle.novelty_flags),
        "evidence_catalog": [
            {"id": fact.id, "label_zh": fact.label_zh, "value": _json_safe(fact.value)}
            for fact in catalog
        ],
    }


def _behavior_summary_json(summary: BehaviorSummary) -> dict[str, Any]:
    """Aggregated behaviour summary (no raw events, no ``intended_task``)."""
    bs: dict[str, Any] = {
        "duration_min": summary.duration_min,
        "actual_focus_min": summary.actual_focus_min,
        "context_switches_per_hour": summary.context_switches_per_hour,
        "longest_focus_block_sec": summary.longest_focus_block_s,
        "social_media_ratio": summary.social_media_ratio,
        "start_delay_min": summary.start_delay_min,
    }
    if summary.baseline_deviation is not None:
        bs["baseline_deviation"] = summary.baseline_deviation
    return bs


def _intervention_history_json(bundle: EvidenceBundle) -> list[dict[str, Any]]:
    """Intervention history rows (no ids, no window titles)."""
    return [
        {
            "type": rec.intervention_type,
            "triggered_at": rec.triggered_at.isoformat(),
            "user_response": rec.user_response,
            "effect_note": rec.effect_note,
        }
        for rec in bundle.intervention_history
    ]


def _json_safe(value: object) -> object:
    """Convert a value to JSON-safe type."""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# Catalog entry types, derived from ``EvidenceFact.source_refs`` (the fact's
# originating bundle section).  Consumers only use ``type`` as a display hint.
_CATALOG_TYPE_BY_SECTION: dict[str, str] = {
    "evidence": "metric",
    "behavior_summary": "summary",
    "intervention_history": "intervention",
    "novelty_flags": "novelty",
}


def _catalog_entry_type(source_refs: tuple[str, ...]) -> str:
    """Map a catalog fact's source section to its entry type."""
    if not source_refs:
        return "other"
    section = source_refs[0].split(".", 1)[0]
    return _CATALOG_TYPE_BY_SECTION.get(section, "other")


def to_prompt_json(
    bundle: EvidenceBundle,
    compressed: bool = True,
    policy: EvidenceCompressionPolicy = DEFAULT_COMPRESSION_POLICY,
) -> str:
    """Serialize an ``EvidenceBundle`` for LLM consumption.

    Rules:
      - Compact JSON (no extra whitespace) to minimise token usage.
      - Human-readable Chinese values are preferred over raw numbers.
      - **No window titles or file paths** are included (NF-S3a).
      - Behavioral metrics use the aggregated summary, not raw events.
      - ``compressed=True`` (default) sends full rows only for actionable
        items and summarises stable metrics (see
        ``EvidenceCompressionPolicy``); ``compressed=False`` reproduces the
        legacy shape for A/B comparison — it is not a production mode.

    Args:
        bundle: The evidence bundle to serialise.
        compressed: Emit the compressed payload (default) or the legacy shape.
        policy: Compression policy — ignored when *compressed* is False.

    Returns:
        A compact JSON string suitable for inclusion in an LLM prompt.
    """
    data: dict[str, Any] = {
        "window": {
            "start": bundle.window[0].isoformat(),
            "end": bundle.window[1].isoformat(),
        },
    }

    if not compressed:
        data.update(_legacy_prompt_payload(bundle))
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    keywords = _recent_intervention_keywords(bundle, policy)
    actionable = [
        item
        for item in bundle.items
        if is_actionable_item(item, bundle, policy, keywords)
    ]
    stable = [
        item
        for item in bundle.items
        if not is_actionable_item(item, bundle, policy, keywords)
    ]
    if (
        policy.stable_metric_limit is not None
        and len({item.metric for item in stable}) > policy.stable_metric_limit
    ):
        # Safety valve: summarise only the first N stable metrics; keep the rest
        # enumerated so nothing silently disappears from the payload.
        summarised: set[str] = set()
        demoted: list[EvidenceItem] = []
        for item in stable:
            if item.metric in summarised or len(summarised) < policy.stable_metric_limit:
                summarised.add(item.metric)
            else:
                demoted.append(item)
        if demoted:
            demoted_set = {id(item) for item in demoted}
            stable = [item for item in stable if id(item) not in demoted_set]
            actionable.extend(demoted)

    evidence_list = [_evidence_row(item, include_info_values=True) for item in actionable]

    data["evidence"] = evidence_list
    if stable:
        # Only pay for the summary when something was actually collapsed.
        data["stable_summary"] = _stable_summary(stable, len(bundle.items))
    data["behavior_summary"] = _behavior_summary_json(bundle.behavior_summary)
    data["intervention_history"] = _intervention_history_json(bundle)
    data["novelty_flags"] = list(bundle.novelty_flags)

    # ── Evidence catalog: the canonical citation namespace ──────────────
    # Carries id + Chinese label + type ONLY.  Values, baselines, quality
    # (confidence) and severity are emitted exactly once, in ``evidence``.
    # The ✱keep-all✱ entry list keeps every citeable id resolvable, so
    # citation recall is unchanged by compression.
    #
    # The entries are positional 3-tuples ``[id, label, type]`` rather than
    # objects: repeating ``{"id":…,"label":…,"type":…}`` on every row made the
    # catalog ~58% of the whole payload, and the key names carry no
    # information.  Same data, no loss — measured 34% smaller than the legacy
    # payload on a production-shaped 31-item window (21% before this encoding).
    catalog = build_evidence_catalog(bundle)
    data["evidence_catalog"] = [
        [fact.id, fact.label_zh, _catalog_entry_type(fact.source_refs)]
        for fact in catalog
    ]

    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def metric_names(bundle: EvidenceBundle) -> frozenset[str]:
    """Return all metric names present in the bundle.

    Used by the critic agent to validate that every ``[证据: 指标名]`` citation
    in an expert's response refers to a metric that actually exists.

    Args:
        bundle: The evidence bundle.

    Returns:
        A frozenset of metric strings for O(1) membership checks.
    """
    return frozenset(item.metric for item in bundle.items)
