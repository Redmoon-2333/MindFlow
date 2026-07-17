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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

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
    """

    user_id: int
    window: tuple[datetime, datetime]
    items: tuple[EvidenceItem, ...]
    behavior_summary: BehaviorSummary
    intervention_history: tuple[InterventionRecord, ...]
    novelty_flags: tuple[str, ...]


# ═══════════════════════════════════════════════════════════════════════════════
# Serialisation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def to_prompt_json(bundle: EvidenceBundle) -> str:
    """Serialize an ``EvidenceBundle`` for LLM consumption.

    Rules:
      - Compact JSON (no extra whitespace) to minimise token usage.
      - Human-readable Chinese values are preferred over raw numbers.
      - **No window titles or file paths** are included (NF-S3a).
      - Behavioral metrics use the aggregated summary, not raw events.

    Args:
        bundle: The evidence bundle to serialise.

    Returns:
        A compact JSON string suitable for inclusion in an LLM prompt.
    """
    evidence_list: list[dict[str, Any]] = []
    for item in bundle.items:
        entry: dict[str, Any] = {
            "metric": item.metric,
            "severity": item.severity,
            "confidence": item.confidence,
            "human_readable": item.human_readable,
        }
        if item.severity != "info":
            entry["value"] = item.value
            if item.baseline is not None:
                entry["baseline"] = item.baseline
        evidence_list.append(entry)

    # Behaviour summary (aggregated, no raw events)
    summary = bundle.behavior_summary
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

    # Intervention history (no IDs, no window titles)
    interventions: list[dict[str, Any]] = []
    for rec in bundle.intervention_history:
        interventions.append({
            "type": rec.intervention_type,
            "triggered_at": rec.triggered_at.isoformat(),
            "user_response": rec.user_response,
            "effect_note": rec.effect_note,
        })

    data: dict[str, Any] = {
        "window": {
            "start": bundle.window[0].isoformat(),
            "end": bundle.window[1].isoformat(),
        },
        "evidence": evidence_list,
        "behavior_summary": bs,
        "intervention_history": interventions,
        "novelty_flags": list(bundle.novelty_flags),
    }
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
