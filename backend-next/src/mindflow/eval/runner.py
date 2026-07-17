"""Evaluation runner — run analyzers against scenarios, compute metrics, compare.

Provides the core evaluation loop, aggregate statistics, and comparison logic
for G006 (single-expert vs expert-panel benchmarking).

Metrics computed:
  - Top-1 accuracy: Fraction of scenarios where the highest-confidence predicted
    type matches the gold-standard top type.
  - Jaccard index: Mean |predicted_types ∩ expected_types| / |predicted_types ∪ expected_types|
    across all scenarios.
  - Technique match rate: Fraction of scenarios where the predicted CBT technique
    matches the gold-standard technique.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from mindflow.domain.evidence import EvidenceBundle
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType
from mindflow.eval.scenarios import EvalScenario

# ---------------------------------------------------------------------------
# AssessmentLike — the common interface from rule engine and panel
# ---------------------------------------------------------------------------

# Both ProcrastinationAssessment and PanelVerdict share these fields:
#   types: tuple[ProcrastinationType, ...]
#   confidence: Mapping[ProcrastinationType, float]
#   recommended_technique: CBTTechnique | None
#   rationale: str
#   source: str
# We use a broad type to accept both.


@dataclass(frozen=True)
class ScenarioResult:
    """Result of evaluating a single scenario."""

    scenario_id: str
    description: str
    expected_types: tuple[ProcrastinationType, ...]
    predicted_types: tuple[ProcrastinationType, ...]
    expected_technique: CBTTechnique | None
    predicted_technique: CBTTechnique | None
    top1_hit: bool
    jaccard: float
    technique_match: bool
    predicted_source: str
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalReport:
    """Aggregate evaluation report for a single analyzer."""

    analyzer_name: str
    scenario_results: tuple[ScenarioResult, ...]
    top1_accuracy: float
    mean_jaccard: float
    technique_accuracy: float
    timestamp: datetime
    total: int = 0
    hits: int = 0
    misses: int = 0

    def __post_init__(self) -> None:
        # Use object.__setattr__ because frozen dataclass
        if self.total == 0 and self.scenario_results:
            object.__setattr__(self, "total", len(self.scenario_results))
            hits = sum(1 for r in self.scenario_results if r.top1_hit)
            object.__setattr__(self, "hits", hits)
            object.__setattr__(self, "misses", self.total - hits)


@dataclass(frozen=True)
class ComparisonReport:
    """Comparison between baseline (rule engine) and panel (expert or mock)."""

    baseline_name: str
    panel_name: str
    baseline_report: EvalReport
    panel_report: EvalReport
    top1_delta: float
    jaccard_delta: float
    technique_delta: float
    baseline_wins: int
    panel_wins: int
    ties: int
    per_scenario: tuple[ScenarioResult, ...]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _compute_jaccard(
    predicted: tuple[ProcrastinationType, ...],
    expected: tuple[ProcrastinationType, ...],
) -> float:
    """Compute Jaccard index between two type tuples (treated as sets)."""
    pred_set = set(predicted)
    exp_set = set(expected)
    union = pred_set | exp_set
    if not union:
        return 1.0  # both empty → perfect agreement
    intersection = pred_set & exp_set
    return len(intersection) / len(union)


async def run_eval(
    analyzer: Callable[[EvidenceBundle], Awaitable[Any]],
    scenarios: tuple[EvalScenario, ...],
    *,
    analyzer_name: str = "unknown",
) -> EvalReport:
    """Run an analyzer against all scenarios and produce an evaluation report.

    Args:
        analyzer: Async callable that takes EvidenceBundle and returns an
            object with ``types``, ``recommended_technique``, and ``source``.
        scenarios: Tuple of EvalScenario instances.
        analyzer_name: Label for the report (e.g. "rule_engine" or "panel").

    Returns:
        EvalReport with aggregate metrics and per-scenario results.
    """
    results: list[ScenarioResult] = []

    for scenario in scenarios:
        assessment = await analyzer(scenario.bundle)

        predicted_types: tuple[ProcrastinationType, ...] = getattr(assessment, "types", ())
        predicted_technique: CBTTechnique | None = (
            getattr(assessment, "recommended_technique", None)
        )
        predicted_source: str = getattr(assessment, "source", "unknown")

        expected_types = scenario.expected_types
        expected_technique = scenario.expected_technique

        # Top-1 hit: predicted[0] matches expected[0]
        top1_hit = bool(
            predicted_types
            and expected_types
            and predicted_types[0] == expected_types[0]
        )

        jaccard = _compute_jaccard(predicted_types, expected_types)

        technique_match = predicted_technique == expected_technique

        results.append(ScenarioResult(
            scenario_id=scenario.scenario_id,
            description=scenario.description,
            expected_types=expected_types,
            predicted_types=predicted_types,
            expected_technique=expected_technique,
            predicted_technique=predicted_technique,
            top1_hit=top1_hit,
            jaccard=jaccard,
            technique_match=technique_match,
            predicted_source=predicted_source,
        ))

    total = len(results)
    hits = sum(1 for r in results if r.top1_hit)
    top1_accuracy = hits / total if total > 0 else 0.0
    mean_jaccard = sum(r.jaccard for r in results) / total if total > 0 else 0.0
    technique_accuracy = sum(1 for r in results if r.technique_match) / total if total > 0 else 0.0

    return EvalReport(
        analyzer_name=analyzer_name,
        scenario_results=tuple(results),
        top1_accuracy=top1_accuracy,
        mean_jaccard=mean_jaccard,
        technique_accuracy=technique_accuracy,
        timestamp=datetime.now(UTC),
        total=total,
        hits=hits,
        misses=total - hits,
    )


def compare(
    baseline_report: EvalReport,
    panel_report: EvalReport,
    *,
    baseline_name: str = "rule_engine",
    panel_name: str = "panel",
) -> ComparisonReport:
    """Compare two evaluation reports, scoring per-scenario wins/losses/ties.

    Win/loss is determined by Top-1 hit: if baseline hits and panel misses,
    baseline_wins++; if panel hits and baseline misses, panel_wins++.
    If both hit or both miss, ties++.
    """
    top1_delta = panel_report.top1_accuracy - baseline_report.top1_accuracy
    jaccard_delta = panel_report.mean_jaccard - baseline_report.mean_jaccard
    technique_delta = panel_report.technique_accuracy - baseline_report.technique_accuracy

    baseline_wins = 0
    panel_wins = 0
    ties = 0

    per_scenario: list[ScenarioResult] = []

    baseline_map = {r.scenario_id: r for r in baseline_report.scenario_results}
    panel_map = {r.scenario_id: r for r in panel_report.scenario_results}

    all_ids = set(baseline_map) | set(panel_map)
    for sid in sorted(all_ids):
        br = baseline_map.get(sid)
        pr = panel_map.get(sid)

        if br is None or pr is None:
            continue

        # Win/loss determination
        baseline_hit = br.top1_hit
        panel_hit = pr.top1_hit

        if baseline_hit and not panel_hit:
            baseline_wins += 1
        elif panel_hit and not baseline_hit:
            panel_wins += 1
        else:
            ties += 1

        per_scenario.append(pr)

    return ComparisonReport(
        baseline_name=baseline_name,
        panel_name=panel_name,
        baseline_report=baseline_report,
        panel_report=panel_report,
        top1_delta=top1_delta,
        jaccard_delta=jaccard_delta,
        technique_delta=technique_delta,
        baseline_wins=baseline_wins,
        panel_wins=panel_wins,
        ties=ties,
        per_scenario=tuple(per_scenario),
    )
