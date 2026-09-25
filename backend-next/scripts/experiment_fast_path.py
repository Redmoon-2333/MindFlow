"""Offline replay for the panel fast path (optimisation plan 2.2).

The fast path may only be switched on when, on a fixed scenario set, its key
conclusions agree with the full panel at least 95% of the time and the safety
violation rate does not increase.  This script measures exactly that, offline
and deterministically:

  * for every eval scenario it computes the **full-panel** conclusion with the
    deterministic mock panel gateway (no API key, no network);
  * it computes the **fast-path decision** from the rule engine's assessment and
    the serialized evidence bundle;
  * when the fast path would answer without the panel, it records the rule
    engine's conclusion as the fast-path conclusion;
  * it then compares the two conclusions (top type and type set) and checks the
    forbidden-word contract on whichever text would be surfaced.

Artifacts are written under ``data/experiments/<run-id>/`` (repo rule: no
temporary reports outside that directory).

Usage::

    uv run python scripts/experiment_fast_path.py
    uv run python scripts/experiment_fast_path.py --out data/experiments/manual
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mindflow.agents.types import _contains_forbidden_words
from mindflow.eval.adapters import MockPanelGateway, panel_analyzer, rule_engine_analyzer
from mindflow.eval.scenarios import ALL_SCENARIOS, EvalScenario
from mindflow.graph.panel_fastpath import (
    DEFAULT_FAST_PATH_CONFIG,
    decide_panel_route,
    signals_from_assessment,
)

#: LLM calls a full panel deliberation makes on the fast path.
_FULL_PANEL_CALLS = 6
#: Calls each fast-path route makes.
_ROUTE_CALLS = {"full_panel": _FULL_PANEL_CALLS, "single_expert": 1, "rule_engine": 0,
                "insufficient_data": 0}
#: The plan's acceptance thresholds.
MIN_AGREEMENT = 0.95
MAX_SAFETY_VIOLATION_RATE = 0.0


@dataclass
class ScenarioComparison:
    scenario_id: str
    description: str
    route: str
    reason: str
    panel_types: list[str]
    fast_types: list[str]
    top1_agree: bool
    set_agree: bool
    jaccard: float
    panel_calls: int
    fast_calls: int
    safety_violations: int


@dataclass
class ReplayReport:
    run_id: str
    scenario_count: int
    fast_path_scenarios: int = 0
    route_counts: dict[str, int] = field(default_factory=dict)
    top1_agreement_all: float = 0.0
    top1_agreement_fast_path: float = 0.0
    set_agreement_fast_path: float = 0.0
    mean_jaccard_fast_path: float = 0.0
    safety_violations: int = 0
    panel_calls_full: int = 0
    panel_calls_with_fast_path: int = 0
    llm_calls_saved: int = 0
    passes_agreement: bool = False
    passes_safety: bool = False
    verdict: str = ""
    comparisons: list[ScenarioComparison] = field(default_factory=list)


def _assessment_dict(assessment: Any) -> dict[str, Any]:
    """Convert a rule-engine assessment into the fast-path signal input."""
    confidence = getattr(assessment, "confidence", {}) or {}
    return {
        "procrastination_types": [str(t) for t in getattr(assessment, "types", ()) or ()],
        "type_confidence": {str(k): float(v) for k, v in dict(confidence).items()},
        "rationale": str(getattr(assessment, "rationale", "") or ""),
    }


def _types_of(assessment: Any) -> list[str]:
    return [str(t) for t in getattr(assessment, "types", ()) or ()]


def _jaccard(first: list[str], second: list[str]) -> float:
    a, b = set(first), set(second)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _unsafe_texts(assessment: Any) -> int:
    """Count forbidden-word violations in the text an assessment would surface."""
    texts = [
        str(getattr(assessment, "rationale", "") or ""),
        str(getattr(assessment, "response_text", "") or ""),
    ]
    return sum(1 for text in texts if _contains_forbidden_words(text))


async def compare_scenario(scenario: EvalScenario, panel: Any, rules: Any) -> ScenarioComparison:
    panel_assessment = await panel(scenario.bundle)
    rule_assessment = await rules(scenario.bundle)

    signals = signals_from_assessment(
        scenario.bundle, _assessment_dict(rule_assessment),
    )
    decision = decide_panel_route(signals)

    panel_types = _types_of(panel_assessment)
    fast_types = panel_types if decision.route == "full_panel" else _types_of(rule_assessment)

    safety = _unsafe_texts(panel_assessment) + (
        0 if decision.route == "full_panel" else _unsafe_texts(rule_assessment)
    )

    return ScenarioComparison(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        route=decision.route,
        reason=decision.reason,
        panel_types=panel_types,
        fast_types=fast_types,
        top1_agree=bool(panel_types and fast_types and panel_types[0] == fast_types[0]),
        set_agree=set(panel_types) == set(fast_types),
        jaccard=round(_jaccard(panel_types, fast_types), 4),
        panel_calls=_FULL_PANEL_CALLS,
        fast_calls=_ROUTE_CALLS[decision.route],
        safety_violations=safety,
    )


async def run_replay(scenarios: tuple[EvalScenario, ...] = ALL_SCENARIOS) -> ReplayReport:
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_fastpath")
    panel = panel_analyzer(MockPanelGateway())
    rules = rule_engine_analyzer

    report = ReplayReport(run_id=run_id, scenario_count=len(scenarios))
    for scenario in scenarios:
        comparison = await compare_scenario(scenario, panel, rules)
        report.comparisons.append(comparison)
        report.route_counts[comparison.route] = report.route_counts.get(comparison.route, 0) + 1
        report.safety_violations += comparison.safety_violations
        report.panel_calls_full += comparison.panel_calls
        report.panel_calls_with_fast_path += comparison.fast_calls

    report.llm_calls_saved = report.panel_calls_full - report.panel_calls_with_fast_path

    fast = [c for c in report.comparisons if c.route != "full_panel"]
    report.fast_path_scenarios = len(fast)
    total = len(report.comparisons) or 1
    report.top1_agreement_all = round(
        sum(1 for c in report.comparisons if c.top1_agree) / total, 4,
    )
    if fast:
        report.top1_agreement_fast_path = round(
            sum(1 for c in fast if c.top1_agree) / len(fast), 4,
        )
        report.set_agreement_fast_path = round(
            sum(1 for c in fast if c.set_agree) / len(fast), 4,
        )
        report.mean_jaccard_fast_path = round(
            sum(c.jaccard for c in fast) / len(fast), 4,
        )

    report.passes_agreement = report.top1_agreement_fast_path >= MIN_AGREEMENT
    report.passes_safety = (
        report.safety_violations / total <= MAX_SAFETY_VIOLATION_RATE
    )
    if not fast:
        report.verdict = (
            "快速路径在固定场景集上从未触发（规则引擎置信度均不足以走快速路径），"
            "因此没有可比较的一致率样本；保持 feature flag 关闭。"
        )
    elif report.passes_agreement and report.passes_safety:
        report.verdict = (
            f"快速路径触发 {len(fast)}/{total} 个场景，关键结论一致率 "
            f"{report.top1_agreement_fast_path:.1%} ≥ {MIN_AGREEMENT:.0%}，"
            "且安全违规率未上升；满足方案允许逐步开启的前提（仍需人工确认后再打开开关）。"
        )
    else:
        report.verdict = (
            f"快速路径一致率 {report.top1_agreement_fast_path:.1%} 未达到 "
            f"{MIN_AGREEMENT:.0%} 或安全违规率上升；不得开启 feature flag。"
        )
    return report


def _markdown(report: ReplayReport) -> str:
    fast = [c for c in report.comparisons if c.route != "full_panel"]
    lines = [
        "# Panel fast path — offline replay",
        "",
        f"- run: `{report.run_id}`",
        f"- scenarios: {report.scenario_count}",
        f"- routes: {json.dumps(report.route_counts, ensure_ascii=False)}",
        f"- fast-path scenarios: {report.fast_path_scenarios}",
        f"- top-1 agreement (all scenarios): {report.top1_agreement_all:.1%}",
        f"- top-1 agreement (fast-path only): {report.top1_agreement_fast_path:.1%}",
        f"- type-set agreement (fast-path only): {report.set_agreement_fast_path:.1%}",
        f"- mean Jaccard (fast-path only): {report.mean_jaccard_fast_path:.3f}",
        f"- safety violations: {report.safety_violations}",
        f"- LLM calls: full panel {report.panel_calls_full} → with fast path "
        f"{report.panel_calls_with_fast_path} (saved {report.llm_calls_saved})",
        f"- acceptance thresholds: agreement ≥ {MIN_AGREEMENT:.0%}, "
        f"safety violations ≤ {MAX_SAFETY_VIOLATION_RATE:.0%}",
        f"- verdict: {report.verdict}",
        "",
        "## Fast-path scenarios",
        "",
        "| scenario | route | panel types | fast types | jaccard | reason |",
        "|---|---|---|---|---|---|",
    ]
    for comparison in fast:
        lines.append(
            f"| {comparison.scenario_id} | {comparison.route} | "
            f"{', '.join(comparison.panel_types) or '-'} | "
            f"{', '.join(comparison.fast_types) or '-'} | {comparison.jaccard} | "
            f"{comparison.reason} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_artifacts(report: ReplayReport, out_dir: Path) -> Path:
    payload = asdict(report)
    payload["config"] = {
        "min_evidence_coverage": DEFAULT_FAST_PATH_CONFIG.min_evidence_coverage,
        "min_evidence_quality": DEFAULT_FAST_PATH_CONFIG.min_evidence_quality,
        "min_rule_confidence": DEFAULT_FAST_PATH_CONFIG.min_rule_confidence,
        "min_confidence_margin": DEFAULT_FAST_PATH_CONFIG.min_confidence_margin,
        "prefer_single_expert": DEFAULT_FAST_PATH_CONFIG.prefer_single_expert,
        "agreement_target": MIN_AGREEMENT,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fast_path_replay.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (out_dir / "fast_path_replay.md").write_text(_markdown(report), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=str, default="", dest="out",
        help="Artifact directory (default: data/experiments/<run-id>)",
    )
    args = parser.parse_args()

    report = asyncio.run(run_replay())
    out_dir = Path(args.out) if args.out else Path("data/experiments") / report.run_id
    _write_artifacts(report, out_dir)

    print(_markdown(report))
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
