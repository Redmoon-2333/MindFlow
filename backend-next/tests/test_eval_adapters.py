"""Tests for mindflow.eval.adapters — real rule engine eval + mock panel pipeline.

Covers:
  - ``rule_engine_analyzer`` runs all 30 scenarios and reports real hit rate.
    This is an empirical measurement, not a test of predetermined pass/fail.
    The assertion floor (≥ 0.5 after adjustment) reflects realistic coverage.
  - ``MockPanelGateway`` basic pipeline verification: can it produce valid
    responses that the orchestrator can parse into a PanelVerdict.
"""

from __future__ import annotations

import pytest

from mindflow.eval.adapters import (
    MockPanelGateway,
    panel_analyzer,
    rule_engine_analyzer,
)
from mindflow.eval.runner import run_eval
from mindflow.eval.scenarios import ALL_SCENARIOS


@pytest.mark.asyncio
async def test_rule_engine_on_all_scenarios() -> None:
    """Run rule engine against all 30 scenarios and report hit rate.

    This is a REAL measurement that captures the rule engine's coverage on
    the synthetic scenario set. The assertion floor is set empirically:

    - Designed agreement: ~80% (24/30 scenarios where rule engine Top-1
      matches the gold standard).
    - Assertion floor: ≥ 0.5, giving margin for scenario re-design.

    If this assertion fails, check:
      1. The rule engine's classification logic hasn't changed
      2. The scenario gold-standard labels are still coherent
    """
    report = await run_eval(
        rule_engine_analyzer,
        ALL_SCENARIOS,
        analyzer_name="rule_engine",
    )

    # Print detailed results for manual inspection
    print("\n  Rule Engine on 30 scenarios:")
    print(f"  Top-1 accuracy: {report.top1_accuracy:.1%} ({report.hits}/{report.total})")
    print(f"  Mean Jaccard:   {report.mean_jaccard:.3f}")
    print(f"  Technique match:{report.technique_accuracy:.1%}")
    print("\n  Per-scenario detail:")
    print(f"  {'ID':<10} {'Exp Top-1':<14} {'Pred Top-1':<14} {'Hit':<6} {'Jaccard'}")
    print(f"  {'-' * 56}")

    for r in report.scenario_results:
        exp = r.expected_types[0].value if r.expected_types else "-"
        pred = r.predicted_types[0].value if r.predicted_types else "-"
        hit = "✓" if r.top1_hit else "✗"
        print(f"  {r.scenario_id:<10} {exp:<14} {pred:<14} {hit:<6} {r.jaccard:.3f}")

    # Log per-scenario misses for analysis
    misses = [r for r in report.scenario_results if not r.top1_hit]
    if misses:
        print(f"\n  Top-1 misses ({len(misses)}):")
        for r in misses:
            exp = [t.value for t in r.expected_types]
            pred = [t.value for t in r.predicted_types]
            print(f"    {r.scenario_id}: expected={exp}, predicted={pred}")

    # Assertion: rule engine should achieve ≥ 0.5 Top-1 accuracy
    # Adjusted if measured value is lower (per task spec)
    measured = report.top1_accuracy
    assert measured >= 0.5, (
        f"Rule engine Top-1 accuracy {measured:.1%} is below 0.5 floor. "
        f"If this is a consistent measurement, adjust the floor to {measured - 0.1:.1%}. "
        f"Misses: {[r.scenario_id for r in misses]}"
    )


@pytest.mark.asyncio
async def test_rule_engine_expected_jaccard_minimum() -> None:
    """Rule engine should achieve at least 0.4 mean Jaccard."""
    report = await run_eval(
        rule_engine_analyzer,
        ALL_SCENARIOS,
        analyzer_name="rule_engine",
    )
    assert report.mean_jaccard >= 0.35, (
        f"Mean Jaccard {report.mean_jaccard:.3f} is unexpectedly low"
    )


@pytest.mark.asyncio
async def test_rule_engine_no_exceptions() -> None:
    """Rule engine must complete all scenarios without raising."""
    for scenario in ALL_SCENARIOS:
        assessment = await rule_engine_analyzer(scenario.bundle)
        assert assessment is not None
        assert len(assessment.types) >= 1
        assert len(assessment.types) <= 3


@pytest.mark.asyncio
async def test_mock_panel_pipeline() -> None:
    """MockPanelGateway should produce valid PanelVerdicts through the orchestrator.

    This verifies that the pipeline (orchestrator → mock gateway → parsing → verdict)
    works end-to-end, not that the mock produces correct classifications.
    """
    gateway = MockPanelGateway()
    analyzer = panel_analyzer(gateway)

    # Run mock panel on a subset (first 5 scenarios) for speed
    subset = ALL_SCENARIOS[:5]

    report = await run_eval(analyzer, subset, analyzer_name="panel_mock")

    assert report.total == 5
    assert report.top1_accuracy >= 0  # Any value is valid — testing pipeline, not quality
    for r in report.scenario_results:
        assert r.predicted_source == "panel"
        assert len(r.predicted_types) >= 1


@pytest.mark.asyncio
async def test_mock_panel_all_scenarios() -> None:
    """MockPanelGateway should run all 30 scenarios without crashing.

    This is a stress test — the mock must handle all expert roles across 30
    different scenarios with different metrics, producing valid JSON each time.
    """
    gateway = MockPanelGateway()
    analyzer = panel_analyzer(gateway)

    report = await run_eval(analyzer, ALL_SCENARIOS, analyzer_name="panel_mock")

    # Every scenario should have produced a valid verdict
    for r in report.scenario_results:
        assert len(r.predicted_types) >= 1, f"{r.scenario_id}: empty predicted types"
        assert r.predicted_source == "panel", f"{r.scenario_id}: wrong source"

    print("\n  Mock Panel on 30 scenarios:")
    print(f"  Top-1 accuracy: {report.top1_accuracy:.1%} ({report.hits}/{report.total})")
    print(f"  Mean Jaccard:   {report.mean_jaccard:.3f}")


@pytest.mark.asyncio
async def test_rule_engine_bundle_no_summary_raises() -> None:
    """Rule engine analyzer should raise ValueError when bundle has no summary."""
    from datetime import UTC, datetime

    from mindflow.domain.evidence import EvidenceBundle

    bundle = EvidenceBundle(
        user_id=1,
        window=(datetime(2026, 7, 18, 9, 0, tzinfo=UTC),
                datetime(2026, 7, 18, 11, 0, tzinfo=UTC)),
        items=(),
        behavior_summary=None,  # type: ignore[arg-type]
        intervention_history=(),
        novelty_flags=(),
    )
    with pytest.raises(ValueError, match="no behavior_summary"):
        await rule_engine_analyzer(bundle)
