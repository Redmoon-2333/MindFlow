"""Tests for mindflow.eval.runner — metric correctness and comparison logic.

Covers:
  - Jaccard index computation (known values)
  - run_eval with known hits/misses
  - EvalReport aggregation
  - compare() win/loss/tie counting
"""

from __future__ import annotations

from dataclasses import dataclass

from mindflow.domain.evidence import EvidenceBundle, EvidenceItem
from mindflow.domain.procrastination import (
    BehaviorSummary,
    CBTTechnique,
    ProcrastinationType,
)
from mindflow.eval.runner import EvalReport, ScenarioResult, _compute_jaccard, compare, run_eval
from mindflow.eval.scenarios import EvalScenario

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EMPTY_TUPLE: tuple = ()


def _make_stub_bundle() -> EvidenceBundle:
    return EvidenceBundle(
        user_id=1,
        window=(__import__("datetime").datetime(2026, 7, 18, 9, 0),
                __import__("datetime").datetime(2026, 7, 18, 11, 0)),
        items=(
            EvidenceItem(metric="focus_score", value=0.5, baseline=0.7,
                         severity="moderate", confidence=0.8, source="test",
                         human_readable="test"),
        ),
        behavior_summary=BehaviorSummary(
            intended_task="test", duration_min=60.0, actual_focus_min=30.0,
            context_switches_per_hour=10.0, longest_focus_block_s=300.0,
            social_media_ratio=0.3, start_delay_min=5.0,
            keyword_flags=frozenset(), baseline_deviation=None,
        ),
        intervention_history=_EMPTY_TUPLE,
        novelty_flags=_EMPTY_TUPLE,
    )


@dataclass
class _StubAssessment:
    """Minimal assessment-like object for testing run_eval."""
    types: tuple[ProcrastinationType, ...]
    recommended_technique: CBTTechnique | None
    source: str = "stub"


def _make_scenario(
    sid: str,
    expected_types: tuple[ProcrastinationType, ...],
    expected_technique: CBTTechnique | None,
) -> EvalScenario:
    return EvalScenario(
        scenario_id=sid,
        description="test scenario",
        bundle=_make_stub_bundle(),
        expected_types=expected_types,
        expected_technique=expected_technique,
    )


# ===================================================================
# Jaccard index
# ===================================================================


class TestJaccard:
    """_compute_jaccard correctness for known inputs."""

    def test_identical_sets(self) -> None:
        t = (ProcrastinationType.IMPULSIVITY,)
        assert _compute_jaccard(t, t) == 1.0

    def test_disjoint_sets(self) -> None:
        a = (ProcrastinationType.IMPULSIVITY,)
        b = (ProcrastinationType.DECISIONAL,)
        assert _compute_jaccard(a, b) == 0.0

    def test_partial_overlap(self) -> None:
        a = (ProcrastinationType.IMPULSIVITY, ProcrastinationType.DECISIONAL)
        b = (ProcrastinationType.IMPULSIVITY,)
        assert _compute_jaccard(a, b) == 0.5  # intersection={IMP}, union={IMP, DEC}

    def test_three_and_one(self) -> None:
        a = (ProcrastinationType.IMPULSIVITY, ProcrastinationType.DECISIONAL, ProcrastinationType.PERFECTIONISM)
        b = (ProcrastinationType.IMPULSIVITY,)
        assert _compute_jaccard(a, b) == 1 / 3

    def test_both_empty(self) -> None:
        assert _compute_jaccard((), ()) == 1.0

    def test_one_empty(self) -> None:
        a = (ProcrastinationType.IMPULSIVITY,)
        assert _compute_jaccard(a, ()) == 0.0

    def test_same_multiple(self) -> None:
        a = (ProcrastinationType.TASK_AVERSION, ProcrastinationType.EMOTIONAL_REGULATION)
        b = (ProcrastinationType.TASK_AVERSION, ProcrastinationType.EMOTIONAL_REGULATION)
        assert _compute_jaccard(a, b) == 1.0


# ===================================================================
# run_eval
# ===================================================================


class TestRunEval:
    """run_eval correctness with known predictions."""

    async def _all_hit(self, bundle: EvidenceBundle) -> _StubAssessment:  # noqa: ARG002
        return _StubAssessment(
            types=(ProcrastinationType.IMPULSIVITY,),
            recommended_technique=CBTTechnique.STIMULUS_CONTROL,
        )

    async def _all_miss(self, bundle: EvidenceBundle) -> _StubAssessment:  # noqa: ARG002
        return _StubAssessment(
            types=(ProcrastinationType.DECISIONAL,),
            recommended_technique=CBTTechnique.GOAL_SETTING,
        )

    async def _mixed(self, bundle: EvidenceBundle) -> _StubAssessment:  # noqa: ARG002
        return _StubAssessment(
            types=(ProcrastinationType.IMPULSIVITY, ProcrastinationType.DECISIONAL),
            recommended_technique=CBTTechnique.STIMULUS_CONTROL,
        )

    async def test_all_hit(self) -> None:
        scenario = _make_scenario(
            "TST-001",
            (ProcrastinationType.IMPULSIVITY,),
            CBTTechnique.STIMULUS_CONTROL,
        )
        report = await run_eval(self._all_hit, (scenario,), analyzer_name="test")
        assert report.top1_accuracy == 1.0
        assert report.mean_jaccard == 1.0
        assert report.technique_accuracy == 1.0
        assert report.total == 1
        assert report.hits == 1

    async def test_all_miss(self) -> None:
        scenario = _make_scenario(
            "TST-002",
            (ProcrastinationType.IMPULSIVITY,),
            CBTTechnique.STIMULUS_CONTROL,
        )
        report = await run_eval(self._all_miss, (scenario,), analyzer_name="test")
        assert report.top1_accuracy == 0.0
        assert report.mean_jaccard == 0.0
        assert report.technique_accuracy == 0.0
        assert report.hits == 0

    async def test_mixed_prediction(self) -> None:
        """Predicted {IMP, DEC}, expected {IMP} → partial Jaccard."""
        scenario = _make_scenario(
            "TST-003",
            (ProcrastinationType.IMPULSIVITY,),
            CBTTechnique.STIMULUS_CONTROL,
        )
        report = await run_eval(self._mixed, (scenario,), analyzer_name="test")
        # Top-1 hit: predicted[0]=IMP == expected[0]=IMP ✓
        assert report.top1_accuracy == 1.0
        # Jaccard: intersection={IMP}, union={IMP, DEC} = 0.5
        assert report.mean_jaccard == 0.5
        # Technique matches
        assert report.technique_accuracy == 1.0

    async def test_half_half(self) -> None:
        """Two scenarios: one hit, one miss → 50% accuracy."""
        async def analyzer(bundle: EvidenceBundle) -> _StubAssessment:  # noqa: ARG002
            return _StubAssessment(
                types=(ProcrastinationType.IMPULSIVITY,),
                recommended_technique=CBTTechnique.STIMULUS_CONTROL,
            )

        scenarios = (
            _make_scenario("HIT-001", (ProcrastinationType.IMPULSIVITY,), CBTTechnique.STIMULUS_CONTROL),
            _make_scenario("MISS-001", (ProcrastinationType.DECISIONAL,), CBTTechnique.GOAL_SETTING),
        )
        report = await run_eval(analyzer, scenarios, analyzer_name="test")
        assert report.top1_accuracy == 0.5
        assert report.total == 2
        assert report.hits == 1
        assert report.misses == 1

    async def test_empty_scenarios(self) -> None:
        report = await run_eval(self._all_hit, (), analyzer_name="test")
        assert report.top1_accuracy == 0.0
        assert report.total == 0

    async def test_source_propagation(self) -> None:
        async def analyzer(bundle: EvidenceBundle) -> _StubAssessment:  # noqa: ARG002
            return _StubAssessment(
                types=(ProcrastinationType.IMPULSIVITY,),
                recommended_technique=CBTTechnique.STIMULUS_CONTROL,
                source="custom_source",
            )
        scenario = _make_scenario(
            "TST-004",
            (ProcrastinationType.IMPULSIVITY,),
            CBTTechnique.STIMULUS_CONTROL,
        )
        report = await run_eval(analyzer, (scenario,), analyzer_name="test")
        assert report.scenario_results[0].predicted_source == "custom_source"


# ===================================================================
# compare
# ===================================================================


class TestCompare:
    """compare() win/loss/tie counting."""

    def _make_report(
        self,
        results: list[ScenarioResult],
        name: str = "test",
    ) -> EvalReport:
        total = len(results)
        hits = sum(1 for r in results if r.top1_hit)
        jaccards = [r.jaccard for r in results]
        techniques = sum(1 for r in results if r.technique_match)
        return EvalReport(
            analyzer_name=name,
            scenario_results=tuple(results),
            top1_accuracy=hits / total if total else 0.0,
            mean_jaccard=sum(jaccards) / total if total else 0.0,
            technique_accuracy=techniques / total if total else 0.0,
            timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            total=total,
            hits=hits,
            misses=total - hits,
        )

    def _result(self, sid: str, hit: bool) -> ScenarioResult:
        return ScenarioResult(
            scenario_id=sid,
            description="",
            expected_types=(ProcrastinationType.IMPULSIVITY,),
            predicted_types=(ProcrastinationType.IMPULSIVITY,) if hit else (ProcrastinationType.DECISIONAL,),
            expected_technique=CBTTechnique.STIMULUS_CONTROL,
            predicted_technique=CBTTechnique.STIMULUS_CONTROL if hit else CBTTechnique.GOAL_SETTING,
            top1_hit=hit,
            jaccard=1.0 if hit else 0.0,
            technique_match=hit,
            predicted_source="test",
        )

    def test_baseline_wins(self) -> None:
        """Baseline hits, panel misses → baseline_wins=1."""
        baseline = self._make_report([self._result("S-001", hit=True)])
        panel = self._make_report([self._result("S-001", hit=False)])
        cr = compare(baseline, panel)
        assert cr.baseline_wins == 1
        assert cr.panel_wins == 0
        assert cr.ties == 0

    def test_panel_wins(self) -> None:
        """Baseline misses, panel hits → panel_wins=1."""
        baseline = self._make_report([self._result("S-001", hit=False)])
        panel = self._make_report([self._result("S-001", hit=True)])
        cr = compare(baseline, panel)
        assert cr.baseline_wins == 0
        assert cr.panel_wins == 1
        assert cr.ties == 0

    def test_tie_both_hit(self) -> None:
        baseline = self._make_report([self._result("S-001", hit=True)])
        panel = self._make_report([self._result("S-001", hit=True)])
        cr = compare(baseline, panel)
        assert cr.baseline_wins == 0
        assert cr.panel_wins == 0
        assert cr.ties == 1

    def test_tie_both_miss(self) -> None:
        baseline = self._make_report([self._result("S-001", hit=False)])
        panel = self._make_report([self._result("S-001", hit=False)])
        cr = compare(baseline, panel)
        assert cr.baseline_wins == 0
        assert cr.panel_wins == 0
        assert cr.ties == 1

    def test_mixed_results(self) -> None:
        """3 scenarios: baseline wins 1, panel wins 1, tie 1."""
        outcomes = [
            (True, False),   # baseline wins
            (False, True),   # panel wins
            (True, True),    # tie
        ]
        baseline_results = [self._result(f"S-{i}", b) for i, (b, _) in enumerate(outcomes)]
        panel_results = [self._result(f"S-{i}", p) for i, (_, p) in enumerate(outcomes)]
        baseline = self._make_report(baseline_results)
        panel = self._make_report(panel_results)
        cr = compare(baseline, panel)
        assert cr.baseline_wins == 1
        assert cr.panel_wins == 1
        assert cr.ties == 1

    def test_deltas_computed(self) -> None:
        baseline = self._make_report(
            [self._result("S-001", hit=True), self._result("S-002", hit=False)],
        )
        panel = self._make_report(
            [self._result("S-001", hit=True), self._result("S-002", hit=True)],
        )
        cr = compare(baseline, panel)
        assert cr.top1_delta == 0.5  # panel 1.0 - baseline 0.5
