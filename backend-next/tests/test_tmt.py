"""Tests for ``domain/tmt.py`` — TMT decomposition over behavioural proxies."""

from __future__ import annotations

import math

import pytest

from mindflow.domain.tmt import (
    HIGH_LEVEL_THRESHOLD,
    LOW_LEVEL_THRESHOLD,
    PRIOR,
    TMT_NOTE,
    TmtBreakdown,
    TmtComponent,
    TmtInputs,
    compute_tmt,
    deviation_z_mean,
)


def _full_inputs(**overrides: object) -> TmtInputs:
    """A complete input set whose components are all clearly informative."""
    base: dict[str, object] = {
        "feedback_scores": [4.0, 5.0],
        "task_priority": 5.0,
        "task_count": 2,
        "productive_share": 0.5,
        "deviation_z_mean": 0.0,
        "hours_to_deadline": 24.0,
    }
    base.update(overrides)
    return TmtInputs(**base)  # type: ignore[arg-type]


def _z_for(impulsiveness: float) -> float:
    """Inverse of the logistic: the z whose ``I`` is *impulsiveness*."""
    return math.log(impulsiveness / (1.0 - impulsiveness))


def _hours_for(delay: float) -> float:
    """Inverse of ``D = h / (h + 24)``: the hours whose ``D`` is *delay*."""
    return 24.0 * delay / (1.0 - delay)


def _z_for(impulsiveness: float) -> float:
    """Inverse of the logistic: the z whose ``I`` is *impulsiveness*."""
    return math.log(impulsiveness / (1.0 - impulsiveness))


def _hours_for(delay: float) -> float:
    """Inverse of ``D = h / (h + 24)``: the hours whose ``D`` is *delay*."""
    return 24.0 * delay / (1.0 - delay)


class TestBounds:
    def test_every_component_stays_in_the_unit_interval(self) -> None:
        for inputs in (
            TmtInputs(),
            _full_inputs(),
            _full_inputs(feedback_scores=[1.0] * 50, deviation_z_mean=-40.0),
            _full_inputs(feedback_scores=[5.0] * 50, deviation_z_mean=40.0),
            _full_inputs(hours_to_deadline=-100.0),
            _full_inputs(hours_to_deadline=10_000.0),
        ):
            breakdown = compute_tmt(inputs)
            assert 0.0 <= breakdown.motivation <= 1.0
            for component in breakdown.components.as_mapping().values():
                assert 0.0 <= component.score <= 1.0

    def test_motivation_matches_the_tmt_formula(self) -> None:
        breakdown = compute_tmt(_full_inputs())
        c = breakdown.components
        expected = (
            c.expectancy.score
            * c.value.score
            / (1.0 + c.impulsiveness.score * c.delay.score)
        )
        assert breakdown.motivation == pytest.approx(expected, abs=1e-4)

    def test_dataclasses_reject_out_of_range_scores(self) -> None:
        with pytest.raises(ValueError):
            TmtComponent(score=1.5, raw=None, label="x", source="y")
        with pytest.raises(ValueError):
            TmtBreakdown(
                motivation=1.5,
                motivation_level="high",
                dominant_barrier="value",
                components=compute_tmt(TmtInputs()).components,
            )

    def test_to_dict_matches_contract_shape(self) -> None:
        payload = compute_tmt(_full_inputs()).to_dict()
        assert set(payload) == {
            "motivation",
            "motivation_level",
            "dominant_barrier",
            "components",
            "note",
        }
        assert set(payload["components"]) == {
            "expectancy",
            "value",
            "impulsiveness",
            "delay",
        }
        assert set(payload["components"]["expectancy"]) == {
            "score",
            "raw",
            "label",
            "source",
            "evidence_ids",
        }
        assert payload["note"] == TMT_NOTE


class TestExpectancy:
    def test_more_successful_sessions_raise_expectancy(self) -> None:
        low = compute_tmt(_full_inputs(feedback_scores=[1.0, 2.0])).components
        high = compute_tmt(_full_inputs(feedback_scores=[4.0, 5.0])).components
        assert high.expectancy.score > low.expectancy.score

    def test_shrinkage_towards_the_prior(self) -> None:
        # A single 5/5 session cannot reach 1.0: the 0.5 prior has weight 3.
        single = compute_tmt(_full_inputs(feedback_scores=[5.0])).components
        expected = (1.0 + PRIOR * 3.0) / (1.0 + 3.0)
        assert single.expectancy.score == pytest.approx(expected, abs=1e-4)
        assert single.expectancy.raw == pytest.approx(5.0)

    def test_no_feedback_falls_back_to_the_documented_prior(self) -> None:
        component = compute_tmt(_full_inputs(feedback_scores=[])).components.expectancy
        assert component.score == PRIOR
        assert component.raw is None
        assert component.source == "先验（无近14天自评）"

    def test_out_of_range_scores_are_ignored(self) -> None:
        component = compute_tmt(
            _full_inputs(feedback_scores=[0.0, 6.0, 4.0])
        ).components.expectancy
        assert "1 次自评" in component.source


class TestValue:
    def test_higher_priority_raises_value(self) -> None:
        low = compute_tmt(_full_inputs(task_priority=1.0)).components.value
        high = compute_tmt(_full_inputs(task_priority=5.0)).components.value
        assert high.score > low.score

    def test_productive_share_is_blended_in(self) -> None:
        without = compute_tmt(
            _full_inputs(task_priority=5.0, productive_share=None)
        ).components.value
        with_share = compute_tmt(
            _full_inputs(task_priority=5.0, productive_share=0.0)
        ).components.value
        assert without.score == pytest.approx(1.0)
        assert with_share.score == pytest.approx(0.7)

    def test_no_tasks_falls_back_to_the_prior(self) -> None:
        component = compute_tmt(
            _full_inputs(task_priority=None, task_count=0, productive_share=0.9)
        ).components.value
        assert component.score == PRIOR
        assert component.source == "先验"


class TestImpulsiveness:
    def test_positive_z_raises_impulsiveness(self) -> None:
        low = compute_tmt(_full_inputs(deviation_z_mean=-2.0)).components
        high = compute_tmt(_full_inputs(deviation_z_mean=2.0)).components
        assert high.impulsiveness.score > low.impulsiveness.score

    def test_zero_z_is_exactly_the_logistic_midpoint(self) -> None:
        component = compute_tmt(_full_inputs(deviation_z_mean=0.0)).components.impulsiveness
        assert component.score == pytest.approx(0.5)

    def test_missing_deviation_falls_back_to_the_prior(self) -> None:
        component = compute_tmt(
            _full_inputs(deviation_z_mean=None)
        ).components.impulsiveness
        assert component.score == PRIOR
        assert component.source == "先验"

    def test_large_z_saturates_without_overflow(self) -> None:
        component = compute_tmt(_full_inputs(deviation_z_mean=1000.0)).components
        assert component.impulsiveness.score == pytest.approx(1.0)
        negative = compute_tmt(_full_inputs(deviation_z_mean=-1000.0)).components
        assert negative.impulsiveness.score == pytest.approx(0.0)


class TestDelay:
    def test_far_deadline_raises_delay(self) -> None:
        near = compute_tmt(_full_inputs(hours_to_deadline=1.0)).components.delay
        far = compute_tmt(_full_inputs(hours_to_deadline=168.0)).components.delay
        assert far.score > near.score

    def test_half_point_is_24_hours(self) -> None:
        component = compute_tmt(_full_inputs(hours_to_deadline=24.0)).components.delay
        assert component.score == pytest.approx(0.5)

    def test_overdue_clamps_to_zero(self) -> None:
        component = compute_tmt(_full_inputs(hours_to_deadline=-12.0)).components.delay
        assert component.score == 0.0
        assert component.raw == 0.0

    def test_no_deadline_falls_back_to_the_prior(self) -> None:
        component = compute_tmt(_full_inputs(hours_to_deadline=None)).components.delay
        assert component.score == PRIOR
        assert component.source == "无截止时间"


class TestDeviationZMean:
    def test_switch_features_enter_positively(self) -> None:
        deviation = {
            "features": [
                {"feature": "app_switch_count", "robust_z": 2.0},
                {"feature": "domain_switch_count", "robust_z": 1.0},
            ]
        }
        assert deviation_z_mean(deviation) == pytest.approx(1.5)

    def test_longest_segment_enters_negatively(self) -> None:
        # A long uninterrupted segment (high z) means *less* impulsiveness.
        deviation = {"features": [{"feature": "longest_segment_ratio", "robust_z": 2.0}]}
        assert deviation_z_mean(deviation) == pytest.approx(-2.0)

    def test_other_features_are_ignored(self) -> None:
        deviation = {"features": [{"feature": "idle_ratio", "robust_z": 9.0}]}
        assert deviation_z_mean(deviation) is None

    @pytest.mark.parametrize("value", [None, {}, {"features": "nope"}, {"features": []}])
    def test_malformed_payloads_yield_none(self, value: object) -> None:
        assert deviation_z_mean(value) is None  # type: ignore[arg-type]

    def test_non_numeric_z_is_skipped(self) -> None:
        deviation = {
            "features": [
                {"feature": "app_switch_count", "robust_z": "1.0"},
                {"feature": "app_switch_count", "robust_z": True},
                {"feature": "app_switch_count", "robust_z": 1.0},
            ]
        }
        assert deviation_z_mean(deviation) == pytest.approx(1.0)


class TestDominantBarrier:
    def test_low_expectancy_dominates(self) -> None:
        breakdown = compute_tmt(
            _full_inputs(feedback_scores=[1.0] * 20, task_priority=5.0)
        )
        assert breakdown.components.expectancy.score < 0.2
        assert breakdown.dominant_barrier == "expectancy"

    def test_low_value_dominates(self) -> None:
        breakdown = compute_tmt(
            _full_inputs(feedback_scores=[5.0] * 20, task_priority=1.0, productive_share=0.0)
        )
        assert breakdown.dominant_barrier == "value"

    def test_high_impulsiveness_dominates(self) -> None:
        breakdown = compute_tmt(
            _full_inputs(
                feedback_scores=[5.0] * 20,
                task_priority=5.0,
                productive_share=1.0,
                deviation_z_mean=6.0,
                # Impulsiveness only costs motivation when there is delay to
                # discount (the formula's I·D product), so the deadline must
                # be far enough away for I to be the binding barrier.
                hours_to_deadline=168.0,
            )
        )
        assert breakdown.components.impulsiveness.score > 0.9
        assert breakdown.dominant_barrier == "impulsiveness"

    def _ideal_ev_inputs(self, **overrides: object) -> TmtInputs:
        """E and V pinned near their ideals so the I·D joint gain dominates."""
        base: dict[str, object] = {
            "feedback_scores": [5.0] * 30,
            "task_priority": 5.0,
            "productive_share": 1.0,
        }
        base.update(overrides)
        return _full_inputs(**base)

    def test_long_delay_is_reported_as_delay(self) -> None:
        # D = 0.9 vs I = 0.3: the joint I·D gain wins and delay is the larger
        # factor, so the user-facing barrier is the deadline.
        breakdown = compute_tmt(
            self._ideal_ev_inputs(
                deviation_z_mean=_z_for(0.3),
                hours_to_deadline=_hours_for(0.9),
            )
        )
        assert breakdown.components.delay.score == pytest.approx(0.9)
        assert breakdown.components.impulsiveness.score == pytest.approx(0.3)
        assert breakdown.dominant_barrier == "delay"

    def test_high_impulsiveness_is_reported_as_impulsiveness(self) -> None:
        breakdown = compute_tmt(
            self._ideal_ev_inputs(
                deviation_z_mean=_z_for(0.9),
                hours_to_deadline=_hours_for(0.3),
            )
        )
        assert breakdown.components.impulsiveness.score == pytest.approx(0.9)
        assert breakdown.components.delay.score == pytest.approx(0.3)
        assert breakdown.dominant_barrier == "impulsiveness"

    def test_equal_i_and_d_resolve_to_impulsiveness(self) -> None:
        # The I·D product is symmetric, so equal scores are genuinely
        # ambiguous; the canonical order breaks the tie to impulsiveness.
        breakdown = compute_tmt(
            self._ideal_ev_inputs(deviation_z_mean=0.0, hours_to_deadline=24.0)
        )
        assert breakdown.components.impulsiveness.score == pytest.approx(0.5)
        assert breakdown.components.delay.score == pytest.approx(0.5)
        assert breakdown.dominant_barrier == "impulsiveness"

    def test_expectancy_still_wins_when_it_has_the_largest_gain(self) -> None:
        # Same ideal V, but E is collapsed and I·D is only mildly costly, so
        # idealising E gains more than removing the product.
        breakdown = compute_tmt(
            self._ideal_ev_inputs(
                feedback_scores=[1.0] * 30,
                deviation_z_mean=_z_for(0.35),
                hours_to_deadline=_hours_for(0.35),
            )
        )
        assert breakdown.dominant_barrier == "expectancy"

    def test_choice_is_deterministic(self) -> None:
        first = compute_tmt(_full_inputs()).dominant_barrier
        second = compute_tmt(_full_inputs()).dominant_barrier
        assert first == second


class TestLevels:
    def test_all_prior_inputs_are_medium(self) -> None:
        breakdown = compute_tmt(TmtInputs())
        assert breakdown.motivation == pytest.approx(0.2)
        assert breakdown.motivation_level == "medium"

    def test_low_motivation_band(self) -> None:
        breakdown = compute_tmt(
            _full_inputs(feedback_scores=[1.0] * 30, task_priority=1.0, productive_share=0.0)
        )
        assert breakdown.motivation < LOW_LEVEL_THRESHOLD
        assert breakdown.motivation_level == "low"

    def test_high_motivation_band(self) -> None:
        breakdown = compute_tmt(
            _full_inputs(
                feedback_scores=[5.0] * 30,
                task_priority=5.0,
                productive_share=1.0,
                deviation_z_mean=-8.0,
                hours_to_deadline=0.0,
            )
        )
        assert breakdown.motivation >= HIGH_LEVEL_THRESHOLD
        assert breakdown.motivation_level == "high"

    def test_note_states_the_boundary(self) -> None:
        note = compute_tmt(TmtInputs()).note
        assert "不是心理测量" in note
        assert "临床诊断" in note
        assert note == TMT_NOTE

    def test_math_domain_is_respected(self) -> None:
        # Guard against an accidental overflow/NaN path in the logistic.
        component = compute_tmt(_full_inputs(deviation_z_mean=0.0)).components
        assert not math.isnan(component.impulsiveness.score)
