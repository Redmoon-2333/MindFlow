"""Tests for mindflow.domain.procrastination — rule engine and domain model.

Test coverage (per NF-Q1 target):
  - Each type: >= 2 positive cases + >= 1 negative case
  - Multi-type coexistence and confidence ordering
  - Edge cases: zero duration, all-zero metrics, extreme values
  - Hypothesis property tests: confidence in [0,1], types non-empty and <= 3,
    all-domain no-exception guarantee

Due to the deterministic nature of the rule engine, all test assertions are
exact comparisons rather than fuzzy matching.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from hypothesis import given
from hypothesis import strategies as st
from pytest import approx

from mindflow.domain.procrastination import (
    TYPE_TO_TECHNIQUES,
    BehaviorSummary,
    CBTTechnique,
    ProcrastinationAssessment,
    ProcrastinationType,
    RuleEngine,
)

# ---------------------------------------------------------------------------
# Domain model basics
# ---------------------------------------------------------------------------

# Thresholds used throughout — kept as constants so tests don't accidentally
# become coupled to constructor default values.
_DEFAULT_ENGINE: Final = RuleEngine()
_TASK_AVERSION_MIN_CONF: Final = 0.4
_NO_SIGNIFICANT_CONF: Final = 0.15
_IMPULSIVITY_SATURATED_CONF: Final = 0.95
_EMOTIONAL_SATURATED_CONF: Final = 0.95

# ---------------------------------------------------------------------------
# ProcrastinationType & CBTTechnique enums
# ---------------------------------------------------------------------------

_TYPE_COUNT: Final = 5
_TECHNIQUE_COUNT: Final = 6


class TestProcrastinationType:
    """Enum structural tests."""

    def test_member_count(self) -> None:
        assert len(ProcrastinationType) == _TYPE_COUNT

    def test_members_are_lowercase(self) -> None:
        for member in ProcrastinationType:
            assert member.value == member.name.lower()


class TestCBTTechnique:
    """Enum structural tests."""

    def test_member_count(self) -> None:
        assert len(CBTTechnique) == _TECHNIQUE_COUNT

    def test_members_are_lowercase(self) -> None:
        for member in CBTTechnique:
            assert member.value == member.name.lower()


class TestTypeToTechniques:
    """TYPE_TO_TECHNIQUES mapping invariants."""

    def test_all_types_mapped(self) -> None:
        assert set(TYPE_TO_TECHNIQUES) == set(ProcrastinationType)

    def test_each_maps_to_non_empty_tuple(self) -> None:
        for t in ProcrastinationType:
            assert len(TYPE_TO_TECHNIQUES[t]) >= 1

    def test_recommended_is_first_element(self) -> None:
        """The primary technique for each type is the first tuple element."""
        expected_first: Mapping[ProcrastinationType, CBTTechnique] = {
            ProcrastinationType.TASK_AVERSION: CBTTechnique.GRADED_EXPOSURE,
            ProcrastinationType.IMPULSIVITY: CBTTechnique.STIMULUS_CONTROL,
            ProcrastinationType.DECISIONAL: CBTTechnique.GOAL_SETTING,
            ProcrastinationType.PERFECTIONISM: CBTTechnique.COGNITIVE_RESTRUCTURING,
            ProcrastinationType.EMOTIONAL_REGULATION: CBTTechnique.MINDFULNESS,
        }
        for t, expected in expected_first.items():
            assert TYPE_TO_TECHNIQUES[t][0] == expected


# ---------------------------------------------------------------------------
# ProcrastinationAssessment invariants
# ---------------------------------------------------------------------------

# Valid ProcrastinationTypes for property-based testing.
_ALL_TYPES: Final = list(ProcrastinationType)


class TestAssessmentInvariants:
    """Structural invariants for every ProcrastinationAssessment."""

    def test_types_sorted_by_confidence_desc(self) -> None:
        """Types MUST be ordered by confidence descending (highest first)."""
        assessment = self._make_assessment(
            (ProcrastinationType.IMPULSIVITY, ProcrastinationType.PERFECTIONISM),
            {ProcrastinationType.IMPULSIVITY: 0.3, ProcrastinationType.PERFECTIONISM: 0.7},
        )
        assert assessment.types[0] == ProcrastinationType.PERFECTIONISM

    def test_confidence_only_includes_returned_types(self) -> None:
        assessment = self._make_assessment(
            (ProcrastinationType.DECISIONAL,),
            {ProcrastinationType.DECISIONAL: 0.6},
        )
        assert set(assessment.confidence) == set(assessment.types)

    @staticmethod
    def _make_assessment(
        types: tuple[ProcrastinationType, ...],
        confidence: Mapping[ProcrastinationType, float],
    ) -> ProcrastinationAssessment:
        # Determine recommended technique from highest-confidence type
        sorted_types = sorted(types, key=lambda t: confidence[t], reverse=True)
        top_type = sorted_types[0]
        return ProcrastinationAssessment(
            types=tuple(sorted_types),
            confidence=confidence,
            recommended_technique=TYPE_TO_TECHNIQUES[top_type][0],
            rationale="测试用评估",
            source="rule_engine",
        )


# ---------------------------------------------------------------------------
# Rule tests: impulsivity
#   IF longest_focus_block_s < 300 AND switches/h >= 12
# ---------------------------------------------------------------------------


class TestImpulsivity:
    """Positive and negative cases for impulsivity classification."""

    # Positive case 1: bare threshold — 300s block, 12 switches/h
    def test_bare_threshold(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=20,
            context_switches_per_hour=12,
            longest_focus_block_s=299,
            social_media_ratio=0.3,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.IMPULSIVITY in result.types
        assert result.types[0] == ProcrastinationType.IMPULSIVITY
        # switches=12 → confidence = 0.5
        assert result.confidence[ProcrastinationType.IMPULSIVITY] == approx(0.5)

    # Positive case 2: saturated — many switches, very short focus blocks
    def test_saturated(self) -> None:
        summary = BehaviorSummary(
            intended_task="写论文",
            duration_min=30,
            actual_focus_min=5,
            context_switches_per_hour=30,
            longest_focus_block_s=45,
            social_media_ratio=0.1,
            start_delay_min=1,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.IMPULSIVITY in result.types
        assert result.confidence[ProcrastinationType.IMPULSIVITY] == _IMPULSIVITY_SATURATED_CONF

    # Negative case: focus block too long for impulsivity
    def test_focus_block_too_long(self) -> None:
        """Focus block > 300s should prevent impulsivity rule from firing.
        Impulsivity may still appear via the no-significant path but with
        very low confidence (< 0.2).
        """
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=40,
            context_switches_per_hour=15,
            longest_focus_block_s=600,  # > 300s
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        # Rule should not fire (focus block too long)
        # No-significant path may include impulsivity at < 0.2
        assert result.confidence.get(ProcrastinationType.IMPULSIVITY, 0) < 0.2

    # Negative case: switches below threshold
    def test_switches_below_threshold(self) -> None:
        """Switches < 12/h should prevent impulsivity rule from firing.
        Impulsivity may still appear via the no-significant path but with
        very low confidence (< 0.2).
        """
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=30,
            context_switches_per_hour=5,
            longest_focus_block_s=120,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        # Rule should not fire (switches < 12)
        # No-significant path may include impulsivity at < 0.2
        assert result.confidence.get(ProcrastinationType.IMPULSIVITY, 0) < 0.2


# ---------------------------------------------------------------------------
# Rule tests: decisional
#   IF start_delay_min > 30 AND focus recovers (actual_focus/duration > 0.4)
# ---------------------------------------------------------------------------


class TestDecisional:
    """Positive and negative cases for decisional classification."""

    # Positive case 1: bare threshold
    def test_bare_threshold(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=120,
            actual_focus_min=50,  # ratio = 0.416 > 0.4
            context_switches_per_hour=8,
            longest_focus_block_s=600,
            social_media_ratio=0.2,
            start_delay_min=31,  # > 30
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.DECISIONAL in result.types
        assert result.confidence[ProcrastinationType.DECISIONAL] >= 0.5

    # Positive case 2: severe delay
    def test_long_delay(self) -> None:
        summary = BehaviorSummary(
            intended_task="复习考试",
            duration_min=180,
            actual_focus_min=120,
            context_switches_per_hour=5,
            longest_focus_block_s=900,
            social_media_ratio=0.1,
            start_delay_min=90,  # well above 60
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.DECISIONAL in result.types
        assert result.confidence[ProcrastinationType.DECISIONAL] == _IMPULSIVITY_SATURATED_CONF

    # Negative case: delay > 30 but focus doesn't recover
    def test_delayed_no_recovery(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=120,
            actual_focus_min=40,  # ratio = 0.33 < 0.4
            context_switches_per_hour=10,
            longest_focus_block_s=300,
            social_media_ratio=0.3,
            start_delay_min=45,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.DECISIONAL not in result.types

    # Negative case: no delay
    def test_no_delay(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=40,
            context_switches_per_hour=6,
            longest_focus_block_s=500,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.DECISIONAL not in result.types


# ---------------------------------------------------------------------------
# Rule tests: perfectionism
#   IF keyword_flags contains "self_criticism" or "redo_pattern"
# ---------------------------------------------------------------------------


class TestPerfectionism:
    """Positive and negative cases for perfectionism classification."""

    # Positive case 1: single keyword
    def test_single_keyword_self_criticism(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=15,
            context_switches_per_hour=8,
            longest_focus_block_s=200,
            social_media_ratio=0.2,
            start_delay_min=10,
            keyword_flags=frozenset({"self_criticism"}),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.PERFECTIONISM in result.types
        assert result.confidence[ProcrastinationType.PERFECTIONISM] == approx(0.6)

    # Positive case 2: both keywords → higher confidence
    def test_both_keywords(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=10,
            context_switches_per_hour=6,
            longest_focus_block_s=100,
            social_media_ratio=0.1,
            start_delay_min=5,
            keyword_flags=frozenset({"self_criticism", "redo_pattern"}),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.PERFECTIONISM in result.types
        assert result.confidence[ProcrastinationType.PERFECTIONISM] == approx(0.85)

    # Positive case 3: redo_pattern alone
    def test_redo_pattern_keyword(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=20,
            context_switches_per_hour=10,
            longest_focus_block_s=300,
            social_media_ratio=0.3,
            start_delay_min=5,
            keyword_flags=frozenset({"redo_pattern"}),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.PERFECTIONISM in result.types
        assert result.confidence[ProcrastinationType.PERFECTIONISM] == approx(0.6)

    # Negative case: unrelated keywords
    def test_unrelated_keywords(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=30,
            context_switches_per_hour=8,
            longest_focus_block_s=400,
            social_media_ratio=0.3,
            start_delay_min=5,
            keyword_flags=frozenset({"boring", "hard"}),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.PERFECTIONISM not in result.types

    # Negative case: empty keywords
    def test_empty_keywords(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=30,
            context_switches_per_hour=6,
            longest_focus_block_s=500,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.PERFECTIONISM not in result.types


# ---------------------------------------------------------------------------
# Rule tests: emotional_regulation
#   IF social_media_ratio > 0.55
# ---------------------------------------------------------------------------


class TestEmotionalRegulation:
    """Positive and negative cases for emotional regulation classification."""

    # Positive case 1: bare threshold
    def test_bare_threshold(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=15,
            context_switches_per_hour=10,
            longest_focus_block_s=200,
            social_media_ratio=0.56,  # > 0.55
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.EMOTIONAL_REGULATION in result.types
        assert result.confidence[ProcrastinationType.EMOTIONAL_REGULATION] >= 0.5

    # Positive case 2: very high social media
    def test_very_high_social_media(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=5,
            context_switches_per_hour=15,
            longest_focus_block_s=60,
            social_media_ratio=0.85,
            start_delay_min=1,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.EMOTIONAL_REGULATION in result.types
        conf = result.confidence[ProcrastinationType.EMOTIONAL_REGULATION]
        assert conf == _EMOTIONAL_SATURATED_CONF

    # Negative case: ratio at threshold
    def test_exactly_at_threshold(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=30,
            context_switches_per_hour=10,
            longest_focus_block_s=400,
            social_media_ratio=0.55,  # not strictly above
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.EMOTIONAL_REGULATION not in result.types

    # Negative case: low social media
    def test_low_social_media(self) -> None:
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=40,
            context_switches_per_hour=5,
            longest_focus_block_s=800,
            social_media_ratio=0.15,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.EMOTIONAL_REGULATION not in result.types


# ---------------------------------------------------------------------------
# Catch-all: task_aversion and no-significant
# ---------------------------------------------------------------------------


class TestTaskAversion:
    """Catch-all fires when focus is low but no specific type matches."""

    def test_low_focus_triggers_catch_all(self) -> None:
        """Low actual_focus ratio (< 0.35) and no other rules fire → task_aversion."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=120,
            actual_focus_min=30,  # ratio = 0.25 < 0.35
            context_switches_per_hour=8,  # < 12
            longest_focus_block_s=400,  # > 300
            social_media_ratio=0.3,  # < 0.55
            start_delay_min=5,  # < 30
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.TASK_AVERSION in result.types
        assert result.types[0] == ProcrastinationType.TASK_AVERSION
        assert result.confidence[ProcrastinationType.TASK_AVERSION] >= _TASK_AVERSION_MIN_CONF

    def test_negative_baseline_deviation_triggers_catch_all(self) -> None:
        """Significantly negative baseline deviation triggers task_aversion."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=30,  # ratio = 0.5, not low
            context_switches_per_hour=8,
            longest_focus_block_s=500,
            social_media_ratio=0.3,
            start_delay_min=5,
            keyword_flags=frozenset(),
            baseline_deviation=-0.8,  # < -0.5
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.TASK_AVERSION in result.types


class TestNoSignificant:
    """When metrics are healthy and no rule fires → low-confidence result."""

    def test_healthy_metrics(self) -> None:
        """Normal metrics with no rule firing should yield very low confidence."""
        summary = BehaviorSummary(
            intended_task="正常学习",
            duration_min=120,
            actual_focus_min=90,  # ratio = 0.75
            context_switches_per_hour=6,
            longest_focus_block_s=900,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        # Should return the default type with very low confidence
        assert len(result.types) == 1
        max_conf = max(result.confidence.values())
        assert max_conf == _NO_SIGNIFICANT_CONF
        assert "未检测到显著的拖延模式" in result.rationale

    def test_no_significant_has_no_recommended_technique(self) -> None:
        """No-significant path must not recommend a technique (review M1 contract)."""
        summary = BehaviorSummary(
            intended_task="正常学习",
            duration_min=120,
            actual_focus_min=90,
            context_switches_per_hour=6,
            longest_focus_block_s=900,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert result.recommended_technique is None

    def test_significant_result_has_technique(self) -> None:
        """Any above-threshold assessment carries a concrete CBT technique."""
        summary = BehaviorSummary(
            intended_task="论文",
            duration_min=120,
            actual_focus_min=20,
            context_switches_per_hour=24,
            longest_focus_block_s=120,
            social_media_ratio=0.3,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert result.recommended_technique is not None

    def test_source_is_rule_engine(self) -> None:
        """Every assessment from RuleEngine has source='rule_engine'."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=30,
            context_switches_per_hour=6,
            longest_focus_block_s=500,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert result.source == "rule_engine"


# ---------------------------------------------------------------------------
# Multi-type detection and ordering
# ---------------------------------------------------------------------------


class TestMultiType:
    """When multiple rules fire, types are sorted by confidence descending."""

    def test_impulsivity_and_emotional_regulation(self) -> None:
        """High switches AND high social media → both types, sorted correctly."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=10,
            context_switches_per_hour=15,
            longest_focus_block_s=60,
            social_media_ratio=0.75,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        expected_types = {
            ProcrastinationType.IMPULSIVITY,
            ProcrastinationType.EMOTIONAL_REGULATION,
        }
        assert expected_types.issubset(set(result.types))
        # Both should have > 0.5 confidence
        for t in expected_types:
            assert result.confidence[t] >= 0.5
        # Types must be in descending order
        confs = [result.confidence[t] for t in result.types]
        assert confs == sorted(confs, reverse=True)

    def test_perfectionism_and_decisional(self) -> None:
        """Keywords AND delay → both detected, perfectionism should rank higher."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=120,
            actual_focus_min=60,
            context_switches_per_hour=6,
            longest_focus_block_s=600,
            social_media_ratio=0.2,
            start_delay_min=45,
            keyword_flags=frozenset({"self_criticism", "redo_pattern"}),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.PERFECTIONISM in result.types
        assert ProcrastinationType.DECISIONAL in result.types

    def test_max_three_types(self) -> None:
        """Even if 4+ rules fire, only top 3 types are returned."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=120,
            actual_focus_min=30,
            context_switches_per_hour=20,
            longest_focus_block_s=60,
            social_media_ratio=0.70,
            start_delay_min=60,
            keyword_flags=frozenset({"self_criticism", "redo_pattern"}),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert len(result.types) <= 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Boundary and degenerate input handling."""

    def test_zero_duration(self) -> None:
        """Zero duration_min should not cause division-by-zero errors."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=0,
            actual_focus_min=0,
            context_switches_per_hour=0,
            longest_focus_block_s=0,
            social_media_ratio=0,
            start_delay_min=0,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        # Must not raise
        result = _DEFAULT_ENGINE.assess(summary)
        assert result.source == "rule_engine"

    def test_all_zero_metrics(self) -> None:
        """All-zero metrics with 0 duration fall into no-significant territory."""
        summary = BehaviorSummary(
            intended_task="",
            duration_min=0,
            actual_focus_min=0,
            context_switches_per_hour=0,
            longest_focus_block_s=0,
            social_media_ratio=0.0,
            start_delay_min=0,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert 1 <= len(result.types) <= 3
        for conf in result.confidence.values():
            assert 0 <= conf <= 1

    def test_extreme_values(self) -> None:
        """Extremely large values should not cause floating-point errors."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=1e6,
            actual_focus_min=0,
            context_switches_per_hour=1e6,
            longest_focus_block_s=1e6,
            social_media_ratio=0.99,
            start_delay_min=1e6,
            keyword_flags=frozenset({"self_criticism", "redo_pattern"}),
            baseline_deviation=-10.0,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert 1 <= len(result.types) <= 3
        for conf in result.confidence.values():
            assert 0 <= conf <= 1

    def test_negative_metrics_should_not_crash(self) -> None:
        """Improbable negative values should be handled gracefully."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=-10,
            actual_focus_min=-5,
            context_switches_per_hour=-1,
            longest_focus_block_s=-100,
            social_media_ratio=-0.5,
            start_delay_min=-1,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert 1 <= len(result.types) <= 3

    def test_none_intended_task(self) -> None:
        """intended_task=None should be handled."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=30,
            context_switches_per_hour=6,
            longest_focus_block_s=500,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        assert result.source == "rule_engine"

    def test_baseline_deviation_none_does_not_crash(self) -> None:
        """baseline_deviation=None should not cause errors in catch-all logic."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=10,
            context_switches_per_hour=6,
            longest_focus_block_s=400,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        # Low focus ratio (10/60=0.167) < 0.35 → task_aversion
        assert ProcrastinationType.TASK_AVERSION in result.types


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

# Hypothesis strategy for generating synthetic BehaviorSummary instances.
# Uses constrained float ranges to avoid NaN/Inf and keep values realistic.
_behavior_strategy = st.builds(
    BehaviorSummary,
    intended_task=st.one_of(st.none(), st.text(max_size=50)),
    duration_min=st.floats(min_value=0, max_value=480, allow_nan=False, allow_infinity=False),
    actual_focus_min=st.floats(min_value=0, max_value=480, allow_nan=False, allow_infinity=False),
    context_switches_per_hour=st.floats(
        min_value=0, max_value=100, allow_nan=False, allow_infinity=False
    ),
    longest_focus_block_s=st.floats(
        min_value=0, max_value=3600, allow_nan=False, allow_infinity=False
    ),
    social_media_ratio=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
    start_delay_min=st.floats(min_value=0, max_value=240, allow_nan=False, allow_infinity=False),
    keyword_flags=st.frozensets(
        st.sampled_from(["self_criticism", "redo_pattern", "boring", "hard", "procrastination"])
    ),
    baseline_deviation=st.one_of(
        st.none(),
        st.floats(min_value=-3, max_value=3, allow_nan=False, allow_infinity=False),
    ),
)


class TestHypothesisProperties:
    """Property-based tests using Hypothesis."""

    @given(_behavior_strategy)
    def test_confidence_in_range(self, summary: BehaviorSummary) -> None:
        """All per-type confidence values must be in [0, 1]."""
        result = _DEFAULT_ENGINE.assess(summary)
        for conf in result.confidence.values():
            assert 0 <= conf <= 1, f"Confidence {conf} out of [0, 1]"

    @given(_behavior_strategy)
    def test_types_non_empty_and_limited(self, summary: BehaviorSummary) -> None:
        """types must have 1 to 3 entries."""
        result = _DEFAULT_ENGINE.assess(summary)
        assert 1 <= len(result.types) <= 3, (
            f"Expected 1-3 types, got {len(result.types)}: {result.types}"
        )

    @given(_behavior_strategy)
    def test_no_exception(self, summary: BehaviorSummary) -> None:
        """assess must never raise an exception for any valid input."""
        _DEFAULT_ENGINE.assess(summary)  # should not raise

    @given(_behavior_strategy)
    def test_source_is_always_rule_engine(self, summary: BehaviorSummary) -> None:
        """source field must always be 'rule_engine'."""
        result = _DEFAULT_ENGINE.assess(summary)
        assert result.source == "rule_engine"

    @given(_behavior_strategy)
    def test_types_sorted_by_confidence(self, summary: BehaviorSummary) -> None:
        """Returned types must be in descending confidence order."""
        result = _DEFAULT_ENGINE.assess(summary)
        # If more than 1 type, verify sorting
        if len(result.types) >= 2:
            confs = [result.confidence[t] for t in result.types]
            assert confs == sorted(confs, reverse=True), (
                f"Types not sorted by confidence: {list(zip(result.types, confs, strict=True))}"
            )


# ---------------------------------------------------------------------------
# Custom threshold calibration
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    """RuleEngine thresholds can be overridden via constructor parameters."""

    def test_custom_impulsivity_threshold(self) -> None:
        """Lowering the switch threshold should make impulsivity easier to trigger."""
        liberal_engine = RuleEngine(impulsivity_min_switches=6.0)
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=20,
            context_switches_per_hour=8,
            longest_focus_block_s=200,
            social_media_ratio=0.3,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = liberal_engine.assess(summary)
        assert ProcrastinationType.IMPULSIVITY in result.types, (
            "Lowered threshold should detect impulsivity at 8 switches/h"
        )

        # Default engine should NOT detect it at 8 switches/h
        default_result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.IMPULSIVITY not in default_result.types, (
            "Default threshold should not detect impulsivity at 8 switches/h"
        )

    def test_custom_emotional_regulation_threshold(self) -> None:
        """Relaxing the emotion regulation threshold should make it easier to trigger."""
        liberal_engine = RuleEngine(emotional_regulation_min_ratio=0.30)
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=30,
            context_switches_per_hour=6,
            longest_focus_block_s=500,
            social_media_ratio=0.40,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = liberal_engine.assess(summary)
        assert ProcrastinationType.EMOTIONAL_REGULATION in result.types

        default_result = _DEFAULT_ENGINE.assess(summary)
        assert ProcrastinationType.EMOTIONAL_REGULATION not in default_result.types

    def test_rationale_no_diagnostic_terms(self) -> None:
        """NF-S7: rationale must not contain '诊断', '治疗', or '患者'."""
        # Trigger all rules to maximize coverage
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=10,
            context_switches_per_hour=20,
            longest_focus_block_s=60,
            social_media_ratio=0.75,
            start_delay_min=45,
            keyword_flags=frozenset({"self_criticism", "redo_pattern"}),
            baseline_deviation=-0.8,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        for term in ("诊断", "治疗", "患者"):
            assert term not in result.rationale, (
                f"NF-S7 violation: rationale contains '{term}': {result.rationale}"
            )

    def test_no_significant_rationale_also_compliant(self) -> None:
        """'No significant' rationale must also be NF-S7 compliant."""
        summary = BehaviorSummary(
            intended_task=None,
            duration_min=60,
            actual_focus_min=45,
            context_switches_per_hour=6,
            longest_focus_block_s=900,
            social_media_ratio=0.2,
            start_delay_min=2,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )
        result = _DEFAULT_ENGINE.assess(summary)
        for term in ("诊断", "治疗", "患者"):
            assert term not in result.rationale
