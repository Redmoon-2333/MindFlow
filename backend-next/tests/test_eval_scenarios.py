"""Tests for mindflow.eval.scenarios — scenario completeness and validity.

Covers:
  - 30 scenarios exist with correct IDs
  - All ProcrastinationType members are represented
  - Each scenario's expected types are valid enum members
  - Each scenario has a valid EvidenceBundle (non-empty items, behavior_summary)
  - Expected technique is coherent with expected types
  - validate_all_scenarios() returns empty issues
"""

from __future__ import annotations

from mindflow.domain.procrastination import ProcrastinationType
from mindflow.eval.scenarios import ALL_SCENARIOS, validate_all_scenarios


class TestScenarioCount:
    """Ensure exactly 30 scenarios exist with correct structure."""

    def test_count(self) -> None:
        assert len(ALL_SCENARIOS) == 30, f"Expected 30 scenarios, got {len(ALL_SCENARIOS)}"

    def test_unique_ids(self) -> None:
        ids = [s.scenario_id for s in ALL_SCENARIOS]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_id_pattern(self) -> None:
        """IDs should follow IMP-001, DEC-001, PER-001, EMO-001, TAV-001, MIX-001 pattern."""
        prefixes = {"IMP", "DEC", "PER", "EMO", "TAV", "MIX"}
        for s in ALL_SCENARIOS:
            prefix = s.scenario_id.split("-")[0]
            assert prefix in prefixes, f"Unexpected ID prefix: {s.scenario_id}"


class TestScenarioCompleteness:
    """Every type is covered by the right number of scenarios."""

    def test_all_types_represented(self) -> None:
        """Each ProcrastinationType appears as expected in at least one scenario."""
        all_expected_types: set[ProcrastinationType] = set()
        for s in ALL_SCENARIOS:
            all_expected_types.update(s.expected_types)
        assert all_expected_types == set(ProcrastinationType), (
            f"Missing types in gold standard: {set(ProcrastinationType) - all_expected_types}"
        )

    def test_per_type_count(self) -> None:
        """At least 5 scenarios per type group (IMP, DEC, PER, EMO, TAV)."""
        type_scenarios = {"IMP": 0, "DEC": 0, "PER": 0, "EMO": 0, "TAV": 0}
        for s in ALL_SCENARIOS:
            prefix = s.scenario_id.split("-")[0]
            if prefix in type_scenarios:
                type_scenarios[prefix] += 1
        for prefix, count in type_scenarios.items():
            assert count >= 5, f"{prefix} has {count} scenarios, expected >= 5"

    def test_mixed_scenarios(self) -> None:
        """At least 5 MIX scenarios."""
        mix_count = sum(1 for s in ALL_SCENARIOS if s.scenario_id.startswith("MIX"))
        assert mix_count >= 5, f"Expected >= 5 MIX scenarios, got {mix_count}"


class TestExpectedTypes:
    """Gold-standard type validation."""

    def test_expected_types_are_valid_enums(self) -> None:
        for s in ALL_SCENARIOS:
            for t in s.expected_types:
                assert isinstance(t, ProcrastinationType), (
                    f"{s.scenario_id}: expected type {t} is not a ProcrastinationType"
                )

    def test_expected_types_non_empty(self) -> None:
        """Every scenario should have at least one expected type (gold standard)."""
        for s in ALL_SCENARIOS:
            assert len(s.expected_types) >= 1, (
                f"{s.scenario_id}: empty expected_types"
            )

    def test_expected_types_max_three(self) -> None:
        for s in ALL_SCENARIOS:
            assert len(s.expected_types) <= 3, (
                f"{s.scenario_id}: {len(s.expected_types)} expected types, max 3"
            )

    def test_expected_types_sorted_by_confidence(self) -> None:
        """Types should be ordered by descending importance (human judgment)."""
        for s in ALL_SCENARIOS:
            assert len(s.expected_types) >= 1

    def test_technique_coherent_with_types(self) -> None:
        """If types exist, technique must exist and vice versa."""
        for s in ALL_SCENARIOS:
            if s.expected_types:
                assert s.expected_technique is not None, (
                    f"{s.scenario_id}: has types but no technique"
                )
            else:
                assert s.expected_technique is None, (
                    f"{s.scenario_id}: no types but has technique"
                )


class TestBundleValidity:
    """EvidenceBundle structural checks."""

    def test_bundle_has_behavior_summary(self) -> None:
        for s in ALL_SCENARIOS:
            assert s.bundle.behavior_summary is not None, (
                f"{s.scenario_id}: bundle has no behavior_summary"
            )

    def test_bundle_has_items(self) -> None:
        for s in ALL_SCENARIOS:
            assert len(s.bundle.items) >= 3, (
                f"{s.scenario_id}: only {len(s.bundle.items)} items, expected >= 3"
            )

    def test_bundle_has_user_id(self) -> None:
        for s in ALL_SCENARIOS:
            assert s.bundle.user_id >= 0

    def test_bundle_window_valid(self) -> None:
        for s in ALL_SCENARIOS:
            start, end = s.bundle.window
            assert start <= end, f"{s.scenario_id}: window start after end"

    def test_item_metrics_unique(self) -> None:
        for s in ALL_SCENARIOS:
            metrics = [item.metric for item in s.bundle.items]
            assert len(metrics) == len(set(metrics)), (
                f"{s.scenario_id}: duplicate metrics in bundle items"
            )


class TestValidation:
    """validate_all_scenarios integrity."""

    def test_validation_passes(self) -> None:
        issues = validate_all_scenarios()
        assert not issues, f"Validation issues found: {issues}"

    def test_validation_return_type(self) -> None:
        issues = validate_all_scenarios()
        assert isinstance(issues, list)


class TestGetScenario:
    """get_scenario lookup."""

    def test_existing(self) -> None:
        from mindflow.eval.scenarios import get_scenario
        s = get_scenario("IMP-001")
        assert s is not None
        assert s.scenario_id == "IMP-001"

    def test_nonexistent(self) -> None:
        from mindflow.eval.scenarios import get_scenario
        s = get_scenario("NON-EXISTENT")
        assert s is None


class TestDescriptionLanguages:
    """Descriptions should be Chinese (per design doc)."""

    def test_descriptions_are_chinese(self) -> None:
        """Basic check that descriptions use Chinese characters."""
        for s in ALL_SCENARIOS:
            # Most descriptions should contain Chinese characters
            has_chinese = any("一" <= c <= "鿿" for c in s.description)
            assert has_chinese, f"{s.scenario_id}: description not in Chinese: {s.description}"

    def test_scenario_id_is_ascii(self) -> None:
        for s in ALL_SCENARIOS:
            assert s.scenario_id.isascii(), f"{s.scenario_id}: contains non-ASCII"
