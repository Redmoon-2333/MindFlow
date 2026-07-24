"""Tests for student archetype definitions in ``train/user_profiles.py``.

Verifies integrity, completeness, and correctness of the 30 student
profiles, 6 episode definitions, and helper functions.
"""

from __future__ import annotations

import pytest

from mindflow.train.user_profiles import (
    EPISODES,
    PROFILES,
    ProcrastinationEpisode,
    StudentArchetype,
    get_archetype,
    get_episode,
    list_archetype_ids,
)

# ── Expected constants ──────────────────────────────────────────────────────

EXPECTED_PROFILE_COUNT = 30
EXPECTED_EPISODE_COUNT = 6
EXPECTED_PATTERNS = [
    "early_morning",
    "morning_focus",
    "afternoon_mixed",
    "evening_leisure",
    "late_night",
]
EXPECTED_EPISODE_NAMES = [
    "binge_watching",
    "doom_scrolling",
    "gaming_session",
    "social_media_spiral",
    "inspiration_browsing",
    "crash_and_burn",
]


# ── Count & uniqueness ──────────────────────────────────────────────────────


class TestProfileCountAndUniqueness:
    """Verify the expected number of profiles and ID uniqueness."""

    def test_all_30_profiles_exist(self) -> None:
        """PROFILES should contain exactly 30 archetypes (5 grades × 6 majors)."""
        assert len(PROFILES) == EXPECTED_PROFILE_COUNT

    def test_all_profile_ids_unique(self) -> None:
        """Every profile in PROFILES should have a unique profile_id."""
        ids = [p.profile_id for p in PROFILES.values()]
        assert len(ids) == len(set(ids)), "Duplicate profile_id values detected"


# ── Field completeness ──────────────────────────────────────────────────────


class TestFieldCompleteness:
    """Verify every profile has all required fields populated."""

    def test_every_profile_has_all_required_fields(self) -> None:
        """Each profile must have complete app, title, and weight lists."""
        for pid, profile in PROFILES.items():
            assert isinstance(profile, StudentArchetype), f"{pid} is not a StudentArchetype"
            # All 5 pattern keys present in primary_apps
            assert set(profile.primary_apps.keys()) == set(EXPECTED_PATTERNS), (
                f"{pid}: primary_apps missing patterns"
            )
            # All 5 pattern keys present in primary_titles
            assert set(profile.primary_titles.keys()) == set(EXPECTED_PATTERNS), (
                f"{pid}: primary_titles missing patterns"
            )
            # All 5 pattern keys present in primary_weights
            assert set(profile.primary_weights.keys()) == set(EXPECTED_PATTERNS), (
                f"{pid}: primary_weights missing patterns"
            )
            # Non-empty app / title / weight lists per pattern
            for pattern in EXPECTED_PATTERNS:
                assert len(profile.primary_apps[pattern]) > 0, (
                    f"{pid}: primary_apps[{pattern}] empty"
                )
                assert len(profile.primary_titles[pattern]) > 0, (
                    f"{pid}: primary_titles[{pattern}] empty"
                )
                assert len(profile.primary_weights[pattern]) > 0, (
                    f"{pid}: primary_weights[{pattern}] empty"
                )
                # Consistency: apps, titles, weights all same length
                assert len(profile.primary_apps[pattern]) == len(profile.primary_titles[pattern]), (
                    f"{pid}: {pattern} apps/titles length mismatch"
                )
                assert len(profile.primary_apps[pattern]) == len(
                    profile.primary_weights[pattern]
                ), (
                    f"{pid}: {pattern} apps/weights length mismatch"
                )


# ── Weight integrity ────────────────────────────────────────────────────────


class TestWeightIntegrity:
    """Verify that probability distributions sum to one."""

    def test_app_weights_sum_to_one(self) -> None:
        """For every profile × pattern, primary_weights should sum to ~1.0."""
        for pid, profile in PROFILES.items():
            for pattern in EXPECTED_PATTERNS:
                total = sum(profile.primary_weights[pattern])
                assert total == pytest.approx(1.0, abs=0.01), (
                    f"{pid} / {pattern}: weights sum to {total}, expected ~1.0"
                )

    def test_episode_type_weights_sum_to_one(self) -> None:
        """For every profile, episode_type_weights should sum to ~1.0."""
        for pid, profile in PROFILES.items():
            total = sum(profile.episode_type_weights.values())
            assert total == pytest.approx(1.0, abs=0.01), (
                f"{pid}: episode_type_weights sum to {total}, expected ~1.0"
            )


# ── Episode definitions ─────────────────────────────────────────────────────


class TestEpisodeDefinitions:
    """Verify the episode type definitions."""

    def test_all_6_episodes_exist(self) -> None:
        """EPISODES should contain exactly 6 procrastination episodes."""
        assert len(EPISODES) == EXPECTED_EPISODE_COUNT
        for name in EXPECTED_EPISODE_NAMES:
            assert name in EPISODES, f"Missing episode: {name}"

    def test_episode_getter_works(self) -> None:
        """get_episode(\"binge_watching\") should return the correct episode."""
        episode = get_episode("binge_watching")
        assert isinstance(episode, ProcrastinationEpisode)
        assert episode.name == "binge_watching"
        # Spot-check a few known attributes
        assert episode.min_duration_hours == 1.5
        assert episode.max_duration_hours == 5.0
        assert "bilibili" in episode.apps


# ── Helper functions ────────────────────────────────────────────────────────


class TestHelperFunctions:
    """Verify the module-level public helpers."""

    def test_archetype_getter_works(self) -> None:
        """get_archetype(\"freshman_cs\") should return the CS freshman archetype."""
        archetype = get_archetype("freshman_cs")
        assert isinstance(archetype, StudentArchetype)
        assert archetype.profile_id == "freshman_cs"
        assert archetype.grade == "大一"
        assert archetype.major == "计算机/软件"

    def test_archetype_getter_raises_on_unknown(self) -> None:
        """get_archetype(\"nonexistent\") should raise KeyError."""
        with pytest.raises(KeyError):
            get_archetype("nonexistent")

    def test_list_archetype_ids(self) -> None:
        """list_archetype_ids() should return a sorted list of 30 strings."""
        ids = list_archetype_ids()
        assert isinstance(ids, list)
        assert len(ids) == EXPECTED_PROFILE_COUNT
        assert all(isinstance(i, str) for i in ids)
        # Verify sorted
        assert ids == sorted(ids)
        # Verify every PROFILES key is in the list
        assert set(ids) == set(PROFILES.keys())


# ── Data validity ───────────────────────────────────────────────────────────


class TestDataValidity:
    """Verify field-level constraints on every profile."""

    def test_schedule_fields_are_valid(self) -> None:
        """Verify typical_wake_hour and typical_sleep_hour form plausible schedules.

        If sleep > wake, it's a same-day schedule (e.g. wake=7, sleep=23).
        If sleep <= wake, it's an overnight schedule (e.g. wake=8, sleep=0).
        Both patterns are acceptable.
        """
        for pid, profile in PROFILES.items():
            wake = profile.typical_wake_hour
            sleep = profile.typical_sleep_hour
            # Hours must be in valid 0-23 range
            assert 0 <= wake <= 23, f"{pid}: wake_hour={wake} out of range"
            assert 0 <= sleep <= 23, f"{pid}: sleep_hour={sleep} out of range"
            # Same-day schedule: sleep > wake (e.g. 7 → 23)
            if sleep > wake:
                pass  # Normal same-day schedule
            # Overnight schedule: sleep <= wake (e.g. 10 → 2)
            elif sleep <= wake:
                pass  # Overnight schedule (acceptable)
            # If wake == sleep, that's degenerate — flag it
            if wake == sleep:
                pytest.fail(f"{pid}: wake_hour == sleep_hour ({wake}), degenerate schedule")

    def test_probability_fields_in_range(self) -> None:
        """daily_proc_probability must be between 0.0 and 1.0 for all profiles."""
        for pid, profile in PROFILES.items():
            prob = profile.daily_proc_probability
            assert 0.0 <= prob <= 1.0, (
                f"{pid}: daily_proc_probability={prob} outside [0.0, 1.0]"
            )
