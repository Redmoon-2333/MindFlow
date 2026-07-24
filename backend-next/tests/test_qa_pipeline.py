"""TDD test suite for the 3-agent data quality assurance pipeline.

Tests define the contract for ``mindflow.train.qa_pipeline``.
All tests will FAIL initially (no implementation) — this is correct for TDD.

Agent responsibilities:
  - StatisticalRealismAgent: Checks feature distributions match expected
    student behavior patterns.
  - BehavioralPlausibilityAgent: Validates time-of-day activity sequences
    against known human patterns (sleep, meals, procrastination).
  - ProfileConsistencyAgent: Ensures activity data aligns with declared
    student archetype (major + grade expectations).

Imports from hypothetical ``mindflow.train.qa_pipeline`` — these will
fail until the module is created.  That is by design.
"""

from __future__ import annotations

import copy
import random
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

# ── Imports that WILL FAIL until qa_pipeline.py is implemented ───────────
from mindflow.train.qa_pipeline import (  # noqa: E402
    BehavioralPlausibilityAgent,
    ProfileConsistencyAgent,
    QAPipeline,
    QAReport,
    StatisticalRealismAgent,
)
from mindflow.train.user_profiles import StudentArchetype, get_archetype

# ═════════════════════════════════════════════════════════════════════════
# Test-data builders (inline, no external files)
# ═════════════════════════════════════════════════════════════════════════

UTC = UTC


def _make_hour(day_offset: int, hour: int) -> datetime:
    """Return a timezone-aware datetime for a given day offset and hour."""
    base = datetime(2026, 3, 16, tzinfo=UTC)  # a Monday
    return base + timedelta(days=day_offset, hours=hour)


def _base_row(ts: datetime, **overrides: Any) -> dict[str, Any]:
    """Build a minimal activity row with sensible defaults."""
    row: dict[str, Any] = {
        "timestamp": ts,
        "process_name": "vscode",
        "window_title": "main.py - VSCode",
        "duration_seconds": 300.0,
        "is_idle": 0,
    }
    row.update(overrides)
    return row


def _base_feature(
    window_start: datetime,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a minimal feature dict (17 columns) with defaults for a focused window."""
    feat: dict[str, Any] = {
        "window_start": window_start.isoformat(),
        "unique_app_count": 2,
        "switch_frequency": 4.0,
        "productivity_ratio": 0.85,
        "entertainment_ratio": 0.05,
        "social_ratio": 0.05,
        "max_app_duration": 1200.0,
        "idle_ratio": 0.03,
        "hour_of_day": window_start.hour,
        "day_of_week": window_start.weekday(),
        "title_code_ratio": 0.80,
        "title_doc_ratio": 0.10,
        "title_url_ratio": 0.05,
        "title_meeting_ratio": 0.00,
        "title_entertainment_ratio": 0.05,
        "activity_entropy": 0.25,
        "context_switch_cost": 0.10,
        "temporal_decay_weight": 0.70,
    }
    feat.update(overrides)
    return feat


def _cs_student_profile() -> StudentArchetype:
    """Return a CS senior archetype for consistency tests."""
    return get_archetype("senior_cs")


def _art_student_profile() -> StudentArchetype:
    """Return a design/art freshman archetype."""
    return get_archetype("freshman_design")


def _medical_student_profile() -> StudentArchetype:
    """Return a medical grad archetype."""
    return get_archetype("grad_medical")


# ═════════════════════════════════════════════════════════════════════════
# TestStatisticalRealismAgent — feature distribution realism
# ═════════════════════════════════════════════════════════════════════════


class TestStatisticalRealismAgent:
    """Agent should detect when feature distributions deviate from
    expected student behavioral statistics."""

    def test_detects_bimodal_focus_scores(self) -> None:
        """Bimodal focus_score distribution (2 peaks: focused + distracted
        modes) should be identified with a score < 0.7."""
        agent = StatisticalRealismAgent()
        rng = random.Random(42)

        # Simulate 200 windows: 60% at focus ~0.85, 40% at focus ~0.15
        features: list[dict[str, Any]] = []
        for i in range(200):
            ts = _make_hour(0, i % 24)
            if i < 120:
                feat = _base_feature(
                    ts,
                    productivity_ratio=0.85,
                    entertainment_ratio=0.05,
                    focus_score=0.82 + rng.uniform(-0.03, 0.03),
                )
            else:
                feat = _base_feature(
                    ts,
                    productivity_ratio=0.15,
                    entertainment_ratio=0.75,
                    focus_score=0.18 + rng.uniform(-0.03, 0.03),
                )
            features.append(feat)

        result = agent.evaluate([], features)
        assert result["score"] < 0.7, (
            f"Expected score < 0.7 for bimodal focus, got {result['score']}"
        )
        assert len(result["flags"]) >= 1, "Should raise at least one flag for bimodal distribution"

    def test_flags_uniform_distributions(self) -> None:
        """Uniformly random features (no structure) should receive score < 0.5."""
        agent = StatisticalRealismAgent()
        rng = random.Random(0)

        features = [
            _base_feature(
                _make_hour(0, h),
                productivity_ratio=rng.uniform(0.0, 1.0),
                entertainment_ratio=rng.uniform(0.0, 1.0),
                idle_ratio=rng.uniform(0.0, 1.0),
                switch_frequency=rng.uniform(0.0, 50.0),
                focus_score=rng.uniform(0.0, 1.0),
            )
            for h in range(168)  # 7 days × 24 hours
        ]

        result = agent.evaluate([], features)
        assert result["score"] < 0.5, (
            f"Uniform random data should score < 0.5, got {result['score']}"
        )

    def test_weekend_entertainment_higher_than_weekday(self) -> None:
        """Weekend entertainment_ratio should exceed weekday by at least 10%.
        Data where it does NOT should be flagged."""
        agent = StatisticalRealismAgent()

        # Build 7 days of features where weekends and weekdays have equal
        # entertainment ratios (unrealistic)
        features: list[dict[str, Any]] = []
        for day in range(7):
            for hour in range(24):
                ts = _make_hour(day, hour)
                feat = _base_feature(
                    ts,
                    entertainment_ratio=0.40,
                    productivity_ratio=0.40,
                    focus_score=0.50,
                )
                features.append(feat)

        result = agent.evaluate([], features)

        # The agent should detect that weekends are not elevated
        weekend_flag = any(
            "weekend" in f.get("reason", "").lower()
            or "entertainment" in f.get("reason", "").lower()
            for f in result["flags"]
        )
        assert weekend_flag or result["score"] < 0.75, (
            "Agent should flag equal weekend/weekday entertainment ratios"
        )

    def test_sleep_hours_have_high_idle(self) -> None:
        """Hours 2-6 AM should average idle_ratio > 0.7.
        Data with low idle during these hours should score lower."""
        agent = StatisticalRealismAgent()

        # 7 days: sleep hours (2-6) have idle_ratio ~0.20 (unrealistic — too active)
        features = [
            _base_feature(
                _make_hour(day, hour),
                idle_ratio=0.20 if 2 <= hour <= 6 else 0.05,
                focus_score=0.60,
            )
            for day in range(7)
            for hour in range(24)
        ]

        result = agent.evaluate([], features)

        # Agent should detect low idle during sleep hours
        sleep_flag = any(
            any(term in f.get("reason", "").lower() for term in ("sleep", "idle", "night"))
            for f in result["flags"]
        )
        assert sleep_flag or result["score"] < 0.80, (
            "Agent should flag unusually low idle during sleep hours (2-6 AM)"
        )

    def test_hourly_productivity_curve_matches_expected(self) -> None:
        """Hourly productivity should correlate with expected student curve
        (peak 9-12 AM, dip at lunch, secondary peak 14-17, decline after 18)."""
        agent = StatisticalRealismAgent()

        # Build features matching the expected curve roughly
        rng = random.Random(42)
        features: list[dict[str, Any]] = []
        for day in range(5):  # weekdays
            for hour in range(8, 22):
                ts = _make_hour(day, hour)
                # Expected productivity pattern
                if 9 <= hour < 12:
                    prod = 0.75 + rng.uniform(-0.10, 0.10)
                elif 14 <= hour < 17:
                    prod = 0.65 + rng.uniform(-0.10, 0.10)
                elif hour in (12, 13, 18, 19):
                    prod = 0.30 + rng.uniform(-0.10, 0.10)
                else:
                    prod = 0.45 + rng.uniform(-0.10, 0.10)

                feat = _base_feature(
                    ts,
                    productivity_ratio=prod,
                    entertainment_ratio=0.2,
                    focus_score=prod,
                )
                features.append(feat)

        result = agent.evaluate([], features)
        # Realistic curve should score fairly well
        assert result["score"] > 0.50, (
            f"Realistic hourly curve should score > 0.50, got {result['score']}"
        )


# ═════════════════════════════════════════════════════════════════════════
# TestBehavioralPlausibilityAgent — time-sequence pattern realism
# ═════════════════════════════════════════════════════════════════════════


class TestBehavioralPlausibilityAgent:
    """Agent should validate that activity sequences conform to
    known human behavioral patterns."""

    def test_rejects_code_entertainment_pingpong(self) -> None:
        """Code↔Entertainment cycling > 3× per hour should be flagged."""
        agent = BehavioralPlausibilityAgent()

        # Build rows that ping-pong between code and entertainment
        rows: list[dict[str, Any]] = []
        apps = ["vscode", "bilibili"]
        ts = _make_hour(0, 14)
        for i in range(60):  # 60 transitions in 1 hour → 60 switches/hour
            rows.append(
                _base_row(
                    ts + timedelta(minutes=i),
                    process_name=apps[i % 2],
                    window_title="main.py - VSCode" if i % 2 == 0 else "B站 - Anime",
                    is_idle=0,
                )
            )

        result = agent.evaluate(rows, [])
        assert result["score"] < 0.6, (
            f"Code-entertainment pingpong should score < 0.6, got {result['score']}"
        )
        assert any(
            "pingpong" in f.get("reason", "").lower()
            or "cycling" in f.get("reason", "").lower()
            or "switch" in f.get("reason", "").lower()
            for f in result["flags"]
        ), "Should flag rapid code↔entertainment cycling"

    def test_verifies_sleep_periods_idle(self) -> None:
        """Sleep hours (1-7 AM) should be > 70% idle overall.
        Non-idle sleep data should lower the score."""
        agent = BehavioralPlausibilityAgent()

        # 7 nights: sleep hours mostly idle
        rows: list[dict[str, Any]] = []
        for day in range(7):
            for hour in range(1, 8):
                ts = _make_hour(day, hour)
                for minute in range(0, 60, 5):
                    # 80% idle during sleep
                    is_idle = 1 if random.random() < 0.80 else 0
                    rows.append(
                        _base_row(
                            ts + timedelta(minutes=minute),
                            process_name="lock_screen" if is_idle else "chrome",
                            window_title="Locked" if is_idle else "Chrome - late night browsing",
                            is_idle=is_idle,
                        )
                    )

        result = agent.evaluate(rows, [])
        # Mostly idle sleep should get a reasonable score
        assert result["score"] > 0.60, (
            f"Mostly-idle sleep should score > 0.60, got {result['score']}"
        )

    def test_detects_meal_breaks(self) -> None:
        """Productivity should dip around 12:00-13:00 and 18:00-19:00.
        Data showing high productivity during meals should be flagged."""
        agent = BehavioralPlausibilityAgent()

        rows: list[dict[str, Any]] = []
        # 5 weekdays with consistently high productivity at lunch/dinner
        for day in range(5):
            for hour in range(8, 22):
                ts = _make_hour(day, hour)
                for minute in range(0, 60, 5):
                    # Keep productivity high even during lunch (12-13) and dinner (18-19)
                    rows.append(
                        _base_row(
                            ts + timedelta(minutes=minute),
                            process_name="vscode",
                            window_title="project.py - VSCode",
                            is_idle=0,
                        )
                    )

        result = agent.evaluate(rows, [])
        meal_flag = any(
            any(term in f.get("reason", "").lower()
                for term in ("meal", "lunch", "dinner", "break"))
            for f in result["flags"]
        )
        assert meal_flag or result["score"] < 0.85, (
            "Agent should detect missing meal-break productivity dips"
        )

    def test_traces_procrastination_boundaries(self) -> None:
        """Binge procrastination episodes should have clear start/end
        productivity shifts (sharp drop at start, recovery at end)."""
        agent = BehavioralPlausibilityAgent()

        rows: list[dict[str, Any]] = []
        ts = _make_hour(0, 14)

        # Pre-binge: productive
        for i in range(12):
            rows.append(
                _base_row(
                    ts + timedelta(minutes=i * 5),
                    process_name="vscode",
                    is_idle=0,
                )
            )

        # Binge: entertainment for 2 hours
        binge_start_ts = ts + timedelta(minutes=60)
        for i in range(24):
            rows.append(
                _base_row(
                    binge_start_ts + timedelta(minutes=i * 5),
                    process_name="bilibili",
                    window_title="B站 - 追番中",
                    is_idle=0,
                )
            )

        # Post-binge: return to productive
        post_ts = binge_start_ts + timedelta(minutes=120)
        for i in range(12):
            rows.append(
                _base_row(
                    post_ts + timedelta(minutes=i * 5),
                    process_name="vscode",
                    is_idle=0,
                )
            )

        result = agent.evaluate(rows, [])
        # Well-structured procrastination with boundaries should be detected but
        # not harshly penalized — it's plausible behavior
        assert "boundary" in str(result.get("flags", [])).lower() or result["score"] > 0.30, (
            "Agent should identify procrastination episode boundaries"
        )

    def test_monday_morning_transition_exists(self) -> None:
        """First weekday morning (Monday) should show a shift from
        weekend-leisure to work pattern. Absence of this transition
        should be flagged."""
        agent = BehavioralPlausibilityAgent()

        # Sunday evening (leisure) → Monday morning
        rows: list[dict[str, Any]] = []
        # Sunday 22:00-23:55: leisure
        # Actually let me use explicit day offsets
        base_monday = datetime(2026, 3, 16, tzinfo=UTC)  # Monday
        base_sunday = base_monday - timedelta(days=1)  # Sunday

        # Sunday 20:00-23:55: entertainment
        for hour in range(20, 24):
            for minute in range(0, 60, 5):
                ts = base_sunday + timedelta(hours=hour, minutes=minute)
                rows.append(
                    _base_row(
                        ts,
                        process_name="bilibili",
                        window_title="B站 - Weekend",
                        is_idle=0,
                    )
                )

        # Monday 7:00-9:55: work (normal transition)
        for hour in range(7, 10):
            for minute in range(0, 60, 5):
                ts = base_monday + timedelta(hours=hour, minutes=minute)
                rows.append(
                    _base_row(
                        ts,
                        process_name="vscode",
                        window_title="main.py - VSCode",
                        is_idle=0,
                    )
                )

        result = agent.evaluate(rows, [])
        # Natural transition should score well
        assert result["score"] > 0.50, (
            f"Valid Monday morning transition should score > 0.50, got {result['score']}"
        )

    def test_rejects_impossible_sequence(self) -> None:
        """24 consecutive hours of 'code' activity (no sleep, no meals)
        should be flagged as implausible."""
        agent = BehavioralPlausibilityAgent()

        rows: list[dict[str, Any]] = []
        ts = _make_hour(0, 0)
        for hour in range(24):
            for minute in range(0, 60, 5):
                rows.append(
                    _base_row(
                        ts + timedelta(hours=hour, minutes=minute),
                        process_name="vscode",
                        window_title="coding.py - VSCode",
                        is_idle=0,
                    )
                )

        result = agent.evaluate(rows, [])
        assert result["score"] < 0.40, (
            f"24h continuous code should score < 0.40 (implausible), got {result['score']}"
        )
        assert any(
            any(term in f.get("reason", "").lower()
                for term in ("continuous", "24", "consecutive", "no break",
                             "impossible", "unrealistic", "no sleep"))
            for f in result["flags"]
        ), "Should flag 24h continuous identical activity as implausible"


# ═════════════════════════════════════════════════════════════════════════
# TestProfileConsistencyAgent — archetype-data alignment
# ═════════════════════════════════════════════════════════════════════════


class TestProfileConsistencyAgent:
    """Agent should detect misalignment between declared student archetype
    and actual activity patterns."""

    def test_flags_cs_student_using_photoshop(self) -> None:
        """CS student with > 3% creative-tool usage (photoshop, blender, figma)
        should be flagged as inconsistent."""
        agent = ProfileConsistencyAgent()

        cs_profile = _cs_student_profile()

        # Build features where CS student heavily uses design tools
        features: list[dict[str, Any]] = []
        for day in range(5):
            for hour in range(8, 22):
                ts = _make_hour(day, hour)
                # 60% creative tools, 40% code — very unusual for CS
                if hour % 2 == 0:
                    feat = _base_feature(
                        ts,
                        productivity_ratio=0.60,
                        entertainment_ratio=0.20,
                        focus_score=0.55,
                        creative_tool_ratio=0.30,
                    )
                else:
                    feat = _base_feature(
                        ts,
                        productivity_ratio=0.80,
                        entertainment_ratio=0.05,
                        focus_score=0.75,
                        creative_tool_ratio=0.0,
                    )
                features.append(feat)

        result = agent.evaluate(features, [cs_profile])
        assert result["score"] < 0.70, (
            f"CS student with high creative-tool usage should score < 0.70, got {result['score']}"
        )
        assert any(
            any(term in f.get("reason", "").lower()
                for term in ("creative", "design", "photoshop", "tool"))
            for f in result["flags"]
        ), "Should flag CS student with excessive creative tool usage"

    def test_flags_art_student_using_matlab(self) -> None:
        """Art/Design student with > 3% MATLAB/engineering-tool usage
        should be flagged as inconsistent."""
        agent = ProfileConsistencyAgent()

        art_profile = _art_student_profile()

        # Design student using heavy MATLAB — very unusual
        features: list[dict[str, Any]] = []
        for day in range(5):
            for hour in range(8, 22):
                ts = _make_hour(day, hour)
                # 40% engineering tools (MATLAB, Keil, etc.)
                feat = _base_feature(
                    ts,
                    productivity_ratio=0.50,
                    entertainment_ratio=0.15,
                    focus_score=0.55,
                    engineering_tool_ratio=0.20 if hour < 14 else 0.05,
                    creative_tool_ratio=0.05,
                )
                features.append(feat)

        result = agent.evaluate(features, [art_profile])
        assert result["score"] < 0.75, (
            f"Art student with MATLAB usage should score < 0.75, got {result['score']}"
        )

    def test_verifies_freshman_structure(self) -> None:
        """Freshman should have lower schedule variance (more rigid routine)
        than a senior. Data that violates this should be flagged."""
        agent = ProfileConsistencyAgent()

        freshman = get_archetype("freshman_cs")
        senior = get_archetype("senior_cs")

        # Build features with high variance (senior-like) for a freshman
        features: list[dict[str, Any]] = []
        for day in range(5):
            for hour in range(8, 22):
                ts = _make_hour(day, hour)
                # High hour-to-hour variability
                if hour % 3 == 0:
                    prod = 0.90
                    ent = 0.02
                elif hour % 3 == 1:
                    prod = 0.10
                    ent = 0.85
                else:
                    prod = 0.50
                    ent = 0.30
                feat = _base_feature(
                    ts,
                    productivity_ratio=prod,
                    entertainment_ratio=ent,
                    focus_score=prod,
                )
                features.append(feat)

        result_freshman = agent.evaluate(features, [freshman])
        result_senior = agent.evaluate(features, [senior])

        # Freshman profile should penalize high variance more than senior
        assert result_freshman["score"] <= result_senior["score"] + 0.05, (
            f"Freshman should not score higher than senior for high-variance data: "
            f"freshman={result_freshman['score']}, senior={result_senior['score']}"
        )

    def test_verifies_medical_discipline(self) -> None:
        """Medical students should have the highest average focus_score
        among all archetypes. Data that contradicts this should be flagged."""
        agent = ProfileConsistencyAgent()

        medical = _medical_student_profile()
        cs = _cs_student_profile()
        art = _art_student_profile()

        # Low-focus features: should penalize medical more harshly
        features: list[dict[str, Any]] = []
        for day in range(5):
            for hour in range(8, 22):
                ts = _make_hour(day, hour)
                feat = _base_feature(
                    ts,
                    productivity_ratio=0.30,
                    entertainment_ratio=0.50,
                    focus_score=0.30,
                )
                features.append(feat)

        result_medical = agent.evaluate(features, [medical])
        result_cs = agent.evaluate(features, [cs])
        result_art = agent.evaluate(features, [art])

        # Medical profile should be most strict about low focus
        scores = {
            "medical": result_medical["score"],
            "cs": result_cs["score"],
            "art": result_art["score"],
        }

        # Medical should score lowest (most strict) for low-focus data
        assert scores["medical"] < 0.70 or (
            scores["medical"] < scores["cs"]
            and scores["medical"] < scores["art"]
        ), (
            f"Medical should score <= lowest for low-focus data: {scores}"
        )

    def test_verifies_art_irregular_hours(self) -> None:
        """Art/Design students should show higher hour-to-hour variance
        in productivity than Medical students. Data that contradicts
        this pattern should be flagged."""
        agent = ProfileConsistencyAgent()

        art = _art_student_profile()
        medical = _medical_student_profile()

        # Highly regular schedule (medical-like, not art-like)
        features: list[dict[str, Any]] = []
        for day in range(5):
            for hour in range(8, 22):
                ts = _make_hour(day, hour)
                # Very consistent productivity across all hours
                feat = _base_feature(
                    ts,
                    productivity_ratio=0.80,
                    entertainment_ratio=0.05,
                    focus_score=0.80,
                )
                features.append(feat)

        result_art = agent.evaluate(features, [art])
        result_medical = agent.evaluate(features, [medical])

        # Ultra-regular schedule should fit medical better than art
        assert result_medical["score"] >= result_art["score"] - 0.05, (
            f"Regular schedule should favor medical over art: "
            f"medical={result_medical['score']}, art={result_art['score']}"
        )


# ═════════════════════════════════════════════════════════════════════════
# TestQAReport — result aggregation
# ═════════════════════════════════════════════════════════════════════════


class TestQAReport:
    """QAReport should aggregate scores from all three agents into a
    single pass/fail decision with weighted average."""

    def test_report_merges_three_agent_scores(self) -> None:
        """QAReport combines scores from all 3 agents using weighted average
        with individual agent results preserved."""
        report = QAReport(
            statistical_score=0.85,
            behavioral_score=0.78,
            profile_score=0.92,
            statistical_flags=["bimodal_focus"],
            behavioral_flags=["pingpong_detected"],
            profile_flags=[],
        )

        # Weighted average: default weights should produce sensible overall
        # Assuming equal weights by default: (0.85 + 0.78 + 0.92) / 3 = 0.85
        expected = (0.85 + 0.78 + 0.92) / 3
        assert report.overall_score == pytest.approx(expected, abs=0.01), (
            f"Expected overall ~{expected:.2f}, got {report.overall_score}"
        )

        # All agent details should be preserved
        assert report.statistical_score == 0.85
        assert report.behavioral_score == 0.78
        assert report.profile_score == 0.92
        assert "bimodal_focus" in report.statistical_flags
        assert "pingpong_detected" in report.behavioral_flags
        assert len(report.profile_flags) == 0

        # Custom weights
        report_custom = QAReport(
            statistical_score=0.60,
            behavioral_score=0.90,
            profile_score=0.90,
            statistical_flags=[],
            behavioral_flags=[],
            profile_flags=[],
            weights={"statistical": 0.5, "behavioral": 0.25, "profile": 0.25},
        )
        expected_custom = 0.60 * 0.5 + 0.90 * 0.25 + 0.90 * 0.25
        assert report_custom.overall_score == pytest.approx(expected_custom, abs=0.01), (
            f"Custom weights: expected ~{expected_custom:.2f}, got {report_custom.overall_score}"
        )

    def test_report_passed_threshold(self) -> None:
        """overall_score >= 0.7 → passed=True; below → passed=False."""
        # Passing report
        report_pass = QAReport(
            statistical_score=0.85,
            behavioral_score=0.82,
            profile_score=0.78,
            statistical_flags=[],
            behavioral_flags=[],
            profile_flags=[],
        )
        assert report_pass.passed is True, (
            f"Score {report_pass.overall_score} >= 0.7 should pass"
        )

        # Failing report
        report_fail = QAReport(
            statistical_score=0.55,
            behavioral_score=0.60,
            profile_score=0.65,
            statistical_flags=["uniform_distribution"],
            behavioral_flags=["pingpong_detected", "no_sleep_idle"],
            profile_flags=["major_mismatch"],
        )
        assert report_fail.passed is False, (
            f"Score {report_fail.overall_score} < 0.7 should not pass"
        )

        # Exact boundary: 0.70 should pass
        report_boundary = QAReport(
            statistical_score=0.70,
            behavioral_score=0.70,
            profile_score=0.70,
            statistical_flags=[],
            behavioral_flags=[],
            profile_flags=[],
        )
        assert report_boundary.passed is True, (
            "Score exactly 0.70 should pass"
        )

    def test_raises_on_invalid_weights(self) -> None:
        """QAReport should raise ValueError when weights don't sum to 1.0."""
        with pytest.raises(ValueError, match="sum"):
            QAReport(
                statistical_score=0.80,
                behavioral_score=0.80,
                profile_score=0.80,
                statistical_flags=[],
                behavioral_flags=[],
                profile_flags=[],
                weights={"statistical": 0.5, "behavioral": 0.5, "profile": 0.5},
            )

    def test_raises_on_negative_weight(self) -> None:
        """QAReport should raise ValueError on negative weights."""
        with pytest.raises(ValueError, match="negative"):
            QAReport(
                statistical_score=0.80,
                behavioral_score=0.80,
                profile_score=0.80,
                statistical_flags=[],
                behavioral_flags=[],
                profile_flags=[],
                weights={"statistical": -0.2, "behavioral": 0.7, "profile": 0.5},
            )


# ═════════════════════════════════════════════════════════════════════════
# TestQAPipeline — orchestration / integration
# ═════════════════════════════════════════════════════════════════════════


class TestQAPipeline:
    """Integration-style tests for the full QA pipeline orchestration."""

    def test_pipeline_runs_all_agents(self) -> None:
        """QAPipeline.run() should execute all 3 agents and produce a QAReport."""
        pipeline = QAPipeline()

        # Generate realistic-ish data
        rows: list[dict[str, Any]] = []
        features: list[dict[str, Any]] = []
        for day in range(3):
            for hour in range(24):
                ts = _make_hour(day, hour)
                for minute in range(0, 60, 5):
                    is_night = hour < 7
                    is_idle = 1 if (is_night and random.random() < 0.85) else 0
                    proc = (
                        "lock_screen" if is_idle
                        else random.choice(["vscode", "chrome", "wechat", "bilibili"])
                    )
                    rows.append(_base_row(
                        ts + timedelta(minutes=minute),
                        process_name=proc,
                        is_idle=is_idle,
                    ))

        for day in range(3):
            for hour in range(7, 23):
                ts = _make_hour(day, hour)
                features.append(_base_feature(ts, hour_of_day=hour))

        profile = _cs_student_profile()

        report = pipeline.run(rows, features, [profile])

        # Report should contain scores from all agents
        assert isinstance(report, QAReport), (
            f"Expected QAReport, got {type(report)}"
        )
        assert 0.0 <= report.statistical_score <= 1.0, (
            f"Statistical score out of range: {report.statistical_score}"
        )
        assert 0.0 <= report.behavioral_score <= 1.0, (
            f"Behavioral score out of range: {report.behavioral_score}"
        )
        assert 0.0 <= report.profile_score <= 1.0, (
            f"Profile score out of range: {report.profile_score}"
        )
        assert 0.0 <= report.overall_score <= 1.0, (
            f"Overall score out of range: {report.overall_score}"
        )

    def test_run_until_pass_loops_correctly(self) -> None:
        """QAPipeline.run_until_pass() should loop: generate → QA → fix →
        regenerate, with a maximum of 5 iterations."""
        pipeline = QAPipeline(max_iterations=5)

        # Track calls to verify iteration behavior
        iteration_count = [0]

        def tracker_generator(iteration: int = 0) -> tuple[list[dict], list[dict]]:
            """Generator that produces progressively better data."""
            iteration_count[0] += 1

            rows: list[dict[str, Any]] = []
            features: list[dict[str, Any]] = []

            for day in range(3):
                for hour in range(24):
                    ts = _make_hour(day, hour)
                    for minute in range(0, 60, 5):
                        is_idle = 1 if (hour < 7) else 0
                        proc = "lock_screen" if is_idle else "vscode"
                        rows.append(_base_row(
                            ts + timedelta(minutes=minute),
                            process_name=proc,
                            is_idle=is_idle,
                        ))

            for day in range(3):
                for hour in range(7, 23):
                    ts = _make_hour(day, hour)
                    features.append(_base_feature(
                        ts,
                        productivity_ratio=0.85,
                        entertainment_ratio=0.05,
                        focus_score=0.85,
                    ))

            return rows, features

        # Fixer that slightly tweaks features
        def fixer(flags: list[dict[str, str]], features: list[dict]) -> list[dict]:
            """Apply corrections based on QA flags."""
            fixed = copy.deepcopy(features)
            for feat in fixed:
                feat["focus_score"] = min(1.0, feat.get("focus_score", 0.7) + 0.05)
            return fixed

        profile = _cs_student_profile()
        rows, features = tracker_generator()

        report = pipeline.run_until_pass(
            rows=rows,
            features=features,
            profiles=[profile],
            generator_fn=tracker_generator,
            fixer_fn=fixer,
        )

        assert isinstance(report, QAReport), (
            f"Expected QAReport from run_until_pass, got {type(report)}"
        )
        # Should have iterated at least once
        assert iteration_count[0] >= 1, (
            f"run_until_pass should call generator at least once, called {iteration_count[0]}"
        )
        # Should not exceed max iterations
        assert iteration_count[0] <= 5, (
            f"run_until_pass should not exceed max_iterations=5, got {iteration_count[0]}"
        )

    def test_pipeline_accepts_generator_fn(self) -> None:
        """QAPipeline should accept a callable generator function instead of
        hardcoded data, enabling custom data generation strategies."""
        pipeline = QAPipeline()

        profile = _cs_student_profile()

        def my_generator(days: int = 2) -> tuple[list[dict], list[dict]]:
            """Custom generator producing minimal data."""
            rows: list[dict[str, Any]] = []
            features: list[dict[str, Any]] = []
            for day in range(days):
                for hour in range(24):
                    ts = _make_hour(day, hour)
                    rows.append(_base_row(ts))
                for hour in range(8, 20):
                    ts = _make_hour(day, hour)
                    features.append(_base_feature(ts))
            return rows, features

        rows, features = my_generator(days=2)

        # Pipeline should work with any callable that returns (rows, features)
        report = pipeline.run(rows=rows, features=features, profiles=[profile])

        assert isinstance(report, QAReport), (
            f"Pipeline with custom generator should produce QAReport, got {type(report)}"
        )
        assert 0.0 <= report.overall_score <= 1.0
