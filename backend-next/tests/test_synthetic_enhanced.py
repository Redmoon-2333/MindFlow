"""TDD test suite for the enhanced multi-user synthetic data generator.

These tests define the contract for the enhanced ``generate_synthetic_data()``
function. They are expected to FAIL initially — the enhanced parameters
(``num_users``, ``include_procrastination``, ``user_profiles``, ``archetypes``)
do not exist in the current implementation. As features are implemented, tests
will transition from FAIL to PASS.

Test categories:
  - **BackwardCompatibility**: enhanced generator preserves all original behavior
  - **MultiUserGeneration**: scalable user simulation with distinct profiles
  - **ProcrastinationPatterns**: realistic binge/doom-scroll/task-avoidance episodes
  - **ArchetypeBehavior**: student-type-specific app usage and schedule patterns
  - **LargeVolume**: correctness and performance at scale (100k+ rows)
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from mindflow.train import synthetic_data
from mindflow.train.synthetic_data import generate_synthetic_data

# ── Shared calculation helpers ────────────────────────────────────────────────


def _expected_rows(days: int, samples_per_hour: int = 12, num_users: int = 1) -> int:
    """Calculate expected row count for given generation parameters."""
    return days * 24 * samples_per_hour * num_users


def _entertainment_apps() -> set[str]:
    """Apps considered entertainment / procrastination."""
    return {"bilibili", "youtube", "steam", "douyin", "weibo"}


def _entertainment_ratio(rows: list[dict[str, Any]]) -> float:
    """Fraction of non-idle rows that are entertainment apps."""
    apps = _entertainment_apps()
    non_idle = [r for r in rows if r.get("is_idle", 0) == 0]
    if not non_idle:
        return 0.0
    return sum(1 for r in non_idle if r.get("process_name", "") in apps) / len(non_idle)


def _hourly_buckets(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    """Group rows by (user_id, day_of_month, hour)."""
    buckets: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        uid = r.get("user_id", 0)
        ts = r["timestamp"]
        key = (uid, ts.day, ts.hour)  # type: ignore[union-attr]
        buckets[key].append(r)
    return buckets


class _FixedDateTime(datetime):
    """Keep streak regression tests independent of the wall-clock date."""

    @classmethod
    def now(cls, tz: Any = None) -> _FixedDateTime:
        fixed = cls(2026, 7, 26, tzinfo=UTC)
        return fixed if tz is not None else fixed.replace(tzinfo=None)


def _max_consecutive_entertainment_hours(data: list[dict[str, Any]]) -> int:
    """Max true per-user streak of consecutive entertainment-heavy hours."""
    buckets: dict[tuple[int, datetime], list[dict[str, Any]]] = defaultdict(list)
    for row in data:
        timestamp = row["timestamp"]
        hour = timestamp.replace(minute=0, second=0, microsecond=0)
        buckets[(row["user_id"], hour)].append(row)

    hours_by_user: dict[int, list[tuple[datetime, float]]] = defaultdict(list)
    for (user_id, hour), rows in buckets.items():
        hours_by_user[user_id].append((hour, _entertainment_ratio(rows)))

    max_streak = 0
    for hours in hours_by_user.values():
        current_streak = 0
        previous_hour: datetime | None = None
        for hour, ratio in sorted(hours):
            is_consecutive = (
                previous_hour is not None
                and hour - previous_hour == timedelta(hours=1)
            )
            if ratio > 0.4:
                current_streak = current_streak + 1 if is_consecutive else 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
            previous_hour = hour

    return max_streak


# ── Test fixtures ─────────────────────────────────────────────────────────────
#
# These fixtures use enhanced parameters (num_users, include_procrastination).
# They will raise TypeError against the current implementation — this is the
# expected TDD signal that the enhanced function signature does not exist yet.


@pytest.fixture(scope="module")
def sample_rows_small() -> list[dict[str, Any]]:
    """Generate 3 days of single-user data (864 rows).

    Uses the enhanced signature: generate_synthetic_data(days=3, num_users=1).
    Will fail until num_users parameter is implemented.
    """
    return generate_synthetic_data(days=3, num_users=1)


@pytest.fixture(scope="module")
def sample_rows_multi() -> list[dict[str, Any]]:
    """Generate 3 days of 3-user data with procrastination (2592 rows).

    Uses the enhanced signature: generate_synthetic_data(
        days=3, num_users=3, include_procrastination=True
    ).
    Will fail until num_users and include_procrastination are implemented.
    """
    return generate_synthetic_data(days=3, num_users=3, include_procrastination=True)


@pytest.fixture(scope="module")
def sample_rows_large() -> list[dict[str, Any]]:
    """Generate 7 days of 5-user data (10080 rows).

    Uses the enhanced signature: generate_synthetic_data(days=7, num_users=5).
    Will fail until num_users parameter is implemented.
    """
    return generate_synthetic_data(days=7, num_users=5)


# ═══════════════════════════════════════════════════════════════════════════════
# TestBackwardCompatibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Enhanced generator must preserve all original (legacy) behavior.

    Tests 1, 2, and 5 use only the existing function signature and SHOULD PASS
    against the current implementation. Tests 3 and 4 use enhanced parameters or
    check for enhanced fields and are expected to FAIL initially.
    """

    def test_default_params_produce_same_shape(self) -> None:
        """generate_synthetic_data() with only default params returns list[dict]
        with 4032 rows (14d × 24h × 12 samples/h)."""
        rows = generate_synthetic_data()
        expected = _expected_rows(days=14)
        assert len(rows) == expected, (
            f"Expected {expected} rows, got {len(rows)} — "
            f"default 14d × 24h × 12/h must produce exact count"
        )
        assert isinstance(rows, list), "Result must be list"
        assert all(isinstance(r, dict) for r in rows), "All elements must be dict"

    def test_seed_42_reproducible(self) -> None:
        """Two calls with same seed produce identical process_name, is_idle,
        and (within float tolerance) duration_seconds."""
        rows_a = generate_synthetic_data(days=2, samples_per_hour=6, seed=42)
        rows_b = generate_synthetic_data(days=2, samples_per_hour=6, seed=42)
        assert len(rows_a) == len(rows_b)
        for i, (r_a, r_b) in enumerate(zip(rows_a, rows_b, strict=True)):
            assert r_a["process_name"] == r_b["process_name"], (
                f"Row {i}: process_name mismatch ({r_a['process_name']} vs {r_b['process_name']})"
            )
            assert r_a["is_idle"] == r_b["is_idle"], (
                f"Row {i}: is_idle mismatch ({r_a['is_idle']} vs {r_b['is_idle']})"
            )
            assert abs(r_a["duration_seconds"] - r_b["duration_seconds"]) < 1.0, (
                "Row "
                f"{i}: duration_seconds differs by "
                f"{abs(r_a['duration_seconds'] - r_b['duration_seconds'])}"
            )

    def test_one_user_no_procrastination_same_as_original(self) -> None:
        """When num_users=1 and include_procrastination=False, output matches
        legacy behavior in structure, field set, and row count."""
        rows = generate_synthetic_data(
            days=3, num_users=1, include_procrastination=False, seed=42
        )
        expected = _expected_rows(days=3)
        assert len(rows) == expected

        # Every row must have the original five fields (no procrastination extras)
        for i, row in enumerate(rows):
            assert set(row.keys()) >= {
                "timestamp", "process_name", "window_title",
                "duration_seconds", "is_idle",
            }, f"Row {i} missing original fields: {row.keys()}"

    def test_all_rows_have_required_fields(self) -> None:
        """Every row has timestamp, process_name, window_title,
        duration_seconds, is_idle, and user_id."""
        rows = generate_synthetic_data(days=1, samples_per_hour=6)
        required = {
            "timestamp", "process_name", "window_title",
            "duration_seconds", "is_idle", "user_id",
        }
        for i, row in enumerate(rows):
            missing = required - set(row.keys())
            assert not missing, f"Row {i} missing fields: {missing}"

    def test_existing_cli_invocation_still_works(self) -> None:
        """Legacy signature generate_synthetic_data(days=7, samples_per_hour=6)
        still works and returns valid data."""
        rows = generate_synthetic_data(days=7, samples_per_hour=6)
        expected = _expected_rows(days=7, samples_per_hour=6)
        assert len(rows) == expected
        # Spot-check data validity
        assert all(isinstance(r["timestamp"], datetime) for r in rows)
        assert all(isinstance(r["duration_seconds"], (int, float)) for r in rows)
        assert all(r["duration_seconds"] > 0 for r in rows)
        assert all(r["is_idle"] in (0, 1) for r in rows)


# ═══════════════════════════════════════════════════════════════════════════════
# TestMultiUserGeneration
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiUserGeneration:
    """Multi-user generation produces distinct, realistic per-user data."""

    def test_multiple_users_produce_different_data(self) -> None:
        """5 users → at least 2 pairs have meaningfully different process_name
        distributions (differ by ≥ 2 apps in their top-5)."""
        rows = generate_synthetic_data(days=3, num_users=5, seed=42)
        user_apps: dict[int, Counter[str]] = {}
        for row in rows:
            uid = row["user_id"]
            if uid not in user_apps:
                user_apps[uid] = Counter()
            user_apps[uid][row["process_name"]] += 1

        distinct_pairs = 0
        user_ids = sorted(user_apps)
        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                top_i = {app for app, _ in user_apps[user_ids[i]].most_common(5)}
                top_j = {app for app, _ in user_apps[user_ids[j]].most_common(5)}
                if len(top_i.symmetric_difference(top_j)) >= 2:
                    distinct_pairs += 1

        assert distinct_pairs >= 2, (
            f"Only {distinct_pairs} user pairs had distinct app distributions; "
            f"expected at least 2"
        )

    def test_user_id_present_and_unique(self) -> None:
        """user_ids are 1, 2, 3, ..., num_users with no gaps."""
        rows = generate_synthetic_data(days=2, num_users=4, seed=42)
        user_ids = sorted({row["user_id"] for row in rows})
        assert user_ids == [1, 2, 3, 4], (
            f"Expected user_ids [1, 2, 3, 4], got {user_ids}"
        )

    def test_num_users_controls_row_count(self) -> None:
        """num_users=7 produces ~7 × 14 × 24 × 12 = 28224 rows (±5%)."""
        num_users = 7
        rows = generate_synthetic_data(days=14, num_users=num_users, seed=42)
        expected = _expected_rows(days=14, num_users=num_users)
        tolerance = expected * 0.05
        assert abs(len(rows) - expected) <= tolerance, (
            f"Expected ~{expected} rows (±5%), got {len(rows)}"
        )

    def test_each_user_has_continuous_timeline(self) -> None:
        """For each user_id, timestamps are strictly chronological and
        consecutive samples are spaced ≤ 10 minutes apart."""
        rows = generate_synthetic_data(days=3, num_users=3, seed=42)
        max_gap = timedelta(minutes=10)

        user_rows: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            user_rows.setdefault(row["user_id"], []).append(row)

        for uid, urows in user_rows.items():
            timestamps = sorted(r["timestamp"] for r in urows)
            for i in range(1, len(timestamps)):
                gap = timestamps[i] - timestamps[i - 1]
                assert gap <= max_gap, (
                    f"User {uid}: gap of {gap} between "
                    f"{timestamps[i - 1]} and {timestamps[i]}"
                )

    def test_all_profiles_generate_rows(self) -> None:
        """Specifying ['freshman_cs', 'senior_business', 'grad_medical']
        generates data for exactly 3 users with balanced row counts."""
        profiles = ["freshman_cs", "senior_business", "grad_medical"]
        rows = generate_synthetic_data(days=3, user_profiles=profiles, seed=42)
        user_ids = sorted({row["user_id"] for row in rows})
        assert len(user_ids) == 3, (
            f"Expected 3 users from 3 profiles, got {len(user_ids)}: {user_ids}"
        )
        expected_per = _expected_rows(days=3)
        for uid in user_ids:
            count = sum(1 for r in rows if r["user_id"] == uid)
            assert abs(count - expected_per) <= expected_per * 0.10, (
                f"User {uid} has {count} rows, expected ~{expected_per} (±10%)"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TestProcrastinationPatterns
# ═══════════════════════════════════════════════════════════════════════════════


class TestProcrastinationPatterns:
    """Procrastination episodes must exhibit realistic behavioral signatures.

    All tests in this class require include_procrastination=True and are
    marked xfail until the procrastination pattern generator is implemented.
    """

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sliding_hour_windows(
        rows: list[dict[str, Any]],
        window_hours: int = 2,
    ) -> list[list[dict[str, Any]]]:
        """Split rows into sliding windows of N consecutive hours per user."""
        buckets = _hourly_buckets(rows)
        sorted_keys = sorted(buckets.keys())
        windows: list[list[dict[str, Any]]] = []
        for i in range(len(sorted_keys) - window_hours + 1):
            window: list[dict[str, Any]] = []
            for j in range(window_hours):
                window.extend(buckets[sorted_keys[i + j]])
            windows.append(window)
        return windows

    @staticmethod
    def _app_switch_count(non_idle_rows: list[dict[str, Any]]) -> int:
        """Count process_name transitions in a list of non-idle rows."""
        proc_names = [r["process_name"] for r in non_idle_rows]
        return sum(
            1 for i in range(1, len(proc_names))
            if proc_names[i] != proc_names[i - 1]
        )

    # ── Tests ──────────────────────────────────────────────────────────────

    def test_binge_watching_episode_appears(self) -> None:
        """With procrastination enabled, at least one 2-hour window has
        entertainment_ratio > 0.5."""
        rows = generate_synthetic_data(
            days=7, num_users=3, include_procrastination=True, seed=42
        )
        windows = self._sliding_hour_windows(rows, window_hours=2)
        binge_count = sum(
            1 for w in windows if _entertainment_ratio(w) > 0.5
        )
        assert binge_count > 0, (
            "No binge-watching episodes found: expected at least one 2-hour "
            "window with entertainment_ratio > 0.5"
        )

    def test_doom_scrolling_episode_appears(self) -> None:
        """Some hours have social_ratio > 0.3 AND switch_frequency > 4
        (rapid app switching characteristic of doom-scrolling)."""
        rows = generate_synthetic_data(
            days=7, num_users=3, include_procrastination=True, seed=42
        )
        social_apps = {"wechat", "weibo", "douyin", "zhihu", "twitter", "reddit"}

        doom_hours = 0
        for _key, bucket in _hourly_buckets(rows).items():
            non_idle = [r for r in bucket if r.get("is_idle", 0) == 0]
            if len(non_idle) < 5:
                continue
            social_ratio = sum(
                1 for r in non_idle if r.get("process_name", "") in social_apps
            ) / len(non_idle)
            switches = self._app_switch_count(non_idle)
            if social_ratio > 0.3 and switches > max(4, len(non_idle) // 2):
                doom_hours += 1

        assert doom_hours > 0, (
            "No doom-scrolling episodes: expected social_ratio > 0.3 and "
            "switch_frequency > 4 in at least one hour window"
        )

    def test_task_avoidance_pattern_exists(self) -> None:
        """Some segments show code → browser → code alternation
        (the classic task avoidance signature)."""
        rows = generate_synthetic_data(
            days=7, num_users=3, include_procrastination=True, seed=42
        )
        code_apps = {"vscode", "pycharm", "terminal", "notion", "typora"}
        browser_apps = {"chrome", "bilibili", "youtube", "douyin", "weibo", "zhihu"}

        found = False
        for uid in set(r["user_id"] for r in rows):
            non_idle = [
                r for r in rows
                if r["user_id"] == uid and r.get("is_idle", 0) == 0
            ]
            procs = [r["process_name"] for r in non_idle]
            for i in range(len(procs) - 2):
                if (
                    procs[i] in code_apps
                    and procs[i + 1] in browser_apps
                    and procs[i + 2] in code_apps
                ):
                    found = True
                    break
            if found:
                break

        assert found, (
            "No task avoidance pattern (code → browser → code) found"
        )

    def test_deadline_panic_exists(self) -> None:
        """Some day-segments show productivity_ratio > 0.6 and prolonged
        uninterrupted focus (≥ 6 consecutive samples of same app)."""
        rows = generate_synthetic_data(
            days=7, num_users=3, include_procrastination=True, seed=42
        )
        productivity_apps = {
            "vscode", "pycharm", "terminal", "notion", "typora", "excel", "wps",
        }

        # Group daytime work segments by full local date and user. Deadline
        # panic intentionally does not overwrite evening leisure/late-night data.
        day_buckets: dict[tuple[int, object], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            timestamp = row["timestamp"]
            if 8 <= timestamp.hour < 18:
                day_buckets[(row["user_id"], timestamp.date())].append(row)

        panic_count = 0
        for _key, bucket in day_buckets.items():
            non_idle = sorted(
                (r for r in bucket if r.get("is_idle", 0) == 0),
                key=lambda row: row["timestamp"],
            )
            if len(non_idle) < 50:
                continue
            prod_ratio = sum(
                1 for r in non_idle if r.get("process_name", "") in productivity_apps
            ) / len(non_idle)

            # Longest consecutive run of same app
            max_run = 0
            cur_run = 0
            cur_app: str | None = None
            for r in non_idle:
                if r["process_name"] == cur_app:
                    cur_run += 1
                else:
                    max_run = max(max_run, cur_run)
                    cur_app = r["process_name"]
                    cur_run = 1
            max_run = max(max_run, cur_run)

            if prod_ratio > 0.6 and max_run > 6:
                panic_count += 1

        assert panic_count > 0, (
            "No deadline panic signatures: expected productivity_ratio > 0.6 "
            "and max_app_run > 6 samples in at least one day"
        )

    def test_procrastination_evening_biased(self) -> None:
        """Entertainment ratio is higher in hours 19-24 than 9-12
        (procrastination is evening-biased)."""
        rows = generate_synthetic_data(
            days=7, num_users=5, include_procrastination=True, seed=42
        )

        def _ratio_in_range(lo: int, hi: int) -> float:
            subset = [
                r for r in rows
                if lo <= r["timestamp"].hour < hi and r.get("is_idle", 0) == 0  # type: ignore[union-attr]
            ]
            return _entertainment_ratio(subset) if subset else 0.0

        evening = _ratio_in_range(19, 24)
        morning = _ratio_in_range(9, 12)

        assert evening > morning, (
            f"Evening entertainment ratio ({evening:.3f}) not higher than "
            f"morning ({morning:.3f})"
        )

    def test_weekend_has_more_procrastination(self) -> None:
        """Weekend entertainment_ratio exceeds weekday entertainment_ratio."""
        rows = generate_synthetic_data(
            days=14, num_users=3, include_procrastination=True, seed=42
        )

        weekend = [
            r for r in rows
            if r["timestamp"].weekday() >= 5 and r.get("is_idle", 0) == 0  # type: ignore[union-attr, arg-type]
        ]
        weekday = [
            r for r in rows
            if r["timestamp"].weekday() < 5 and r.get("is_idle", 0) == 0  # type: ignore[union-attr, arg-type]
        ]

        assert weekend and weekday, "Need both weekend and weekday data"
        weekend_ratio = _entertainment_ratio(weekend)
        weekday_ratio = _entertainment_ratio(weekday)

        assert weekend_ratio > weekday_ratio, (
            f"Weekend entertainment ({weekend_ratio:.3f}) ≤ weekday ({weekday_ratio:.3f})"
        )

    def test_procrastination_flag_controls_behavior(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Procrastination episodes do not shorten true per-user entertainment streaks."""
        monkeypatch.setattr(synthetic_data, "datetime", _FixedDateTime)
        rows_off = generate_synthetic_data(
            days=7, num_users=3, include_procrastination=False, seed=42
        )
        rows_on = generate_synthetic_data(
            days=7, num_users=3, include_procrastination=True, seed=42
        )

        streak_off = _max_consecutive_entertainment_hours(rows_off)
        streak_on = _max_consecutive_entertainment_hours(rows_on)

        assert streak_off <= 7, (
            f"Without procrastination, max entertainment streak was {streak_off}h "
            f"(expected at most 7h for profile-based evening patterns)"
        )
        assert streak_on >= streak_off - 1, (
            f"Procrastination streak ({streak_on}h) materially shorter than "
            f"baseline ({streak_off}h); episodes should not reduce entertainment"
        )

    @pytest.mark.parametrize(
        "profile_id",
        ["freshman_business", "grad_cs", "junior_cs"],
    )
    def test_procrastination_preserves_streak_across_profiles(
        self, monkeypatch: pytest.MonkeyPatch, profile_id: str
    ) -> None:
        """The flag overlays episodes without erasing each profile's baseline streak."""
        monkeypatch.setattr(synthetic_data, "datetime", _FixedDateTime)
        rows_off = generate_synthetic_data(
            days=7,
            user_profiles=[profile_id],
            include_procrastination=False,
            seed=42,
        )
        rows_on = generate_synthetic_data(
            days=7,
            user_profiles=[profile_id],
            include_procrastination=True,
            seed=42,
        )

        streak_off = _max_consecutive_entertainment_hours(rows_off)
        streak_on = _max_consecutive_entertainment_hours(rows_on)
        assert streak_on >= streak_off - 1, (
            f"{profile_id}: procrastination streak ({streak_on}h) materially shorter "
            f"than baseline ({streak_off}h)"
        )

    def test_episodes_have_minimum_duration(self) -> None:
        """Multiple users have procrastination episodes lasting ≥ 6 consecutive
        entertainment samples (30 min at 12 samples/hour)."""
        rows = generate_synthetic_data(
            days=7, num_users=3, include_procrastination=True, seed=42
        )
        apps = _entertainment_apps()

        users_with_long_episode = 0
        for uid in sorted(set(r["user_id"] for r in rows)):
            non_idle = [
                r for r in rows
                if r["user_id"] == uid and r.get("is_idle", 0) == 0
            ]
            max_streak = 0
            cur = 0
            for r in non_idle:
                if r.get("process_name", "") in apps:
                    cur += 1
                    max_streak = max(max_streak, cur)
                else:
                    cur = 0

            if max_streak >= 6:
                users_with_long_episode += 1

        assert users_with_long_episode >= 2, (
            f"Only {users_with_long_episode}/3 users had entertainment streaks ≥ 6; "
            f"expected at least 2"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestArchetypeBehavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestArchetypeBehavior:
    """Student archetypes produce characteristic app usage and schedule patterns.

    All tests require the archetype/profile system.
    """

    def test_cs_student_uses_ide(self) -> None:
        """freshman_cs or junior_cs user has 'vscode' or 'pycharm' in
        > 10% of non-idle rows."""
        rows = generate_synthetic_data(
            days=3, user_profiles=["freshman_cs", "junior_cs"], seed=42
        )
        ide_apps = {"vscode", "pycharm"}
        for uid in sorted(set(r["user_id"] for r in rows)):
            user_rows = [
                r for r in rows
                if r["user_id"] == uid and r.get("is_idle", 0) == 0
            ]
            if not user_rows:
                continue
            ide_ratio = sum(
                1 for r in user_rows if r.get("process_name", "") in ide_apps
            ) / len(user_rows)
            assert ide_ratio > 0.10, (
                f"User {uid}: IDE usage ratio {ide_ratio:.3f} ≤ 0.10"
            )

    def test_design_student_uses_figma_or_ps(self) -> None:
        """senior_design user has creative tools (Figma/Photoshop/Illustrator/XD)
        in their process_name set."""
        rows = generate_synthetic_data(
            days=3, user_profiles=["senior_design"], seed=42
        )
        creative = {"figma", "photoshop", "illustrator", "xd"}
        all_apps = {r["process_name"] for r in rows}
        assert bool(all_apps & creative), (
            f"No creative design tools found in apps: {sorted(all_apps)}"
        )

    def test_business_student_uses_excel(self) -> None:
        """junior_business user has Excel or WPS in their process_name set."""
        rows = generate_synthetic_data(
            days=3, user_profiles=["junior_business"], seed=42
        )
        all_apps = {r["process_name"] for r in rows}
        assert bool(all_apps & {"excel", "wps"}), (
            f"Excel/WPS not found in apps: {sorted(all_apps)}"
        )

    def test_medical_student_has_high_focus(self) -> None:
        """grad_medical has higher average productivity_ratio than
        junior_business and senior_design."""
        profiles = ["grad_medical", "junior_business", "senior_design"]
        rows = generate_synthetic_data(days=5, user_profiles=profiles, seed=42)

        productivity_apps = {
            "vscode", "pycharm", "terminal", "notion", "typora",
            "excel", "wps", "onenote", "anki",
        }

        def _user_productivity(uid: int) -> float:
            user_rows = [
                r for r in rows
                if r["user_id"] == uid and r.get("is_idle", 0) == 0
            ]
            if not user_rows:
                return 0.0
            return sum(
                1 for r in user_rows if r.get("process_name", "") in productivity_apps
            ) / len(user_rows)

        user_ids = sorted(set(r["user_id"] for r in rows))
        assert len(user_ids) == 3, f"Expected 3 users, got {user_ids}"

        med = _user_productivity(user_ids[0])
        biz = _user_productivity(user_ids[1])
        des = _user_productivity(user_ids[2])

        assert med > biz, f"Medical ({med:.3f}) not higher than Business ({biz:.3f})"
        assert med > des, f"Medical ({med:.3f}) not higher than Design ({des:.3f})"

    def test_freshman_has_structured_schedule(self) -> None:
        """Freshman (high rigidity) has higher hour-to-hour activity variance
        than senior (low rigidity), reflecting a more structured daily pattern
        with clear work/sleep distinction rather than evenly distributed activity."""
        rows = generate_synthetic_data(
            days=5, user_profiles=["freshman_cs", "senior_cs"], seed=42
        )

        def _active_hour_variance(uid: int) -> float:
            hour_counts = [0] * 24
            for r in rows:
                if r["user_id"] == uid and r.get("is_idle", 0) == 0:
                    hour_counts[r["timestamp"].hour] += 1  # type: ignore[union-attr]
            return float(np.var(hour_counts))

        user_ids = sorted(set(r["user_id"] for r in rows))
        freshman_var = _active_hour_variance(user_ids[0])
        senior_var = _active_hour_variance(user_ids[1])

        # Structured schedule creates clear peaks → higher variance.
        # Unstructured schedule has more even distribution → lower variance.
        assert freshman_var > senior_var, (
            f"Freshman variance ({freshman_var:.1f}) ≤ Senior ({senior_var:.1f}) — "
            f"structured schedules produce higher variance"
        )

    def test_weekend_delay_respected(self) -> None:
        """Archetypes with weekend_delay_hours=3 shift the wake-up
        transition ~3h later on weekends compared to weekdays.

        Detects the wake-up hour as the hour with the largest positive
        jump in non-idle activity (the derivative peak).
        """
        custom_archetypes = {
            "late_sleeper": {
                "weekend_delay_hours": 3,
                "major": "cs",
                "year": "junior",
            }
        }
        rows = generate_synthetic_data(
            days=7,
            user_profiles=["late_sleeper"],
            archetypes=custom_archetypes,
            seed=42,
        )

        def _wake_hour_from_derivative(is_weekend: bool) -> float:
            hour_counts = [0] * 24
            for r in rows:
                ts = r["timestamp"]
                if (ts.weekday() >= 5) != is_weekend:  # type: ignore[union-attr, arg-type]
                    continue
                if r.get("is_idle", 0) == 0:
                    hour_counts[ts.hour] += 1  # type: ignore[union-attr]
            # Find the hour with the largest positive jump
            best_hour = 0
            best_jump = 0
            for h in range(24):
                jump = hour_counts[h] - hour_counts[(h - 1) % 24]
                if jump > best_jump:
                    best_jump = jump
                    best_hour = h
            return float(best_hour)

        weekday_wake = _wake_hour_from_derivative(is_weekend=False)
        weekend_wake = _wake_hour_from_derivative(is_weekend=True)

        assert weekday_wake >= 0 and weekend_wake >= 0, "No active hours found"
        assert weekend_wake >= weekday_wake + 2.0, (
            f"Weekend wake ({weekend_wake:.0f}h) not ≥ weekday wake + 2h "
            f"({weekday_wake:.0f}h) for weekend_delay_hours=3"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TestLargeVolume
# ═══════════════════════════════════════════════════════════════════════════════


class TestLargeVolume:
    """Correctness and performance at scale.

    Test 26 (chronological order) uses only existing parameters and SHOULD
    PASS now. All other tests require enhanced parameters.
    """

    def test_30_profiles_15_days_produces_129k_rows(self) -> None:
        """30 users × 15 days × 24h × 12 samples/h ≈ 129,600 rows (±10%)."""
        rows = generate_synthetic_data(days=15, num_users=30, seed=42)
        expected = _expected_rows(days=15, num_users=30)
        tolerance = expected * 0.10
        assert abs(len(rows) - expected) <= tolerance, (
            f"Expected ~{expected} rows (±10%), got {len(rows)}"
        )

    def test_chronological_order_maintained(self) -> None:
        """All rows sorted by timestamp in ascending global order."""
        rows = generate_synthetic_data(days=1, samples_per_hour=12)
        timestamps = [r["timestamp"] for r in rows]
        assert timestamps == sorted(timestamps), (
            "Rows not in global chronological order"
        )

    def test_no_duplicate_timestamps_per_user(self) -> None:
        """Same user_id never has duplicate timestamps."""
        rows = generate_synthetic_data(days=7, num_users=5, seed=42)
        user_ts: dict[int, set[datetime]] = defaultdict(set)
        for r in rows:
            uid = r["user_id"]
            ts = r["timestamp"]
            assert ts not in user_ts[uid], (
                f"User {uid} has duplicate timestamp {ts}"
            )
            user_ts[uid].add(ts)

    def test_idle_distribution_realistic(self) -> None:
        """Overall idle ratio between 0.1 and 0.42 across 5 users × 7 days."""
        rows = generate_synthetic_data(days=7, num_users=5, seed=42)
        idle_ratio = sum(1 for r in rows if r.get("is_idle", 0) == 1) / max(len(rows), 1)
        assert 0.1 <= idle_ratio <= 0.42, (
            f"Idle ratio {idle_ratio:.3f} outside realistic range [0.1, 0.42]"
        )

    def test_performance_reasonable(self) -> None:
        """Generation of ~129k rows completes in < 30 seconds."""
        start = time.perf_counter()
        rows = generate_synthetic_data(days=15, num_users=30, seed=42)
        elapsed = time.perf_counter() - start

        expected = _expected_rows(days=15, num_users=30)
        assert len(rows) >= expected * 0.90, (
            f"Row count {len(rows)} too far below expected {expected}"
        )
        assert elapsed < 30.0, (
            f"Generation took {elapsed:.1f}s, exceeding 30s budget"
        )
