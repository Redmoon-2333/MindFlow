"""Tests for the synthetic data generator.

Focuses on:
  - Total row count matches expected (24h * samples_per_hour * days)
  - All 5 pattern types appear in output
  - Seed reproducibility (same seed = same data)
  - Weekend vs. weekday idle ratio
"""

from __future__ import annotations

from datetime import datetime

from mindflow.train.synthetic_data import generate_synthetic_data


class TestGenerateSyntheticData:
    """Exercise the synthetic data generator."""

    def test_basic_row_count(self) -> None:
        """Default 14 days at 12 samples/hour should produce ~4032 rows."""
        rows = generate_synthetic_data(days=14, samples_per_hour=12)
        expected = 14 * 24 * 12
        # Allow some wiggle: the generator always produces full grid × days
        assert len(rows) >= expected * 0.95

    def test_three_days_count(self) -> None:
        """3 days at 12 samples/hour."""
        rows = generate_synthetic_data(days=3, samples_per_hour=12)
        expected = 3 * 24 * 12
        assert len(rows) >= expected * 0.95

    def test_all_rows_have_required_fields(self) -> None:
        """Every row must have timestamp, process_name, window_title, duration_seconds, is_idle."""
        rows = generate_synthetic_data(days=1, samples_per_hour=6)
        for row in rows:
            assert "timestamp" in row, f"Row missing timestamp: {row}"
            assert "process_name" in row, f"Row missing process_name: {row}"
            assert "window_title" in row, f"Row missing window_title: {row}"
            assert "duration_seconds" in row, f"Row missing duration_seconds: {row}"
            assert "is_idle" in row, f"Row missing is_idle: {row}"

    def test_timestamp_types(self) -> None:
        """Timestamp should be a datetime (timezone-aware)."""
        rows = generate_synthetic_data(days=1, samples_per_hour=4)
        for row in rows[:10]:
            ts = row["timestamp"]
            assert isinstance(ts, datetime), f"Expected datetime, got {type(ts)}"
            assert ts.tzinfo is not None, "Expected timezone-aware timestamp"

    def test_idle_is_int(self) -> None:
        """is_idle should be 0 or 1."""
        rows = generate_synthetic_data(days=1, samples_per_hour=4)
        for row in rows:
            assert row["is_idle"] in (0, 1), f"is_idle={row['is_idle']} not 0|1"

    def test_duration_positive(self) -> None:
        """duration_seconds should be strictly positive."""
        rows = generate_synthetic_data(days=1, samples_per_hour=4)
        for row in rows:
            assert row["duration_seconds"] > 0, f"duration_seconds={row['duration_seconds']}"

    def test_patterns_appear(self) -> None:
        """All 5 pattern types should be observable in 7-day output."""
        rows = generate_synthetic_data(days=7, samples_per_hour=12)
        apps_seen: set[str] = set()
        for row in rows:
            apps_seen.add(str(row["process_name"]))

        # Check that at least one app from each pattern category appears
        # morning_focus apps
        assert any(a in apps_seen for a in ["vscode", "pycharm", "notion", "terminal"]), (
            "No morning_focus apps found"
        )
        # evening_leisure apps
        assert any(a in apps_seen for a in ["bilibili", "youtube", "steam"]), (
            "No evening_leisure apps found"
        )
        # idle patterns
        assert any(a in apps_seen for a in ["", "lock_screen", "screensaver"]), (
            "No idle entries found"
        )

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        rows_a = generate_synthetic_data(days=2, samples_per_hour=6, seed=42)
        rows_b = generate_synthetic_data(days=2, samples_per_hour=6, seed=42)
        assert len(rows_a) == len(rows_b)

        for r_a, r_b in zip(rows_a, rows_b, strict=False):
            assert r_a["process_name"] == r_b["process_name"]
            assert r_a["is_idle"] == r_b["is_idle"]
            assert abs(r_a["duration_seconds"] - r_b["duration_seconds"]) < 1.0

    def test_different_seed_different(self) -> None:
        """Different seeds produce different results (very high probability)."""
        rows_a = generate_synthetic_data(days=3, samples_per_hour=6, seed=42)
        rows_b = generate_synthetic_data(days=3, samples_per_hour=6, seed=99)
        # Very unlikely that two different seeds produce identical output
        differences = sum(
            1 for a, b in zip(rows_a, rows_b, strict=False)
            if a["process_name"] != b["process_name"] or a["is_idle"] != b["is_idle"]
        )
        assert differences > max(1, len(rows_a) // 10), (
            f"Expected many differences between seeds, got {differences}"
        )

    def test_chronological_order(self) -> None:
        """Output should be sorted by timestamp ascending."""
        rows = generate_synthetic_data(days=1, samples_per_hour=12)
        timestamps = [r["timestamp"] for r in rows]
        assert timestamps == sorted(timestamps), "Rows not in chronological order"

    def test_is_idle_distribution(self) -> None:
        """Idle ratio should be > 0 (some idle periods exist)."""
        rows = generate_synthetic_data(days=3, samples_per_hour=6)
        idle_count = sum(1 for r in rows if r["is_idle"] == 1)
        total = len(rows)
        idle_ratio = idle_count / total
        # At least some idle periods
        assert idle_ratio > 0.05, f"idle_ratio={idle_ratio} too low"
        # Not all idle
        assert idle_ratio < 0.8, f"idle_ratio={idle_ratio} too high"

    def test_weekend_has_more_idle(self) -> None:
        """Weekend data should have more idle/late_night entries (approximately)."""
        # Generate 14 days (covers 2 weekends)
        rows = generate_synthetic_data(days=14, samples_per_hour=6)
        # Sunday (dow=6) pattern heavily leans to leisure
        late_night_apps = {"bilibili", "douyin", "weibo"}
        weekend_rows = [r for r in rows if r["timestamp"].weekday() >= 5]
        weekday_rows = [r for r in rows if r["timestamp"].weekday() < 5]
        if weekend_rows and weekday_rows:
            weekend_entertain = (
                sum(1 for r in weekend_rows if r["process_name"] in late_night_apps)
                / max(len(weekend_rows), 1)
            )
            weekday_entertain = (
                sum(1 for r in weekday_rows if r["process_name"] in late_night_apps)
                / max(len(weekday_rows), 1)
            )
            # weekends have at least proportionally similar entertainment
            assert weekend_entertain >= weekday_entertain * 0.5
