"""Tests for baseline model (pure stdlib, no pandas)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mindflow.domain.baseline import BaselineModel


def _make_feature_row(
    hour: int = 12,
    dow: int = 0,
    switch_frequency: float = 15.0,
    unique_app_count: float = 4.0,
    productivity_ratio: float = 0.5,
    entertainment_ratio: float = 0.2,
    social_ratio: float = 0.1,
    max_app_duration: float = 600.0,
    idle_ratio: float = 0.05,
    title_code_ratio: float = 0.0,
    title_doc_ratio: float = 0.0,
    title_url_ratio: float = 0.0,
    title_meeting_ratio: float = 0.0,
    title_entertainment_ratio: float = 0.0,
    process_name: str = "code.exe",
) -> dict:
    return {
        "hour_of_day": hour,
        "day_of_week": dow,
        "switch_frequency": switch_frequency,
        "unique_app_count": unique_app_count,
        "productivity_ratio": productivity_ratio,
        "entertainment_ratio": entertainment_ratio,
        "social_ratio": social_ratio,
        "max_app_duration": max_app_duration,
        "idle_ratio": idle_ratio,
        "title_code_ratio": title_code_ratio,
        "title_doc_ratio": title_doc_ratio,
        "title_url_ratio": title_url_ratio,
        "title_meeting_ratio": title_meeting_ratio,
        "title_entertainment_ratio": title_entertainment_ratio,
        "process_name": process_name,
    }


def _make_rows(n: int = 5, **overrides) -> list[dict]:
    """Generate n feature rows for testing."""
    return [_make_feature_row(**overrides) for _ in range(n)]


class TestBaselineModelInit:
    def test_initialization(self):
        model = BaselineModel(user_id=1)
        assert model.user_id == 1
        assert model.total_days == 0
        assert not model.has_sufficient_data(30)
        assert model.created_at is not None
        assert model.updated_at is not None

    def test_initialization_different_users(self):
        model_a = BaselineModel(user_id=42)
        model_b = BaselineModel(user_id=99)
        assert model_a.user_id == 42
        assert model_b.user_id == 99

    def test_buckets_initialized(self):
        model = BaselineModel(user_id=1)
        assert len(model._stats) == 24
        assert len(model._top_apps) == 24
        for hour in range(24):
            assert len(model._stats[hour]) == 7
            assert len(model._top_apps[hour]) == 7

    def test_empty_rows_returns_zero(self):
        model = BaselineModel(user_id=1)
        n = model.update([])
        assert n == 0
        assert not model.has_sufficient_data(1)


class TestBaselineModelUpdate:
    def test_single_bucket_update(self):
        model = BaselineModel(user_id=1)
        rows = _make_rows(3, hour=10, dow=2, switch_frequency=10.0)
        n = model.update(rows)
        assert n == 3
        assert model.has_sufficient_data(1)
        stats = model.get_stats(10, 2)
        assert stats["switch_frequency"]["n"] == 3
        assert stats["switch_frequency"]["mean"] == 10.0
        assert stats["switch_frequency"]["std"] == 0.0

    def test_multi_bucket_update(self):
        model = BaselineModel(user_id=1)
        rows = [
            _make_feature_row(hour=9, dow=0, switch_frequency=5.0),
            _make_feature_row(hour=10, dow=0, switch_frequency=15.0),
            _make_feature_row(hour=9, dow=1, switch_frequency=8.0),
        ]
        model.update(rows)
        assert model.get_stats(9, 0)["switch_frequency"]["n"] == 1
        assert model.get_stats(10, 0)["switch_frequency"]["n"] == 1
        assert model.get_stats(9, 1)["switch_frequency"]["n"] == 1

    def test_welford_convergence(self):
        """With identical values, mean should be exact and std 0."""
        model = BaselineModel(user_id=1)
        rows = _make_rows(10, hour=14, dow=3, switch_frequency=12.5)
        model.update(rows)
        stats = model.get_stats(14, 3)
        assert stats["switch_frequency"]["mean"] == 12.5
        assert stats["switch_frequency"]["std"] == 0.0

    def test_welford_variance(self):
        """With known values, verify Welford std approximates population std."""
        model = BaselineModel(user_id=1)
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        rows = [_make_feature_row(hour=9, dow=0, switch_frequency=v) for v in values]
        model.update(rows)
        stats = model.get_stats(9, 0)
        # sample std for [10,12,14,16,18] = 3.1623
        assert round(stats["switch_frequency"]["std"], 1) == 3.2

    def test_missing_feature_skipped(self):
        model = BaselineModel(user_id=1)
        rows = [
            {"hour_of_day": 10, "day_of_week": 0, "switch_frequency": 5.0},
            {"hour_of_day": 10, "day_of_week": 0},
        ]
        n = model.update(rows)
        assert n == 2
        stats = model.get_stats(10, 0)
        assert stats["switch_frequency"]["n"] == 1

    def test_non_numeric_feature_skipped(self):
        model = BaselineModel(user_id=1)
        rows = [
            {"hour_of_day": 10, "day_of_week": 0, "switch_frequency": "invalid"},
        ]
        n = model.update(rows)
        assert n == 1
        stats = model.get_stats(10, 0)
        assert "switch_frequency" not in stats

    def test_app_tracking(self):
        model = BaselineModel(user_id=1)
        rows = [
            _make_feature_row(hour=10, dow=0, process_name="chrome.exe"),
            _make_feature_row(hour=10, dow=0, process_name="chrome.exe"),
            _make_feature_row(hour=10, dow=0, process_name="code.exe"),
        ]
        model.update(rows)
        apps = model.get_top_apps(10, 0)
        assert apps[0]["app"] == "chrome.exe"
        assert apps[0]["count"] == 2
        assert apps[1]["app"] == "code.exe"
        assert apps[1]["count"] == 1

    def test_app_tracking_limit(self):
        model = BaselineModel(user_id=1)
        rows = [_make_feature_row(hour=10, dow=0, process_name=f"app{i}.exe") for i in range(10)]
        model.update(rows)
        apps = model.get_top_apps(10, 0, limit=3)
        assert len(apps) == 3

    def test_unknown_process_default(self):
        model = BaselineModel(user_id=1)
        rows = [{"hour_of_day": 10, "day_of_week": 0, "switch_frequency": 5.0}]
        model.update(rows)
        apps = model.get_top_apps(10, 0)
        assert apps[0]["app"] == "unknown"

    def test_total_days_tracking(self):
        model = BaselineModel(user_id=1)
        rows = [
            {**_make_feature_row(hour=10, dow=0), "window_start": "2026-01-01T10:00:00"},
            {**_make_feature_row(hour=11, dow=0), "window_start": "2026-01-01T11:00:00"},
            {**_make_feature_row(hour=10, dow=1), "window_start": "2026-01-02T10:00:00"},
        ]
        model.update(rows)
        assert model.total_days >= 2


class TestBaselineModelHasSufficientData:
    def test_below_threshold(self):
        model = BaselineModel(user_id=1)
        model.update(_make_rows(5))
        assert not model.has_sufficient_data(100)

    def test_above_threshold(self):
        model = BaselineModel(user_id=1)
        model.update(_make_rows(50))
        assert model.has_sufficient_data(30)


class TestBaselineModelPersistence:
    def test_round_trip(self):
        model = BaselineModel(user_id=1)
        model.update(_make_rows(10, hour=10, dow=0, switch_frequency=15.0))

        data = model.to_dict()
        assert data["user_id"] == 1
        assert "stats" in data
        assert "top_apps" in data
        assert "updated_at" in data

        restored = BaselineModel.from_dict(data)
        assert restored.user_id == model.user_id
        assert restored.total_days == model.total_days
        restored_stats = restored.get_stats(10, 0)
        orig_stats = model.get_stats(10, 0)
        assert restored_stats["switch_frequency"]["mean"] == orig_stats["switch_frequency"]["mean"]

    def test_save_load(self):
        model = BaselineModel(user_id=1)
        model.update(_make_rows(10, hour=14, dow=3))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            model.save(path)
            assert path.exists()

            loaded = BaselineModel.load(path)
            assert loaded.user_id == 1
            assert loaded.total_days == model.total_days
            assert loaded.has_sufficient_data(1)

    def test_load_empty_baseline(self):
        model = BaselineModel(user_id=1)
        data = model.to_dict()
        restored = BaselineModel.from_dict(data)
        assert restored.user_id == 1
        assert restored.total_days == 0
        assert not restored.has_sufficient_data(1)

    def test_json_serializable(self):
        model = BaselineModel(user_id=1)
        model.update(_make_rows(3))
        data = model.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["user_id"] == 1
        assert "stats" in parsed

    def test_save_load_missing_apps(self):
        """from_dict should handle missing top_apps gracefully."""
        model = BaselineModel(user_id=1)
        # Build minimal dict without top_apps
        data = model.to_dict()
        data.pop("top_apps", None)
        restored = BaselineModel.from_dict(data)
        assert restored.user_id == 1


class TestBucketSufficiency:
    """Per-bucket data sufficiency (review M3 contract)."""

    def test_empty_bucket_is_insufficient(self) -> None:
        model = BaselineModel(user_id=1)
        assert model.has_bucket_sufficient_data(hour=9, dow=0) is False

    def test_populated_bucket_reaches_sufficiency(self) -> None:
        model = BaselineModel(user_id=1)
        rows = [
            {"hour_of_day": 9, "day_of_week": 0, "switch_frequency": 8.0 + i} for i in range(3)
        ]
        model.update(rows)
        assert model.has_bucket_sufficient_data(hour=9, dow=0, min_samples=2) is True
        # Other buckets remain insufficient even though overall data exists
        assert model.has_bucket_sufficient_data(hour=14, dow=3, min_samples=2) is False
