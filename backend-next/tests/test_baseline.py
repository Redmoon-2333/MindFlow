"""Tests for the V2 baseline model (pure stdlib, no pandas).

Covers: the single 24-feature V2 vocabulary, schema-version persistence,
``window_start_utc`` + ``features_json`` row consumption, timezone-correct
(hour, weekday) and unique local-date derivation, Welford serialization,
and the inclusive 30-sample readiness threshold.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from mindflow.domain.baseline import (
    MAD_TO_SIGMA,
    ROBUST_RESERVOIR_SIZE,
    BaselineModel,
)
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES

_SH = ZoneInfo("Asia/Shanghai")

# Baseline local (hour, weekday) base week: 2026-07-27 is a Monday (dow 0).
_BASE_LOCAL_DATE = date(2026, 7, 27)

V2_FEATURES_DEFAULT: dict[str, float] = {
    "app_switch_count": 3.0,
    "domain_switch_count": 2.0,
    "longest_segment_ratio": 0.6,
    "idle_ratio": 0.1,
    "keypress_rate_per_min": 25.0,
    "mouse_click_rate_per_min": 12.0,
    "scroll_rate_per_min": 5.0,
    "mouse_distance_per_min": 200.0,
    "input_active_ratio": 0.7,
    "interaction_bursts_per_min": 2.0,
    "click_key_ratio": 0.5,
    "browser_ratio": 0.3,
    "audible_browser_ratio": 0.1,
    "active_seconds_ratio": 0.8,
    "top_app_ratio": 0.7,
    "top_domain_ratio": 0.5,
    "interaction_interval_mean_s": 10.0,
    "interaction_interval_std_s": 5.0,
    "interaction_interval_cv": 0.5,
    "hour_sin": 0.5,
    "hour_cos": 0.5,
    "weekday_sin": 0.5,
    "weekday_cos": 0.5,
    "task_type_code": 0.0,
    # v4 task-context columns (feature schema 4)
    "task_type_entropy": 0.0,
    "task_type_dominant_ratio": 0.0,
    "task_context_transition": 0.0,
    "task_unknown_ratio": 1.0,
}


def _shanghai_utc(hour: int, dow: int, *, day: int = 27) -> datetime:
    """UTC instant whose Asia/Shanghai local time is *hour* on a *dow* day.

    Anchors to 2026-07-*day* in the base week (2026-07-27 = Monday, dow 0)
    and shifts to the requested weekday.
    """
    local = datetime.combine(
        date(_BASE_LOCAL_DATE.year, _BASE_LOCAL_DATE.month, day),
        time(hour=hour),
        tzinfo=_SH,
    )
    shift = (dow - local.weekday()) % 7
    return (local + timedelta(days=shift)).astimezone(UTC)


def _make_feature_row(
    start_utc: datetime | None = None,
    process_name: str = "code.exe",
    features: Mapping[str, Any] | None = None,
    features_json: str | None = None,
    **feature_overrides: Any,
) -> dict[str, Any]:
    """V2 feature-window row: ``window_start_utc`` + flattened ``features_json``."""
    merged = {**V2_FEATURES_DEFAULT, **(features or {}), **feature_overrides}
    return {
        "window_start_utc": start_utc if start_utc is not None else _shanghai_utc(12, 0),
        "features_json": features_json if features_json is not None else json.dumps(merged),
        "process_name": process_name,
    }


def _make_rows(
    n: int = 5,
    hour: int = 12,
    dow: int = 0,
    start_utc: datetime | None = None,
    **overrides: Any,
) -> list[dict[str, Any]]:
    """Generate *n* V2 feature rows; default bucket is local (hour, dow)."""
    start = start_utc if start_utc is not None else _shanghai_utc(hour, dow)
    return [_make_feature_row(start, **overrides) for _ in range(n)]


def _typed_rows(n: int = 5, **overrides: Any) -> list[Mapping[str, Any]]:
    return [dict(row) for row in _make_rows(n, **overrides)]


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
        assert len(model._stats) == 24  # noqa: SLF001
        assert len(model._top_apps) == 24  # noqa: SLF001
        for hour in range(24):
            assert len(model._stats[hour]) == 7  # noqa: SLF001
            assert len(model._top_apps[hour]) == 7  # noqa: SLF001

    def test_empty_rows_returns_zero(self):
        model = BaselineModel(user_id=1)
        n = model.update([])
        assert n == 0
        assert not model.has_sufficient_data(1)


class TestV2FeatureVocabulary:
    """The baseline consumes exactly the single authoritative V2 vocabulary."""

    def test_feature_vocabulary_is_the_current_v2_names(self) -> None:
        assert list(V2_FEATURE_NAMES) == BaselineModel.FEATURE_COLS
        assert len(BaselineModel.FEATURE_COLS) == len(V2_FEATURE_NAMES)

    def test_feature_schema_version_is_current(self) -> None:
        assert BaselineModel.FEATURE_SCHEMA_VERSION == FEATURE_SCHEMA_VERSION


class TestV2Update:
    def test_bucket_derived_from_window_start_in_timezone(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        # 04:00 UTC == 12:00 local Monday (dow 0) in Asia/Shanghai.
        rows = _make_rows(3, start_utc=datetime(2026, 7, 27, 4, 0, tzinfo=UTC))
        n = model.update(rows)
        assert n == 3
        assert model.has_sufficient_data(1)
        stats = model.get_stats(12, 0)
        assert stats["app_switch_count"]["n"] == 3
        assert stats["app_switch_count"]["mean"] == 3.0
        assert stats["app_switch_count"]["std"] == 0.0

    def test_welford_convergence(self):
        """With identical values, mean should be exact and std 0."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        rows = _make_rows(10, hour=14, dow=3, app_switch_count=12.5)
        model.update(rows)
        stats = model.get_stats(14, 3)
        assert stats["app_switch_count"]["mean"] == 12.5
        assert stats["app_switch_count"]["std"] == 0.0

    def test_welford_variance(self):
        """With known values, verify Welford std approximates sample std."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        values = [10.0, 12.0, 14.0, 16.0, 18.0]
        rows = [
            _make_feature_row(_shanghai_utc(9, 0), app_switch_count=v) for v in values
        ]
        model.update(rows)
        stats = model.get_stats(9, 0)
        # sample std for [10,12,14,16,18] = 3.1623
        assert round(stats["app_switch_count"]["std"], 1) == 3.2

    def test_features_json_accepts_string_and_dict(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        start = _shanghai_utc(10, 0)
        rows = [
            {"window_start_utc": start, "features_json": json.dumps({"app_switch_count": 5.0})},
            {"window_start_utc": start, "features_json": {"app_switch_count": 5.0}},
        ]
        assert model.update(rows) == 2
        assert model.get_stats(10, 0)["app_switch_count"]["n"] == 2

    def test_missing_feature_skipped(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        start = _shanghai_utc(10, 0)
        rows = [
            {"window_start_utc": start, "features_json": json.dumps({"app_switch_count": 5.0})},
            {"window_start_utc": start, "features_json": json.dumps({"idle_ratio": 0.2})},
        ]
        n = model.update(rows)
        assert n == 2
        stats = model.get_stats(10, 0)
        assert stats["app_switch_count"]["n"] == 1
        assert stats["idle_ratio"]["n"] == 1

    def test_non_numeric_feature_skipped(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        start = _shanghai_utc(10, 0)
        rows = [{
            "window_start_utc": start,
            "features_json": json.dumps({"app_switch_count": "invalid", "idle_ratio": 0.1}),
        }]
        n = model.update(rows)
        assert n == 1
        stats = model.get_stats(10, 0)
        assert "app_switch_count" not in stats
        assert stats["idle_ratio"]["n"] == 1

    def test_app_tracking(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        start = _shanghai_utc(10, 0)
        rows = [
            _make_feature_row(start, process_name="chrome.exe"),
            _make_feature_row(start, process_name="chrome.exe"),
            _make_feature_row(start, process_name="code.exe"),
        ]
        model.update(rows)
        apps = model.get_top_apps(10, 0)
        assert apps[0]["app"] == "chrome.exe"
        assert apps[0]["count"] == 2
        assert apps[1]["app"] == "code.exe"
        assert apps[1]["count"] == 1

    def test_unknown_process_default(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update([{"window_start_utc": _shanghai_utc(10, 0), "features_json": "{}"}])
        apps = model.get_top_apps(10, 0)
        assert apps[0]["app"] == "unknown"


class TestTimezoneBucketing:
    def test_utc_1600_lands_on_next_local_date_and_hour(self):
        """16:00 UTC == 00:00 next day in Asia/Shanghai: hour 0, next weekday."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        # 2026-07-27 Monday, 16:00 UTC -> 2026-07-28 00:00 Tuesday local.
        model.update([_make_feature_row(datetime(2026, 7, 27, 16, 0, tzinfo=UTC))])
        stats = model.get_stats(0, 1)
        assert stats["app_switch_count"]["n"] == 1
        assert model.total_days == 1
        # The recorded local date is the *next* day, not the UTC day.
        restored = BaselineModel.from_dict(model.to_dict())
        assert restored.total_days == 1

    def test_unique_local_dates_across_batches(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        # 15:59 UTC on 07-27 -> 23:59 local 07-27.
        model.update([_make_feature_row(datetime(2026, 7, 27, 15, 59, tzinfo=UTC))])
        # 16:00 UTC on 07-27 -> 00:00 local 07-28 (a different local day).
        model.update([_make_feature_row(datetime(2026, 7, 27, 16, 0, tzinfo=UTC))])
        assert model.total_days == 2

    def test_explicit_timezone_changes_bucket(self):
        """The same UTC instant buckets differently in a different timezone."""
        start = datetime(2026, 7, 27, 4, 0, tzinfo=UTC)  # 12:00 Shanghai, 09:30 Mumbai
        shanghai = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        shanghai.update([_make_feature_row(start)])
        assert shanghai.get_stats(12, 0)["app_switch_count"]["n"] == 1
        assert shanghai.get_stats(9, 0) == {}


class TestV2Persistence:
    def test_round_trip_retains_schema_timezone_timestamps_and_counts(self):
        model = BaselineModel(user_id=7, timezone="Asia/Shanghai")
        model.update(_typed_rows(10, hour=9, dow=0, app_switch_count=15.0))
        model.update(_typed_rows(5, hour=14, dow=3, app_switch_count=8.0))

        restored = BaselineModel.from_dict(model.to_dict())

        assert restored.FEATURE_SCHEMA_VERSION == FEATURE_SCHEMA_VERSION
        assert restored.timezone == "Asia/Shanghai"
        assert restored.total_samples() == model.total_samples()
        assert restored.get_stats(9, 0) == model.get_stats(9, 0)
        assert restored.get_stats(14, 3) == model.get_stats(14, 3)
        assert restored.total_days == model.total_days
        assert restored.created_at == model.created_at
        assert restored.updated_at == model.updated_at

    def test_save_load(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(_make_rows(10, hour=14, dow=3))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            model.save(path)
            assert path.exists()

            loaded = BaselineModel.load(path)
            assert loaded.user_id == 1
            assert loaded.FEATURE_SCHEMA_VERSION == FEATURE_SCHEMA_VERSION
            assert loaded.timezone == "Asia/Shanghai"
            assert loaded.total_days == model.total_days
            assert loaded.has_sufficient_data(1)

    def test_load_empty_baseline(self):
        model = BaselineModel(user_id=1)
        restored = BaselineModel.from_dict(model.to_dict())
        assert restored.user_id == 1
        assert restored.total_days == 0
        assert not restored.has_sufficient_data(1)

    def test_json_serializable(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(_make_rows(3))
        data = model.to_dict()
        parsed = json.loads(json.dumps(data))
        assert parsed["user_id"] == 1
        assert parsed["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert parsed["timezone"] == "Asia/Shanghai"
        assert "stats" in parsed

    def test_save_load_missing_apps(self):
        """from_dict should handle missing top_apps gracefully."""
        data = BaselineModel(user_id=1).to_dict()
        data.pop("top_apps", None)
        restored = BaselineModel.from_dict(data)
        assert restored.user_id == 1

    def test_legacy_payload_without_schema_version_is_detectable(self):
        """Old V1 payloads load but remain detectable as not-v2 (no mapper)."""
        v1 = {"user_id": 3, "created_at": "2026-01-01T00:00:00+00:00",
              "updated_at": "2026-01-01T00:00:00+00:00", "total_days": 1,
              "stats": {}, "top_apps": {}}
        restored = BaselineModel.from_dict(v1)
        assert restored.FEATURE_SCHEMA_VERSION != 2
        assert restored.user_id == 3
        # Re-serializing must not silently upgrade the stored version.
        assert BaselineModel.from_dict(restored.to_dict()).FEATURE_SCHEMA_VERSION != 2

    def test_reloaded_model_keeps_accumulating_local_dates(self) -> None:
        """total_days accumulation survives a round-trip (rollup idempotency)."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update([_make_feature_row(datetime(2026, 7, 27, 4, 0, tzinfo=UTC))])
        restored = BaselineModel.from_dict(model.to_dict())
        assert restored.total_days == 1
        # A later window on another local date increments exactly once.
        restored.update([_make_feature_row(datetime(2026, 7, 28, 4, 0, tzinfo=UTC))])
        assert restored.total_days == 2


class TestBucketSufficiency:
    """Per-bucket data sufficiency (review M3 contract)."""

    def test_empty_bucket_is_insufficient(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        assert model.has_bucket_sufficient_data(hour=9, dow=0) is False

    def test_populated_bucket_reaches_sufficiency(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        rows = [
            _make_feature_row(_shanghai_utc(9, 0), app_switch_count=8.0 + i)
            for i in range(3)
        ]
        model.update(rows)
        assert model.has_bucket_sufficient_data(hour=9, dow=0, min_samples=2) is True
        # Other buckets remain insufficient even though overall data exists
        assert model.has_bucket_sufficient_data(hour=14, dow=3, min_samples=2) is False


class TestBaselineSerializationContract:
    """Formal V2 serialization + readiness contract.

    Locks the V2 vocabulary, Welford state preservation across round-trips,
    sample-count semantics, and the inclusive readiness threshold so later
    baseline work cannot drift silently.
    """

    def test_feature_vocabulary_is_v2_24_features(self) -> None:
        assert list(V2_FEATURE_NAMES) == BaselineModel.FEATURE_COLS
        assert "app_switch_count" in BaselineModel.FEATURE_COLS
        assert "switch_frequency" not in BaselineModel.FEATURE_COLS
        assert "productivity_ratio" not in BaselineModel.FEATURE_COLS

    def test_round_trip_preserves_welford_state_and_counts(self) -> None:
        model = BaselineModel(user_id=7, timezone="Asia/Shanghai")
        model.update(_typed_rows(10, hour=9, dow=0, app_switch_count=15.0))
        model.update(_typed_rows(5, hour=14, dow=3, app_switch_count=8.0))

        restored = BaselineModel.from_dict(model.to_dict())

        assert restored.total_samples() == model.total_samples()
        assert restored.get_stats(9, 0) == model.get_stats(9, 0)
        assert restored.get_stats(14, 3) == model.get_stats(14, 3)
        assert restored.total_days == model.total_days
        assert restored.created_at == model.created_at
        assert restored.updated_at == model.updated_at

    def test_total_samples_counts_every_feature_value_per_row(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(_typed_rows(5, hour=10, dow=0))
        assert model.total_samples() == 5 * len(BaselineModel.FEATURE_COLS)

    def test_has_sufficient_data_threshold_is_inclusive(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(_typed_rows(2, hour=10, dow=0))
        samples = 2 * len(BaselineModel.FEATURE_COLS)
        assert model.total_samples() == samples
        assert model.has_sufficient_data(samples) is True
        assert model.has_sufficient_data(samples + 1) is False

    def test_has_sufficient_data_flips_at_thirty_samples(self) -> None:
        """Readiness flips at the existing 30-sample threshold (inclusive)."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        # One full row (every current feature) + one partial row sized so the
        # two together are exactly the 30-sample readiness threshold.
        start = _shanghai_utc(9, 0)
        partial_size = max(1, 30 - len(V2_FEATURE_NAMES))
        partial_features = {
            name: V2_FEATURES_DEFAULT[name] for name in V2_FEATURE_NAMES[:partial_size]
        }
        model.update([
            _make_feature_row(start),
            _make_feature_row(start, features_json=json.dumps(partial_features)),
        ])
        assert model.total_samples() == 30
        assert model.has_sufficient_data(30) is True
        assert model.has_sufficient_data(31) is False

    def test_overall_mean_is_sample_weighted_across_buckets(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(_typed_rows(3, hour=9, dow=0, app_switch_count=10.0))
        model.update(_typed_rows(1, hour=14, dow=3, app_switch_count=20.0))
        # (3 * 10 + 1 * 20) / 4 == 12.5
        assert model.overall_mean("app_switch_count") == 12.5

    def test_overall_mean_unknown_feature_returns_none(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(_typed_rows(2))
        assert model.overall_mean("not_a_feature") is None

    def test_serialization_keeps_iso_timestamps(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(_typed_rows(1))
        data = model.to_dict()
        assert "T" in data["created_at"]
        assert "T" in data["updated_at"]


class TestMalformedPolicy:
    """Malformed timestamps / nonnumeric values must not corrupt counts."""

    def test_malformed_timestamp_skipped_without_count_corruption(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        rows = [
            {"window_start_utc": "not-a-timestamp", "features_json": "{}"},
            {"window_start_utc": None, "features_json": "{}"},
            {"features_json": "{}"},  # no timestamp at all
            _make_feature_row(),  # valid control row
        ]
        assert model.update(rows) == 1
        assert model.total_samples() == len(V2_FEATURE_NAMES)

    def test_invalid_features_json_skipped(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        rows = [
            {"window_start_utc": _shanghai_utc(10, 0), "features_json": "{not json"},
            _make_feature_row(),
        ]
        assert model.update(rows) == 1
        assert model.total_samples() == len(V2_FEATURE_NAMES)

    def test_nonnumeric_feature_does_not_corrupt_counts(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        start = _shanghai_utc(10, 0)
        rows = [
            {"window_start_utc": start, "features_json": json.dumps(
                {"app_switch_count": "oops", "idle_ratio": 0.2})},
            {"window_start_utc": start, "features_json": json.dumps(
                {"app_switch_count": 4.0})},
            {"window_start_utc": start, "features_json": json.dumps(
                {"app_switch_count": 6.0})},
        ]
        assert model.update(rows) == 3
        stats = model.get_stats(10, 0)
        # Nonnumeric value contributes nothing; the two valid values are
        # counted exactly once each (n == 2, not 3).
        assert stats["app_switch_count"]["n"] == 2
        assert stats["app_switch_count"]["mean"] == 5.0
        assert stats["idle_ratio"]["n"] == 1

    def test_non_finite_feature_skipped(self) -> None:
        """NaN/Inf feature values are skipped before Welford updates."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        start = _shanghai_utc(10, 0)
        rows = [
            {"window_start_utc": start, "features_json": json.dumps(
                {"app_switch_count": float("nan"), "idle_ratio": 0.2})},
            {"window_start_utc": start, "features_json": json.dumps(
                {"app_switch_count": float("inf")})},
            {"window_start_utc": start, "features_json": json.dumps(
                {"app_switch_count": 4.0})},
            {"window_start_utc": start, "features_json": json.dumps(
                {"app_switch_count": 6.0})},
        ]
        assert model.update(rows) == 4
        stats = model.get_stats(10, 0)
        # Non-finite values contribute nothing; the two valid values are
        # counted exactly once each (n == 2, not 4) with a finite mean.
        assert stats["app_switch_count"]["n"] == 2
        assert stats["app_switch_count"]["mean"] == 5.0
        assert stats["idle_ratio"]["n"] == 1
        assert model.overall_mean("app_switch_count") == 5.0

    def test_persisted_non_finite_bucket_mean_returns_none(self) -> None:
        """A persisted poisoned bucket must not emit a non-finite overall mean."""
        model = BaselineModel(user_id=1)
        data = model.to_dict()
        # Simulate a previously-persisted payload whose Welford state already
        # holds a non-finite mean (NaN or Infinity).
        data["stats"]["12"]["0"]["app_switch_count"] = {"n": 5.0, "mean": float("nan"), "M2": 0.0}
        data["stats"]["13"]["0"]["app_switch_count"] = {"n": 5.0, "mean": float("inf"), "M2": 0.0}
        restored = BaselineModel.from_dict(data)
        assert restored.overall_mean("app_switch_count") is None


class TestRobustBaseline:
    """Empirical-Bayes / MAD statistics behind ``DeviationSummary`` (§2).

    The expected value is a James–Stein posterior mean of the bucket's Welford
    mean and the user's global median; the dispersion is the same posterior
    combination of the bucket variance and the global MAD-based σ. The global
    estimate comes from a bounded ring buffer of recent values per feature, so
    it has a 50% breakdown point without inflating the persisted JSON.

    These tests pin the arithmetic, the cold-start fallbacks (empty bucket →
    global; short ring → pooled Welford), the zero-dispersion guard, the
    outlier resistance that motivates the estimator, and the persistence
    contract (the ``robust`` block is optional in both directions).
    """

    #: Global ring values of ``_two_bucket_model``: 10, 12, 14, 16, 18 and
    #: 100, 102, 104, 106, 108. Median (18+100)/2; MAD of the deviations
    #: (41,41,43,43,45,45,47,47,49,49) is 45.
    _GLOBAL_MEDIAN = 59.0
    _GLOBAL_MAD = 45.0
    _GLOBAL_SIGMA = 45.0 * MAD_TO_SIGMA

    @staticmethod
    def _two_bucket_model() -> BaselineModel:
        """(10,0) holds 10–18, (11,0) holds 100–108; medians 14 and 104."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        rows = [
            _make_feature_row(_shanghai_utc(10, 0), app_switch_count=float(v))
            for v in (10, 12, 14, 16, 18)
        ] + [
            _make_feature_row(_shanghai_utc(11, 0), app_switch_count=float(v))
            for v in (100, 102, 104, 106, 108)
        ]
        model.update(rows)
        return model

    def test_global_robust_stats_are_median_and_scaled_mad(self) -> None:
        model = self._two_bucket_model()
        stats = model.global_robust_stats("app_switch_count")
        assert stats is not None
        assert stats["n"] == 10.0
        assert stats["median"] == self._GLOBAL_MEDIAN
        assert stats["mad"] == self._GLOBAL_MAD
        assert stats["sigma"] == pytest.approx(self._GLOBAL_SIGMA, rel=1e-9)
        # A feature observed with no dispersion still gets a ring entry, but
        # its MAD is 0 — the zero-dispersion guard in shrunk_expected is what
        # stops that from becoming a division by ~0.
        flat = model.global_robust_stats("keypress_rate_per_min")
        assert flat is not None and flat["mad"] == 0.0

    def test_global_welford_stats_pool_every_bucket(self) -> None:
        """The pooled mean is sample-weighted, the variance the union's."""
        model = self._two_bucket_model()
        stats = model.global_welford_stats("app_switch_count")
        assert stats is not None
        assert stats["n"] == 10.0
        assert stats["mean"] == pytest.approx(59.0)
        # Values 10..18 / 100..108 → Σ M2 = 40 + 40 = 80 over N-1 = 9.
        assert stats["std"] == pytest.approx(math.sqrt(80.0 / 9.0), rel=1e-9)
        assert model.global_welford_stats("keypress_rate_per_min") is not None
        assert model.global_welford_stats("keypress_rate_per_min")["std"] == 0.0

    def test_shrinkage_math(self) -> None:
        """μ̂ and ŝ are both posterior combinations, not just the mean."""
        model = self._two_bucket_model()
        k = 10
        n_b = 5.0
        mean_b = 14.0
        var_b = 10.0  # M2 40 / (5 - 1)
        estimate = model.shrunk_expected("app_switch_count", 10, 0, k=k)
        assert estimate is not None

        expected = (n_b * mean_b + k * self._GLOBAL_MEDIAN) / (n_b + k)
        variance = (n_b * var_b + k * self._GLOBAL_SIGMA**2) / (n_b + k)
        assert estimate["expected"] == pytest.approx(expected)
        assert estimate["scale"] == pytest.approx(math.sqrt(variance))
        assert estimate["shrinkage"] == pytest.approx(k / (n_b + k))
        assert estimate["n_bucket"] == n_b
        assert estimate["n_global"] == 10.0

    def test_scale_never_drops_below_the_relative_floor(self) -> None:
        """A large mean with a tiny spread must not report huge z-scores."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        # Values hovering at 0.95 ± 0.001: absolute spread 1e-3, relative 1e-3.
        model.update(
            [
                _make_feature_row(
                    _shanghai_utc(10, 0), longest_segment_ratio=0.95 + delta
                )
                for delta in (0.0, 0.001, -0.001, 0.002, -0.002)
            ]
        )
        estimate = model.shrunk_expected("longest_segment_ratio", 10, 0)
        assert estimate is not None
        floor = 1e-6 + 0.05 * abs(estimate["expected"])
        assert estimate["scale"] == pytest.approx(floor)
        assert estimate["scale"] > math.sqrt(1e-6)

    def test_cold_start_falls_back_to_global(self) -> None:
        """An empty bucket uses the global estimate verbatim (shrinkage 1.0)."""
        model = self._two_bucket_model()
        estimate = model.shrunk_expected("app_switch_count", 3, 4)
        assert estimate is not None
        assert estimate["n_bucket"] == 0.0
        assert estimate["shrinkage"] == 1.0
        assert estimate["expected"] == self._GLOBAL_MEDIAN
        assert estimate["scale"] == pytest.approx(self._GLOBAL_SIGMA, rel=1e-9)

    def test_unobserved_feature_returns_none(self) -> None:
        """No data anywhere → None, so callers skip the feature."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        assert model.shrunk_expected("app_switch_count", 10, 0) is None
        assert model.global_robust_stats("app_switch_count") is None
        assert model.global_welford_stats("app_switch_count") is None
        assert model.robust_z("app_switch_count", 3.0, 10, 0) is None

    def test_short_ring_falls_back_to_pooled_welford(self) -> None:
        """Fewer than MIN_ROBUST_SAMPLES values → the MAD is not trusted yet."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(
            [
                _make_feature_row(_shanghai_utc(10, 0), app_switch_count=float(v))
                for v in (10, 12, 14, 16, 18)
            ]
        )
        assert model.global_robust_stats("app_switch_count") is None
        welford = model.global_welford_stats("app_switch_count")
        assert welford is not None

        # Inside the observed bucket the Welford prior is combined as usual.
        estimate = model.shrunk_expected("app_switch_count", 10, 0)
        assert estimate is not None
        assert estimate["expected"] == pytest.approx(14.0)
        assert estimate["n_global"] == 5.0
        # Cold bucket: the pooled Welford mean/std *is* the estimate.
        cold = model.shrunk_expected("app_switch_count", 3, 4)
        assert cold is not None
        assert cold["expected"] == pytest.approx(welford["mean"])
        assert cold["scale"] == pytest.approx(welford["std"])
        assert cold["shrinkage"] == 1.0

    def test_zero_dispersion_feature_is_withheld(self) -> None:
        """A feature that never varied has no honest scale — skip, don't cap."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update(
            [
                _make_feature_row(_shanghai_utc(10, 0), app_switch_count=14.0)
                for _ in range(10)
            ]
        )
        assert model.global_robust_stats("app_switch_count") is None or (
            model.global_robust_stats("app_switch_count")["sigma"] == 0.0
        )
        assert model.shrunk_expected("app_switch_count", 10, 0) is None
        assert model.robust_z("app_switch_count", 500.0, 10, 0) is None

    def test_mad_resists_a_single_extreme_window(self) -> None:
        """One 1000-switch window must not inflate the scale (breakdown 50%)."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        values = [9.0, 10.0, 11.0] * 3 + [1000.0]
        model.update(
            [
                _make_feature_row(_shanghai_utc(10, 0), app_switch_count=v)
                for v in values
            ]
        )
        stats = model.global_robust_stats("app_switch_count")
        assert stats is not None
        assert stats["median"] == 10.0
        assert stats["mad"] == 1.0
        assert stats["sigma"] == pytest.approx(MAD_TO_SIGMA)
        # The Welford σ, by contrast, is dominated by the outlier.
        welford = model.get_stats(10, 0)["app_switch_count"]
        assert welford["std"] > 200.0
        # ... and it drags the shrunk scale up with it, which is exactly why the
        # robust prior is worth having: the global median is still 10.
        estimate = model.shrunk_expected("app_switch_count", 10, 0)
        assert estimate is not None
        assert estimate["expected"] == pytest.approx(59.5)

    def test_robust_z_uses_shrunk_expectation(self) -> None:
        model = self._two_bucket_model()
        estimate = model.shrunk_expected("app_switch_count", 10, 0)
        assert estimate is not None
        z = model.robust_z("app_switch_count", 500.0, 10, 0)
        assert z is not None
        assert z == (500.0 - estimate["expected"]) / estimate["scale"]
        assert model.robust_z("app_switch_count", float("nan"), 10, 0) is None

    def test_ring_is_bounded_and_keeps_recent_values(self) -> None:
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        values = [float(v) for v in range(ROBUST_RESERVOIR_SIZE + 20)]
        model.update(
            [
                _make_feature_row(_shanghai_utc(10, 0), app_switch_count=v)
                for v in values
            ]
        )
        assert ROBUST_RESERVOIR_SIZE == 256
        stats = model.global_robust_stats("app_switch_count")
        assert stats is not None
        assert stats["n"] == float(ROBUST_RESERVOIR_SIZE)
        # The ring keeps the most recent values, so the median tracks the tail
        # rather than the whole history.
        tail = values[-ROBUST_RESERVOIR_SIZE:]
        assert stats["median"] == (tail[len(tail) // 2 - 1] + tail[len(tail) // 2]) / 2

    def test_robust_block_round_trips(self) -> None:
        model = self._two_bucket_model()
        restored = BaselineModel.from_dict(model.to_dict())
        assert restored.global_robust_stats(
            "app_switch_count"
        ) == model.global_robust_stats("app_switch_count")
        assert restored.shrunk_expected(
            "app_switch_count", 10, 0
        ) == model.shrunk_expected("app_switch_count", 10, 0)

    def test_legacy_payload_without_robust_block_loads(self) -> None:
        """Payloads written before this upgrade must load and stay usable."""
        model = self._two_bucket_model()
        payload = model.to_dict()
        del payload["robust"]
        restored = BaselineModel.from_dict(payload)
        # Welford stats survive untouched.
        assert restored.get_stats(10, 0)["app_switch_count"]["mean"] == 14.0
        # No ring yet: the robust prior is absent, so the estimate degrades to
        # the pooled Welford statistics rather than disappearing or crashing.
        assert restored.global_robust_stats("app_switch_count") is None
        welford = restored.global_welford_stats("app_switch_count")
        assert welford is not None
        estimate = restored.shrunk_expected("app_switch_count", 10, 0)
        assert estimate is not None
        assert estimate["expected"] == pytest.approx(
            (5.0 * 14.0 + 10.0 * welford["mean"]) / 15.0
        )
        assert estimate["n_global"] == 10.0
        # The next updates repopulate the ring from new windows only.
        for value in (1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0):
            restored.update(
                [_make_feature_row(_shanghai_utc(10, 0), app_switch_count=value)]
            )
        stats = restored.global_robust_stats("app_switch_count")
        assert stats is not None and stats["n"] == 8.0
        assert stats["median"] == 8.0

    def test_new_payload_shape_is_additive(self) -> None:
        """The ``robust`` key is additive; every legacy key keeps its shape."""
        payload = self._two_bucket_model().to_dict()
        for key in (
            "user_id",
            "feature_schema_version",
            "timezone",
            "created_at",
            "updated_at",
            "total_days",
            "local_dates",
            "stats",
            "top_apps",
        ):
            assert key in payload
        assert payload["robust"]["reservoir_size"] == ROBUST_RESERVOIR_SIZE
        # One global ring per feature — no per-bucket reservoirs in the JSON.
        assert set(payload["robust"]) == {"reservoir_size", "global"}
        assert payload["robust"]["global"]["app_switch_count"] == [
            10.0, 12.0, 14.0, 16.0, 18.0, 100.0, 102.0, 104.0, 106.0, 108.0,
        ]
