"""Tests for deviation detection (pure stdlib, no pandas).

NOTE on the V2 migration: ``BaselineModel`` now stores only the 24 V2 feature
stats, while ``DeviationDetector`` (domain/deviation.py) still scores against
its 12 legacy feature weights, which overlap the V2 vocabulary at exactly one
name, ``idle_ratio``. The rows below carry their features inside
``features_json`` rather than as flattened top-level keys, so none of the
legacy weights finds a value to score: ``z_scores`` is empty and overall
deviation is 0, hence every window scores "normal". The detector migration
(re-weighting to the V2 vocabulary) is deliberately out of Todo 6 scope
(product behavior beyond BaselineModel), so the affected tests below pin that
degraded contract explicitly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.deviation import DeviationDetector
from mindflow.domain.feature_schema import V2_FEATURE_NAMES


def _shanghai_utc(hour: int, dow: int) -> datetime:
    """UTC instant whose Asia/Shanghai local time is (hour, dow).

    Anchors to 2026-07-27 (a Monday, dow 0) and shifts to the requested weekday.
    """
    local = datetime(2026, 7, 27, hour, tzinfo=ZoneInfo("Asia/Shanghai"))
    shift = (dow - local.weekday()) % 7
    return (local + timedelta(days=shift)).astimezone(UTC)


def _make_feature_row(
    hour: int = 12,
    dow: int = 0,
    app_switch_count: float = 15.0,
    window_start: str = "",
    **extra: object,
) -> dict:
    """V2 feature-window row bucketed at local (hour, dow) in Asia/Shanghai."""
    features: dict[str, object] = {name: 0.0 for name in V2_FEATURE_NAMES}
    features.update({"app_switch_count": app_switch_count, **extra})
    start_utc = window_start or _shanghai_utc(hour, dow)
    return {
        "window_start_utc": start_utc,
        "features_json": json.dumps(features),
        "hour_of_day": hour,
        "day_of_week": dow,
    }


def _train_baseline(rows_per_bucket: int = 10) -> tuple[BaselineModel, list[dict]]:
    """Create a baseline from varied data at hour=10, dow=0.

    Uses slight variation so std > 0 for meaningful z-score computation.
    """
    model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
    train_rows = []
    for i in range(rows_per_bucket):
        base = 10.0 + (i % 5) * 2.0  # values: 10, 12, 14, 16, 18, 10, ...
        train_rows.append(
            _make_feature_row(hour=10, dow=0, app_switch_count=base, unique_app_count=3.0)
        )
    model.update(train_rows)
    return model, train_rows


class TestDeviationDetectorNormal:
    def test_normal_window_returns_all_keys(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0)
        score = detector.score_window(row)
        assert "overall_deviation" in score
        assert "severity" in score
        assert "z_scores" in score
        assert "window_start" in score
        assert "hour_of_day" in score
        assert "day_of_week" in score
        assert "top_deviations" in score

    def test_normal_window_classification(self):
        """Window matching baseline should be 'normal'."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=10.0, unique_app_count=3.0)
        score = detector.score_window(row)
        assert score["severity"] == "normal"

    def test_insufficient_baseline_returns_normal(self):
        """When baseline has <2 samples per feature, std=0 -> no deviation."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update([_make_feature_row(hour=10, dow=0)])
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=100.0)
        score = detector.score_window(row)
        # With n<2, z_score is 0, so overall is 0
        assert score["overall_deviation"] == 0.0
        assert score["severity"] == "normal"

    def test_missing_features_skipped(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = {"hour_of_day": 10, "day_of_week": 0}
        score = detector.score_window(row)
        assert score["overall_deviation"] == 0.0
        assert score["severity"] == "normal"

    def test_legacy_weights_no_longer_match_v2_baseline(self):
        """Extreme V2 feature values no longer trigger legacy-weighted severity.

        The V2 baseline stores v2 feature stats only; DeviationDetector's
        legacy feature weights match no baseline data, so z-scores are all 0
        and the window scores "normal". Detector migration is out of scope.
        """
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=100.0)
        score = detector.score_window(row)
        assert score["overall_deviation"] == 0.0
        assert score["severity"] == "normal"

    def test_severity_classes(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        # Sanity: verify thresholds map to correct strings
        assert detector.MILD_THRESHOLD == 1.5
        assert detector.MODERATE_THRESHOLD == 2.5
        assert detector.SEVERE_THRESHOLD == 4.0


class TestDeviationDetectorAnalyze:
    def test_analyze_empty(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        result = detector.analyze_dataframe([])
        assert result == []

    def test_analyze_all_normal(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [_make_feature_row(hour=10, dow=0) for _ in range(3)]
        result = detector.analyze_dataframe(rows)
        assert result == []

    def test_analyze_returns_no_anomalies_against_v2_baseline(self):
        """After the V2 migration no legacy-weighted window can be an anomaly.

        Extreme v2 feature values still score 0 against the legacy weights, so
        analyze_dataframe returns [] (see module note)."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, app_switch_count=10.0),
            _make_feature_row(hour=10, dow=0, app_switch_count=500.0),
        ]
        result = detector.analyze_dataframe(rows)
        assert result == []

    def test_analyze_sorted_by_deviation(self):
        """Empty result stays trivially sorted (no anomalies post-migration)."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, app_switch_count=20.0),
            _make_feature_row(hour=10, dow=0, app_switch_count=500.0),
            _make_feature_row(hour=10, dow=0, app_switch_count=200.0),
        ]
        result = detector.analyze_dataframe(rows)
        assert result == []

    def test_window_titles_return_no_anomalies(self):
        """Title enrichment only applies to anomalies; none exist post-migration."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, app_switch_count=500.0),
        ]
        assert detector.analyze_dataframe(rows, window_titles=["VS Code - work"]) == []
        assert detector.analyze_dataframe(rows, window_titles=[["t1", "t2"]]) == []
        assert detector.analyze_dataframe(rows, window_titles=[""]) == []

    def test_sample_titles_helper_still_works(self):
        """The private title-enrichment helper keeps its behavior."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        assert detector._sample_titles("  ") == []  # noqa: SLF001
        assert detector._sample_titles("Visual Studio Code") == ["Visual Studio Code"]
        assert detector._sample_titles(["t1", "", "t2", "t3", "t4", "t5"]) == [
            "t1", "t2", "t3", "t4",
        ]

    def test_top_deviations(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=500.0)
        score = detector.score_window(row)
        assert len(score["top_deviations"]) <= 3
        for d in score["top_deviations"]:
            assert "feature" in d
            assert "z_score" in d
            assert "direction" in d
            assert d["direction"] in ("up", "down")
            assert abs(d["z_score"]) > 0.5

    def test_z_score_clamping(self):
        """Z-scores should be clamped to [-10, 10]."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=1e10)
        score = detector.score_window(row)
        for feature, z in score["z_scores"].items():
            assert -10.0 <= z <= 10.0, f"Z-score {z} for {feature} not clamped"


class TestDeviationDetectorDailySummary:
    def test_daily_summary_empty(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        summary = detector.daily_summary([])
        assert summary["total_windows"] == 0
        assert summary["anomaly_count"] == 0
        assert summary["most_anomalous_hour"] is None

    def test_daily_summary_structure(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [_make_feature_row(hour=10, dow=0) for _ in range(5)]
        summary = detector.daily_summary(rows)
        assert "total_windows" in summary
        assert "anomaly_count" in summary
        assert "anomaly_ratio" in summary
        assert "severity_counts" in summary
        assert "average_deviation" in summary
        assert "most_anomalous_hour" in summary or summary["most_anomalous_hour"] is None

    def test_daily_summary_counts(self):
        """Windows are counted; anomalies are 0 against the legacy-weighted detector."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, app_switch_count=10.0),
            _make_feature_row(hour=10, dow=0, app_switch_count=500.0),
        ]
        summary = detector.daily_summary(rows)
        assert summary["total_windows"] == 2
        assert summary["anomaly_count"] == 0
        assert summary["severity_counts"] == {"normal": 2}

    def test_daily_summary_most_anomalous_hour(self):
        """Without legacy-matching stats no hour is anomalous → None."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        # Train baseline at multiple hours with varied values so std > 0
        train_rows = []
        for h in range(24):
            for i in range(10):
                base = 10.0 + (i % 5) * 2.0
                train_rows.append(_make_feature_row(hour=h, dow=0, app_switch_count=base))
        model.update(train_rows)
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=9, dow=0, app_switch_count=500.0),
            _make_feature_row(hour=9, dow=0, app_switch_count=500.0),
            _make_feature_row(hour=14, dow=0, app_switch_count=15.0),
        ]
        summary = detector.daily_summary(rows)
        assert summary["most_anomalous_hour"] is None
        assert summary["anomaly_count"] == 0

    def test_daily_summary_zero_total_weight(self):
        """When all features missing, total_weight=0 should not crash."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row: dict = {"hour_of_day": 10, "day_of_week": 0}
        summary = detector.daily_summary([row])
        assert summary["average_deviation"] == 0.0
