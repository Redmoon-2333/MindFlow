"""Tests for deviation detection (pure stdlib, no pandas).

``DeviationDetector`` scores a 5-minute V2 feature window against the user's
own baseline for the same local (hour, weekday) bucket. Two estimators do the
work, both from ``domain/baseline.py``:

* the expected value is an empirical-Bayes (James–Stein) shrunk median, so a
  sparse bucket leans on the user's global behaviour;
* the dispersion is the median absolute deviation scaled by 1.4826, so a
  handful of extreme days cannot inflate σ and mask the deviation.

Feature weights are expressed over the V2 vocabulary and renormalised over the
features actually present in a row (persisted rows carry their vector inside
``features_json``; in-memory rows pass flattened keys — both are accepted).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

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


#: Median of the training values below (10, 12, 14, 16, 18 repeating) — a row
#: carrying it is, by construction, exactly typical for the bucket.
_TYPICAL_SWITCHES = 14.0


def _train_baseline(rows_per_bucket: int = 10) -> tuple[BaselineModel, list[dict]]:
    """Create a baseline from varied data at hour=10, dow=0.

    Uses slight variation so both the Welford and MAD scales are non-zero.
    Every other feature stays constant at 0.0, so those features carry no
    dispersion and the detector skips them instead of emitting a spurious
    z-score from the scale floor.
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
        assert "scored_features" in score

    def test_normal_window_classification(self):
        """Window matching baseline should be 'normal'."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(
            hour=10, dow=0, app_switch_count=_TYPICAL_SWITCHES, unique_app_count=3.0
        )
        score = detector.score_window(row)
        assert score["severity"] == "normal"

    def test_insufficient_baseline_returns_normal(self):
        """A single observation yields no robust scale, so nothing is scored."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update([_make_feature_row(hour=10, dow=0)])
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=100.0)
        score = detector.score_window(row)
        assert score["overall_deviation"] == 0.0
        assert score["severity"] == "normal"
        assert score["z_scores"] == {}

    def test_missing_features_skipped(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = {"hour_of_day": 10, "day_of_week": 0}
        score = detector.score_window(row)
        assert score["overall_deviation"] == 0.0
        assert score["severity"] == "normal"

    def test_extreme_v2_feature_is_detected(self):
        """A 25x jump in app switches must reach a non-normal severity.

        This is the regression the V2 re-weighting fixes: before it, only
        ``idle_ratio`` from the legacy weight table overlapped the V2
        vocabulary, so this window scored 0 and reported "normal".
        """
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=500.0)
        score = detector.score_window(row)
        assert score["z_scores"]["app_switch_count"] > 5.0
        assert score["severity"] == "severe"
        assert score["overall_deviation"] >= detector.SEVERE_THRESHOLD

    def test_severity_classes(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        # Sanity: verify thresholds map to correct strings
        assert detector.MILD_THRESHOLD == 1.5
        assert detector.MODERATE_THRESHOLD == 2.5
        assert detector.SEVERE_THRESHOLD == 4.0
        assert detector.severity_for(0.0) == "normal"
        assert detector.severity_for(1.5) == "mild"
        assert detector.severity_for(2.5) == "moderate"
        assert detector.severity_for(4.0) == "severe"

    def test_constant_feature_is_not_scored(self):
        """A feature that never varies has no honest scale — skip, don't cap."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=14.0, idle_ratio=0.9)
        score = detector.score_window(row)
        assert "idle_ratio" not in score["z_scores"]


class TestDeviationDetectorAnalyze:
    def test_analyze_empty(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        result = detector.analyze_dataframe([])
        assert result == []

    def test_analyze_all_normal(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, app_switch_count=_TYPICAL_SWITCHES)
            for _ in range(3)
        ]
        result = detector.analyze_dataframe(rows)
        assert result == []

    def test_analyze_flags_extreme_window(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, app_switch_count=_TYPICAL_SWITCHES),
            _make_feature_row(hour=10, dow=0, app_switch_count=500.0),
        ]
        result = detector.analyze_dataframe(rows)
        assert len(result) == 1
        assert result[0]["severity"] == "severe"

    def test_analyze_sorted_by_deviation(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, app_switch_count=40.0),
            _make_feature_row(hour=10, dow=0, app_switch_count=500.0),
            _make_feature_row(hour=10, dow=0, app_switch_count=120.0),
        ]
        result = detector.analyze_dataframe(rows)
        assert len(result) == 3
        deviations = [row["overall_deviation"] for row in result]
        assert deviations == sorted(deviations, reverse=True)

    def test_window_titles_attached_to_anomalies(self):
        """Title enrichment applies to flagged windows only."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [_make_feature_row(hour=10, dow=0, app_switch_count=500.0)]
        result = detector.analyze_dataframe(rows, window_titles=["VS Code - work"])
        assert len(result) == 1
        assert result[0]["sample_titles"] == ["VS Code - work"]

        list_result = detector.analyze_dataframe(rows, window_titles=[["t1", "t2"]])
        assert list_result[0]["sample_titles"] == ["t1", "t2"]

        empty_result = detector.analyze_dataframe(rows, window_titles=[""])
        assert empty_result[0]["sample_titles"] == []

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
        assert score["top_deviations"][0]["feature"] == "app_switch_count"
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
        assert score["z_scores"]["app_switch_count"] == 10.0
        for feature, z in score["z_scores"].items():
            assert -10.0 <= z <= 10.0, f"Z-score {z} for {feature} not clamped"


class TestDeviationSummary:
    """``to_summary`` produces the API contract's ``DeviationSummary`` (§2)."""

    def test_summary_shape_and_rank(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        summary = detector.to_summary(_make_feature_row(hour=10, dow=0, app_switch_count=500.0))
        assert summary is not None
        assert summary["method"] == "eb_shrinkage_mad"
        assert summary["severity"] == "severe"
        assert set(summary["bucket"]) == {
            "hour_of_day",
            "day_of_week",
            "n_samples",
            "shrinkage",
        }
        assert summary["bucket"]["hour_of_day"] == 10
        assert summary["bucket"]["day_of_week"] == 0
        assert summary["bucket"]["n_samples"] == 10
        # k / (n + k) with k = 10 and n = 10.
        assert summary["bucket"]["shrinkage"] == pytest.approx(0.5)
        assert len(summary["features"]) <= 6
        top = summary["features"][0]
        assert top["feature"] == "app_switch_count"
        # Chinese label comes from the shared domain/feature_labels map.
        assert top["label"] == "应用切换次数"
        assert top["value"] == 500.0
        assert top["expected"] == pytest.approx(_TYPICAL_SWITCHES)
        assert top["robust_z"] > 5.0
        assert 0.0 < top["weight"] <= 1.0

    def test_summary_feature_cap(self):
        """At most 6 features, ordered by |z| * weight."""
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        varied = {
            "app_switch_count",
            "domain_switch_count",
            "idle_ratio",
            "active_seconds_ratio",
            "keypress_rate_per_min",
            "browser_ratio",
            "top_app_ratio",
            "longest_segment_ratio",
        }
        rows = []
        for i in range(10):
            extra = {name: float(i) for name in varied if name != "app_switch_count"}
            rows.append(_make_feature_row(hour=10, dow=0, app_switch_count=float(i), **extra))
        model.update(rows)
        detector = DeviationDetector(model)
        row = _make_feature_row(
            hour=10,
            dow=0,
            app_switch_count=100.0,
            **{name: 100.0 for name in varied if name != "app_switch_count"},
        )
        summary = detector.to_summary(row)
        assert summary is not None
        assert len(summary["features"]) == 6
        scores = [
            abs(item["robust_z"]) * item["weight"] for item in summary["features"]
        ]
        assert scores == sorted(scores, reverse=True)

    def test_summary_none_without_baseline_data(self):
        """No feature in the window has baseline data → no summary."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        assert detector.to_summary({"hour_of_day": 10, "day_of_week": 0}) is None

    def test_features_json_and_flattened_row_agree(self):
        """A persisted row (vector in ``features_json``) scores like a flat one.

        Persisted windows carry their vector as JSON; in-memory rows pass the
        values as top-level keys. Both must reach the same score so
        ``EvidenceBundle`` and the API share one scoring path.
        """
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, app_switch_count=500.0)
        flat = {
            **{name: 0.0 for name in V2_FEATURE_NAMES},
            "app_switch_count": 500.0,
            "hour_of_day": 10,
            "day_of_week": 0,
        }
        assert detector.score_window(row)["overall_deviation"] == pytest.approx(
            detector.score_window(flat)["overall_deviation"]
        )
        assert detector.score_window(row)["z_scores"] == detector.score_window(flat)["z_scores"]


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
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, app_switch_count=_TYPICAL_SWITCHES),
            _make_feature_row(hour=10, dow=0, app_switch_count=500.0),
        ]
        summary = detector.daily_summary(rows)
        assert summary["total_windows"] == 2
        assert summary["anomaly_count"] == 1
        assert summary["anomaly_ratio"] == pytest.approx(0.5)
        assert summary["severity_counts"] == {"normal": 1, "severe": 1}

    def test_daily_summary_most_anomalous_hour(self):
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        # Train every hour with the same varied values so each bucket has a
        # meaningful scale.
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
            _make_feature_row(hour=14, dow=0, app_switch_count=_TYPICAL_SWITCHES),
        ]
        summary = detector.daily_summary(rows)
        assert summary["anomaly_count"] == 2
        assert summary["most_anomalous_hour"] == 9

    def test_daily_summary_zero_total_weight(self):
        """When all features missing, total_weight=0 should not crash."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row: dict = {"hour_of_day": 10, "day_of_week": 0}
        summary = detector.daily_summary([row])
        assert summary["average_deviation"] == 0.0
