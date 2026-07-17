"""Tests for deviation detection (pure stdlib, no pandas)."""

from __future__ import annotations

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.deviation import DeviationDetector


def _make_feature_row(
    hour: int = 12,
    dow: int = 0,
    switch_frequency: float = 15.0,
    unique_app_count: float = 4.0,
    max_app_duration: float = 600.0,
    idle_ratio: float = 0.05,
    productivity_ratio: float = 0.5,
    entertainment_ratio: float = 0.2,
    social_ratio: float = 0.1,
    title_code_ratio: float = 0.0,
    title_doc_ratio: float = 0.0,
    title_url_ratio: float = 0.0,
    title_meeting_ratio: float = 0.0,
    title_entertainment_ratio: float = 0.0,
    window_start: str = "",
) -> dict:
    return {
        "hour_of_day": hour,
        "day_of_week": dow,
        "switch_frequency": switch_frequency,
        "unique_app_count": unique_app_count,
        "max_app_duration": max_app_duration,
        "idle_ratio": idle_ratio,
        "productivity_ratio": productivity_ratio,
        "entertainment_ratio": entertainment_ratio,
        "social_ratio": social_ratio,
        "title_code_ratio": title_code_ratio,
        "title_doc_ratio": title_doc_ratio,
        "title_url_ratio": title_url_ratio,
        "title_meeting_ratio": title_meeting_ratio,
        "title_entertainment_ratio": title_entertainment_ratio,
        "window_start": window_start,
    }


def _train_baseline(rows_per_bucket: int = 10) -> tuple[BaselineModel, list[dict]]:
    """Create a baseline from varied data at hour=10, dow=0.

    Uses slight variation so std > 0 for meaningful z-score computation.
    """
    model = BaselineModel(user_id=1)
    train_rows = []
    for i in range(rows_per_bucket):
        base = 10.0 + (i % 5) * 2.0  # values: 10, 12, 14, 16, 18, 10, ...
        train_rows.append(
            _make_feature_row(hour=10, dow=0, switch_frequency=base, unique_app_count=3.0)
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
        row = _make_feature_row(hour=10, dow=0, switch_frequency=10.0, unique_app_count=3.0)
        score = detector.score_window(row)
        assert score["severity"] == "normal"

    def test_insufficient_baseline_returns_normal(self):
        """When baseline has <2 samples per feature, std=0 -> no deviation."""
        model = BaselineModel(user_id=1)
        model.update([_make_feature_row(hour=10, dow=0)])
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, switch_frequency=100.0)
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

    def test_severity_mild(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        # Extreme switch_frequency to trigger deviation
        row = _make_feature_row(hour=10, dow=0, switch_frequency=100.0)
        score = detector.score_window(row)
        assert score["severity"] in ("mild", "moderate", "severe")

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

    def test_analyze_with_anomalies(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, switch_frequency=10.0),  # normal
            _make_feature_row(hour=10, dow=0, switch_frequency=500.0),  # anomaly
        ]
        result = detector.analyze_dataframe(rows)
        assert len(result) >= 1
        for a in result:
            assert a["severity"] != "normal"

    def test_analyze_sorted_by_deviation(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, switch_frequency=20.0),
            _make_feature_row(hour=10, dow=0, switch_frequency=500.0),
            _make_feature_row(hour=10, dow=0, switch_frequency=200.0),
        ]
        result = detector.analyze_dataframe(rows)
        deviations = [a["overall_deviation"] for a in result]
        assert deviations == sorted(deviations, reverse=True)

    def test_analyze_with_window_titles(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, switch_frequency=500.0),
        ]
        titles = ["Visual Studio Code - focus work"]
        result = detector.analyze_dataframe(rows, window_titles=titles)
        assert len(result) == 1
        assert "sample_titles" in result[0]
        assert result[0]["sample_titles"] == ["Visual Studio Code - focus work"]

    def test_analyze_with_list_titles(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, switch_frequency=500.0),
        ]
        titles = [["title1", "title2", "title3"]]
        result = detector.analyze_dataframe(rows, window_titles=titles)
        assert result[0]["sample_titles"] == ["title1", "title2", "title3"]

    def test_empty_titles_skipped(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, switch_frequency=500.0),
        ]
        result = detector.analyze_dataframe(rows, window_titles=[""])
        assert "sample_titles" not in result[0] or result[0]["sample_titles"] == []

    def test_top_deviations(self):
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row = _make_feature_row(hour=10, dow=0, switch_frequency=500.0)
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
        row = _make_feature_row(hour=10, dow=0, switch_frequency=1e10)
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
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=10, dow=0, switch_frequency=10.0),
            _make_feature_row(hour=10, dow=0, switch_frequency=500.0),
        ]
        summary = detector.daily_summary(rows)
        assert summary["total_windows"] == 2
        assert summary["anomaly_count"] >= 1

    def test_daily_summary_most_anomalous_hour(self):
        """Test that most_anomalous_hour returns the hour with highest deviation."""
        model = BaselineModel(user_id=1)
        # Train baseline at multiple hours with varied values so std > 0
        train_rows = []
        for h in range(24):
            for i in range(10):
                base = 10.0 + (i % 5) * 2.0
                train_rows.append(_make_feature_row(hour=h, dow=0, switch_frequency=base))
        model.update(train_rows)
        detector = DeviationDetector(model)
        rows = [
            _make_feature_row(hour=9, dow=0, switch_frequency=500.0),
            _make_feature_row(hour=9, dow=0, switch_frequency=500.0),
            _make_feature_row(hour=14, dow=0, switch_frequency=15.0),
        ]
        summary = detector.daily_summary(rows)
        assert summary["most_anomalous_hour"] == 9

    def test_daily_summary_zero_total_weight(self):
        """When all features missing, total_weight=0 should not crash."""
        model, _ = _train_baseline()
        detector = DeviationDetector(model)
        row: dict = {"hour_of_day": 10, "day_of_week": 0}
        summary = detector.daily_summary([row])
        assert summary["average_deviation"] == 0.0
