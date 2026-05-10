import json
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from mindflow.analyzer.baseline import BaselineModel
from mindflow.analyzer.deviation import DeviationDetector
from mindflow.analyzer.data_pipeline import generate_synthetic_data, BehaviorFeatureExtractor


def test_baseline_initialization():
    model = BaselineModel(user_id=1)
    assert model.user_id == 1
    assert model.total_days == 0
    assert not model.has_sufficient_data(30)


def test_baseline_update():
    model = BaselineModel(user_id=1)
    raw = generate_synthetic_data(days=3, samples_per_hour=12)
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features = extractor.extract_session_features(raw)
    n = model.update(features)
    assert n > 0
    assert model.has_sufficient_data(30)


def test_baseline_get_stats():
    model = BaselineModel(user_id=1)
    raw = generate_synthetic_data(days=7, samples_per_hour=12)
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features = extractor.extract_session_features(raw)
    model.update(features)

    stats = model.get_stats(10, 1)
    assert "switch_frequency" in stats
    assert stats["switch_frequency"]["n"] > 0
    assert stats["switch_frequency"]["mean"] > 0


def test_baseline_get_top_apps():
    model = BaselineModel(user_id=1)
    raw = generate_synthetic_data(days=3, samples_per_hour=12)
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features = extractor.extract_session_features(raw)
    model.update(features)

    apps = model.get_top_apps(10, 1, limit=5)
    assert isinstance(apps, list)
    if apps:
        assert "app" in apps[0]
        assert "count" in apps[0]


def test_baseline_save_load():
    model = BaselineModel(user_id=1)
    raw = generate_synthetic_data(days=3, samples_per_hour=12)
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features = extractor.extract_session_features(raw)
    model.update(features)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.json"
        model.save(path)
        assert path.exists()

        loaded = BaselineModel.load(path)
        assert loaded.user_id == 1
        assert loaded.total_days == model.total_days
        assert loaded.has_sufficient_data(30)


def test_deviation_detector_normal():
    model = BaselineModel(user_id=1)
    raw = generate_synthetic_data(days=7, samples_per_hour=12)
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features = extractor.extract_session_features(raw)

    mid = len(features) // 2
    model.update(features.iloc[:mid])

    detector = DeviationDetector(model)
    row = features.iloc[mid]
    score = detector.score_window(row)
    assert "overall_deviation" in score
    assert "severity" in score
    assert "z_scores" in score
    assert score["severity"] in ("normal", "mild", "moderate", "severe")


def test_deviation_daily_summary():
    model = BaselineModel(user_id=1)
    raw = generate_synthetic_data(days=7, samples_per_hour=12)
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features = extractor.extract_session_features(raw)

    mid = len(features) // 2
    model.update(features.iloc[:mid])

    detector = DeviationDetector(model)
    test_features = features.iloc[mid:]
    summary = detector.daily_summary(test_features)
    assert "total_windows" in summary
    assert "anomaly_count" in summary
    assert "severity_counts" in summary
    assert summary["total_windows"] == len(test_features)


def test_deviation_analyze_dataframe():
    model = BaselineModel(user_id=1)
    raw = generate_synthetic_data(days=7, samples_per_hour=12)
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features = extractor.extract_session_features(raw)

    mid = len(features) // 2
    model.update(features.iloc[:mid])

    detector = DeviationDetector(model)
    anomalies = detector.analyze_dataframe(features.iloc[mid:])
    assert isinstance(anomalies, list)
    for a in anomalies:
        assert a["severity"] != "normal"
