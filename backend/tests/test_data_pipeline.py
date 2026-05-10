import pandas as pd

from mindflow.analyzer.data_pipeline import (
    BehaviorFeatureExtractor,
    AppClassifier,
    generate_synthetic_data,
)


def test_generate_synthetic_data():
    df = generate_synthetic_data(days=3, samples_per_hour=6)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "process_name" in df.columns
    assert "is_idle" in df.columns
    assert df["is_idle"].mean() > 0


def test_app_classifier_productivity():
    classifier = AppClassifier()
    assert classifier.classify("vscode.exe", "main.py - VSCode") == "code"
    assert classifier.classify("notion.exe", "notes") == "document"
    assert classifier.classify("bilibili.exe", "B站视频") == "entertainment"
    assert classifier.get_productivity_score("code") == 1.0
    assert classifier.get_productivity_score("entertainment") == 0.0
    assert classifier.get_productivity_score("other") == 0.3


def test_app_classifier_browser():
    classifier = AppClassifier()
    result = classifier.classify("chrome.exe", "github.com/MindFlow")
    assert result in ("code", "document", "browser_work", "communication",
                       "entertainment", "social", "other")


def test_app_classifier_unknown():
    classifier = AppClassifier()
    result = classifier.classify("weird_app.exe", "something weird")
    assert result == "other"


def test_feature_extractor_empty():
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    empty_df = pd.DataFrame()
    result = extractor.extract_session_features(empty_df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_feature_extractor():
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    raw = generate_synthetic_data(days=1, samples_per_hour=12)
    result = extractor.extract_session_features(raw)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "productivity_ratio" in result.columns
    assert "switch_frequency" in result.columns
    assert "idle_ratio" in result.columns
    for col in ("productivity_ratio", "entertainment_ratio", "social_ratio", "idle_ratio"):
        assert result[col].between(0.0, 1.0).all()


def test_daily_features():
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    raw = generate_synthetic_data(days=3, samples_per_hour=12)
    daily = extractor.extract_daily_features(raw)
    assert isinstance(daily, pd.DataFrame)
    assert not daily.empty
    assert "session_count" in daily.columns
    assert "total_active_hours" in daily.columns


def test_feature_names():
    extractor = BehaviorFeatureExtractor()
    names = extractor.get_feature_names()
    assert isinstance(names, list)
    assert len(names) >= 8
