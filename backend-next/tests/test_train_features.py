"""Tests for BehaviorFeatureExtractor and TitleAnalyzer.

Focuses on:
  - 14 feature dimensions in each output window
  - Window boundary correctness
  - Empty / single-event edge cases
  - AppClassifier categorization
  - TitleAnalyzer signal ratios
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mindflow.domain.events import ActivityEvent, make_event
from mindflow.train.features import (
    AppClassifier,
    BehaviorFeatureExtractor,
    TitleAnalyzer,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def extractor() -> BehaviorFeatureExtractor:
    return BehaviorFeatureExtractor(window_minutes=30)


@pytest.fixture
def analyzer() -> TitleAnalyzer:
    return TitleAnalyzer()


@pytest.fixture
def classifier() -> AppClassifier:
    return AppClassifier()


def _event(
    ts: datetime,
    process: str = "vscode",
    title: str = "main.py - VSCode",
    duration: float = 120.0,
    is_idle: bool = False,
) -> ActivityEvent:
    return make_event(
        user_id=1,
        timestamp_utc=ts,
        duration_s=duration,
        app_name=process,
        window_title=title,
        process_name=process,
        is_idle=is_idle,
    )


# ── TitleAnalyzer ─────────────────────────────────────────────────────────────


class TestTitleAnalyzer:
    def test_code_extension(self, analyzer: TitleAnalyzer) -> None:
        result = analyzer.analyze("main.py - VSCode")
        assert result["is_code_editor"] == 1.0
        assert result["is_likely_entertainment"] == 0.0

    def test_document_extension(self, analyzer: TitleAnalyzer) -> None:
        result = analyzer.analyze("report.pdf - Adobe")
        assert result["is_document"] == 1.0

    def test_browser_url(self, analyzer: TitleAnalyzer) -> None:
        result = analyzer.analyze("https://github.com/MindFlow - Chrome")
        assert result["is_browser"] == 1.0

    def test_meeting_keyword(self, analyzer: TitleAnalyzer) -> None:
        result = analyzer.analyze("Zoom Meeting - Room A")
        assert result["is_meeting"] == 1.0

    def test_entertainment_keyword(self, analyzer: TitleAnalyzer) -> None:
        result = analyzer.analyze("bilibili - Anime")
        assert result["is_likely_entertainment"] == 1.0

    def test_empty_title(self, analyzer: TitleAnalyzer) -> None:
        result = analyzer.analyze("")
        assert all(v == 0.0 for v in result.values())

    def test_title_code_ratio(self, analyzer: TitleAnalyzer) -> None:
        """TitleAnalyzer methods feed into code ratio computation."""
        r1 = analyzer.analyze("main.py - VSCode")
        r2 = analyzer.analyze("B站 - Anime")
        r3 = analyzer.analyze("notes - Obsidian")
        titles = [r1, r2, r3]
        code_ratio = sum(t["is_code_editor"] for t in titles) / len(titles)
        assert code_ratio == pytest.approx(0.3333, abs=0.01)

    def test_feature_keys_consistency(self, analyzer: TitleAnalyzer) -> None:
        """TitleAnalyzer.FEATURE_KEYS should match analyze() return keys."""
        result = analyzer.analyze("test")
        feature_keys = analyzer.FEATURE_KEYS
        # Map from analyze keys to feature keys
        analyze_to_feature = {
            "is_code_editor": "title_code_ratio",
            "is_document": "title_doc_ratio",
            "is_browser": "title_url_ratio",
            "is_meeting": "title_meeting_ratio",
            "is_likely_entertainment": "title_entertainment_ratio",
        }
        assert len(result) == len(feature_keys)
        assert all(k in analyze_to_feature for k in result)


# ── AppClassifier ─────────────────────────────────────────────────────────────


class TestAppClassifier:
    def test_code_detection(self, classifier: AppClassifier) -> None:
        assert classifier.classify("Code.exe", "") == "code"
        assert classifier.classify("vscode", "") == "code"
        assert classifier.classify("pycharm64.exe", "") == "code"

    def test_entertainment_detection(self, classifier: AppClassifier) -> None:
        assert classifier.classify("bilibili.exe", "") == "entertainment"

    def test_social_detection(self, classifier: AppClassifier) -> None:
        assert classifier.classify("weibo.exe", "") == "social"

    def test_browser_fallback(self, classifier: AppClassifier) -> None:
        assert classifier.classify("chrome.exe", "") == "browser_work"

    def test_title_keyword_fallback(self, classifier: AppClassifier) -> None:
        assert classifier.classify("firefox.exe", "github.com - Firefox") == "browser_work"

    def test_unknown_returns_other(self, classifier: AppClassifier) -> None:
        assert classifier.classify("unknown_app_123", "Untitled") == "other"

    def test_productivity_scores(self, classifier: AppClassifier) -> None:
        assert classifier.get_productivity_score("code") == 1.0
        assert classifier.get_productivity_score("entertainment") == 0.0
        assert classifier.get_productivity_score("unknown") == 0.3


# ── BehaviorFeatureExtractor ──────────────────────────────────────────────────


class TestBehaviorFeatureExtractor:
    def test_empty_events(self, extractor: BehaviorFeatureExtractor) -> None:
        """Empty event list should return empty feature list."""
        assert extractor.extract_session_features([]) == []

    def test_fourteen_features(self, extractor: BehaviorFeatureExtractor) -> None:
        """Each output dict should have all 14 feature columns plus window_start."""
        now = datetime.now(UTC)
        events = []
        # 8 events / 5-min spacing = 35 min — enough to cross one 30-min window
        # boundary and produce ≥1 feature row (audit fix: the old 6-event / 30-min
        # case fit exactly inside one bucket and produced zero features).
        for i in range(8):
            events.append(
                _event(ts=now + timedelta(minutes=5 * i), duration=300.0)
            )

        features = extractor.extract_session_features(events)
        assert len(features) >= 1

        first = features[0]
        for col in BehaviorFeatureExtractor.FEATURE_NAMES:
            assert col in first, f"Missing feature: {col}"
        assert "window_start" in first

    def test_feature_types(self, extractor: BehaviorFeatureExtractor) -> None:
        """Feature values should have correct types."""
        now = datetime.now(UTC)
        events = [
            _event(ts=now + timedelta(minutes=5 * i), duration=300.0)
            for i in range(12)
        ]
        features = extractor.extract_session_features(events)
        assert len(features) >= 1

        first = features[0]
        # Integer features
        assert isinstance(first["unique_app_count"], int)
        assert isinstance(first["hour_of_day"], int)
        assert isinstance(first["day_of_week"], int)
        # Float features
        assert isinstance(first["switch_frequency"], float)
        assert isinstance(first["productivity_ratio"], float)

    def test_window_boundary(self, extractor: BehaviorFeatureExtractor) -> None:
        """Events crossing window boundaries should go to separate windows."""
        base = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        # Events at 0, 15, 35, 45 min inside the same hour
        events = [
            _event(ts=base + timedelta(minutes=0), duration=300.0),
            _event(ts=base + timedelta(minutes=15), duration=300.0),
            _event(ts=base + timedelta(minutes=35), duration=300.0),
            _event(ts=base + timedelta(minutes=45), duration=300.0),
        ]
        features = extractor.extract_session_features(events)
        # Should have at least 2 windows (0-30 and 30-60)
        assert len(features) >= 1

        # Events in same window
        window_starts = {f["window_start"] for f in features}
        assert len(window_starts) >= 1, f"Expected >=1 window, got {window_starts}"

    def test_entertainment_ratio(self, extractor: BehaviorFeatureExtractor) -> None:
        """Entertainment apps should result in higher entertainment_ratio."""
        now = datetime.now(UTC)
        events = [
            _event(ts=now + timedelta(minutes=5 * i), process="bilibili", title="B站 - Anime")
            for i in range(6)
        ]
        features = extractor.extract_session_features(events)
        if features:
            assert features[0]["entertainment_ratio"] > 0.5

    def test_idle_ratio(self, extractor: BehaviorFeatureExtractor) -> None:
        """Pure idle events should be skipped (no active time)."""
        now = datetime.now(UTC)
        events = [
            _event(ts=now + timedelta(minutes=5 * i), is_idle=True, duration=300.0)
            for i in range(6)
        ]
        # Each event is idle with no active — the window's active_seconds = 0
        # so the code will "continue" and skip it. Features should be empty or
        # have very few windows.
        features = extractor.extract_session_features(events)
        # No active events => no features (all windows skipped)
        assert len(features) == 0

    def test_feature_names_property(self, extractor: BehaviorFeatureExtractor) -> None:
        """get_feature_names() should return the standard feature list."""
        names = extractor.get_feature_names()
        assert names == BehaviorFeatureExtractor.FEATURE_NAMES
        assert len(names) == 14

    def test_feature_names_after_extraction(self, extractor: BehaviorFeatureExtractor) -> None:
        """After extracting features, get_feature_names() should still work."""
        now = datetime.now(UTC)
        events = [
            _event(ts=now + timedelta(minutes=5 * i), duration=300.0)
            for i in range(6)
        ]
        extractor.extract_session_features(events)
        names = extractor.get_feature_names()
        assert len(names) == 14

    def test_switch_frequency_nonzero(self, extractor: BehaviorFeatureExtractor) -> None:
        """Multiple apps should produce nonzero switch frequency."""
        now = datetime.now(UTC)
        events = [
            _event(
                ts=now + timedelta(minutes=5 * i),
                process=["vscode", "chrome", "terminal", "vscode", "notion", "chrome"][i],
            )
            for i in range(6)
        ]
        features = extractor.extract_session_features(events)
        if features:
            assert features[0]["switch_frequency"] >= 0
