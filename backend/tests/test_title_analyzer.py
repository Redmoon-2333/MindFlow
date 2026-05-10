import pytest
from mindflow.analyzer.title_analyzer import TitleAnalyzer, BehaviorOnlyLabeler


@pytest.fixture
def analyzer():
    return TitleAnalyzer()


def test_empty_title(analyzer):
    result = analyzer.analyze("")
    assert result["is_browser"] is False
    assert result["is_code_editor"] is False
    assert result["url_domain"] is None


def test_none_title(analyzer):
    result = analyzer.analyze(None)
    assert result["is_browser"] is False


def test_url_detection_github(analyzer):
    result = analyzer.analyze("github.com/MindFlow/main.py - Google Chrome")
    assert result["is_browser"] is True
    assert result["url_domain"] == "github.com"


def test_url_detection_bilibili(analyzer):
    result = analyzer.analyze("https://www.bilibili.com/video/BV1xx - Google Chrome")
    assert result["is_browser"] is True
    assert result["url_domain"] == "bilibili.com"


def test_code_file_detection(analyzer):
    result = analyzer.analyze("main.py - MindFlow - Visual Studio Code")
    assert result["is_code_editor"] is True
    assert result["file_extension"] == ".py"


def test_notebook_detection(analyzer):
    result = analyzer.analyze("analysis.ipynb - Jupyter")
    assert result["is_code_editor"] is True
    assert result["file_extension"] == ".ipynb"


def test_document_detection(analyzer):
    result = analyzer.analyze("论文初稿.md - Typora")
    assert result["is_document"] is True
    assert result["file_extension"] == ".md"


def test_pdf_detection(analyzer):
    result = analyzer.analyze("paper.pdf - Adobe Acrobat")
    assert result["is_document"] is True


def test_meeting_detection_zoom(analyzer):
    result = analyzer.analyze("会议中 - Zoom Meeting")
    assert result["is_meeting"] is True


def test_meeting_detection_teams(analyzer):
    result = analyzer.analyze("Weekly Standup - Microsoft Teams")
    assert result["is_meeting"] is True


def test_entertainment_anime(analyzer):
    result = analyzer.analyze("B站 - 番剧播放 - 第12集")
    assert result["is_likely_entertainment"] is True


def test_entertainment_live(analyzer):
    result = analyzer.analyze("直播间 - 游戏直播")
    assert result["is_likely_entertainment"] is True


def test_normal_webpage_no_special_flags(analyzer):
    result = analyzer.analyze("Google Search - Chrome")
    assert result["is_browser"] is False  # no URL pattern
    assert result["is_likely_entertainment"] is False


def test_behavior_only_labeler_code_file():
    labeler = BehaviorOnlyLabeler()
    row = {
        "switch_frequency": 5,
        "unique_app_count": 2,
        "max_app_duration": 1200,
        "idle_ratio": 0.02,
        "hour_of_day": 10,
    }
    title_features = {"is_code_editor": True, "is_document": False,
                      "is_likely_entertainment": False, "url_domain": None,
                      "is_meeting": False}
    label, conf = labeler.label(row, title_features)
    assert label == 1
    assert conf > 0.5


def test_behavior_only_labeler_entertainment():
    labeler = BehaviorOnlyLabeler()
    row = {
        "switch_frequency": 20,
        "unique_app_count": 4,
        "max_app_duration": 300,
        "idle_ratio": 0.05,
        "hour_of_day": 22,
    }
    title_features = {"is_code_editor": False, "is_document": False,
                      "is_likely_entertainment": True, "url_domain": "bilibili.com",
                      "is_meeting": False}
    label, conf = labeler.label(row, title_features)
    assert label == 0
    # Low confidence expected: sf=20 is ambiguous, only title signal strongly votes distraction
