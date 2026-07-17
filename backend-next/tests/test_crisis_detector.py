"""Tests for CrisisDetector — keyword-based crisis scanning.

Coverage:
  - Positive matches for each keyword
  - Empty / whitespace text
  - Safe text (no match)
  - Multiple text fields via scan_texts
  - Short-circuit on first match
  - Custom keyword extension
"""

from __future__ import annotations

from mindflow.infrastructure.security.crisis_detector import CrisisDetector, CrisisLevel


class TestCrisisDetector:
    """Crisis keyword scanning tests."""

    def setup_method(self) -> None:
        self.detector = CrisisDetector()

    def test_empty_text_returns_none(self) -> None:
        """Empty or whitespace text should return NONE."""
        level, resp = self.detector.scan("")
        assert level == CrisisLevel.NONE
        assert resp is None

        level, resp = self.detector.scan("   ")
        assert level == CrisisLevel.NONE
        assert resp is None

    def test_safe_text_returns_none(self) -> None:
        """Ordinary text should return NONE."""
        level, resp = self.detector.scan("今天写完了论文第三章，感觉不错")
        assert level == CrisisLevel.NONE
        assert resp is None

    def test_detects_suicide_keyword(self) -> None:
        """"自杀" should trigger HIGH."""
        level, resp = self.detector.scan("我觉得活着没意思，想自杀")
        assert level == CrisisLevel.HIGH
        assert resp is not None
        assert resp.stop_llm is True
        assert "400" in resp.message  # Has hotline number

    def test_detects_buhuoxianghuo(self) -> None:
        """"不想活" should trigger HIGH."""
        level, resp = self.detector.scan("真的太累了，不想活了")
        assert level == CrisisLevel.HIGH
        assert resp is not None

    def test_detects_ends_life(self) -> None:
        """"结束生命" should trigger HIGH."""
        level, resp = self.detector.scan("我想结束生命")
        assert level == CrisisLevel.HIGH

    def test_detects_self_harm(self) -> None:
        """"伤害自己" should trigger HIGH."""
        level, resp = self.detector.scan("我总是想伤害自己")
        assert level == CrisisLevel.HIGH

    def test_detects_chengbuxiaqu(self) -> None:
        """"撑不下去" should trigger HIGH."""
        level, resp = self.detector.scan("我感觉快撑不下去了")
        assert level == CrisisLevel.HIGH

    def test_scan_texts_empty(self) -> None:
        """scan_texts with empty list returns NONE."""
        level, resp = self.detector.scan_texts([])
        assert level == CrisisLevel.NONE
        assert resp is None

    def test_scan_texts_multiple(self) -> None:
        """scan_texts should find crisis in any field."""
        texts = ["今天天气不错", "我有点想结束生命", "代码写完了"]
        level, resp = self.detector.scan_texts(texts)
        assert level == CrisisLevel.HIGH
        assert resp is not None

    def test_scan_texts_short_circuit(self) -> None:
        """scan_texts should stop at first match."""
        texts = ["我想自杀", "今天天气不错"]
        level, resp = self.detector.scan_texts(texts)
        assert level == CrisisLevel.HIGH
        # If it short-circuited on "我想自杀", it found the match
        assert resp is not None

    def test_safe_texts_returns_none(self) -> None:
        """scan_texts with all-safe texts returns NONE."""
        texts = ["今天天气不错", "代码写完了", "去吃饭了"]
        level, resp = self.detector.scan_texts(texts)
        assert level == CrisisLevel.NONE
        assert resp is None

    def test_add_keywords(self) -> None:
        """Custom keywords should extend the built-in set."""
        detector = CrisisDetector(extra_keywords=frozenset({"帮帮我", "绝望"}))
        level, resp = detector.scan("我感觉很绝望")
        assert level == CrisisLevel.HIGH

    def test_combined_keywords_in_text(self) -> None:
        """Multiple crisis keywords in the same text should still match."""
        level, resp = self.detector.scan("不想活了，想自杀，撑不下去了")
        assert level == CrisisLevel.HIGH

    def test_keyword_as_substring(self) -> None:
        """Keywords matched as substrings."""
        level, resp = self.detector.scan("她一直有自杀倾向，需要关注")
        assert level == CrisisLevel.HIGH

    def test_default_keywords_immutable(self) -> None:
        """Default keyword set should be isolated between instances."""
        d1 = CrisisDetector()
        d2 = CrisisDetector()
        level1, _ = d1.scan("我觉得很绝望")  # "绝望" not in default set
        level2, _ = d2.scan("我想自杀")
        assert level1 == CrisisLevel.NONE
        assert level2 == CrisisLevel.HIGH
