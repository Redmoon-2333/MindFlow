"""Tests for CrisisDetector — keyword-based crisis scanning.

Coverage:
  - Positive matches for each keyword
  - Empty / whitespace text
  - Safe text (no match)
  - Multiple text fields via scan_texts
  - Short-circuit on first match
  - Custom keyword extension
  - P2-3: new suggested crisis terms (割腕, 跳楼, 服毒, etc.)
  - P2-3: whitespace-bypass detection (割 腕, 跳   楼, 自\\n杀)
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

    # ── P2-3 RED: new suggested crisis terms ─────────────────────────────────

    def test_detects_new_term_gewang(self) -> None:
        """RED: "割腕" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("有割腕的念头")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '割腕' should now be detected"
        assert resp is not None

    def test_detects_new_term_tiaolou(self) -> None:
        """RED: "跳楼" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("想跳楼")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '跳楼' should now be detected"
        assert resp is not None

    def test_detects_new_term_fudu(self) -> None:
        """RED: "服毒" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("服毒的想法")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '服毒' should now be detected"

    def test_detects_new_term_anmianyao(self) -> None:
        """RED: "安眠药" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("吃了一瓶安眠药")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '安眠药' should now be detected"

    def test_detects_new_term_yishu(self) -> None:
        """RED: "遗书" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("写好了遗书")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '遗书' should now be detected"

    def test_detects_new_term_jiujiuwo(self) -> None:
        """RED: "救救我" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("我受不了了，救救我")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '救救我' should now be detected"

    def test_detects_new_term_wo_shoubuliao(self) -> None:
        """RED: "我受不了了" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("我真的我受不了了")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '我受不了了' should now be detected"

    def test_detects_new_term_zoutouwulu(self) -> None:
        """RED: "走投无路" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("感觉走投无路了")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '走投无路' should now be detected"

    def test_detects_new_term_shangdiao(self) -> None:
        """RED: "上吊" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("想用上吊的方式结束一切")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '上吊' should now be detected"

    def test_detects_new_term_gemai(self) -> None:
        """RED: "割脉" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("割脉自尽")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '割脉' should now be detected"

    def test_detects_new_term_huogoule(self) -> None:
        """RED: "活够了" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("真的活够了")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '活够了' should now be detected"

    def test_detects_new_term_huonile(self) -> None:
        """RED: "活腻了" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("感觉活腻了")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '活腻了' should now be detected"

    def test_detects_new_term_tailile(self) -> None:
        """RED: "太累了" should trigger HIGH (P2-3 new term)."""
        level, resp = self.detector.scan("太累了不想继续")
        assert level == CrisisLevel.HIGH, "P2-3 RED: '太累了' should now be detected"

    # ── P2-3 RED: whitespace-bypass detection ────────────────────────────────

    def test_whitespace_bypass_gewang_with_space(self) -> None:
        """RED: "割 腕" (with space) should trigger HIGH."""
        level, resp = self.detector.scan("割 腕自尽")
        assert level == CrisisLevel.HIGH, (
            "P2-3 RED: whitespace bypass '割 腕' should trigger after normalization"
        )

    def test_whitespace_bypass_tiaolou_with_multiple_spaces(self) -> None:
        """RED: "跳   楼" (with multiple spaces) should trigger HIGH."""
        level, resp = self.detector.scan("想跳   楼")
        assert level == CrisisLevel.HIGH, (
            "P2-3 RED: whitespace bypass '跳   楼' should trigger after normalization"
        )

    def test_whitespace_bypass_zisha_with_newline(self) -> None:
        """RED: "自\\n杀" (with newline) should trigger HIGH."""
        level, resp = self.detector.scan("有自\n杀倾向")
        assert level == CrisisLevel.HIGH, (
            "P2-3 RED: whitespace bypass '自\\n杀' should trigger after normalization"
        )

    def test_whitespace_bypass_zican_with_tab(self) -> None:
        """RED: "自\\t残" (with tab) should trigger HIGH."""
        level, resp = self.detector.scan("有自\t残行为")
        assert level == CrisisLevel.HIGH, (
            "P2-3 RED: whitespace bypass '自\\t残' should trigger after normalization"
        )

    def test_normal_benign_text_no_false_positive(self) -> None:
        """Normal benign text with spaces should remain NONE (no false positive)."""
        level, resp = self.detector.scan("今天我割了院子里的杂草，做完后吃了药休息")
        assert level == CrisisLevel.NONE, (
            "P2-3: normal text should not trigger false positive"
        )
