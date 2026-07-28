"""Characterization tests for P1-2 forbidden-word constant centralization.

Post-refactoring state: ``mindflow.domain.forbidden_words`` is the single
source of truth for ``FORBIDDEN_MEDICAL_TERMS`` (4 canonical terms) and
``CRISIS_KEYWORDS`` (union of safety_guard + CrisisDetector terms, 15 unique).

These tests lock the refactored behavior:
  - ``forbidden_words.py`` provides the canonical constants.
  - Schemas and agents import the canonical 4-term set.
  - Safety guard imports the 4-term set and adds 8 locally → effective 12.
  - Both safety guard and CrisisDetector import the united CRISIS_KEYWORDS (15).
  - Behavioral surface tests remain green.
"""

from __future__ import annotations

# ── Canonical source: forbidden_words module ────────────────────────────────

class TestForbiddenWordsModule:
    """The canonical module must exist and export the correct constants."""

    def test_module_exists_and_exports_constants(self) -> None:
        """GREEN: forbidden_words module exists and exports both constants."""
        from mindflow.domain.forbidden_words import (
            CRISIS_KEYWORDS,
            FORBIDDEN_MEDICAL_TERMS,
        )

        assert FORBIDDEN_MEDICAL_TERMS is not None
        assert CRISIS_KEYWORDS is not None

    def test_forbidden_medical_terms_has_exactly_4(self) -> None:
        from mindflow.domain.forbidden_words import FORBIDDEN_MEDICAL_TERMS

        assert len(FORBIDDEN_MEDICAL_TERMS) == 4
        assert "诊断" in FORBIDDEN_MEDICAL_TERMS
        assert "治疗" in FORBIDDEN_MEDICAL_TERMS
        assert "患者" in FORBIDDEN_MEDICAL_TERMS
        assert "处方" in FORBIDDEN_MEDICAL_TERMS

    def test_crisis_keywords_has_28_union_terms(self) -> None:
        from mindflow.domain.forbidden_words import CRISIS_KEYWORDS

        assert len(CRISIS_KEYWORDS) == 28  # P2-3: expanded from 15 → 28
        # From safety_guard (8)
        assert "自杀" in CRISIS_KEYWORDS
        assert "自残" in CRISIS_KEYWORDS
        assert "轻生" in CRISIS_KEYWORDS
        assert "一了百了" in CRISIS_KEYWORDS
        # From CrisisDetector — unique additions
        assert "想死" in CRISIS_KEYWORDS
        assert "死了算了" in CRISIS_KEYWORDS
        assert "没有意义" in CRISIS_KEYWORDS
        assert "自伤" in CRISIS_KEYWORDS
        assert "结束自己的生命" in CRISIS_KEYWORDS
        # Shared by both
        assert "结束生命" in CRISIS_KEYWORDS
        assert "不想活" in CRISIS_KEYWORDS
        assert "活不下去" in CRISIS_KEYWORDS
        assert "伤害自己" in CRISIS_KEYWORDS
        assert "不想活了" in CRISIS_KEYWORDS
        assert "撑不下去" in CRISIS_KEYWORDS


# ── GREEN single-source: schemas, agents, safety_guard all trace to same object ─

class TestSingleSourceIdentity:
    """GREEN: All consumers reference the same canonical frozenset object."""

    def test_schemas_and_agents_reference_same_object(self) -> None:
        """After refactoring, both schemas and agents import the same constant."""
        from mindflow.agents.types import FORBIDDEN_WORDS
        from mindflow.domain.forbidden_words import FORBIDDEN_MEDICAL_TERMS

        # FORBIDDEN_WORDS is a re-export alias pointing to FORBIDDEN_MEDICAL_TERMS
        assert FORBIDDEN_WORDS is FORBIDDEN_MEDICAL_TERMS
        assert FORBIDDEN_WORDS == FORBIDDEN_MEDICAL_TERMS

    def test_safety_guard_medical_base_is_canonical(self) -> None:
        """Safety guard's base 4 terms are the canonical set."""
        from mindflow.domain.forbidden_words import FORBIDDEN_MEDICAL_TERMS
        from mindflow.services.safety_guard import _FORBIDDEN_MEDICAL

        # The canonical 4 are all present in safety guard's 12-term set
        for word in FORBIDDEN_MEDICAL_TERMS:
            assert word in _FORBIDDEN_MEDICAL

    def test_crisis_detector_uses_canonical_crisis_keywords(self) -> None:
        """CrisisDetector's keyword set IS the canonical CRISIS_KEYWORDS."""
        from mindflow.domain.forbidden_words import CRISIS_KEYWORDS
        from mindflow.infrastructure.security.crisis_detector import (
            _CRISIS_KEYWORDS,
        )

        # _CRISIS_KEYWORDS is aliased directly to CRISIS_KEYWORDS
        assert _CRISIS_KEYWORDS is CRISIS_KEYWORDS
        assert _CRISIS_KEYWORDS == CRISIS_KEYWORDS


# ── Schemas: 4-term import, validator behavior ─────────────────────────────

class TestSchemasForbiddenWords:
    """Post-refactoring: schemas uses canonical FORBIDDEN_MEDICAL_TERMS."""

    def test_schemas_imports_canonical_4_terms(self) -> None:
        from mindflow.infrastructure.llm.schemas import (
            FORBIDDEN_MEDICAL_TERMS,
        )

        assert len(FORBIDDEN_MEDICAL_TERMS) == 4
        assert "诊断" in FORBIDDEN_MEDICAL_TERMS
        assert "治疗" in FORBIDDEN_MEDICAL_TERMS
        assert "患者" in FORBIDDEN_MEDICAL_TERMS
        assert "处方" in FORBIDDEN_MEDICAL_TERMS

    def test_schemas_does_not_contain_safety_guard_extensions(self) -> None:
        from mindflow.infrastructure.llm.schemas import (
            FORBIDDEN_MEDICAL_TERMS,
        )

        assert "药物" not in FORBIDDEN_MEDICAL_TERMS
        assert "剂量" not in FORBIDDEN_MEDICAL_TERMS
        assert "复诊" not in FORBIDDEN_MEDICAL_TERMS
        assert "手术" not in FORBIDDEN_MEDICAL_TERMS

    # ── Behavioral: LLM validator still rejects forbidden words ─────────────

    def test_validator_rejects_zhenduan(self) -> None:
        import pytest
        from pydantic import ValidationError

        from mindflow.infrastructure.llm.schemas import LLMAttributionResult

        payload = {
            "procrastination_types": ["impulsivity"],
            "type_confidence": {"impulsivity": 0.8},
            "cbt_technique": "stimulus_control",
            "response_text": "我诊断你有拖延症",
            "next_action": "试试番茄钟",
        }
        with pytest.raises(ValidationError, match="NF-S7"):
            LLMAttributionResult.model_validate(payload)

    def test_validator_rejects_zhiliao_in_next_action(self) -> None:
        import pytest
        from pydantic import ValidationError

        from mindflow.infrastructure.llm.schemas import LLMAttributionResult

        payload = {
            "procrastination_types": ["impulsivity"],
            "type_confidence": {"impulsivity": 0.8},
            "cbt_technique": "stimulus_control",
            "response_text": "试试番茄钟",
            "next_action": "建议接受治疗并开具处方",
        }
        with pytest.raises(ValidationError):
            LLMAttributionResult.model_validate(payload)

    def test_validator_allows_non_medical_text(self) -> None:
        from mindflow.infrastructure.llm.schemas import LLMAttributionResult

        payload = {
            "procrastination_types": ["impulsivity"],
            "type_confidence": {"impulsivity": 0.82},
            "cbt_technique": "stimulus_control",
            "response_text": "试试把任务拆成更小的步骤。",
            "next_action": "写一个最小可行草稿",
        }
        result = LLMAttributionResult.model_validate(payload)
        assert result.response_text == "试试把任务拆成更小的步骤。"


# ── Agents/types: 4-term import, backward compat ───────────────────────────

class TestAgentsTypesForbiddenWords:
    """Post-refactoring: types.FORBIDDEN_WORDS is a re-export of the canonical set."""

    def test_has_exactly_4_terms(self) -> None:
        from mindflow.agents.types import FORBIDDEN_WORDS

        assert len(FORBIDDEN_WORDS) == 4

    def test_contains_canonical_medical_terms(self) -> None:
        from mindflow.agents.types import FORBIDDEN_WORDS

        assert "诊断" in FORBIDDEN_WORDS
        assert "治疗" in FORBIDDEN_WORDS
        assert "患者" in FORBIDDEN_WORDS
        assert "处方" in FORBIDDEN_WORDS

    def test_forbidden_words_detector_function(self) -> None:
        from mindflow.agents.types import _contains_forbidden_words

        assert _contains_forbidden_words("诊断结果") == "诊断"
        assert _contains_forbidden_words("治疗建议") == "治疗"
        assert _contains_forbidden_words("安全文本") is None


# ── Safety guard: 12 medical terms (4 canonical + 8 local) ─────────────────

class TestSafetyGuardMedicalTerms:
    """Post-refactoring: safety guard has 4 canonical + 8 local = 12 medical terms."""

    def test_has_exactly_12_terms(self) -> None:
        from mindflow.services.safety_guard import _FORBIDDEN_MEDICAL

        assert len(_FORBIDDEN_MEDICAL) == 12

    def test_contains_all_4_canonical(self) -> None:
        from mindflow.services.safety_guard import _FORBIDDEN_MEDICAL

        assert "诊断" in _FORBIDDEN_MEDICAL
        assert "治疗" in _FORBIDDEN_MEDICAL
        assert "患者" in _FORBIDDEN_MEDICAL
        assert "处方" in _FORBIDDEN_MEDICAL

    def test_contains_8_local_extensions(self) -> None:
        from mindflow.services.safety_guard import _FORBIDDEN_MEDICAL

        assert "药物" in _FORBIDDEN_MEDICAL
        assert "剂量" in _FORBIDDEN_MEDICAL
        assert "复诊" in _FORBIDDEN_MEDICAL
        assert "挂号" in _FORBIDDEN_MEDICAL
        assert "住院" in _FORBIDDEN_MEDICAL
        assert "手术" in _FORBIDDEN_MEDICAL
        assert "服药" in _FORBIDDEN_MEDICAL
        assert "副作用" in _FORBIDDEN_MEDICAL

    def test_check_forbidden_content_blocks_medical(self) -> None:
        from mindflow.services.safety_guard import check_forbidden_content

        result = check_forbidden_content("通知", "这是治疗建议")
        assert result.level == "block"
        assert result.category == "medical_language"

    def test_check_forbidden_content_blocks_extension(self) -> None:
        from mindflow.services.safety_guard import check_forbidden_content

        result = check_forbidden_content("通知", "请按时服药")
        assert result.level == "block"
        assert result.category == "medical_language"


# ── Safety guard: crisis terms now use CRISIS_KEYWORDS (15 terms) ──────────

class TestSafetyGuardCrisisTerms:
    """Post-refactoring: safety guard crisis terms are the full union (28 after P2-3)."""

    def test_has_28_crisis_terms(self) -> None:
        from mindflow.services.safety_guard import _FORBIDDEN_CRISIS

        assert len(_FORBIDDEN_CRISIS) == 28  # P2-3: expanded from 15 → 28

    def test_contains_original_safety_guard_terms(self) -> None:
        from mindflow.services.safety_guard import _FORBIDDEN_CRISIS

        assert "自杀" in _FORBIDDEN_CRISIS
        assert "自残" in _FORBIDDEN_CRISIS
        assert "轻生" in _FORBIDDEN_CRISIS
        assert "一了百了" in _FORBIDDEN_CRISIS

    def test_now_also_contains_crisis_detector_only_terms(self) -> None:
        """Safety guard now catches CrisisDetector-only legacy terms too."""
        from mindflow.services.safety_guard import _FORBIDDEN_CRISIS

        assert "想死" in _FORBIDDEN_CRISIS
        assert "死了算了" in _FORBIDDEN_CRISIS
        assert "没有意义" in _FORBIDDEN_CRISIS
        assert "自伤" in _FORBIDDEN_CRISIS

    def test_check_forbidden_content_blocks_crisis(self) -> None:
        from mindflow.services.safety_guard import check_forbidden_content

        result = check_forbidden_content("通知", "我想自杀")
        assert result.level == "block"
        assert result.category == "crisis_language"

    def test_check_forbidden_content_blocks_legacy_crisis(self) -> None:
        """Safety guard now blocks CrisisDetector-only legacy terms."""
        from mindflow.services.safety_guard import check_forbidden_content

        # "想死" was previously only in CrisisDetector
        result = check_forbidden_content("通知", "我想死")
        assert result.level == "block"
        assert result.category == "crisis_language"


# ── CrisisDetector: now uses CRISIS_KEYWORDS (15 terms) ────────────────────

class TestCrisisDetectorKeywords:
    """Post-refactoring: CrisisDetector uses the united CRISIS_KEYWORDS (28 after P2-3)."""

    def test_has_28_terms(self) -> None:
        from mindflow.infrastructure.security.crisis_detector import (
            _CRISIS_KEYWORDS,
        )

        assert len(_CRISIS_KEYWORDS) == 28  # P2-3: expanded from 15 → 28

    def test_detects_legacy_only_terms(self) -> None:
        from mindflow.infrastructure.security.crisis_detector import CrisisDetector

        detector = CrisisDetector()
        level, _ = detector.scan("我想死")
        assert level.value == "high"

        level, _ = detector.scan("感觉死了算了")
        assert level.value == "high"

        level, _ = detector.scan("活着没有意义了")
        assert level.value == "high"

    def test_now_detects_safety_guard_only_terms(self) -> None:
        """CrisisDetector now detects terms previously only in safety_guard."""
        from mindflow.infrastructure.security.crisis_detector import CrisisDetector

        detector = CrisisDetector()
        # "自残" was in safety_guard but NOT in old CrisisDetector
        level, _ = detector.scan("我有自残倾向")
        assert level.value == "high", (
            "Post-refactoring: CrisisDetector should detect safety_guard-only terms"
        )

        # "一了百了" was in safety_guard but NOT in old CrisisDetector
        level, _ = detector.scan("感觉一了百了")
        assert level.value == "high"

        # "轻生" was in safety_guard but NOT in old CrisisDetector
        level, _ = detector.scan("有轻生念头")
        assert level.value == "high"

    def test_safe_text_returns_none(self) -> None:
        from mindflow.infrastructure.security.crisis_detector import CrisisDetector

        detector = CrisisDetector()
        level, _ = detector.scan("今天天气不错")
        assert level.value == "none"


# ── GREEN gate: single-source proof ─────────────────────────────────────────

class TestGreenSingleSourceGate:
    """GREEN: post-refactoring, all consumers trace to one canonical source."""

    def test_forbidden_medical_terms_module_exists(self) -> None:
        import importlib

        try:
            importlib.import_module("mindflow.domain.forbidden_words")
            exists = True
        except ImportError:
            exists = False
        assert exists, "GREEN: forbidden_words module should exist after refactoring"

    def test_schemas_types_share_same_object_post_refactor(self) -> None:
        """After refactoring, importing FORBIDDEN_MEDICAL_TERMS from either
        schemas or the canonical source yields the same object."""
        from mindflow.domain.forbidden_words import FORBIDDEN_MEDICAL_TERMS
        from mindflow.infrastructure.llm import schemas

        assert schemas.FORBIDDEN_MEDICAL_TERMS is FORBIDDEN_MEDICAL_TERMS
        assert schemas.FORBIDDEN_MEDICAL_TERMS == FORBIDDEN_MEDICAL_TERMS


# ── P2-3 RED: expanded CRISIS_KEYWORDS terms ──────────────────────────────────

class TestP2CrisisKeywordsExpanded:
    """RED: P2-3 should expand CRISIS_KEYWORDS to include all suggested terms."""

    def test_crisis_keywords_should_have_28_unique_terms(self) -> None:
        """RED: After P2-3, CRISIS_KEYWORDS should have 28 unique terms
        (15 legacy + 13 genuinely new from 17 suggestions, with 4 overlaps)."""
        from mindflow.domain.forbidden_words import CRISIS_KEYWORDS

        assert len(CRISIS_KEYWORDS) == 28, (
            f"P2-3 RED: expected 28 unique terms, got {len(CRISIS_KEYWORDS)}"
        )

    _new_p2_terms = frozenset({
        "活够了", "活腻了", "割腕", "割脉", "跳楼",
        "服毒", "上吊", "遗书", "救救我", "我受不了了",
        "太累了", "走投无路", "安眠药",
    })

    def test_all_new_p2_terms_are_present(self) -> None:
        """RED: Every genuinely new P2-3 suggested term should be in the set."""
        from mindflow.domain.forbidden_words import CRISIS_KEYWORDS

        for term in self._new_p2_terms:
            assert term in CRISIS_KEYWORDS, (
                f"P2-3 RED: '{term}' should be in CRISIS_KEYWORDS"
            )

    def test_all_17_suggested_terms_are_present(self) -> None:
        """RED: All 17 terms from the plan (both new and existing) should be present."""
        from mindflow.domain.forbidden_words import CRISIS_KEYWORDS

        all_suggested = {
            "活够了", "活腻了", "一了百了", "割腕", "割脉", "跳楼",
            "服毒", "上吊", "遗书", "救救我", "我受不了了", "太累了",
            "撑不下去", "走投无路", "自残", "自伤", "安眠药",
        }
        for term in all_suggested:
            assert term in CRISIS_KEYWORDS, (
                f"P2-3 RED: suggested term '{term}' should be in CRISIS_KEYWORDS"
            )

    def test_all_15_legacy_terms_are_still_present(self) -> None:
        """RED: All 15 legacy crisis terms must remain intact in the expanded set."""
        from mindflow.domain.forbidden_words import CRISIS_KEYWORDS

        legacy_15 = frozenset({
            "自杀", "自残", "伤害自己", "不想活", "活不下去",
            "结束生命", "一了百了", "轻生", "结束自己的生命",
            "自伤", "撑不下去", "不想活了", "没有意义", "死了算了", "想死",
        })
        for term in legacy_15:
            assert term in CRISIS_KEYWORDS, (
                f"P2-3 RED: legacy term '{term}' should still be in CRISIS_KEYWORDS"
            )

    # ── safety_guard sees expanded crisis terms ─────────────────────────

    def test_safety_guard_crisis_should_have_28_terms(self) -> None:
        """RED: After P2-3, safety_guard's _FORBIDDEN_CRISIS should also have 28 terms."""
        from mindflow.services.safety_guard import _FORBIDDEN_CRISIS

        assert len(_FORBIDDEN_CRISIS) == 28, (
            f"P2-3 RED: expected 28 terms in safety_guard crisis, got {len(_FORBIDDEN_CRISIS)}"
        )

    def test_crisis_detector_keywords_should_have_28_terms(self) -> None:
        """RED: After P2-3, _CRISIS_KEYWORDS in crisis_detector should have 28 terms."""
        from mindflow.infrastructure.security.crisis_detector import _CRISIS_KEYWORDS

        assert len(_CRISIS_KEYWORDS) == 28, (
            f"P2-3 RED: expected 28 terms in crisis_detector, got {len(_CRISIS_KEYWORDS)}"
        )

    # ── Single-source proof still holds ──────────────────────────────────

    def test_crisis_keywords_single_source_still_holds(self) -> None:
        """Post-P2-3, all consumers must still reference the same frozenset object."""
        from mindflow.domain.forbidden_words import CRISIS_KEYWORDS
        from mindflow.infrastructure.security.crisis_detector import (
            _CRISIS_KEYWORDS,
        )
        from mindflow.services.safety_guard import _FORBIDDEN_CRISIS

        assert _CRISIS_KEYWORDS is CRISIS_KEYWORDS, (
            "P2-3: crisis_detector._CRISIS_KEYWORDS must be the same object as canonical"
        )
        assert _FORBIDDEN_CRISIS is CRISIS_KEYWORDS, (
            "P2-3: safety_guard._FORBIDDEN_CRISIS must be the same object as canonical"
        )
