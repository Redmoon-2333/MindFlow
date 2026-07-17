"""Tests for LLMAttributionResult schema.

Coverage:
  - Valid full payload parses correctly
  - Valid minimal payload (optional fields omitted)
  - Invalid JSON raises ValidationError
  - Forbidden words in response_text raise ValidationError (NF-S7)
  - TYPE CONFIDENCE outside [0,1] raises ValidationError
  - Extra fields forbidden (strict mode)
  - Too many procrastination types (>3) raises ValidationError
  - Type confidence missing key for a declared type raises ValidationError
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mindflow.infrastructure.llm.schemas import LLMAttributionResult


class TestLLMAttributionResult:
    """LLM output contract validation tests."""

    VALID_PAYLOAD = {
        "procrastination_types": ["impulsivity", "emotional_regulation"],
        "type_confidence": {"impulsivity": 0.82, "emotional_regulation": 0.67},
        "cognitive_distortions": ["all-or-nothing thinking", "catastrophizing"],
        "cbt_technique": "stimulus_control",
        "response_text": "我注意到你今天的注意力集中时间较短，切换频繁。试试一个番茄钟？",
        "next_action": "设置一个番茄钟，专注25分钟",
    }

    def test_valid_full_payload(self) -> None:
        """A fully populated valid payload should parse successfully."""
        result = LLMAttributionResult.model_validate(self.VALID_PAYLOAD)
        assert len(result.procrastination_types) == 2
        assert result.cbt_technique == "stimulus_control"
        assert len(result.response_text) <= 500

    def test_valid_minimal_payload(self) -> None:
        """A minimal valid payload (optional fields omitted) should parse."""
        payload = {
            "procrastination_types": ["task_aversion"],
            "type_confidence": {"task_aversion": 0.75},
            "cbt_technique": "graded_exposure",
            "response_text": "试试把任务拆成更小的步骤。",
            "next_action": "写一个最小可行草稿",
        }
        result = LLMAttributionResult.model_validate(payload)
        assert len(result.procrastination_types) == 1
        assert result.cognitive_distortions == []

    def test_validates_from_json_string(self) -> None:
        """model_validate_json should parse a JSON string."""
        json_str = json.dumps(self.VALID_PAYLOAD, ensure_ascii=False)
        result = LLMAttributionResult.model_validate_json(json_str)
        assert result.cbt_technique == "stimulus_control"

    def test_rejects_forbidden_word_zhenduan(self) -> None:
        """"诊断" in response_text should raise ValidationError."""
        payload = dict(self.VALID_PAYLOAD)
        payload["response_text"] = "我诊断你有拖延症"
        with pytest.raises(ValidationError, match="NF-S7"):
            LLMAttributionResult.model_validate(payload)

    def test_rejects_forbidden_word_zhiliao(self) -> None:
        """"治疗" in response_text should raise ValidationError."""
        payload = dict(self.VALID_PAYLOAD)
        payload["response_text"] = "建议你接受治疗"
        with pytest.raises(ValidationError, match="NF-S7"):
            LLMAttributionResult.model_validate(payload)

    def test_rejects_forbidden_word_huanzhe(self) -> None:
        """"患者" in response_text should raise ValidationError."""
        payload = dict(self.VALID_PAYLOAD)
        payload["response_text"] = "患者应注意作息"
        with pytest.raises(ValidationError, match="NF-S7"):
            LLMAttributionResult.model_validate(payload)

    def test_rejects_forbidden_word_chufang(self) -> None:
        """"处方" in response_text should raise ValidationError."""
        payload = dict(self.VALID_PAYLOAD)
        payload["response_text"] = "这是给你的处方建议"
        with pytest.raises(ValidationError, match="NF-S7"):
            LLMAttributionResult.model_validate(payload)

    def test_rejects_confidence_out_of_range_high(self) -> None:
        """Confidence > 1.0 should raise ValidationError."""
        payload = dict(self.VALID_PAYLOAD)
        payload["type_confidence"]["impulsivity"] = 1.5
        with pytest.raises(ValidationError, match="confidence"):
            LLMAttributionResult.model_validate(payload)

    def test_rejects_confidence_out_of_range_low(self) -> None:
        """Confidence < 0.0 should raise ValidationError."""
        payload = dict(self.VALID_PAYLOAD)
        payload["type_confidence"]["impulsivity"] = -0.1
        with pytest.raises(ValidationError, match="confidence"):
            LLMAttributionResult.model_validate(payload)

    def test_rejects_extra_fields(self) -> None:
        """Extra fields should be forbidden in strict mode."""
        payload = dict(self.VALID_PAYLOAD)
        payload["extra_field"] = "should_not_exist"
        with pytest.raises(ValidationError, match="extra"):
            LLMAttributionResult.model_validate(payload)

    def test_rejects_too_many_types(self) -> None:
        """More than 3 procrastination types should raise."""
        payload = dict(self.VALID_PAYLOAD)
        payload["procrastination_types"] = [
            "task_aversion",
            "impulsivity",
            "decisional",
            "perfectionism",
        ]
        with pytest.raises(ValidationError, match="procrastination_types"):
            LLMAttributionResult.model_validate(payload)

    def test_rejects_empty_types(self) -> None:
        """Empty procrastination_types should raise."""
        payload = dict(self.VALID_PAYLOAD)
        payload["procrastination_types"] = []
        with pytest.raises(ValidationError):
            LLMAttributionResult.model_validate(payload)

    def test_confidence_missing_key_for_type(self) -> None:
        """Missing confidence key for a declared type should raise."""
        payload = dict(self.VALID_PAYLOAD)
        payload["procrastination_types"] = ["impulsivity", "perfectionism"]
        # perfectionism not in type_confidence
        with pytest.raises(ValidationError, match="confidence"):
            LLMAttributionResult.model_validate(payload)

    def test_response_text_exceeds_max_length(self) -> None:
        """Response text longer than 500 chars should raise."""
        payload = dict(self.VALID_PAYLOAD)
        payload["response_text"] = "a" * 501
        with pytest.raises(ValidationError, match="response_text"):
            LLMAttributionResult.model_validate(payload)

    def test_invalid_cbt_technique(self) -> None:
        """Invalid cbt_technique value should raise."""
        payload = dict(self.VALID_PAYLOAD)
        payload["cbt_technique"] = "invalid_technique"
        with pytest.raises(ValidationError):
            LLMAttributionResult.model_validate(payload)

    def test_invalid_procrastination_type(self) -> None:
        """Invalid procrastination type should raise."""
        payload = dict(self.VALID_PAYLOAD)
        payload["procrastination_types"] = ["invalid_type"]
        with pytest.raises(ValidationError):
            LLMAttributionResult.model_validate(payload)


class TestReviewRegressionFixes:
    """Regressions for wave 6 review P1-1 / P2-2 (NF-S7 full coverage)."""

    def _base(self) -> dict:
        return {
            "procrastination_types": ["impulsivity"],
            "type_confidence": {"impulsivity": 0.8},
            "cognitive_distortions": [],
            "cbt_technique": "stimulus_control",
            "response_text": "试试番茄钟，把任务拆成 5 分钟的小块。",
            "next_action": "关闭聊天软件，专注 5 分钟",
        }

    def test_forbidden_word_in_next_action_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        from mindflow.infrastructure.llm.schemas import LLMAttributionResult

        data = self._base()
        data["next_action"] = "建议接受治疗并开具处方"
        with pytest.raises(ValidationError):
            LLMAttributionResult.model_validate(data)

    def test_forbidden_word_in_cognitive_distortions_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        from mindflow.infrastructure.llm.schemas import LLMAttributionResult

        data = self._base()
        data["cognitive_distortions"] = ["非黑即白思维", "患者式自我否定"]
        with pytest.raises(ValidationError):
            LLMAttributionResult.model_validate(data)

    def test_extra_confidence_keys_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        from mindflow.infrastructure.llm.schemas import LLMAttributionResult

        data = self._base()
        data["type_confidence"] = {"impulsivity": 0.8, "perfectionism": 0.3}
        with pytest.raises(ValidationError):
            LLMAttributionResult.model_validate(data)
