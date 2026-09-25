"""Phase 1.3 regressions: prompt ⇄ schema alignment and deterministic semantics.

The panel prompts (``agents/experts.py``) and the Pydantic schemas
(``agents/schemas.py``) must describe the *same* JSON contract:

* every field a prompt asks for exists in the schema (nothing is silently
  dropped),
* ``extra="forbid"`` turns field drift into a loud parse failure,
* an "opinion" without an argument, without a legal type, without an evidence
  citation, or with an out-of-range confidence is not a valid opinion,
* an ``insufficient_data`` verdict must state its ``evidence_gaps``, and an
  empty object is never a verdict.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from mindflow.agents.experts import (
    ANALYST,
    ATTRIBUTION_EXPERTS,
    CRITIC,
    MODERATOR,
)
from mindflow.agents.orchestrator import (
    _parse_analyst_opinion,
    _parse_critic,
    _parse_expert_opinion,
    _parse_verdict,
)
from mindflow.agents.schemas import (
    AnalystOutput,
    AttributionOutput,
    CriticOutput,
    ModeratorOutput,
    validate_opinion_semantics,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Prompt / schema field alignment
# ═══════════════════════════════════════════════════════════════════════════════

ANALYST_PAYLOAD = {
    "patterns": [{"name": "专注度下降", "severity": "moderate", "description": "低于基线"}],
    "anomalies": [{"metric": "focus.longest_block", "detail": "仅3分钟"}],
    "top_concerns": ["专注度下降", "切换频率过高"],
    "evidence_citations": ["focus.focus_score"],
}

ATTRIBUTION_PAYLOAD = {
    "attribution_types": ["impulsivity"],
    "confidence": {"impulsivity": 0.82},
    "argument": "切换频繁 [证据: focus.switch_rate]",
    "evidence_citations": ["focus.switch_rate"],
    "cognitive_distortions": ["全或无思维"],
    "tmt_factors": {"Expectancy": "低", "Value": "低", "Impulsiveness": "高", "Delay": "高"},
    "emotion_pattern": "社交媒体避难",
    "is_emotion_driven": True,
}

MODERATOR_PAYLOAD = {
    "types": ["impulsivity"],
    "confidence": {"impulsivity": 0.8},
    "recommended_technique": "stimulus_control",
    "rationale": "综合各方意见",
    "dissent": ["TMT 认为启动延迟更关键"],
    "insufficient_data": False,
    "uncertainty": 0.2,
    "evidence_gaps": [],
}

CRITIC_PAYLOAD = {
    "approved": True,
    "issues": [],
    "critique_detail": "通过。",
}


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (AnalystOutput, ANALYST_PAYLOAD),
        (AttributionOutput, ATTRIBUTION_PAYLOAD),
        (ModeratorOutput, MODERATOR_PAYLOAD),
        (CriticOutput, CRITIC_PAYLOAD),
    ],
)
def test_every_prompt_field_round_trips(schema: type, payload: dict) -> None:
    """Each field the prompts advertise must survive schema validation."""
    model = schema.model_validate_json(json.dumps(payload, ensure_ascii=False))
    dumped = model.model_dump()
    for key in payload:
        assert key in dumped, f"{schema.__name__} dropped prompt field {key!r}"


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (AnalystOutput, ANALYST_PAYLOAD),
        (AttributionOutput, ATTRIBUTION_PAYLOAD),
        (ModeratorOutput, MODERATOR_PAYLOAD),
        (CriticOutput, CRITIC_PAYLOAD),
    ],
)
def test_extra_fields_are_rejected(schema: type, payload: dict) -> None:
    """extra="forbid": unknown keys are errors, not silent drops."""
    polluted = {**payload, "unexpected_field_from_prompt_drift": 1}
    with pytest.raises(ValidationError):
        schema.model_validate_json(json.dumps(polluted, ensure_ascii=False))


def test_prompt_advertised_fields_exist_in_prompt_text() -> None:
    """The prompts still advertise the fields the schemas now carry."""
    for field in ("top_concerns", "evidence_citations", "patterns", "anomalies"):
        assert field in ANALYST.system_prompt
    for expert in ATTRIBUTION_EXPERTS:
        for field in ("attribution_types", "confidence", "argument", "evidence_citations"):
            assert field in expert.system_prompt
    assert "cognitive_distortions" in ATTRIBUTION_EXPERTS[0].system_prompt
    assert "critique_detail" in CRITIC.system_prompt
    for field in ("types", "confidence", "recommended_technique", "rationale", "dissent"):
        assert field in MODERATOR.system_prompt
    assert "insufficient_data" in MODERATOR.system_prompt
    assert "evidence_gaps" in MODERATOR.system_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Moderator verdict validity
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("empty object", "{}"),
        ("missing rationale", '{"types": ["impulsivity"], "confidence": {"impulsivity": 0.8}}'),
        ("blank rationale", '{"types": ["impulsivity"], "rationale": "   "}'),
        ("no types", '{"types": [], "rationale": "无结论"}'),
        ("unknown type only", '{"types": ["not_a_type"], "rationale": "非法"}'),
        (
            "insufficient_data without gaps",
            '{"insufficient_data": true, "types": [], "rationale": "证据不足"}',
        ),
        (
            "unknown technique",
            '{"types": ["impulsivity"], "recommended_technique": "hypnosis", "rationale": "x"}',
        ),
    ],
)
def test_invalid_moderator_verdicts_are_not_accepted(label: str, payload: str) -> None:
    assert _parse_verdict(payload) is None, label


def test_moderator_abstention_requires_gaps() -> None:
    ok = _parse_verdict(json.dumps({
        "types": [],
        "confidence": {},
        "rationale": "证据不足，暂不裁决",
        "insufficient_data": True,
        "evidence_gaps": ["缺少输入活跃度数据"],
    }, ensure_ascii=False))
    assert ok is not None
    assert ok["insufficient_data"] is True
    assert ok["evidence_gaps"] == ["缺少输入活跃度数据"]


def test_moderator_chinese_aliases_are_canonicalised() -> None:
    verdict = _parse_verdict(json.dumps({
        "types": ["冲动型拖延"],
        "confidence": {"冲动型拖延": 0.7},
        "rationale": "别名归一化",
    }, ensure_ascii=False))
    assert verdict is not None
    assert verdict["types"] == ["impulsivity"]
    assert verdict["confidence"] == {"impulsivity": 0.7}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Expert opinion validity
# ═══════════════════════════════════════════════════════════════════════════════

_EXPERT = ATTRIBUTION_EXPERTS[0]


def _opinion(payload: dict):
    return _parse_expert_opinion(json.dumps(payload, ensure_ascii=False), _EXPERT)


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("empty argument", {
            "attribution_types": ["impulsivity"], "confidence": {"impulsivity": 0.8},
            "argument": "", "evidence_citations": ["focus.switch_rate"],
        }),
        ("blank argument", {
            "attribution_types": ["impulsivity"], "confidence": {"impulsivity": 0.8},
            "argument": "   ", "evidence_citations": ["focus.switch_rate"],
        }),
        ("no legal type", {
            "attribution_types": ["not_a_type"], "confidence": {"not_a_type": 0.8},
            "argument": "有论据", "evidence_citations": ["focus.switch_rate"],
        }),
        ("no citations", {
            "attribution_types": ["impulsivity"], "confidence": {"impulsivity": 0.8},
            "argument": "有论据但没有引用", "evidence_citations": [],
        }),
        ("confidence out of range", {
            "attribution_types": ["impulsivity"], "confidence": {"impulsivity": 1.4},
            "argument": "有论据", "evidence_citations": ["focus.switch_rate"],
        }),
        ("missing confidence", {
            "attribution_types": ["impulsivity"], "confidence": {},
            "argument": "有论据", "evidence_citations": ["focus.switch_rate"],
        }),
    ],
)
def test_semantically_invalid_opinions_are_skipped(label: str, payload: dict) -> None:
    opinion = _opinion(payload)
    assert opinion.skipped is True, label
    assert opinion.argument == ""


def test_valid_opinion_survives_and_drops_alias_types() -> None:
    opinion = _opinion({
        "attribution_types": ["冲动型拖延", "not_a_type"],
        "confidence": {"冲动型拖延": 0.8, "not_a_type": 0.9},
        "argument": "切换频繁 [证据: focus.switch_rate]",
        "evidence_citations": ["focus.switch_rate"],
    })
    assert opinion.skipped is False
    assert opinion.attribution_types == ("impulsivity",)
    assert opinion.confidence == {"impulsivity": 0.8}


def test_semantic_helper_reports_every_issue() -> None:
    issues = validate_opinion_semantics(
        attribution_types=(),
        confidence={"impulsivity": 1.2},
        evidence_citations=(),
        argument="  ",
    )
    assert "论据为空" in issues
    assert "未给出合法拖延类型" in issues
    assert "缺少证据引用" in issues


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Analyst + critic contracts
# ═══════════════════════════════════════════════════════════════════════════════


def test_empty_analyst_report_is_not_an_opinion() -> None:
    for payload in ('{"patterns": [], "anomalies": []}', '{"patterns": []}', "{}"):
        opinion = _parse_analyst_opinion(payload, ANALYST)
        assert opinion.skipped is True


def test_analyst_top_concerns_reach_the_argument() -> None:
    opinion = _parse_analyst_opinion(json.dumps(ANALYST_PAYLOAD, ensure_ascii=False), ANALYST)
    assert opinion.skipped is False
    assert "重点关注" in opinion.argument
    assert "切换频率过高" in opinion.argument


def test_critic_rejection_without_issues_still_carries_a_reason() -> None:
    result = _parse_critic('{"approved": false, "issues": [], "critique_detail": "证据不足"}')
    assert result.approved is False
    assert result.issues == ("证据不足",)


def test_critic_parse_failure_fails_closed() -> None:
    result = _parse_critic("not json at all")
    assert result.approved is False
    assert result.issues


def test_critic_extra_field_is_rejected_but_rejection_is_safe() -> None:
    """Drift fails loudly at the schema, and the panel still fails closed."""
    with pytest.raises(ValidationError):
        CriticOutput.model_validate_json(
            '{"approved": true, "issues": [], "notes": "drift"}'
        )
    assert _parse_critic('{"approved": true, "issues": [], "notes": "drift"}').approved is False
