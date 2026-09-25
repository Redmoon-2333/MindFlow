"""Phase 2.3 regressions: the Claim Ledger contract.

Free-text expert reasoning is unverifiable, uncompressed and easy to pad.  The
ledger replaces it with claims whose evidence ids, confidence, support and
competing explanation are machine-checked *before* anything downstream sees
them, and the moderator receives only the validated table, the evidence table
and the conflict summary.
"""

from __future__ import annotations

import json

import pytest
from test_agents_orchestrator import _make_bundle

from mindflow.agents.claims import (
    ABSTENTION_AGREEMENT_THRESHOLD,
    DISAGREEMENT_CONFIDENCE_PENALTY,
    LEGACY_SUPPORT_MAX_CHARS,
    Claim,
    ClaimLedger,
    apply_disagreement_penalty,
    claim_ledger_for,
    guard_verdict_confidence,
    ledger_from_claims_payload,
    render_claims_argument,
    render_conflict_summary,
    render_ledger_table,
    summarize_conflicts,
    validate_claim,
    validate_ledger,
)
from mindflow.agents.experts import ATTRIBUTION_EXPERTS, CBT, TMT
from mindflow.agents.orchestrator import (
    _build_moderator_claims_prompt,
    _parse_expert_opinion,
)
from mindflow.agents.schemas import AttributionOutput
from mindflow.agents.types import ExpertOpinion
from mindflow.domain.evidence import to_prompt_json
from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids
from mindflow.graph.panel_graph import PanelGraph

_VALID_IDS = tuple(evidence_catalog_ids(build_evidence_catalog(_make_bundle())))
_FIRST_ID = _VALID_IDS[0]


def _claim(**overrides: object) -> Claim:
    payload: dict[str, object] = {
        "type": "impulsivity",
        "confidence": 0.68,
        "evidence_ids": (_FIRST_ID,),
        "support": "频繁切换且最长专注块较短",
        "alternative": "可能是任务需要多应用协作",
    }
    payload.update(overrides)
    return Claim(**payload)  # type: ignore[arg-type]


CLAIM_RESPONSE = json.dumps({
    "claims": [
        {
            "type": "impulsivity",
            "confidence": 0.68,
            "evidence_ids": [_FIRST_ID],
            "support": "频繁切换且最长专注块较短",
            "alternative": "可能是任务需要多应用协作",
        }
    ],
    "insufficient_data": False,
    "evidence_gaps": [],
}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Claim validation
# ═══════════════════════════════════════════════════════════════════════════════


def test_valid_claim_passes_every_check() -> None:
    assert validate_claim(_claim(), _VALID_IDS) == ()


@pytest.mark.parametrize(
    ("label", "claim", "expected"),
    [
        ("no evidence", _claim(evidence_ids=()), "claim 缺少证据: impulsivity"),
        (
            "unknown evidence",
            _claim(evidence_ids=("no.such.metric",)),
            "claim 引用了不存在的证据: no.such.metric",
        ),
        ("blank support", _claim(support="   "), "claim 缺少论据: impulsivity"),
        (
            "no alternative",
            _claim(alternative=""),
            "claim 缺少替代解释: impulsivity",
        ),
    ],
)
def test_invalid_claims_are_reported(label: str, claim: Claim, expected: str) -> None:
    issues = validate_claim(claim, _VALID_IDS)
    assert expected in issues, label


def test_confidence_range_is_enforced_by_validation() -> None:
    assert "claim 置信度越界: impulsivity=1.5" in validate_claim(_claim(confidence=1.5))


def test_ledger_without_claims_is_invalid() -> None:
    assert "未提供任何有效 claim" in validate_ledger(ClaimLedger())


def test_abstention_requires_evidence_gaps() -> None:
    assert "insufficient_data=true 时必须提供 evidence_gaps" in validate_ledger(
        ClaimLedger(insufficient_data=True)
    )
    assert validate_ledger(
        ClaimLedger(insufficient_data=True, evidence_gaps=("缺少输入活跃度",))
    ) == ()


# ═══════════════════════════════════════════════════════════════════════════════
# Schema + parser integration
# ═══════════════════════════════════════════════════════════════════════════════


def test_claim_ledger_response_is_accepted_by_the_schema() -> None:
    parsed = AttributionOutput.model_validate_json(CLAIM_RESPONSE)
    assert parsed.claims[0].type == "impulsivity"
    assert parsed.claims[0].evidence_ids == [_FIRST_ID]


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps({"claims": [{
            "type": "impulsivity", "confidence": 0.6,
            "support": "x", "alternative": "y",
        }]}),
        json.dumps({"claims": [{
            "type": "impulsivity", "confidence": 0.6,
            "evidence_ids": ["m"], "alternative": "y",
        }]}),
        json.dumps({"claims": [{
            "type": "impulsivity", "confidence": 0.6,
            "evidence_ids": ["m"], "support": "x",
        }]}),
        json.dumps({"claims": [], "insufficient_data": True}),
    ],
)
def test_malformed_claim_payloads_are_rejected(payload: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AttributionOutput.model_validate_json(payload)


def test_claim_response_becomes_a_validated_opinion() -> None:
    opinion = _parse_expert_opinion(CLAIM_RESPONSE, CBT, valid_metrics=_VALID_IDS)

    assert opinion.skipped is False
    assert opinion.attribution_types == ("impulsivity",)
    assert opinion.confidence == {"impulsivity": 0.68}
    assert opinion.evidence_citations == (_FIRST_ID,)
    assert len(opinion.claims) == 1
    assert _FIRST_ID in opinion.argument


def test_claim_with_hallucinated_evidence_id_is_skipped() -> None:
    payload = json.dumps({
        "claims": [{
            "type": "impulsivity",
            "confidence": 0.68,
            "evidence_ids": ["hallucinated.metric"],
            "support": "x",
            "alternative": "y",
        }],
        "insufficient_data": False,
        "evidence_gaps": [],
    }, ensure_ascii=False)
    opinion = _parse_expert_opinion(payload, CBT, valid_metrics=_VALID_IDS)

    assert opinion.skipped is True
    assert opinion.claims == ()


def test_abstaining_expert_does_not_count_as_a_valid_opinion() -> None:
    payload = json.dumps({
        "claims": [],
        "insufficient_data": True,
        "evidence_gaps": ["缺少输入活跃度数据"],
    }, ensure_ascii=False)
    opinion = _parse_expert_opinion(payload, TMT, valid_metrics=_VALID_IDS)

    assert opinion.skipped is True


def test_legacy_opinion_is_converted_to_an_equivalent_ledger() -> None:
    legacy = _parse_expert_opinion(
        json.dumps({
            "attribution_types": ["impulsivity"],
            "confidence": {"impulsivity": 0.82},
            "argument": "切换频率高 [证据: focus.switch_rate]",
            "evidence_citations": [_FIRST_ID],
        }, ensure_ascii=False),
        CBT,
        valid_metrics=_VALID_IDS,
    )

    ledger = claim_ledger_for(legacy)
    assert len(ledger.claims) == 1
    assert ledger.claims[0].type == "impulsivity"
    assert ledger.claims[0].evidence_ids == (_FIRST_ID,)
    assert "旧格式" in ledger.claims[0].alternative


def test_legacy_prose_digest_is_bounded_for_the_moderator() -> None:
    """The moderator gets a bounded digest, not the expert's whole essay."""
    long_argument = "切换频率高，专注块短。" * 200
    legacy = _parse_expert_opinion(
        json.dumps({
            "attribution_types": ["impulsivity"],
            "confidence": {"impulsivity": 0.8},
            "argument": long_argument,
            "evidence_citations": [_FIRST_ID],
        }, ensure_ascii=False),
        CBT,
        valid_metrics=_VALID_IDS,
    )
    assert legacy.skipped is False

    support = claim_ledger_for(legacy).claims[0].support
    assert len(support) <= LEGACY_SUPPORT_MAX_CHARS + 1  # + the ellipsis
    assert support.endswith("…")


def test_short_legacy_arguments_are_not_truncated() -> None:
    legacy = _parse_expert_opinion(
        json.dumps({
            "attribution_types": ["impulsivity"],
            "confidence": {"impulsivity": 0.8},
            "argument": "切换频率高 [证据: switch_rate]",
            "evidence_citations": [_FIRST_ID],
        }, ensure_ascii=False),
        CBT,
        valid_metrics=_VALID_IDS,
    )
    support = claim_ledger_for(legacy).claims[0].support
    assert support == "切换频率高 [证据: switch_rate]"


# ═══════════════════════════════════════════════════════════════════════════════
# Conflict summary + disagreement handling
# ═══════════════════════════════════════════════════════════════════════════════


def test_conflicting_claims_are_detected() -> None:
    ledgers = {
        "CBT归因专家": ClaimLedger(claims=(_claim(confidence=0.85),)),
        "TMT归因专家": ClaimLedger(claims=(_claim(confidence=0.4),)),
        "情绪调节归因专家": ClaimLedger(insufficient_data=True, evidence_gaps=("缺失",)),
    }
    summary = summarize_conflicts(ledgers, agreement_strength=0.4)

    assert summary.has_conflict
    conflict = summary.conflicts[0]
    assert conflict.type == "impulsivity"
    assert conflict.contested is True
    assert conflict.supporters == ("CBT归因专家", "TMT归因专家")
    assert "情绪调节归因专家" in summary.abstained_roles


def test_agreement_of_distinct_types_is_not_a_conflict() -> None:
    ledgers = {
        "a": ClaimLedger(claims=(_claim(type="impulsivity", confidence=0.8),)),
        "b": ClaimLedger(claims=(_claim(type="task_aversion", confidence=0.7),)),
    }
    summary = summarize_conflicts(ledgers, agreement_strength=0.9)

    assert summary.has_conflict is False
    assert {c.type for c in summary.conflicts} == {"impulsivity", "task_aversion"}


def test_high_disagreement_lowers_confidence() -> None:
    ledger = ClaimLedger(claims=(_claim(confidence=0.8),))
    penalised = apply_disagreement_penalty(ledger, agreement_strength=0.4)

    assert penalised.insufficient_data is False
    assert penalised.claims[0].confidence == pytest.approx(
        0.8 * DISAGREEMENT_CONFIDENCE_PENALTY
    )


def test_extreme_disagreement_forces_abstention() -> None:
    ledger = ClaimLedger(claims=(_claim(),))
    abstained = apply_disagreement_penalty(
        ledger, agreement_strength=ABSTENTION_AGREEMENT_THRESHOLD - 0.01,
    )

    assert abstained.claims == ()
    assert abstained.insufficient_data is True
    assert abstained.evidence_gaps


def test_strong_agreement_leaves_confidence_untouched() -> None:
    ledger = ClaimLedger(claims=(_claim(confidence=0.8),))
    assert apply_disagreement_penalty(ledger, agreement_strength=0.9) == ledger


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict-level consensus guard
# ═══════════════════════════════════════════════════════════════════════════════


def _verdict(confidence: float = 0.9) -> dict[str, object]:
    return {
        "types": ["impulsivity"],
        "confidence": {"impulsivity": confidence},
        "recommended_technique": "stimulus_control",
        "rationale": "综合意见",
        "dissent": [],
        "insufficient_data": False,
        "evidence_gaps": [],
    }


def test_consensus_guard_leaves_high_agreement_verdicts_alone() -> None:
    verdict = _verdict()
    assert guard_verdict_confidence(verdict, 0.9) == verdict


def test_consensus_guard_lowers_confidence_at_partial_agreement() -> None:
    guarded = guard_verdict_confidence(_verdict(0.9), 0.4)

    assert guarded["confidence"] == {
        "impulsivity": pytest.approx(0.9 * DISAGREEMENT_CONFIDENCE_PENALTY)
    }
    assert guarded["insufficient_data"] is False
    assert any("共识强度偏低" in item for item in guarded["dissent"])  # type: ignore[union-attr]


def test_consensus_guard_forces_abstention_at_extreme_disagreement() -> None:
    guarded = guard_verdict_confidence(_verdict(0.9), 0.2)

    assert guarded["insufficient_data"] is True
    assert guarded["evidence_gaps"]
    assert guarded["confidence"]["impulsivity"] <= 0.2 + 1e-9  # type: ignore[index]


def test_consensus_guard_does_not_mutate_its_input() -> None:
    verdict = _verdict(0.9)
    before = json.loads(json.dumps(verdict))
    guard_verdict_confidence(verdict, 0.2)
    assert verdict == before


# ═══════════════════════════════════════════════════════════════════════════════
# Moderator prompt wiring
# ═══════════════════════════════════════════════════════════════════════════════


async def test_panel_moderator_receives_the_validated_ledger() -> None:
    """The live graph hands the moderator the ledger table, not raw prose."""
    from test_agents_orchestrator import (
        _ANALYST_JSON,
        _ATTRIBUTION_IMPULSIVITY,
        _CRITIC_APPROVE,
        _MODERATOR_JSON,
        FP_ANALYST,
        FP_CBT,
        FP_CRITIC,
        FP_EMOTION,
        FP_MODERATOR,
        FP_TMT,
        MockGateway,
        _make_bundle,
    )

    from mindflow.domain.evidence import to_prompt_json
    from mindflow.domain.evidence_facts import build_evidence_catalog, evidence_catalog_ids
    from mindflow.graph.panel_graph import PanelGraph, PanelGraphState

    class _RecordingModeratorGateway(MockGateway):
        def __init__(self, responses: dict[str, list[str]]) -> None:
            super().__init__(responses=responses)
            self.moderator_prompts: list[str] = []

        async def complete(self, system: str, user: str, model: str = "chat", **kw: object) -> str:
            if FP_MODERATOR in system:
                self.moderator_prompts.append(user)
            return await super().complete(system=system, user=user, model=model)

    gateway = _RecordingModeratorGateway(responses={
        FP_ANALYST: [_ANALYST_JSON],
        FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
        FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
        FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
        FP_MODERATOR: [_MODERATOR_JSON],
        FP_CRITIC: [_CRITIC_APPROVE],
    })
    bundle = _make_bundle()
    state: PanelGraphState = {  # type: ignore[typeddict-item]
        "bundle_json": to_prompt_json(bundle),
        "valid_metrics": tuple(evidence_catalog_ids(build_evidence_catalog(bundle))),
        "attribution_opinions": (),
        "transcript": (),
        "analyst_opinion": None,
        "conflict_report": None,
        "escalated": False,
        "moderator_verdict": None,
        "critic_result": None,
        "critic_retries": 0,
        "moderator_redo_count": 0,
        "call_count": 0,
        "disagreement_summary": None,
        "rebuttal_delta": None,
        "_expert_index": 0,
    }

    result = await PanelGraph(gateway=gateway).ainvoke(state)

    assert result["panel_terminal"] == "approved"
    assert gateway.moderator_prompts, "moderator was never called"
    prompt = gateway.moderator_prompts[0]
    assert "已校验的 Claim Ledger" in prompt
    assert "跨专家冲突摘要" in prompt
    assert _FIRST_ID in prompt
    assert "insufficient_data=true" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Moderator prompt
# ═══════════════════════════════════════════════════════════════════════════════


def _opinion(role: str, claims: tuple[Claim, ...]) -> ExpertOpinion:
    return ExpertOpinion(
        role=role,
        perspective="测试视角",
        attribution_types=tuple(dict.fromkeys(c.type for c in claims)),
        confidence={c.type: c.confidence for c in claims},
        evidence_citations=tuple(e for c in claims for e in c.evidence_ids),
        argument=render_claims_argument(ClaimLedger(claims=claims)),
        claims=claims,
    )


def test_moderator_prompt_carries_validated_claims_not_raw_prose() -> None:
    from mindflow.agents.conflict import detect_conflict

    opinions = [
        _opinion("CBT归因专家", (_claim(confidence=0.8),)),
        _opinion("TMT归因专家", (_claim(confidence=0.45),)),
    ]
    analyst = ExpertOpinion(
        role="数据分析师",
        perspective="行为模式分析视角",
        attribution_types=(),
        confidence={},
        evidence_citations=(),
        argument="[moderate] 专注度低于基线\n异常-focus.switch_rate: 切换偏高",
    )
    conflict = detect_conflict(list(opinions))

    prompt = _build_moderator_claims_prompt(
        to_prompt_json(_make_bundle()),
        analyst,
        opinions,
        conflict,
        agreement_strength=0.4,
    )

    assert "已校验的 Claim Ledger" in prompt
    assert "跨专家冲突摘要" in prompt
    assert "analyst" not in prompt.lower() or "数据分析师要点" in prompt
    # The claim rows are present with their evidence ids.
    assert _FIRST_ID in prompt
    assert "impulsivity" in prompt
    # The moderator is told how to abstain instead of guessing.
    assert "insufficient_data=true" in prompt


def test_ledger_table_marks_abstaining_experts() -> None:
    table = render_ledger_table({
        "CBT归因专家": ClaimLedger(claims=(_claim(),)),
        "TMT归因专家": ClaimLedger(insufficient_data=True, evidence_gaps=("缺少数据",)),
    })
    assert "（弃权）" in table
    assert "缺少数据" in table


def test_conflict_summary_renders_the_penalty_guidance() -> None:
    text = render_conflict_summary(
        summarize_conflicts({"a": ClaimLedger(claims=(_claim(),))}, agreement_strength=0.2)
    )
    assert "分歧较大" in text


def test_every_attribution_prompt_advertises_the_ledger() -> None:
    for expert in ATTRIBUTION_EXPERTS:
        assert "claims" in expert.system_prompt
        assert "evidence_ids" in expert.system_prompt
        assert "alternative" in expert.system_prompt


def test_ledger_helper_is_importable_from_the_graph_layer() -> None:
    """The graph layer consumes ledgers through the same public helpers."""
    assert PanelGraph is not None
    ledger = ledger_from_claims_payload([
        {
            "type": "impulsivity",
            "confidence": 0.5,
            "evidence_ids": [_FIRST_ID],
            "support": "s",
            "alternative": "a",
        }
    ])
    assert ledger.claims[0].evidence_ids == (_FIRST_ID,)
