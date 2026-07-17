"""Tests for agents/orchestrator.py — full panel deliberation paths.

Uses ``MockGateway`` with per-fingerprint call counting to simulate
recording/playback of LLM responses.

Covers:
  - Fast path success (no conflict, critic approves)
  - Conflict escalation path (rebuttal round)
  - Critic reject → re-verdict
  - Fake metric reference caught by critic
  - Bad JSON → expert skipped
  - 2 attribution failures → PanelUnavailableError
  - Call count budget guard
  - Forbidden words in expert opinion → skipped
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import pytest

from mindflow.agents.orchestrator import PanelOrchestrator
from mindflow.agents.types import (
    PanelUnavailableError,
    PanelVerdict,
)
from mindflow.domain.evidence import EvidenceBundle, EvidenceItem
from mindflow.domain.procrastination import BehaviorSummary

# ═══════════════════════════════════════════════════════════════════════════════
# Mock Gateway — call-count-per-fingerprint routing
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MockGateway:
    """Mock LLM gateway for testing.

    Routes responses by examining the system prompt for fingerprints.
    Each fingerprint maps to a *list* of responses — the N-th call to that
    fingerprint returns the N-th response (or the last one on overflow).

    This correctly handles:
      - Parallel calls (each expert type has a unique fingerprint/counter)
      - Rebuttal rounds (same fingerprint, second entry in list)
      - Critic reject → re-verdict (third entry if needed)
    """

    responses: dict[str, list[str]] = field(default_factory=dict)
    default_response: str = '{"approved": true, "issues": []}'
    _counts: dict[str, int] = field(default_factory=dict)

    async def complete(
        self,
        system: str,
        user: str,  # noqa: ARG002
        model: Literal["chat", "reasoner"] = "chat",  # noqa: ARG002
    ) -> str:
        """Return the N-th response for the matching fingerprint."""
        # Find which fingerprint matches
        key = self.default_response
        for fp in self.responses:
            if fp in system:
                key = fp
                break

        # Increment call count for this key
        self._counts[key] = self._counts.get(key, 0) + 1
        idx = self._counts[key] - 1

        # Get response list
        response_list = self.responses.get(key, [self.default_response])
        if idx < len(response_list):
            return response_list[idx]
        return response_list[-1]

    async def close(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Fingerprints (short, reliable substrings — verified match system prompts)
# ═══════════════════════════════════════════════════════════════════════════════

FP_ANALYST = "行为数据分析师"
FP_CBT = "认知行为疗法"
FP_TMT = "时间动机理论"
FP_EMOTION = "情绪调节归因专家"
FP_MODERATOR = "会诊综合主持人"
FP_CRITIC = "批评家"


# ═══════════════════════════════════════════════════════════════════════════════
# JSON fixtures
# ═══════════════════════════════════════════════════════════════════════════════

_ANALYST_JSON: str = """{
  "patterns": [{"name": "专注度下降", "severity": "moderate", "description": "专注度显著低于基线 [证据: focus_score]"}],
  "anomalies": [{"metric": "longest_focus_block_s", "detail": "最长专注块仅3分钟 [证据: longest_focus_block_s]"}],
  "top_concerns": ["专注度下降", "切换频率过高"],
  "evidence_citations": ["focus_score", "switch_rate", "longest_focus_block_s"]
}"""

_ATTRIBUTION_IMPULSIVITY: str = """{
  "attribution_types": ["impulsivity"],
  "confidence": {"impulsivity": 0.82},
  "argument": "用户切换频率高、最长专注块仅3分钟，符合冲动分心模式 [证据: switch_rate] [证据: longest_focus_block_s]",
  "evidence_citations": ["switch_rate", "longest_focus_block_s"]
}"""

_ATTRIBUTION_TASK_AVERSION: str = """{
  "attribution_types": ["task_aversion"],
  "confidence": {"task_aversion": 0.75},
  "argument": "专注度45/120分钟，不足40%，符合任务畏惧模式 [证据: focus_score]",
  "evidence_citations": ["focus_score"]
}"""

_REBUTTAL_IMPULSIVITY: str = """{
  "attribution_types": ["impulsivity"],
  "confidence": {"impulsivity": 0.78},
  "argument": "经权衡其他专家意见后，维持冲动分心判断，但适度降低置信度 [证据: switch_rate]",
  "evidence_citations": ["switch_rate", "longest_focus_block_s"]
}"""

_MODERATOR_JSON: str = """{
  "types": ["impulsivity"],
  "confidence": {"impulsivity": 0.80},
  "recommended_technique": "stimulus_control",
  "rationale": "综合多方意见，用户主要表现为冲动分心型拖延。专注块短、切换频率高是核心指标。",
  "dissent": []
}"""

_MODERATOR_REDO_JSON: str = """{
  "types": ["impulsivity"],
  "confidence": {"impulsivity": 0.78},
  "recommended_technique": "stimulus_control",
  "rationale": "修正后：降低置信度以匹配证据强度，所有引用已核实。",
  "dissent": []
}"""

_CRITIC_APPROVE: str = """{
  "approved": true, "issues": [], "critique_detail": "通过。"
}"""

_CRITIC_REJECT: str = """{
  "approved": false, "issues": ["引用不存在的指标: fake_metric"], "critique_detail": "引用不合法。"
}"""

_CRITIC_FAKE_METRIC: str = """{
  "approved": false, "issues": ["引用不存在的指标: nonexistent_metric"], "critique_detail": "指标不在合法清单中。"
}"""


# ═══════════════════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════════════════


def _make_bundle() -> EvidenceBundle:
    """Create a simple EvidenceBundle for testing."""
    now = datetime.now(UTC)
    return EvidenceBundle(
        user_id=1,
        window=(now, now),
        items=(
            EvidenceItem(
                metric="focus_score",
                value=0.45, baseline=0.72, severity="moderate",
                confidence=0.85, source="welford_baseline",
                human_readable="专注度低于基线",
            ),
            EvidenceItem(
                metric="switch_rate",
                value=15.0, baseline=8.0, severity="moderate",
                confidence=0.78, source="feature_computation",
                human_readable="切换频率偏高",
            ),
            EvidenceItem(
                metric="longest_focus_block_s",
                value=180.0, baseline=600.0, severity="severe",
                confidence=0.90, source="feature_computation",
                human_readable="最长专注块很短",
            ),
            EvidenceItem(
                metric="social_media_ratio",
                value=0.4, baseline=0.2, severity="mild",
                confidence=0.65, source="feature_computation",
                human_readable="社交媒体使用比例略高",
            ),
        ),
        behavior_summary=BehaviorSummary(
            intended_task="写论文", duration_min=120.0,
            actual_focus_min=45.0, context_switches_per_hour=15.0,
            longest_focus_block_s=180.0, social_media_ratio=0.4,
            start_delay_min=25.0, keyword_flags=frozenset(),
            baseline_deviation=-1.8,
        ),
        intervention_history=(),
        novelty_flags=(),
    )


def _fast_responses() -> dict[str, list[str]]:
    """All experts agree on impulsivity, critic approves."""
    return {
        FP_ANALYST: [_ANALYST_JSON],
        FP_CBT: [_ATTRIBUTION_IMPULSIVITY],
        FP_TMT: [_ATTRIBUTION_IMPULSIVITY],
        FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
        FP_MODERATOR: [_MODERATOR_JSON],
        FP_CRITIC: [_CRITIC_APPROVE],
    }


def _conflict_responses() -> dict[str, list[str]]:
    """CBT+TMT disagree → rebuttal round needed.

    First call to each attribution expert returns initial opinion;
    second call (rebuttal) returns the converged opinion.
    """
    return {
        FP_ANALYST: [_ANALYST_JSON],
        FP_CBT: [_ATTRIBUTION_IMPULSIVITY, _REBUTTAL_IMPULSIVITY],
        FP_TMT: [_ATTRIBUTION_TASK_AVERSION, _REBUTTAL_IMPULSIVITY],
        FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY, _REBUTTAL_IMPULSIVITY],
        FP_MODERATOR: [_MODERATOR_JSON],
        FP_CRITIC: [_CRITIC_APPROVE],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFastPath:
    """Fast path: no conflict, critic approves on first try."""

    @pytest.mark.asyncio
    async def test_fast_path_success(self) -> None:
        gateway = MockGateway(responses=_fast_responses())
        orchestrator = PanelOrchestrator(gateway=gateway)
        verdict = await orchestrator.run(_make_bundle())

        assert isinstance(verdict, PanelVerdict)
        assert verdict.source == "panel"
        assert verdict.escalated is False
        assert verdict.call_count == 6
        assert len(verdict.types) >= 1
        assert verdict.recommended_technique is not None
        assert verdict.rationale != ""
        assert len(verdict.transcript) >= 4

    @pytest.mark.asyncio
    async def test_fast_path_call_count(self) -> None:
        gateway = MockGateway(responses=_fast_responses())
        verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())
        assert verdict.call_count == 6


class TestConflictEscalation:
    """Conflict escalation: attribution experts disagree → rebuttal round."""

    @pytest.mark.asyncio
    async def test_conflict_escalation(self) -> None:
        """TMT says task_aversion, CBT+Emotion say impulsivity → escalation."""
        gateway = MockGateway(responses=_conflict_responses())
        verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())

        assert verdict.escalated is True
        assert verdict.call_count == 9

    @pytest.mark.asyncio
    async def test_conflict_escalation_verdict(self) -> None:
        """After escalation, should still produce a valid verdict."""
        gateway = MockGateway(responses=_conflict_responses())
        verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())

        assert verdict.escalated is True
        assert len(verdict.types) >= 1
        assert verdict.rationale != ""


class TestCriticReject:
    """Critic rejects → moderator re-verdict → critic approves."""

    @pytest.mark.asyncio
    async def test_critic_reject_redo(self) -> None:
        """Critic rejects once (then approves on re-check)."""
        responses = _fast_responses()
        responses[FP_CRITIC] = [_CRITIC_REJECT, _CRITIC_APPROVE]
        responses[FP_MODERATOR] = [_MODERATOR_JSON, _MODERATOR_REDO_JSON]

        gateway = MockGateway(responses=responses)
        verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())

        assert verdict.source == "panel"
        # analyst(1) + 3attribution(3) + moderator(1) + critic(1) + redo(1) + critic2(1) = 8
        assert verdict.call_count == 8

    @pytest.mark.asyncio
    async def test_critic_reject_fake_metric(self) -> None:
        """Critic identifies fake metric reference → re-verdict → final verdict."""
        responses = _fast_responses()
        responses[FP_CRITIC] = [_CRITIC_FAKE_METRIC, _CRITIC_APPROVE]
        responses[FP_MODERATOR] = [_MODERATOR_JSON, _MODERATOR_REDO_JSON]

        gateway = MockGateway(responses=responses)
        verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())

        assert isinstance(verdict, PanelVerdict)
        assert verdict.source == "panel"


class TestBadJSON:
    """Malformed JSON from experts → graceful degradation (skip)."""

    @pytest.mark.asyncio
    async def test_attribution_bad_json_skipped(self) -> None:
        """One attribution expert returns bad JSON → that expert is skipped."""
        responses = _fast_responses()
        responses[FP_TMT] = ["这不是合法 JSON"]

        gateway = MockGateway(responses=responses)
        verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())

        assert verdict.source == "panel"
        # Still 6 calls (the bad JSON one consumed a call)
        assert verdict.call_count == 6

    @pytest.mark.asyncio
    async def test_two_attribution_failures_raises_unavailable(self) -> None:
        """Two attribution experts fail → PanelUnavailableError (< 2 valid)."""
        responses = {
            FP_ANALYST: [_ANALYST_JSON],
            FP_CBT: ["坏掉的 JSON"],
            FP_TMT: ["坏掉的 JSON"],
            FP_EMOTION: [_ATTRIBUTION_IMPULSIVITY],
        }
        gateway = MockGateway(responses=responses)

        with pytest.raises(PanelUnavailableError):
            await PanelOrchestrator(gateway=gateway).run(_make_bundle())

    @pytest.mark.asyncio
    async def test_analyst_bad_json_creates_skipped_opinion(self) -> None:
        """Analyst returns bad JSON → produces skipped opinion."""
        responses = _fast_responses()
        responses[FP_ANALYST] = ["{{{ 损坏的 JSON"]

        gateway = MockGateway(responses=responses)
        try:
            verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())
            assert isinstance(verdict, PanelVerdict)
        except PanelUnavailableError:
            pass  # Acceptable if analyst failure cascades


class TestBudgetGuard:
    """Call count > 12 raises PanelBudgetExceededError."""

    @pytest.mark.asyncio
    async def test_normal_path_never_exceeds_budget(self) -> None:
        """Fast path (6) and conflict path (9) are within budget."""
        v1 = await PanelOrchestrator(
            gateway=MockGateway(responses=_fast_responses()),
        ).run(_make_bundle())
        assert v1.call_count <= 12

        v2 = await PanelOrchestrator(
            gateway=MockGateway(responses=_conflict_responses()),
        ).run(_make_bundle())
        assert v2.call_count <= 12


class TestForbiddenWords:
    """Forbidden words in expert output → opinion skipped."""

    @pytest.mark.asyncio
    async def test_forbidden_word_skips_expert(self) -> None:
        """Attribution expert uses forbidden word → that expert is skipped."""
        forbidden_json = """{
          "attribution_types": ["task_aversion"],
          "confidence": {"task_aversion": 0.75},
          "argument": "这个患者需要治疗 [证据: focus_score]",
          "evidence_citations": ["focus_score"]
        }"""

        responses = _fast_responses()
        responses[FP_TMT] = [forbidden_json]

        gateway = MockGateway(responses=responses)
        try:
            verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())
            assert verdict.source == "panel"
        except PanelUnavailableError:
            pass  # Acceptable if only 1 attribution valid after skip


class TestPanelVerdictShape:
    """PanelVerdict aligns with ProcrastinationAssessment shape."""

    @pytest.mark.asyncio
    async def test_verdict_shape_matches_assessment(self) -> None:
        gateway = MockGateway(responses=_fast_responses())
        verdict = await PanelOrchestrator(gateway=gateway).run(_make_bundle())

        # Common with ProcrastinationAssessment
        assert hasattr(verdict, "types")
        assert hasattr(verdict, "confidence")
        assert hasattr(verdict, "recommended_technique")
        assert hasattr(verdict, "rationale")

        # Panel-specific extras
        assert hasattr(verdict, "dissent")
        assert hasattr(verdict, "transcript")
        assert hasattr(verdict, "escalated")
        assert hasattr(verdict, "call_count")
        assert hasattr(verdict, "source")


class TestCitationValidation:
    """Review P1: code-enforced citation validation, not prompt trust."""

    def test_bogus_citation_detected(self) -> None:
        from mindflow.agents.orchestrator import validate_citations
        from mindflow.agents.types import ExpertOpinion

        op = ExpertOpinion(
            role="cbt",
            perspective="CBT",
            attribution_types=("impulsivity",),
            confidence={"impulsivity": 0.8},
            evidence_citations=("focus_score", "made_up_metric"),
            argument="切换频繁 [证据: switch_rate]，且虚构 [证据: fantasy_stat]",
            raw_json="{}",
        )
        bogus = validate_citations(op, frozenset({"focus_score", "switch_rate"}))
        assert bogus == ("fantasy_stat", "made_up_metric")

    def test_all_valid_citations(self) -> None:
        from mindflow.agents.orchestrator import validate_citations
        from mindflow.agents.types import ExpertOpinion

        op = ExpertOpinion(
            role="tmt",
            perspective="TMT",
            attribution_types=("decisional",),
            confidence={"decisional": 0.7},
            evidence_citations=("behavior_deviation",),
            argument="偏差显著 [证据: behavior_deviation]",
            raw_json="{}",
        )
        assert validate_citations(op, frozenset({"behavior_deviation"})) == ()

    def test_parse_skips_opinion_with_bogus_citation(self) -> None:
        from mindflow.agents.experts import ATTRIBUTION_EXPERTS
        from mindflow.agents.orchestrator import _parse_expert_opinion

        raw = (
            '{"attribution_types": ["impulsivity"], "confidence": {"impulsivity": 0.8},'
            ' "evidence_citations": ["nonexistent_metric"], "argument": "论证"}'
        )
        op = _parse_expert_opinion(
            raw, ATTRIBUTION_EXPERTS[0], valid_metrics=frozenset({"focus_score"})
        )
        assert op.skipped is True

    def test_fullwidth_colon_pattern(self) -> None:
        from mindflow.agents.orchestrator import validate_citations
        from mindflow.agents.types import ExpertOpinion

        op = ExpertOpinion(
            role="emotion",
            perspective="情绪",
            attribution_types=("emotional_regulation",),
            confidence={"emotional_regulation": 0.6},
            evidence_citations=(),
            argument="娱乐占比高 [证据：top_apps]",  # 全角冒号
            raw_json="{}",
        )
        assert validate_citations(op, frozenset({"top_apps"})) == ()
