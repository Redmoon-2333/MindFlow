"""Tests for agents/orchestrator.py — module-level panel deliberation helpers.

The legacy ``PanelOrchestrator`` class was removed when the v2 ``PanelGraph``
became the only active panel path.  This file now exercises the module-level
parsing / citation-validation helpers that ``PanelGraph`` depends on.

Also hosts the shared ``MockGateway`` and JSON fixtures used by
``test_langgraph_orchestrator`` (which drives the full PanelGraph).

Covers:
  - Citation validation (validate_citations) — bogus vs valid references
  - Expert opinion parsing skips opinions with hallucinated citations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

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
  "patterns": [{"name": "专注度下降", "severity": "moderate", "description": "专注度显著低于基线 [证据: focus.focus_score]"}],
  "anomalies": [{"metric": "focus.longest_block", "detail": "最长专注块仅3分钟 [证据: focus.longest_block]"}],
  "top_concerns": ["专注度下降", "切换频率过高"],
  "evidence_citations": ["focus.focus_score", "focus.switch_rate", "focus.longest_block"]
}"""

_ATTRIBUTION_IMPULSIVITY: str = """{
  "attribution_types": ["impulsivity"],
  "confidence": {"impulsivity": 0.82},
  "argument": "用户切换频率高、最长专注块仅3分钟，符合冲动分心模式 [证据: focus.switch_rate] [证据: focus.longest_block]",
  "evidence_citations": ["focus.switch_rate", "focus.longest_block"]
}"""

_ATTRIBUTION_TASK_AVERSION: str = """{
  "attribution_types": ["task_aversion"],
  "confidence": {"task_aversion": 0.75},
  "argument": "专注度45/120分钟，不足40%，符合任务畏惧模式 [证据: focus.focus_score]",
  "evidence_citations": ["focus.focus_score"]
}"""

_REBUTTAL_IMPULSIVITY: str = """{
  "attribution_types": ["impulsivity"],
  "confidence": {"impulsivity": 0.78},
  "argument": "经权衡其他专家意见后，维持冲动分心判断，但适度降低置信度 [证据: focus.switch_rate]",
  "evidence_citations": ["focus.switch_rate", "focus.longest_block"]
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
                metric="longest_block",
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
# Citation validation — review P1: code-enforced citation validation, not prompt trust
# ═══════════════════════════════════════════════════════════════════════════════


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
            evidence_citations=("focus.focus_score", "made_up_metric"),
            argument="切换频繁 [证据: focus.switch_rate]，且虚构 [证据: fantasy_stat]",
            raw_json="{}",
        )
        bogus = validate_citations(op, frozenset({"focus.focus_score", "focus.switch_rate"}))
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
            raw, ATTRIBUTION_EXPERTS[0], valid_metrics=frozenset({"focus.focus_score"})
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
