"""Tests for agents/types.py — ExpertOpinion, PanelVerdict, TranscriptEntry, exceptions."""

from __future__ import annotations

from mindflow.agents.types import (
    FORBIDDEN_WORDS,
    CriticResult,
    ExpertOpinion,
    PanelBudgetExceededError,
    PanelUnavailableError,
    PanelVerdict,
    TranscriptEntry,
)
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType


class TestTranscriptEntry:
    """TranscriptEntry frozen dataclass."""

    def test_creates_with_required_fields(self) -> None:
        entry = TranscriptEntry(role="数据分析师", content="摘要", round=0)
        assert entry.role == "数据分析师"
        assert entry.content == "摘要"
        assert entry.round == 0

    def test_is_frozen(self) -> None:
        entry = TranscriptEntry(role="批评家", content="通过", round=3)
        try:
            entry.role = "主持人"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except Exception:
            pass


class TestExpertOpinion:
    """ExpertOpinion frozen dataclass."""

    def test_creates_with_minimal_fields(self) -> None:
        opinion = ExpertOpinion(
            role="CBT归因专家",
            perspective="认知行为理论视角",
            attribution_types=("impulsivity",),
            confidence={"impulsivity": 0.82},
            evidence_citations=("switch_rate", "longest_focus_block_s"),
            argument="用户切换频率高，专注块短，符合冲动分心模式",
        )
        assert opinion.role == "CBT归因专家"
        assert opinion.attribution_types == ("impulsivity",)
        assert opinion.confidence["impulsivity"] == 0.82
        assert len(opinion.evidence_citations) == 2
        assert opinion.skipped is False
        assert opinion.raw_json is None

    def test_skipped_default_is_false(self) -> None:
        opinion = ExpertOpinion(
            role="测试",
            perspective="测试视角",
            attribution_types=(),
            confidence={},
            evidence_citations=(),
            argument="",
        )
        assert opinion.skipped is False

    def test_explicit_skipped(self) -> None:
        opinion = ExpertOpinion(
            role="测试",
            perspective="测试视角",
            attribution_types=(),
            confidence={},
            evidence_citations=(),
            argument="",
            skipped=True,
        )
        assert opinion.skipped is True


class TestPanelVerdict:
    """PanelVerdict frozen dataclass — aligned with ProcrastinationAssessment."""

    def test_creates_with_all_fields(self) -> None:
        transcript = (
            TranscriptEntry(role="分析师", content="分析摘要", round=0),
            TranscriptEntry(role="主持人", content="裁决摘要", round=2),
        )
        verdict = PanelVerdict(
            types=(ProcrastinationType.IMPULSIVITY,),
            confidence={ProcrastinationType.IMPULSIVITY: 0.82},
            recommended_technique=CBTTechnique.STIMULUS_CONTROL,
            rationale="用户表现出冲动分心模式，建议刺激控制技术",
            dissent=("TMT专家认为更接近情绪调节型",),
            transcript=transcript,
            escalated=False,
            call_count=6,
            source="panel",
        )

        assert len(verdict.types) == 1
        assert verdict.types[0] == ProcrastinationType.IMPULSIVITY
        assert verdict.recommended_technique == CBTTechnique.STIMULUS_CONTROL
        assert verdict.source == "panel"
        assert verdict.escalated is False
        assert verdict.call_count == 6
        assert len(verdict.transcript) == 2

    def test_empty_dissent(self) -> None:
        verdict = PanelVerdict(
            types=(ProcrastinationType.TASK_AVERSION,),
            confidence={ProcrastinationType.TASK_AVERSION: 0.7},
            recommended_technique=CBTTechnique.GRADED_EXPOSURE,
            rationale="测试",
            dissent=(),
            transcript=(),
            escalated=False,
            call_count=6,
            source="panel",
        )
        assert verdict.dissent == ()


class TestPanelBudgetExceededError:
    """PanelBudgetExceededError runtime error."""

    def test_default_message(self) -> None:
        err = PanelBudgetExceededError()
        assert "12" in str(err)
        assert err.call_count == 0

    def test_with_call_count(self) -> None:
        err = PanelBudgetExceededError(call_count=13)
        assert err.call_count == 13
        assert "13" in str(err)


class TestPanelUnavailableError:
    """PanelUnavailableError runtime error."""

    def test_with_reason(self) -> None:
        err = PanelUnavailableError(reason="主持人解析失败", call_count=4)
        assert "主持人解析失败" in str(err)
        assert err.reason == "主持人解析失败"
        assert err.call_count == 4

    def test_default_call_count(self) -> None:
        err = PanelUnavailableError(reason="测试")
        assert err.call_count == 0


class TestCriticResult:
    """CriticResult frozen dataclass."""

    def test_approved(self) -> None:
        result = CriticResult(approved=True, issues=())
        assert result.approved
        assert result.issues == ()

    def test_rejected_with_issues(self) -> None:
        result = CriticResult(approved=False, issues=("引用不存在指标", "禁词检测失败"))
        assert not result.approved
        assert len(result.issues) == 2


class TestForbiddenWords:
    """FORBIDDEN_WORDS constant (NF-S7)."""

    def test_contains_medical_terms(self) -> None:
        assert "诊断" in FORBIDDEN_WORDS
        assert "治疗" in FORBIDDEN_WORDS
        assert "患者" in FORBIDDEN_WORDS
        assert "处方" in FORBIDDEN_WORDS
        assert len(FORBIDDEN_WORDS) == 4
