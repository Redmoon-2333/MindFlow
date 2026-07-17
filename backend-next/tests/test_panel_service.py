"""Tests for PanelService (services/panel_service.py).

Covers:
  - Normal panel path: EvidenceBundleBuilder → PanelOrchestrator → PanelVerdict
  - Degradation to single-expert (PanelUnavailableError from orchestrator)
  - Degradation to single-expert (PanelBudgetExceededError from orchestrator)
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import AsyncMock

import pytest

from mindflow.agents.types import (
    PanelBudgetExceededError,
    PanelUnavailableError,
    PanelVerdict,
    TranscriptEntry,
)
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType
from mindflow.services.panel_service import PanelService


@pytest.fixture
def mock_orchestrator() -> AsyncMock:
    """Create a mock PanelOrchestrator."""
    return AsyncMock()


@pytest.fixture
def mock_llm_service() -> AsyncMock:
    """Create a mock LLMService."""
    service = AsyncMock()
    # Default: successful outcome
    service.analyze.return_value = type(
        "Outcome",
        (),
        {
            "assessment": {
                "procrastination_types": ["impulsivity"],
                "type_confidence": {"impulsivity": 0.72},
                "cbt_technique": "stimulus_control",
                "response_text": "单专家分析结果（降级模式）",
            },
            "source": "deepseek",
            "cached": False,
            "degraded": True,
        },
    )()
    return service


@pytest.fixture
def mock_builder() -> AsyncMock:
    """Create a mock EvidenceBundleBuilder."""
    builder = AsyncMock()
    bundle = type(
        "Bundle",
        (),
        {
            "user_id": 1,
            "window": (datetime(2026, 7, 18, 0, 0), datetime(2026, 7, 19, 0, 0)),
            "items": (),
            "behavior_summary": None,
            "intervention_history": (),
            "novelty_flags": (),
        },
    )()
    builder.build.return_value = bundle
    return builder


@pytest.fixture
def panel_service(
    mock_builder: AsyncMock,
    mock_orchestrator: AsyncMock,
    mock_llm_service: AsyncMock,
) -> PanelService:
    """Create a PanelService with all mocks."""
    service = PanelService.__new__(PanelService)
    service._builder = mock_builder
    service._orchestrator = mock_orchestrator
    service._llm_service = mock_llm_service
    return service


def _make_verdict(**overrides: object) -> PanelVerdict:
    """Build a sample PanelVerdict."""
    defaults: dict[str, object] = {
        "types": (ProcrastinationType.IMPULSIVITY,),
        "confidence": {ProcrastinationType.IMPULSIVITY: 0.85},
        "recommended_technique": CBTTechnique.STIMULUS_CONTROL,
        "rationale": "测试会诊结果",
        "dissent": (),
        "transcript": (
            TranscriptEntry(role="数据分析师", content="模式分析完成", round=0),
            TranscriptEntry(role="CBT归因专家", content="归因完成", round=1),
            TranscriptEntry(role="综合主持人", content="裁决完成", round=3),
        ),
        "escalated": False,
        "call_count": 6,
        "source": "panel",
    }
    defaults.update(overrides)
    return PanelVerdict(**defaults)  # type: ignore[arg-type]


class TestPanelServiceNormal:
    """Normal panel flow — orchestrator succeeds."""

    async def test_panel_success(self, panel_service: PanelService) -> None:
        """Successful panel returns verdict with source='panel'."""
        panel_service._orchestrator.run = AsyncMock(return_value=_make_verdict())
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.source == "panel"
        assert verdict.call_count == 6
        assert verdict.types == (ProcrastinationType.IMPULSIVITY,)

    async def test_panel_escalated(self, panel_service: PanelService) -> None:
        """Panel with conflict escalation returns escalated=True."""
        panel_service._orchestrator.run = AsyncMock(
            return_value=_make_verdict(escalated=True, call_count=9),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.source == "panel"
        assert verdict.escalated is True
        assert verdict.call_count == 9

    async def test_panel_with_dissent(self, panel_service: PanelService) -> None:
        """Panel with recorded dissent."""
        panel_service._orchestrator.run = AsyncMock(
            return_value=_make_verdict(
                dissent=("TMT专家认为情绪调节是主要因素",),
            ),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert len(verdict.dissent) == 1
        assert "TMT" in verdict.dissent[0]

    async def test_panel_transcript_present(self, panel_service: PanelService) -> None:
        """Panel transcript contains expert entries."""
        panel_service._orchestrator.run = AsyncMock(return_value=_make_verdict())
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert len(verdict.transcript) == 3


class TestPanelServiceDegradation:
    """Degradation to single-expert LLM service."""

    async def test_panel_unavailable_fallback(self, panel_service: PanelService) -> None:
        """PanelUnavailableError triggers single-expert fallback."""
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelUnavailableError(reason="仅1份有效归因", call_count=4),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.source == "single_expert"
        assert verdict.types == (ProcrastinationType.IMPULSIVITY,)
        assert verdict.call_count == 0
        # Verify LLM service was called
        panel_service._llm_service.analyze.assert_awaited_once()

    async def test_panel_budget_exceeded_fallback(self, panel_service: PanelService) -> None:
        """PanelBudgetExceededError triggers single-expert fallback."""
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelBudgetExceededError(call_count=12),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.source == "single_expert"
        assert verdict.call_count == 0
        panel_service._llm_service.analyze.assert_awaited_once()

    async def test_degraded_verdict_has_rationale(self, panel_service: PanelService) -> None:
        """Degraded verdict has the fallback LLM's rationale."""
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelUnavailableError(reason="专家解析失败"),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.rationale == "单专家分析结果（降级模式）"

    async def test_degraded_no_transcript(self, panel_service: PanelService) -> None:
        """Degraded verdict has no transcript entries."""
        panel_service._orchestrator.run = AsyncMock(
            side_effect=PanelUnavailableError(reason="专家解析失败"),
        )
        verdict = await panel_service.run_daily_panel(user_id=1, target_date=date(2026, 7, 18))
        assert verdict.transcript == ()
        assert verdict.dissent == ()
