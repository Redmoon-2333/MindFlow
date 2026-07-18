"""Tests for LangChain tool declarations (agents/langchain_tools.py).

Covers:
  - Each of the 4 tools on the happy path
  - run_panel per-session cap (1 max)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.agents.langchain_tools import (
    current_session_id,
    current_user_id,
    make_get_latest_analysis,
    make_query_evidence,
    make_query_interventions,
    make_run_panel,
    session_panel_usage,
)
from mindflow.domain.procrastination import BehaviorSummary
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.services.evidence_service import EvidenceBundleBuilder
from mindflow.services.panel_service import PanelService


def _make_empty_bundle() -> MagicMock:
    """Create a mock EvidenceBundle with empty fields."""
    bundle = MagicMock()
    bundle.user_id = 1
    bundle.window = (
        datetime(2026, 7, 18, 0, 0, tzinfo=UTC),
        datetime(2026, 7, 18, 23, 59, tzinfo=UTC),
    )
    bundle.items = ()
    bundle.behavior_summary = BehaviorSummary(
        intended_task=None,
        duration_min=0.0,
        actual_focus_min=0.0,
        context_switches_per_hour=0.0,
        longest_focus_block_s=0.0,
        social_media_ratio=0.0,
        start_delay_min=0.0,
        keyword_flags=frozenset(),
        baseline_deviation=None,
    )
    bundle.intervention_history = ()
    bundle.novelty_flags = ()
    return bundle


@pytest.fixture(autouse=True)
def _reset_context() -> None:
    """Reset context vars and panel usage before each test."""
    current_user_id.set(0)
    current_session_id.set(None)
    session_panel_usage.clear()


@pytest.fixture
def mock_evidence_builder() -> AsyncMock:
    """Create a mock EvidenceBundleBuilder."""
    builder = AsyncMock(spec=EvidenceBundleBuilder)
    builder.build = AsyncMock(return_value=_make_empty_bundle())
    return builder


@pytest.fixture
def mock_analysis_repo() -> AsyncMock:
    """Create a mock analysis repository."""
    repo = AsyncMock(spec=SQLAlchemyProcrastinationAnalysisRepository)
    repo.get_by_date = AsyncMock(return_value=None)
    return repo


@pytest.fixture
def mock_panel_service() -> AsyncMock:
    """Create a mock PanelService."""
    return AsyncMock(spec=PanelService)


@pytest.fixture
def mock_intervention_repo() -> AsyncMock:
    """Create a mock InterventionLogRepository."""
    repo = AsyncMock(spec=InterventionLogRepository)
    repo.query_range_by_date = AsyncMock(return_value=[])
    return repo


class TestQueryEvidence:
    """query_evidence tool happy path."""

    async def test_query_evidence_returns_json(
        self,
        mock_evidence_builder: AsyncMock,
    ) -> None:
        """Tool returns a non-empty JSON string."""
        current_user_id.set(1)
        tool = make_query_evidence(mock_evidence_builder)
        result = await tool.ainvoke({"days_back": 7})

        assert isinstance(result, str)
        assert len(result) > 0
        assert result.startswith("{")
        assert '"window"' in result
        # EvidenceBundle was built with the correct user
        mock_evidence_builder.build.assert_called_once()


class TestGetLatestAnalysis:
    """get_latest_analysis tool happy path."""

    async def test_get_latest_analysis_returns_data(
        self,
        mock_analysis_repo: AsyncMock,
    ) -> None:
        """Tool returns analysis JSON when data exists."""
        current_user_id.set(1)
        mock_analysis_repo.get_by_date = AsyncMock(
            return_value={
                "procrastination_types": ["impulsivity"],
                "type_confidence": {"impulsivity": 0.8},
            }
        )

        tool = make_get_latest_analysis(mock_analysis_repo)
        result = await tool.ainvoke({})

        assert "impulsivity" in result
        mock_analysis_repo.get_by_date.assert_called_once()

    async def test_get_latest_analysis_not_found(
        self,
        mock_analysis_repo: AsyncMock,
    ) -> None:
        """Tool returns 'not found' message when no data exists."""
        current_user_id.set(1)
        mock_analysis_repo.get_by_date = AsyncMock(return_value=None)

        tool = make_get_latest_analysis(mock_analysis_repo)
        result = await tool.ainvoke({})

        assert "暂无分析数据" in result


class TestRunPanel:
    """run_panel tool happy path and cap."""

    async def test_run_panel_returns_verdict(
        self,
        mock_panel_service: AsyncMock,
    ) -> None:
        """Tool returns panel verdict JSON."""
        current_user_id.set(1)
        current_session_id.set("test-session-1")

        mock_verdict = MagicMock()
        mock_verdict.types = ()
        mock_verdict.confidence = {}
        mock_verdict.rationale = "会诊完成"
        mock_panel_service.run_daily_panel = AsyncMock(return_value=mock_verdict)

        tool = make_run_panel(mock_panel_service)
        result = await tool.ainvoke({})

        assert "会诊完成" in result
        mock_panel_service.run_daily_panel.assert_called_once()

    async def test_run_panel_per_session_cap(
        self,
        mock_panel_service: AsyncMock,
    ) -> None:
        """run_panel enforces 1-call-per-session limit."""
        current_user_id.set(1)
        current_session_id.set("cap-test-session")

        mock_verdict = MagicMock()
        mock_verdict.types = ()
        mock_verdict.confidence = {}
        mock_verdict.rationale = "会诊完成"
        mock_panel_service.run_daily_panel = AsyncMock(return_value=mock_verdict)

        tool = make_run_panel(mock_panel_service)

        # First call: succeeds
        result1 = await tool.ainvoke({})
        assert "会诊完成" in result1
        assert mock_panel_service.run_daily_panel.call_count == 1

        # Second call: rejected by cap
        result2 = await tool.ainvoke({})
        assert "已超出" in result2
        # run_daily_panel not called again
        assert mock_panel_service.run_daily_panel.call_count == 1


class TestQueryInterventions:
    """query_interventions tool happy path."""

    async def test_query_interventions_returns_data(
        self,
        mock_intervention_repo: AsyncMock,
    ) -> None:
        """Tool returns intervention JSON when records exist."""
        current_user_id.set(1)
        mock_intervention_repo.query_range_by_date = AsyncMock(
            return_value=[
                {
                    "intervention_type": "nudge",
                    "triggered_at": "2026-07-18T10:00:00Z",
                    "user_response": "accepted",
                }
            ]
        )

        tool = make_query_interventions(mock_intervention_repo)
        result = await tool.ainvoke({"days_back": 7})

        assert "nudge" in result
        mock_intervention_repo.query_range_by_date.assert_called_once()

    async def test_query_interventions_not_found(
        self,
        mock_intervention_repo: AsyncMock,
    ) -> None:
        """Tool returns 'not found' message when no records exist."""
        current_user_id.set(1)
        mock_intervention_repo.query_range_by_date = AsyncMock(return_value=[])

        tool = make_query_interventions(mock_intervention_repo)
        result = await tool.ainvoke({"days_back": 7})

        assert "暂无干预记录" in result
