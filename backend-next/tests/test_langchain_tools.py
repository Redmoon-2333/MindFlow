"""Tests for LangChain tool declarations (agents/langchain_tools.py).

Covers:
  - Each of the 4 tools on the happy path
  - run_panel per-session cap (1 max)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mindflow.agents.langchain_tools as langchain_tools
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

    async def test_get_latest_analysis_uses_business_timezone(
        self, mock_analysis_repo: AsyncMock
    ) -> None:
        current_user_id.set(1)
        mock_analysis_repo.get_by_date = AsyncMock(return_value={"type": "ok"})
        with patch(
            "mindflow.agents.langchain_tools.business_today",
            return_value=date(2026, 7, 26),
        ) as business_today_mock:
            tool = make_get_latest_analysis(mock_analysis_repo, "Asia/Shanghai")
            await tool.ainvoke({})
        business_today_mock.assert_called_once_with("Asia/Shanghai")
        mock_analysis_repo.get_by_date.assert_awaited_once_with(1, date(2026, 7, 26))

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

    async def test_run_panel_same_session_limit_is_atomic(
        self,
        mock_panel_service: AsyncMock,
    ) -> None:
        """A concurrent second call is rejected while the first is in flight."""
        current_user_id.set(1)
        current_session_id.set("atomic-cap-session")
        entered = asyncio.Event()
        release = asyncio.Event()

        mock_verdict = MagicMock(types=(), confidence={}, rationale="panel complete")

        async def blocked_panel(*args: object, **kwargs: object) -> MagicMock:
            entered.set()
            await release.wait()
            return mock_verdict

        mock_panel_service.run_daily_panel = AsyncMock(side_effect=blocked_panel)
        panel_tool = make_run_panel(mock_panel_service)

        first = asyncio.create_task(panel_tool.ainvoke({}))
        await entered.wait()
        second = asyncio.create_task(panel_tool.ainvoke({}))
        try:
            await asyncio.sleep(0.05)
            assert second.done()
            assert "\u5df2\u8d85\u51fa" in await second
        finally:
            release.set()
            await first

        assert mock_panel_service.run_daily_panel.call_count == 1

    async def test_run_panel_usage_tracking_is_bounded(
        self,
        mock_panel_service: AsyncMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Completed session usage evicts oldest entries above the tracking cap."""
        monkeypatch.setattr(langchain_tools, "_MAX_SESSION_PANEL_USAGE", 3)
        current_user_id.set(1)
        mock_verdict = MagicMock(types=(), confidence={}, rationale="panel complete")
        mock_panel_service.run_daily_panel = AsyncMock(return_value=mock_verdict)
        panel_tool = make_run_panel(mock_panel_service)

        for index in range(4):
            current_session_id.set(f"bounded-session-{index}")
            assert "panel complete" in await panel_tool.ainvoke({})

        assert len(session_panel_usage) == 3
        assert "bounded-session-0" not in session_panel_usage


class TestPanelLockAsyncSafety:
    """Panel session reservation lock MUST use asyncio.Lock, not threading.Lock.

    threading.Lock is not event-loop-safe: if the lock-holding code ever
    yields to the event loop (e.g., adding I/O inside the critical section),
    other coroutines trying to acquire the lock block the entire event loop
    thread instead of just their own coroutine.  asyncio.Lock makes lock
    acquisition cooperative — contenders ``await`` the lock and yield to the
    event loop until the holder releases.
    """

    async def test_panel_lock_is_asyncio_lock(self) -> None:
        """RED: _session_panel_usage_lock must be asyncio.Lock, not threading.Lock."""
        import asyncio as _asyncio

        assert isinstance(
            langchain_tools._session_panel_usage_lock, _asyncio.Lock
        ), (
            "RED: _session_panel_usage_lock is a threading.Lock — this blocks the "
            "entire event-loop thread when contended in async code.  Replace with "
            "asyncio.Lock so that contending coroutines cooperatively yield."
        )

    async def test_panel_async_lock_cooperatively_awaits(self) -> None:
        """When the lock is held, contenders await cooperatively (no event-loop block)."""
        import asyncio as _asyncio

        lock = _asyncio.Lock()
        holder_entered = _asyncio.Event()
        release_signal = _asyncio.Event()

        async def hold_across_await() -> None:
            async with lock:
                holder_entered.set()
                await release_signal.wait()

        async def contender() -> bool:
            await holder_entered.wait()
            # asyncio.Lock: this await yields to the event loop cooperatively
            async with lock:
                return True

        t_hold = _asyncio.create_task(hold_across_await())
        await holder_entered.wait()

        t_contend = _asyncio.create_task(contender())
        # The contender must NOT be done — it's awaiting the lock
        done_early, _ = await _asyncio.wait([t_contend], timeout=0.3)
        assert not done_early, (
            "asyncio.Lock acquired while held — lock was not cooperative"
        )

        release_signal.set()
        await t_contend  # now it should complete
        await t_hold


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
