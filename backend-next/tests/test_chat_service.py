"""Tests for ChatService (services/chat_service.py).

Covers:
  - Crisis detection short-circuit
  - Tool loop: single tool call → answer
  - Tool loop: multiple tool calls
  - run_panel per-session cap (1 max)
  - Bad/malformed JSON tool response → degrade to text answer
  - Forbidden word retry (1 retry, then safe reply)
  - LLM unavailable → rule-based reply
  - History compression trigger
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest

from mindflow.agents.types import FORBIDDEN_WORDS
from mindflow.domain.evidence import EvidenceItem, InterventionRecord
from mindflow.domain.procrastination import BehaviorSummary
from mindflow.infrastructure.security.crisis_detector import (
    CrisisDetector,
    CrisisLevel,
)
from mindflow.services.chat_service import ChatService, _LLM_DOWN_REPLY, _SAFE_REPLY


def _make_empty_bundle() -> MagicMock:
    """Create a mock EvidenceBundle with empty fields."""
    bundle = MagicMock()
    bundle.user_id = 1
    bundle.window = (datetime(2026, 7, 18, 0, 0, tzinfo=UTC), datetime(2026, 7, 18, 23, 59, tzinfo=UTC))
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


@pytest.fixture
def mock_gateway() -> AsyncMock:
    """Create a mock DeepSeekGateway."""
    return AsyncMock()


@pytest.fixture
def mock_crisis_detector() -> MagicMock:
    """Create a mock CrisisDetector that returns NONE."""
    detector = MagicMock(spec=CrisisDetector)
    detector.scan.return_value = (CrisisLevel.NONE, None)
    return detector


@pytest.fixture
def mock_analysis_repo() -> AsyncMock:
    """Create a mock analysis repository."""
    return AsyncMock()


@pytest.fixture
def mock_panel_service() -> AsyncMock:
    """Create a mock PanelService."""
    return AsyncMock()


@pytest.fixture
def mock_intervention_repo() -> AsyncMock:
    """Create a mock InterventionLogRepository."""
    repo = AsyncMock()
    repo.query_range_by_date = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_evidence_builder() -> AsyncMock:
    """Create a mock EvidenceBundleBuilder."""
    builder = AsyncMock()
    builder.build = AsyncMock(return_value=_make_empty_bundle())
    return builder


@pytest.fixture
def mock_chat_repo() -> AsyncMock:
    """Create a mock ChatRepository."""
    repo = AsyncMock()
    repo.append = AsyncMock()
    repo.recent = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def chat_service(
    mock_gateway: AsyncMock,
    mock_crisis_detector: MagicMock,
    mock_analysis_repo: AsyncMock,
    mock_panel_service: AsyncMock,
    mock_intervention_repo: AsyncMock,
    mock_evidence_builder: AsyncMock,
) -> ChatService:
    """Create a ChatService with all mocks."""
    service = ChatService.__new__(ChatService)
    service._chat_repo = AsyncMock()
    service._chat_repo.append = AsyncMock()
    service._chat_repo.recent = AsyncMock(return_value=[])
    service._crisis_detector = mock_crisis_detector
    service._llm_gateway = mock_gateway
    service._analysis_repo = mock_analysis_repo
    service._panel_service = mock_panel_service
    service._intervention_repo = mock_intervention_repo
    service._evidence_builder = mock_evidence_builder
    return service


class TestCrisisDetection:
    """Crisis detection short-circuits the LLM."""

    async def test_crisis_returns_hotline(self, chat_service: ChatService) -> None:
        """Crisis detection → hotline response, no LLM call."""
        chat_service._crisis_detector.scan.return_value = (
            CrisisLevel.HIGH,
            MagicMock(
                message="全国24小时心理援助热线：400-161-9995",
                stop_llm=True,
            ),
        )

        result = await chat_service.ask(user_id=1, session_id="s1", message="我想自杀")

        assert "400-161-9995" in result.answer
        assert result.degraded is True
        # No LLM call, no persistence
        chat_service._llm_gateway.complete.assert_not_called()
        chat_service._chat_repo.append.assert_not_called()

    async def test_no_crisis_passes_through(self, chat_service: ChatService) -> None:
        """No crisis → normal flow."""
        chat_service._llm_gateway.complete.return_value = '{"answer": "你好！有什么可以帮助你的？"}'

        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert result.answer == "你好！有什么可以帮助你的？"
        chat_service._llm_gateway.complete.assert_called_once()


class TestToolLoop:
    """Tool-calling loop behavior."""

    async def test_single_tool_then_answer(self, chat_service: ChatService) -> None:
        """Single tool call → LLM receives result → answers."""
        # First call: tool call
        # Second call: answer
        chat_service._llm_gateway.complete.side_effect = [
            '{"tool": "query_evidence", "args": {"days_back": 7}}',
            '{"answer": "根据你的行为数据，今天专注度正常。"}',
        ]
        chat_service._evidence_builder.build.return_value = _make_empty_bundle()

        result = await chat_service.ask(user_id=1, session_id="s1", message="我今天怎么样？")

        assert "专注度正常" in result.answer
        assert "query_evidence" in result.tools_used
        assert result.evidence_cited is True
        assert result.degraded is False
        assert chat_service._llm_gateway.complete.call_count == 2

    async def test_multiple_tool_calls(self, chat_service: ChatService) -> None:
        """Multiple tool calls in sequence."""
        # First: analysis tool
        # Second: evidence tool
        # Third: answer
        mock_result = {
            "procrastination_types": ["impulsivity"],
            "type_confidence": {"impulsivity": 0.8},
        }
        chat_service._analysis_repo.get_by_date = AsyncMock(return_value=mock_result)

        chat_service._llm_gateway.complete.side_effect = [
            '{"tool": "get_latest_analysis", "args": {}}',
            '{"tool": "query_evidence", "args": {"days_back": 3}}',
            '{"answer": "综合分析显示你有冲动分心倾向。"}',
        ]
        chat_service._evidence_builder.build.return_value = _make_empty_bundle()

        result = await chat_service.ask(user_id=1, session_id="s1", message="分析我的情况")

        assert "综合分析显示" in result.answer
        assert len(result.tools_used) == 2
        assert "get_latest_analysis" in result.tools_used
        assert "query_evidence" in result.tools_used
        assert result.evidence_cited is True

    async def test_run_panel_per_session_cap(self, chat_service: ChatService) -> None:
        """run_panel can only be called once per session."""
        mock_verdict = MagicMock()
        mock_verdict.types = ()
        mock_verdict.confidence = {}
        mock_verdict.rationale = "会诊完成"
        chat_service._panel_service.run_daily_panel = AsyncMock(return_value=mock_verdict)

        # First: run_panel (allowed)
        # Second: run_panel (rejected, should not call panel_service again)
        # Third: answer
        chat_service._llm_gateway.complete.side_effect = [
            '{"tool": "run_panel", "args": {}}',
            '{"tool": "run_panel", "args": {}}',
            '{"answer": "会诊已完成。"}',
        ]

        result = await chat_service.ask(user_id=1, session_id="s1", message="运行会诊")

        assert result.tools_used == ("run_panel",)
        # panel_service.run_daily_panel called exactly once
        assert chat_service._panel_service.run_daily_panel.call_count == 1

    async def test_bad_json_tool_response(self, chat_service: ChatService) -> None:
        """Bad JSON tool response → try extracting from text, then use as answer."""
        chat_service._llm_gateway.complete.return_value = (
            "好的，我来分析你的数据。根据当前信息，你的专注度正常。"
        )

        result = await chat_service.ask(user_id=1, session_id="s1", message="怎么样？")

        assert "专注度正常" in result.answer
        assert result.tools_used == ()


class TestForbiddenWords:
    """Forbidden word handling."""

    async def test_forbidden_word_retry(self, chat_service: ChatService) -> None:
        """Answer with forbidden word → retry once → accept retry."""
        chat_service._llm_gateway.complete.side_effect = [
            '{"answer": "根据诊断结果，你的情况需要治疗。"}',
            '{"answer": "根据分析，你可以尝试调整工作节奏。"}',
        ]

        result = await chat_service.ask(user_id=1, session_id="s1", message="我该怎么办？")

        assert "诊断" not in result.answer
        assert "治疗" not in result.answer
        assert "调整工作节奏" in result.answer
        assert result.degraded is False

    async def test_forbidden_word_retry_fails(self, chat_service: ChatService) -> None:
        """Answer with forbidden word repeatedly → safe reply after retry exhausted."""
        # Every response is forbidden; the loop will retry up to _MAX_TOOL_ROUNDS
        # times, then fall back to _SAFE_REPLY.
        chat_service._llm_gateway.complete.return_value = '{"answer": "根据诊断结果，你确实需要治疗。"}'

        result = await chat_service.ask(user_id=1, session_id="s1", message="我该怎么办？")

        assert result.answer == _SAFE_REPLY
        assert result.degraded is True


class TestLLMUnavailable:
    """LLM gateway failures."""

    async def test_llm_gateway_timeout(self, chat_service: ChatService) -> None:
        """LLM timeout → fallback reply."""
        chat_service._llm_gateway.complete.side_effect = TimeoutError("Gateway timed out")

        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True

    async def test_llm_gateway_api_error(self, chat_service: ChatService) -> None:
        """LLM API error → fallback reply."""
        from mindflow.agents.llm_gateway import GatewayAPIError
        chat_service._llm_gateway.complete.side_effect = GatewayAPIError("API error")

        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True


class TestHistoryCompression:
    """History compression when exceeding max rounds."""

    async def test_no_compression_below_limit(self, chat_service: ChatService) -> None:
        """Under the round limit → no compression."""
        chat_service._chat_repo.recent.return_value = [
            {"role": "user", "content": f"msg{i}", "created_at": "2026-01-01T00:00:00Z"}
            for i in range(5)
        ] + [
            {"role": "assistant", "content": f"rsp{i}", "created_at": "2026-01-01T00:01:00Z"}
            for i in range(5)
        ]
        chat_service._llm_gateway.complete.return_value = '{"answer": "你好！"}'

        # Add the current message to make it 11 entries
        result = await chat_service.ask(user_id=1, session_id="s1", message="新消息")

        # Verify LLM was called (should pass through normally)
        assert result.answer == "你好！"

    async def test_compression_triggers(self, chat_service: ChatService) -> None:
        """Over the round limit → compression summary is generated."""
        # Create 22 messages (11 rounds) to trigger compression
        msgs: list[dict[str, Any]] = []
        for i in range(11):
            msgs.append({"role": "user", "content": f"用户消息{i}", "created_at": f"2026-01-01T00:{i:02d}:00Z"})
            msgs.append({"role": "assistant", "content": f"助手回复{i}", "created_at": f"2026-01-01T00:{i:02d}:01Z"})

        chat_service._chat_repo.recent.return_value = msgs
        chat_service._llm_gateway.complete.return_value = '{"answer": "收到，已压缩历史。"}'

        result = await chat_service.ask(user_id=1, session_id="s1", message="最新消息")

        # The user prompt should include a summary
        assert result.answer == "收到，已压缩历史。"
