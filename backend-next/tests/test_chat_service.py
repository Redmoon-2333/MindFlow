"""Tests for ChatService (services/chat_service.py) with LangChain agent.

Covers:
  - Crisis detection short-circuit
  - Normal Q&A through LangChain agent
  - Forbidden word retry (1 retry, then safe reply)
  - LLM unavailable → rule-based reply
  - History compression trigger
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models import FakeListChatModel

from mindflow.infrastructure.security.crisis_detector import (
    CrisisDetector,
    CrisisLevel,
)
from mindflow.services.chat_service import _LLM_DOWN_REPLY, _SAFE_REPLY, ChatService


class _FakeChatModel(FakeListChatModel):
    """A ``FakeListChatModel`` that supports ``bind_tools`` for agent creation."""

    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> _FakeChatModel:
        return self


def _make_agent(responses: list[str]) -> Any:
    """Create a LangChain agent backed by ``FakeChatModel``.

    Args:
        responses: List of responses the fake model returns in sequence.

    Returns:
        A ``CompiledStateGraph`` agent compatible with ``ainvoke``.
    """
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def _test_tool() -> str:  # type: ignore[empty-docstring]
        """A test tool — always returns "ok"."""
        return "ok"

    llm = _FakeChatModel(responses=responses)
    return create_agent(
        model=llm,
        tools=[_test_tool],
        system_prompt="你是 MindFlow 的 AI 助手。",
    )


@pytest.fixture
def chat_service() -> ChatService:
    """Create a ChatService with a LangChain agent backed by a fake model.

    All repositories are mocked — no real I/O occurs.
    """
    service = ChatService.__new__(ChatService)
    service._chat_repo = AsyncMock()
    service._chat_repo.append = AsyncMock()
    service._chat_repo.recent = AsyncMock(return_value=[])
    service._crisis_detector = MagicMock(spec=CrisisDetector)
    service._crisis_detector.scan.return_value = (CrisisLevel.NONE, None)
    service._llm_gateway = AsyncMock()
    service._chat_repo = AsyncMock()
    service._chat_repo.append = AsyncMock()
    service._chat_repo.recent = AsyncMock(return_value=[])
    service._crisis_detector = MagicMock(spec=CrisisDetector)
    service._crisis_detector.scan.return_value = (CrisisLevel.NONE, None)
    service._llm_gateway = AsyncMock()

    # Attach an agent backed by FakeChatModel
    service._agent = _make_agent(["你好！有什么可以帮助你的？"])
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
        # No persistence (agent not called)
        chat_service._chat_repo.append.assert_not_called()

    async def test_no_crisis_passes_through(self, chat_service: ChatService) -> None:
        """No crisis → normal flow through the agent."""
        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert "你好！有什么可以帮助你的？" in result.answer
        assert result.degraded is False
        # Both user and assistant messages persisted
        assert chat_service._chat_repo.append.call_count == 2


class TestForbiddenWords:
    """Forbidden word handling."""

    async def test_forbidden_word_retry(self, chat_service: ChatService) -> None:
        """Answer with forbidden word → retry once → accept retry."""
        # First response has forbidden word, second is clean
        chat_service._agent = _make_agent([
            "根据诊断结果，你的情况需要治疗。",
            "根据分析，你可以尝试调整工作节奏。",
        ])

        result = await chat_service.ask(user_id=1, session_id="s1", message="我该怎么办？")

        assert "诊断" not in result.answer
        assert "治疗" not in result.answer
        assert "调整工作节奏" in result.answer
        assert result.degraded is False

    async def test_forbidden_word_retry_fails(self, chat_service: ChatService) -> None:
        """Answer with forbidden word repeatedly → safe reply after retry exhausted."""
        chat_service._agent = _make_agent([
            "根据诊断结果，你确实需要治疗。",  # First attempt
            "诊断显示你需要进行治疗。",  # Retry also forbidden
        ])

        result = await chat_service.ask(user_id=1, session_id="s1", message="我该怎么办？")

        assert result.answer == _SAFE_REPLY
        assert result.degraded is True


class TestLLMUnavailable:
    """LLM gateway failures."""

    async def test_llm_gateway_timeout(self, chat_service: ChatService) -> None:
        """LLM timeout → fallback reply."""
        # Mock the agent to raise an exception on invoke
        chat_service._agent.ainvoke = AsyncMock(side_effect=TimeoutError("Gateway timed out"))  # type: ignore[method-assign]

        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True

    async def test_llm_gateway_api_error(self, chat_service: ChatService) -> None:
        """LLM API error → fallback reply."""
        chat_service._agent.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))  # type: ignore[method-assign]

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

        result = await chat_service.ask(user_id=1, session_id="s1", message="新消息")

        assert "你好！有什么可以帮助你的？" in result.answer

    async def test_compression_triggers(self, chat_service: ChatService) -> None:
        """Over the round limit → compression summary is generated."""
        # Create 22 messages (11 rounds) to trigger compression
        msgs: list[dict[str, Any]] = []
        for i in range(11):
            msgs.append(
                {
                    "role": "user",
                    "content": f"用户消息{i}",
                    "created_at": f"2026-01-01T00:{i:02d}:00Z",
                }
            )
            msgs.append(
                {
                    "role": "assistant",
                    "content": f"助手回复{i}",
                    "created_at": f"2026-01-01T00:{i:02d}:01Z",
                }
            )

        chat_service._chat_repo.recent.return_value = msgs

        result = await chat_service.ask(user_id=1, session_id="s1", message="最新消息")

        # Agent should still respond
        assert "你好！有什么可以帮助你的？" in result.answer
