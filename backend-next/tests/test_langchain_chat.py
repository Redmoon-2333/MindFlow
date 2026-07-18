"""Tests for LangChain agent integration in ChatService.

Covers:
  - Normal Q&A with agent
  - Crisis detection short-circuit (pre-LLM)
  - LLM failure → safe fallback reply
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
from mindflow.services.chat_service import ChatService


class FakeChatModel(FakeListChatModel):
    """A ``FakeListChatModel`` that supports ``bind_tools``, enabling usage
    with ``create_agent`` even when tools are registered."""

    def bind_tools(
        self,
        tools: Any,
        **kwargs: Any,
    ) -> FakeChatModel:
        return self


def _make_mock_gateway() -> AsyncMock:
    """Create a backward-compat mock DeepSeekGateway."""
    gw = AsyncMock()
    gw._api_key = "test-key"
    gw._base_url = "https://api.deepseek.com"
    return gw


@pytest.fixture
def chat_service() -> ChatService:
    """Create a ChatService that uses a FakeChatModel for the agent.

    The underlying repositories are mocked so no real DB/LLM I/O occurs.
    """
    # Build with __new__ + manual fixture (matches test_chat_service.py pattern)
    service = ChatService.__new__(ChatService)
    service._chat_repo = AsyncMock()
    service._chat_repo.append = AsyncMock()
    service._chat_repo.recent = AsyncMock(return_value=[])
    service._crisis_detector = MagicMock(spec=CrisisDetector)
    service._crisis_detector.scan.return_value = (CrisisLevel.NONE, None)
    service._llm_gateway = _make_mock_gateway()

    # Replace the real agent with one backed by FakeChatModel
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def _dummy_tool() -> str:  # type: ignore[empty-docstring]
        """Placeholder tool for agent construction."""
        return "ok"

    llm = FakeChatModel(responses=["你好！我是 MindFlow 助手，有什么可以帮助你的？"])
    service._agent = create_agent(
        model=llm,
        tools=[_dummy_tool],
        system_prompt="你是 MindFlow 的 AI 助手。",
    )
    return service


class TestChatAgentNormal:
    """Normal Q&A through the LangChain agent."""

    async def test_normal_qa(self, chat_service: ChatService) -> None:
        """Agent returns a direct answer for a simple greeting."""
        result = await chat_service.ask(
            user_id=1, session_id="s1", message="你好",
        )

        assert result.answer == "你好！我是 MindFlow 助手，有什么可以帮助你的？"
        assert result.degraded is False
        # Message was persisted (user + assistant)
        assert chat_service._chat_repo.append.call_count == 2


class TestChatCrisisDetection:
    """Crisis detection short-circuits the agent."""

    async def test_crisis_returns_hotline(self, chat_service: ChatService) -> None:
        """Crisis detection → hotline response, no agent call."""
        chat_service._crisis_detector.scan.return_value = (
            CrisisLevel.HIGH,
            MagicMock(
                message="全国24小时心理援助热线：400-161-9995",
                stop_llm=True,
            ),
        )

        result = await chat_service.ask(
            user_id=1, session_id="s1", message="我想自杀",
        )

        assert "400-161-9995" in result.answer
        assert result.degraded is True
        # No persistence (agent not called)
        chat_service._chat_repo.append.assert_not_called()


class TestChatLLMFailure:
    """LLM failure → safe fallback."""

    async def test_llm_down_fallback(self, chat_service: ChatService) -> None:
        """When the agent raises, fall back to safe reply."""
        from langchain.agents import create_agent
        from langchain_core.tools import tool

        @tool
        def _dummy_tool() -> str:
            """Placeholder."""
            return "ok"

        # Make an agent that raises on invoke by using an empty responses list
        # that forces IndexError when the LLM tries to get a response.
        chat_service._agent = create_agent(
            model=FakeChatModel(responses=[]),
            tools=[_dummy_tool],
            system_prompt="test",
        )
        result = await chat_service.ask(
            user_id=1, session_id="s1", message="你好",
        )

        # Should fall back to safe reply
        assert isinstance(result.answer, str)
        assert result.degraded is True
