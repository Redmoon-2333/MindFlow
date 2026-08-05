"""Tests for ChatService (services/chat_service.py) — v2 ChatGraph delegation.

The legacy LangChain ``create_agent`` path and shadow-mode comparison were
removed with the V1 cleanup; ChatGraph is now the only chat path.

Covers:
  - Crisis detection short-circuit (pre-LLM gate in ChatService.ask)
  - Normal Q&A via the injected ChatGraph
  - ChatGraph unavailable / raising → degraded reply
  - Per-session locking serialises concurrent asks
  - Constructor builds a ChatGraph; no API key → degraded mode
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.infrastructure.security.crisis_detector import (
    CrisisDetector,
    CrisisLevel,
)
from mindflow.services.chat_service import _LLM_DOWN_REPLY, ChatAnswer, ChatService


def _make_mock_gateway() -> AsyncMock:
    """Create a backward-compat mock DeepSeekGateway."""
    gw = AsyncMock()
    gw._api_key = "test-key"
    gw._base_url = "https://api.deepseek.com"
    return gw


def _make_service(graph: Any = None, *, detector_scan: Any = None) -> ChatService:
    """Build a ChatService via __new__ with the given fake ChatGraph."""
    service = ChatService.__new__(ChatService)
    service._chat_repo = AsyncMock()
    service._chat_repo.append = AsyncMock()
    service._chat_repo.recent = AsyncMock(return_value=[])
    service._crisis_detector = MagicMock(spec=CrisisDetector)
    service._crisis_detector.scan.return_value = (
        detector_scan if detector_scan is not None else (CrisisLevel.NONE, None)
    )
    service._llm_gateway = _make_mock_gateway()
    service._session_locks = {}
    service._chat_graph = graph
    return service


@pytest.fixture
def chat_service() -> ChatService:
    """Create a ChatService with a fake ChatGraph injected."""
    fake_graph = MagicMock()
    fake_graph.ask = AsyncMock(
        return_value=ChatAnswer(
            answer="你好！有什么可以帮助你的？", session_id="s1",
        )
    )
    return _make_service(fake_graph)


class TestCrisisDetection:
    """Crisis detection short-circuits the LLM."""

    async def test_crisis_returns_hotline(self, chat_service: ChatService) -> None:
        """Crisis detection → hotline response, no graph call."""
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
        # No persistence (graph not called)
        chat_service._chat_graph.ask.assert_not_called()

    async def test_no_crisis_passes_through(self, chat_service: ChatService) -> None:
        """No crisis → normal flow through the ChatGraph."""
        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert "你好！有什么可以帮助你的？" in result.answer
        assert result.degraded is False
        chat_service._chat_graph.ask.assert_awaited_once()


class TestGraphDelegation:
    """ChatService delegates to the ChatGraph (v2 is the only chat path)."""

    async def test_ask_delegates_to_graph(self) -> None:
        """ask() forwards user/session/message to ChatGraph.ask()."""
        fake_graph = MagicMock()
        expected = ChatAnswer(answer="来自新图的回复", session_id="g1")
        fake_graph.ask = AsyncMock(return_value=expected)
        service = _make_service(fake_graph)

        result = await service.ask(user_id=1, session_id="g1", message="你好")

        assert result is expected
        fake_graph.ask.assert_awaited_once_with(1, "g1", "你好")

    async def test_graph_unavailable_degrades(self) -> None:
        """No ChatGraph injected → degraded safe reply."""
        service = _make_service(None)

        result = await service.ask(user_id=1, session_id="g2", message="你好")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True

    async def test_graph_raise_degrades(self) -> None:
        """ChatGraph raising → degraded safe reply."""
        fake_graph = MagicMock()
        fake_graph.ask = AsyncMock(side_effect=TimeoutError("Gateway timed out"))
        service = _make_service(fake_graph)

        result = await service.ask(user_id=1, session_id="g3", message="你好")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True


class TestChatConcurrency:
    """Concurrent asks on the same session are serialized by the session lock."""

    async def test_same_session_asks_are_serialized(self) -> None:
        fake_graph = MagicMock()
        entered = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def blocked_ask(
            user_id: int, session_id: str, message: str
        ) -> ChatAnswer:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                entered.set()
                await release.wait()
            return ChatAnswer(answer="done", session_id=session_id)

        fake_graph.ask = AsyncMock(side_effect=blocked_ask)
        service = _make_service(fake_graph)

        first = asyncio.create_task(
            service.ask(user_id=1, session_id="serial-session", message="first")
        )
        await entered.wait()
        second = asyncio.create_task(
            service.ask(user_id=1, session_id="serial-session", message="second")
        )
        try:
            await asyncio.sleep(0.05)
            assert call_count == 1
        finally:
            release.set()
            await asyncio.gather(first, second)

        assert call_count == 2


class TestConstructor:
    """Constructor builds a ChatGraph; no API key → degraded mode."""

    def _gateway(self) -> MagicMock:
        gw = MagicMock()
        gw._api_key = ""
        gw._base_url = ""
        return gw

    async def test_constructor_without_api_key_builds_degraded_graph(self) -> None:
        """No API key → ChatGraph built with model=None → degraded reply."""
        repository = MagicMock()
        repository.append = AsyncMock()
        repository.recent = AsyncMock(return_value=[])
        detector = MagicMock()
        detector.scan.return_value = (CrisisLevel.NONE, None)

        service = ChatService(
            session_factory=MagicMock(),
            crisis_detector=detector,
            llm_gateway=self._gateway(),
            analysis_repo=MagicMock(),
            panel_service=None,
            intervention_repo=MagicMock(),
            evidence_builder=MagicMock(),
            chat_repo=repository,
        )

        assert service._chat_graph is not None
        result = await service.ask(user_id=1, session_id="offline", message="??")
        assert result.degraded is True

    async def test_constructor_accepts_injected_graph(self) -> None:
        """An injected ChatGraph is used as-is."""
        fake_graph = MagicMock()
        repository = MagicMock()
        repository.append = AsyncMock()
        repository.recent = AsyncMock(return_value=[])
        detector = MagicMock()
        detector.scan.return_value = (CrisisLevel.NONE, None)

        service = ChatService(
            session_factory=MagicMock(),
            crisis_detector=detector,
            llm_gateway=self._gateway(),
            analysis_repo=MagicMock(),
            panel_service=None,
            intervention_repo=MagicMock(),
            evidence_builder=MagicMock(),
            chat_repo=repository,
            chat_graph=fake_graph,
        )

        assert service._chat_graph is fake_graph


class TestSharedResourceCleanup:
    """Shutdown closes shared resources once, not twice."""

    async def test_aclose_noop_with_registry(self) -> None:
        """With registry injected, aclose() is a no-op (registry manages pools)."""
        service = ChatService.__new__(ChatService)
        service._registry = MagicMock()  # registry present
        service._llm_gateway = AsyncMock()
        service._agent_model = MagicMock()

        # Should not raise, should not close anything
        await service.aclose()

        service._llm_gateway.close.assert_not_called()
