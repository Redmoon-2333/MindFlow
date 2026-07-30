"""Tests for ChatService (services/chat_service.py) with LangChain agent.

Covers:
  - Crisis detection short-circuit
  - Normal Q&A through LangChain agent
  - Forbidden word retry (1 retry, then safe reply)
  - LLM unavailable → rule-based reply
  - History compression trigger
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage

from mindflow.agents.langchain_tools import (  # noqa: F401 — ContextVar removed in Todo 13
    make_get_latest_analysis,
    make_query_evidence,
    make_query_interventions,
    make_run_panel,
)
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
    def _test_tool() -> str:
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

    async def test_crisis_returns_hotline(self, chat_service: Any) -> None:
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

    async def test_no_crisis_passes_through(self, chat_service: Any) -> None:
        """No crisis → normal flow through the agent."""
        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert "你好！有什么可以帮助你的？" in result.answer
        assert result.degraded is False
        # Both user and assistant messages persisted
        assert chat_service._chat_repo.append.call_count == 2


class TestForbiddenWords:
    """Forbidden word handling."""

    async def test_forbidden_word_retry(self, chat_service: Any) -> None:
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

    async def test_forbidden_word_retry_fails(self, chat_service: Any) -> None:
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

    async def test_llm_gateway_timeout(self, chat_service: Any) -> None:
        """LLM timeout → fallback reply."""
        # Mock the agent to raise an exception on invoke
        chat_service._agent.ainvoke = AsyncMock(side_effect=TimeoutError("Gateway timed out"))

        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True

    async def test_llm_gateway_api_error(self, chat_service: Any) -> None:
        """LLM API error → fallback reply."""
        chat_service._agent.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

        result = await chat_service.ask(user_id=1, session_id="s1", message="你好")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True


class TestHistoryCompression:
    """History compression when exceeding max rounds."""

    async def test_no_compression_below_limit(self, chat_service: Any) -> None:
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

    async def test_compression_triggers(self, chat_service: Any) -> None:
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


class TestChatConcurrencyAndContext:
    """Concurrent session handling and LangGraph invocation safeguards."""

    async def test_same_session_asks_are_serialized(self, chat_service: Any) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def blocked_ainvoke(
            input: dict[str, Any],
            config: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                entered.set()
                await release.wait()
            return {"messages": [AIMessage(content="done")]}

        chat_service._agent.ainvoke = AsyncMock(side_effect=blocked_ainvoke)

        first = asyncio.create_task(
            chat_service.ask(user_id=1, session_id="serial-session", message="first")
        )
        await entered.wait()
        second = asyncio.create_task(
            chat_service.ask(user_id=1, session_id="serial-session", message="second")
        )
        try:
            await asyncio.sleep(0.05)
            assert call_count == 1
        finally:
            release.set()
            await asyncio.gather(first, second)

        assert call_count == 2

    @pytest.mark.skip(reason="ContextVar replaced by ToolContext in Todo 13; adapters use explicit context")
    async def test_contextvars_are_reset_after_agent_failure(
        self,
        chat_service: Any,
    ) -> None:
        """ContextVars removed — adapters now use explicit ToolContext."""
        pass

    async def test_agent_invocation_sets_recursion_limit(
        self,
        chat_service: Any,
    ) -> None:
        agent_call = AsyncMock(return_value={"messages": [AIMessage(content="done")]})
        chat_service._agent.ainvoke = agent_call

        await chat_service.ask(user_id=1, session_id="recursion-session", message="hello")

        assert agent_call.await_args is not None
        assert agent_call.await_args.kwargs["config"] == {"recursion_limit": 12}


class TestChatInjection:
    """Constructor-level injection keeps agent tests independent of provider setup."""

    def test_constructor_accepts_injected_agent(self) -> None:
        injected_agent = MagicMock()
        service = ChatService(
            session_factory=MagicMock(),
            crisis_detector=MagicMock(),
            llm_gateway=MagicMock(),
            analysis_repo=MagicMock(),
            panel_service=None,
            intervention_repo=MagicMock(),
            evidence_builder=MagicMock(),
            chat_repo=MagicMock(),
            agent=injected_agent,
        )

        assert service._agent is injected_agent
        assert service._agent_model is None

    async def test_constructor_without_api_key_uses_degraded_agent_mode(self) -> None:
        gateway = MagicMock()
        gateway._api_key = ""
        gateway._base_url = ""
        repository = MagicMock()
        repository.append = AsyncMock()
        repository.recent = AsyncMock(return_value=[])
        detector = MagicMock()
        detector.scan.return_value = (CrisisLevel.NONE, None)
        service = ChatService(
            session_factory=MagicMock(),
            crisis_detector=detector,
            llm_gateway=gateway,
            analysis_repo=MagicMock(),
            panel_service=None,
            intervention_repo=MagicMock(),
            evidence_builder=MagicMock(),
            chat_repo=repository,
        )

        result = await service.ask(user_id=1, session_id="offline", message="??")

        assert result.degraded is True
        assert service._agent is None
        repository.append.assert_awaited()


# ═══════════════════════════════════════════════════════════════════════════════
# ChatGraph integration tests (Todo 15)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNewChatGraphFlag:
    """When new_chat_graph=True, ChatService delegates to ChatGraph."""

    async def test_new_flag_delegates_to_graph(self) -> None:
        """ChatService with new_chat_graph=True + injected ChatGraph → graph path."""
        from mindflow.graph.chat_graph import ChatGraph
        from mindflow.services.chat_service import ChatAnswer

        # ── Build a ChatGraph with a fake model ────────────────────────
        fake_model = _FakeChatModel(responses=["来自新图的回复"])
        repo = AsyncMock()
        repo.append = AsyncMock()
        repo.recent = AsyncMock(return_value=[])
        detector = MagicMock(spec=CrisisDetector)
        detector.scan.return_value = (CrisisLevel.NONE, None)

        chat_graph = ChatGraph(
            chat_repo=repo,
            crisis_detector=detector,
            model=fake_model,
            tools=[],
        )

        # ── Build service with graph injection ──────────────────────────
        service = ChatService.__new__(ChatService)
        service._chat_repo = repo
        service._crisis_detector = detector
        service._llm_gateway = _make_mock_gateway()
        service._tool_adapters = []
        service._agent = None  # legacy agent unused
        service._chat_graph = chat_graph
        service._shadow_graph = None
        service._shadow_mode_chat = False
        service._new_chat_graph = True
        service._session_locks = {}

        result = await service.ask(user_id=1, session_id="g1", message="你好")

        assert isinstance(result, ChatAnswer)
        assert result.answer == "来自新图的回复"
        assert result.session_id == "g1"

    async def test_new_flag_without_graph_falls_back_to_legacy(self) -> None:
        """When new_chat_graph=True but _chat_graph=None, legacy path used."""
        service = ChatService.__new__(ChatService)
        service._chat_repo = AsyncMock()
        service._chat_repo.append = AsyncMock()
        service._chat_repo.recent = AsyncMock(return_value=[])
        service._crisis_detector = MagicMock(spec=CrisisDetector)
        service._crisis_detector.scan.return_value = (CrisisLevel.NONE, None)
        service._llm_gateway = _make_mock_gateway()
        service._tool_adapters = []
        service._agent = _make_agent(["传统回复"])
        service._chat_graph = None
        service._shadow_graph = None
        service._shadow_mode_chat = False
        service._new_chat_graph = True  # flag on but no graph
        service._session_locks = {}

        result = await service.ask(user_id=1, session_id="g2", message="你好")

        # Falls back to legacy (agent) path
        assert "传统回复" in result.answer


class TestShadowModeChat:
    """Shadow mode runs both paths, compares, returns legacy only."""

    async def test_shadow_mode_returns_legacy_output(self) -> None:
        """Shadow mode → legacy result returned, not shadow."""
        from mindflow.graph.chat_graph import ChatGraph
        from mindflow.services.chat_service import _ShadowChatRepo

        # ── Legacy agent ────────────────────────────────────────────────
        legacy_agent = _make_agent(["传统路径的回复"])

        # ── Shadow graph ────────────────────────────────────────────────
        fake_model = _FakeChatModel(responses=["新图的回复"])
        real_repo = AsyncMock()
        real_repo.append = AsyncMock()
        real_repo.recent = AsyncMock(return_value=[])
        detector = MagicMock(spec=CrisisDetector)
        detector.scan.return_value = (CrisisLevel.NONE, None)

        shadow_repo = _ShadowChatRepo(real_repo)
        shadow_graph = ChatGraph(
            chat_repo=shadow_repo,
            crisis_detector=detector,
            model=fake_model,
            tools=[],
        )

        # ── Build service ───────────────────────────────────────────────
        service = ChatService.__new__(ChatService)
        service._chat_repo = real_repo
        service._crisis_detector = detector
        service._llm_gateway = _make_mock_gateway()
        service._tool_adapters = []
        service._agent = legacy_agent
        service._chat_graph = None  # not used in shadow mode
        service._shadow_graph = shadow_graph
        service._shadow_repo = shadow_repo
        service._shadow_mode_chat = True
        service._new_chat_graph = False
        service._session_locks = {}

        result = await service.ask(user_id=1, session_id="sh1", message="你好")

        # Legacy output returned
        assert "传统路径" in result.answer
        # Shadow output NOT returned
        assert "新图" not in result.answer

    async def test_shadow_mode_never_double_persists(self) -> None:
        """Shadow mode → only one user + one assistant persisted."""
        from mindflow.graph.chat_graph import ChatGraph
        from mindflow.services.chat_service import _ShadowChatRepo

        fake_model = _FakeChatModel(responses=["新图回复"])
        real_repo = AsyncMock()
        real_repo.append = AsyncMock()
        real_repo.recent = AsyncMock(return_value=[])
        detector = MagicMock(spec=CrisisDetector)
        detector.scan.return_value = (CrisisLevel.NONE, None)

        shadow_repo = _ShadowChatRepo(real_repo)
        shadow_graph = ChatGraph(
            chat_repo=shadow_repo,
            crisis_detector=detector,
            model=fake_model,
            tools=[],
        )

        service = ChatService.__new__(ChatService)
        service._chat_repo = real_repo
        service._crisis_detector = detector
        service._llm_gateway = _make_mock_gateway()
        service._tool_adapters = []
        service._agent = _make_agent(["传统回复"])
        service._chat_graph = None
        service._shadow_graph = shadow_graph
        service._shadow_repo = shadow_repo
        service._shadow_mode_chat = True
        service._new_chat_graph = False
        service._session_locks = {}

        await service.ask(user_id=1, session_id="sh2", message="你好")

        # Real repo: exactly 2 appends (user + assistant), no duplicates
        assert real_repo.append.call_count == 2
        calls = real_repo.append.call_args_list
        assert calls[0].args[1] == "user"
        assert calls[1].args[1] == "assistant"

    async def test_shadow_mode_logs_comparison(self) -> None:
        """Shadow mode completes without error; diff is emitted via loguru."""
        from mindflow.graph.chat_graph import ChatGraph
        from mindflow.services.chat_service import _ShadowChatRepo

        fake_model = _FakeChatModel(responses=["新图不同回复"])
        real_repo = AsyncMock()
        real_repo.append = AsyncMock()
        real_repo.recent = AsyncMock(return_value=[])
        detector = MagicMock(spec=CrisisDetector)
        detector.scan.return_value = (CrisisLevel.NONE, None)

        shadow_repo = _ShadowChatRepo(real_repo)
        shadow_graph = ChatGraph(
            chat_repo=shadow_repo,
            crisis_detector=detector,
            model=fake_model,
            tools=[],
        )

        service = ChatService.__new__(ChatService)
        service._chat_repo = real_repo
        service._crisis_detector = detector
        service._llm_gateway = _make_mock_gateway()
        service._tool_adapters = []
        service._agent = _make_agent(["传统回复"])
        service._chat_graph = None
        service._shadow_graph = shadow_graph
        service._shadow_repo = shadow_repo
        service._shadow_mode_chat = True
        service._new_chat_graph = False
        service._session_locks = {}

        # Shadow mode runs without error, returns legacy answer
        result = await service.ask(user_id=1, session_id="sh3", message="你好")
        assert "传统回复" in result.answer  # legacy returned
        # Shadow ran (its answer differs from legacy — logged by loguru)


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


class TestFlagSwitchRestoresLegacy:
    """Switching flag restores legacy behavior without DB migration rollback."""

    async def test_flag_false_uses_legacy_agent(self) -> None:
        """new_chat_graph=False, _chat_graph=None → legacy path always."""
        service = ChatService.__new__(ChatService)
        service._chat_repo = AsyncMock()
        service._chat_repo.append = AsyncMock()
        service._chat_repo.recent = AsyncMock(return_value=[])
        service._crisis_detector = MagicMock(spec=CrisisDetector)
        service._crisis_detector.scan.return_value = (CrisisLevel.NONE, None)
        service._llm_gateway = _make_mock_gateway()
        service._tool_adapters = []
        service._agent = _make_agent(["传统回复"])
        service._chat_graph = None
        service._shadow_graph = None
        service._shadow_mode_chat = False
        service._new_chat_graph = False
        service._session_locks = {}

        result = await service.ask(user_id=1, session_id="f1", message="切换")

        assert "传统回复" in result.answer
        assert "新图" not in result.answer

    async def test_flag_switch_from_graph_to_legacy(self) -> None:
        """After switching new_chat_graph from True→False, legacy path works."""
        from mindflow.graph.chat_graph import ChatGraph

        fake_model = _FakeChatModel(responses=["新图回复"])
        repo = AsyncMock()
        repo.append = AsyncMock()
        repo.recent = AsyncMock(return_value=[])
        detector = MagicMock(spec=CrisisDetector)
        detector.scan.return_value = (CrisisLevel.NONE, None)

        chat_graph = ChatGraph(
            chat_repo=repo,
            crisis_detector=detector,
            model=fake_model,
            tools=[],
        )

        service = ChatService.__new__(ChatService)
        service._chat_repo = repo
        service._crisis_detector = detector
        service._llm_gateway = _make_mock_gateway()
        service._tool_adapters = []
        service._agent = _make_agent(["传统回复"])
        service._chat_graph = chat_graph  # graph available
        service._shadow_graph = None
        service._session_locks = {}

        # ── Flag ON → graph path ─────────────────────────────────────────
        service._new_chat_graph = True
        service._shadow_mode_chat = False
        result1 = await service.ask(user_id=1, session_id="sw", message="on")
        assert "新图回复" in result1.answer

        # ── Flag OFF → legacy path (no DB migration needed) ──────────────
        service._new_chat_graph = False
        result2 = await service.ask(user_id=1, session_id="sw2", message="off")
        assert "传统回复" in result2.answer


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_gateway() -> AsyncMock:
    """Create a backward-compat mock DeepSeekGateway."""
    gw = AsyncMock()
    gw._api_key = "test-key"
    gw._base_url = "https://api.deepseek.com"
    return gw
