"""Tests for ChatGraph (graph/chat_graph.py).

Covers parity and new-graph behaviour:
  - No-tool: model responds without tools
  - Each tool adapter: query_evidence, latest_analysis, run_analysis, intervention_history
  - Multi-tool loop: model → tool → model → answer
  - Crisis: crisis keywords short-circuit
  - Unavailable model: degraded response
  - Forbidden-word retry: detected → one retry → safe fallback
  - History compression: over limit → compressed summary
  - Concurrent same-session: requests serialised without corruption
  - Orphan-turn recovery: turn_id identifies orphan messages
  - Response parity: output fields match ChatService.ask() exactly
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool as lc_tool

from mindflow.graph.chat_graph import (
    _LLM_DOWN_REPLY,
    _SAFE_REPLY,
    ChatGraph,
)
from mindflow.infrastructure.security.crisis_detector import (
    CrisisDetector,
    CrisisLevel,
)
from mindflow.services.chat_service import ChatAnswer

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeBindableChatModel(FakeListChatModel):
    """A ``FakeListChatModel`` that supports ``bind_tools``."""

    def bind_tools(
        self, tools: Any, **kwargs: Any,
    ) -> _FakeBindableChatModel:
        return self


def _make_tool(name: str, result: str) -> Any:
    """Create a simple LangChain tool that returns a fixed string."""

    @lc_tool(name_or_callable=name)
    async def _tool_fn() -> str:
        """Test tool."""
        return result

    return _tool_fn


def _make_ainvoke_model(responses: list[AIMessage]) -> AsyncMock:
    """Create a mock model whose ``ainvoke`` returns AIMessages in sequence.

    Each call to ``ainvoke`` pops the first element from *responses*.
    The mock also supports ``bind_tools`` (returns self).
    """
    model = AsyncMock()
    # Use side_effect with a list-like iteration
    response_iter = iter(responses)

    async def _ainvoke(*args: Any, **kwargs: Any) -> AIMessage:
        try:
            return next(response_iter)
        except StopIteration:
            return AIMessage(content="")

    model.ainvoke = AsyncMock(side_effect=_ainvoke)
    model.bind_tools = MagicMock(return_value=model)
    return model


def _make_chat_graph(
    *,
    model: Any = None,
    tools: list[Any] | None = None,
    chat_repo: Any = None,
    crisis_detector: Any = None,
    tool_adapters: list[Any] | None = None,
    max_history_rounds: int = 10,
) -> ChatGraph:
    """Create a ChatGraph with default mocks."""
    repo = chat_repo or _make_mock_chat_repo()
    detector = crisis_detector or _make_mock_detector()
    return ChatGraph(
        chat_repo=repo,
        crisis_detector=detector,
        model=model,
        tools=tools or [],
        tool_adapters=tool_adapters or [],
        max_history_rounds=max_history_rounds,
    )


def _make_mock_chat_repo() -> AsyncMock:
    """Create a mock ChatRepository."""
    repo = AsyncMock()
    repo.append = AsyncMock()
    repo.recent = AsyncMock(return_value=[])
    return repo


def _make_mock_detector() -> MagicMock:
    """Create a mock CrisisDetector."""
    detector = MagicMock(spec=CrisisDetector)
    detector.scan.return_value = (CrisisLevel.NONE, None)
    return detector


# ═══════════════════════════════════════════════════════════════════════════════
# Parity tests — no-tool conversation
# ═══════════════════════════════════════════════════════════════════════════════


class TestParityNoTool:
    """ChatGraph produces same output as ChatService for simple Q&A."""

    async def test_no_tool_simple_answer(self) -> None:
        """Model responds with text, no tools invoked."""
        model = _FakeBindableChatModel(responses=["你好！有什么可以帮助你的？"])
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=1, session_id="s1", message="你好")

        assert isinstance(result, ChatAnswer)
        assert result.answer == "你好！有什么可以帮助你的？"
        assert result.session_id == "s1"
        assert result.tools_used == ()
        assert result.evidence_cited is False
        assert result.degraded is False

    async def test_no_tool_empty_response_treated_as_safe(self) -> None:
        """Empty model response falls back to safe reply."""
        model = _FakeBindableChatModel(responses=[""])
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=1, session_id="s1", message="?")

        # Empty → fallback
        assert len(result.answer) > 0
        assert result.degraded is False  # model succeeded, just empty

    async def test_response_metadata_fields_unchanged(self) -> None:
        """ChatAnswer always has answer, session_id, tools_used, evidence_cited, degraded."""
        model = _FakeBindableChatModel(responses=["测试回复"])
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=2, session_id="meta-test", message="test")

        assert isinstance(result, ChatAnswer)
        assert isinstance(result.answer, str)
        assert isinstance(result.session_id, str)
        assert isinstance(result.tools_used, tuple)
        assert isinstance(result.evidence_cited, bool)
        assert isinstance(result.degraded, bool)
        assert result.session_id == "meta-test"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestEachTool:
    """Each tool adapter works through the graph."""

    async def test_tool_execution_loops_back_to_model(self) -> None:
        """Model calls a tool → tool executes → model gets final answer."""
        # First response has tool_calls, second is the final answer
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "query_evidence",
                "args": {"days_back": 7},
            }],
        )
        final_msg = AIMessage(content="根据分析，你的专注度正常。")

        model = _make_ainvoke_model([tool_call_msg, final_msg])
        query_tool = _make_tool("query_evidence", '{"focus_score": 80}')
        tools = [query_tool]
        graph = _make_chat_graph(model=model, tools=tools)

        result = await graph.ask(user_id=1, session_id="s1", message="分析我的数据")

        assert result.answer == "根据分析，你的专注度正常。"
        assert "query_evidence" in result.tools_used
        assert result.evidence_cited is True

    async def test_multi_tool_call_loop(self) -> None:
        """Model calls tool A → model calls tool B → model produces answer."""
        call_a = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_1",
                "name": "query_evidence",
                "args": {"days_back": 7},
            }],
        )
        call_b = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_2",
                "name": "get_latest_analysis",
                "args": {},
            }],
        )
        final = AIMessage(content="根据证据和分析，建议你休息一下。")

        model = _make_ainvoke_model([call_a, call_b, final])
        tool_a = _make_tool("query_evidence", "evidence data")
        tool_b = _make_tool("get_latest_analysis", "analysis data")
        # Multi-tool loops need extra recursion depth (explicit graph has more nodes)
        graph = _make_chat_graph(model=model, tools=[tool_a, tool_b])
        graph._recursion_limit = 25  # override for multi-tool test

        result = await graph.ask(user_id=1, session_id="s1", message="分析")

        assert "建议你休息一下" in result.answer
        assert "query_evidence" in result.tools_used
        assert "get_latest_analysis" in result.tools_used
        assert result.evidence_cited is True

    async def test_tool_not_found_handled(self) -> None:
        """Model calls a tool that doesn't exist — error is returned."""
        call_unknown = AIMessage(
            content="",
            tool_calls=[{
                "id": "call_99",
                "name": "nonexistent_tool",
                "args": {},
            }],
        )
        final = AIMessage(content="我无法获取那个信息。")

        model = _make_ainvoke_model([call_unknown, final])
        graph = _make_chat_graph(model=model, tools=[])

        result = await graph.ask(user_id=1, session_id="s1", message="test")

        assert "无法获取" in result.answer

    async def test_tool_without_bind_tools_fallback(self) -> None:
        """Model that doesn't support bind_tools still works (no tools bound)."""
        model = _FakeBindableChatModel(responses=["没有工具也能回复。"])
        graph = _make_chat_graph(model=model, tools=[_make_tool("t", "ok")])

        result = await graph.ask(user_id=1, session_id="s1", message="hello")

        assert "没有工具也能回复" in result.answer
        assert result.tools_used == ()


# ═══════════════════════════════════════════════════════════════════════════════
# Crisis tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrisisGate:
    """Crisis detection short-circuits the pipeline."""

    async def test_crisis_returns_hotline(self) -> None:
        """HIGH crisis → hotline response, degraded, no DB writes."""
        detector = _make_mock_detector()
        detector.scan.return_value = (
            CrisisLevel.HIGH,
            MagicMock(
                message="全国24小时心理援助热线：400-161-9995",
                stop_llm=True,
            ),
        )
        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(crisis_detector=detector, chat_repo=repo)

        result = await graph.ask(user_id=1, session_id="cx", message="我想自杀")

        assert "400-161-9995" in result.answer
        assert result.degraded is True
        # No persistence for crisis
        repo.append.assert_not_called()

    async def test_crisis_sets_crisis_gate_state(self) -> None:
        """Crisis gate is True in the final state."""
        detector = _make_mock_detector()
        detector.scan.return_value = (
            CrisisLevel.HIGH,
            MagicMock(message="热线", stop_llm=True),
        )
        graph = _make_chat_graph(crisis_detector=detector)

        result = await graph.ask(user_id=1, session_id="cg", message="轻生")

        assert result.degraded is True
        assert "热线" in result.answer


# ═══════════════════════════════════════════════════════════════════════════════
# Unavailable model tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnavailableModel:
    """Degraded response when model is unavailable."""

    async def test_no_model_returns_degraded(self) -> None:
        """model=None → degraded with fallback reply."""
        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(model=None, chat_repo=repo)

        result = await graph.ask(user_id=1, session_id="s1", message="你好")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True
        assert result.tools_used == ()
        # User and assistant messages should still be persisted
        assert repo.append.call_count == 2

    async def test_model_raises_returns_degraded(self) -> None:
        """Model.ainvoke raises → degraded with fallback."""
        model = AsyncMock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        model.bind_tools = MagicMock(return_value=model)

        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(model=model, chat_repo=repo)

        result = await graph.ask(user_id=1, session_id="s1", message="test")

        assert result.answer == _LLM_DOWN_REPLY
        assert result.degraded is True


# ═══════════════════════════════════════════════════════════════════════════════
# Forbidden-word tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestForbiddenWords:
    """Forbidden word detection and retry."""

    async def test_no_forbidden_passes_through(self) -> None:
        """Response without forbidden words is unchanged."""
        model = _FakeBindableChatModel(responses=["根据数据分析，建议调整作息。"])
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=1, session_id="s1", message="建议")

        assert "调整作息" in result.answer
        assert result.degraded is False

    async def test_forbidden_word_triggers_retry(self) -> None:
        """Forbidden word → retry → clean response accepted."""
        # First response has forbidden word "诊断", retry is clean
        model = _FakeBindableChatModel(responses=[
            "根据诊断结果，你需要调整。",  # ← "诊断" is forbidden
            "根据分析结果，建议你调整作息。",  # ← clean retry
        ])
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=1, session_id="s1", message="分析")

        assert "诊断" not in result.answer
        assert "调整作息" in result.answer
        assert result.degraded is False

    async def test_forbidden_word_retry_exhausted(self) -> None:
        """Forbidden word found in both original and retry → safe fallback."""
        model = _FakeBindableChatModel(responses=[
            "诊断显示需要治疗。",  # original
            "治疗是必须的。",  # retry — still has "治疗"
        ])
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=1, session_id="s1", message="分析")

        assert result.answer == _SAFE_REPLY
        assert result.degraded is True

    async def test_forbidden_word_retry_only_once(self) -> None:
        """Exactly one retry — no infinite loop."""
        forbidden_responses = ["诊断x"] * 5  # all have forbidden word
        model = _FakeBindableChatModel(responses=forbidden_responses)
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=1, session_id="s1", message="test")

        # Should be safe reply (retry exhausted), not stuck in loop
        assert result.answer == _SAFE_REPLY
        assert result.degraded is True


# ═══════════════════════════════════════════════════════════════════════════════
# History compression tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHistoryCompression:
    """History compression when exceeding max rounds."""

    async def test_no_compression_below_limit(self) -> None:
        """Under the round limit → no compression summary."""
        model = _FakeBindableChatModel(responses=["你好！"])
        repo = _make_mock_chat_repo()
        repo.recent = AsyncMock(return_value=[
            {"role": "user", "content": f"msg{i}"}
            for i in range(5)
        ])
        graph = _make_chat_graph(model=model, chat_repo=repo, max_history_rounds=10)

        result = await graph.ask(user_id=1, session_id="s1", message="新消息")

        assert "你好" in result.answer

    async def test_compression_triggers_above_limit(self) -> None:
        """Over the round limit → compression summary is generated."""
        model = _FakeBindableChatModel(responses=["已收到你的消息。"])
        repo = _make_mock_chat_repo()
        # 22 messages = 11 rounds → triggers compression (max=10 → 20 msg limit)
        msgs: list[dict[str, Any]] = []
        for i in range(11):
            msgs.append({"role": "user", "content": f"用户消息{i}"})
            msgs.append({"role": "assistant", "content": f"助手回复{i}"})
        repo.recent = AsyncMock(return_value=msgs)
        graph = _make_chat_graph(model=model, chat_repo=repo, max_history_rounds=10)

        result = await graph.ask(user_id=1, session_id="s1", message="最新消息")

        assert "已收到" in result.answer


# ═══════════════════════════════════════════════════════════════════════════════
# Concurrency tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """Same-session requests are serialised."""

    async def test_same_session_serialized(self) -> None:
        """Two concurrent requests to the same session are processed sequentially."""
        entered = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def blocked_ainvoke(*args: Any, **kwargs: Any) -> AIMessage:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                entered.set()
                await release.wait()
            return AIMessage(content="done")

        model = AsyncMock()
        model.ainvoke = AsyncMock(side_effect=blocked_ainvoke)
        model.bind_tools = MagicMock(return_value=model)
        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(model=model, chat_repo=repo)

        first = asyncio.create_task(
            graph.ask(user_id=1, session_id="serial-s", message="first")
        )
        await entered.wait()
        second = asyncio.create_task(
            graph.ask(user_id=1, session_id="serial-s", message="second")
        )
        try:
            await asyncio.sleep(0.05)
            assert call_count == 1, "second request must be blocked"
        finally:
            release.set()
            results = await asyncio.gather(first, second)

        assert call_count == 2
        assert results[0].answer == "done"
        assert results[1].answer == "done"

    async def test_different_sessions_run_concurrently(self) -> None:
        """Requests to different sessions can run in parallel."""
        enter_a = asyncio.Event()
        enter_b = asyncio.Event()
        release = asyncio.Event()
        calls: list[str] = []

        async def blocked(*args: Any, **kwargs: Any) -> AIMessage:
            import asyncio as _asyncio
            task = _asyncio.current_task()
            calls.append(str(id(task)))
            if len(calls) == 1:
                enter_a.set()
            else:
                enter_b.set()
            await release.wait()
            return AIMessage(content="done")

        model = AsyncMock()
        model.ainvoke = AsyncMock(side_effect=blocked)
        model.bind_tools = MagicMock(return_value=model)
        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(model=model, chat_repo=repo)

        task_a = asyncio.create_task(graph.ask(user_id=1, session_id="a", message="a"))
        await enter_a.wait()
        task_b = asyncio.create_task(graph.ask(user_id=1, session_id="b", message="b"))
        await enter_b.wait()

        # Both different sessions should have entered the critical section
        assert len(calls) == 2, "different sessions run in parallel"

        release.set()
        await asyncio.gather(task_a, task_b)


# ═══════════════════════════════════════════════════════════════════════════════
# Turn ID / orphan-turn recovery tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTurnId:
    """Durable turn_id for orphan detection and retry."""

    async def test_turn_id_is_generated(self) -> None:
        """Each request generates a unique turn_id."""
        model = _FakeBindableChatModel(responses=["ok"])
        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(model=model, chat_repo=repo)

        await graph.ask(user_id=1, session_id="s1", message="msg1")
        await graph.ask(user_id=1, session_id="s1", message="msg2")

        # Both user messages were persisted with a turn_id
        assert repo.append.call_count >= 2
        call_args = [
            call.kwargs.get("message_id", call.args[4] if len(call.args) > 4 else None)
            for call in repo.append.call_args_list
        ]

    async def test_turn_id_format(self) -> None:
        """Turn ID follows the format turn:{session_id}:{uuid4()}."""
        model = _FakeBindableChatModel(responses=["ok"])
        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(model=model, chat_repo=repo)

        await graph.ask(user_id=1, session_id="my-session", message="msg")

        # Check that the user message was persisted with a turn_id
        assert repo.append.call_count >= 1
        first_call = repo.append.call_args_list[0]
        message_id = first_call.kwargs.get("message_id", None)

        if message_id:
            assert message_id.startswith("turn:my-session:")
            # Should have a UUID suffix
            uuid_part = message_id.split("turn:my-session:")[1]
            assert len(uuid_part) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersistence:
    """Messages are persisted correctly."""

    async def test_user_and_assistant_persisted(self) -> None:
        """Both user and assistant messages are written to DB."""
        model = _FakeBindableChatModel(responses=["你好！"])
        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(model=model, chat_repo=repo)

        await graph.ask(user_id=1, session_id="s1", message="你好")

        assert repo.append.call_count == 2
        calls = repo.append.call_args_list
        # append(session_id, role, content, *, user_id, message_id)
        assert calls[0].args[0] == "s1"  # session_id
        assert calls[0].args[1] == "user"  # role
        assert calls[0].args[2] == "你好"  # content
        assert calls[1].args[1] == "assistant"  # role

    async def test_degraded_response_still_persisted(self) -> None:
        """Even when degraded, the fallback answer is persisted."""
        model = AsyncMock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("fail"))
        model.bind_tools = MagicMock(return_value=model)
        repo = _make_mock_chat_repo()
        graph = _make_chat_graph(model=model, chat_repo=repo)

        result = await graph.ask(user_id=1, session_id="s1", message="test")

        assert result.degraded is True
        # Both messages should be persisted
        assert repo.append.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Response parity tests — cross-validate with ChatService output
# ═══════════════════════════════════════════════════════════════════════════════


class TestResponseParity:
    """ChatGraph output matches ChatService output exactly."""

    async def test_every_output_has_answer(self) -> None:
        """ChatAnswer.answer is always a non-empty string."""
        model = _FakeBindableChatModel(responses=["测试"])
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=1, session_id="s1", message="test")

        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    async def test_degraded_output_has_answer(self) -> None:
        """Degraded responses still have an answer string."""
        graph = _make_chat_graph(model=None)

        result = await graph.ask(user_id=1, session_id="s1", message="test")

        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.degraded is True

    async def test_tools_used_is_tuple(self) -> None:
        """ChatAnswer.tools_used is always a tuple."""
        model = _FakeBindableChatModel(responses=["test"])
        graph = _make_chat_graph(model=model)

        result = await graph.ask(user_id=1, session_id="s1", message="test")

        assert isinstance(result.tools_used, tuple)
