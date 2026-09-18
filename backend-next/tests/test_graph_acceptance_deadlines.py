"""A13 regressions: deterministic expiry, cancellation and private tool history."""

import asyncio
import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger
from sqlalchemy.exc import IntegrityError

from mindflow.agents.types import PanelUnavailableError
from mindflow.graph import analysis_graph as analysis
from mindflow.graph import chat_graph as chat
from mindflow.infrastructure.repositories.chat import ChatRepository
from mindflow.infrastructure.security.crisis_detector import CrisisLevel
from mindflow.ports import AnalysisRequest


def chat_fixture(model=None, tools=()):
    rows = []

    async def append(session, role, content, **kwargs):
        row = {
            "id": kwargs.get("message_id") or f"m{len(rows)}",
            "user_id": kwargs["user_id"],
            "session_id": session, "role": role, "content": content,
        }
        rows.append(row)
        return row

    async def recent(session, **kwargs):
        return [row for row in rows if row["session_id"] == session]

    repo = SimpleNamespace(append=AsyncMock(side_effect=append), recent=recent)
    detector = MagicMock()
    detector.scan.return_value = (CrisisLevel.NONE, None)
    graph = chat.ChatGraph(repo, detector, model=model, tools=list(tools), recursion_limit=30)
    return graph, rows


def model_fixture(callback):
    model = MagicMock()
    model.bind_tools.return_value = model
    model.ainvoke = AsyncMock(side_effect=callback)
    return model


class SyntheticProviderError(RuntimeError):
    status_code = 503


def private_provider_error(error_type=SyntheticProviderError):
    exc = error_type("SYNTHETIC_OPAQUE_KEY SYNTHETIC_PRIVATE_REASONING")
    exc.__cause__ = RuntimeError("SYNTHETIC_PRIVATE_REASONING")
    return exc


@pytest.fixture
def graph_error_logs():
    rendered, records = [], []

    def sink(message):
        rendered.append(str(message))
        records.append(message.record)

    handler = logger.add(sink, level="DEBUG")
    try:
        yield rendered, records
    finally:
        logger.remove(handler)


def assert_private_errors_absent(rendered, records, *outputs):
    text = "\n".join(rendered) + repr(outputs)
    assert "SYNTHETIC_OPAQUE_KEY" not in text
    assert "SYNTHETIC_PRIVATE_REASONING" not in text
    assert records
    assert all(record["exception"] is None for record in records)


@pytest.mark.parametrize("path", ["model", "correction", "outer"])
async def test_chat_provider_errors_are_allowlisted(path, graph_error_logs):
    error = private_provider_error()
    replies = [AIMessage(content="诊断"), error] if path == "correction" else [error]
    graph, rows = chat_fixture(model_fixture(replies))
    if path == "outer":
        graph._chat_repo.append.side_effect = error
    result = await graph.ask(1, "session", "question")
    rendered, records = graph_error_logs
    assert result.degraded
    assert_private_errors_absent(rendered, records, rows, result)
    assert any("type=SyntheticProviderError status=503" in line for line in rendered)


@pytest.mark.parametrize("unavailable", [False, True])
async def test_analysis_panel_errors_are_allowlisted(unavailable, graph_error_logs):
    error = private_provider_error(PanelUnavailableError if unavailable else SyntheticProviderError)
    runtime = analysis.AnalysisRunContext(
        panel_graph=SimpleNamespace(ainvoke=AsyncMock(side_effect=error)),
    )
    result = await analysis.panel_graph_node({
        "runtime": runtime, "user_id": 1, "target_date": date(2026, 9, 19),
        "bundle_json": "{}",
    })
    runtime.panel_graph.ainvoke.assert_awaited_once()
    assert result["panel_succeeded"] is False
    assert_private_errors_absent(*graph_error_logs, result)


async def test_analysis_outer_error_persistence_is_allowlisted(graph_error_logs):
    graph, runs, _ = analysis_fixture()
    graph._compiled = SimpleNamespace(ainvoke=AsyncMock(side_effect=private_provider_error()))
    result = await graph.run_analysis(AnalysisRequest(user_id=1, target_date=date(2026, 9, 19)))
    assert_private_errors_absent(*graph_error_logs, result, runs.update_status.call_args_list)
    assert runs.update_status.await_args.kwargs["error"] == "type=SyntheticProviderError status=503"


@pytest.mark.parametrize("module", [chat, analysis])
@pytest.mark.parametrize("status", ["SYNTHETIC_OPAQUE_KEY", True, None, 503])
def test_error_metadata_accepts_only_integer_status(module, status):
    error = private_provider_error()
    error.status_code = status
    text = module._safe_error_metadata(error)
    assert text == "type=SyntheticProviderError" + (" status=503" if type(status) is int else "")


def tool_call(name="query_evidence"):
    return AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "PRIVATE_PROTOCOL"},
        tool_calls=[{"id": "call-1", "name": name, "args": {}}],
    )


@pytest.mark.parametrize(("name", "payload", "expected"), [
    ("query_evidence", '{"error":"unavailable"}', False),
    ("query_evidence", "Tool error: unavailable", False),
    ("query_evidence", "暂无数据", False),
    ("query_evidence", "{}", False),
    ("query_evidence", '{"evidence":[]}', False),
    ("query_evidence", '{"evidence":[{"metric":"focus_score","value":0}]}', True),
    ("get_latest_analysis", "暂无分析数据", False),
    ("get_latest_analysis", '{"error":"failed","types":["impulsivity"]}', False),
    ("get_latest_analysis", '{"procrastination_types":[]}', False),
    ("get_latest_analysis", '{"procrastination_types":["impulsivity"]}', True),
])
async def test_evidence_requires_successful_structured_data(name, payload, expected):
    replies = iter([tool_call(name), AIMessage(content="Final answer")])
    model = model_fixture(lambda messages: next(replies))
    tool = SimpleNamespace(name=name, ainvoke=AsyncMock(return_value=payload))
    graph, _ = chat_fixture(model, [tool])
    result = await graph.ask(1, "session", "question")
    assert result.evidence_cited is expected


async def test_protocol_survives_next_turn_but_never_enters_display_history():
    captured = []
    replies = iter([
        tool_call(), AIMessage(content="First answer"), AIMessage(content="Second answer"),
    ])

    async def invoke(messages):
        captured.append(list(messages))
        return next(replies)

    tool = SimpleNamespace(
        name="query_evidence", ainvoke=AsyncMock(return_value='{"error":"offline"}'),
    )
    graph, rows = chat_fixture(model_fixture(invoke), [tool])
    await graph.ask(1, "session", "same question")
    await graph.ask(1, "session", "same question")
    followup = captured[-1]
    assert [m.type for m in followup] == ["system", "human", "ai", "tool", "ai", "human"]
    assert followup[2].additional_kwargs["reasoning_content"] == "PRIVATE_PROTOCOL"
    assert followup[3].tool_call_id == followup[2].tool_calls[0]["id"]
    assert "PRIVATE_PROTOCOL" not in json.dumps(rows)
    assert all(row["role"] in ("user", "assistant") for row in rows)
    assert len([m for m in followup if isinstance(m, HumanMessage)]) == 2
    graph._protocol_turns.clear()
    rebuilt = chat._build_messages_from_state({
        "runtime": chat.ChatRunContext(protocol_turns=graph._protocol_turns),
        "user_id": 1, "session_id": "session",
        "messages": rows, "user_message": "after restart", "turn_id": "new",
    })
    assert not any(isinstance(m, ToolMessage) or getattr(m, "tool_calls", []) for m in rebuilt)


async def test_correction_keeps_current_tool_protocol():
    captured = []
    replies = iter([tool_call(), AIMessage(content="诊断"), AIMessage(content="Safe answer")])

    async def invoke(messages):
        captured.append(list(messages))
        return next(replies)

    tool = SimpleNamespace(name="query_evidence", ainvoke=AsyncMock(return_value="{}"))
    graph, rows = chat_fixture(model_fixture(invoke), [tool])
    result = await graph.ask(1, "session", "question")
    assert result.answer == "Safe answer"
    assert any(isinstance(m, ToolMessage) for m in captured[-1])
    assert any(
        m.additional_kwargs.get("reasoning_content") == "PRIVATE_PROTOCOL"
        for m in captured[-1]
    )
    assert rows[-1]["content"] == "Safe answer"


async def test_real_db_ids_and_same_session_user_isolation(session_factory, create_tables):
    repo = ChatRepository(session_factory)
    captured = []
    replies = iter([
        tool_call(), AIMessage(content="USER_A_ANSWER"),
        AIMessage(content="USER_B_ANSWER"), AIMessage(content="USER_A_FOLLOWUP"),
    ])

    async def invoke(messages):
        captured.append(list(messages))
        return next(replies)

    tool = SimpleNamespace(name="query_evidence", ainvoke=AsyncMock(return_value="{}"))
    graph, _ = chat_fixture(model_fixture(invoke), [tool])
    graph._chat_repo = repo
    await graph.ask(101, "shared-session", "USER_A_QUESTION")
    a_rows = await repo.recent("shared-session", user_id=101)
    assistant = next(row for row in a_rows if row["role"] == "assistant")
    assert UUID(assistant["id"]).version == 7
    # The real primary key rejects reuse even by another user/session.
    with pytest.raises(IntegrityError):
        await repo.append(
            "other-session", "assistant", "duplicate", user_id=202,
            message_id=assistant["id"],
        )
    await graph.ask(202, "shared-session", "USER_B_QUESTION")
    b_prompt = captured[-1]
    assert not any("USER_A_" in str(m.content) for m in b_prompt)
    assert not any(m.additional_kwargs.get("reasoning_content") for m in b_prompt)
    assert not any(isinstance(m, ToolMessage) for m in b_prompt)
    await graph.ask(101, "shared-session", "USER_A_NEXT_QUESTION")
    a_prompt = captured[-1]
    assert not any("USER_B_" in str(m.content) for m in a_prompt)
    assert any(m.additional_kwargs.get("reasoning_content") == "PRIVATE_PROTOCOL" for m in a_prompt)
    key = (101, "shared-session", assistant["id"])
    assert key in graph._protocol_turns
    display_rows = [
        *await repo.recent("shared-session", user_id=101),
        *await repo.recent("shared-session", user_id=202),
    ]
    assert "PRIVATE_PROTOCOL" not in json.dumps(display_rows)


@pytest.mark.parametrize("other_scope", [(202, "session"), (101, "other-session")])
def test_protocol_cache_key_cannot_alias_another_scope(other_scope):
    runtime = chat.ChatRunContext()
    runtime.protocol_turns[(101, "session", "same-id")] = [
        tool_call(), ToolMessage(content="PRIVATE_TOOL", tool_call_id="call-1"),
        AIMessage(content="private answer"),
    ]
    owner = chat._build_messages_from_state({
        "runtime": runtime, "user_id": 101, "session_id": "session",
        "messages": [{
            "id": "same-id", "user_id": 101, "session_id": "session",
            "role": "assistant", "content": "public answer",
        }],
        "user_message": "new question", "turn_id": "new",
    })
    assert any(isinstance(m, ToolMessage) for m in owner)
    user_id, session_id = other_scope
    rebuilt = chat._build_messages_from_state({
        "runtime": runtime, "user_id": user_id, "session_id": session_id,
        "messages": [{
            "id": "same-id", "user_id": user_id, "session_id": session_id,
            "role": "assistant", "content": "public answer",
        }],
        "user_message": "new question", "turn_id": "new",
    })
    assert not any(isinstance(m, ToolMessage) for m in rebuilt)
    assert not any(m.additional_kwargs.get("reasoning_content") for m in rebuilt)
    assert rebuilt[1].content == "public answer"


async def test_history_is_scoped_before_summary_and_unowned_rows_are_dropped():
    rows = [
        {"user_id": 202, "session_id": "s", "role": "assistant", "content": "OTHER_USER"},
        {"user_id": 101, "session_id": "other", "role": "assistant", "content": "OTHER_SESSION"},
        {"session_id": "s", "role": "assistant", "content": "UNKNOWN_OWNER"},
        {"user_id": 101, "session_id": "s", "role": "user", "content": "OWN_HISTORY"},
    ]
    runtime = chat.ChatRunContext(chat_repo=SimpleNamespace(recent=AsyncMock(return_value=rows)))
    result = await chat.history_load_node({
        "runtime": runtime, "user_id": 101, "session_id": "s",
    })
    assert result["messages"] == [rows[-1]]


async def test_new_graph_after_restart_rebuilds_text_only_from_real_db(
    session_factory, create_tables,
):
    repo = ChatRepository(session_factory)
    replies = iter([tool_call(), AIMessage(content="Durable final answer")])
    tool = SimpleNamespace(name="query_evidence", ainvoke=AsyncMock(return_value="{}"))
    first, _ = chat_fixture(model_fixture(lambda messages: next(replies)), [tool])
    first._chat_repo = repo
    await first.ask(101, "session", "Durable question")
    assert first._protocol_turns

    captured = []

    async def invoke(messages):
        captured.extend(messages)
        return AIMessage(content="After restart answer")

    restarted, _ = chat_fixture(model_fixture(invoke), [tool])
    restarted._chat_repo = repo
    assert not restarted._protocol_turns
    await restarted.ask(101, "session", "After restart question")
    assert any(m.content == "Durable final answer" for m in captured)
    assert any(m.content == "Durable question" for m in captured)
    assert not any(isinstance(m, ToolMessage) or getattr(m, "tool_calls", []) for m in captured)
    assert not any(m.additional_kwargs.get("reasoning_content") for m in captured)
    assert "PRIVATE_PROTOCOL" not in json.dumps(await repo.recent("session"))


@pytest.fixture
def deadlines(monkeypatch):
    """Expose real asyncio deadlines so tests can expire them without waiting."""
    original = asyncio.timeout
    captured = []

    def timeout(delay):
        timer = original(delay)
        captured.append((delay, timer))
        return timer

    monkeypatch.setattr(asyncio, "timeout", timeout)
    return captured


@pytest.mark.parametrize("expire_in", ["tool", "second_model"])
async def test_chat_whole_turn_deadline_cancels_current_await(deadlines, expire_in):
    cancelled = []
    calls = 0

    async def expire():
        assert deadlines[0][0] == 600
        deadlines[0][1].reschedule(asyncio.get_running_loop().time() - 1)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    async def invoke(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            return tool_call()
        await expire()

    async def execute(args):
        if expire_in == "tool":
            await expire()
        return "{}"

    tool = SimpleNamespace(name="query_evidence", ainvoke=execute)
    graph, rows = chat_fixture(model_fixture(invoke), [tool])
    result = await graph.ask(1, "session", "question")
    assert result.degraded and not result.evidence_cited
    assert cancelled == [True]
    assert len(deadlines) == 1
    assert all(row["role"] == "user" for row in rows)
    assert not graph._session_locks[(1, "session")].locked()


async def test_chat_external_cancel_is_not_swallowed():
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def invoke(messages):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    graph, _ = chat_fixture(model_fixture(invoke))
    task = asyncio.create_task(graph.ask(1, "session", "question"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert not graph._session_locks[(1, "session")].locked()


def analysis_fixture():
    runs, budget = AsyncMock(), AsyncMock()
    runs.save_run.return_value = "run"
    graph = analysis.AnalysisGraph(AsyncMock(), runs, budget, AsyncMock(), MagicMock())
    return graph, runs, budget


@pytest.mark.parametrize("external_cancel", [False, True])
async def test_late_panel_failure_fallback_uses_remaining_deadline(
    deadlines, monkeypatch, external_cancel,
):
    graph, runs, budget = analysis_fixture()
    cancelled = asyncio.Event()
    entered = asyncio.Event()
    ollama = AsyncMock()
    rule = AsyncMock()
    monkeypatch.setattr(analysis, "ollama_node", ollama)
    monkeypatch.setattr(analysis, "rule_engine_node", rule)

    async def fail_panel(state):
        raise PanelUnavailableError("late failure")

    async def fallback(payload):
        entered.set()
        if not external_cancel:
            deadlines[0][1].reschedule(asyncio.get_running_loop().time() - 1)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    graph._panel_graph = SimpleNamespace(ainvoke=fail_panel)
    graph._deepseek_client = SimpleNamespace(analyze=fallback)

    async def invoke(state, config=None):
        runtime = analysis._runtime_of(state)
        runtime.budget_owned = True
        state.update(await analysis.panel_graph_node(state))
        state["summary_json"] = "{}"
        assert deadlines[0][0] == 600
        return await analysis._fallback_chain_node(state)

    graph._compiled = SimpleNamespace(ainvoke=invoke)
    request = AnalysisRequest(user_id=1, target_date=date(2026, 9, 19), idempotency_key="key")
    task = asyncio.create_task(graph.run_analysis(request))
    await entered.wait()
    if external_cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if external_cancel else TimeoutError):
        await task
    assert cancelled.is_set()
    budget.release.assert_awaited_once_with("key")
    assert runs.update_status.await_args.args[1] == "failed"
    ollama.assert_not_awaited()
    rule.assert_not_awaited()


async def test_analysis_cancel_does_not_release_another_runs_budget():
    graph, _, budget = analysis_fixture()
    budget.try_reserve.return_value = False

    async def invoke(state, config=None):
        raise asyncio.CancelledError

    graph._compiled = SimpleNamespace(ainvoke=invoke)
    with pytest.raises(asyncio.CancelledError):
        await graph.run_analysis(AnalysisRequest(user_id=1, target_date=date(2026, 9, 19)))
    budget.release.assert_not_awaited()
