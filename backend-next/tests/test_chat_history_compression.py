"""Phase 1.4 regressions: the exact payload handed to the chat provider.

The prompt assembled on every turn must be exactly::

    system prompt
    + at most one history summary (only if old turns were compressed)
    + the last 4–6 complete turns, verbatim
    + the current user message

These tests assert on the real message list passed to the model (a recording
gateway), covering:

  - folded turns are never re-sent, on this or any later turn
  - a tool call and its tool result are always both present or both absent,
    including an assistant message carrying several parallel tool calls
  - the recent window keeps 4–6 verbatim turns
  - the summary stays exactly one bounded message across consecutive turns
  - folded history still passes the forbidden-word guard
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from mindflow.graph import chat_graph as chat_module
from mindflow.graph.chat_graph import (
    _RECENT_TURNS,
    _SUMMARY_HEADER,
    _SUMMARY_MAX_CHARS,
    CHAT_SYSTEM_PROMPT,
    ChatGraph,
)
from mindflow.infrastructure.security.crisis_detector import (
    CrisisDetector,
    CrisisLevel,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Doubles: an in-memory chat repo and a recording chat model
# ═══════════════════════════════════════════════════════════════════════════════

_USER_ID = 1
_SESSION_ID = "s1"


class _RecordingRepo:
    """ChatRepository double with the real id/ordering semantics."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._seq = 0

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        user_id: int = 1,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        self._seq += 1
        row: dict[str, Any] = {
            "id": message_id or f"row-{self._seq}",
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
        }
        self.rows.append(row)
        return dict(row)

    async def recent(
        self,
        session_id: str,
        limit: int = 20,
        *,
        user_id: int = 1,
    ) -> list[dict[str, Any]]:
        """Newest rows for the session, oldest-first (like the SQL repo)."""
        scoped = [
            row for row in self.rows
            if row["session_id"] == session_id and row["user_id"] == user_id
        ]
        return [dict(row) for row in scoped[-limit:]]


class _RecordingModel:
    """Chat model double that records every payload it is invoked with."""

    def __init__(self, replies: list[AIMessage] | None = None) -> None:
        self.payloads: list[list[Any]] = []
        self._replies = list(replies or [])

    def bind_tools(self, tools: Any, **kwargs: Any) -> _RecordingModel:
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.payloads.append(list(messages))
        if self._replies:
            return self._replies.pop(0)
        return AIMessage(content=f"自动回复{len(self.payloads)}")


def _make_tool(name: str, result: str) -> Any:
    """Create a simple LangChain tool that returns a fixed string."""

    @lc_tool(name_or_callable=name)
    async def _tool_fn() -> str:
        """Test tool."""
        return result

    return _tool_fn


def _make_graph(
    model: _RecordingModel,
    repo: _RecordingRepo,
    *,
    tools: list[Any] | None = None,
) -> ChatGraph:
    detector = MagicMock(spec=CrisisDetector)
    detector.scan.return_value = (CrisisLevel.NONE, None)
    return ChatGraph(
        chat_repo=repo,
        crisis_detector=detector,
        model=model,
        tools=tools or [],
    )


def _summary_messages(payload: list[Any]) -> list[Any]:
    """Payload messages that are history summaries (not the system prompt)."""
    return [
        message for message in payload
        if isinstance(message, SystemMessage) and message.content != CHAT_SYSTEM_PROMPT
    ]


def _verbatim_turn_count(payload: list[Any]) -> int:
    """Complete turns sent verbatim, excluding the trailing current message."""
    return len([m for m in payload if isinstance(m, HumanMessage)]) - 1


def _message_contents(payload: list[Any]) -> list[str]:
    return [str(getattr(message, "content", "")) for message in payload]


# ═══════════════════════════════════════════════════════════════════════════════
# Payload shape
# ═══════════════════════════════════════════════════════════════════════════════


class TestPayloadShape:
    """The payload is system + ≤1 summary + recent turns + current message."""

    async def test_payload_is_summary_recent_turns_and_current_message(self) -> None:
        """A long conversation sends one summary, six turns, and the question."""
        model = _RecordingModel([AIMessage(content=f"回复{i}") for i in range(12)])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo)

        for i in range(12):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        payload = model.payloads[-1]

        assert isinstance(payload[0], SystemMessage)
        assert payload[0].content == CHAT_SYSTEM_PROMPT

        summaries = _summary_messages(payload)
        assert len(summaries) == 1
        assert str(summaries[0].content).startswith(_SUMMARY_HEADER)

        assert isinstance(payload[-1], HumanMessage)
        assert payload[-1].content == "问题11"
        assert _message_contents(payload).count("问题11") == 1

        # 11 completed turns: the last six are verbatim, five were folded.
        assert _verbatim_turn_count(payload) == _RECENT_TURNS

    async def test_folded_turns_are_absent_from_every_later_payload(self) -> None:
        """Once a turn is compressed it never reappears as a payload message."""
        model = _RecordingModel([AIMessage(content=f"回复{i}") for i in range(12)])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo)

        for i in range(12):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        for ask_index in range(_RECENT_TURNS + 1, 12):
            payload = model.payloads[ask_index]
            contents = _message_contents(payload)
            # At ask k there are k completed turns; those older than the
            # verbatim window are folded and must not be sent as messages.
            for folded_index in range(ask_index - _RECENT_TURNS):
                assert f"问题{folded_index}" not in contents
                assert f"回复{folded_index}" not in contents
            # The summary is the only place folded history survives.
            summary = "\n".join(
                str(m.content) for m in _summary_messages(payload)
            )
            assert f"问题{ask_index - _RECENT_TURNS - 1}" in summary

    async def test_no_summary_while_history_fits_the_window(self) -> None:
        """Short conversations get no summary message at all."""
        model = _RecordingModel([AIMessage(content=f"回复{i}") for i in range(4)])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo)

        for i in range(4):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        for payload in model.payloads:
            assert _summary_messages(payload) == []

        last = model.payloads[-1]
        assert _verbatim_turn_count(last) == 3  # turns 0-2 are complete

    async def test_recent_window_is_within_four_to_six_turns(self) -> None:
        """The named window constant is inside the required 4–6 range."""
        model = _RecordingModel([AIMessage(content=f"回复{i}") for i in range(20)])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo)

        for i in range(20):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        payload = model.payloads[-1]
        verbatim = _verbatim_turn_count(payload)
        assert 4 <= verbatim <= 6
        assert verbatim == _RECENT_TURNS

        # The verbatim turns are complete pairs, oldest-first, with no gaps.
        kept = list(range(20 - _RECENT_TURNS - 1, 19))
        expected_sequence: list[str] = []
        for index in kept:
            expected_sequence.extend([f"问题{index}", f"回复{index}"])
        expected_sequence.append("问题19")
        assert _message_contents(payload)[2:] == expected_sequence


# ═══════════════════════════════════════════════════════════════════════════════
# Summary boundedness
# ═══════════════════════════════════════════════════════════════════════════════


class TestSummaryBounded:
    """Re-summarising every turn stays a single bounded message."""

    async def test_summary_message_count_does_not_grow_across_turns(self) -> None:
        """Every turn after compression sends exactly one summary message."""
        model = _RecordingModel([AIMessage(content=f"回复{i}") for i in range(15)])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo)

        for i in range(15):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        for ask_index in range(_RECENT_TURNS + 1, 15):
            payload = model.payloads[ask_index]
            summaries = _summary_messages(payload)
            assert len(summaries) == 1
            content = str(summaries[0].content)
            # The summary is rewritten, never appended to itself.
            assert content.count(_SUMMARY_HEADER) == 1
            assert len(content) <= _SUMMARY_MAX_CHARS + len(_SUMMARY_HEADER) + 1
            # system + summary + six turns + current message, constant size.
            assert len(payload) == 2 + _RECENT_TURNS * 2 + 1

    async def test_summary_stays_bounded_with_long_folded_history(self) -> None:
        """A very long folded history cannot inflate the summary message."""
        model = _RecordingModel([
            AIMessage(content=f"回复{i}" + "细节" * 400) for i in range(12)
        ])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo)

        for i in range(12):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}" + "内容" * 400)

        payload = model.payloads[-1]
        summaries = _summary_messages(payload)
        assert len(summaries) == 1
        content = str(summaries[0].content)
        assert len(content) <= _SUMMARY_MAX_CHARS + len(_SUMMARY_HEADER) + 1

    async def test_folded_summary_sanitises_forbidden_words(self) -> None:
        """Folded history passes the forbidden-word guard before the model."""
        model = _RecordingModel([AIMessage(content=f"回复{i}") for i in range(10)])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo)

        for i in range(10):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}关于诊断与治疗")

        payload = model.payloads[-1]
        summaries = _summary_messages(payload)
        assert len(summaries) == 1
        content = str(summaries[0].content)
        assert "诊断" not in content
        assert "治疗" not in content
        assert "***" in content


# ═══════════════════════════════════════════════════════════════════════════════
# Tool protocol integrity
# ═══════════════════════════════════════════════════════════════════════════════


def _tool_call(call_id: str, name: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {}}])


class TestToolProtocolIntegrity:
    """Tool calls and their results are always sent as complete pairs."""

    async def test_tool_exchange_kept_complete_inside_the_window(self) -> None:
        """A tool turn in the recent window keeps call + result together."""
        model = _RecordingModel([
            _tool_call("call-1", "query_evidence"),
            AIMessage(content="工具回答"),
            *[AIMessage(content=f"回复{i}") for i in range(3)],
        ])
        repo = _RecordingRepo()
        graph = _make_graph(
            model, repo, tools=[_make_tool("query_evidence", '{"evidence":[]}')],
        )

        await graph.ask(_USER_ID, _SESSION_ID, "问题0")
        for i in range(1, 4):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        payload = model.payloads[-1]
        calls = [m for m in payload if getattr(m, "tool_calls", None)]
        results = [m for m in payload if isinstance(m, ToolMessage)]
        assert len(calls) == 1
        assert len(results) == 1
        assert results[0].tool_call_id == calls[0].tool_calls[0]["id"]

    async def test_folded_tool_turn_drops_call_and_result_together(self) -> None:
        """A folded tool turn leaves neither half of the protocol behind."""
        model = _RecordingModel([
            _tool_call("call-1", "query_evidence"),
            AIMessage(content="工具回答"),
            *[AIMessage(content=f"回复{i}") for i in range(8)],
        ])
        repo = _RecordingRepo()
        graph = _make_graph(
            model, repo, tools=[_make_tool("query_evidence", '{"evidence":[]}')],
        )

        await graph.ask(_USER_ID, _SESSION_ID, "问题0")
        for i in range(1, 9):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        payload = model.payloads[-1]
        assert not [m for m in payload if isinstance(m, ToolMessage)]
        assert not [m for m in payload if getattr(m, "tool_calls", None)]
        assert "问题0" not in _message_contents(payload)
        summary = _summary_messages(payload)[0]
        assert "问题0" in str(summary.content)

    async def test_parallel_tool_calls_keep_all_results_together(self) -> None:
        """An assistant message with N tool calls keeps all N results."""
        parallel = AIMessage(content="", tool_calls=[
            {"id": "call-a", "name": "query_evidence", "args": {}},
            {"id": "call-b", "name": "get_latest_analysis", "args": {}},
        ])
        model = _RecordingModel([
            parallel,
            AIMessage(content="合并回答"),
            *[AIMessage(content=f"回复{i}") for i in range(3)],
        ])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo, tools=[
            _make_tool("query_evidence", '{"evidence":[]}'),
            _make_tool("get_latest_analysis", '{"procrastination_types":[]}'),
        ])

        await graph.ask(_USER_ID, _SESSION_ID, "问题0")
        for i in range(1, 4):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        payload = model.payloads[-1]
        calls = [m for m in payload if getattr(m, "tool_calls", None)]
        results = [m for m in payload if isinstance(m, ToolMessage)]
        assert len(calls) == 1
        assert len(calls[0].tool_calls) == 2
        assert {result.tool_call_id for result in results} == {"call-a", "call-b"}

    async def test_folded_parallel_tool_turn_drops_every_half(self) -> None:
        """A folded parallel tool turn leaves no call and no result behind."""
        parallel = AIMessage(content="", tool_calls=[
            {"id": "call-a", "name": "query_evidence", "args": {}},
            {"id": "call-b", "name": "get_latest_analysis", "args": {}},
        ])
        model = _RecordingModel([
            parallel,
            AIMessage(content="合并回答"),
            *[AIMessage(content=f"回复{i}") for i in range(8)],
        ])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo, tools=[
            _make_tool("query_evidence", '{"evidence":[]}'),
            _make_tool("get_latest_analysis", '{"procrastination_types":[]}'),
        ])

        await graph.ask(_USER_ID, _SESSION_ID, "问题0")
        for i in range(1, 9):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        payload = model.payloads[-1]
        assert not [m for m in payload if isinstance(m, ToolMessage)]
        assert not [m for m in payload if getattr(m, "tool_calls", None)]

    async def test_incomplete_cached_protocol_is_dropped_whole(self) -> None:
        """A protocol missing one of its results never sends the other half."""
        parallel = AIMessage(content="", tool_calls=[
            {"id": "call-a", "name": "query_evidence", "args": {}},
            {"id": "call-b", "name": "get_latest_analysis", "args": {}},
        ])
        model = _RecordingModel([
            parallel,
            AIMessage(content="合并回答"),
            AIMessage(content="后续回复"),
        ])
        repo = _RecordingRepo()
        graph = _make_graph(model, repo, tools=[
            _make_tool("query_evidence", '{"evidence":[]}'),
            _make_tool("get_latest_analysis", '{"procrastination_types":[]}'),
        ])

        await graph.ask(_USER_ID, _SESSION_ID, "问题0")
        # Simulate a partially evicted exchange: one tool result is gone.
        cached = next(iter(graph._protocol_turns.values()))
        assert len([m for m in cached if isinstance(m, ToolMessage)]) == 2
        cached.pop(1)
        await graph.ask(_USER_ID, _SESSION_ID, "问题1")

        payload = model.payloads[-1]
        assert not [m for m in payload if isinstance(m, ToolMessage)]
        assert not [m for m in payload if getattr(m, "tool_calls", None)]
        # The durable final answer is still sent, so context is not lost.
        assert "合并回答" in _message_contents(payload)

    async def test_payload_json_has_no_orphan_tool_entries(self) -> None:
        """Every tool result in the payload answers a retained tool call."""
        model = _RecordingModel([
            _tool_call("call-1", "query_evidence"),
            AIMessage(content="工具回答"),
            *[AIMessage(content=f"回复{i}") for i in range(7)],
        ])
        repo = _RecordingRepo()
        graph = _make_graph(
            model, repo, tools=[_make_tool("query_evidence", '{"evidence":[]}')],
        )

        await graph.ask(_USER_ID, _SESSION_ID, "问题0")
        for i in range(1, 8):
            await graph.ask(_USER_ID, _SESSION_ID, f"问题{i}")

        for payload in model.payloads:
            open_calls = {
                call["id"]
                for message in payload
                for call in (getattr(message, "tool_calls", None) or [])
            }
            answered = {
                getattr(message, "tool_call_id", "")
                for message in payload if isinstance(message, ToolMessage)
            }
            assert answered <= open_calls
            assert json.dumps(_message_contents(payload), ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _row(role: str, content: str, row_id: str) -> dict[str, Any]:
    return {
        "id": row_id, "user_id": _USER_ID, "session_id": _SESSION_ID,
        "role": role, "content": content,
    }


class TestBoundaryHelpers:
    """The turn grouping and the compression boundary are exact."""

    def test_group_turns_keeps_assistant_rows_with_their_user_row(self) -> None:
        rows = [
            _row("user", "u0", "0u"),
            _row("assistant", "a0", "0a"),
            _row("assistant", "a0b", "0ab"),  # extra reply row stays in the turn
            _row("user", "u1", "1u"),
            _row("assistant", "a1", "1a"),
        ]

        turns = chat_module._group_turns(rows)

        assert [[row["content"] for row in turn] for turn in turns] == [
            ["u0", "a0", "a0b"],
            ["u1", "a1"],
        ]

    def test_group_turns_keeps_a_leading_assistant_row_addressable(self) -> None:
        rows = [_row("assistant", "orphan", "0a"), _row("user", "u1", "1u")]

        turns = chat_module._group_turns(rows)

        assert [[row["content"] for row in turn] for turn in turns] == [
            ["orphan"],
            ["u1"],
        ]

    def test_split_history_holds_out_the_current_turn_and_the_window(
        self,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for i in range(9):
            rows.append(_row("user", f"u{i}", f"{i}u"))
            rows.append(_row("assistant", f"a{i}", f"{i}a"))
        rows.append(_row("user", "question", "current"))

        folded, recent = chat_module._split_history({
            "runtime": chat_module.ChatRunContext(),
            "user_id": _USER_ID,
            "session_id": _SESSION_ID,
            "messages": rows,
            "user_message": "question",
            "turn_id": "current",
        })

        assert [[row["content"] for row in turn] for turn in recent] == [
            [f"u{i}", f"a{i}"] for i in range(3, 9)
        ]
        assert [[row["content"] for row in turn] for turn in folded] == [
            [f"u{i}", f"a{i}"] for i in range(3)
        ]
        # The halves are disjoint and cover the whole completed history.
        assert [turn[0]["content"] for turn in [*folded, *recent]] == [
            f"u{i}" for i in range(9)
        ]

    def test_zero_window_folds_every_completed_turn(
        self, monkeypatch: Any,
    ) -> None:
        """A degenerate window still folds instead of re-sending everything."""
        monkeypatch.setattr(chat_module, "_RECENT_TURNS", 0)
        rows = [
            _row("user", "u0", "0u"), _row("assistant", "a0", "0a"),
            _row("user", "u1", "1u"), _row("assistant", "a1", "1a"),
        ]

        folded, recent = chat_module._split_history({
            "runtime": chat_module.ChatRunContext(),
            "user_id": _USER_ID,
            "session_id": _SESSION_ID,
            "messages": rows,
            "user_message": "question",
            "turn_id": "current",
        })

        assert recent == []
        assert len(folded) == 2

    def test_incomplete_tool_exchange_is_detected(self) -> None:
        parallel = _tool_call("call-a", "query_evidence")
        parallel.tool_calls.append({"id": "call-b", "name": "other", "args": {}})
        complete = [
            parallel,
            ToolMessage(content="a", tool_call_id="call-a"),
            ToolMessage(content="b", tool_call_id="call-b"),
            AIMessage(content="done"),
        ]

        assert chat_module._is_complete_tool_exchange(complete)
        assert not chat_module._is_complete_tool_exchange(complete[:2])
        # A result for a call that was never sent is also incomplete.
        assert not chat_module._is_complete_tool_exchange([complete[1]])
