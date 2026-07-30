"""ChatGraph — explicit LangGraph reproduction of ``create_agent`` behavior.

Replaces the ``create_agent``-backed ``ChatService._ask_serialized`` with
explicit graph nodes for full observability, checkpointing, and durable
turn tracking.

Pipeline nodes (in order):
  1. crisis_gate_node         — scan for crisis keywords, short-circuit
  2. user_message_persist_node — save user message to DB immediately
  3. history_load_node         — load recent messages from DB
  4. history_compress_node     — compress oldest rounds if over limit
  5. model_call_node            — invoke LLM with messages + tools
  6. tools_condition_router     — route based on tool_calls in response
  7. tool_execution_node        — execute tool via LangChain tools
  8. answer_extraction_node     — extract final text from LLM response
  9. forbidden_word_validation_node — check for forbidden medical terms
  10. correction_loop_node      — one retry if forbidden words found
  11. assistant_message_persist_node — save assistant response to DB

Design constraints:
  - Output metadata (ChatAnswer fields) MUST be identical to current
    ``ChatService.ask()`` output.
  - Does NOT mix ``AnalysisState`` into chat state.
  - Does NOT change frontend response metadata fields.
  - Uses typed tool adapters from ``mindflow.graph.tools`` (Todo 13).
  - Durable ``turn_id`` per user message: ``turn:{session_id}:{uuid4()}``.
  - Preserves recursion limit (12) and session serialisation.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from mindflow.agents.types import FORBIDDEN_WORDS

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_HISTORY_ROUNDS: int = 10
_RECURSION_LIMIT: int = 12

_EVIDENCE_TOOLS: frozenset[str] = frozenset({"query_evidence", "get_latest_analysis"})

CHAT_SYSTEM_PROMPT: str = (
    "你是 MindFlow 的 AI 助手，帮助用户分析专注力模式和拖延行为。"
    "\n\n"
    "【回答要求】\n"
    "- 使用中文\n"
    "- 根据用户的行为数据给出个性化建议\n"
    "- 引用具体证据，例如「根据你的行为数据……」\n"
    '- 禁止使用以下词汇：诊断、治疗、患者、处方\n'
    "- 友善、鼓励、具体"
)

_LLM_DOWN_REPLY: str = (
    "当前 AI 对话不可用，你可以查看今日报告 /api/v1/focus 了解你的专注情况。"
)

_SAFE_REPLY: str = (
    "我暂时无法回答这个问题，请稍后再试。"
    "你可以查看今日报告 /api/v1/focus 了解你的专注情况。"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime context — non-serializable dependencies
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ChatRunContext:
    """Live dependencies injected at graph invocation time.

    NOT stored in checkpointable state — LangGraph cannot serialise
    repository references, model clients, or tool adapters.
    """

    # ── Repositories ──
    chat_repo: Any = None  # ChatRepository

    # ── Crisis ──
    crisis_detector: Any = None  # CrisisDetector

    # ── LLM ──
    model: Any = None  # BaseChatModel | None (None = degraded)

    # ── Tools ──
    tools: list[Any] = field(default_factory=list)  # list[BaseTool]
    tool_adapters: list[Any] = field(default_factory=list)

    # ── Configuration ──
    max_history_rounds: int = _MAX_HISTORY_ROUNDS
    recursion_limit: int = _RECURSION_LIMIT

    # ── Session lock ──
    session_locks: dict[str, asyncio.Lock] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph state — checkpointable TypedDict
# ═══════════════════════════════════════════════════════════════════════════════


class ChatGraphState(TypedDict, total=False):
    """State flowing through the chat conversation graph.

    Fields matching ``ChatState`` (state.py Todo 3):
        messages: Conversation history as role/content dicts.
        tool_messages: Accumulated tool-call and tool-result records.
        errors: Unique error records keyed by error message.
        crisis_gate: True if pre-LLM crisis detection triggered.
        retry_count: Number of retry loops (forbidden word, tool error).
        graph_version: Schema version for state migration awareness.

    Additional fields for graph execution:
        user_id: The user identifier (input).
        session_id: The conversation session identifier (input).
        user_message: The raw user message text (input).
        turn_id: Durable per-turn UUID identifier.
        answer: The final assistant response text (output).
        degraded: True if the response fell back to rule-based reply (output).
        tools_used: Names of tools invoked during this turn (output).
        evidence_cited: True if evidence-gathering tools were used (output).
        runtime: Live dependencies (not checkpointed).
        history_summary: Compression summary string, or None.
        model_messages_raw: Accumulated LangChain message objects (transient).
    """

    # ── Input ──
    user_id: int
    session_id: str
    user_message: str
    turn_id: str

    # ── Runtime (not checkpointed) ──
    runtime: ChatRunContext

    # ── ChatState fields (Todo 3) ──
    messages: list[dict[str, object]]
    tool_messages: list[dict[str, str]]
    errors: list[dict[str, str]]
    crisis_gate: bool
    retry_count: int
    graph_version: int

    # ── Execution fields ──
    answer: str
    degraded: bool
    tools_used: list[str]
    evidence_cited: bool
    history_summary: str | None
    model_messages_raw: list[Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: forbidden word check
# ═══════════════════════════════════════════════════════════════════════════════


def _contains_forbidden_words(text: str) -> str | None:
    """Return the first forbidden word found in *text*, or None."""
    for word in FORBIDDEN_WORDS:
        if word in text:
            return word
    return None


def _compress_history(
    history: list[dict[str, Any]],
    max_history_rounds: int = _MAX_HISTORY_ROUNDS,
) -> str | None:
    """Compress oldest conversation rounds into a text summary.

    When the history exceeds *max_history_rounds* rounds, the earliest
    messages are summarised. The summary passes through the forbidden-word
    check (replacing matches with ``***``).
    """
    max_messages = max_history_rounds * 2
    if len(history) <= max_messages:
        return None

    extra = len(history) - max_messages
    to_compress = history[:extra]

    parts: list[str] = ["之前的对话摘要:"]
    for msg in to_compress:
        label = "用户" if msg.get("role") == "user" else "AI助手"
        text = str(msg.get("content", ""))[:200]
        parts.append(f"[{label}]: {text}")

    summary = "\n".join(parts)

    for word in FORBIDDEN_WORDS:
        if word in summary:
            summary = summary.replace(word, "***")

    return summary


def _extract_answer_from_messages(messages: list[Any]) -> str:
    """Extract the final answer text from a list of LangChain messages.

    Returns the content of the last AI message, or the hardcoded fallback.
    """
    if not messages:
        return _LLM_DOWN_REPLY

    # Walk backwards to find the last non-tool-response AI message
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if content and isinstance(content, str) and content.strip():
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role in ("ai", "assistant"):
                return str(content)

    # Fallback: return content of the very last message
    last = messages[-1]
    content = getattr(last, "content", "") if hasattr(last, "content") else str(last)
    return str(content) if content else _LLM_DOWN_REPLY


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1: crisis_gate_node
# ═══════════════════════════════════════════════════════════════════════════════


async def crisis_gate_node(state: ChatGraphState) -> dict[str, Any]:
    """Scan the user message for crisis keywords.

    When a HIGH crisis level is detected, short-circuit the entire
    pipeline — return the crisis hotline information, set ``degraded=True``,
    and set ``crisis_gate=True`` so downstream nodes know to skip.
    """
    from mindflow.infrastructure.security.crisis_detector import CrisisLevel

    runtime: ChatRunContext = state.get("runtime", ChatRunContext())  # type: ignore[arg-type]
    user_message = state.get("user_message", "")
    user_id = state.get("user_id", 0)
    session_id = state.get("session_id", "")

    crisis_level, crisis_response = runtime.crisis_detector.scan(user_message)

    if crisis_level == CrisisLevel.HIGH and crisis_response is not None:
        logger.warning(
            "ChatGraph: Crisis detected in chat message, user_id={}", user_id
        )
        return {
            "crisis_gate": True,
            "answer": crisis_response.message,
            "degraded": True,
            "session_id": session_id,
            "tools_used": [],
            "evidence_cited": False,
        }

    return {"crisis_gate": False}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2: user_message_persist_node
# ═══════════════════════════════════════════════════════════════════════════════


async def user_message_persist_node(state: ChatGraphState) -> dict[str, Any]:
    """Persist the incoming user message to the DB immediately.

    This ensures no user message is lost even if the LLM call fails later.
    The message is persisted with the durable ``turn_id`` for orphan detection.
    """
    runtime: ChatRunContext = state.get("runtime", ChatRunContext())  # type: ignore[arg-type]
    session_id = state["session_id"]
    user_message = state["user_message"]
    user_id = state.get("user_id", 0)
    turn_id = state.get("turn_id", "")

    await runtime.chat_repo.append(
        session_id, "user", user_message, user_id=user_id,
        message_id=turn_id if turn_id else None,
    )

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3: history_load_node
# ═══════════════════════════════════════════════════════════════════════════════


async def history_load_node(state: ChatGraphState) -> dict[str, Any]:
    """Load recent conversation messages from the DB.

    Fetches up to ``max_history_rounds * 2 + 2`` messages so the
    compression node has enough context.
    """
    runtime: ChatRunContext = state.get("runtime", ChatRunContext())  # type: ignore[arg-type]
    session_id = state["session_id"]
    max_rounds = runtime.max_history_rounds

    history = await runtime.chat_repo.recent(
        session_id, limit=max_rounds * 2 + 2,
    )

    return {"messages": history}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4: history_compress_node
# ═══════════════════════════════════════════════════════════════════════════════


async def history_compress_node(state: ChatGraphState) -> dict[str, Any]:
    """Compress oldest conversation rounds into a summary if over limit.

    When the loaded history exceeds ``max_history_rounds`` rounds, the
    earliest messages are summarised and stored in ``history_summary``.
    The summary is sanitised for forbidden words.
    """
    runtime: ChatRunContext = state.get("runtime", ChatRunContext())  # type: ignore[arg-type]
    history = state.get("messages", [])

    summary = _compress_history(list(history), runtime.max_history_rounds)
    return {"history_summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5: model_call_node
# ═══════════════════════════════════════════════════════════════════════════════


async def model_call_node(state: ChatGraphState) -> dict[str, Any]:
    """Invoke the LLM with the current message history + tools.

    When ``model_messages_raw`` is already populated (after tool execution),
    the accumulated messages are reused.  Otherwise, a fresh message list
    is built from the DB history and current user message.

    When the model is unavailable (None), sets ``degraded=True`` and
    returns the fallback reply.
    """

    runtime: ChatRunContext = state.get("runtime", ChatRunContext())  # type: ignore[arg-type]
    model = runtime.model
    tools = runtime.tools

    # ── Degraded path: no model ──────────────────────────────────────────
    if model is None:
        return {
            "answer": _LLM_DOWN_REPLY,
            "degraded": True,
            "tools_used": [],
            "evidence_cited": False,
        }

    # ── Build or reuse LangChain message list ──────────────────────────────
    existing_raw: list[Any] = list(state.get("model_messages_raw", []))
    if existing_raw:
        # Continue from previous state (after tool execution or correction)
        messages: list[Any] = existing_raw
    else:
        # Fresh build from DB history
        messages = _build_messages_from_state(state)

    # ── Bind tools to model ───────────────────────────────────────────────
    bound_model = model
    if tools:
        try:
            bound_model = model.bind_tools(tools)
        except (AttributeError, NotImplementedError):
            # Model doesn't support tool binding — proceed without tools
            pass

    # ── Invoke LLM ─────────────────────────────────────────────────────────
    try:
        result = await bound_model.ainvoke(messages)
    except Exception as exc:
        logger.warning("ChatGraph: Model invocation failed: {}", exc)
        return {
            "answer": _LLM_DOWN_REPLY,
            "degraded": True,
            "tools_used": [],
            "evidence_cited": False,
        }

    # ── Collect tool call info ─────────────────────────────────────────────
    messages.append(result)
    tools_used: list[str] = list(state.get("tools_used", []))
    evidence_cited = state.get("evidence_cited", False)

    tool_call_objects = getattr(result, "tool_calls", None) or []
    for tc in tool_call_objects:
        t_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        if t_name:
            if t_name not in tools_used:
                tools_used.append(t_name)
            if t_name in _EVIDENCE_TOOLS:
                evidence_cited = True

        # Record as tool message
        tool_msg: dict[str, str] = {
            "type": "call",
            "name": t_name,
            "content": str(tc.get("args", "")) if isinstance(tc, dict) else str(getattr(tc, "args", "")),
        }
        existing_tool_msgs: list[dict[str, str]] = list(state.get("tool_messages", []))
        existing_tool_msgs.append(tool_msg)

        return {
            "model_messages_raw": messages,
            "tool_messages": existing_tool_msgs,
            "tools_used": tools_used,
            "evidence_cited": evidence_cited,
        }

    # No tool calls — this is the final answer
    return {
        "model_messages_raw": messages,
        "tools_used": tools_used,
        "evidence_cited": evidence_cited,
    }


def _build_messages_from_state(state: ChatGraphState) -> list[Any]:
    """Build LangChain message list from DB history and current user message."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    history = state.get("messages", [])
    user_message = state["user_message"]
    summary = state.get("history_summary")

    messages: list[Any] = []
    if summary:
        messages.append(SystemMessage(content=summary))

    for msg in history:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # Ensure current user message is the last user message
    if not any(
        isinstance(m, HumanMessage) and m.content == user_message
        for m in messages
    ):
        messages.append(HumanMessage(content=user_message))

    return messages


# ═══════════════════════════════════════════════════════════════════════════════
# Router 6: tools_condition_router
# ═══════════════════════════════════════════════════════════════════════════════


def tools_condition_router(state: ChatGraphState) -> Literal["tool_execution", "answer_extraction"]:
    """Route based on whether the last model response contains tool_calls.

    When tool_calls are present, route to ``tool_execution_node``.
    Otherwise, route to ``answer_extraction_node``.
    """
    # Check if already degraded — skip tool execution
    if state.get("degraded", False):
        return "answer_extraction"

    model_messages = state.get("model_messages_raw", [])
    if not model_messages:
        return "answer_extraction"

    last_msg = model_messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if tool_calls:
        return "tool_execution"

    return "answer_extraction"


# ═══════════════════════════════════════════════════════════════════════════════
# Node 7: tool_execution_node
# ═══════════════════════════════════════════════════════════════════════════════


async def tool_execution_node(state: ChatGraphState) -> dict[str, Any]:
    """Execute the tool calls requested by the model.

    Uses LangChain tool invocation — each tool is called with its
    arguments, and the results are added to the message history.
    After execution, the flow loops back to ``model_call_node`` so the
    model can process the tool results.

    Recursion limit (12) is enforced by the graph config — after 12
    tool/model cycles, the graph engine will raise a RecursionError.
    """
    from langchain_core.messages import ToolMessage

    runtime: ChatRunContext = state.get("runtime", ChatRunContext())  # type: ignore[arg-type]
    model_messages: list[Any] = list(state.get("model_messages_raw", []))
    if not model_messages:
        return {}

    last_msg = model_messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if not tool_calls:
        return {}

    # Build tool name → callable map
    tool_map: dict[str, Any] = {}
    for tool in runtime.tools:
        tool_map[tool.name] = tool

    tool_msgs: list[dict[str, str]] = list(state.get("tool_messages", []))

    for tc in tool_calls:
        t_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        t_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        t_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

        tool_fn = tool_map.get(t_name)
        if tool_fn is None:
            result_text = f"Tool '{t_name}' not found"
        else:
            try:
                result = await tool_fn.ainvoke(t_args)
                result_text = str(result) if result is not None else ""
            except Exception as exc:
                result_text = f"Tool error: {exc}"

        tool_msg = ToolMessage(content=result_text, tool_call_id=t_id, name=t_name)
        model_messages.append(tool_msg)

        tool_msgs.append({
            "type": "result",
            "name": t_name,
            "content": result_text,
        })

    return {
        "model_messages_raw": model_messages,
        "tool_messages": tool_msgs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 8: answer_extraction_node
# ═══════════════════════════════════════════════════════════════════════════════


async def answer_extraction_node(state: ChatGraphState) -> dict[str, Any]:
    """Extract the final answer text from the model's messages.

    Searches backward through the accumulated messages for the last
    assistant/AI message with non-empty content.
    """
    model_messages = state.get("model_messages_raw", [])
    answer = _extract_answer_from_messages(model_messages)

    return {"answer": answer}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 9: forbidden_word_validation_node
# ═══════════════════════════════════════════════════════════════════════════════


async def forbidden_word_validation_node(state: ChatGraphState) -> dict[str, Any]:
    """Check the extracted answer for forbidden medical terms.

    When a forbidden word is found, sets the state so the
    ``correction_loop_router`` can decide whether to retry or fall back.
    If already degraded (e.g. LLM down), skip this check.
    """
    if state.get("degraded", False):
        return {}

    answer = state.get("answer", "")
    bad_word = _contains_forbidden_words(answer)

    if bad_word is not None:
        logger.warning("ChatGraph: Forbidden word '{}' found in response", bad_word)
        return {
            "errors": list(state.get("errors", [])) + [{
                "key": f"forbidden_{bad_word}",
                "message": f"Output contained forbidden word: {bad_word}",
            }],
        }

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Router 10: correction_loop_router
# ═══════════════════════════════════════════════════════════════════════════════


def correction_loop_router(
    state: ChatGraphState,
) -> Literal["correction_loop", "assistant_message_persist"]:
    """Route after forbidden word check.

    If the latest error is a forbidden-word error AND retry_count < 1,
    route to ``correction_loop_node`` for one retry.
    Otherwise, proceed to persistence.
    """
    errors = state.get("errors", [])

    # Check if the latest error is a forbidden-word error
    has_forbidden = False
    for err in reversed(errors):
        key = err.get("key", "")
        if key.startswith("forbidden_"):
            has_forbidden = True
            break

    retry_count = state.get("retry_count", 0)

    if has_forbidden and retry_count < 1:
        return "correction_loop"

    return "assistant_message_persist"


# ═══════════════════════════════════════════════════════════════════════════════
# Node 10 (retry): correction_loop_node
# ═══════════════════════════════════════════════════════════════════════════════


async def correction_loop_node(state: ChatGraphState) -> dict[str, Any]:
    """One retry when forbidden words are detected.

    Appends a correction instruction to the message history and invokes
    the model again.  If the retry still contains forbidden words, the
    answer is replaced with the safe fallback reply and ``degraded`` is set.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    runtime: ChatRunContext = state.get("runtime", ChatRunContext())  # type: ignore[arg-type]
    model = runtime.model
    answer = state.get("answer", "")

    # Find the first forbidden word for the correction message
    bad_word = _contains_forbidden_words(answer)
    correction_text = (
        f"回答包含禁用词汇「{bad_word or '某些'}」，"
        "请用中文重新回答，不要使用诊断、治疗、患者、处方等词汇。"
    )

    # Build retry messages: original messages + correction instruction
    history = state.get("messages", [])
    user_message = state["user_message"]
    summary = state.get("history_summary")

    retry_messages: list[Any] = []
    if summary:
        retry_messages.append(SystemMessage(content=summary))

    for msg in history:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))
        if role == "user":
            retry_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            retry_messages.append(AIMessage(content=content))

    # Ensure current user message is last user message
    if not any(
        isinstance(m, HumanMessage) and m.content == user_message
        for m in retry_messages
    ):
        retry_messages.append(HumanMessage(content=user_message))

    # Append correction instruction
    retry_messages.append(SystemMessage(content=correction_text))

    retry_count = state.get("retry_count", 0) + 1

    if model is None:
        return {
            "answer": _SAFE_REPLY,
            "degraded": True,
            "retry_count": retry_count,
        }

    try:
        result = await model.ainvoke(retry_messages)
        retry_answer = _extract_answer_from_messages([result])

        if _contains_forbidden_words(retry_answer) is not None:
            return {
                "answer": _SAFE_REPLY,
                "degraded": True,
                "retry_count": retry_count,
            }

        return {
            "answer": retry_answer,
            "retry_count": retry_count,
        }

    except Exception as exc:
        logger.warning("ChatGraph: Correction retry failed: {}", exc)
        return {
            "answer": _SAFE_REPLY,
            "degraded": True,
            "retry_count": retry_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 11: assistant_message_persist_node
# ═══════════════════════════════════════════════════════════════════════════════


async def assistant_message_persist_node(state: ChatGraphState) -> dict[str, Any]:
    """Persist the assistant's final answer to the DB.

    Always persists, even when ``degraded=True`` — the user should
    always see a response.
    """
    runtime: ChatRunContext = state.get("runtime", ChatRunContext())  # type: ignore[arg-type]
    session_id = state["session_id"]
    answer = state.get("answer", _SAFE_REPLY)
    user_id = state.get("user_id", 0)

    await runtime.chat_repo.append(
        session_id, "assistant", answer, user_id=user_id,
    )

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Crisis router — after crisis gate, decide whether to short-circuit
# ═══════════════════════════════════════════════════════════════════════════════


def crisis_router(state: ChatGraphState) -> Literal["__end__", "user_message_persist"]:
    """Route: crisis_gate=True → END, normal → user_message_persist."""
    if state.get("crisis_gate", False):
        return "__end__"
    return "user_message_persist"


# ═══════════════════════════════════════════════════════════════════════════════
# ChatGraph — graph definition class
# ═══════════════════════════════════════════════════════════════════════════════


class ChatGraph:
    """Explicit LangGraph reproduction of ``create_agent`` chat behaviour.

    Builds a ``StateGraph`` with 11 explicit nodes that mirror the
    ``ChatService._ask_serialized`` pipeline, enabling full observability,
    checkpointing, and durable turn tracking.

    Args:
        chat_repo: Chat message repository for persistence.
        crisis_detector: Pre-LLM crisis keyword scanner.
        model: Bound LangChain chat model (or None for degraded mode).
        tools: LangChain tool list (from ``langchain_tools.py`` factories).
        tool_adapters: Typed tool adapters (from ``mindflow.graph.tools``).
        max_history_rounds: Max conversation rounds kept verbatim.
        recursion_limit: Max model/tool loop depth.
    """

    def __init__(  # noqa: PLR0913
        self,
        chat_repo: Any,
        crisis_detector: Any,
        model: Any = None,
        tools: list[Any] | None = None,
        tool_adapters: list[Any] | None = None,
        max_history_rounds: int = _MAX_HISTORY_ROUNDS,
        recursion_limit: int = _RECURSION_LIMIT,
    ) -> None:
        self._chat_repo = chat_repo
        self._crisis_detector = crisis_detector
        self._model = model
        self._tools = tools or []
        self._tool_adapters = tool_adapters or []
        self._max_history_rounds = max_history_rounds
        self._recursion_limit = recursion_limit
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._compiled: CompiledStateGraph[Any, Any, Any, Any] | None = None

    # ── Public API ──────────────────────────────────────────────────────

    async def ask(
        self,
        user_id: int,
        session_id: str,
        message: str,
    ) -> Any:
        """Process a user message and return a ChatAnswer.

        The response metadata is IDENTICAL to ``ChatService.ask()``:
        ``ChatAnswer(answer, session_id, tools_used, evidence_cited, degraded)``.

        Args:
            user_id: The user identifier.
            session_id: The conversation session identifier.
            message: The user's text message.

        Returns:
            A ``ChatAnswer`` with the response and metadata.
        """


        # ── Generate durable turn_id ────────────────────────────────────
        turn_id = f"turn:{session_id}:{uuid.uuid4()}"

        # ── Serialise per-session access ──────────────────────────────────
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            return await self._ask_serialized(
                user_id, session_id, message, turn_id,
            )

    async def _ask_serialized(
        self,
        user_id: int,
        session_id: str,
        message: str,
        turn_id: str,
    ) -> Any:
        """Run the chat graph while holding the session lock."""
        from mindflow.graph.tools import ToolContext
        from mindflow.services.chat_service import ChatAnswer

        # ── Build runtime context ────────────────────────────────────────
        runtime = ChatRunContext(
            chat_repo=self._chat_repo,
            crisis_detector=self._crisis_detector,
            model=self._model,
            tools=self._tools,
            tool_adapters=self._tool_adapters,
            max_history_rounds=self._max_history_rounds,
            recursion_limit=self._recursion_limit,
            session_locks=self._session_locks,
        )

        # ── Set ToolContext on adapters ────────────────────────────────────
        ctx = ToolContext(
            user_id=user_id,
            session_id=session_id,
            run_id=turn_id,
        )
        for adapter in self._tool_adapters:
            adapter.context = ctx

        try:
            # ── Build initial state ───────────────────────────────────────
            initial_state: ChatGraphState = {
                "user_id": user_id,
                "session_id": session_id,
                "user_message": message,
                "turn_id": turn_id,
                "runtime": runtime,
                "messages": [],
                "tool_messages": [],
                "errors": [],
                "crisis_gate": False,
                "retry_count": 0,
                "graph_version": 1,
                "answer": "",
                "degraded": False,
                "tools_used": [],
                "evidence_cited": False,
                "history_summary": None,
                "model_messages_raw": [],
            }

            # ── Run the graph ─────────────────────────────────────────────
            graph = self._get_compiled_graph()
            final_state = await graph.ainvoke(
                initial_state,
                config={"recursion_limit": self._recursion_limit},
            )

            # ── Normalise output ──────────────────────────────────────────
            if isinstance(final_state, dict):
                answer = final_state.get("answer", _LLM_DOWN_REPLY)
                tools_used: tuple[str, ...] = tuple(final_state.get("tools_used", []))
                evidence_cited = final_state.get("evidence_cited", False)
                degraded = final_state.get("degraded", False)
            else:
                answer = _LLM_DOWN_REPLY
                tools_used = ()
                evidence_cited = False
                degraded = True

            return ChatAnswer(
                answer=answer,
                session_id=session_id,
                tools_used=tools_used,
                evidence_cited=evidence_cited,
                degraded=degraded,
            )

        except Exception as exc:
            logger.warning("ChatGraph invocation failed: {}", exc)
            return ChatAnswer(
                answer=_LLM_DOWN_REPLY,
                session_id=session_id,
                degraded=True,
            )

        finally:
            # Clear context on adapters after invocation
            for adapter in self._tool_adapters:
                adapter.context = None

    # ── Graph construction ───────────────────────────────────────────────

    def _get_compiled_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        """Build and compile the LangGraph StateGraph once (lazy init)."""
        if self._compiled is not None:
            return self._compiled

        builder = StateGraph(ChatGraphState)

        # ── Add nodes ────────────────────────────────────────────────────
        builder.add_node("crisis_gate", crisis_gate_node)
        builder.add_node("user_message_persist", user_message_persist_node)
        builder.add_node("history_load", history_load_node)
        builder.add_node("history_compress", history_compress_node)
        builder.add_node("model_call", model_call_node)
        builder.add_node("tool_execution", tool_execution_node)
        builder.add_node("answer_extraction", answer_extraction_node)
        builder.add_node("forbidden_word_validation", forbidden_word_validation_node)
        builder.add_node("correction_loop", correction_loop_node)
        builder.add_node("assistant_message_persist", assistant_message_persist_node)

        # ── Add edges and conditional routes ──────────────────────────────

        # START → crisis_gate
        builder.set_entry_point("crisis_gate")

        # crisis_gate → [crisis? → END | normal → user_message_persist]
        builder.add_conditional_edges(
            "crisis_gate",
            crisis_router,
            {
                "__end__": END,
                "user_message_persist": "user_message_persist",
            },
        )

        # user_message_persist → history_load → history_compress → model_call
        builder.add_edge("user_message_persist", "history_load")
        builder.add_edge("history_load", "history_compress")
        builder.add_edge("history_compress", "model_call")

        # model_call → [tool_calls? → tool_execution | no → answer_extraction]
        builder.add_conditional_edges(
            "model_call",
            tools_condition_router,
            {
                "tool_execution": "tool_execution",
                "answer_extraction": "answer_extraction",
            },
        )

        # tool_execution → model_call (loop back for next model pass)
        builder.add_edge("tool_execution", "model_call")

        # answer_extraction → forbidden_word_validation
        builder.add_edge("answer_extraction", "forbidden_word_validation")

        # forbidden_word_validation → [forbidden? → correction_loop | clean → assistant_message_persist]
        builder.add_conditional_edges(
            "forbidden_word_validation",
            correction_loop_router,
            {
                "correction_loop": "correction_loop",
                "assistant_message_persist": "assistant_message_persist",
            },
        )

        # correction_loop → assistant_message_persist (one retry only)
        builder.add_edge("correction_loop", "assistant_message_persist")

        # assistant_message_persist → END
        builder.add_edge("assistant_message_persist", END)

        self._compiled = builder.compile()
        return self._compiled
