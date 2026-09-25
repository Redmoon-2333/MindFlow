"""ChatGraph — explicit LangGraph reproduction of ``create_agent`` behavior.

Replaces the ``create_agent``-backed ``ChatService._ask_serialized`` with
explicit graph nodes for full observability, checkpointing, and durable
turn tracking.

Pipeline nodes (in order):
  1. crisis_gate_node         — scan for crisis keywords, short-circuit
  2. user_message_persist_node — save user message to DB immediately
  3. history_load_node         — load recent messages from DB
  4. history_compress_node     — fold turns older than the verbatim window
                                 into one bounded summary message
  5. model_call_node            — invoke LLM with messages + tools
  6. tools_condition_router     — route based on tool_calls in response
  7. tool_execution_node        — execute tool via LangChain tools
  8. answer_extraction_node     — extract final text from LLM response
  9. forbidden_word_validation_node — check for forbidden medical terms
  10. correction_loop_node      — one retry if forbidden words found
  11. assistant_message_persist_node — save assistant response to DB

Payload contract (Phase 1.4): every model call receives exactly
``system prompt + at most one history summary + the last _RECENT_TURNS
complete turns verbatim + the current user message``.  Folded turns are never
re-sent, and tool calls always travel with all of their tool results.

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
import json
import math
import time
import uuid
from collections import OrderedDict
from contextlib import suppress
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
_CHAT_TURN_TIMEOUT_S: float = 600.0
_MAX_PROTOCOL_TURNS: int = 128

# ── Observability labels (optimisation plan 2.5) ─────────────────────────────
# Chat is its own graph, and each of its two generation calls is its own node,
# so the panel's per-role/per-node aggregates stay distinguishable from chat's.
_CHAT_GRAPH_LABEL: str = "chat"
_MODEL_CALL_NODE: str = "model_call"
_CORRECTION_NODE: str = "correction_loop"

# ── History window (Phase 1.4) ──────────────────────────────────────────────
# The payload handed to the provider on every turn is exactly:
#
#     system prompt
#     + at most one history summary (only if older turns were compressed)
#     + the last _RECENT_TURNS complete turns, verbatim
#     + the current user message
#
# ``_RECENT_TURNS`` is the verbatim window: the plan fixes it in the 4–6 range
# and 6 keeps the widest recent context the token budget allows without
# re-sending anything that was already folded into the summary.  A *turn* is
# one user message plus the assistant's complete reply (including every tool
# call and tool result); it is the atomic unit of both the verbatim window and
# the compression boundary.
_RECENT_TURNS: int = 6
# The folded summary is a single message, rebuilt from scratch every turn.
# These caps keep it bounded (and stable in size) no matter how long the
# conversation runs.
_SUMMARY_HEADER: str = "之前的对话摘要:"
_SUMMARY_MAX_CHARS: int = 1200
_SUMMARY_ITEM_MAX_CHARS: int = 200


def _safe_error_metadata(exc: Exception) -> str:
    """Never include provider response bodies or exception chains."""
    status = getattr(exc, "status_code", None)
    suffix = f" status={status}" if type(status) is int else ""
    return f"type={type(exc).__name__}{suffix}"


def _status_class(status: object) -> str:
    """Map a provider status code to its ``2xx``/``4xx``/``5xx`` class."""
    if type(status) is not int:
        return ""
    return f"{status // 100}xx"


def _error_category(exc: Exception) -> str:
    """Sanitized transport category — never the message, body, or traceback.

    Values match the vocabulary the HTTP layer already emits
    (``timeout``/``transport``), so chat failures aggregate with panel ones.
    """
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "transport"


def _is_budget_refusal(exc: Exception) -> bool:
    """True when the exception is a caller-imposed request-budget refusal.

    Matched by type *name* so this module never imports the harness that owns
    the budget. A refused attempt is still recorded (it happened) but must not
    be counted as a request that reached the provider.
    """
    return type(exc).__name__ in {"BudgetRefusedError", "PanelBudgetExceededError"}


def _token_counts(response: object) -> tuple[int, int, int]:
    """Read provider-reported ``(input, output, reasoning)`` token counts.

    ``usage_metadata`` is the LangChain v1 field; older or proxied providers
    only populate ``response_metadata["token_usage"]``. Reasoning tokens arrive
    under ``output_token_details`` (OpenAI-compatible) or
    ``completion_tokens_details``. Every count is ``0`` when the provider omits
    usage entirely — a missing number is never estimated from text length.
    """
    raw: Any = getattr(response, "usage_metadata", None)
    if not isinstance(raw, dict):
        metadata = getattr(response, "response_metadata", None)
        raw = None
        if isinstance(metadata, dict):
            raw = metadata.get("token_usage") or metadata.get("usage")
    if not isinstance(raw, dict):
        return (0, 0, 0)

    def _count(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0
        return max(0, int(value))

    input_tokens = _count(raw.get("input_tokens")) or _count(raw.get("prompt_tokens"))
    output_tokens = _count(raw.get("output_tokens")) or _count(
        raw.get("completion_tokens"),
    )
    reasoning_tokens = 0
    details = raw.get("output_token_details")
    if isinstance(details, dict):
        reasoning_tokens = _count(details.get("reasoning"))
    if not reasoning_tokens:
        reasoning_tokens = _count(raw.get("reasoning_tokens"))
    if not reasoning_tokens:
        completion_details = raw.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning_tokens = _count(completion_details.get("reasoning_tokens"))
    return (input_tokens, output_tokens, reasoning_tokens)


def _model_label(model: object) -> str:
    """Best-effort model id for a client that may be a mock (never raises)."""
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, "")
        if isinstance(value, str) and value:
            return value
    return ""


def _record_chat_call(  # noqa: PLR0913 - one record per measured call, by design
    runtime: ChatRunContext,
    *,
    node: str,
    model: object,
    response: object = None,
    total_latency_ms: float = 0.0,
    retry_count: int = 0,
    ok: bool = True,
    error: Exception | None = None,
) -> None:
    """Emit one aggregated observability record for a chat generation call.

    Same shape and same privacy contract as the panel/attribution path: labels,
    latency, status, retry count and provider-reported token usage — and nothing
    that could hold a prompt, an API key, a completion, or reasoning content.
    Total latency is the measured wall time of the call; the HTTP share is the
    same span, because a single ``ainvoke`` *is* the HTTP request here (any
    tool-calling rounds are separate calls and get their own records).

    A caller-imposed budget refusal happens *before* the transport, so it is
    recorded with zero latency: the attempt is visible, but it is an event, not
    a request that reached the provider.
    """
    from mindflow.agents.policies import ROLE_CHAT  # noqa: PLC0415
    from mindflow.services.llm_observability import (  # noqa: PLC0415
        LLMRequestRecord,
        record_llm_request,
    )

    refused = error is not None and _is_budget_refusal(error)
    total_ms = 0.0 if refused else max(0.0, float(total_latency_ms))
    input_tokens, output_tokens, reasoning_tokens = _token_counts(response)
    status_class = _status_class(getattr(error, "status_code", None))
    if not status_class and error is None:
        # A ``None``-raising ``ainvoke`` only ever returns a provider response,
        # so the call ended in a 2xx even though the message object does not
        # carry the status code.
        status_class = "2xx"
    record_llm_request(LLMRequestRecord(
        graph=_CHAT_GRAPH_LABEL,
        node=node,
        role=ROLE_CHAT,
        provider=runtime.provider,
        model=_model_label(model),
        queue_latency_ms=0.0,
        # One ``ainvoke`` is one HTTP request here, so the HTTP share is the
        # whole measured span (never more than the total it is part of).
        http_latency_ms=total_ms,
        total_latency_ms=total_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        retry_count=max(0, retry_count),
        http_status_class=status_class,
        error_category="" if refused else (
            _error_category(error) if error is not None else ""
        ),
        fallback_reason="request_budget_refused" if refused else "",
        ok=ok,
    ))


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
    "当前 AI 对话不可用，你可以查看「专注分析」页面了解你的专注情况。"
)

_SAFE_REPLY: str = (
    "我暂时无法回答这个问题，请稍后再试。"
    "你可以查看「专注分析」页面了解你的专注情况。"
)


def _provider_label(provider: str) -> str:
    """Resolve the provider label recorded on chat calls.

    An explicit label wins (the composition root knows the registry it built);
    otherwise the **resolved L1 target** decides, using the same vocabulary as
    ``LangChainGateway._provider_label`` so panel and chat records for the same
    provider agree: ``"ecnu"`` for the compat campus gateway, ``"deepseek"`` for
    the production DeepSeek pin. Deriving it from the legacy ``is_ecnu`` sniffer
    would label DeepSeek traffic "ecnu" whenever an old ECNU ``.env`` is still
    present. An unavailable settings object yields an empty label, never a guess.
    """
    if provider:
        return provider
    try:
        from mindflow.config import get_settings  # noqa: PLC0415

        return "ecnu" if get_settings().llm.l1_target().is_ecnu else "deepseek"
    except Exception:  # noqa: BLE001 - a label must never break a chat turn
        return ""


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

    # ── Observability ──
    #: Provider label recorded on every chat call ("ecnu" / "generic" / …).
    #: Injected by the composition root; resolved from settings otherwise.
    provider: str = ""

    # ── Tools ──
    tools: list[Any] = field(default_factory=list)  # list[BaseTool]
    tool_adapters: list[Any] = field(default_factory=list)

    # ── Configuration ──
    max_history_rounds: int = _MAX_HISTORY_ROUNDS
    recursion_limit: int = _RECURSION_LIMIT

    # ── Session lock ──
    session_locks: dict[tuple[int, str], asyncio.Lock] = field(default_factory=dict)
    # Private complete assistant/tool exchanges, never written to display history.
    protocol_turns: OrderedDict[tuple[int, str, str], list[Any]] = field(
        default_factory=OrderedDict,
    )


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
        history_summary: The single folded-history summary message (or None).
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
# Helpers: forbidden word check + history window
# ═══════════════════════════════════════════════════════════════════════════════


def _contains_forbidden_words(text: str) -> str | None:
    """Return the first forbidden word found in *text*, or None."""
    for word in FORBIDDEN_WORDS:
        if word in text:
            return word
    return None


def _scoped_rows(state: ChatGraphState) -> list[dict[str, Any]]:
    """Return this user's/session's history rows, oldest-first.

    ``history_load_node`` already scopes the rows it loads, but the message
    builder and the compressor can also run on directly supplied state or on
    injected repositories, so ownership is re-checked here.
    """
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    return [
        row for row in state.get("messages", [])
        if row.get("user_id") == user_id and row.get("session_id") == session_id
    ]


def _group_turns(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group oldest-first history rows into turns.

    A turn starts at every user row and owns all following assistant rows, so
    a turn is exactly "one user message plus the assistant's full reply".  The
    grouping never splits a tool protocol: an assistant row and every tool
    result belonging to it stay inside the same turn, and the compression
    boundary only ever moves between turns.
    """
    turns: list[list[dict[str, Any]]] = []
    for row in rows:
        if not turns or row.get("role") == "user":
            turns.append([row])
        else:
            turns[-1].append(row)
    return turns


def _split_history(
    state: ChatGraphState,
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Split history into ``(folded_turns, recent_turns)``.

    This is the single compression boundary shared by the compressor and the
    message builder, so a turn can never be summarised *and* re-sent.

    The current user message is persisted before the model call, so its row is
    the newest one; it is held out of the verbatim window and re-sent as the
    final ``HumanMessage`` instead.  ``folded_turns`` is everything older than
    the last ``_RECENT_TURNS`` complete turns — the only source of the
    summary — and those turns never appear in the payload again.
    """
    turns = _group_turns(_scoped_rows(state))

    turn_id = str(state.get("turn_id", ""))
    if turns and turn_id and str(turns[-1][-1].get("id", "")) == turn_id:
        turns = turns[:-1]

    if _RECENT_TURNS <= 0:
        return turns, []
    return turns[:-_RECENT_TURNS], turns[-_RECENT_TURNS:]


def _is_complete_tool_exchange(exchange: list[Any]) -> bool:
    """True when every ``assistant(tool_calls)`` has all of its tool results.

    Parallel calls are checked individually: an assistant message carrying N
    tool calls only counts as complete once all N matching ``ToolMessage``s
    have been seen.  A partial exchange returns False so the caller drops the
    whole group instead of sending half a protocol pair.
    """
    pending: list[str] = []
    for msg in exchange:
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            if pending:
                return False
            pending = [
                str(call.get("id", "") if isinstance(call, dict) else getattr(call, "id", ""))
                for call in calls
            ]
            if any(not call_id for call_id in pending):
                return False
            continue
        if getattr(msg, "type", None) == "tool":
            call_id = str(getattr(msg, "tool_call_id", ""))
            if call_id not in pending:
                return False
            pending.remove(call_id)
            continue
        if pending:
            # A non-tool message between a call and its results breaks the pair.
            return False
    return not pending


def _render_turn(
    turn: list[dict[str, Any]],
    runtime: ChatRunContext,
    user_id: int,
    session_id: str,
) -> list[Any]:
    """Render one verbatim turn as LangChain messages.

    An assistant row with a complete cached protocol exchange is expanded into
    ``assistant(tool_calls) → tool(result)… → assistant(final)``.  When the
    exchange is missing or incomplete, the whole exchange is dropped and the
    durable final answer is used instead — never half of a pair.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    rendered: list[Any] = []
    for row in turn:
        content = str(row.get("content", ""))
        if row.get("role") == "user":
            rendered.append(HumanMessage(content=content))
            continue
        protocol = runtime.protocol_turns.get(
            (user_id, session_id, str(row.get("id", ""))),
        )
        if protocol and _is_complete_tool_exchange(protocol):
            rendered.extend(protocol)
        else:
            # Eviction/restart drops the whole protocol exchange, not half of
            # a tool call. The durable final answer remains usable.
            rendered.append(AIMessage(content=content))
    return rendered


def _summarise_turns(turns: list[list[dict[str, Any]]]) -> str | None:
    """Fold *turns* into a single bounded summary message, or None if empty.

    The summary is rebuilt from scratch on every turn — it is never appended
    to a previous summary — so consecutive turns always produce exactly one
    summary message of bounded size (``_SUMMARY_MAX_CHARS``) instead of a
    growing list.  The newest folded turns are kept first, because when the
    cap bites the most recent context matters most.  The text passes through
    the forbidden-word check (matches replaced with ``***``).
    """
    if not turns:
        return None

    blocks: list[list[str]] = []
    used = 0
    # Newest folded turns first so the cap drops the oldest context.
    for turn in reversed(turns):
        block: list[str] = []
        for row in turn:
            label = "用户" if row.get("role") == "user" else "AI助手"
            text = " ".join(str(row.get("content", "")).split())
            if text:
                block.append(f"[{label}]: {text[:_SUMMARY_ITEM_MAX_CHARS]}")
        if not block:
            continue
        cost = sum(len(line) + 1 for line in block)
        if blocks and used + cost > _SUMMARY_MAX_CHARS:
            break
        blocks.append(block)
        used += cost

    if not blocks:
        return None

    blocks.reverse()  # back to chronological order
    lines = [line for block in blocks for line in block]
    summary = "\n".join([_SUMMARY_HEADER, *lines])

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

    runtime: ChatRunContext = state.get("runtime", ChatRunContext())
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
    runtime: ChatRunContext = state.get("runtime", ChatRunContext())
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

    Loads enough rows for both halves of the payload: the verbatim recent
    window (``_RECENT_TURNS`` complete turns) plus ``max_history_rounds``
    turns of older context that the compression node may fold into the
    summary.  A turn is one user row plus one assistant row, hence ``* 2``.
    """
    runtime: ChatRunContext = state.get("runtime", ChatRunContext())
    session_id = state["session_id"]
    # Never load fewer turns than the verbatim window needs, otherwise the
    # summary would have to be built from turns that are still sent verbatim.
    keep_rounds = max(runtime.max_history_rounds, _RECENT_TURNS + 1)

    history = await runtime.chat_repo.recent(
        session_id, limit=keep_rounds * 2 + 2, user_id=state["user_id"],
    )

    # Recheck ownership before summarisation/model input, including injected repositories.
    return {"messages": [
        row for row in history
        if row.get("user_id") == state["user_id"]
        and row.get("session_id") == session_id
    ]}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4: history_compress_node
# ═══════════════════════════════════════════════════════════════════════════════


async def history_compress_node(state: ChatGraphState) -> dict[str, Any]:
    """Fold the turns outside the verbatim window into one summary message.

    Only turns older than the last ``_RECENT_TURNS`` complete turns are
    summarised, using the same boundary as ``_build_messages_from_state``.
    Folded turns are dropped from the payload instead of being re-sent, so the
    prompt is exactly ``system prompt + summary + recent turns + current user
    message``.  The summary is always a single, size-capped message.
    """
    folded_turns, _recent_turns = _split_history(state)

    return {"history_summary": _summarise_turns(folded_turns)}


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

    runtime: ChatRunContext = state.get("runtime", ChatRunContext())
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
        with suppress(AttributeError, NotImplementedError):
            # Model doesn't support tool binding — proceed without tools
            bound_model = model.bind_tools(tools)

    # ── Invoke LLM ─────────────────────────────────────────────────────────
    started_at = time.perf_counter()
    try:
        result = await bound_model.ainvoke(messages)
    except Exception as exc:
        # The failed call is a measured request too: without its record the
        # chat failure rate would read 0% no matter how often the provider dies.
        _record_chat_call(
            runtime,
            node=_MODEL_CALL_NODE,
            model=bound_model,
            total_latency_ms=(time.perf_counter() - started_at) * 1000.0,
            ok=False,
            error=exc,
        )
        logger.opt(exception=False).warning(
            "ChatGraph: Model invocation failed: {}", _safe_error_metadata(exc),
        )
        return {
            "answer": _LLM_DOWN_REPLY,
            "degraded": True,
            "tools_used": [],
            "evidence_cited": False,
            "errors": list(state.get("errors", [])) + [{
                "key": "model_call_failed",
                "message": _safe_error_metadata(exc),
            }],
        }

    _record_chat_call(
        runtime,
        node=_MODEL_CALL_NODE,
        model=bound_model,
        response=result,
        total_latency_ms=(time.perf_counter() - started_at) * 1000.0,
    )

    # ── Collect tool call info ─────────────────────────────────────────────
    messages.append(result)
    tools_used: list[str] = list(state.get("tools_used", []))
    evidence_cited = state.get("evidence_cited", False)

    tool_call_objects = getattr(result, "tool_calls", None) or []
    existing_tool_msgs: list[dict[str, str]] = list(state.get("tool_messages", []))
    for tc in tool_call_objects:
        t_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        if t_name and t_name not in tools_used:
            tools_used.append(t_name)
            # NOTE: proposing a tool call is not evidence. `evidence_cited` is
            # only set once the tool actually returns data (see
            # tool_execution_node), because an unsuccessful or fabricated call
            # would otherwise be reported to the user as sourced.

        # Record as tool message
        tool_msg: dict[str, str] = {
            "type": "call",
            "name": t_name,
            "content": (
                str(tc.get("args", ""))
                if isinstance(tc, dict)
                else str(getattr(tc, "args", ""))
            ),
        }
        existing_tool_msgs.append(tool_msg)

    if tool_call_objects:
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
    """Build the provider payload for this turn.

    The payload is exactly::

        system prompt
        + at most one history summary (only if older turns were compressed)
        + the last _RECENT_TURNS complete turns, verbatim
        + the current user message

    Turns folded into the summary are **not** re-sent: ``_split_history`` is
    the single boundary shared with ``history_compress_node``, so compressed
    history disappears from every later payload and lives on only inside the
    summary message.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    runtime = state.get("runtime", ChatRunContext())
    _folded_turns, recent_turns = _split_history(state)

    messages: list[Any] = [SystemMessage(content=CHAT_SYSTEM_PROMPT)]
    summary = state.get("history_summary")
    if summary:
        messages.append(SystemMessage(content=summary))

    for turn in recent_turns:
        messages.extend(
            _render_turn(turn, runtime, state["user_id"], state["session_id"]),
        )

    # The current user message is always the last message sent.
    messages.append(HumanMessage(content=state["user_message"]))

    return messages


def _has_tool_evidence(name: str, text: str) -> bool:
    """Recognise data in the two evidence tools' existing JSON contracts."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if name == "query_evidence":
        evidence = data.get("evidence")
        summary = data.get("behavior_summary")
        duration = summary.get("duration_min") if isinstance(summary, dict) else None
        observed_activity = (
            isinstance(duration, (int, float)) and not isinstance(duration, bool)
            and math.isfinite(duration) and duration > 0
        )
        # The real builder emits named info items even for an empty window.
        # Info values are omitted on the wire, so use observed duration for those.
        # Explicit measurements still count when zero; names alone never do.
        return isinstance(evidence, list) and any(
            isinstance(item, dict) and bool(item.get("metric"))
            and (
                observed_activity
                or (
                    type(item.get("value")) in (int, float)
                    and math.isfinite(item["value"])
                )
                or (isinstance(item.get("value"), str) and bool(item["value"].strip()))
            )
            for item in evidence
        )
    if name == "get_latest_analysis":
        types = data.get("procrastination_types", data.get("types"))
        return isinstance(types, list) and any(isinstance(t, str) and t for t in types)
    return False


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

    runtime: ChatRunContext = state.get("runtime", ChatRunContext())
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
    evidence_cited = state.get("evidence_cited", False)

    for tc in tool_calls:
        t_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        t_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        t_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

        tool_fn = tool_map.get(t_name)
        succeeded = False
        if tool_fn is None:
            result_text = f"Tool '{t_name}' not found"
        else:
            try:
                result = await tool_fn.ainvoke(t_args)
                result_text = str(result) if result is not None else ""
                # A tool that ran but returned nothing is not evidence either.
                succeeded = _has_tool_evidence(t_name, result_text)
            except Exception as exc:
                result_text = f"Tool error: {_safe_error_metadata(exc)}"

        if succeeded and t_name in _EVIDENCE_TOOLS:
            evidence_cited = True

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
        "evidence_cited": evidence_cited,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 8: answer_extraction_node
# ═══════════════════════════════════════════════════════════════════════════════


async def answer_extraction_node(state: ChatGraphState) -> dict[str, Any]:
    """Extract the final answer text from the model's messages.

    Searches backward through the accumulated messages for the last
    assistant/AI message with non-empty content.
    """
    if state.get("degraded", False):
        return {}
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
    from langchain_core.messages import SystemMessage

    runtime: ChatRunContext = state.get("runtime", ChatRunContext())
    model = runtime.model
    answer = state.get("answer", "")

    # Find the first forbidden word for the correction message
    bad_word = _contains_forbidden_words(answer)
    correction_text = (
        f"回答包含禁用词汇「{bad_word or '某些'}」，"
        "请用中文重新回答，不要使用诊断、治疗、患者、处方等词汇。"
    )

    # Build retry messages: original messages + correction instruction
    retry_messages = list(state.get("model_messages_raw", [])) or _build_messages_from_state(state)

    # Append correction instruction
    retry_messages.append(SystemMessage(content=correction_text))

    retry_count = state.get("retry_count", 0) + 1

    if model is None:
        return {
            "answer": _SAFE_REPLY,
            "degraded": True,
            "retry_count": retry_count,
        }

    started_at = time.perf_counter()
    try:
        result = await model.ainvoke(retry_messages)
        retry_answer = _extract_answer_from_messages([result])
        _record_chat_call(
            runtime,
            node=_CORRECTION_NODE,
            model=model,
            response=result,
            total_latency_ms=(time.perf_counter() - started_at) * 1000.0,
            retry_count=retry_count,
        )

        if _contains_forbidden_words(retry_answer) is not None:
            return {
                "answer": _SAFE_REPLY,
                "degraded": True,
                "retry_count": retry_count,
            }

        return {
            "answer": retry_answer,
            "retry_count": retry_count,
            "model_messages_raw": [*retry_messages, result],
        }

    except Exception as exc:
        _record_chat_call(
            runtime,
            node=_CORRECTION_NODE,
            model=model,
            total_latency_ms=(time.perf_counter() - started_at) * 1000.0,
            retry_count=retry_count,
            ok=False,
            error=exc,
        )
        logger.opt(exception=False).warning(
            "ChatGraph: Correction retry failed: {}", _safe_error_metadata(exc),
        )
        return {
            "answer": _SAFE_REPLY,
            "degraded": True,
            "retry_count": retry_count,
            "errors": list(state.get("errors", [])) + [{
                "key": "correction_loop_failed",
                "message": _safe_error_metadata(exc),
            }],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 11: assistant_message_persist_node
# ═══════════════════════════════════════════════════════════════════════════════


async def assistant_message_persist_node(state: ChatGraphState) -> dict[str, Any]:
    """Persist the assistant's final answer to the DB.

    Always persists, even when ``degraded=True`` — the user should
    always see a response.
    """
    runtime: ChatRunContext = state.get("runtime", ChatRunContext())
    session_id = state["session_id"]
    answer = state.get("answer", _SAFE_REPLY)
    user_id = state.get("user_id", 0)

    row = await runtime.chat_repo.append(
        session_id, "assistant", answer, user_id=user_id,
    )
    if (
        isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]
        and row.get("user_id") == user_id and row.get("session_id") == session_id
        and row.get("role") == "assistant" and not state.get("degraded", False)
    ):
        raw = state.get("model_messages_raw", [])
        # Cache only this completed turn. Never retain an orphan tool exchange.
        start = next(
            (i + 1 for i in range(len(raw) - 1, -1, -1)
             if getattr(raw[i], "type", None) == "human"),
            len(raw),
        )
        if start < len(raw):
            runtime.protocol_turns[(user_id, session_id, row["id"])] = raw[start:]
            while len(runtime.protocol_turns) > _MAX_PROTOCOL_TURNS:
                runtime.protocol_turns.popitem(last=False)

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
        max_history_rounds: Max conversation turns loaded from the DB and made
            available to the compression node.  The verbatim window is
            ``_RECENT_TURNS``; anything older than that (within this load
            bound) is folded into the single summary message.
        recursion_limit: Max model/tool loop depth.
        provider: Provider label recorded in LLM observability
            (``"ecnu"`` / ``"generic"``).  Empty resolves from settings.
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
        provider: str = "",
    ) -> None:
        self._chat_repo = chat_repo
        self._crisis_detector = crisis_detector
        self._model = model
        self._tools = tools or []
        self._tool_adapters = tool_adapters or []
        self._max_history_rounds = max_history_rounds
        self._recursion_limit = recursion_limit
        self._provider = _provider_label(provider)
        self._session_locks: dict[tuple[int, str], asyncio.Lock] = {}
        # Same-process continuation only. On restart, DB answers reconstruct
        # text history without any orphan tool calls or private reasoning.
        self._protocol_turns: OrderedDict[tuple[int, str, str], list[Any]] = OrderedDict()
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
        lock = self._session_locks.setdefault((user_id, session_id), asyncio.Lock())
        try:
            # One deadline covers queueing, tools, all model calls and persistence.
            async with asyncio.timeout(_CHAT_TURN_TIMEOUT_S):
                async with lock:
                    return await self._ask_serialized(
                        user_id, session_id, message, turn_id,
                    )
        except TimeoutError:
            from mindflow.services.chat_service import ChatAnswer

            return ChatAnswer(
                answer=_LLM_DOWN_REPLY, session_id=session_id, degraded=True,
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
            protocol_turns=self._protocol_turns,
            provider=self._provider,
        )

        # ── Set ToolContext on adapters ────────────────────────────────────
        ctx = ToolContext(
            user_id=user_id,
            session_id=session_id,
            run_id=turn_id,
        )
        context_tokens: list[tuple[Any, Any]] = []

        try:
            for adapter in self._tool_adapters:
                context_tokens.append((adapter, adapter.bind_context(ctx)))

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
            # Ambient labels keep any LLM call reached from inside a chat turn
            # (a gateway-backed tool, a subgraph) grouped under the chat graph
            # even when the caller never labelled it.
            from mindflow.agents.policies import ROLE_CHAT  # noqa: PLC0415
            from mindflow.services.llm_observability import llm_call_context  # noqa: PLC0415

            graph = self._get_compiled_graph()
            with llm_call_context(graph=_CHAT_GRAPH_LABEL, role=ROLE_CHAT):
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
            logger.opt(exception=False).warning(
                "ChatGraph invocation failed: {}", _safe_error_metadata(exc),
            )
            return ChatAnswer(
                answer=_LLM_DOWN_REPLY,
                session_id=session_id,
                degraded=True,
            )

        finally:
            for adapter, token in reversed(context_tokens):
                adapter.reset_context(token)

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

        # forbidden_word_validation → [forbidden? → correction_loop
        #                              | clean → assistant_message_persist]
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
