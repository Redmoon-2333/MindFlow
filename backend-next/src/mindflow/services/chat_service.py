"""L2 conversational assistant service — G004.

Implements the chat agent loop (07-agent-upgrade-design.md §6) using
LangChain's ``create_agent`` with tool-calling loop:

  1. Crisis detection (pre-LLM gate)
  2. Session history loading with compression
  3. LangChain agent invocation (tool loop managed internally)
  4. Forbidden word check with retry
  5. Message persistence

Tools are declared in ``agents/langchain_tools.py`` as LangChain ``@tool``
factories and wired into the agent during ``__init__``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_deepseek import ChatDeepSeek
from loguru import logger
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.agents.langchain_tools import (
    make_get_latest_analysis,
    make_query_evidence,
    make_query_interventions,
    make_run_panel,
)
from mindflow.agents.llm_gateway import DeepSeekGateway
from mindflow.agents.types import FORBIDDEN_WORDS, _contains_forbidden_words
from mindflow.config import get_settings
from mindflow.graph.chat_graph import ChatGraph
from mindflow.graph.tools import (
    InterventionHistoryTool,
    LatestAnalysisTool,
    QueryEvidenceTool,
    RunAnalysisTool,
    ToolContext,
)
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
)
from mindflow.infrastructure.repositories.chat import ChatRepository
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.infrastructure.security.crisis_detector import (
    CrisisDetector,
    CrisisLevel,
)
from mindflow.services.evidence_service import EvidenceBundleBuilder
from mindflow.services.panel_service import PanelService
from mindflow.time_utils import TimezoneLike

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_HISTORY_ROUNDS: int = 10
"""Maximum conversation rounds (1 round = user + assistant) kept verbatim."""

_LLM_DOWN_REPLY: str = (
    "当前 AI 对话不可用，你可以查看「专注分析」页面了解你的专注情况。"
)
"""Fallback reply when the LLM gateway is entirely unavailable."""

_SAFE_REPLY: str = (
    "我暂时无法回答这个问题，请稍后再试。"
    "你可以查看「专注分析」页面了解你的专注情况。"
)
"""Fallback reply when the LLM output fails the forbidden-word check."""

_EVIDENCE_TOOLS: frozenset[str] = frozenset({"query_evidence", "get_latest_analysis"})
"""Tools whose usage implies evidence was cited in the final answer."""

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
"""Base system prompt passed to create_agent (tool schemas managed by LangChain)."""

# ═══════════════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChatAnswer:
    """Response from the chat assistant.

    Attributes:
        answer: The assistant's response text.
        session_id: The conversation session identifier.
        tools_used: Names of tools invoked during this turn.
        evidence_cited: True if evidence-gathering tools were used.
        degraded: True if the response fell back to a rule-based reply.
    """

    answer: str = ""
    session_id: str = ""
    tools_used: tuple[str, ...] = ()
    evidence_cited: bool = False
    degraded: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Shadow repo — shadow-mode proxy that never writes to the real DB
# ═══════════════════════════════════════════════════════════════════════════════


class _ShadowChatRepo:
    """Wraps a real ``ChatRepository`` for shadow-mode comparison.

    ``append()`` writes only to an in-memory buffer — never to the real DB —
    so the shadow graph can load its own user message via ``recent()`` without
    double-persisting.  ``recent()`` delegates to the real repo for historical
    context and prepends any buffered messages for the current session.

    The buffer is cleared before each shadow request.
    """

    def __init__(self, real_repo: Any) -> None:
        self._real = real_repo
        self._buffer: list[dict[str, Any]] = []

    def clear(self) -> None:
        """Clear the in-memory buffer (call before each shadow request)."""
        self._buffer.clear()

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        user_id: int | None = None,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        """Record message in-memory only — NEVER write to the real DB."""
        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "session_id": session_id,
            "user_id": user_id,
        }
        self._buffer.append(entry)
        return {"id": message_id or "shadow-msg", **entry}

    async def recent(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return real history + buffered messages for *session_id*."""
        real_history = await self._real.recent(session_id, limit=limit)
        buffered = [m for m in self._buffer if m.get("session_id") == session_id]
        # Buffered messages (user) go first so the graph sees the current
        # user message before loading older history.
        return buffered + real_history

    async def list_sessions(
        self,
        user_id: int,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Delegate to the real repository."""
        return await self._real.list_sessions(user_id=user_id, limit=limit)


# ═══════════════════════════════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════════════════════════════


class ChatService:
    """L2 conversational assistant — the LangChain-powered chat agent loop.

    Manages the conversation lifecycle: crisis gate, history management,
    LangChain agent (tool-augmented LLM), forbidden word enforcement,
    and persistence.

    Args:
        session_factory: SQLAlchemy session factory (used to construct a
            default ``ChatRepository`` when one is not injected).
        crisis_detector: Pre-LLM crisis keyword scanner.
        llm_gateway: LLM gateway for generating responses (kept for backward
            compat; the LangChain agent uses ``ChatDeepSeek`` instead).
        analysis_repo: Repository for procrastination analysis results.
        panel_service: Expert panel service (None if unavailable).
        intervention_repo: Intervention history repository.
        evidence_builder: Evidence bundle builder for behavioral data.
        chat_repo: Chat message repository. Defaults to a ``ChatRepository``
            built from *session_factory* (kept optional so existing call
            sites need no change), matching how other services receive
            their repos injected.
        provider_registry: Optional shared ProviderRegistry. When provided,
            the chat model is obtained from the registry and HTTP pool
            lifecycle is managed by the registry (aclose becomes a no-op).
        chat_graph: Optional ``ChatGraph`` for v2 graph-based chat path.
            When provided and ``new_chat_graph`` config is True, ``ask()``
            delegates to the graph instead of ``create_agent``.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        crisis_detector: CrisisDetector,
        llm_gateway: DeepSeekGateway,
        analysis_repo: SQLAlchemyProcrastinationAnalysisRepository,
        panel_service: PanelService | None,
        intervention_repo: InterventionLogRepository,
        evidence_builder: EvidenceBundleBuilder,
        chat_repo: ChatRepository | None = None,
        max_history_rounds: int | None = None,
        agent: Any | None = None,
        model: BaseChatModel | None = None,
        timezone: TimezoneLike | None = None,
        provider_registry: Any | None = None,
        chat_graph: ChatGraph | None = None,
    ) -> None:
        self._chat_repo = chat_repo or ChatRepository(session_factory=session_factory)
        settings = get_settings()
        self._max_history_rounds = (
            max_history_rounds
            if max_history_rounds is not None
            else settings.max_history_rounds
        )
        self._timezone: TimezoneLike = timezone or settings.timezone
        self._crisis_detector = crisis_detector
        self._analysis_repo = analysis_repo
        self._panel_service = panel_service
        self._intervention_repo = intervention_repo
        self._evidence_builder = evidence_builder
        self._session_locks: dict[str, asyncio.Lock] = {}

        # ── Backward compat: keep _llm_gateway for existing test fixtures ───
        self._llm_gateway = llm_gateway
        self._registry = provider_registry

        # ── Graph integration (Wave 4) ────────────────────────────────────
        self._chat_graph: ChatGraph | None = chat_graph
        self._shadow_graph: ChatGraph | None = None
        self._shadow_repo: _ShadowChatRepo | None = None
        self._new_chat_graph: bool = settings.new_chat_graph
        self._shadow_mode_chat: bool = settings.shadow_mode_chat

        if agent is not None:
            self._agent = agent
            self._agent_model: ChatDeepSeek | None = None
            return

        # ── Build typed tool adapters ──────────────────────────────────────────
        evidence_adapter = QueryEvidenceTool(
            evidence_builder=evidence_builder, timezone=self._timezone
        )
        analysis_adapter = LatestAnalysisTool(
            analysis_repo=analysis_repo, timezone=self._timezone
        )
        panel_adapter = RunAnalysisTool(
            panel_service=panel_service, timezone=self._timezone
        )
        intervention_adapter = InterventionHistoryTool(
            intervention_repo=intervention_repo, timezone=self._timezone
        )
        self._tool_adapters: list = [
            evidence_adapter,
            analysis_adapter,
            panel_adapter,
            intervention_adapter,
        ]

        # ── Build LangChain tools from adapters ───────────────────────────────
        tools: list[BaseTool] = [
            make_query_evidence(evidence_adapter),
            make_get_latest_analysis(analysis_adapter),
            make_run_panel(panel_adapter),
            make_query_interventions(intervention_adapter),
        ]

        # ── Build LangChain model ───────────────────────────────────────────
        # Read api_key/base_url from the injected gateway's state (E2E finding:
        # the app must assemble services without a key so degradation paths
        # stay reachable — ChatDeepSeek is only initialised when a key exists).
        api_key: str = getattr(llm_gateway, "_api_key", "")
        base_url: str = getattr(llm_gateway, "_base_url", "")

        # Reconstruct the base URL for the LangChain client (strip /chat/completions
        # or keep as-is based on what ChatDeepSeek expects).
        llm: BaseChatModel | None = model
        owned_model: ChatDeepSeek | None = None
        if llm is None and api_key:
            owned_model = ChatDeepSeek(
                model="deepseek-chat",
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=0.7,
                max_tokens=2048,
            )
            llm = owned_model

        # Keep a reference so aclose() can release this model's httpx pool.
        # create_agent does not expose the model back to us, and dropping the
        # reference alone leaks the pool until GC (review C2 connection leak).
        self._agent_model = owned_model

        # ── Build agent ─────────────────────────────────────────────────────
        if llm is None:
            self._agent = None
        else:
            self._agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt=CHAT_SYSTEM_PROMPT,
                name="mindflow_chat_agent",
            )

        # ── Build ChatGraph if not already injected ──────────────────────────
        if self._new_chat_graph and self._chat_graph is None:
            # Build ChatGraph from local components (app.py may inject instead)
            self._chat_graph = ChatGraph(
                chat_repo=self._chat_repo,
                crisis_detector=self._crisis_detector,
                model=llm,
                tools=list(tools),
                tool_adapters=list(self._tool_adapters),
                max_history_rounds=self._max_history_rounds,
            )

        # ── Build shadow graph (shadow-mode comparison) ─────────────────────
        if self._shadow_mode_chat and self._chat_graph is not None:
            self._shadow_repo = _ShadowChatRepo(self._chat_repo)
            self._shadow_graph = ChatGraph(
                chat_repo=self._shadow_repo,
                crisis_detector=self._crisis_detector,
                model=self._chat_graph._model,
                tools=list(self._chat_graph._tools),
                tool_adapters=list(self._tool_adapters),
                max_history_rounds=self._max_history_rounds,
            )

    async def aclose(self) -> None:
        """Close the LLM HTTP clients held by this service.

        When a ``ProviderRegistry`` is injected, pool lifecycle is managed
        by the registry — this method becomes a no-op (the registry's
        ``shutdown()`` is called separately during application shutdown).

        Otherwise (standalone clients), close both the injected gateway
        and the standalone ``ChatDeepSeek`` built for the agent (review
        C2 connection leak).
        """
        import contextlib

        if self._registry is not None:
            return  # registry manages pool lifecycle

        with contextlib.suppress(Exception):
            await self._llm_gateway.close()

        model = self._agent_model
        if model is not None:
            async_client = getattr(model, "root_async_client", None)
            if async_client is not None and hasattr(async_client, "close"):
                with contextlib.suppress(Exception):
                    await async_client.close()
            sync_client = getattr(model, "root_client", None)
            if sync_client is not None and hasattr(sync_client, "close"):
                with contextlib.suppress(Exception):
                    sync_client.close()
            self._agent_model = None

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    async def ask(
        self,
        user_id: int,
        session_id: str,
        message: str,
    ) -> ChatAnswer:
        """Process a user message and return the assistant's response.

        The full pipeline:
          1. Crisis detection — hit → return hotline info, no LLM, no storage.
          2. Persist user message.
          3. Load session history (compress oldest rounds if > 10).
          4. LangChain agent invocation (tool loop managed internally).
          5. Forbidden word check (1 retry, then safe reply).
          6. Persist assistant answer.

        Args:
            user_id: The user identifier.
            session_id: The conversation session identifier.
            message: The user's text message.

        Returns:
            A ``ChatAnswer`` with the response and metadata.
        """
        # ── 1. Crisis detection (pre-LLM gate) ──────────────────────────
        crisis_level, crisis_response = self._crisis_detector.scan(message)
        if crisis_level == CrisisLevel.HIGH and crisis_response is not None:
            logger.warning("Crisis detected in chat message, user_id={}", user_id)
            return ChatAnswer(
                answer=crisis_response.message,
                session_id=session_id,
                degraded=True,
            )

        async with self._get_session_lock(session_id):
            # ── Route: new graph → shadow mode → legacy ─────────────────
            if getattr(self, "_shadow_mode_chat", False) and self._shadow_graph is not None:
                return await self._ask_shadow_mode(user_id, session_id, message)
            if getattr(self, "_new_chat_graph", False) and self._chat_graph is not None:
                return await self._ask_via_graph(user_id, session_id, message)
            return await self._ask_serialized(user_id, session_id, message)

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Return the lock that serializes one conversation session."""
        locks: dict[str, asyncio.Lock]
        try:
            locks = self._session_locks
        except AttributeError:
            locks = {}
            self._session_locks = locks
        return locks.setdefault(session_id, asyncio.Lock())

    async def _ask_serialized(
        self,
        user_id: int,
        session_id: str,
        message: str,
    ) -> ChatAnswer:
        """Run the mutable chat pipeline while holding the session lock."""
        # ── 2. Persist user message ─────────────────────────────────────
        await self._chat_repo.append(
            session_id, "user", message, user_id=user_id,
        )

        # ── 3. Load and prepare history ─────────────────────────────────
        max_rounds = getattr(self, "_max_history_rounds", _MAX_HISTORY_ROUNDS)
        history = await self._chat_repo.recent(
            session_id, limit=max_rounds * 2 + 2,
        )
        system_summary = self._compress_history(history, max_rounds)

        # ── 4. LangChain agent invocation ───────────────────────────────
        degraded = False
        final_answer = _LLM_DOWN_REPLY
        tools_used: list[str] = []
        evidence_cited = False

        # Set context on tool adapters (replaces ContextVars)
        ctx = ToolContext(user_id=user_id, session_id=session_id)
        for adapter in getattr(self, "_tool_adapters", []):
            adapter.context = ctx
        try:
            # Build LangChain message list
            messages: list[BaseMessage] = []
            if system_summary:
                messages.append(SystemMessage(content=system_summary))

            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

            # Current user message is already the last "user" entry in history
            # (we persisted it in step 2 and loaded it in step 3).  If it's not
            # in history yet, add it explicitly — unlikely but keeps the invariant.
            if not any(
                isinstance(m, HumanMessage) and m.content == message for m in messages
            ):
                messages.append(HumanMessage(content=message))

            if self._agent is None:
                degraded = True
            else:
                try:
                    result = await self._agent.ainvoke(
                        {"messages": messages},
                        config={"recursion_limit": 12},
                    )
                    final_answer = self._extract_answer(result)

                    # Extract tool names from message history
                    for msg_obj in result.get("messages", []):
                        tc = getattr(msg_obj, "tool_calls", None) or []
                        for call in tc:
                            t_name = (
                                call.get("name", "")
                                if isinstance(call, dict)
                                else getattr(call, "name", "")
                            )
                            if t_name:
                                tools_used.append(t_name)
                                if t_name in _EVIDENCE_TOOLS:
                                    evidence_cited = True

                except Exception as exc:  # noqa: BLE001
                    logger.warning("LangChain agent invocation failed: {}", exc)
                    degraded = True

            # ── 5. Forbidden word check (1 retry) ───────────────────────────
            if not degraded:
                assert self._agent is not None
                bad_word = _contains_forbidden_words(final_answer)
                if bad_word is not None:
                    logger.warning("Forbidden word '{}' found in response", bad_word)
                    # One retry: append a correction instruction
                    retry_messages = list(messages)
                    retry_messages.append(
                        SystemMessage(
                            content=(
                                f"回答包含禁用词汇「{bad_word}」，"
                                "请用中文重新回答，不要使用诊断、治疗、患者、处方等词汇。"
                            )
                        )
                    )
                    try:
                        retry_result = await self._agent.ainvoke(
                            {"messages": retry_messages},
                            config={"recursion_limit": 12},
                        )
                        retry_answer = self._extract_answer(retry_result)
                        if _contains_forbidden_words(retry_answer) is None:
                            final_answer = retry_answer
                        else:
                            final_answer = _SAFE_REPLY
                            degraded = True
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("LangChain agent retry failed: {}", exc)
                        final_answer = _SAFE_REPLY
                        degraded = True

            # ── 6. Persist assistant answer ─────────────────────────────────
            await self._chat_repo.append(
                session_id, "assistant", final_answer, user_id=user_id,
            )

            return ChatAnswer(
                answer=final_answer,
                session_id=session_id,
                tools_used=tuple(tools_used),
                evidence_cited=evidence_cited,
                degraded=degraded,
            )

        finally:
            # Clear context on adapters after agent invocation
            for adapter in getattr(self, "_tool_adapters", []):
                adapter.context = None

    async def _ask_via_graph(
        self,
        user_id: int,
        session_id: str,
        message: str,
    ) -> ChatAnswer:
        """Delegate the full pipeline to ``ChatGraph.ask()``.

        The graph internally handles persistence, tool execution, forbidden
        word checks, and history compression — producing the same
        ``ChatAnswer`` shape as ``_ask_serialized``.
        """
        assert self._chat_graph is not None, "_ask_via_graph requires _chat_graph"

        try:
            result = await self._chat_graph.ask(user_id, session_id, message)
            if not isinstance(result, ChatAnswer):
                return ChatAnswer(
                    answer=_LLM_DOWN_REPLY,
                    session_id=session_id,
                    degraded=True,
                )
            return result
        except Exception as exc:
            logger.warning("ChatGraph invocation failed in _ask_via_graph: {}", exc)
            return ChatAnswer(
                answer=_LLM_DOWN_REPLY,
                session_id=session_id,
                degraded=True,
            )

    async def _ask_shadow_mode(
        self,
        user_id: int,
        session_id: str,
        message: str,
    ) -> ChatAnswer:
        """Run both legacy and new paths, compare, return legacy output.

        The legacy path persists to the real DB normally.  The shadow
        (new) path uses ``_ShadowChatRepo`` — its writes are in-memory
        only and discarded after comparison.  Shadow output is never
        returned to the caller.

        Comparison is logged at INFO level for answer text diffs and at
        DEBUG level for metadata diffs (tools_used, evidence_cited,
        degraded).
        """
        assert self._shadow_graph is not None, "_ask_shadow_mode requires _shadow_graph"
        assert self._shadow_repo is not None, "_ask_shadow_mode requires _shadow_repo"

        # ── 1. Legacy path (persists normally) ──────────────────────────
        legacy_result = await self._ask_serialized(user_id, session_id, message)

        # ── 2. Shadow path (no real DB writes) ──────────────────────────
        self._shadow_repo.clear()
        shadow_result: ChatAnswer | None = None
        try:
            shadow_result = await self._shadow_graph.ask(
                user_id, session_id, message,
            )
        except Exception as exc:
            logger.warning("Shadow graph invocation failed: {}", exc)

        # ── 3. Compare ─────────────────────────────────────────────────
        if shadow_result is not None:
            if legacy_result.answer != shadow_result.answer:
                logger.info(
                    "Shadow mode diff [answer]: session={}, legacy='{}', new='{}'",
                    session_id,
                    legacy_result.answer[:200],
                    shadow_result.answer[:200],
                )
            else:
                logger.debug(
                    "Shadow mode match [answer]: session={}", session_id,
                )
            if legacy_result.tools_used != shadow_result.tools_used:
                logger.info(
                    "Shadow mode diff [tools_used]: legacy={}, new={}",
                    list(legacy_result.tools_used),
                    list(shadow_result.tools_used),
                )
            if legacy_result.evidence_cited != shadow_result.evidence_cited:
                logger.info(
                    "Shadow mode diff [evidence_cited]: legacy={}, new={}",
                    legacy_result.evidence_cited,
                    shadow_result.evidence_cited,
                )
            if legacy_result.degraded != shadow_result.degraded:
                logger.info(
                    "Shadow mode diff [degraded]: legacy={}, new={}",
                    legacy_result.degraded,
                    shadow_result.degraded,
                )
        else:
            logger.info(
                "Shadow mode: shadow path failed for session={}, legacy returned",
                session_id,
            )

        # ── 4. Return legacy output ONLY ────────────────────────────────
        return legacy_result

    async def list_sessions(
        self,
        user_id: int,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List the user's most recent chat sessions.

        Public entry point so routes don't reach into the private
        ``_chat_repo`` (encapsulation — E5). Delegates to the repository.
        """
        return await self._chat_repo.list_sessions(user_id=user_id, limit=limit)

    async def get_messages(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the messages for a session, oldest-first.

        Public entry point so routes don't reach into the private
        ``_chat_repo`` (encapsulation — E5). Delegates to the repository.
        """
        return await self._chat_repo.recent(session_id, limit=limit)

    # ══════════════════════════════════════════════════════════════════════
    # Agent output helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_answer(result: dict[str, Any]) -> str:
        """Extract the final answer text from a LangChain agent result.

        Args:
            result: The agent invocation result dict (``AgentState``).

        Returns:
            The last AI message's content, or a safe fallback.
        """
        messages = result.get("messages", [])
        if not messages:
            return _LLM_DOWN_REPLY

        last = messages[-1]
        content = last.content if hasattr(last, "content") else str(last)
        return str(content) if content else _LLM_DOWN_REPLY

    # ══════════════════════════════════════════════════════════════════════
    # History compression
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compress_history(
        history: list[dict[str, Any]],
        max_history_rounds: int = _MAX_HISTORY_ROUNDS,
    ) -> str | None:
        """Compress oldest conversation rounds into a text summary.

        When the history exceeds *max_history_rounds* rounds (20 messages),
        the earliest messages are summarized. The summary also passes through
        the forbidden-word check.

        Args:
            history: Full message list from the repository (oldest-first).
            max_history_rounds: Max rounds to keep verbatim (default from config).

        Returns:
            A summary string, or None if no compression is needed.
        """
        max_messages = max_history_rounds * 2
        if len(history) <= max_messages:
            return None

        extra = len(history) - max_messages
        to_compress = history[:extra]

        parts: list[str] = ["之前的对话摘要:"]
        for msg in to_compress:
            label = "用户" if msg.get("role") == "user" else "AI助手"
            text = msg.get("content", "")[:200]
            parts.append(f"[{label}]: {text}")

        summary = "\n".join(parts)

        # Forbidden word check on summary
        for word in FORBIDDEN_WORDS:
            if word in summary:
                summary = summary.replace(word, "***")

        return summary


