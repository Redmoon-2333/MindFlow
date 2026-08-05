"""L2 conversational assistant service — G004.

Implements the chat loop (07-agent-upgrade-design.md §6) by delegating to the
v2 ``ChatGraph`` (``mindflow.graph.chat_graph``):

  1. Crisis detection (pre-LLM gate)
  2. Session history loading with compression
  3. Graph-based model/tool loop (tool calling managed internally)
  4. Forbidden word check with retry
  5. Message persistence

Tools are declared in ``agents/langchain_tools.py`` as LangChain ``@tool``
factories and wired into the ``ChatGraph`` during ``__init__``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
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
from mindflow.config import get_settings
from mindflow.graph.chat_graph import ChatGraph
from mindflow.graph.tools import (
    InterventionHistoryTool,
    LatestAnalysisTool,
    QueryEvidenceTool,
    RunAnalysisTool,
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

_LLM_DOWN_REPLY: str = (
    "当前 AI 对话不可用，你可以查看「专注分析」页面了解你的专注情况。"
)
"""Fallback reply when the LLM gateway is entirely unavailable."""

_SAFE_REPLY: str = (
    "我暂时无法回答这个问题，请稍后再试。"
    "你可以查看「专注分析」页面了解你的专注情况。"
)
"""Fallback reply when the LLM output fails the forbidden-word check."""

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
# Service
# ═══════════════════════════════════════════════════════════════════════════════


class ChatService:
    """L2 conversational assistant — delegates to the v2 ``ChatGraph``.

    Manages the conversation lifecycle: crisis gate, per-session locking,
    and delegation to the v2 ``ChatGraph`` (which handles history
    management, tool-augmented model calls, forbidden word enforcement,
    and persistence).

    Args:
        session_factory: SQLAlchemy session factory (used to construct a
            default ``ChatRepository`` when one is not injected).
        crisis_detector: Pre-LLM crisis keyword scanner.
        llm_gateway: LLM gateway for generating responses (kept for backward
            compat; the ``ChatGraph`` model is built from its api_key/base_url).
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
        chat_graph: Optional ``ChatGraph``. When not injected, the service
            builds one from local components in ``__init__``.
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

        # ── Graph integration (v2 — ChatGraph is the only chat path) ───────
        self._chat_graph: ChatGraph | None = chat_graph

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
        self._agent_model = owned_model

        # ── Build ChatGraph if not already injected (v2 is the only path) ───
        if self._chat_graph is None:
            self._chat_graph = ChatGraph(
                chat_repo=self._chat_repo,
                crisis_detector=self._crisis_detector,
                model=llm,
                tools=list(tools),
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
            # ── v2 ChatGraph is the only chat path ──────────────────────
            if self._chat_graph is None:
                return ChatAnswer(
                    answer=_LLM_DOWN_REPLY,
                    session_id=session_id,
                    degraded=True,
                )
            return await self._ask_via_graph(user_id, session_id, message)

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Return the lock that serializes one conversation session."""
        locks: dict[str, asyncio.Lock]
        try:
            locks = self._session_locks
        except AttributeError:
            locks = {}
            self._session_locks = locks
        return locks.setdefault(session_id, asyncio.Lock())

    async def _ask_via_graph(
        self,
        user_id: int,
        session_id: str,
        message: str,
    ) -> ChatAnswer:
        """Delegate the full pipeline to ``ChatGraph.ask()``.

        The graph internally handles persistence, tool execution, forbidden
        word checks, and history compression — producing the
        ``ChatAnswer`` response shape.
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


