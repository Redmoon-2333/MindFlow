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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.agents.langchain_tools import (
    current_session_id,
    make_get_latest_analysis,
    make_query_evidence,
    make_query_interventions,
    make_run_panel,
)
from mindflow.agents.langchain_tools import (
    current_user_id as _tools_user_id,
)
from mindflow.agents.llm_gateway import DeepSeekGateway
from mindflow.agents.types import FORBIDDEN_WORDS
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

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_HISTORY_ROUNDS: int = 10
"""Maximum conversation rounds (1 round = user + assistant) kept verbatim."""

_LLM_DOWN_REPLY: str = (
    "当前 AI 对话不可用，你可以查看今日报告 /api/v1/focus 了解你的专注情况。"
)
"""Fallback reply when the LLM gateway is entirely unavailable."""

_SAFE_REPLY: str = (
    "我暂时无法回答这个问题，请稍后再试。"
    "你可以查看今日报告 /api/v1/focus 了解你的专注情况。"
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
# Service
# ═══════════════════════════════════════════════════════════════════════════════


class ChatService:
    """L2 conversational assistant — the LangChain-powered chat agent loop.

    Manages the conversation lifecycle: crisis gate, history management,
    LangChain agent (tool-augmented LLM), forbidden word enforcement,
    and persistence.

    Args:
        session_factory: SQLAlchemy session factory for the chat repository.
        crisis_detector: Pre-LLM crisis keyword scanner.
        llm_gateway: LLM gateway for generating responses (kept for backward
            compat; the LangChain agent uses ``ChatDeepSeek`` instead).
        analysis_repo: Repository for procrastination analysis results.
        panel_service: Expert panel service (None if unavailable).
        intervention_repo: Intervention history repository.
        evidence_builder: Evidence bundle builder for behavioral data.
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
    ) -> None:
        self._chat_repo = ChatRepository(session_factory=session_factory)
        self._crisis_detector = crisis_detector
        self._analysis_repo = analysis_repo
        self._panel_service = panel_service
        self._intervention_repo = intervention_repo
        self._evidence_builder = evidence_builder

        # ── Backward compat: keep _llm_gateway for existing test fixtures ───
        self._llm_gateway = llm_gateway

        # ── Build LangChain tools ───────────────────────────────────────────
        tools: list[Callable[..., Awaitable[str]]] = [
            make_query_evidence(evidence_builder),
            make_get_latest_analysis(analysis_repo),
            make_run_panel(panel_service),
            make_query_interventions(intervention_repo),
        ]

        # ── Build LangChain model ───────────────────────────────────────────
        # Read api_key/base_url from the injected gateway's state (E2E finding:
        # the app must assemble services without a key so degradation paths
        # stay reachable — ChatDeepSeek is only initialised when a key exists).
        api_key: str = getattr(llm_gateway, "_api_key", "")
        base_url: str = getattr(llm_gateway, "_base_url", "")

        # Reconstruct the base URL for the LangChain client (strip /chat/completions
        # or keep as-is based on what ChatDeepSeek expects).
        llm: ChatDeepSeek | None = None
        if api_key:
            llm = ChatDeepSeek(
                model="deepseek-chat",
                api_key=api_key,
                base_url=base_url,
                temperature=0.7,
                max_tokens=2048,
            )

        # ── Build agent ─────────────────────────────────────────────────────
        self._agent = create_agent(
            model=llm if llm is not None else "deepseek-chat",  # type: ignore[arg-type]
            tools=tools,
            system_prompt=CHAT_SYSTEM_PROMPT,
            name="mindflow_chat_agent",
        )

    async def aclose(self) -> None:
        """Close the underlying LLM gateway HTTP client.

        Cleanup hook for application shutdown (review P2 connection leak).
        """
        import contextlib

        with contextlib.suppress(Exception):
            await self._llm_gateway.close()

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

        # ── 2. Persist user message ─────────────────────────────────────
        await self._chat_repo.append(
            session_id, "user", message, user_id=user_id,
        )

        # ── 3. Load and prepare history ─────────────────────────────────
        history = await self._chat_repo.recent(
            session_id, limit=_MAX_HISTORY_ROUNDS * 2 + 2,
        )
        system_summary = self._compress_history(history)

        # ── 4. LangChain agent invocation ───────────────────────────────
        degraded = False
        final_answer = _LLM_DOWN_REPLY
        tools_used: list[str] = []
        evidence_cited = False

        # Set context for tool factories
        current_session_id.set(session_id)
        _tools_user_id.set(user_id)

        # Build LangChain message list
        messages: list = []
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

        try:
            result = await self._agent.ainvoke({"messages": messages})
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
            bad_word = self._check_forbidden(final_answer)
            if bad_word is not None:
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
                        {"messages": retry_messages}
                    )
                    retry_answer = self._extract_answer(retry_result)
                    if self._check_forbidden(retry_answer) is None:
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
    ) -> str | None:
        """Compress oldest conversation rounds into a text summary.

        When the history exceeds ``_MAX_HISTORY_ROUNDS`` rounds (20 messages),
        the earliest messages are summarized. The summary also passes through
        the forbidden-word check.

        Args:
            history: Full message list from the repository (oldest-first).

        Returns:
            A summary string, or None if no compression is needed.
        """
        max_messages = _MAX_HISTORY_ROUNDS * 2
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

    # ══════════════════════════════════════════════════════════════════════
    # Forbidden word check
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _check_forbidden(text: str) -> str | None:
        """Check if *text* contains any forbidden word.

        Args:
            text: The text to check.

        Returns:
            The first forbidden word found, or None if clean.
        """
        for word in FORBIDDEN_WORDS:
            if word in text:
                logger.warning("Forbidden word '{}' found in response", word)
                return word
        return None
