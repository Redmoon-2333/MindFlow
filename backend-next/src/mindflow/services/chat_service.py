"""L2 conversational assistant service — G004.

Implements the chat agent loop (07-agent-upgrade-design.md §6):

  1. Crisis detection (pre-LLM gate)
  2. Session history loading with compression
  3. Tool loop (max 3 rounds): LLM decides tool or answer
  4. Forbidden word check with retry
  5. Message persistence

Tools available:
  - query_evidence: Fetch behavior evidence from the ML sensing layer
  - get_latest_analysis: Retrieve today's (or yesterday's) procrastination analysis
  - run_panel: Trigger expert panel deliberation (max 1 per session)
  - query_interventions: Query recent intervention history

All tools reuse existing services/repositories — zero new business logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.agents.llm_gateway import DeepSeekGateway, GatewayAPIError
from mindflow.agents.types import FORBIDDEN_WORDS
from mindflow.domain.evidence import to_prompt_json
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

_MAX_TOOL_ROUNDS: int = 3
"""Maximum tool-call iterations before forcing a direct answer."""

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

_TOOL_RESULT_MAX_CHARS: int = 3000
"""Maximum characters of a tool result to include in the LLM context."""


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


@dataclass
class _ToolCall:
    """A parsed tool call from the LLM response."""

    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ParsedResponse:
    """Parsed LLM response — either a tool call or a direct answer."""

    tool_call: _ToolCall | None = None
    answer: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════════════════════════════


class ChatService:
    """L2 conversational assistant — the chat agent loop.

    Manages the conversation lifecycle: crisis gate, history management,
    tool-augmented LLM loop, forbidden word enforcement, and persistence.

    Args:
        session_factory: SQLAlchemy session factory for the chat repository.
        crisis_detector: Pre-LLM crisis keyword scanner.
        llm_gateway: LLM gateway for generating responses.
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
        self._llm_gateway = llm_gateway
        self._analysis_repo = analysis_repo
        self._panel_service = panel_service
        self._intervention_repo = intervention_repo
        self._evidence_builder = evidence_builder

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
          4. Tool loop (max 3 rounds): LLM decides tool call or direct answer.
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
        conversation = [
            {"role": m["role"], "content": m["content"]} for m in history
        ]
        system_summary = self._compress_history(history)

        # ── 4. Tool loop ────────────────────────────────────────────────
        tools_used: list[str] = []
        evidence_cited = False
        panel_used = False
        final_answer = _LLM_DOWN_REPLY
        degraded = False

        for attempt in range(_MAX_TOOL_ROUNDS + 1):
            user_prompt = self._build_user_prompt(conversation)

            try:
                raw = await self._llm_gateway.complete(
                    system=self._build_system_prompt(system_summary),
                    user=user_prompt,
                )
            except (GatewayAPIError, httpx.TimeoutException, httpx.HTTPError) as exc:
                logger.warning("LLM gateway unavailable in chat: {}", exc)
                degraded = True
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unexpected LLM error in chat: {}", exc)
                degraded = True
                break

            parsed = self._parse_response(raw)

            # ── Direct answer path ──────────────────────────────────────
            if parsed.answer is not None:
                bad_word = self._check_forbidden(parsed.answer)
                if bad_word is None:
                    final_answer = parsed.answer
                elif attempt < _MAX_TOOL_ROUNDS:
                    conversation.append({
                        "role": "system",
                        "content": (
                            f"回答包含禁用词汇「{bad_word}」，"
                            "请用中文重新回答，不要使用诊断、治疗、患者、处方等词汇。"
                        ),
                    })
                    continue
                else:
                    final_answer = _SAFE_REPLY
                    degraded = True
                break

            # ── Tool call path ──────────────────────────────────────────
            if parsed.tool_call is not None:
                tool_name = parsed.tool_call.name

                # Enforce run_panel cap
                if tool_name == "run_panel" and panel_used:
                    conversation.append({
                        "role": "system",
                        "content": "run_panel 每会话最多 1 次，已超出。请直接回答。",
                    })
                    continue

                result = await self._execute_tool(
                    tool_name, parsed.tool_call.args, user_id,
                )

                if tool_name == "run_panel":
                    panel_used = True
                tools_used.append(tool_name)
                if tool_name in _EVIDENCE_TOOLS:
                    evidence_cited = True

                conversation.append({
                    "role": "system",
                    "content": (
                        f"工具 {tool_name} 返回:\n{result[:_TOOL_RESULT_MAX_CHARS]}"
                    ),
                })

        # ── 5. Persist assistant answer ─────────────────────────────────
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
    # System prompt
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_system_prompt(summary: str | None = None) -> str:
        """Build the system prompt with tool schemas and output rules.

        Args:
            summary: Optional compressed history summary to include.

        Returns:
            The complete system prompt string.
        """
        base = (
            "你是 MindFlow 的 AI 助手，帮助用户分析专注力模式和拖延行为。"
            "\n\n"
            "【可用工具】\n"
            "需要获取信息时，返回 JSON 格式工具调用：\n"
            '{"tool": "工具名", "args": {}}\n'
            "可以回答时，返回：\n"
            '{"answer": "你的回答"}\n'
            "\n"
            "工具列表：\n"
            "1. query_evidence(days_back: int, ≤30)\n"
            "   查询最近N天的行为证据（专注度、切换频率、基线偏差等）\n"
            "2. get_latest_analysis()\n"
            "   获取最新的拖延类型分析结果（今日或昨日）\n"
            "3. run_panel()\n"
            "   运行专家团会诊（每会话最多1次）\n"
            "4. query_interventions(days_back: int, ≤30)\n"
            "   查询最近N天的干预记录\n"
            "\n"
            "【回答要求】\n"
            "- 使用中文\n"
            "- 根据用户的行为数据给出个性化建议\n"
            "- 引用具体证据，例如「根据你的行为数据……」\n"
            '- 禁止使用以下词汇：诊断、治疗、患者、处方\n'
            "- 友善、鼓励、具体\n"
            "- 一次只调用一个工具，等待结果后再决定下一步"
        )

        if summary:
            base = f"{base}\n\n【对话背景】\n{summary}"

        return base

    @staticmethod
    def _build_user_prompt(
        conversation: list[dict[str, str]],
    ) -> str:
        """Serialize the conversation into a single user prompt string.

        The system prompt is sent separately via the ``system`` parameter;
        everything here goes into the ``user`` role message.

        Args:
            conversation: List of dicts with ``role`` and ``content`` keys.

        Returns:
            A formatted prompt string.
        """
        parts: list[str] = ["当前对话（最新在最后）:\n"]

        role_labels = {
            "user": "用户",
            "assistant": "助手",
            "system": "系统",
        }

        for msg in conversation:
            label = role_labels.get(msg["role"], msg["role"])
            parts.append(f"{label}: {msg['content']}")

        parts.append(
            "\n请分析以上对话并回答用户的问题。"
            "如果需要获取数据，使用工具；否则直接回答。"
        )

        return "\n\n".join(parts)

    # ══════════════════════════════════════════════════════════════════════
    # Response parsing
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_response(raw: str) -> _ParsedResponse:
        """Parse the LLM's JSON response.

        Handles both valid JSON and fallback text responses.

        Args:
            raw: The raw response string from the LLM.

        Returns:
            A ``_ParsedResponse`` with either a tool call or an answer.
        """
        import re as _re

        # Try strict JSON parsing first
        try:
            data: dict[str, object] = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract a JSON object from the text
            match = _re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", raw, _re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except (json.JSONDecodeError, ValueError):
                    return _ParsedResponse(answer=raw)
            else:
                return _ParsedResponse(answer=raw)

        if "answer" in data:
            return _ParsedResponse(answer=str(data["answer"]))

        if "tool" in data:
            name = str(data["tool"])
            args_raw = data.get("args")
            args: dict[str, Any] = {}
            if isinstance(args_raw, dict):
                args = dict(args_raw)
            return _ParsedResponse(tool_call=_ToolCall(name=name, args=args))

        # Default: treat the whole thing as an answer
        return _ParsedResponse(answer=raw)

    # ══════════════════════════════════════════════════════════════════════
    # Tool execution
    # ══════════════════════════════════════════════════════════════════════

    async def _execute_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_id: int,
    ) -> str:
        """Execute a tool call and return its result as a string.

        Args:
            tool_name: The tool name.
            args: Tool arguments.
            user_id: The user identifier.

        Returns:
            The tool result as a string, or an error message.
        """
        try:
            if tool_name == "query_evidence":
                return await self._do_query_evidence(args, user_id)
            if tool_name == "get_latest_analysis":
                return await self._do_get_latest_analysis(user_id)
            if tool_name == "run_panel":
                return await self._do_run_panel(user_id)
            if tool_name == "query_interventions":
                return await self._do_query_interventions(args, user_id)
            return f"未知工具: {tool_name}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tool '{}' failed: {}", tool_name, exc)
            return f"工具 {tool_name} 执行出错: {exc}"

    async def _do_query_evidence(
        self,
        args: dict[str, Any],
        user_id: int,
    ) -> str:
        """Build an EvidenceBundle and serialize to prompt JSON.

        Args:
            args: Tool arguments (expects ``days_back`` ≤ 30).
            user_id: The user identifier.

        Returns:
            JSON string of the evidence bundle.
        """
        days_back = min(int(args.get("days_back", 7)), 30)
        window_end = datetime.now(UTC)
        window_start = window_end - timedelta(days=days_back)

        bundle = await self._evidence_builder.build(user_id, window_start, window_end)
        return to_prompt_json(bundle)

    async def _do_get_latest_analysis(
        self,
        user_id: int,
    ) -> str:
        """Fetch today's analysis, falling back to yesterday.

        Args:
            user_id: The user identifier.

        Returns:
            JSON string of the analysis, or a "not found" message.
        """
        today = date.today()
        result = await self._analysis_repo.get_by_date(user_id, today)

        if result is None:
            yesterday = today - timedelta(days=1)
            result = await self._analysis_repo.get_by_date(user_id, yesterday)

        if result is None:
            return "暂无分析数据"

        return json.dumps(result, ensure_ascii=False)

    async def _do_run_panel(
        self,
        user_id: int,
    ) -> str:
        """Run the expert panel and return a summary.

        Args:
            user_id: The user identifier.

        Returns:
            JSON string of the panel verdict summary, or an error message
            if the panel service is unavailable.
        """
        if self._panel_service is None:
            return "专家会诊服务暂不可用"

        target_date = date.today()

        try:
            verdict = await self._panel_service.run_daily_panel(user_id, target_date)
            return json.dumps({
                "types": [str(t) for t in verdict.types],
                "confidence": {
                    str(k): float(v) for k, v in verdict.confidence.items()
                },
                "rationale": verdict.rationale,
            }, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Panel execution failed in chat: {}", exc)
            return f"会诊执行失败: {exc}"

    async def _do_query_interventions(
        self,
        args: dict[str, Any],
        user_id: int,
    ) -> str:
        """Query recent intervention history.

        Args:
            args: Tool arguments (expects ``days_back`` ≤ 30).
            user_id: The user identifier.

        Returns:
            JSON string of intervention records, or a "not found" message.
        """
        days_back = min(int(args.get("days_back", 7)), 30)
        start_date = date.today() - timedelta(days=days_back)
        end_date = date.today()

        logs = await self._intervention_repo.query_range_by_date(
            user_id, start_date, end_date,
        )

        if not logs:
            return "暂无干预记录"

        summary = [
            {
                "type": log.get("intervention_type", "unknown"),
                "time": log.get("triggered_at", ""),
                "response": log.get("user_response", "pending"),
            }
            for log in logs
        ]

        return json.dumps(summary, ensure_ascii=False)

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
