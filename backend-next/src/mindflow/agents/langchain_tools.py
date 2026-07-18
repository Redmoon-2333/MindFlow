"""MindFlow tools declared as LangChain ``@tool`` for use with ``create_agent``.

Each tool wraps an existing service or repository call and returns a string
suitable for inclusion in the LLM context window.  Tools that require
dependencies (repositories, services) capture them via closure at factory
time — the exported ``make_*`` functions take the dependency and return the
tool callable.

Per-session caps:
  - ``run_panel``: 1 invocation per session (tracked via ``session_panel_usage``).

Context:
  - ``current_user_id``: ContextVar set by ``ChatService.ask`` before
    agent invocation, read by tools that need the user identity.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import UTC, date, datetime, timedelta
from typing import Any

from langchain_core.tools import tool

from mindflow.domain.evidence import to_prompt_json
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.services.evidence_service import EvidenceBundleBuilder
from mindflow.services.panel_service import PanelService

# ── Session-panel usage tracking ─────────────────────────────────────────

session_panel_usage: dict[str, int] = {}
"""Per-session ``run_panel`` call count (modify is not thread-safe — guarded
by the GIL + single-threaded async model)."""


# ── Context for implicit tool arguments ──────────────────────────────────

current_user_id: ContextVar[int] = ContextVar("current_user_id", default=0)
"""User id for the current request, set by ``ChatService.ask`` before the
agent invocation."""

current_session_id: ContextVar[str | None] = ContextVar(
    "current_session_id", default=None
)
"""Session id for the current request, set by ``ChatService.ask`` before the
agent invocation.  Read by tools that need session-level state (e.g. panel cap)."""


# ── Tool factory: query_evidence ─────────────────────────────────────────


def make_query_evidence(
    evidence_builder: EvidenceBundleBuilder,
) -> Callable[..., Awaitable[str]]:
    """Return a ``query_evidence`` tool bound to *evidence_builder*.

    The tool signature exposed to the LLM::

        query_evidence(days_back: int = 7) -> str
    """

    @tool
    async def query_evidence(days_back: int = 7) -> str:
        """Query behavior evidence from the ML sensing layer.

        Fetches focus score, switch rate, longest focus block, behavior
        deviation, intervention history, and novelty flags for the last
        N days (capped at 30).

        Args:
            days_back: Number of days to look back (max 30).

        Returns:
            JSON string of the evidence bundle.
        """
        uid = current_user_id.get()
        if uid == 0:
            return '{"error": "user_id not set"}'

        capped = min(days_back, 30)
        window_end = datetime.now(UTC)
        window_start = window_end - timedelta(days=capped)

        bundle = await evidence_builder.build(uid, window_start, window_end)
        return to_prompt_json(bundle)

    return query_evidence


# ── Tool factory: get_latest_analysis ────────────────────────────────────


def make_get_latest_analysis(
    analysis_repo: SQLAlchemyProcrastinationAnalysisRepository,
) -> Callable[..., Awaitable[str]]:
    """Return a ``get_latest_analysis`` tool bound to *analysis_repo*.

    The tool signature exposed to the LLM::

        get_latest_analysis() -> str
    """

    @tool
    async def get_latest_analysis() -> str:
        """Retrieve today's (or yesterday's) procrastination analysis.

        Returns the latest procrastination-type diagnosis with confidence
        scores from the ML pipeline.

        Returns:
            JSON string of the analysis result, or a not-found message.
        """
        uid = current_user_id.get()
        if uid == 0:
            return '{"error": "user_id not set"}'

        today = date.today()
        result: dict[str, Any] | None = await analysis_repo.get_by_date(uid, today)

        if result is None:
            yesterday = today - timedelta(days=1)
            result = await analysis_repo.get_by_date(uid, yesterday)

        if result is None:
            return "暂无分析数据"

        return json.dumps(result, ensure_ascii=False)

    return get_latest_analysis


# ── Tool factory: run_panel ──────────────────────────────────────────────


def make_run_panel(
    panel_service: PanelService | None,
) -> Callable[..., Awaitable[str]]:
    """Return a ``run_panel`` tool bound to *panel_service*.

    The tool signature exposed to the LLM::

        run_panel() -> str

    Per-session cap (1 call) is enforced via ``session_panel_usage``.
    The *session_id* is read from ``current_session_id`` contextvar.
    """

    @tool
    async def run_panel() -> str:
        """Run the expert panel deliberation on today's data.

        Triggers a multi-expert analysis (analyst, attribution expert,
        moderator, critic) to produce a procrastination-type verdict
        with CBT recommendations.

        Limited to **1 invocation per session**.

        Returns:
            JSON string of the panel verdict, or an error/skip message.
        """
        uid = current_user_id.get()
        if uid == 0:
            return '{"error": "user_id not set"}'

        # Per-session cap — relies on current_session_id
        sid = current_session_id.get()
        if sid is not None and session_panel_usage.get(sid, 0) >= 1:
            return "run_panel 每会话最多 1 次，已超出。"

        if panel_service is None:
            return "专家会诊服务暂不可用"

        target_date = date.today()
        try:
            verdict = await panel_service.run_daily_panel(uid, target_date)
            if sid is not None:
                session_panel_usage[sid] = session_panel_usage.get(sid, 0) + 1
            return json.dumps(
                {
                    "types": [str(t) for t in verdict.types],
                    "confidence": {str(k): float(v) for k, v in verdict.confidence.items()},
                    "rationale": verdict.rationale,
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            from loguru import logger

            logger.warning("Panel execution failed in chat tool: {}", exc)
            return f"会诊执行失败: {exc}"

    return run_panel


# ── Tool factory: query_interventions ────────────────────────────────────


def make_query_interventions(
    intervention_repo: InterventionLogRepository,
) -> Callable[..., Awaitable[str]]:
    """Return a ``query_interventions`` tool bound to *intervention_repo*.

    The tool signature exposed to the LLM::

        query_interventions(days_back: int = 7) -> str
    """

    @tool
    async def query_interventions(days_back: int = 7) -> str:
        """Query recent intervention history.

        Returns nudge, task-breakdown, reframe, and environment-mod
        intervention records triggered in the last N days (capped at 30).

        Args:
            days_back: Number of days to look back (max 30).

        Returns:
            JSON string of intervention records, or a not-found message.
        """
        uid = current_user_id.get()
        if uid == 0:
            return '{"error": "user_id not set"}'

        capped = min(days_back, 30)
        start_date = date.today() - timedelta(days=capped)
        end_date = date.today()

        logs = await intervention_repo.query_range_by_date(uid, start_date, end_date)

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

    return query_interventions
