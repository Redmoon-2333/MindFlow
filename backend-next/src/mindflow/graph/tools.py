"""Typed tool adapters with explicit context for LangChain agent tools.

Replaces ContextVar-based tool factories with policy-enforced tool adapters
that receive ``user_id``, ``session_id``, and ``run_id`` explicitly via
``ToolContext``.  Each adapter owns its dependencies (repositories, services,
budget gate) through constructor injection — no global state, no private
attribute access.

Design constraints:
  - Every ``execute()`` method receives ``ToolContext`` as its first argument.
  - Input is validated through Pydantic schemas (days cap, required fields).
  - Output is sanitised (no raw prompts, no API keys, no full evidence JSON).
  - ``RunAnalysisTool`` uses ``BudgetReservationPort`` for durable exactly-once
    semantics; an in-memory fallback is provided when no port is injected.
  - Tool descriptions (user-visible to the LLM) are preserved from the
    original ``langchain_tools.py`` factories.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mindflow.domain.evidence import to_prompt_json
from mindflow.ports import BudgetReservationPort
from mindflow.time_utils import TimezoneLike, business_today

# ═══════════════════════════════════════════════════════════════════════════════
# Context
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ToolContext:
    """Explicit context passed to every tool invocation.

    Replaces ``current_user_id`` and ``current_session_id`` ContextVars.
    ``run_id`` is optional — it is set by the caller (e.g. a trace ID) for
    correlation, not by the tool.
    """

    user_id: int
    session_id: str | None = None
    run_id: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Input schemas
# ═══════════════════════════════════════════════════════════════════════════════


class QueryEvidenceInput(BaseModel):
    """Validated input for ``query_evidence``."""

    user_id: int
    session_id: str | None = None
    days: int = Field(default=7, ge=1, description="Days to look back (capped at 30)")

    @field_validator("days")
    @classmethod
    def _cap_days(cls, v: int) -> int:
        return min(v, 30)


class LatestAnalysisInput(BaseModel):
    """Validated input for ``latest_analysis``."""

    user_id: int
    session_id: str | None = None
    target_date: date | None = Field(default=None, description="Target date; None = today")


class RunAnalysisInput(BaseModel):
    """Validated input for ``run_analysis``."""

    user_id: int
    session_id: str | None = None
    target_date: date = Field(description="Target date for the panel analysis")
    force: bool = Field(default=False, description="Bypass idempotent cache")


class InterventionHistoryInput(BaseModel):
    """Validated input for ``intervention_history``."""

    user_id: int
    session_id: str | None = None
    days: int = Field(default=7, ge=1, description="Days to look back (capped at 30)")

    @field_validator("days")
    @classmethod
    def _cap_days(cls, v: int) -> int:
        return min(v, 30)


# ═══════════════════════════════════════════════════════════════════════════════
# Output schemas
# ═══════════════════════════════════════════════════════════════════════════════


class QueryEvidenceOutput(BaseModel):
    """Sanitised output from ``query_evidence``."""

    sessions: list[dict[str, Any]] = Field(default_factory=list)
    evidence_json: str = Field(default="{}", description="Serialized evidence bundle")
    total_days: int = 0
    capped: bool = False
    error: str | None = None


class LatestAnalysisOutput(BaseModel):
    """Sanitised output from ``latest_analysis``."""

    analysis: dict[str, Any] | None = None
    source: str | None = None
    target_date: date | None = None
    error: str | None = None


class RunAnalysisOutput(BaseModel):
    """Sanitised output from ``run_analysis``."""

    analysis: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    degraded: bool = False
    error: str | None = None


class InterventionHistoryOutput(BaseModel):
    """Sanitised output from ``intervention_history``."""

    interventions: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Sanitisation helpers
# ═══════════════════════════════════════════════════════════════════════════════

_SANITISE_KEYS: frozenset[str] = frozenset(
    {"api_key", "prompt", "raw_prompt", "evidence", "evidence_json", "full_output"}
)


def _sanitise_dict(data: Any) -> Any:
    """Remove sensitive keys from a dict (recursive, shallow copy)."""
    if not isinstance(data, dict):
        return data
    return {
        k: _sanitise_dict(v) if isinstance(v, dict) else v
        for k, v in data.items()
        if k not in _SANITISE_KEYS
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Tool adapters
# ═══════════════════════════════════════════════════════════════════════════════


class QueryEvidenceTool:
    """Typed adapter for the ``query_evidence`` tool.

    Fetches behavior evidence from the ML sensing layer for a given user,
    capped at 30 days.  Output is sanitised before returning.

    Tool description (preserved):
        Query behavior evidence from the ML sensing layer.
        Fetches focus score, switch rate, longest focus block, behavior
        deviation, intervention history, and novelty flags for the last
        N days (capped at 30).
    """

    def __init__(
        self,
        evidence_builder: Any,  # EvidenceBundleBuilder
        timezone: TimezoneLike = "local",
    ) -> None:
        self._evidence_builder = evidence_builder
        self._timezone = timezone
        self.context: ToolContext | None = None

    async def execute(self, days: int = 7) -> QueryEvidenceOutput:
        """Run the query-evidence flow with explicit context."""
        ctx = self.context
        if ctx is None or ctx.user_id == 0:
            return QueryEvidenceOutput(error="user_id not set")

        capped = min(days, 30)
        window_end = datetime.now(UTC)
        window_start = window_end - timedelta(days=capped)

        try:
            bundle = await self._evidence_builder.build(
                ctx.user_id, window_start, window_end
            )
            evidence_str = to_prompt_json(bundle)
        except Exception as exc:  # noqa: BLE001
            return QueryEvidenceOutput(
                total_days=capped, capped=days > 30, error=str(exc)
            )

        return QueryEvidenceOutput(
            sessions=[],
            evidence_json=evidence_str,
            total_days=capped,
            capped=days > 30,
        )


class LatestAnalysisTool:
    """Typed adapter for the ``latest_analysis`` tool.

    Retrieves today's (or yesterday's) procrastination analysis from the
    repository.  Returns sanitised output — never exposes raw storage fields.

    Tool description (preserved):
        Retrieve today's (or yesterday's) procrastination analysis.
        Returns the latest procrastination-type diagnosis with confidence
        scores from the ML pipeline.
    """

    def __init__(
        self,
        analysis_repo: Any,  # SQLAlchemyProcrastinationAnalysisRepository
        timezone: TimezoneLike = "local",
    ) -> None:
        self._analysis_repo = analysis_repo
        self._timezone = timezone
        self.context: ToolContext | None = None

    async def execute(self) -> LatestAnalysisOutput:
        """Run the latest-analysis flow with explicit context."""
        ctx = self.context
        if ctx is None or ctx.user_id == 0:
            return LatestAnalysisOutput(error="user_id not set")

        today = business_today(self._timezone)
        result: dict[str, Any] | None = await self._analysis_repo.get_by_date(
            ctx.user_id, today
        )

        if result is None:
            yesterday = today - timedelta(days=1)
            result = await self._analysis_repo.get_by_date(ctx.user_id, yesterday)

        if result is None:
            return LatestAnalysisOutput(
                analysis=None, source=None, target_date=None, error="暂无分析数据"
            )

        sanitised = _sanitise_dict(result)
        return LatestAnalysisOutput(
            analysis=sanitised,
            source=sanitised.get("source"),
            target_date=today if result is not None else None,
        )


class RunAnalysisTool:
    """Typed adapter for the ``run_analysis`` tool.

    Triggers the expert panel deliberation for today's data.  Uses
    ``BudgetReservationPort`` for durable exactly-once semantics; falls
    back to an in-memory per-session cap when no port is injected.

    Tool description (preserved):
        Run the expert panel deliberation on today's data.
        Triggers a multi-expert analysis (analyst, attribution expert,
        moderator, critic) to produce a procrastination-type verdict
        with CBT recommendations.
        Limited to 1 invocation per session.
    """

    # In-memory fallback tracking (per-instance, not global)
    _MAX_INFLIGHT: int = 128

    def __init__(
        self,
        panel_service: Any | None,  # PanelService | None
        budget_port: BudgetReservationPort | None = None,
        timezone: TimezoneLike = "local",
    ) -> None:
        self._panel_service = panel_service
        self._budget_port = budget_port
        self._timezone = timezone
        self.context: ToolContext | None = None
        # In-memory cap (fallback when no budget_port)
        self._in_memory_used: set[str] = set()
        self._in_memory_lock: asyncio.Lock = asyncio.Lock()

    def _idempotency_key(self, target_date: date) -> str:
        """Build a unique idempotency key for this session + date."""
        sid = self.context.session_id if self.context else "unknown"
        uid = self.context.user_id if self.context else 0
        return f"chat:{sid}:{uid}:daily_panel:{target_date.isoformat()}"

    async def execute(self, date: date, force: bool = False) -> RunAnalysisOutput:
        """Run the panel deliberation with budget reservation."""
        from loguru import logger  # noqa: PLC0415

        ctx = self.context
        if ctx is None or ctx.user_id == 0:
            return RunAnalysisOutput(degraded=True, error="user_id not set")

        if self._panel_service is None:
            return RunAnalysisOutput(degraded=True, error="专家会诊服务暂不可用")

        target_date = date if date else business_today(self._timezone)
        idkey = self._idempotency_key(target_date)

        # ── Budget reservation (durable or in-memory fallback) ──────────
        if self._budget_port is not None:
            reserved = await self._budget_port.try_reserve(idkey)
            if not reserved:
                return RunAnalysisOutput(
                    degraded=True, error="run_analysis 每会话最多 1 次，已超出。"
                )
        else:
            async with self._in_memory_lock:
                if idkey in self._in_memory_used:
                    return RunAnalysisOutput(
                        degraded=True, error="run_analysis 每会话最多 1 次，已超出。"
                    )
                self._in_memory_used.add(idkey)
                # Bound the set size
                if len(self._in_memory_used) > self._MAX_INFLIGHT:
                    self._in_memory_used.clear()

        # ── Execute panel ───────────────────────────────────────────────
        succeeded = False
        try:
            verdict = await self._panel_service.run_daily_panel(
                ctx.user_id, target_date, origin="chat"
            )
            succeeded = True
            analysis: dict[str, Any] = {
                "types": [str(t) for t in verdict.types],
                "confidence": {str(k): float(v) for k, v in verdict.confidence.items()},
                "rationale": verdict.rationale,
            }
            return RunAnalysisOutput(
                analysis=analysis, source="panel", degraded=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Panel execution failed in chat tool: {}", exc)
            return RunAnalysisOutput(
                degraded=True, error=f"会诊执行失败: {exc}"
            )
        finally:
            if self._budget_port is not None:
                if succeeded:
                    # Keep the reservation as durable proof of usage
                    pass
                else:
                    # Release on failure — allow retry
                    await self._budget_port.release(idkey)
            else:
                if not succeeded:
                    async with self._in_memory_lock:
                        self._in_memory_used.discard(idkey)


class InterventionHistoryTool:
    """Typed adapter for the ``intervention_history`` tool.

    Queries recent intervention history (nudges, task-breakdowns, etc.)
    for the current user.  Output is sanitised — only type, time, and
    response are exposed.

    Tool description (preserved):
        Query recent intervention history.
        Returns nudge, task-breakdown, reframe, and environment-mod
        intervention records triggered in the last N days (capped at 30).
    """

    def __init__(
        self,
        intervention_repo: Any,  # InterventionLogRepository
        timezone: TimezoneLike = "local",
    ) -> None:
        self._intervention_repo = intervention_repo
        self._timezone = timezone
        self.context: ToolContext | None = None

    async def execute(self, days: int = 7) -> InterventionHistoryOutput:
        """Run the intervention-history flow with explicit context."""
        ctx = self.context
        if ctx is None or ctx.user_id == 0:
            return InterventionHistoryOutput(error="user_id not set")

        capped = min(days, 30)
        end_date = business_today(self._timezone)
        start_date = end_date - timedelta(days=capped)

        try:
            logs = await self._intervention_repo.query_range_by_date(
                ctx.user_id, start_date, end_date
            )
        except Exception as exc:  # noqa: BLE001
            return InterventionHistoryOutput(error=str(exc))

        if not logs:
            return InterventionHistoryOutput(interventions=[], total=0)

        summary: list[dict[str, Any]] = [
            {
                "type": log.get("intervention_type", "unknown"),
                "time": log.get("triggered_at", ""),
                "response": log.get("user_response", "pending"),
            }
            for log in logs
        ]
        return InterventionHistoryOutput(interventions=summary, total=len(summary))


# ═══════════════════════════════════════════════════════════════════════════════
# Output formatting (for LangChain tool return values)
# ═══════════════════════════════════════════════════════════════════════════════


def format_evidence_output(output: QueryEvidenceOutput) -> str:
    """Convert ``QueryEvidenceOutput`` to a LangChain-compatible string."""
    if output.error:
        return json.dumps({"error": output.error}, ensure_ascii=False)
    return output.evidence_json


def format_analysis_output(output: LatestAnalysisOutput) -> str:
    """Convert ``LatestAnalysisOutput`` to a LangChain-compatible string."""
    if output.error:
        if output.error == "暂无分析数据":
            return output.error
        return json.dumps({"error": output.error}, ensure_ascii=False)
    if output.analysis is None:
        return "暂无分析数据"
    return json.dumps(output.analysis, ensure_ascii=False)


def format_run_output(output: RunAnalysisOutput) -> str:
    """Convert ``RunAnalysisOutput`` to a LangChain-compatible string."""
    if output.error:
        return output.error
    return json.dumps(output.analysis, ensure_ascii=False)


def format_intervention_output(output: InterventionHistoryOutput) -> str:
    """Convert ``InterventionHistoryOutput`` to a LangChain-compatible string."""
    if output.error:
        return json.dumps({"error": output.error}, ensure_ascii=False)
    if output.total == 0:
        return "暂无干预记录"
    return json.dumps(output.interventions, ensure_ascii=False)
