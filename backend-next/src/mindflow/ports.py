"""Runtime service ports — Protocol interfaces for dependency inversion.

Repository protocols allow services to depend on abstract interfaces
rather than concrete SQLAlchemy implementations.  This keeps the service
layer testable and decoupled from persistence details.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Protocol

from mindflow.agents.types import PanelVerdict
from mindflow.domain.baseline import BaselineModel
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.domain.intervention import ThrottleStats

# ── Framework-neutral workflow value objects ───────────────────────────

OriginType = Literal["scheduler", "api", "chat", "auto_intervention"]
RunStatus = Literal["pending", "running", "completed", "failed"]


@dataclass(frozen=True)
class AnalysisRequest:
    """Request to run an analysis within a workflow.

    Attributes:
        user_id: The user to analyse.
        target_date: The date to analyse.
        analysis_kind: Storage/cache kind for this workflow (for example,
            ``"daily_panel"`` or ``"daily_attribution"``).
        force: If True, bypass idempotent cache and re-run.
        origin: Which entry point triggered this run.
        idempotency_key: Client-supplied key for exactly-once semantics.
    """

    user_id: int
    target_date: date
    force: bool = False
    origin: OriginType = "api"
    idempotency_key: str = ""
    retry_if_degraded: bool = False
    analysis_kind: str = "daily_attribution"


@dataclass(frozen=True)
class AnalysisResult:
    """Result of a single analysis within a workflow.

    Attributes:
        verdict: The panel verdict produced by the analysis.
        run_id: Identifier of the workflow run this result belongs to.
        created_at: When the analysis completed.
    """

    verdict: PanelVerdict
    run_id: str = ""
    created_at: datetime | None = None


@dataclass(frozen=True)
class WorkflowRunRequest:
    """Outer request to trigger an entire workflow run.

    Attributes:
        user_id: The user to analyse.
        target_date: The date to analyse.
        force_refresh: If True, bypass caches and force fresh analysis.
        origin: Which entry point triggered this run.
        idempotency_key: Client-supplied key for exactly-once semantics.
    """

    user_id: int
    target_date: date
    force_refresh: bool = False
    origin: OriginType = "api"
    idempotency_key: str = ""


@dataclass(frozen=True)
class WorkflowRunResult:
    """Result of a complete workflow run with status tracking.

    Attributes:
        run_id: Unique identifier for this workflow run.
        status: Current run status.
        analysis_result: The analysis outcome, populated on completion.
        error_message: Human-readable error if status is ``"failed"``.
        started_at: When the run began.
        completed_at: When the run ended (success or failure).
    """

    run_id: str
    status: RunStatus
    analysis_result: AnalysisResult | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

# ── Scheduled jobs ──────────────────────────────────────────────────────


class ScheduledJobRunsPort(Protocol):
    async def claim(
        self, job_name: str, local_date: date, *, retry_failed: bool = False
    ) -> int | None: ...
    async def heartbeat(
        self, job_name: str, local_date: date, *, attempt_count: int
    ) -> bool: ...
    async def has_succeeded(self, job_name: str, local_date: date) -> bool: ...
    async def mark_succeeded(
        self, job_name: str, local_date: date, *, attempt_count: int
    ) -> bool: ...
    async def mark_failed(
        self, job_name: str, local_date: date, *, attempt_count: int, error: str
    ) -> bool: ...
    async def mark_cancelled(
        self, job_name: str, local_date: date, *, attempt_count: int
    ) -> bool: ...


# ── Activity ────────────────────────────────────────────────────────────


class ActivityRepositoryPort(Protocol):
    """Read-side activity event queries used by services."""

    async def query_range(
        self, user_id: int, start: Any, end: Any
    ) -> list[Any]: ...
    async def query_overlapping_range(
        self, user_id: int, start: Any, end: Any
    ) -> list[Any]: ...


# ── Procrastination analysis ────────────────────────────────────────────


class ProcrastinationAnalysisRepositoryPort(Protocol):
    """Persistence for LLM attribution and panel verdict results."""

    async def get_by_date(
        self, user_id: int, target_date: date, *,
        analysis_kind: str | None = None,
    ) -> dict[str, Any] | None: ...
    async def upsert(
        self, user_id: int, target_date: date, *,
        procrastination_types: list[str],
        type_confidence: dict[str, float],
        cognitive_distortions: list[str],
        cbt_technique: str | None,
        response_text: str,
        llm_model: str | None = None,
        llm_cost_usd: float = 0.0,
        panel_transcript: dict[str, Any] | None = None,
        analysis_kind: str = "daily_attribution",
        source: str | None = None,
    ) -> None: ...
    async def exists(self, user_id: int, target_date: date) -> bool: ...


# ── Intervention ────────────────────────────────────────────────────────


class InterventionLogRepositoryPort(Protocol):
    """Persistence for intervention dispatch and tracking."""

    async def log_triggered(
        self,
        user_id: int,
        intervention_type: str,
        cbt_technique: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        intervention_id: str | None = None,
        triggered_at: datetime | None = None,
    ) -> dict[str, Any]: ...
    async def update_response(
        self,
        intervention_id: str,
        user_response: Literal["accepted", "ignored", "dismissed"],
        latency_s: float = 0.0,
    ) -> dict[str, Any] | None: ...
    async def count_today(self, user_id: int) -> int: ...
    async def count_today_by_type(self, user_id: int, intervention_type: str) -> int: ...
    async def ignore_rate_7d(self, user_id: int) -> float: ...
    async def query_range(
        self, user_id: int, start: datetime, end: datetime,
    ) -> list[dict[str, Any]]: ...
    async def update_feedback(
        self,
        intervention_id: str,
        rating: Literal["helpful", "neutral", "annoying"],
        comment: str | None = None,
    ) -> dict[str, Any] | None: ...
    async def annoying_count_7d_by_type(
        self, user_id: int, intervention_type: str,
    ) -> int: ...
    async def get_throttle_stats(
        self, user_id: int, intervention_type: str, *,
        now: datetime, today_start: datetime,
        cutoff_7d: datetime, cooldown_lower_bound: datetime,
    ) -> ThrottleStats: ...
    async def get_by_id(self, intervention_id: str) -> dict[str, Any] | None: ...
    async def query_range_by_date(
        self, user_id: int, start_date: date, end_date: date,
    ) -> list[dict[str, Any]]: ...


# ── Chat ─────────────────────────────────────────────────────────────────


class ChatRepositoryPort(Protocol):
    """Persistence for chat conversations."""

    async def append(
        self, session_id: str, role: str, content: str, *,
        user_id: int | None = None,
    ) -> None: ...
    async def recent(
        self, session_id: str, *, limit: int = 20,
    ) -> list[dict[str, Any]]: ...
    async def list_sessions(
        self, user_id: int, *, limit: int = 10,
    ) -> list[dict[str, Any]]: ...


# ── User preferences ────────────────────────────────────────────────────────


class PreferencesRepositoryPort(Protocol):
    """Persistence for per-user JSON preferences."""

    async def get(self, user_id: int) -> dict[str, Any]: ...
    async def set(self, user_id: int, preferences: dict[str, Any]) -> None: ...


# ── Baseline model ──────────────────────────────────────────────────────────


class BaselineRepositoryPort(Protocol):
    """Persistence for the user's behaviour baseline model (one row per user)."""

    async def get_latest(self, user_id: int) -> BaselineModel | None: ...

    async def upsert(self, model: BaselineModel) -> None: ...


# ── App classification rules ────────────────────────────────────────────────


class AppClassificationRulesRepositoryPort(Protocol):
    """Persistence for user-defined app classification rules."""

    async def get_all(self, user_id: int) -> list[dict[str, Any]]: ...
    async def add(
        self, user_id: int, rule: dict[str, Any],
    ) -> dict[str, Any]: ...
    async def replace_all(
        self, user_id: int, rules: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...
    async def delete(self, rule_id: str) -> None: ...
    async def get_unknown_apps(
        self, user_id: int, *, limit: int = 20,
    ) -> list[dict[str, Any]]: ...


# ── Telemetry ───────────────────────────────────────────────────────────────


class TelemetryRepositoryPort(Protocol):
    """Persistence for interaction buckets, browser segments, and features."""

    async def save_interaction_bucket(self, **values: Any) -> dict[str, Any]: ...
    async def save_browser_heartbeat(self, **values: Any) -> dict[str, Any]: ...
    async def save_authenticated_browser_heartbeat(
        self, token_hash: str, *,
        heartbeat: dict[str, Any] | None,
    ) -> tuple[bool, dict[str, Any] | None]: ...
    async def last_browser_segment_before(
        self, user_id: int, timestamp: datetime,
    ) -> dict[str, Any] | None: ...
    async def list_browser_segments(
        self, user_id: int, start: datetime, end: datetime,
    ) -> list[dict[str, Any]]: ...
    async def save_focus_feedback(
        self, user_id: int, session_id: str,
        label: str, score: int, task_type: str | None,
    ) -> dict[str, Any]: ...
    async def list_focus_feedback(
        self, user_id: int,
    ) -> list[dict[str, Any]]: ...
    async def list_interaction_buckets(
        self, user_id: int, start: datetime, end: datetime,
    ) -> list[dict[str, Any]]: ...
    async def save_feature_window(
        self, user_id: int,
        window_start_utc: datetime, window_end_utc: datetime,
        feature_schema_version: int, features_json: str,
        label: str | None = None,
    ) -> None: ...
    async def upsert_feature_windows(
        self, rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...
    async def latest_feature_window(
        self, user_id: int, feature_schema_version: int = FEATURE_SCHEMA_VERSION,
    ) -> dict[str, Any] | None: ...
    async def list_feature_windows(
        self, user_id: int, feature_schema_version: int = FEATURE_SCHEMA_VERSION,
    ) -> list[dict[str, Any]]: ...
    async def list_feature_windows_in_range(
        self, user_id: int, start: datetime, end: datetime,
        feature_schema_version: int = FEATURE_SCHEMA_VERSION,
    ) -> list[dict[str, Any]]: ...
    async def cleanup_old_telemetry(
        self, interaction_cutoff: datetime,
        activity_cutoff: datetime, feature_cutoff: datetime,
    ) -> int: ...
    async def save_browser_token(
        self, user_id: int, token_hash: str,
    ) -> None: ...
    async def verify_browser_token(self, token_hash: str) -> bool: ...
    async def revoke_browser_tokens(self, user_id: int) -> int: ...
    async def get_status(
        self, user_id: int, target_date: date,
    ) -> dict[str, Any]: ...
    async def delete_scope(
        self, user_id: int,
        scope: Literal["interaction", "browser", "feedback", "all"],
    ) -> int: ...


# ── Workflow orchestration ─────────────────────────────────────────────────


class AnalysisWorkflowPort(Protocol):
    """Framework-neutral entry point for running an expert panel analysis.

    Implementations may use LangGraph, a manual state machine, or a
    lightweight task runner — only the request/result contract matters.
    """

    async def run_analysis(self, request: AnalysisRequest) -> AnalysisResult: ...


# ── Model provider ──────────────────────────────────────────────────────────


class ModelProviderPort(Protocol):
    """Access chat model and structured attribution without framework coupling.

    Hides whether the provider is LangChain, a raw HTTP client, or a local
    Ollama model. Consumers only see typed inputs and outputs.
    """

    async def generate(self, system_prompt: str, user_message: str) -> str: ...

    async def structured_attribution(
        self,
        summary_json: str,
    ) -> object: ...


# ── Workflow run store ──────────────────────────────────────────────────────


class WorkflowRunStorePort(Protocol):
    """Status tracking and run metadata persistence for workflow runs.

    Separate from ``ScheduledJobRunsPort`` — that port tracks cron attempts
    per (job_name, local_date); this port tracks individual workflow runs
    by a generated run ID, supporting idempotency and audit.
    """

    async def save_run(self, request: WorkflowRunRequest) -> str: ...

    async def get_run(self, run_id: str) -> WorkflowRunResult | None: ...

    async def update_status(
        self, run_id: str, status: RunStatus, *,
        result: AnalysisResult | None = None,
        error: str | None = None,
    ) -> None: ...


# ── Budget reservation ──────────────────────────────────────────────────────


class BudgetReservationPort(Protocol):
    """Atomic budget check-and-reserve for workflow runs.

    Guarantees exactly-once execution for a given idempotency key: the first
    reservation succeeds, all subsequent reservations for the same key fail
    instantly (no waiting, no partial state).
    """

    async def try_reserve(
        self, idempotency_key: str, *,
        cost_estimate: float = 1.0,
    ) -> bool: ...

    async def release(self, idempotency_key: str) -> None: ...


# ── Collector intervals ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CollectorIntervalRecord:
    """Immutable snapshot of one CollectorService run interval.

    Attributes:
        id: UUIDv7 string returned by ``open()`` and used to target
            ``close()``.
        user_id: User identifier the interval belongs to.
        started_at: UTC ISO8601 timestamp when the interval opened.
        ended_at: UTC ISO8601 timestamp when the interval closed, or
            ``None`` while the interval is still open.
        reason: Human-readable reason recorded at open/close time.
        manual_stop: Flag — the interval was stopped manually.
        failure: Flag — the interval ended in failure.
        sleep: Flag — the interval ended due to system sleep.
        last_error: Error text recorded for failed intervals.
    """

    id: str
    user_id: int
    started_at: str
    ended_at: str | None
    reason: str | None
    manual_stop: bool
    failure: bool
    sleep: bool
    last_error: str | None


class CollectorIntervalsPort(Protocol):
    """Runtime collector-interval lifecycle storage.

    ``open()`` creates exactly one row; ``close()`` updates that exact
    row by ``id`` and is idempotent — a second close must not rewrite
    terminal facts. List operations are user-scoped and ordered by
    ``started_at``.
    """

    async def open(
        self,
        user_id: int,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord: ...

    async def close(
        self,
        interval_id: str,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord | None: ...

    async def list_by_user(
        self, user_id: int, *, limit: int = 100
    ) -> list[CollectorIntervalRecord]: ...

    async def list_by_user_range(
        self, user_id: int, start: datetime, end: datetime
    ) -> list[CollectorIntervalRecord]: ...
