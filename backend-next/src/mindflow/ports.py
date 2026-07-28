"""Runtime service ports — Protocol interfaces for dependency inversion.

Repository protocols allow services to depend on abstract interfaces
rather than concrete SQLAlchemy implementations.  This keeps the service
layer testable and decoupled from persistence details.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Protocol

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.intervention import ThrottleStats

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
    """Read-only access to the user's behaviour baseline model."""

    async def get_latest(self, user_id: int) -> BaselineModel | None: ...


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
    ) -> None: ...
    async def latest_feature_window(
        self, user_id: int, feature_schema_version: int = 2,
    ) -> dict[str, Any] | None: ...
    async def list_feature_windows(
        self, user_id: int, feature_schema_version: int = 2,
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
