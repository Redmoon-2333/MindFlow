"""Typed request and response schemas for the public MindFlow API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from mindflow.domain.prediction import FocusPredictionStatus


class CollectorStatusResponse(BaseModel):
    status: str
    running: bool
    message: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tools_used: list[str] = Field(default_factory=list)
    evidence_cited: bool = False
    degraded: bool = False


class PanelTranscriptEntry(BaseModel):
    role: str
    content: str
    round: int


class PanelMeta(BaseModel):
    degraded: bool


class PanelResponse(BaseModel):
    types: list[str]
    confidence: dict[str, float]
    technique: str | None
    rationale: str
    dissent: list[str]
    transcript: list[PanelTranscriptEntry]
    escalated: bool
    call_count: int
    degraded: bool
    meta: PanelMeta


InterventionIntensityValue = Literal["gentle", "standard", "strict"]
InterventionUserResponse = Literal["accepted", "ignored", "dismissed"]
InterventionFeedbackRating = Literal["helpful", "neutral", "annoying"]


class InterventionTriggerRequest(BaseModel):
    intensity: InterventionIntensityValue = "standard"


class InterventionResponseRequest(BaseModel):
    response: InterventionUserResponse
    latency_s: float = Field(default=0.0, ge=0.0)


class InterventionFeedbackRequest(BaseModel):
    rating: InterventionFeedbackRating
    comment: str | None = None


class InterventionPayload(BaseModel):
    id: str
    intervention_type: str
    title: str
    message: str
    dismissible: bool
    created_at: datetime


class InterventionTriggerResponse(BaseModel):
    intervention: InterventionPayload | None
    skipped: bool
    skip_reason: str | None = None


class InterventionCommandResponse(BaseModel):
    status: Literal["ok"] = "ok"
    intervention_id: str
    user_response: InterventionUserResponse | None = None
    feedback_rating: InterventionFeedbackRating | None = None


class InterventionHistoryResponse(BaseModel):
    items: list[dict[str, Any]]
    count: int
    has_more: bool = False
    next_cursor: str | None = None


# ── AI Diagnostics schemas ──────────────────────────────────────────────


class NodeEventSummary(BaseModel):
    """Sanitised node event: structural metadata only — no prompts or content."""

    node_name: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    error_category: str | None = None


class RunSummary(BaseModel):
    """Abbreviated workflow run for list endpoints."""

    run_id: str
    workflow_name: str
    status: str
    graph_version: str | None = None
    source: str | None = None
    origin: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    call_count: int | None = None
    degraded: bool = False


class RunDetail(BaseModel):
    """Full workflow run with node events — allowlisted, no PII or content."""

    run_id: str
    workflow_name: str
    status: str
    graph_version: str | None = None
    source: str | None = None
    origin: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    call_count: int | None = None
    token_estimate: int | None = None
    degraded: bool = False
    node_events: list[NodeEventSummary] = Field(default_factory=list)


class DiagnosticsListResponse(BaseModel):
    """Paginated list of workflow runs."""

    items: list[RunSummary]
    count: int
    has_more: bool = False
    next_offset: int | None = None


# ── Training readiness schemas ───────────────────────────────────────────


GateStatus = Literal["passed", "failed", "not_evaluated", "not_implemented"]


class V2GateCheck(BaseModel):
    """One of the seven V2 quality-gate checks with honest status."""

    key: str
    label: str
    passed: bool
    status: GateStatus
    actual: str
    threshold: str
    message: str  # Chinese, user-readable
    blocker_code: str  # Machine-readable code when failed


class ActivityEventsSummary(BaseModel):
    """Aggregate summary of raw activity events (from activity_events table)."""

    total_events: int
    coverage_days: int
    oldest_timestamp: str | None
    newest_timestamp: str | None


class V2WindowsSummary(BaseModel):
    """V2 (24-dim) feature-window summary including matched eligibility."""

    total: int
    schema_version: int = 2
    date_range_days: int
    eligible_count: int  # matched to explicit feedback via time overlap
    matched_focus_count: int  # eligible windows labelled as focus
    matched_distract_count: int  # eligible windows labelled as distract
    newest_window_start: str | None


class FeedbackLabelSummary(BaseModel):
    """Raw explicit feedback distribution from focus_session_feedback table."""

    focus: int
    distract: int
    mixed: int
    total: int


class Blocker(BaseModel):
    """A single gate-blocker with machine code and Chinese message."""

    code: str
    message: str


# ── Training job schemas ────────────────────────────────────────────────


JobStatus = Literal[
    "pending", "preparing_data", "training",
    "succeeded", "failed", "cancelled",
]


class TrainingJobSummary(BaseModel):
    """Lightweight job summary (used in readiness response)."""

    job_id: str
    status: JobStatus
    started_at: str | None = None
    completed_at: str | None = None


class TrainingJobResponse(BaseModel):
    """Full training-job lifecycle status with optional report/error."""

    job_id: str
    status: JobStatus
    source: str = "db"
    model_mode: str = "rule_engine_only"
    started_at: str | None = None
    completed_at: str | None = None
    activated: bool = False
    version_tag: str | None = None
    feature_schema_version: int | None = None
    quality_gate: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    error: str | None = None


class CreateTrainingJobResponse(BaseModel):
    """202 response after a training job is accepted."""

    job_id: str
    status: JobStatus = "pending"


class TrainingReadinessResponse(BaseModel):
    """Complete training readiness assessment for V2 feature-schema models.

    Reports raw activity events (from activity_events), V2 feature windows
    matched to explicit feedback via time overlap, feedback label
    distribution, trainability (>=10 eligible matched windows with >=2
    unique labels), evaluability (>=10 explicit matched samples with
    enough distinct dates for grouped evaluation), baseline readiness,
    current model mode, the seven V2 gate checks, blocker codes, and
    the active/latest training job status (if any).
    """

    raw_events: ActivityEventsSummary
    v2_windows: V2WindowsSummary
    feedback_labels: FeedbackLabelSummary
    trainable: bool
    trainable_window_count: int  # >=10, >=2 classes: matched windows
    trainable_class_count: int   # number of unique label classes
    evaluable: bool
    evaluable_explicit_count: int   # >=10: explicit matched samples
    evaluable_date_count: int       # distinct feedback days
    baseline_ready: bool
    current_mode: str  # ready / shadow / rule_engine_only
    gates: list[V2GateCheck]
    blockers: list[Blocker]
    current_training_job: TrainingJobSummary | None


# ── Report schemas ────────────────────────────────────────────────────


class TopAppEntry(BaseModel):
    """An application entry with name and usage duration."""

    app: str
    minutes: float


# Exact data-state vocabularies shared by the service and the API contract.
DailyDataState = Literal[
    "ready",
    "no_activity",
    "events_only",
    "neutral_only",
    "no_focus",
    "future",
]
WeeklyDataState = Literal["ready", "partial", "no_activity", "future"]


class DailyReportResponse(BaseModel):
    """Typed daily report returned by ``GET /reports/daily``.

    ``top_apps`` is always a list (possibly empty) — never null.  The
    transient frontend fields (``total_focus_minutes`` /
    ``total_sessions`` / ``total_distractions`` / ``hourly_distribution``)
    and ``data_state`` are attached by the report service on both the
    freshly generated and the cached paths, so every 200 response exposes
    the same canonical field set.
    """

    id: str
    user_id: int
    date: str
    total_focus_min: float
    total_distraction_min: float
    focus_score: float
    top_apps: list[TopAppEntry] = Field(default_factory=list)
    switch_frequency: float
    pattern_summary: str
    created_at: str | None = None
    total_focus_minutes: float
    total_sessions: int
    total_distractions: int
    hourly_distribution: dict[str, float]
    data_state: DailyDataState


class WeeklyReportResponse(BaseModel):
    """Typed weekly report returned by ``GET /reports/weekly``.

    Expected empty/future weeks return 200 with ``daily_reports == []`` and
    an explicit ``data_state`` instead of a 404.  ``daily_summary`` entries
    mirror the canonical per-day totals consumed by the frontend.
    """

    week_start: str
    week_end: str
    daily_reports: list[DailyReportResponse] = Field(default_factory=list)
    averages: dict[str, float] = Field(default_factory=dict)
    trend: dict[str, Any] = Field(default_factory=dict)
    week_number: int
    intervention_effectiveness: dict[str, Any] | None = None
    total_focus_minutes: float
    total_sessions: int
    total_distractions: int
    avg_focus_score: float
    daily_summary: list[dict[str, Any]] = Field(default_factory=list)
    data_state: WeeklyDataState


# ── Focus prediction schemas ─────────────────────────────────────────────


class FocusPredictionResponse(BaseModel):
    """Canonical typed response for ``GET /telemetry/focus-prediction``.

    ``focus_probability`` is always present: a number in [0, 1] for
    ``ready``, ``None`` for every non-ready status. ``status`` uses exactly
    the domain's six-value Literal so the API contract cannot drift from
    ``domain/prediction.py``. ``mode`` is preserved as-is (e.g.
    ``"rule_engine_only"``) and ``reason`` carries the human-readable
    explanation.
    """

    focus_probability: float | None = Field(ge=0.0, le=1.0)
    status: FocusPredictionStatus
    mode: str
    reason: str
