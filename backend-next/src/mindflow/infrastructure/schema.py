"""Single source of truth for all SQLAlchemy Table definitions.

All ``sa.Table(...)`` definitions live here rather than in individual
repository files, so Alembic migrations, repository queries, and any
other schema consumers share one ``metadata`` object.  This prevents
schema drift — every consumer sees the same column set, constraint
set, and type mapping.

Every table here mirrors an Alembic migration.  If you add a column
here, add it in the migration too, and vice versa.

Tables owned by this module:
  - procrastination_analyses        (migration 0001, 0002, 0011)
  - intervention_logs               (migration 0001)
  - chat_messages                   (migration 0003)
  - user_preferences                (migration 0001)
  - baseline_models                 (migration 0001; also used by train/)
  - app_classification_rules        (migration 0006)
  - interaction_buckets             (migration 0001)
  - browser_segments                (migration 0001)
  - focus_session_feedback          (migration 0001)
  - browser_tokens                  (migration 0001)
  - behavior_feature_windows        (migration 0001)
  - workflow_runs                   (migration 0013)
  - workflow_node_events            (migration 0013)
  - workflow_budget_reservations    (migration 0013)

NOTE: ``activity_events`` is intentionally NOT here — it lives in
``repositories/activity.py`` because its computed columns (JSON
extract) make the definition non-trivial to centralise.  That table
will be moved here in a follow-up.
"""

from __future__ import annotations

import sqlalchemy as sa

# ── Shared metadata — all tables bind to this one MetaData ──────────────

metadata = sa.MetaData()

# ── procrastination_analyses (from repositories/analysis.py) ──────────
# Matches migrations 0001, 0002, and 0011.

procrastination_analyses = sa.Table(
    "procrastination_analyses",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("date", sa.Text(), nullable=False),
    sa.Column("procrastination_types_json", sa.Text(), nullable=True),
    sa.Column("type_confidence_json", sa.Text(), nullable=True),
    sa.Column("cognitive_distortions_json", sa.Text(), nullable=True),
    sa.Column("cbt_technique", sa.Text(), nullable=True),
    sa.Column("response_text", sa.Text(), nullable=True),
    sa.Column("llm_model", sa.Text(), nullable=True),
    sa.Column("llm_cost_usd", sa.Float(), nullable=True),
    sa.Column("panel_transcript_json", sa.Text(), nullable=True),
    sa.Column(
        "created_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
    sa.Column("analysis_kind", sa.Text(), nullable=False, default="daily_attribution"),
    sa.Column("source", sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint("id"),
    sa.UniqueConstraint("user_id", "date", "analysis_kind"),
)

# ── intervention_logs (from repositories/intervention.py) ─────────────
# Matches migration 0001.

intervention_logs = sa.Table(
    "intervention_logs",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("triggered_at", sa.Text(), nullable=False),
    sa.Column("intervention_type", sa.Text(), nullable=False),
    sa.Column("cbt_technique", sa.Text(), nullable=True),
    sa.Column("context_json", sa.Text(), nullable=True),
    sa.Column("title", sa.Text(), nullable=True),
    sa.Column("message", sa.Text(), nullable=True),
    sa.Column("user_response", sa.Text(), nullable=True),
    sa.Column("response_latency_s", sa.Float(), nullable=True),
    sa.Column("feedback_rating", sa.Text(), nullable=True),
    sa.Column("feedback_comment", sa.Text(), nullable=True),
    sa.Column(
        "created_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
)

# ── chat_messages (from repositories/chat.py) ─────────────────────────
# Matches migration 0003 and 0012.

chat_messages = sa.Table(
    "chat_messages",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("session_id", sa.Text(), nullable=False),
    sa.Column("role", sa.Text(), nullable=False),
    sa.Column("content", sa.Text(), nullable=False),
    sa.Column(
        "created_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
    sa.CheckConstraint("role IN ('user', 'assistant')"),
    sa.Index("idx_chat_session_recent", "session_id", "created_at", "id"),
)

# ── user_preferences (from repositories/preferences.py) ───────────────
# Matches migration 0001.

user_preferences = sa.Table(
    "user_preferences",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("preferences_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
    sa.Column("updated_at", sa.Text(), nullable=False),
    sa.UniqueConstraint("user_id"),
)

# ── baseline_models (from repositories/baseline.py) ───────────────────
# Matches migration 0001.  Also referenced by train/.

baseline_models = sa.Table(
    "baseline_models",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("model_json", sa.Text(), nullable=False),
    sa.Column("training_events_count", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.Column("updated_at", sa.Text(), nullable=False),
    sa.UniqueConstraint("user_id"),
)

# ── app_classification_rules (from repositories/app_classification.py) ─
# Matches migration 0006.

app_classification_rules = sa.Table(
    "app_classification_rules",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("process_name", sa.Text(), nullable=False),
    sa.Column("window_title_pattern", sa.Text(), nullable=True),
    sa.Column("category", sa.Text(), nullable=False),
    sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.Column("updated_at", sa.Text(), nullable=False),
)

# ── Telemetry tables (from repositories/telemetry.py) ─────────────────
# Matches migration 0001.

interaction_buckets = sa.Table(
    "interaction_buckets",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("window_start_utc", sa.Text(), nullable=False),
    sa.Column("duration_s", sa.Float(), nullable=False),
    sa.Column("context_key", sa.Text(), nullable=False),
    sa.Column("keypress_count", sa.Integer(), nullable=False),
    sa.Column("mouse_click_count", sa.Integer(), nullable=False),
    sa.Column("scroll_delta", sa.Integer(), nullable=False),
    sa.Column("mouse_distance_px", sa.Float(), nullable=False),
    sa.Column("input_active_s", sa.Float(), nullable=False),
    sa.Column("interaction_burst_count", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.Text(), nullable=False),
)

browser_segments = sa.Table(
    "browser_segments",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("timestamp", sa.Text(), nullable=False),
    sa.Column("duration_s", sa.Float(), nullable=False),
    sa.Column("browser_name", sa.Text(), nullable=False),
    sa.Column("domain", sa.Text(), nullable=False),
    sa.Column("audible", sa.Boolean(), nullable=False),
    sa.Column("context_key", sa.Text(), nullable=False),
    sa.Column("created_at", sa.Text(), nullable=False),
)

focus_session_feedback = sa.Table(
    "focus_session_feedback",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("session_id", sa.Text(), nullable=False),
    sa.Column("label", sa.Text(), nullable=False),
    sa.Column("score", sa.Integer(), nullable=False),
    sa.Column("task_type", sa.Text(), nullable=True),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.UniqueConstraint("user_id", "session_id"),
)

browser_tokens = sa.Table(
    "browser_tokens",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("token_hash", sa.Text(), nullable=False, unique=True),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.Column("last_used_at", sa.Text(), nullable=True),
    sa.Column("revoked_at", sa.Text(), nullable=True),
)

behavior_feature_windows = sa.Table(
    "behavior_feature_windows",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("window_start_utc", sa.Text(), nullable=False),
    sa.Column("window_end_utc", sa.Text(), nullable=False),
    sa.Column("feature_schema_version", sa.Integer(), nullable=False),
    sa.Column("features_json", sa.Text(), nullable=False),
    sa.Column("label", sa.Text(), nullable=True),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.UniqueConstraint("user_id", "window_start_utc", "feature_schema_version"),
)

# ── Workflow orchestration tables (from repositories/workflow_runs.py) ──
# Matches migration 0013.
#
# These tables track workflow runs, per-node events, and atomic budget
# reservations.  They intentionally store ONLY runtime metadata (status,
# timestamps, counters, privacy-safe trace IDs) — NO chat text, raw
# prompts, evidence payloads, or expert content is duplicated here.

workflow_runs = sa.Table(
    "workflow_runs",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("workflow_name", sa.Text(), nullable=False),
    sa.Column("run_id", sa.Text(), nullable=False),
    sa.Column(
        "status",
        sa.Text(),
        nullable=False,
        server_default=sa.text("'pending'"),
    ),
    sa.Column("graph_version", sa.Text(), nullable=True),
    sa.Column("source", sa.Text(), nullable=True),
    sa.Column("origin", sa.Text(), nullable=False),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("target_date", sa.Text(), nullable=False),
    sa.Column("idempotency_key", sa.Text(), nullable=True, unique=True),
    sa.Column("started_at", sa.Text(), nullable=True),
    sa.Column("completed_at", sa.Text(), nullable=True),
    sa.Column("retry_reason", sa.Text(), nullable=True),
    sa.Column("degradation_reason", sa.Text(), nullable=True),
    sa.Column("token_count", sa.Integer(), nullable=True),
    sa.Column("call_count", sa.Integer(), nullable=True),
    sa.Column("trace_id", sa.Text(), nullable=True),
    sa.Column(
        "created_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
    sa.Column(
        "updated_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
    sa.UniqueConstraint("idempotency_key"),
)

workflow_node_events = sa.Table(
    "workflow_node_events",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("run_id", sa.Text(), nullable=False),
    sa.Column("node_name", sa.Text(), nullable=False),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("started_at", sa.Text(), nullable=False),
    sa.Column("completed_at", sa.Text(), nullable=True),
    sa.Column("duration_ms", sa.Integer(), nullable=True),
    sa.Column("error_category", sa.Text(), nullable=True),
)

# ── Intervention daily slot reservations (atomic throttle concurrency) ──
# Matches migration (in-memory for tests; no separate migration file).
#
# Each (user_id, date, slot_index) uniquely reserves one intervention slot
# per day.  The UNIQUE constraint acts as the atomic gate — exactly one
# concurrent caller wins the slot via INSERT ON CONFLICT DO NOTHING.
# This prevents the TOCTOU race between throttle read-check and log insert.

intervention_slot_reservations = sa.Table(
    "intervention_slot_reservations",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("date", sa.Text(), nullable=False),
    sa.Column("slot_index", sa.Integer(), nullable=False),
    sa.Column("intervention_type", sa.Text(), nullable=False),
    sa.Column(
        "reserved_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
    sa.UniqueConstraint("user_id", "date", "slot_index"),
)

workflow_budget_reservations = sa.Table(
    "workflow_budget_reservations",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("workflow_name", sa.Text(), nullable=False),
    sa.Column("run_id", sa.Text(), nullable=True),
    sa.Column("origin", sa.Text(), nullable=False),
    sa.Column("idempotency_key", sa.Text(), nullable=False, unique=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("target_date", sa.Text(), nullable=False),
    sa.Column("budget_type", sa.Text(), nullable=False),
    sa.Column(
        "reserved_at",
        sa.Text(),
        nullable=False,
        server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
    ),
    sa.Column("expires_at", sa.Text(), nullable=True),
    sa.Column("released_at", sa.Text(), nullable=True),
    sa.UniqueConstraint("idempotency_key"),
)
