"""MindFlow database schema — initial migration.

Creates all 7 core tables per architecture design (§2.2, §2.3):

  1. activity_events    — Event stream (append-mostly)
  2. focus_sessions     — Focus session projections
  3. daily_reports      — Idempotent daily reports
  4. procrastination_analyses — LLM attribution analysis results
  5. intervention_logs  — Intervention history
  6. baseline_models    — Welford online statistics (per-user JSON)
  7. user_preferences   — User settings key-value store

All primary keys use TEXT (UUIDv7).
All timestamps use TEXT (ISO8601 UTC with timezone).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_create_core_tables"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all 7 core tables with indexes."""

    # --- activity_events (append-mostly event stream) ---
    op.create_table(
        "activity_events",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Text(), nullable=False),
        sa.Column("duration_s", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("data_json", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False, server_default=sa.text("'window_snapshot'")),
        sa.Column("created_at", sa.Text(), nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_events_user_time", "activity_events", ["user_id", "timestamp"])
    op.create_index("idx_events_type", "activity_events", ["user_id", "event_type", "timestamp"])

    # --- focus_sessions (aggregated from event stream) ---
    op.create_table(
        "focus_sessions",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Text(), nullable=False),
        sa.Column("start_time", sa.Text(), nullable=False),
        sa.Column("end_time", sa.Text(), nullable=False),
        sa.Column("session_type", sa.Text(), nullable=False),
        sa.Column("dominant_app", sa.Text(), nullable=True),
        sa.Column("focus_score", sa.Float(), nullable=True),
        sa.Column("switch_count", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_sessions_user_date", "focus_sessions", ["user_id", "date"])

    # --- daily_reports (idempotent — one per user per date) ---
    op.create_table(
        "daily_reports",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Text(), nullable=False),
        sa.Column("total_focus_min", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("total_distraction_min", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("focus_score", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("top_apps_json", sa.Text(), nullable=True),
        sa.Column("switch_frequency", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("pattern_summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "date"),
    )

    # --- procrastination_analyses (LLM attribution, idempotent per date) ---
    op.create_table(
        "procrastination_analyses",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Text(), nullable=False),
        sa.Column("procrastination_types_json", sa.Text(), nullable=True),
        sa.Column("type_confidence_json", sa.Text(), nullable=True),
        sa.Column("cognitive_distortions_json", sa.Text(), nullable=True),
        sa.Column("cbt_technique", sa.Text(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("llm_model", sa.Text(), nullable=True),
        sa.Column("llm_cost_usd", sa.Float(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "date"),
    )

    # --- intervention_logs ---
    op.create_table(
        "intervention_logs",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("triggered_at", sa.Text(), nullable=False),
        sa.Column("intervention_type", sa.Text(), nullable=False),
        sa.Column("cbt_technique", sa.Text(), nullable=True),
        sa.Column("context_json", sa.Text(), nullable=True),
        sa.Column("user_response", sa.Text(), nullable=True),
        sa.Column("response_latency_s", sa.Float(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- baseline_models (per-user JSON, UNIQUE on user_id) ---
    op.create_table(
        "baseline_models",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("model_json", sa.Text(), nullable=False),
        sa.Column("training_events_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.Text(), nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")),
        sa.Column("updated_at", sa.Text(), nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )

    # --- user_preferences (per-user JSON, UNIQUE on user_id) ---
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("preferences_json", sa.Text(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("updated_at", sa.Text(), nullable=False, server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )


def downgrade() -> None:
    """Drop all 7 core tables."""
    op.drop_table("user_preferences")
    op.drop_table("baseline_models")
    op.drop_table("intervention_logs")
    op.drop_table("procrastination_analyses")
    op.drop_table("daily_reports")
    op.drop_table("focus_sessions")
    op.drop_table("activity_events")
