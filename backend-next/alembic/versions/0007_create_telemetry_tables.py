"""Create privacy-preserving telemetry tables.

Revision ID: 0007_create_telemetry_tables
Revises: 0006_create_app_classification_rules
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0007_create_telemetry_tables"
down_revision: str | None = "0006_create_app_classification_rules"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "interaction_buckets",
        sa.Column("id", sa.Text(), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_interaction_user_time",
        "interaction_buckets",
        ["user_id", "window_start_utc"],
    )

    op.create_table(
        "browser_segments",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.Text(), nullable=False),
        sa.Column("duration_s", sa.Float(), nullable=False),
        sa.Column("browser_name", sa.Text(), nullable=False),
        sa.Column("domain", sa.Text(), nullable=False),
        sa.Column("audible", sa.Boolean(), nullable=False),
        sa.Column("context_key", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_browser_user_time",
        "browser_segments",
        ["user_id", "timestamp"],
    )

    op.create_table(
        "focus_session_feedback",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=False),
        sa.Column("task_type", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "session_id"),
    )
    op.create_index(
        "idx_feedback_user_created",
        "focus_session_feedback",
        ["user_id", "created_at"],
    )

    op.create_table(
        "browser_tokens",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("last_used_at", sa.Text(), nullable=True),
        sa.Column("revoked_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash"),
    )

    op.create_table(
        "behavior_feature_windows",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("window_start_utc", sa.Text(), nullable=False),
        sa.Column("window_end_utc", sa.Text(), nullable=False),
        sa.Column("feature_schema_version", sa.Integer(), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "window_start_utc", "feature_schema_version"),
    )
    op.create_index(
        "idx_feature_windows_user_time",
        "behavior_feature_windows",
        ["user_id", "window_start_utc"],
    )


def downgrade() -> None:
    op.drop_index("idx_feature_windows_user_time", table_name="behavior_feature_windows")
    op.drop_table("behavior_feature_windows")
    op.drop_table("browser_tokens")
    op.drop_index("idx_feedback_user_created", table_name="focus_session_feedback")
    op.drop_table("focus_session_feedback")
    op.drop_index("idx_browser_user_time", table_name="browser_segments")
    op.drop_table("browser_segments")
    op.drop_index("idx_interaction_user_time", table_name="interaction_buckets")
    op.drop_table("interaction_buckets")
