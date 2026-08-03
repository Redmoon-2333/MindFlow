"""Add feedback time snapshots and intervention check audit table.

Revision ID: 0018_add_feedback_snapshot_checks
Revises: 0017_create_ml_shadow_predictions
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0018_add_feedback_snapshot_checks"
down_revision: str | None = "0017_create_ml_shadow_predictions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add feedback time snapshots and the intervention_checks audit table."""
    op.add_column(
        "focus_session_feedback",
        sa.Column("session_start_utc", sa.Text(), nullable=True),
    )
    op.add_column(
        "focus_session_feedback",
        sa.Column("session_end_utc", sa.Text(), nullable=True),
    )

    # Backfill from the current focus_sessions projection so existing feedback
    # participates in training even if the session table is rebuilt later.
    op.execute(
        """
        UPDATE focus_session_feedback AS f
        SET session_start_utc = (
            SELECT s.start_time FROM focus_sessions AS s
            WHERE s.id = f.session_id AND s.user_id = f.user_id
        ),
        session_end_utc = (
            SELECT s.end_time FROM focus_sessions AS s
            WHERE s.id = f.session_id AND s.user_id = f.user_id
        )
        """
    )

    op.create_table(
        "intervention_checks",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("checked_at", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("intervention_type", sa.Text(), nullable=True),
        sa.Column("throttle_reason", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("ml_status", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.Text(),
            nullable=False,
            server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
        ),
    )
    op.create_index(
        "ix_intervention_checks_user_time",
        "intervention_checks",
        ["user_id", "checked_at"],
    )


def downgrade() -> None:
    """Drop the audit table and feedback snapshot columns."""
    op.drop_index("ix_intervention_checks_user_time", table_name="intervention_checks")
    op.drop_table("intervention_checks")
    op.drop_column("focus_session_feedback", "session_end_utc")
    op.drop_column("focus_session_feedback", "session_start_utc")
