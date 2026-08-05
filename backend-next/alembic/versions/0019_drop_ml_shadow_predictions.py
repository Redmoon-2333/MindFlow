"""Drop the obsolete ml_shadow_predictions table (V1 legacy, unused).

This migration is intentionally destructive: the table is no longer read by
the application, and SQLite ``DROP TABLE`` cannot preserve its rows.  The
downgrade restores the exact 0017 schema as an empty table; deployments that
need the old data must restore a database backup before downgrading.

Revision ID: 0019_drop_ml_shadow_predictions
Revises: 0018_add_feedback_snapshot_checks
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0019_drop_ml_shadow_predictions"
down_revision: str | None = "0018_add_feedback_snapshot_checks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the unused shadow-prediction table."""
    op.drop_table("ml_shadow_predictions")


def downgrade() -> None:
    """Restore the 0017 table shape; rows removed by upgrade are unrecoverable."""
    op.create_table(
        "ml_shadow_predictions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("window_start_utc", sa.Text(), nullable=False),
        sa.Column("candidate_version", sa.Text(), nullable=False),
        sa.Column("active_version", sa.Text(), nullable=True),
        sa.Column("candidate_proba", sa.Float(), nullable=False),
        sa.Column("active_proba", sa.Float(), nullable=True),
        sa.Column("delta", sa.Float(), nullable=True),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'shadow'"),
        ),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.UniqueConstraint("user_id", "window_start_utc", "candidate_version"),
    )
