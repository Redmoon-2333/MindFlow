# -*- coding: utf-8 -*-
"""Create ml_shadow_predictions table.

Revision ID: 0017_create_ml_shadow_predictions
"""

from alembic import op
import sqlalchemy as sa

revision: str = "0017_create_ml_shadow_predictions"
down_revision: str | None = "0016_add_node_event_payload"


def upgrade() -> None:
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


def downgrade() -> None:
    op.drop_table("ml_shadow_predictions")
