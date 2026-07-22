"""Add feedback columns to intervention_logs.

Adds user-provided feedback on intervention effectiveness:
  - feedback_rating: "helpful" | "neutral" | "annoying"
  - feedback_comment: optional free-text comment

The rating drives throttle adjustments: 3+ "annoying" ratings in 7 days
for a type reduces that type's daily limit.

Revision ID: 0005_add_intervention_feedback
Revises: 0004_add_intervention_logs_index
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_add_intervention_feedback"
down_revision: Union[str, None] = "0004_add_intervention_logs_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add feedback_rating and feedback_comment columns."""
    op.add_column(
        "intervention_logs",
        sa.Column("feedback_rating", sa.Text(), nullable=True),
    )
    op.add_column(
        "intervention_logs",
        sa.Column("feedback_comment", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Drop feedback columns."""
    op.drop_column("intervention_logs", "feedback_comment")
    op.drop_column("intervention_logs", "feedback_rating")
