"""Add task context, quality record and label provenance columns.

Three additive, backward-compatible changes:

1. ``focus_session_feedback.context_type`` / ``goal_alignment`` — the optional
   task-relationship context. Existing rows keep NULL, which callers treat as
   ``unknown``. They are never back-filled with a guessed value: a fabricated
   ``goal_alignment`` would be indistinguishable from a real user answer.

2. ``behavior_feature_windows.quality_json`` — the per-window data-quality
   record (which collectors were enabled/available, coverage, gaps). NULL on
   historical rows means "not recorded", not "everything was fine".

3. ``focus_session_feedback.label_source`` — where the label came from
   (``user_confirmed`` / ``window_annotation`` / ``rule_inferred`` /
   ``llm_assisted``). NULL on historical rows is reported as ``unknown``.

Revision ID: 0026_add_task_context_and_quality
Revises: 0025_add_response_source
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0026_add_task_context_and_quality"
down_revision: str | None = "0025_add_response_source"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the context, provenance and quality columns."""
    op.add_column(
        "focus_session_feedback",
        sa.Column("context_type", sa.Text(), nullable=True),
    )
    op.add_column(
        "focus_session_feedback",
        sa.Column("goal_alignment", sa.Text(), nullable=True),
    )
    op.add_column(
        "focus_session_feedback",
        sa.Column("label_source", sa.Text(), nullable=True),
    )
    op.add_column(
        "behavior_feature_windows",
        sa.Column("quality_json", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_focus_feedback_context",
        "focus_session_feedback",
        ["user_id", "context_type"],
    )


def downgrade() -> None:
    """Drop the added columns (SQLite needs batch mode)."""
    op.drop_index("idx_focus_feedback_context", table_name="focus_session_feedback")
    with op.batch_alter_table("behavior_feature_windows") as batch:
        batch.drop_column("quality_json")
    with op.batch_alter_table("focus_session_feedback") as batch:
        batch.drop_column("label_source")
        batch.drop_column("goal_alignment")
        batch.drop_column("context_type")
