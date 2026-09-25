"""Feature schema v4: task-context rule store + f25..f28 feature columns.

Three additive changes behind the v3 → v4 feature-schema bump:

1. ``behavior_feature_windows.f25..f28`` — one REAL column per new v4 feature
   (``task_type_entropy``, ``task_type_dominant_ratio``,
   ``task_context_transition``, ``task_unknown_ratio``).  Existing windows keep
   NULL there, which readers treat as "not recorded"; ``task_type_code``
   (``f24``) keeps its column but becomes a real value instead of the 0.0
   placeholder.

2. ``task_context_rules`` — the small user-editable mapping layer that turns
   the existing activity categories (``app_classification_rules``) and browser
   *domains* into the observed task context.  ``match_type`` is ``category`` or
   ``domain``; a ``domain`` row stores a bare host only (never a URL path), so
   the existing domain-only privacy boundary is preserved.

3. No data is rewritten.  Old (v3 and earlier) windows stay readable by their
   own version and are ignored by v4 readers, which query by
   ``feature_schema_version``; ``TelemetryService.rebuild_feature_windows()``
   regenerates them from raw activity events when the operator wants the
   history upgraded.

Revision ID: 0027_task_context_rules_and_feature_columns
Revises: 0026_add_task_context_and_quality
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0027_task_context_rules_and_feature_columns"
down_revision: str | None = "0026_add_task_context_and_quality"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: v4 task-context feature columns (order matches the tail of
#: ``domain.feature_schema.V2_FEATURE_NAMES``).
_FEATURE_COLUMNS: tuple[str, ...] = ("f25", "f26", "f27", "f28")


def upgrade() -> None:
    """Add the v4 feature columns and the task-context rule table."""
    for col in _FEATURE_COLUMNS:
        op.add_column(
            "behavior_feature_windows",
            sa.Column(col, sa.Float(), nullable=True),
        )
    op.create_table(
        "task_context_rules",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("match_type", sa.Text(), nullable=False),
        sa.Column("match_value", sa.Text(), nullable=False),
        sa.Column("task_context", sa.Text(), nullable=False),
        sa.Column(
            "priority", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
    )
    op.create_index(
        "idx_task_context_rules_user",
        "task_context_rules",
        ["user_id", "priority"],
    )


def downgrade() -> None:
    """Drop the rule table and the v4 feature columns."""
    op.drop_index("idx_task_context_rules_user", table_name="task_context_rules")
    op.drop_table("task_context_rules")
    with op.batch_alter_table("behavior_feature_windows") as batch:
        for col in reversed(_FEATURE_COLUMNS):
            batch.drop_column(col)
