# -*- coding: utf-8 -*-
"""Create app_classification_rules table.

Stores per-process-name and per-window-title classification rules that
determine how MindFlow categorises activity.  Rules are ordered by priority
(descending); the first matching rule wins.

Columns:
  - id:                  UUIDv7 primary key
  - user_id:             1 (single-user desktop app)
  - process_name:        e.g. "bilibili.exe", "notion.exe"
  - window_title_pattern:  optional SQL LIKE pattern for matching window titles
  - category:            one of "code"/"document"/"browser_work"/
                         "communication"/"entertainment"/"social"/"other"
  - priority:            higher value = checked first (default 0)
  - created_at / updated_at: ISO8601 UTC text timestamps

Revision ID: 0006_create_app_classification_rules
Revises: 0005_add_intervention_feedback
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_create_app_classification_rules"
down_revision: Union[str, None] = "0005_add_intervention_feedback"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create app_classification_rules table with priority index."""
    op.create_table(
        "app_classification_rules",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("process_name", sa.Text(), nullable=False),
        sa.Column("window_title_pattern", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_app_rules_user_priority",
        "app_classification_rules",
        ["user_id", sa.text("priority DESC")],
    )


def downgrade() -> None:
    """Drop app_classification_rules table and index."""
    op.drop_index("idx_app_rules_user_priority", table_name="app_classification_rules")
    op.drop_table("app_classification_rules")
