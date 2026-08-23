# -*- coding: utf-8 -*-
"""Create tasks and blocked_sites tables (intervention execution, plan G).

``tasks`` is the first real data source for the ``smart_prioritization``
intervention: pending tasks ranked by deadline proximity and priority.
``blocked_sites`` backs the ``environment_optimization`` intervention's
execution path — the browser extension polls the enabled rows and applies
declarativeNetRequest rules.

Revision ID: 0023_create_tasks_and_blocked_sites
Revises: 0022_feature_window_columns
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0023_create_tasks_and_blocked_sites"
down_revision: str | None = "0022_feature_window_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the tasks and blocked_sites tables."""
    op.create_table(
        "tasks",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("priority", sa.Integer(), nullable=False, server_default=sa.text("3")),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("deadline_utc", sa.Text(), nullable=True),
        sa.Column("estimated_minutes", sa.Integer(), nullable=True),
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
        sa.CheckConstraint("priority >= 1 AND priority <= 5"),
        sa.CheckConstraint("status IN ('pending', 'in_progress', 'done')"),
    )
    op.create_index(
        "idx_tasks_user_status_deadline",
        "tasks",
        ["user_id", "status", "deadline_utc"],
    )

    op.create_table(
        "blocked_sites",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("domain", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("reason", sa.Text(), nullable=True),
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
        sa.UniqueConstraint("user_id", "domain"),
    )


def downgrade() -> None:
    """Drop blocked_sites then tasks."""
    op.drop_table("blocked_sites")
    op.drop_table("tasks")
