# -*- coding: utf-8 -*-
"""Add scheduled job ownership heartbeat.

Revision ID: 0010_add_scheduled_job_heartbeat
Revises: 0009_create_scheduled_job_runs
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010_add_scheduled_job_heartbeat"
down_revision: str | None = "0009_create_scheduled_job_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "scheduled_job_runs",
        sa.Column("heartbeat_at", sa.Text(), nullable=True),
    )
    op.execute(
        "UPDATE scheduled_job_runs SET heartbeat_at = started_at "
        "WHERE heartbeat_at IS NULL"
    )
    with op.batch_alter_table("scheduled_job_runs") as batch_op:
        batch_op.alter_column(
            "heartbeat_at",
            existing_type=sa.Text(),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("scheduled_job_runs") as batch_op:
        batch_op.drop_column("heartbeat_at")
