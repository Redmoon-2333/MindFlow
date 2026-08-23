# -*- coding: utf-8 -*-
"""Create persistent scheduled job run claims.

Revision ID: 0009_create_scheduled_job_runs
Revises: 0008_optimize_activity_telemetry
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009_create_scheduled_job_runs"
down_revision: str | None = "0008_optimize_activity_telemetry"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "scheduled_job_runs",
        sa.Column("job_name", sa.Text(), nullable=False),
        sa.Column("local_date", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("finished_at", sa.Text(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("job_name", "local_date"),
    )


def downgrade() -> None:
    op.drop_table("scheduled_job_runs")
