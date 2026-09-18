"""Create training_jobs table (persistent training job lifecycle).

``TrainingJobService`` previously kept job state only in memory, so a process
restart lost all record of an in-flight run: the UI could show nothing, or a
stale ``training`` job could look like it never existed. A training run can
also be killed mid-write by a crash or a hard stop, and nothing recorded that
the model directory might contain a partially published candidate.

This table stores one row per job. On startup the service marks any row still
in a non-terminal state as ``interrupted`` (see
``TrainingJobRepository.mark_interrupted``), so the lifecycle is honest across
restarts without pretending the run completed.

Revision ID: 0024_create_training_jobs
Revises: 0023_create_tasks_and_blocked_sites
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0024_create_training_jobs"
down_revision: str | None = "0023_create_tasks_and_blocked_sites"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the training_jobs table."""
    op.create_table(
        "training_jobs",
        sa.Column("job_id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("source", sa.Text(), nullable=False, server_default=sa.text("'db'")),
        sa.Column(
            "model_mode",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'rule_engine_only'"),
        ),
        sa.Column("started_at", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.Text(), nullable=True),
        sa.Column("activated", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("version_tag", sa.Text(), nullable=True),
        sa.Column("feature_schema_version", sa.Integer(), nullable=True),
        sa.Column("quality_gate_json", sa.Text(), nullable=True),
        sa.Column("evaluation_json", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('pending', 'preparing_data', 'training', 'succeeded', "
            "'failed', 'cancelled', 'interrupted')"
        ),
    )
    op.create_index(
        "idx_training_jobs_user_started",
        "training_jobs",
        ["user_id", "started_at"],
    )
    op.create_index(
        "idx_training_jobs_status",
        "training_jobs",
        ["status"],
    )


def downgrade() -> None:
    """Drop the training_jobs table."""
    op.drop_index("idx_training_jobs_status", table_name="training_jobs")
    op.drop_index("idx_training_jobs_user_started", table_name="training_jobs")
    op.drop_table("training_jobs")
