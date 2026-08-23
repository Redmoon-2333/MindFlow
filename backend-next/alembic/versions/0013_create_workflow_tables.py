# -*- coding: utf-8 -*-
"""Create workflow_runs, workflow_node_events, and workflow_budget_reservations.

Adds durable workflow telemetry tables for tracking workflow runs, node-level
events, and atomic budget reservations.  No chat text, raw prompts, evidence
payloads, or expert content is stored in these telemetry tables — only runtime
metadata (status, timestamps, counters, trace IDs).

Tables:
  - workflow_runs: Per-run status, graph version, origin, token/call counters,
    privacy-safe trace IDs, and idempotency key with UNIQUE constraint.
  - workflow_node_events: Per-node start/completion events with duration and
    error category (no prompt/payload content).
  - workflow_budget_reservations: Atomic reservation via UNIQUE on
    idempotency_key — guarantees exactly-one winner for concurrent calls.

Revision ID: 0013_create_workflow_tables
Revises: 0012_add_chat_session_recent_index
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0013_create_workflow_tables"
down_revision: str | None = "0012_add_chat_session_recent_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create workflow_runs, workflow_node_events, workflow_budget_reservations."""
    op.create_table(
        "workflow_runs",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("workflow_name", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("graph_version", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("origin", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("target_date", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=True, unique=True),
        sa.Column("started_at", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.Text(), nullable=True),
        sa.Column("retry_reason", sa.Text(), nullable=True),
        sa.Column("degradation_reason", sa.Text(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("call_count", sa.Integer(), nullable=True),
        sa.Column("trace_id", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )

    op.create_table(
        "workflow_node_events",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("node_name", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
        ),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_category", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "workflow_budget_reservations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("workflow_name", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=True),
        sa.Column("origin", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False, unique=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("target_date", sa.Text(), nullable=False),
        sa.Column("budget_type", sa.Text(), nullable=False),
        sa.Column(
            "reserved_at",
            sa.Text(),
            nullable=False,
            server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
        ),
        sa.Column("expires_at", sa.Text(), nullable=True),
        sa.Column("released_at", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )

    op.create_index(
        "ix_wfr_status_target_date",
        "workflow_runs",
        ["status", "target_date"],
    )
    op.create_index(
        "ix_wfr_user_date",
        "workflow_runs",
        ["user_id", "target_date"],
    )
    op.create_index(
        "ix_wne_run_node",
        "workflow_node_events",
        ["run_id", "node_name"],
    )
    op.create_index(
        "ix_wbr_key",
        "workflow_budget_reservations",
        ["idempotency_key"],
    )


def downgrade() -> None:
    """Drop workflow tables and their indexes."""
    op.drop_index("ix_wbr_key", table_name="workflow_budget_reservations")
    op.drop_index("ix_wne_run_node", table_name="workflow_node_events")
    op.drop_index("ix_wfr_user_date", table_name="workflow_runs")
    op.drop_index("ix_wfr_status_target_date", table_name="workflow_runs")
    op.drop_table("workflow_budget_reservations")
    op.drop_table("workflow_node_events")
    op.drop_table("workflow_runs")
