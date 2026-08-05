"""Create the collector_intervals runtime lifecycle table.

One row per CollectorService run: ``open()`` inserts a row with
``started_at``, ``close()`` updates that exact row by ``id`` (guarded on
``ended_at IS NULL`` so a second close cannot rewrite terminal facts),
and list operations are user-scoped with ``started_at`` ordering backed
by ``idx_collector_intervals_user_started``.

The table definition mirrors ``schema.py`` byte-for-byte.

Revision ID: 0021_create_collector_intervals
Revises: 0020_create_intervention_slot_reservations
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0021_create_collector_intervals"
down_revision: str | None = "0020_create_intervention_slot_reservations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the collector_intervals table and its user/started index."""
    op.create_table(
        "collector_intervals",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("ended_at", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "manual_stop",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "failure",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "sleep",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_collector_intervals_user_started",
        "collector_intervals",
        ["user_id", "started_at"],
    )


def downgrade() -> None:
    """Drop the collector_intervals table and its index."""
    op.drop_index(
        "idx_collector_intervals_user_started",
        table_name="collector_intervals",
    )
    op.drop_table("collector_intervals")
