"""Create the intervention_slot_reservations atomic daily-slot table.

Supports the intervention reliability transaction in
``InterventionService.maybe_intervene()``: after ``can_intervene()`` passes,
a daily slot is reserved via ``INSERT … ON CONFLICT DO NOTHING`` keyed on
``(user_id, date, slot_index)``.  The UNIQUE constraint acts as the atomic
gate — exactly one concurrent caller wins each slot, closing the TOCTOU race
between the throttle's read-only check and the ``log_triggered()`` INSERT.

The table definition mirrors ``schema.py`` byte-for-byte; no explicit
secondary indexes exist in the schema (the UNIQUE constraint supplies the
index the ``ON CONFLICT`` target requires).

Revision ID: 0020_create_intervention_slot_reservations
Revises: 0019_drop_ml_shadow_predictions
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0020_create_intervention_slot_reservations"
down_revision: str | None = "0019_drop_ml_shadow_predictions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the intervention_slot_reservations table."""
    op.create_table(
        "intervention_slot_reservations",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("date", sa.Text(), nullable=False),
        sa.Column("slot_index", sa.Integer(), nullable=False),
        sa.Column("intervention_type", sa.Text(), nullable=False),
        sa.Column(
            "reserved_at",
            sa.Text(),
            nullable=False,
            server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
        ),
        sa.UniqueConstraint("user_id", "date", "slot_index"),
    )


def downgrade() -> None:
    """Drop the intervention_slot_reservations table."""
    op.drop_table("intervention_slot_reservations")
