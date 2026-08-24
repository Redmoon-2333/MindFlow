"""Add (user_id, triggered_at) index to intervention_logs.

intervention_logs was created in 0001 without an index, despite being
queried by (user_id, triggered_at) range on every maybe_intervene() call
(30-min auto-check + manual triggers) via intervention_throttle.py. This
caused a full table scan on every throttle check.

Revision ID: 0004_add_intervention_logs_index
Revises: 0003_create_chat_messages
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0004_add_intervention_logs_index"
down_revision: str | None = "0003_create_chat_messages"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create (user_id, triggered_at) index on intervention_logs."""
    op.create_index(
        "idx_intervention_user_time",
        "intervention_logs",
        ["user_id", "triggered_at"],
    )


def downgrade() -> None:
    """Drop the (user_id, triggered_at) index on intervention_logs."""
    op.drop_index("idx_intervention_user_time", table_name="intervention_logs")
