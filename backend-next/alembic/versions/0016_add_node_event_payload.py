"""Add trace payload to workflow_node_events.

Revision ID: 0016_add_node_event_payload
"""

import sqlalchemy as sa

from alembic import op

revision: str = "0016_add_node_event_payload"
down_revision: str | None = "0015_add_panel_degradation_meta"


def upgrade() -> None:
    op.add_column("workflow_node_events", sa.Column("payload_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("workflow_node_events", "payload_json")
