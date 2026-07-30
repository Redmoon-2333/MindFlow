"""Add title and message columns to intervention_logs.

Revision ID: 0014_add_intervention_title_message
"""

from alembic import op
import sqlalchemy as sa

revision: str = "0014_add_intervention_title_message"
down_revision: str | None = "0013_create_workflow_tables"


def upgrade() -> None:
    op.add_column("intervention_logs", sa.Column("title", sa.Text(), nullable=True))
    op.add_column("intervention_logs", sa.Column("message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("intervention_logs", "message")
    op.drop_column("intervention_logs", "title")
