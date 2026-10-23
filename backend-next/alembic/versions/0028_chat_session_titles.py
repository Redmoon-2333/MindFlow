"""Add optional user-owned chat titles, leaving message history intact."""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0028_chat_session_titles"
down_revision: str | None = "0027_task_context_rules_and_feature_columns"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("user_id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.Text(), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("chat_sessions")
