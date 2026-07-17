"""Create chat_messages table for G004 conversational assistant.

Stores conversation history for the L2 chat interface:
  - One row per message (user or assistant)
  - Sessions identified by session_id (UUIDv7)
  - Role CHECK constraint enforces user/assistant
  - Composite index for session history retrieval

Revision ID: 0003_create_chat_messages
Revises: 0002_add_panel_transcript
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_create_chat_messages"
down_revision: Union[str, None] = "0002_add_panel_transcript"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create chat_messages table with role CHECK constraint and index."""
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.Text(),
            nullable=False,
            server_default=sa.text("(strftime('%Y-%m-%dT%H:%M:%SZ','now'))"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("role IN ('user', 'assistant')", name="ck_chat_role"),
    )
    op.create_index(
        "idx_chat_session_time",
        "chat_messages",
        ["user_id", "session_id", "created_at"],
    )


def downgrade() -> None:
    """Drop chat_messages table and its index."""
    op.drop_index("idx_chat_session_time", table_name="chat_messages")
    op.drop_table("chat_messages")
