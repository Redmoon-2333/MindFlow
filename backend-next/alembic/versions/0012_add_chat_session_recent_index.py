# -*- coding: utf-8 -*-
"""Add composite index idx_chat_session_recent for ChatRepository.recent().

The ``recent()`` query filters on ``session_id``, then orders by
``created_at DESC, id DESC``.  The existing ``idx_chat_session_time``
(user_id, session_id, created_at) covers ``list_sessions()`` but cannot
lead with ``session_id``, so ``recent()`` falls back to a full table scan.

This migration adds ``idx_chat_session_recent(session_id, created_at, id)``
without touching the existing index, giving SQLite a covering index that
directly satisfies the ``recent()`` predicate + sort.

Revision ID: 0012_add_chat_session_recent_index
Revises: 0011_add_analysis_kind
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0012_add_chat_session_recent_index"
down_revision: str | None = "0011_add_analysis_kind"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create idx_chat_session_recent covering recent() predicate + sort.

    Index columns: (session_id, created_at, id)

    ``IF NOT EXISTS`` guards against repeated upgrade invocation (makes the
    migration idempotent, which is a safe practice for SQLite index-only
    migrations).
    """
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_session_recent "
        "ON chat_messages(session_id, created_at, id)"
    )


def downgrade() -> None:
    """Drop idx_chat_session_recent only — preserves idx_chat_session_time."""
    op.drop_index("idx_chat_session_recent", table_name="chat_messages")
