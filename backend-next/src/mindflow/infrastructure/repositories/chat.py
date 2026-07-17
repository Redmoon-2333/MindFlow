"""SQLAlchemy-backed ChatRepository for G004 conversational assistant.

Stores and queries chat messages (one row per user/assistant message).

Table schema matches the Alembic migration (0003_create_chat_messages):

  chat_messages:
    id          TEXT PK (UUIDv7)
    user_id     INTEGER NOT NULL
    session_id  TEXT NOT NULL
    role        TEXT NOT NULL (CHECK: 'user' | 'assistant')
    content     TEXT NOT NULL
    created_at  TEXT NOT NULL (ISO8601 UTC)

Design:
  - Follows the InterventionLogRepository pattern (SQLAlchemy Core + async).
  - Session-based: all messages for a conversation share a session_id.
  - User-bound: user_id on each row for user-scoped queries.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id

# ── Table definition (matches migration 0003_create_chat_messages) ─────

chat_messages = sa.Table(
    "chat_messages",
    sa.MetaData(),
    sa.Column("id", sa.Text(), primary_key=True),
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
)


# ── Repository ─────────────────────────────────────────────────────────


class ChatRepository:
    """Chat message repository backed by SQLAlchemy Core + async SQLite.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory

    # ── Public API ────────────────────────────────────────────────────

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        user_id: int = 1,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        """Append a message to a chat session.

        Args:
            session_id: The conversation session identifier.
            role: "user" or "assistant".
            content: The message text.
            user_id: User identifier (default 1 for single-user mode).
            message_id: Override the auto-generated ID (for testing).

        Returns:
            The inserted row as a dict.
        """
        row = {
            "id": message_id or new_id(),
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
        }

        async with self._session_factory() as session, session.begin():
            await session.execute(chat_messages.insert().values(**row))

        return {**row}

    async def recent(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return the most recent messages for a session, oldest-first.

        Internally queries DESC + LIMIT for efficiency, then reverses in
        Python to maintain oldest-first ordering.

        Args:
            session_id: The conversation session identifier.
            limit: Maximum number of messages to return (default 20).

        Returns:
            List of message dicts sorted by created_at ascending.
        """
        stmt = (
            sa.select(chat_messages)
            .where(chat_messages.c.session_id == session_id)
            .order_by(chat_messages.c.created_at.desc(), chat_messages.c.id.desc())
            .limit(limit)
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = [_row_to_dict(row) for row in result.fetchall()]
            # Reverse to oldest-first
            rows.reverse()
            return rows

    async def list_sessions(
        self,
        user_id: int,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """List the most recent distinct sessions for a user.

        Returns session_id and the timestamp of the last message in each
        session, ordered by most recent activity first.  Uses the message
        ``id`` (UUIDv7, time-sortable) as a tiebreaker when multiple
        sessions have the same ``created_at`` second.

        Args:
            user_id: User identifier.
            limit: Maximum number of sessions to return (default 10).

        Returns:
            List of dicts with ``session_id`` and ``last_message_at`` keys.
        """
        # Use a window-function approach to get the latest message per session
        latest = (
            sa.select(
                chat_messages.c.session_id,
                chat_messages.c.created_at,
                chat_messages.c.id,
                sa.func.row_number()
                .over(
                    partition_by=chat_messages.c.session_id,
                    order_by=sa.desc(chat_messages.c.created_at),
                )
                .label("rn"),
            )
            .where(chat_messages.c.user_id == user_id)
            .subquery()
        )

        # Filter to rn=1 (latest message per session), order by created_at DESC, id DESC
        stmt = (
            sa.select(
                latest.c.session_id,
                latest.c.created_at.label("last_message_at"),
            )
            .where(latest.c.rn == 1)
            .order_by(sa.desc(latest.c.created_at), sa.desc(latest.c.id))
            .limit(limit)
        )

        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return [
                {
                    "session_id": row.session_id,
                    "last_message_at": row.last_message_at,
                }
                for row in result.fetchall()
            ]

    def __repr__(self) -> str:
        return "<ChatRepository>"


# ── Serialisation helper ───────────────────────────────────────────────


def _row_to_dict(row: sa.Row[Any]) -> dict[str, Any]:
    """Convert a ``chat_messages`` row to a plain dict."""
    return {
        "id": row.id,
        "user_id": row.user_id,
        "session_id": row.session_id,
        "role": row.role,
        "content": row.content,
        "created_at": row.created_at,
    }
