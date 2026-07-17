"""Tests for ChatRepository (infrastructure/repositories/chat.py).

Covers:
  - append: basic insert with and without custom ID
  - append: different roles (user, assistant)
  - recent: returns messages in order, respects limit
  - recent: scoped to session_id
  - list_sessions: returns distinct sessions ordered by most recent
  - list_sessions: respects limit
"""

from __future__ import annotations

from typing import Any

import pytest

from mindflow.infrastructure.repositories.chat import ChatRepository, chat_messages


@pytest.fixture
async def chat_tables(engine):
    """Create the chat_messages table."""
    async with engine.begin() as conn:
        await conn.run_sync(chat_messages.metadata.create_all)


@pytest.fixture
def repo(session_factory, chat_tables) -> ChatRepository:
    """Create a ChatRepository with tables created."""
    return ChatRepository(session_factory=session_factory)


class TestAppend:
    """Message insertion tests."""

    async def test_append_user_message(self, repo: ChatRepository) -> None:
        """Append a user message returns the row with correct fields."""
        result = await repo.append(
            session_id="session-001",
            role="user",
            content="今天专注怎么样？",
            user_id=1,
        )
        assert result["session_id"] == "session-001"
        assert result["role"] == "user"
        assert result["content"] == "今天专注怎么样？"
        assert result["user_id"] == 1
        assert "id" in result

    async def test_append_assistant_message(self, repo: ChatRepository) -> None:
        """Append an assistant message."""
        result = await repo.append(
            session_id="session-001",
            role="assistant",
            content="你好！我是 MindFlow 助手。",
            user_id=1,
        )
        assert result["role"] == "assistant"
        assert result["content"] == "你好！我是 MindFlow 助手。"

    async def test_append_with_custom_id(self, repo: ChatRepository) -> None:
        """Custom message ID is persisted."""
        result = await repo.append(
            session_id="session-001",
            role="user",
            content="test",
            user_id=1,
            message_id="custom-id-001",
        )
        assert result["id"] == "custom-id-001"

    async def test_append_default_user_id(self, repo: ChatRepository) -> None:
        """Default user_id is 1."""
        result = await repo.append(
            session_id="session-001",
            role="user",
            content="test",
        )
        assert result["user_id"] == 1

    async def test_append_different_sessions(self, repo: ChatRepository) -> None:
        """Messages in different sessions are stored independently."""
        await repo.append(session_id="s1", role="user", content="msg1", user_id=1)
        await repo.append(session_id="s2", role="user", content="msg2", user_id=1)

        s1_msgs = await repo.recent("s1")
        s2_msgs = await repo.recent("s2")

        assert len(s1_msgs) == 1
        assert s1_msgs[0]["content"] == "msg1"
        assert len(s2_msgs) == 1
        assert s2_msgs[0]["content"] == "msg2"


class TestRecent:
    """Session history retrieval tests."""

    async def test_recent_empty_session(self, repo: ChatRepository) -> None:
        """Empty session returns empty list."""
        msgs = await repo.recent("non-existent-session")
        assert msgs == []

    async def test_recent_ordered(self, repo: ChatRepository) -> None:
        """Messages are returned in chronological order."""
        await repo.append(session_id="s1", role="user", content="first", user_id=1)
        await repo.append(session_id="s1", role="assistant", content="second", user_id=1)
        await repo.append(session_id="s1", role="user", content="third", user_id=1)

        msgs = await repo.recent("s1")
        assert len(msgs) == 3
        assert msgs[0]["content"] == "first"
        assert msgs[1]["content"] == "second"
        assert msgs[2]["content"] == "third"

    async def test_recent_respects_limit(self, repo: ChatRepository) -> None:
        """Limit parameter returns only the last N messages."""
        for i in range(10):
            await repo.append(session_id="s1", role="user", content=f"msg{i}", user_id=1)
            await repo.append(session_id="s1", role="assistant", content=f"rsp{i}", user_id=1)

        msgs = await repo.recent("s1", limit=5)
        assert len(msgs) == 5
        # Oldest-first: the 5 most recent messages in chronological order.
        # With 10 pairs, the most recent 5 are the last 5 inserted: rsp7, msg8, rsp8, msg9, rsp9
        # (oldest-first after the DESC+reverse). The first message is rsp7.
        assert msgs[0]["content"] == "rsp7"

    async def test_recent_scoped_to_session(self, repo: ChatRepository) -> None:
        """Recent only returns messages for the given session."""
        await repo.append(session_id="s1", role="user", content="s1-msg", user_id=1)
        await repo.append(session_id="s2", role="user", content="s2-msg", user_id=1)

        msgs = await repo.recent("s1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "s1-msg"


class TestListSessions:
    """Session listing tests."""

    async def test_list_sessions_empty(self, repo: ChatRepository) -> None:
        """No sessions returns empty list."""
        sessions = await repo.list_sessions(user_id=1)
        assert sessions == []

    async def test_list_sessions_orders_by_recent(self, repo: ChatRepository) -> None:
        """Sessions ordered by last message time, most recent first."""
        await repo.append(session_id="old", role="user", content="old msg", user_id=1)
        await repo.append(session_id="new", role="user", content="new msg", user_id=1)

        sessions = await repo.list_sessions(user_id=1)
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "new"
        assert sessions[1]["session_id"] == "old"

    async def test_list_sessions_respects_limit(self, repo: ChatRepository) -> None:
        """Limit caps the number of sessions returned."""
        for i in range(5):
            await repo.append(session_id=f"s{i}", role="user", content="msg", user_id=1)

        sessions = await repo.list_sessions(user_id=1, limit=3)
        assert len(sessions) == 3

    async def test_list_sessions_scoped_to_user(self, repo: ChatRepository) -> None:
        """Sessions are scoped by user_id."""
        await repo.append(session_id="s1", role="user", content="u1 msg", user_id=1)
        await repo.append(session_id="s2", role="user", content="u2 msg", user_id=2)

        sessions_u1 = await repo.list_sessions(user_id=1)
        sessions_u2 = await repo.list_sessions(user_id=2)

        assert len(sessions_u1) == 1
        assert sessions_u1[0]["session_id"] == "s1"
        assert len(sessions_u2) == 1
        assert sessions_u2[0]["session_id"] == "s2"

    async def test_list_sessions_has_last_message_time(self, repo: ChatRepository) -> None:
        """Each session entry includes a last_message_at timestamp."""
        await repo.append(session_id="s1", role="user", content="msg", user_id=1)

        sessions = await repo.list_sessions(user_id=1)
        assert len(sessions) == 1
        assert "last_message_at" in sessions[0]
        assert sessions[0]["last_message_at"] is not None
