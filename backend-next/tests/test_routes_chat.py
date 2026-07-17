"""Tests for /api/v1/chat endpoints.

Covers:
  - POST /api/v1/chat (send message)
    - With existing session_id
    - Without session_id (creates new UUID)
    - Degraded response
  - GET /api/v1/chat/sessions (list sessions)
  - GET /api/v1/chat/{session_id}/messages (get session messages)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.chat import router as chat_router
from mindflow.services.chat_service import ChatAnswer


def _make_app(chat_service_mock: object | None = None) -> FastAPI:
    """Build a minimal FastAPI app with the chat route registered."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(chat_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True
    if chat_service_mock is not None:
        app.state.chat_service = chat_service_mock
    return app


class TestPostChat:
    """POST /api/v1/chat endpoint tests."""

    def test_chat_with_session(self) -> None:
        """200 with existing session_id."""
        mock_service = AsyncMock()
        mock_service.ask = AsyncMock(
            return_value=ChatAnswer(
                answer="你好！根据你的行为数据，今天专注度正常。",
                session_id="existing-session-001",
                tools_used=("query_evidence",),
                evidence_cited=True,
                degraded=False,
            ),
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/chat",
            json={
                "message": "我今天怎么样？",
                "session_id": "existing-session-001",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "你好！根据你的行为数据，今天专注度正常。"
        assert data["session_id"] == "existing-session-001"
        assert data["tools_used"] == ["query_evidence"]
        assert data["evidence_cited"] is True
        assert data["degraded"] is False

    def test_chat_without_session(self) -> None:
        """200 with new UUID session_id generated."""
        mock_service = AsyncMock()
        mock_service.ask = AsyncMock(
            return_value=ChatAnswer(
                answer="你好！有什么可以帮助你的？",
                session_id="new-uuid-session",
                degraded=False,
            ),
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/chat",
            json={"message": "你好"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "你好！有什么可以帮助你的？"
        # Session ID should be a UUID
        assert len(data["session_id"]) > 0

    def test_chat_degraded(self) -> None:
        """200 with degraded=true when LLM is unavailable."""
        mock_service = AsyncMock()
        mock_service.ask = AsyncMock(
            return_value=ChatAnswer(
                answer="当前 AI 对话不可用",
                session_id="s1",
                degraded=True,
            ),
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/chat",
            json={"message": "你好", "session_id": "s1"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is True
        assert data["evidence_cited"] is False

    def test_chat_empty_message_rejected(self) -> None:
        """422 for empty message."""
        mock_service = AsyncMock()
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/chat",
            json={"message": ""},
        )

        assert resp.status_code == 422

    def test_chat_tools_used_evidence_cited(self) -> None:
        """200 with tools_used and evidence_cited."""
        mock_service = AsyncMock()
        mock_service.ask = AsyncMock(
            return_value=ChatAnswer(
                answer="分析显示你的专注度偏低。",
                session_id="s1",
                tools_used=("query_evidence", "get_latest_analysis"),
                evidence_cited=True,
                degraded=False,
            ),
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/chat",
            json={"message": "分析我的数据", "session_id": "s1"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tools_used"]) == 2
        assert data["evidence_cited"] is True


class TestGetSessions:
    """GET /api/v1/chat/sessions endpoint tests."""

    def test_get_sessions_success(self) -> None:
        """200 with list of sessions."""
        mock_repo = AsyncMock()
        mock_repo.list_sessions = AsyncMock(
            return_value=[
                {"session_id": "s1", "last_message_at": "2026-07-18T10:00:00Z"},
                {"session_id": "s2", "last_message_at": "2026-07-17T10:00:00Z"},
            ],
        )
        mock_service = AsyncMock()
        mock_service._chat_repo = mock_repo
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.get("/api/v1/chat/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["session_id"] == "s1"
        assert data[1]["session_id"] == "s2"

    def test_get_sessions_empty(self) -> None:
        """200 with empty list when no sessions exist."""
        mock_repo = AsyncMock()
        mock_repo.list_sessions = AsyncMock(return_value=[])
        mock_service = AsyncMock()
        mock_service._chat_repo = mock_repo
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.get("/api/v1/chat/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert data == []


class TestGetMessages:
    """GET /api/v1/chat/{session_id}/messages endpoint tests."""

    def test_get_messages_success(self) -> None:
        """200 with list of messages for a session."""
        mock_repo = AsyncMock()
        mock_repo.recent = AsyncMock(
            return_value=[
                {"id": "m1", "role": "user", "content": "你好", "created_at": "2026-07-18T10:00:00Z"},
                {"id": "m2", "role": "assistant", "content": "你好！", "created_at": "2026-07-18T10:00:05Z"},
            ],
        )
        mock_service = AsyncMock()
        mock_service._chat_repo = mock_repo
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.get("/api/v1/chat/session-001/messages")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[0]["content"] == "你好"
        assert data[1]["role"] == "assistant"

    def test_get_messages_empty(self) -> None:
        """200 with empty list for a session with no messages."""
        mock_repo = AsyncMock()
        mock_repo.recent = AsyncMock(return_value=[])
        mock_service = AsyncMock()
        mock_service._chat_repo = mock_repo
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.get("/api/v1/chat/non-existent/messages")

        assert resp.status_code == 200
        data = resp.json()
        assert data == []
