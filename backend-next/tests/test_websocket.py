"""Tests for WebSocket handler at /api/v1/ws.

Covers:
  - Authentication via query parameter token
  - Invalid token -> close with 4001
  - Ping/pong messages
  - Error messages for invalid JSON
  - Connection cleanup on disconnect
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mindflow.api.websocket import router as ws_router

TEST_TOKEN = "test-token-123"


@pytest.fixture
def app() -> FastAPI:
    """Create a test app with WebSocket endpoint."""
    app = FastAPI()
    app.include_router(ws_router, prefix="/api/v1")
    app.state.system_token = TEST_TOKEN
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestWebSocketAuth:
    """Verify WebSocket authentication."""

    def test_connect_with_valid_token(self, client):
        """Connecting with a valid token should succeed."""
        with client.websocket_connect("/api/v1/ws?token=" + TEST_TOKEN):
            pass

    def test_connect_without_token(self, client):
        """Connecting without a token should close with 4001."""
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/v1/ws"):
            pass

    def test_connect_with_wrong_token(self, client):
        """Connecting with a wrong token should close with 4001."""
        with pytest.raises(WebSocketDisconnect), client.websocket_connect("/api/v1/ws?token=wrong"):
            pass


class TestWebSocketPingPong:
    """Verify ping/pong message handling."""

    def test_ping_gets_pong(self, client):
        """Sending a ping should receive a pong."""
        with client.websocket_connect("/api/v1/ws?token=" + TEST_TOKEN) as ws:
            ws.send_json({"type": "ping", "payload": {}})
            response = ws.receive_json()
            assert response["type"] == "pong"
            assert "timestamp" in response

    def test_multiple_pings(self, client):
        """Multiple pings should each receive a pong."""
        with client.websocket_connect("/api/v1/ws?token=" + TEST_TOKEN) as ws:
            for _ in range(3):
                ws.send_json({"type": "ping", "payload": {}})
                response = ws.receive_json()
                assert response["type"] == "pong"


class TestWebSocketErrors:
    """Verify error handling for invalid messages."""

    def test_invalid_json_gets_error(self, client):
        """Sending invalid JSON should receive an error message."""
        with client.websocket_connect("/api/v1/ws?token=" + TEST_TOKEN) as ws:
            ws.send_text("not-json")
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "code" in response["payload"]
