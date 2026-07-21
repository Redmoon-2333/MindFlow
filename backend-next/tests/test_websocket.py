"""Tests for WebSocket handler at /api/v1/ws.

Covers:
  - Authentication via query parameter token
  - Invalid token -> close with 4001
  - Ping/pong messages
  - Error messages for invalid JSON
  - Connection cleanup on disconnect
  - F3: Host validation (defense-in-depth), connection cap, message-flood guard
"""

from __future__ import annotations

from contextlib import suppress

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from mindflow.api.websocket import _MAX_CONNECTIONS
from mindflow.api.websocket import router as ws_router

TEST_TOKEN = "test-token-123"

# TestClient's websocket_connect defaults to `Host: testserver`, which the
# F3 Host check correctly rejects (it isn't localhost/127.0.0.1/[::1]) — so
# every test that expects a *successful* connection needs to explicitly pass
# a trusted Host header, exactly like test_middleware_host.py does for the
# HTTP-side equivalent.
_TRUSTED_HOST_HEADERS = {"host": "localhost"}


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
        with client.websocket_connect(
            "/api/v1/ws?token=" + TEST_TOKEN, headers=_TRUSTED_HOST_HEADERS
        ):
            pass

    def test_connect_without_token(self, client):
        """Connecting without a token should close with 4001."""
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/api/v1/ws", headers=_TRUSTED_HOST_HEADERS),
        ):
            pass

    def test_connect_with_wrong_token(self, client):
        """Connecting with a wrong token should close with 4001."""
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect("/api/v1/ws?token=wrong", headers=_TRUSTED_HOST_HEADERS),
        ):
            pass


class TestWebSocketHostValidation:
    """F3: verify Host header validation (DNS-rebinding defense-in-depth)."""

    def test_untrusted_host_rejected_before_accept(self, client):
        """An untrusted Host header should close with 1008, even with a valid token."""
        with pytest.raises(WebSocketDisconnect) as exc_info, client.websocket_connect(
            "/api/v1/ws?token=" + TEST_TOKEN, headers={"host": "evil.com"}
        ):
            pass
        assert exc_info.value.code == 1008

    def test_default_testclient_host_rejected(self, client):
        """TestClient's default Host (testserver) is untrusted and must be rejected."""
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/api/v1/ws?token=" + TEST_TOKEN),
        ):
            pass
        assert exc_info.value.code == 1008

    def test_trusted_ipv4_host_allowed(self, client):
        """127.0.0.1 should be accepted like localhost."""
        with client.websocket_connect(
            "/api/v1/ws?token=" + TEST_TOKEN, headers={"host": "127.0.0.1:8765"}
        ):
            pass


class TestWebSocketConnectionCap:
    """F3: verify the concurrent-connection cap."""

    def test_connection_cap_enforced(self, client):
        """Once _MAX_CONNECTIONS is reached, further connections are rejected."""
        opened = []
        try:
            for _ in range(_MAX_CONNECTIONS):
                ctx = client.websocket_connect(
                    "/api/v1/ws?token=" + TEST_TOKEN, headers=_TRUSTED_HOST_HEADERS
                )
                opened.append(ctx)
                ctx.__enter__()

            with (
                pytest.raises(WebSocketDisconnect) as exc_info,
                client.websocket_connect(
                    "/api/v1/ws?token=" + TEST_TOKEN, headers=_TRUSTED_HOST_HEADERS
                ),
            ):
                pass
            assert exc_info.value.code == 1008
        finally:
            for ctx in opened:
                with suppress(Exception):
                    ctx.__exit__(None, None, None)


class TestWebSocketPingPong:
    """Verify ping/pong message handling."""

    def test_ping_gets_pong(self, client):
        """Sending a ping should receive a pong."""
        with client.websocket_connect(
            "/api/v1/ws?token=" + TEST_TOKEN, headers=_TRUSTED_HOST_HEADERS
        ) as ws:
            ws.send_json({"type": "ping", "payload": {}})
            response = ws.receive_json()
            assert response["type"] == "pong"
            assert "timestamp" in response

    def test_multiple_pings(self, client):
        """Multiple pings should each receive a pong."""
        with client.websocket_connect(
            "/api/v1/ws?token=" + TEST_TOKEN, headers=_TRUSTED_HOST_HEADERS
        ) as ws:
            for _ in range(3):
                ws.send_json({"type": "ping", "payload": {}})
                response = ws.receive_json()
                assert response["type"] == "pong"


class TestWebSocketErrors:
    """Verify error handling for invalid messages."""

    def test_invalid_json_gets_error(self, client):
        """Sending invalid JSON should receive an error message."""
        with client.websocket_connect(
            "/api/v1/ws?token=" + TEST_TOKEN, headers=_TRUSTED_HOST_HEADERS
        ) as ws:
            ws.send_text("not-json")
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "code" in response["payload"]


class TestWebSocketMessageFlood:
    """F3: verify the per-connection message-rate guard."""

    def test_message_flood_triggers_disconnect(self, client):
        """Sending messages faster than the flood guard allows should disconnect."""
        with client.websocket_connect(
            "/api/v1/ws?token=" + TEST_TOKEN, headers=_TRUSTED_HOST_HEADERS
        ) as ws:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                # _MSG_MAX_PER_WINDOW is 20 per _MSG_WINDOW_S (1s); sending
                # them back-to-back in a tight loop is always "too fast".
                for _ in range(30):
                    ws.send_json({"type": "ping", "payload": {}})
                    ws.receive_json()
            assert exc_info.value.code == 1008

