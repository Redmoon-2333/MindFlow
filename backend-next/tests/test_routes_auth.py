"""Tests for one-time local authentication routes."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.routes import auth
from mindflow.infrastructure.security.token_manager import (
    BootstrapTicketStore,
    SessionTokenStore,
)


def _make_client() -> TestClient:
    app = FastAPI()
    app.state.system_token = "test-system-token"
    app.state.bootstrap_tickets = BootstrapTicketStore()
    app.state.browser_sessions = SessionTokenStore()
    app.include_router(auth.router, prefix="/api/v1")
    return TestClient(app)


def test_legacy_login_route_is_removed() -> None:
    response = _make_client().post("/api/v1/auth/login")
    assert response.status_code == 404
    assert "test-system-token" not in response.text


def test_issue_ticket_returns_short_lived_ticket() -> None:
    response = _make_client().post("/api/v1/auth/bootstrap/ticket")
    assert response.status_code == 200
    assert len(response.json()["ticket"]) >= 32
    assert response.json()["expires_in_s"] == 60


def test_exchange_ticket_sets_http_only_cookie() -> None:
    client = _make_client()
    ticket = client.post("/api/v1/auth/bootstrap/ticket").json()["ticket"]
    response = client.post("/api/v1/auth/bootstrap", json={"ticket": ticket})
    assert response.status_code == 204
    session_id = response.cookies.get("mindflow_session")
    assert session_id and session_id != "test-system-token"
    assert client.app.state.browser_sessions.verify(session_id) is True
    assert "HttpOnly" in response.headers["set-cookie"]


def test_exchange_rejects_invalid_ticket() -> None:
    response = _make_client().post(
        "/api/v1/auth/bootstrap", json={"ticket": "x" * 32}
    )
    assert response.status_code == 401
