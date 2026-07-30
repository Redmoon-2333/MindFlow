"""Security tests for one-time local desktop bootstrap authentication."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.middleware.auth import AuthMiddleware
from mindflow.api.routes.auth import router as auth_router
from mindflow.infrastructure.security.token_manager import (
    BootstrapTicketStore,
    SessionTokenStore,
)


def _app() -> FastAPI:
    app = FastAPI()
    app.state.system_token = "system-secret"
    app.state.bootstrap_tickets = BootstrapTicketStore(ttl_s=60, max_entries=4)
    app.state.browser_sessions = SessionTokenStore(ttl_s=3600, max_entries=4)
    app.add_middleware(AuthMiddleware)
    app.include_router(auth_router, prefix="/api/v1")

    @app.get("/api/v1/protected")
    async def protected() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/panel")
    async def panel() -> dict[str, bool]:
        return {"ui": True}

    @app.get("/api/v1/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "alive"}

    return app


def test_hardcoded_login_and_anonymous_token_disclosure_are_removed() -> None:
    with TestClient(_app()) as client:
        response = client.post("/api/v1/auth/login")
    assert response.status_code == 401
    assert "token" not in response.text


def test_authenticated_launcher_can_issue_and_exchange_one_time_ticket() -> None:
    with TestClient(_app()) as client:
        issued = client.post(
            "/api/v1/auth/bootstrap/ticket",
            headers={"Authorization": "Bearer system-secret"},
        )
        assert issued.status_code == 200
        ticket = issued.json()["ticket"]

        exchanged = client.post("/api/v1/auth/bootstrap", json={"ticket": ticket})
        assert exchanged.status_code == 204
        cookie = exchanged.headers["set-cookie"]
        assert "mindflow_session=" in cookie
        assert "system-secret" not in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=strict" in cookie

        protected = client.get("/api/v1/protected")
        assert protected.status_code == 200

        replay = client.post("/api/v1/auth/bootstrap", json={"ticket": ticket})
        assert replay.status_code == 401


def test_bootstrap_ticket_store_is_one_time_expiring_and_bounded() -> None:
    now = [100.0]
    store = BootstrapTicketStore(ttl_s=5, max_entries=2, clock=lambda: now[0])
    first = store.issue()
    second = store.issue()
    third = store.issue()

    assert store.consume(first) is False
    assert store.consume(second) is True
    assert store.consume(second) is False

    now[0] = 106.0
    assert store.consume(third) is False


def test_auth_prefix_is_not_blanket_exempt() -> None:
    with TestClient(_app()) as client:
        response = client.post("/api/v1/auth/bootstrap/ticket")
    assert response.status_code == 401


def test_non_api_frontend_paths_are_public_but_api_paths_remain_protected() -> None:
    with TestClient(_app()) as client:
        assert client.get("/panel").status_code == 200
        assert client.get("/api/v1/protected").status_code == 401


def test_browser_session_store_is_independent_expiring_and_bounded() -> None:
    now = [100.0]
    store = SessionTokenStore(ttl_s=5, max_entries=2, clock=lambda: now[0])
    first = store.issue()
    second = store.issue()
    third = store.issue()

    assert store.verify(first) is False
    assert store.verify(second) is True
    assert store.verify(third) is True
    now[0] = 106.0
    assert store.verify(second) is False


# ── Public route exemption characterisation ────────────────────────────────


def test_documented_public_paths_accessible_without_auth() -> None:
    """Baseline: documented public paths must remain accessible without auth."""
    with TestClient(_app()) as client:
        assert client.get("/api/v1/health/live").status_code == 200
        assert client.get("/panel").status_code == 200
        # /api/v1/auth/bootstrap is public (ticket IS the auth);
        # POST without body gets 422 validation error, not 401.
        resp = client.post("/api/v1/auth/bootstrap", json={})
        assert resp.status_code != 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        pytest.param("POST", "/api/v1/auth/bootstrap/ticket", id="bootstrap-ticket"),
        pytest.param("POST", "/api/v1/auth/bootstrap/ticket-extra", id="bootstrap-ticket-extra"),
        pytest.param("POST", "/api/v1/auth/login", id="auth-login"),
        pytest.param("GET", "/api/v1/protected", id="generic-protected"),
    ],
)
def test_auth_required_for_protected_api_paths(method: str, path: str) -> None:
    """Protected /api/ paths must reject unauthenticated requests with 401.

    Regression guard: ``/api/v1/auth/bootstrap/ticket`` MUST require a
    valid Bearer token — it is the launcher-only ticket-issuance endpoint.
    """
    with TestClient(_app()) as client:
        response = client.request(method, path)
    assert response.status_code == 401, (
        f"Expected 401 for unauthenticated {method} {path}, "
        f"got {response.status_code}"
    )
