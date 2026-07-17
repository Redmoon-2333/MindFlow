"""Tests for AuthMiddleware (Bearer token validation).

Covers:
  - Valid token → 200
  - Missing token → 401 with RFC 9457
  - Invalid token → 401
  - Wrong token type → 401
  - Exempt paths (health, docs) bypass auth
  - Edge cases: empty token, malformed header
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.middleware.auth import AuthMiddleware


def _make_app(expected_token: str = "test-token-123") -> FastAPI:
    """Create a test app with AuthMiddleware."""
    app = FastAPI()

    @app.get("/api/v1/protected")
    async def protected():
        return {"data": "secret"}

    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/docs")
    async def docs():
        return {"docs": True}

    app.state.system_token = expected_token
    register_exception_handlers(app)
    app.add_middleware(AuthMiddleware)
    return app


class TestAuthMiddleware:
    """Verify AuthMiddleware token validation."""

    @pytest.fixture
    def client(self) -> TestClient:
        app = _make_app()
        return TestClient(app)

    def test_valid_token(self, client):
        """A valid Bearer token should pass authentication."""
        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "Bearer test-token-123"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"data": "secret"}

    def test_missing_token(self, client):
        """No Authorization header should return 401."""
        resp = client.get("/api/v1/protected")
        assert resp.status_code == 401
        data = resp.json()
        assert data["type"] == "https://mindflow.app/errors/auth-required"
        assert data["status"] == 401

    def test_invalid_token(self, client):
        """An incorrect token should return 401."""
        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        data = resp.json()
        assert data["type"] == "https://mindflow.app/errors/auth-required"

    def test_missing_bearer_prefix(self, client):
        """A token without 'Bearer ' prefix should return 401."""
        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "test-token-123"},
        )
        assert resp.status_code == 401

    def test_empty_token(self, client):
        """An empty token value should return 401."""
        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401

    def test_health_endpoint_exempt(self, client):
        """The health endpoint should be accessible without a token."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_docs_endpoint_exempt(self, client):
        """The docs endpoint should be accessible without a token."""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_wrong_scheme(self, client):
        """A non-Bearer scheme (e.g. Basic) should return 401."""
        resp = client.get(
            "/api/v1/protected",
            headers={"Authorization": "Basic dGVzdDp0ZXN0"},
        )
        assert resp.status_code == 401

    def test_problem_detail_format(self, client):
        """Auth errors should use RFC 9457 format."""
        resp = client.get("/api/v1/protected")
        assert resp.headers.get("content-type") == "application/problem+json"
