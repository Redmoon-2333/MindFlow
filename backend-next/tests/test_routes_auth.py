"""Tests for the local authentication route."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.routes import auth


def _make_client() -> TestClient:
    app = FastAPI()
    app.state.system_token = "test-system-token"
    app.include_router(auth.router, prefix="/api/v1")
    return TestClient(app)


def test_login_without_body_returns_system_token() -> None:
    response = _make_client().post("/api/v1/auth/login")

    assert response.status_code == 200
    assert response.json() == {"token": "test-system-token"}


def test_login_with_valid_credentials_returns_system_token(monkeypatch) -> None:
    monkeypatch.setattr(auth, "_HARDCODED_USER", "test-user")
    monkeypatch.setattr(auth, "_HARDCODED_PASS", "test-password")

    response = _make_client().post(
        "/api/v1/auth/login",
        json={"username": "test-user", "password": "test-password"},
    )

    assert response.status_code == 200
    assert response.json() == {"token": "test-system-token"}


def test_login_with_invalid_credentials_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(auth, "_HARDCODED_USER", "test-user")
    monkeypatch.setattr(auth, "_HARDCODED_PASS", "test-password")

    response = _make_client().post(
        "/api/v1/auth/login",
        json={"username": "test-user", "password": "wrong-password"},
    )

    assert response.status_code == 401
    assert response.json()["detail"]


def test_login_body_validation_rejects_missing_password() -> None:
    response = _make_client().post(
        "/api/v1/auth/login",
        json={"username": "test-user"},
    )

    assert response.status_code == 422
