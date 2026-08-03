"""Tests for liveness, readiness, and compatible health endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.health import router as health_router


class _ConnectionContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> None:
        return None


class _HealthyEngine:
    def connect(self) -> _ConnectionContext:
        return _ConnectionContext()


class _BrokenEngine:
    def connect(self) -> _ConnectionContext:
        raise RuntimeError("database unavailable")


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(health_router, prefix="/api/v1")

    mock_collector = type("MockCollector", (), {"status": "stopped"})()
    app.state.collector_service = mock_collector
    app.state.engine = _HealthyEngine()
    app.state.migration_applied = True
    app.state.db_integrity_ok = True
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestLivenessEndpoint:
    def test_live_returns_200_without_database_readiness(
        self, app: FastAPI, client: TestClient
    ) -> None:
        app.state.engine = _BrokenEngine()
        app.state.migration_applied = False
        app.state.db_integrity_ok = False

        response = client.get("/api/v1/health/live")

        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestReadinessEndpoint:
    def test_ready_returns_200_when_all_checks_pass(self, client: TestClient) -> None:
        response = client.get("/api/v1/health/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    @pytest.mark.parametrize(
        ("state_name", "state_value"),
        [("migration_applied", False), ("db_integrity_ok", False)],
    )
    def test_ready_returns_503_for_failed_startup_checks(
        self, app: FastAPI, client: TestClient, state_name: str, state_value: bool
    ) -> None:
        setattr(app.state, state_name, state_value)

        response = client.get("/api/v1/health/ready")

        assert response.status_code == 503
        assert response.json()["status"] == "not_ready"

    def test_ready_returns_503_when_database_connection_fails(
        self, app: FastAPI, client: TestClient
    ) -> None:
        app.state.engine = _BrokenEngine()

        response = client.get("/api/v1/health/ready")

        assert response.status_code == 503
        assert response.json()["database"]["connected"] is False


class TestCompatibleHealthEndpoint:
    def test_health_remains_200_when_not_ready(
        self, app: FastAPI, client: TestClient
    ) -> None:
        app.state.engine = _BrokenEngine()
        app.state.migration_applied = False
        app.state.db_integrity_ok = False

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_contains_existing_component_fields(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()

        assert "version" in data
        assert data["collector"]["status"] == "stopped"
        assert data["database"]["connected"] is True
        assert data["migration"]["applied"] is True
        assert "timestamp" in data

    def test_health_contains_observability_fields(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        obs = data["observability"]
        expected = {"last_activity_at", "last_intervention_at", "scheduler_heartbeat_at", "ml_mode"}
        assert expected <= set(obs)
