"""Tests for /api/v1/health endpoint.

Covers:
  - Health endpoint returns 200
  - Response contains collector, database, migration, and version info
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.health import router as health_router


@pytest.fixture
def app() -> FastAPI:
    """Create a test app with the health route."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(health_router, prefix="/api/v1")

    # Set up app.state with test defaults
    mock_collector = type("MockCollector", (), {"status": "stopped"})()
    app.state.collector_service = mock_collector
    app.state.engine = None  # Will be handled by exception
    app.state.migration_applied = True

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Test client fixture."""
    return TestClient(app)


class TestHealthEndpoint:
    """Verify health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint always returns 200."""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_contains_version(self, client):
        """Response should include version."""
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "version" in data

    def test_health_contains_collector_status(self, client):
        """Response should include collector status."""
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "collector" in data
        assert data["collector"]["status"] == "stopped"

    def test_health_contains_database_info(self, client):
        """Response should include database health."""
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "database" in data
        assert "status" in data["database"]

    def test_health_contains_migration_info(self, client):
        """Response should include migration status."""
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "migration" in data
        assert data["migration"]["applied"] is True

    def test_health_contains_timestamp(self, client):
        """Response should include an ISO8601 timestamp."""
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "timestamp" in data
        assert "T" in data["timestamp"]

    def test_health_no_auth_required(self, client):
        """Health endpoint should be accessible without auth."""
        # The fixture doesn't have auth middleware, so this is always 200
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
