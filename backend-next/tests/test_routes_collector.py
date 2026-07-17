"""Tests for /api/v1/collector endpoints.

Covers:
  - GET /collector: status (stopped, running)
  - POST /collector: start (idempotent)
  - POST /collector/stop: stop (idempotent)
  - Error cases: collector not available
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.collector import router as collector_router


class MockCollectorService:
    """A minimal mock CollectorService for testing."""

    def __init__(self, initial_status: str = "stopped") -> None:
        self._status = initial_status

    @property
    def status(self) -> str:
        return self._status

    async def start(self) -> None:
        self._status = "running"

    async def stop(self) -> None:
        self._status = "stopped"


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(collector_router, prefix="/api/v1")
    app.state.collector_service = MockCollectorService(initial_status="stopped")
    app.state.engine = None
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestCollectorRoutes:
    """Verify collector endpoints."""

    def test_get_status_stopped(self, client):
        """GET /collector should return stopped status."""
        resp = client.get("/api/v1/collector")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_start_collector(self, client):
        """POST /collector should start the collector."""
        resp = client.post("/api/v1/collector")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"

    def test_start_is_idempotent(self, client):
        """Starting an already running collector should return running."""
        client.post("/api/v1/collector")  # Start first
        resp = client.post("/api/v1/collector")  # Start again
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"

    def test_stop_collector(self, client):
        """POST /collector/stop should stop the collector."""
        client.post("/api/v1/collector")  # Start first
        resp = client.post("/api/v1/collector/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_stop_is_idempotent(self, client):
        """Stopping an already stopped collector should return stopped."""
        resp = client.post("/api/v1/collector/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_get_status_after_start(self, client):
        """GET /collector should reflect started state."""
        client.post("/api/v1/collector")
        resp = client.get("/api/v1/collector")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"


def test_collector_not_available():
    """When collector_service is None, endpoints should return 503."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(collector_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.engine = None

    client = TestClient(app)

    resp = client.get("/api/v1/collector")
    assert resp.status_code == 503
    data = resp.json()
    assert data["type"] == "https://mindflow.app/errors/collector-not-running"

    resp = client.post("/api/v1/collector")
    assert resp.status_code == 503

    resp = client.post("/api/v1/collector/stop")
    assert resp.status_code == 503
