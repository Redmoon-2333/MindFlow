"""Tests for /api/v1/autonomy endpoints (G005).

Covers (3 endpoints x 3 paths each):
  - GET  /api/v1/autonomy (enabled, paused, no prefs)
  - POST /api/v1/autonomy/pause (hours=2, hours=0.1 enforced min, invalid body)
  - POST /api/v1/autonomy/resume (resume after pause, resume when not paused)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.autonomy import router as autonomy_router
from mindflow.services.autonomy_service import AutonomyService


def _make_mock_service() -> MagicMock:
    """Create a mock AutonomyService with all async methods mocked."""
    svc = MagicMock(spec=AutonomyService)
    svc.is_enabled = AsyncMock()
    svc.pause = AsyncMock()
    svc.resume = AsyncMock()
    svc.get_status = AsyncMock()
    return svc


class TestGetAutonomy:
    """GET /api/v1/autonomy."""

    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(autonomy_router, prefix="/api/v1")
        return app

    def test_get_enabled(self, app: FastAPI) -> None:
        """200 with enabled=true."""
        mock_svc = _make_mock_service()
        mock_svc.get_status.return_value = {"enabled": True, "paused_until": None}
        app.state.autonomy_service = mock_svc

        client = TestClient(app)
        resp = client.get("/api/v1/autonomy")

        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["paused_until"] is None

    def test_get_paused(self, app: FastAPI) -> None:
        """200 with enabled=false and paused_until."""
        future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        mock_svc = _make_mock_service()
        mock_svc.get_status.return_value = {"enabled": False, "paused_until": future}
        app.state.autonomy_service = mock_svc

        client = TestClient(app)
        resp = client.get("/api/v1/autonomy")

        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False
        assert data["paused_until"] == future

    def test_get_empty_state(self, app: FastAPI) -> None:
        """200 on empty/initial state."""
        mock_svc = _make_mock_service()
        mock_svc.get_status.return_value = {"enabled": True, "paused_until": None}
        app.state.autonomy_service = mock_svc

        client = TestClient(app)
        resp = client.get("/api/v1/autonomy")

        assert resp.status_code == 200
        assert resp.json()["enabled"] is True


class TestPostPause:
    """POST /api/v1/autonomy/pause."""

    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(autonomy_router, prefix="/api/v1")
        return app

    def test_pause_2h(self, app: FastAPI) -> None:
        """200 with hours=2."""
        mock_svc = _make_mock_service()
        future = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
        mock_svc.get_status.return_value = {"enabled": False, "paused_until": future}
        mock_svc.is_enabled.return_value = False
        app.state.autonomy_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/autonomy/pause", json={"hours": 2.0})

        assert resp.status_code == 200
        mock_svc.pause.assert_awaited_once_with(hours=2.0)
        data = resp.json()
        assert data["enabled"] is False
        assert data["paused_until"] == future

    def test_pause_default_hours(self, app: FastAPI) -> None:
        """200 with default hours=1."""
        mock_svc = _make_mock_service()
        future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        mock_svc.get_status.return_value = {"enabled": False, "paused_until": future}
        app.state.autonomy_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/autonomy/pause", json={})

        assert resp.status_code == 200
        mock_svc.pause.assert_awaited_once_with(hours=1.0)

    def test_pause_hours_below_min(self, app: FastAPI) -> None:
        """422 when hours < 0.5 (minimum enforcement)."""
        mock_svc = _make_mock_service()
        app.state.autonomy_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/autonomy/pause", json={"hours": 0.1})

        assert resp.status_code == 422  # Validation error


class TestPostResume:
    """POST /api/v1/autonomy/resume."""

    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(autonomy_router, prefix="/api/v1")
        return app

    def test_resume_after_pause(self, app: FastAPI) -> None:
        """200 after pause."""
        mock_svc = _make_mock_service()
        mock_svc.get_status.return_value = {"enabled": True, "paused_until": None}
        app.state.autonomy_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/autonomy/resume")

        assert resp.status_code == 200
        mock_svc.resume.assert_awaited_once()
        data = resp.json()
        assert data["enabled"] is True
        assert data["paused_until"] is None

    def test_resume_when_not_paused(self, app: FastAPI) -> None:
        """200 when not paused (idempotent)."""
        mock_svc = _make_mock_service()
        mock_svc.get_status.return_value = {"enabled": True, "paused_until": None}
        app.state.autonomy_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/autonomy/resume")

        assert resp.status_code == 200
        mock_svc.resume.assert_awaited_once()
