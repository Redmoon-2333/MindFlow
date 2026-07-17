"""Tests for /api/v1/intervention endpoints.

Covers (3 endpoints x 3 paths each):
  - POST /intervention/trigger: manual trigger (success, skipped, invalid intensity)
  - POST /intervention/{id}/response: record response (accepted, not found, invalid)
  - GET  /intervention/history: history (with data, empty, invalid days)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.intervention import router as intervention_router
from mindflow.domain.intervention import (
    Intervention,
    InterventionIntensity,
)
from mindflow.services.intervention_service import (
    InterventionResult,
    InterventionService,
)


def _make_mock_service() -> MagicMock:
    """Create a mock InterventionService with async methods."""
    svc = MagicMock(spec=InterventionService)
    svc.maybe_intervene = AsyncMock()
    svc.record_response = AsyncMock()
    svc.get_history = AsyncMock()
    return svc


class TestTriggerIntervention:
    """POST /api/v1/intervention/trigger."""

    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(intervention_router, prefix="/api/v1")
        return app

    def test_trigger_success(self, app) -> None:
        """Happy path: intervention triggered successfully."""
        now = __import__("datetime").datetime.now(__import__("datetime").UTC)
        intervention = Intervention(
            id="trig-001",
            user_id=1,
            intervention_type="nudge",
            cbt_technique="behavioral_experiment",
            title="测试标题",
            message="测试消息",
            dismissible=True,
            created_at=now,
        )
        mock_svc = _make_mock_service()
        mock_svc.maybe_intervene.return_value = InterventionResult(
            intervention=intervention,
        )
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/intervention/trigger")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] is False
        assert data["intervention"]["id"] == "trig-001"
        assert data["intervention"]["intervention_type"] == "nudge"

    def test_trigger_with_intensity(self, app) -> None:
        """Custom intensity parameter is passed through."""
        mock_svc = _make_mock_service()
        mock_svc.maybe_intervene.return_value = InterventionResult(
            skipped=True, skip_reason="跳过"
        )
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/intervention/trigger?intensity=strict")
        assert resp.status_code == 200

        # Verify intensity was passed
        call_kwargs = mock_svc.maybe_intervene.await_args[1]
        assert call_kwargs["intensity"] == InterventionIntensity.STRICT

    def test_trigger_invalid_intensity(self, app) -> None:
        """Invalid intensity defaults to standard."""
        mock_svc = _make_mock_service()
        mock_svc.maybe_intervene.return_value = InterventionResult(
            skipped=True, skip_reason="跳过"
        )
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/intervention/trigger?intensity=invalid")
        assert resp.status_code == 200
        call_kwargs = mock_svc.maybe_intervene.await_args[1]
        assert call_kwargs["intensity"] == InterventionIntensity.STANDARD

    def test_trigger_skipped(self, app) -> None:
        """When service skips, response reflects that."""
        mock_svc = _make_mock_service()
        mock_svc.maybe_intervene.return_value = InterventionResult(
            skipped=True,
            skip_reason="当前处于深度专注状态",
        )
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post("/api/v1/intervention/trigger")
        assert resp.status_code == 200
        data = resp.json()
        assert data["skipped"] is True
        assert data["intervention"] is None


class TestRespondToIntervention:
    """POST /api/v1/intervention/{id}/response."""

    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(intervention_router, prefix="/api/v1")
        return app

    def test_response_accepted(self, app) -> None:
        """Accepted response returns ok."""
        mock_svc = _make_mock_service()
        mock_svc.record_response.return_value = {"id": "resp-001", "user_response": "accepted"}
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post(
            "/api/v1/intervention/resp-001/response?response=accepted&latency_s=3.5"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["user_response"] == "accepted"

    def test_response_ignored(self, app) -> None:
        """Ignored response."""
        mock_svc = _make_mock_service()
        mock_svc.record_response.return_value = {"id": "resp-002", "user_response": "ignored"}
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post(
            "/api/v1/intervention/resp-002/response?response=ignored"
        )
        assert resp.status_code == 200
        assert resp.json()["user_response"] == "ignored"

    def test_response_not_found(self, app) -> None:
        """Non-existent intervention returns 404."""
        mock_svc = _make_mock_service()
        mock_svc.record_response.return_value = None
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post(
            "/api/v1/intervention/ghost/response?response=accepted"
        )
        assert resp.status_code == 404

    def test_response_invalid_value(self, app) -> None:
        """Invalid response value returns error, not 404."""
        mock_svc = _make_mock_service()
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post(
            "/api/v1/intervention/some-id/response?response=maybe"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_response_with_latency(self, app) -> None:
        """Latency parameter is passed through."""
        mock_svc = _make_mock_service()
        mock_svc.record_response.return_value = {"id": "resp-003", "user_response": "dismissed"}
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.post(
            "/api/v1/intervention/resp-003/response?response=dismissed&latency_s=12.0"
        )
        assert resp.status_code == 200

        call_args = mock_svc.record_response.await_args
        assert call_args is not None
        # latency_s is passed as the third positional arg
        assert len(call_args.args) >= 3
        assert call_args.args[2] == 12.0


class TestInterventionHistory:
    """GET /api/v1/intervention/history."""

    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(intervention_router, prefix="/api/v1")
        return app

    def test_history_with_data(self, app) -> None:
        """History returns the count and list."""
        mock_svc = _make_mock_service()
        mock_svc.get_history.return_value = [
            {"id": "h1", "intervention_type": "nudge"},
            {"id": "h2", "intervention_type": "task_breakdown"},
        ]
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.get("/api/v1/intervention/history?days=14")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["interventions"]) == 2

    def test_history_empty(self, app) -> None:
        """Empty history returns zero count."""
        mock_svc = _make_mock_service()
        mock_svc.get_history.return_value = []
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.get("/api/v1/intervention/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["interventions"] == []

    def test_history_invalid_days(self, app) -> None:
        """Invalid days parameter should return 422."""
        mock_svc = _make_mock_service()
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.get("/api/v1/intervention/history?days=-1")
        assert resp.status_code == 422

    def test_history_default_days(self, app) -> None:
        """Default days should be 7."""
        mock_svc = _make_mock_service()
        mock_svc.get_history.return_value = []
        app.state.intervention_service = mock_svc

        client = TestClient(app)
        resp = client.get("/api/v1/intervention/history")
        assert resp.status_code == 200

        call_kwargs = mock_svc.get_history.await_args[1]
        assert call_kwargs["days"] == 7
