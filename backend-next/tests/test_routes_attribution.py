"""Tests for POST /api/v1/analytics/attribution.

Coverage:
  - Success (LLM responds, 200 with assessment)
  - Cache hit returns immediately
  - Degraded to rule engine (200 with meta.degraded=true)
  - No events for date (404 not-found)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.attribution import router as attribution_router
from mindflow.services.llm_service import AttributionOutcome


def _make_app(llm_service_mock=None) -> FastAPI:
    """Build a minimal FastAPI app with the attribution route registered."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(attribution_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True
    if llm_service_mock is not None:
        app.state.llm_service = llm_service_mock
    return app


def _success_outcome(**overrides) -> AttributionOutcome:
    """Build a successful AttributionOutcome."""
    defaults = {
        "assessment": {
            "procrastination_types": ["impulsivity"],
            "type_confidence": {"impulsivity": 0.82},
            "cognitive_distortions": ["all-or-nothing thinking"],
            "cbt_technique": "stimulus_control",
            "response_text": "你今天的模式反映了冲动分心倾向。",
            "next_action": "设置一个番茄钟",
        },
        "source": "deepseek",
        "cached": False,
        "degraded": False,
        "crisis_detected": False,
    }
    defaults.update(overrides)
    return AttributionOutcome(**defaults)


class TestAttributionRoute:
    """POST /analytics/attribution endpoint tests."""

    def test_success(self) -> None:
        """200 with assessment data."""
        mock_service = MagicMock()
        mock_service.analyze = AsyncMock(return_value=_success_outcome())
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/analytics/attribution", json={"date": "2026-07-17"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "deepseek"
        assert data["cached"] is False
        assert data["meta"]["degraded"] is False
        assert "procrastination_types" in data["assessment"]
        assert data["assessment"]["cbt_technique"] == "stimulus_control"

    def test_success_defaults_to_today(self) -> None:
        """Without body, defaults to today."""
        mock_service = MagicMock()
        mock_service.analyze = AsyncMock(return_value=_success_outcome())
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/analytics/attribution")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "deepseek"

    def test_cache_hit(self) -> None:
        """Cached result returns immediately."""
        mock_service = MagicMock()
        mock_service.analyze = AsyncMock(
            return_value=_success_outcome(cached=True, source="deepseek")
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/analytics/attribution", json={"date": "2026-07-17"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is True

    def test_degraded_to_rule_engine(self) -> None:
        """Full degradation returns 200 with meta.degraded=true."""
        mock_service = MagicMock()
        mock_service.analyze = AsyncMock(
            return_value=_success_outcome(source="rule_engine", degraded=True)
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/analytics/attribution", json={"date": "2026-07-17"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "rule_engine"
        assert data["meta"]["degraded"] is True

    def test_force_flag(self) -> None:
        """force=True should be passed through to the service."""
        mock_service = MagicMock()
        mock_service.analyze = AsyncMock(return_value=_success_outcome())
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/analytics/attribution",
            json={"date": "2026-07-17", "force": True},
        )

        assert resp.status_code == 200
        # Verify force=True was passed
        mock_service.analyze.assert_called_once()
        assert mock_service.analyze.call_args[1].get("force") is True

    def test_not_found(self) -> None:
        """404 when no events exist for the date."""
        from mindflow.api.errors import _not_found

        mock_service = MagicMock()
        mock_service.analyze = AsyncMock(side_effect=_not_found("暂无活动数据"))
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/analytics/attribution", json={"date": "2026-07-17"})

        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]

    def test_no_activity_domain_error_maps_to_404(self) -> None:
        """A service-raised NoActivityDataError is mapped to 404 by the handler (E4)."""
        from mindflow.errors import NoActivityDataError

        mock_service = MagicMock()
        mock_service.analyze = AsyncMock(
            side_effect=NoActivityDataError("暂无活动数据，请先开始采集")
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/analytics/attribution", json={"date": "2026-07-17"})

        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]
        assert "暂无活动数据" in data["detail"]

    def test_force_bypasses_cache(self) -> None:
        """force=True should reach the service."""
        mock_service = MagicMock()
        mock_service.analyze = AsyncMock(
            return_value=_success_outcome(cached=False, source="deepseek")
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post(
            "/api/v1/analytics/attribution",
            json={"date": "2026-07-16", "force": True},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is False
        # Verify service received the correct args
        mock_service.analyze.assert_called_once()
        call_kwargs = mock_service.analyze.call_args[1]
        assert call_kwargs.get("force") is True
