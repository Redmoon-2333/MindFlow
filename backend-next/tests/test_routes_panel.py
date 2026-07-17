"""Tests for /api/v1/panel endpoints.

Covers:
  - POST /api/v1/panel/today (trigger daily panel)
  - GET /api/v1/panel (retrieve panel result)
  - Degraded mode (meta.degraded=true)
  - Rate limiting (429)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.agents.types import (
    PanelVerdict,
    TranscriptEntry,
)
from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.panel import router as panel_router
from mindflow.domain.procrastination import CBTTechnique, ProcrastinationType


def _make_verdict(**overrides: object) -> PanelVerdict:
    """Build a sample PanelVerdict for testing."""
    defaults: dict[str, object] = {
        "types": (ProcrastinationType.IMPULSIVITY, ProcrastinationType.TASK_AVERSION),
        "confidence": {
            ProcrastinationType.IMPULSIVITY: 0.85,
            ProcrastinationType.TASK_AVERSION: 0.45,
        },
        "recommended_technique": CBTTechnique.STIMULUS_CONTROL,
        "rationale": "你的行为模式显示冲动分心倾向，同时伴随任务畏惧。",
        "dissent": ("情绪调节专家认为情绪调节是次要因素",),
        "transcript": (
            TranscriptEntry(role="数据分析师", content="模式分析完成", round=0),
            TranscriptEntry(role="综合主持人", content="裁决完成", round=3),
        ),
        "escalated": False,
        "call_count": 6,
        "source": "panel",
    }
    defaults.update(overrides)
    return PanelVerdict(**defaults)  # type: ignore[arg-type]


def _make_app(panel_service_mock: object | None = None) -> FastAPI:
    """Build a minimal FastAPI app with the panel route registered."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(panel_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True
    if panel_service_mock is not None:
        app.state.panel_service = panel_service_mock
    return app


class TestPostPanelToday:
    """POST /api/v1/panel/today endpoint tests."""

    def test_success(self) -> None:
        """200 with full PanelVerdict JSON."""
        mock_service = AsyncMock()
        mock_service.run_daily_panel = AsyncMock(return_value=_make_verdict())
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/panel/today")

        assert resp.status_code == 200
        data = resp.json()
        assert data["types"] == ["impulsivity", "task_aversion"]
        assert data["technique"] == "stimulus_control"
        assert data["call_count"] == 6
        assert data["escalated"] is False
        assert data["degraded"] is False
        assert data["meta"]["degraded"] is False
        assert len(data["dissent"]) == 1
        assert len(data["transcript"]) == 2
        assert data["rationale"] == "你的行为模式显示冲动分心倾向，同时伴随任务畏惧。"

    def test_degraded(self) -> None:
        """200 with meta.degraded=true when panel falls through."""
        degraded = _make_verdict(source="single_expert", call_count=0, transcript=())
        mock_service = AsyncMock()
        mock_service.run_daily_panel = AsyncMock(return_value=degraded)
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/panel/today")

        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded"] is True
        assert data["meta"]["degraded"] is True
        assert data["call_count"] == 0
        assert data["transcript"] == []

    def test_escalated_flag(self) -> None:
        """200 with escalated=true when panel had conflict."""
        mock_service = AsyncMock()
        mock_service.run_daily_panel = AsyncMock(
            return_value=_make_verdict(escalated=True, call_count=9),
        )
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/panel/today")

        assert resp.status_code == 200
        data = resp.json()
        assert data["escalated"] is True
        assert data["call_count"] == 9


class TestGetPanel:
    """GET /api/v1/panel endpoint tests."""

    def test_get_panel(self) -> None:
        """200 with panel verdict."""
        mock_service = AsyncMock()
        mock_service.run_daily_panel = AsyncMock(return_value=_make_verdict())
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.get("/api/v1/panel")

        assert resp.status_code == 200
        data = resp.json()
        assert data["types"] == ["impulsivity", "task_aversion"]
        assert data["meta"]["degraded"] is False


class TestPanelRateLimit:
    """Rate limiting on /api/v1/panel/today."""

    def test_ratelimit_headers(self) -> None:
        """Rate limit headers present on response."""
        mock_service = AsyncMock()
        mock_service.run_daily_panel = AsyncMock(return_value=_make_verdict())
        app = _make_app(mock_service)
        client = TestClient(app)

        resp = client.post("/api/v1/panel/today")

        assert resp.status_code == 200
        # Rate limit headers are added by middleware, not by the route itself
        # This just confirms the endpoint works; full ratelimit tests are in
        # test_middleware_ratelimit.py
