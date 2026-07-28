"""Tests for /api/v1/analytics endpoints.

Covers:
  - GET /analytics/patterns: pattern analysis with/without data
  - GET /analytics/baseline: placeholder response
  - GET /analytics/profile: behavioural profile with/without data
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.analytics import router as analytics_router
from mindflow.domain.events import make_event
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.baseline import (
    BaselineRepository,
    baseline_models,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
    focus_sessions,
)
from mindflow.services.analysis_service import AnalysisService
from mindflow.train.models import ModelManager


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


@pytest.fixture
async def seeded_app(engine, session_factory) -> FastAPI:
    """Test app with seed data for analytics."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(baseline_models.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True

    activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    analysis_svc = AnalysisService(
        activity_repo=activity_repo, focus_repo=focus_repo
    )

    # Seed events
    for i in range(20):
        ev = make_event(
            user_id=1,
            timestamp_utc=_BASE + timedelta(seconds=i * 5),
            duration_s=5.0,
            process_name="Code.exe",
        )
        await activity_repo.append_event(ev)

    # Seed sessions
    await focus_repo.save_sessions(1, [
        {
            "date": "2026-07-17",
            "start_time": _utc("2026-07-17T08:00:00").isoformat(),
            "end_time": _utc("2026-07-17T08:30:00").isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 85.0,
            "switch_count": 0,
        },
    ])

    app.state.activity_repository = activity_repo
    app.state.analysis_service = analysis_svc
    app.state.focus_repository = focus_repo
    app.state.baseline_repository = BaselineRepository(session_factory=session_factory)
    return app


@pytest.fixture
async def empty_app(engine, session_factory) -> FastAPI:
    """Test app with no data."""
    async with engine.begin() as conn:
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(baseline_models.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True

    activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    analysis_svc = AnalysisService(
        activity_repo=activity_repo, focus_repo=focus_repo
    )

    app.state.activity_repository = activity_repo
    app.state.analysis_service = analysis_svc
    app.state.focus_repository = focus_repo
    app.state.baseline_repository = BaselineRepository(session_factory=session_factory)
    return app


class TestPatterns:
    """Pattern analysis endpoint tests."""

    def test_patterns_success(self, seeded_app):
        """GET /analytics/patterns should return pattern data."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/analytics/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert "high_switch_periods" in data
        assert "trigger_apps" in data
        assert "heatmap" in data
        assert data["total_sessions"] >= 1

    def test_patterns_empty(self, empty_app):
        """GET /analytics/patterns with no data should return 404."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/analytics/patterns")
        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]


class TestBaseline:
    """Baseline endpoint tests."""

    def test_baseline_not_found_no_data(self, seeded_app):
        """GET /analytics/baseline with no baseline data should return 404."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]

    def test_baseline_not_found_empty(self, empty_app):
        """GET /analytics/baseline with no data should return 404."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]


class TestProfile:
    """Behavioural profile endpoint tests."""

    def test_profile_success(self, seeded_app):
        """GET /analytics/profile should return profile data."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/analytics/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert "peak_focus_hours" in data
        assert "top_apps" in data
        assert "avg_focus_block_min" in data
        assert data["total_events_analysed"] >= 1

    def test_profile_empty(self, empty_app):
        """GET /analytics/profile with no data should return 404."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/analytics/profile")
        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]


@pytest.mark.skip(reason="V1 model_manager removed — test needs V2 fixture update")
def test_model_status_reports_unready_classifier(tmp_path) -> None:
    app = FastAPI()
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.model_manager = ModelManager(models_dir=tmp_path / "models")

    response = TestClient(app).get("/api/v1/analytics/model-status")

    assert response.status_code == 200
    data = response.json()
    assert data["loaded"] is True
    assert data["ready"] is False
    assert data["mode"] == "rule_engine_only"
    assert "classifier_not_fitted" in data["reasons"]


@pytest.mark.skip(reason="V1 model_manager removed — test needs V2 fixture update")
def test_model_status_reports_shadow_mode_without_active_model() -> None:
    app = FastAPI()
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.model_manager = None
    app.state.model_training_mode = "shadow"

    data = TestClient(app).get("/api/v1/analytics/model-status").json()

    assert data["ready"] is False
    assert data["mode"] == "shadow"


def test_model_status_prefers_ready_v2_model() -> None:
    class ReadyV2Manager:
        current_version_tag = "20260724"

        @staticmethod
        def readiness_status():
            return {"ready": True, "reasons": []}

        @staticmethod
        def list_versions():
            return ["20260724"]

    v2_manager = ReadyV2Manager()
    app = FastAPI()
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.model_manager = None
    app.state.v2_model_manager = v2_manager

    data = TestClient(app).get("/api/v1/analytics/model-status").json()

    assert data["mode"] == "ready"
    assert data["feature_schema_version"] == 2
