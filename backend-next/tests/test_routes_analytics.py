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
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
    focus_sessions,
)
from mindflow.services.analysis_service import AnalysisService


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


@pytest.fixture
async def seeded_app(engine, session_factory) -> FastAPI:
    """Test app with seed data for analytics."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)

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
    return app


@pytest.fixture
async def empty_app(engine, session_factory) -> FastAPI:
    """Test app with no data."""
    async with engine.begin() as conn:
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(activity_events.metadata.create_all)

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

    def test_baseline_returns_stub(self, seeded_app):
        """GET /analytics/baseline should return a placeholder."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "message" in data

    def test_baseline_not_found(self, empty_app):
        """No baseline should return the stub contract (200 with status pending).

        NOTE: The /analytics/baseline endpoint is a Wave 6 placeholder that
        always returns a stub dict (status='pending'), never 404.  Once Wave 6
        integrates BaselineModel, this test should be updated to assert a 404
        or empty structure depending on the new behaviour.
        """
        client = TestClient(empty_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "message" in data
        assert "note" in data


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
