"""Tests for /api/v1/focus endpoints.

Covers:
  - GET /focus: returns sessions, auto-generates if missing
  - GET /focus/trend: trend data with date grouping
  - Edge cases: empty data, valid date filtering
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.focus import router as focus_router
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
    """Test app with seeded events for focus endpoint."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(focus_router, prefix="/api/v1")
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
    app.state.report_service = None

    # Insert events so session identification has data
    for i in range(60):
        ev = make_event(
            user_id=1,
            timestamp_utc=_BASE + timedelta(seconds=i * 5),
            duration_s=5.0,
            process_name="Code.exe",
            app_name="VS Code",
        )
        await activity_repo.append_event(ev)

    return app


class TestFocusRoutes:
    """Focus endpoint tests."""

    def test_get_focus_success(self, seeded_app):
        """GET /focus should return session data."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/focus")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "date" in data

    def test_get_focus_with_date(self, seeded_app):
        """GET /focus?date=2026-07-17 should work."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/focus?date=2026-07-17")
        assert resp.status_code == 200

    def test_get_focus_trend(self, seeded_app):
        """GET /focus/trend should return trend data."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/focus/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert "daily" in data
        assert "total_sessions" in data


class TestFocusEmpty:
    """Focus endpoint tests with empty data."""

    @pytest.fixture
    async def empty_app(self, engine, session_factory) -> FastAPI:
        async with engine.begin() as conn:
            await conn.run_sync(focus_sessions.metadata.create_all)
            await conn.run_sync(activity_events.metadata.create_all)

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(focus_router, prefix="/api/v1")
        app.state.collector_service = None
        app.state.migration_applied = True

        focus_repo = SQLAlchemyFocusSessionRepository(
            session_factory=session_factory
        )
        activity_repo = SQLAlchemyActivityRepository(
            session_factory=session_factory
        )
        analysis_svc = AnalysisService(
            activity_repo=activity_repo, focus_repo=focus_repo
        )

        app.state.activity_repository = activity_repo
        app.state.analysis_service = analysis_svc
        app.state.focus_repository = focus_repo
        app.state.report_service = None
        return app

    def test_focus_trend_empty(self, empty_app):
        """GET /focus/trend with no data should return zeros."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/focus/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_sessions"] == 0

    def test_trend_days_boundary(self, seeded_app):
        """days param boundary: 0→422, 90→200, 91→422 (ge=1, le=90)."""
        client = TestClient(seeded_app)

        # Below lower bound
        resp = client.get("/api/v1/focus/trend?days=0")
        assert resp.status_code == 422

        # At upper bound
        resp = client.get("/api/v1/focus/trend?days=90")
        assert resp.status_code == 200

        # Above upper bound
        resp = client.get("/api/v1/focus/trend?days=91")
        assert resp.status_code == 422
