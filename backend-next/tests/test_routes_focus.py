"""Tests for /api/v1/focus endpoints.

Covers:
  - GET /focus: returns sessions, auto-generates if missing
  - GET /focus/trend: trend data with date grouping
  - Edge cases: empty data, valid date filtering
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mindflow.api.routes.focus as focus_module
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
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import focus_session_feedback
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.telemetry_service import TelemetryService


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


@pytest.fixture
async def seeded_app(engine, session_factory, tmp_path) -> FastAPI:
    """Test app with seeded events for focus endpoint."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(focus_session_feedback.create, checkfirst=True)

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
    app.state.telemetry_service = TelemetryService(
        TelemetryRepository(session_factory),
        PreferencesRepository(session_factory),
        data_dir=tmp_path,
    )

    # Insert events so session identification has data. Unique window titles
    # keep the events distinct (heartbeat merging matches app+process+title),
    # matching the AnalysisService test seeding pattern.
    for i in range(60):
        ev = make_event(
            user_id=1,
            timestamp_utc=_BASE + timedelta(seconds=i * 5),
            duration_s=5.0,
            process_name="Code.exe",
            app_name="VS Code",
            window_title=f"window-{i}",
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

    def test_get_focus_defaults_to_business_today(self, seeded_app, monkeypatch):
        business_today = MagicMock(return_value=date(2026, 7, 26))
        monkeypatch.setattr(
            focus_module,
            "business_today",
            business_today,
            raising=False,
        )
        monkeypatch.setattr(
            focus_module,
            "utc_today",
            lambda: date(2026, 7, 25),
            raising=False,
        )
        seeded_app.state.settings = SimpleNamespace(timezone="Asia/Shanghai")

        response = TestClient(seeded_app).get("/api/v1/focus")

        assert response.status_code == 200
        assert response.json()["date"] == "2026-07-26"
        business_today.assert_called_once_with("Asia/Shanghai")

    def test_get_focus_with_date(self, seeded_app):
        """GET /focus?date=2026-07-17 should work."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/focus?date=2026-07-17")
        assert resp.status_code == 200

    async def test_get_focus_current_day_refreshes_projection(self, seeded_app, monkeypatch):
        """Current business day must recompute from latest events, not serve a stale projection.

        The stale row shares its (date, start_time) with the recomputed session, so the
        id is reused (feedback stays linked) while end_time reflects the event stream.
        """
        focus_repo = seeded_app.state.focus_repository
        stale = await focus_repo.save_sessions(1, [{
            "date": "2026-07-17",
            "start_time": _utc("2026-07-17T08:00:00").isoformat(),
            "end_time": _utc("2026-07-17T08:00:30").isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 100.0,
            "switch_count": 0,
        }])
        stale_id = stale[0]["id"]

        monkeypatch.setattr(
            focus_module,
            "business_today",
            MagicMock(return_value=date(2026, 7, 17)),
        )
        seeded_app.state.settings = SimpleNamespace(timezone="Asia/Shanghai")

        resp = TestClient(seeded_app).get("/api/v1/focus?date=2026-07-17")
        assert resp.status_code == 200
        data = resp.json()

        assert data["date"] == "2026-07-17"
        assert len(data["sessions"]) == 1
        session = data["sessions"][0]
        # Recomputed end (60 events * 5s from 08:00:00Z) replaces the stale one
        assert session["end_time"] == "2026-07-17T08:05:00+00:00"
        # Same (date, start_time) → id reused so feedback remains linked
        assert session["id"] == stale_id
        # Each session carries its canonical persisted date for display
        assert session["date"] == "2026-07-17"

    async def test_get_focus_past_date_returns_persisted_as_is(self, seeded_app):
        """Past dates must read existing sessions, not recompute them."""
        focus_repo = seeded_app.state.focus_repository
        saved = await focus_repo.save_sessions(1, [{
            "date": "2026-07-16",
            "start_time": _utc("2026-07-16T08:00:00").isoformat(),
            "end_time": _utc("2026-07-16T08:30:00").isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 90.0,
            "switch_count": 1,
        }])

        resp = TestClient(seeded_app).get("/api/v1/focus?date=2026-07-16")
        assert resp.status_code == 200
        data = resp.json()

        assert data["date"] == "2026-07-16"
        assert len(data["sessions"]) == 1
        session = data["sessions"][0]
        # Persisted projection returned unchanged (no recompute for past dates)
        assert session["id"] == saved[0]["id"]
        assert session["end_time"] == _utc("2026-07-16T08:30:00").isoformat()
        assert session["focus_score"] == 90.0
        assert session["date"] == "2026-07-16"

    def test_get_focus_trend(self, seeded_app):
        """GET /focus/trend should return trend data."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/focus/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert "daily" in data
        assert "total_sessions" in data

    def test_focus_trend_ends_on_business_today(self, seeded_app, monkeypatch):
        business_today = MagicMock(return_value=date(2026, 7, 26))
        monkeypatch.setattr(
            focus_module,
            "business_today",
            business_today,
            raising=False,
        )
        monkeypatch.setattr(
            focus_module,
            "utc_today",
            lambda: date(2026, 7, 25),
            raising=False,
        )
        seeded_app.state.settings = SimpleNamespace(timezone="Asia/Shanghai")

        response = TestClient(seeded_app).get("/api/v1/focus/trend?days=2")

        assert response.status_code == 200
        assert response.json()["start_date"] == "2026-07-25"
        assert response.json()["end_date"] == "2026-07-26"
        business_today.assert_called_once_with("Asia/Shanghai")


class TestFocusEmpty:
    """Focus endpoint tests with empty data."""

    @pytest.fixture
    async def empty_app(self, engine, session_factory, tmp_path) -> FastAPI:
        async with engine.begin() as conn:
            await conn.run_sync(focus_sessions.metadata.create_all)
            await conn.run_sync(activity_events.metadata.create_all)
            await conn.run_sync(focus_session_feedback.create, checkfirst=True)

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
        app.state.telemetry_service = TelemetryService(
            TelemetryRepository(session_factory),
            PreferencesRepository(session_factory),
            data_dir=tmp_path,
        )
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
