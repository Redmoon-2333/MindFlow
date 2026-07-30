"""Tests for /api/v1/focus endpoints.

Covers:
  - GET /focus: returns sessions, auto-generates if missing
  - GET /focus/trend: trend data with date grouping
  - Edge cases: empty data, valid date filtering
  - Dependency wiring: telemetry_service is required by GET /focus
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
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
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.telemetry_service import TelemetryService


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


@pytest.fixture
async def seeded_app(engine, session_factory, tmp_path: Path) -> FastAPI:
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

    telemetry_repo = TelemetryRepository(session_factory=session_factory)
    telemetry_svc = TelemetryService(
        repository=telemetry_repo,
        preferences_repository=MagicMock(spec=PreferencesRepository),
        data_dir=tmp_path,
    )

    app.state.activity_repository = activity_repo
    app.state.analysis_service = analysis_svc
    app.state.focus_repository = focus_repo
    app.state.report_service = None
    app.state.telemetry_service = telemetry_svc

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
    async def empty_app(self, engine, session_factory, tmp_path: Path) -> FastAPI:
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

        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        telemetry_svc = TelemetryService(
            repository=telemetry_repo,
            preferences_repository=MagicMock(spec=PreferencesRepository),
            data_dir=tmp_path,
        )

        app.state.activity_repository = activity_repo
        app.state.analysis_service = analysis_svc
        app.state.focus_repository = focus_repo
        app.state.report_service = None
        app.state.telemetry_service = telemetry_svc
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


class TestFocusDependencyWiring:
    """Verify the focus route's dependency chain is complete and correct."""

    def test_production_app_wires_telemetry_service(self, tmp_path: Path) -> None:
        """Characterization: create_app must set app.state.telemetry_service.

        This guards against the dependency being silently dropped from the
        production startup path — if TelemetryService construction is removed
        or renamed in app.py, this test fails immediately.
        """
        from mindflow.config import Settings
        from mindflow.app import create_app

        settings = Settings(
            data_dir=tmp_path,
            db_url=f"sqlite+aiosqlite:///{tmp_path / 'mindflow.db'}",
            run_scheduler=False,
            run_collectors=False,
        )
        app = create_app(settings)
        # telemetry_service is set during lifespan startup, not at create_app
        # time, so we must enter the TestClient context to trigger lifespan.
        with TestClient(app) as client:
            assert hasattr(app.state, "telemetry_service"), (
                "Production app must wire telemetry_service into app.state "
                "for focus route dependency resolution"
            )
            assert app.state.telemetry_service is not None

    def test_incomplete_app_missing_telemetry_service_fails_at_request_time(
        self, engine, session_factory
    ):
        """Failing-first: a test app without telemetry_service must surface
        the missing dependency clearly when the focus route is invoked.

        This test captures the contract: the focus route REQUIRES
        telemetry_service. If the dependency were made optional (e.g. via
        getattr-with-None fallback), this test would need to be updated to
        reflect that intentional design change — not silently pass.
        """
        import asyncio

        async def _build_app():
            async with engine.begin() as conn:
                await conn.run_sync(activity_events.metadata.create_all)
                await conn.run_sync(focus_sessions.metadata.create_all)

            app = FastAPI()
            register_exception_handlers(app)
            app.include_router(focus_router, prefix="/api/v1")
            app.state.collector_service = None
            app.state.migration_applied = True

            activity_repo = SQLAlchemyActivityRepository(
                session_factory=session_factory
            )
            focus_repo = SQLAlchemyFocusSessionRepository(
                session_factory=session_factory
            )
            app.state.activity_repository = activity_repo
            app.state.analysis_service = AnalysisService(
                activity_repo=activity_repo, focus_repo=focus_repo
            )
            app.state.focus_repository = focus_repo
            app.state.report_service = None
            # Deliberately omit: app.state.telemetry_service
            return app

        incomplete_app = asyncio.run(_build_app())
        client = TestClient(incomplete_app, raise_server_exceptions=False)
        resp = client.get("/api/v1/focus")
        # The request must fail — not silently succeed with degraded data.
        assert resp.status_code == 500, (
            "Focus route without telemetry_service must fail, not degrade silently"
        )
