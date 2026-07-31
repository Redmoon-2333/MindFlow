"""Tests for /api/v1/reports endpoints.

Covers:
  - GET /reports/daily: report generation, 404 for absent data
  - GET /reports/weekly: weekly summary
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mindflow.api.routes.reports as reports_module
from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.reports import router as reports_router
from mindflow.domain.events import make_event
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
    focus_sessions,
)
from mindflow.infrastructure.repositories.report import (
    SQLAlchemyDailyReportRepository,
    daily_reports,
)
from mindflow.services import report_service as report_service_module
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.report_service import ReportService


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


async def _build_report_app(engine, session_factory) -> FastAPI:
    """Return a reports router app wired to real repos (no seed data).

    Repos are attached to ``app.state`` so individual tests can seed the
    exact fixture they need.
    """
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(daily_reports.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(reports_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True

    activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    report_repo = SQLAlchemyDailyReportRepository(session_factory=session_factory)
    report_svc = ReportService(
        activity_repo=activity_repo,
        focus_repo=focus_repo,
        report_repo=report_repo,
        timezone="UTC",
    )

    app.state.activity_repository = activity_repo
    app.state.focus_repository = focus_repo
    app.state.report_repository = report_repo
    app.state.report_service = report_svc
    return app


class TestDailyReport:
    """Daily report endpoint tests."""

    @pytest.fixture
    async def seeded_app(self, engine, session_factory) -> FastAPI:
        """Test app with seeded data for report generation."""
        async with engine.begin() as conn:
            await conn.run_sync(activity_events.metadata.create_all)
            await conn.run_sync(focus_sessions.metadata.create_all)
            await conn.run_sync(daily_reports.metadata.create_all)

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(reports_router, prefix="/api/v1")
        app.state.collector_service = None
        app.state.migration_applied = True

        activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
        focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
        report_repo = SQLAlchemyDailyReportRepository(session_factory=session_factory)
        analysis_svc = AnalysisService(
            activity_repo=activity_repo, focus_repo=focus_repo
        )
        report_svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        # Seed events and sessions
        for i in range(30):
            ev = make_event(
                user_id=1,
                timestamp_utc=_BASE + timedelta(minutes=i * 10),
                duration_s=300.0,
                process_name="Code.exe",
            )
            await activity_repo.append_event(ev)

        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T08:00:00").isoformat(),
                "end_time": _utc("2026-07-17T08:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])

        # Wire services
        app.state.activity_repository = activity_repo
        app.state.analysis_service = analysis_svc
        app.state.focus_repository = focus_repo
        app.state.report_repository = report_repo
        app.state.report_service = report_svc
        return app

    def test_daily_report_success(self, seeded_app):
        """GET /reports/daily should return a report."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/reports/daily?date=2026-07-17")
        assert resp.status_code == 200
        data = resp.json()
        assert data["date"] == "2026-07-17"
        assert "focus_score" in data
        assert "pattern_summary" in data

    def test_daily_report_defaults_to_business_today(self, seeded_app, monkeypatch):
        business_today = MagicMock(return_value=date(2026, 7, 17))
        monkeypatch.setattr(
            reports_module,
            "business_today",
            business_today,
            raising=False,
        )
        monkeypatch.setattr(
            reports_module,
            "utc_today",
            lambda: date(2026, 7, 16),
            raising=False,
        )
        seeded_app.state.settings = SimpleNamespace(timezone="Asia/Shanghai")

        response = TestClient(seeded_app).get("/api/v1/reports/daily")

        assert response.status_code == 200
        assert response.json()["date"] == "2026-07-17"
        business_today.assert_called_once_with("Asia/Shanghai")

    def test_daily_report_invalid_date(self, seeded_app):
        """Invalid date format should return 422."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/reports/daily?date=not-a-date")
        assert resp.status_code == 422


class TestWeeklyReport:
    """Weekly report endpoint tests."""

    @pytest.fixture
    async def seeded_app(self, engine, session_factory) -> FastAPI:
        async with engine.begin() as conn:
            await conn.run_sync(activity_events.metadata.create_all)
            await conn.run_sync(focus_sessions.metadata.create_all)
            await conn.run_sync(daily_reports.metadata.create_all)

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(reports_router, prefix="/api/v1")
        app.state.collector_service = None
        app.state.migration_applied = True

        activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
        focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
        report_repo = SQLAlchemyDailyReportRepository(session_factory=session_factory)
        analysis_svc = AnalysisService(
            activity_repo=activity_repo, focus_repo=focus_repo
        )
        report_svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        for i in range(30):
            ev = make_event(
                user_id=1,
                timestamp_utc=_BASE + timedelta(minutes=i * 10),
                duration_s=300.0,
                process_name="Code.exe",
            )
            await activity_repo.append_event(ev)

        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T08:00:00").isoformat(),
                "end_time": _utc("2026-07-17T08:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])

        app.state.activity_repository = activity_repo
        app.state.analysis_service = analysis_svc
        app.state.focus_repository = focus_repo
        app.state.report_repository = report_repo
        app.state.report_service = report_svc
        return app

    def test_weekly_report_success(self, seeded_app):
        """GET /reports/weekly should return 7-day summary."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/reports/weekly?week_start=2026-07-13")
        assert resp.status_code == 200
        data = resp.json()
        assert data["week_start"] == "2026-07-13"
        assert "averages" in data
        assert "daily_reports" in data

    def test_weekly_report_defaults_to_current_business_week(
        self,
        seeded_app,
        monkeypatch,
    ):
        business_today = MagicMock(return_value=date(2026, 7, 17))
        monkeypatch.setattr(
            reports_module,
            "business_today",
            business_today,
            raising=False,
        )
        monkeypatch.setattr(
            reports_module,
            "utc_today",
            lambda: date(2026, 7, 12),
            raising=False,
        )
        seeded_app.state.settings = SimpleNamespace(timezone="Asia/Shanghai")

        response = TestClient(seeded_app).get("/api/v1/reports/weekly")

        assert response.status_code == 200
        assert response.json()["week_start"] == "2026-07-13"
        business_today.assert_called_once_with("Asia/Shanghai")

    def test_weekly_report_empty_week(self, seeded_app):
        """Empty week should return structure with zero values, not 404."""
        client = TestClient(seeded_app)
        # A week with no data — report_service.weekly_report always returns a
        # structure with 7 daily reports (each filled with zeros), so the
        # route should return 200, never 404.
        resp = client.get("/api/v1/reports/weekly?week_start=2025-01-06")
        assert resp.status_code == 200
        data = resp.json()
        assert "week_start" in data
        assert "daily_reports" in data
        assert len(data["daily_reports"]) == 7
        assert "averages" in data
        assert "trend" in data
        assert "week_number" in data

    def test_weekly_report_invalid_week_start(self, seeded_app):
        """Invalid date format should return 422."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/reports/weekly?week_start=not-a-date")
        assert resp.status_code == 422
        err = resp.json()
        assert "detail" in err


class TestDailyReportDataStates:
    """Todo 15: daily endpoint returns typed 200 responses for every state.

    Expected empty/future states are 200 with a ``data_state`` — never 404
    and never fabricated nonzero bars.  Malformed dates stay 422 (already
    covered by ``TestDailyReport.test_daily_report_invalid_date``).
    """

    @pytest.fixture
    async def report_app(self, engine, session_factory) -> FastAPI:
        return await _build_report_app(engine, session_factory)

    @pytest.fixture
    async def events_app(self, engine, session_factory) -> FastAPI:
        app = await _build_report_app(engine, session_factory)
        for i in range(3):
            await app.state.activity_repository.append_event(
                make_event(
                    user_id=1,
                    timestamp_utc=_utc("2026-07-17T08:00:00") + timedelta(minutes=i * 10),
                    duration_s=300.0,
                    process_name="Code.exe",
                )
            )
        return app

    @pytest.fixture
    async def neutral_app(self, engine, session_factory) -> FastAPI:
        app = await _build_report_app(engine, session_factory)
        await app.state.focus_repository.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T08:00:00").isoformat(),
                "end_time": _utc("2026-07-17T09:00:00").isoformat(),
                "session_type": "neutral",
                "dominant_app": "Code.exe",
                "focus_score": 50.0,
                "switch_count": 1,
            },
        ])
        return app

    @pytest.fixture
    async def distraction_app(self, engine, session_factory) -> FastAPI:
        app = await _build_report_app(engine, session_factory)
        await app.state.focus_repository.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T10:00:00").isoformat(),
                "end_time": _utc("2026-07-17T10:30:00").isoformat(),
                "session_type": "distraction",
                "dominant_app": "Chrome.exe",
                "focus_score": 30.0,
                "switch_count": 3,
            },
        ])
        return app

    @pytest.fixture
    async def ready_app(self, engine, session_factory) -> FastAPI:
        app = await _build_report_app(engine, session_factory)
        await app.state.focus_repository.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T08:00:00").isoformat(),
                "end_time": _utc("2026-07-17T08:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        return app

    @staticmethod
    def _fixed_today(monkeypatch) -> None:
        monkeypatch.setattr(
            report_service_module,
            "business_today",
            lambda _timezone: date(2026, 7, 17),
        )

    def test_future_date(self, report_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(report_app).get("/api/v1/reports/daily?date=2026-07-18")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_state"] == "future"
        assert data["top_apps"] == []
        assert data["total_sessions"] == 0
        assert data["total_distractions"] == 0
        assert len(data["hourly_distribution"]) == 24

    def test_no_activity(self, report_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(report_app).get("/api/v1/reports/daily?date=2026-07-16")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_state"] == "no_activity"
        assert data["top_apps"] == []

    def test_events_only(self, events_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(events_app).get("/api/v1/reports/daily?date=2026-07-17")
        assert resp.status_code == 200
        assert resp.json()["data_state"] == "events_only"

    def test_neutral_only(self, neutral_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(neutral_app).get("/api/v1/reports/daily?date=2026-07-17")
        assert resp.status_code == 200
        assert resp.json()["data_state"] == "neutral_only"

    def test_no_focus(self, distraction_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(distraction_app).get("/api/v1/reports/daily?date=2026-07-17")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_state"] == "no_focus"
        assert data["total_distractions"] == 1

    def test_ready(self, ready_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(ready_app).get("/api/v1/reports/daily?date=2026-07-17")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_state"] == "ready"
        assert data["total_sessions"] == 1


class TestWeeklyReportDataStates:
    """Todo 15: weekly endpoint returns typed 200 responses for every state.

    The expected-empty future week returns 200 with ``data_state`` instead
    of the old 404 (``test_weekly_report_empty_week`` already pins the 200
    for a past empty week).
    """

    @pytest.fixture
    async def report_app(self, engine, session_factory) -> FastAPI:
        return await _build_report_app(engine, session_factory)

    @pytest.fixture
    async def mid_week_app(self, engine, session_factory) -> FastAPI:
        app = await _build_report_app(engine, session_factory)
        await app.state.focus_repository.save_sessions(1, [
            {
                "date": "2026-07-14",
                "start_time": _utc("2026-07-14T08:00:00").isoformat(),
                "end_time": _utc("2026-07-14T09:00:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        return app

    @pytest.fixture
    async def full_week_app(self, engine, session_factory) -> FastAPI:
        app = await _build_report_app(engine, session_factory)
        for offset in range(7):
            day = date(2026, 7, 6) + timedelta(days=offset)
            await app.state.focus_repository.save_sessions(1, [
                {
                    "date": day.isoformat(),
                    "start_time": _utc(f"{day.isoformat()}T08:00:00").isoformat(),
                    "end_time": _utc(f"{day.isoformat()}T09:00:00").isoformat(),
                    "session_type": "focus",
                    "dominant_app": "Code.exe",
                    "focus_score": 80.0,
                    "switch_count": 0,
                },
            ])
        return app

    @staticmethod
    def _fixed_today(monkeypatch) -> None:
        monkeypatch.setattr(
            report_service_module,
            "business_today",
            lambda _timezone: date(2026, 7, 17),
        )

    def test_future_week(self, report_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(report_app).get("/api/v1/reports/weekly?week_start=2026-07-20")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_state"] == "future"
        assert data["daily_reports"] == []
        assert data["daily_summary"] == []

    def test_past_week_all_no_activity(self, report_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(report_app).get("/api/v1/reports/weekly?week_start=2026-07-06")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_state"] == "no_activity"
        assert len(data["daily_reports"]) == 7

    def test_current_week_partial(self, mid_week_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(mid_week_app).get("/api/v1/reports/weekly?week_start=2026-07-13")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_state"] == "partial"
        assert [d["date"] for d in data["daily_reports"]] == [
            "2026-07-13",
            "2026-07-14",
            "2026-07-15",
            "2026-07-16",
            "2026-07-17",
        ]

    def test_past_week_all_ready(self, full_week_app, monkeypatch):
        self._fixed_today(monkeypatch)
        resp = TestClient(full_week_app).get("/api/v1/reports/weekly?week_start=2026-07-06")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data_state"] == "ready"
        assert all(d["data_state"] == "ready" for d in data["daily_reports"])
