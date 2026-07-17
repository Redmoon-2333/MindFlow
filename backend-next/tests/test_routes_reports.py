"""Tests for /api/v1/reports endpoints.

Covers:
  - GET /reports/daily: report generation, 404 for absent data
  - GET /reports/weekly: weekly summary
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.report_service import ReportService


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


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
