"""Tests for ReportService.

Covers:
  - generate_daily_report: idempotency, session aggregation, Chinese summary
  - weekly_report: 7-day trend, week-over-week comparison
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

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
from mindflow.services.report_service import ReportService


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


@pytest.fixture
async def repos(engine, session_factory):
    """Create all needed repositories with tables."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(daily_reports.metadata.create_all)

    activity_repo = SQLAlchemyActivityRepository(
        session_factory=session_factory, pulsetime_s=10
    )
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    report_repo = SQLAlchemyDailyReportRepository(session_factory=session_factory)
    return activity_repo, focus_repo, report_repo


@pytest.fixture
async def seeded_repos(repos):
    """Repositories with pre-seeded events and sessions."""
    activity_repo, focus_repo, report_repo = repos

    # Events across the day
    for i in range(30):
        ev = make_event(
            user_id=1,
            timestamp_utc=_BASE + timedelta(minutes=i * 10),
            duration_s=300.0,  # 5 min per event
            process_name="Code.exe" if i % 3 != 0 else "Chrome.exe",
            app_name="VS Code" if i % 3 != 0 else "Chrome",
        )
        await activity_repo.append_event(ev)

    # Focus sessions
    await focus_repo.save_sessions(1, [
        {
            "date": "2026-07-17",
            "start_time": _utc("2026-07-17T08:00:00").isoformat(),
            "end_time": _utc("2026-07-17T09:00:00").isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 85.0,
            "switch_count": 0,
        },
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
    return repos


class TestGenerateDailyReport:
    """Daily report generation tests."""

    async def test_generates_report(self, seeded_repos):
        """Report should be generated with correct structure."""
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert report["date"] == "2026-07-17"
        assert report["total_focus_min"] >= 0
        assert report["total_distraction_min"] >= 0
        assert 0 <= report["focus_score"] <= 100
        assert report["switch_frequency"] >= 0
        assert report["pattern_summary"] is not None

    async def test_pattern_summary_nonempty(self, seeded_repos):
        """Chinese pattern summary should be non-empty."""
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert len(report["pattern_summary"]) > 10

    async def test_idempotent(self, seeded_repos):
        """Second call should return existing report without recomputing."""
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        r1 = await svc.generate_daily_report(1, date(2026, 7, 17))
        r2 = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert r1["id"] == r2["id"]
        assert r1["focus_score"] == r2["focus_score"]

    async def test_different_date(self, seeded_repos):
        """Different dates should produce separate reports."""
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        r1 = await svc.generate_daily_report(1, date(2026, 7, 17))
        r2 = await svc.generate_daily_report(1, date(2026, 7, 18))
        assert r1["date"] != r2["date"]

    async def test_top_apps_present(self, seeded_repos):
        """Top apps should be populated."""
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert len(report.get("top_apps", [])) > 0


class TestWeeklyReport:
    """Weekly report tests."""

    async def test_generates_weekly(self, seeded_repos):
        """Weekly report should have 7-day structure."""
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        week_start = date(2026, 7, 13)  # Monday
        report = await svc.weekly_report(1, week_start)
        assert report["week_start"] == "2026-07-13"
        assert report["week_end"] == "2026-07-19"
        assert len(report["daily_reports"]) == 7
        assert "averages" in report
        assert "trend" in report

    async def test_weekly_averages_present(self, seeded_repos):
        """Weekly averages should be computed."""
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        week_start = date(2026, 7, 13)
        report = await svc.weekly_report(1, week_start)
        assert "avg_focus_min" in report["averages"]
        assert "avg_focus_score" in report["averages"]
        assert report["week_number"] is not None
