"""Tests for ReportService.

Covers:
  - generate_daily_report: idempotency, session aggregation, Chinese summary
  - weekly_report: 7-day trend, week-over-week comparison
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from typing import Any

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
from mindflow.services import report_service as report_service_module
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

    async def test_recomputes_report_created_before_business_day_completed(
        self,
        seeded_repos,
        engine,
    ):
        """A provisional zero report must not permanently mask later activity."""
        activity_repo, focus_repo, report_repo = seeded_repos
        async with engine.begin() as conn:
            await conn.execute(
                daily_reports.insert().values(
                    id="provisional",
                    user_id=1,
                    date="2026-07-17",
                    total_focus_min=0.0,
                    total_distraction_min=0.0,
                    focus_score=0.0,
                    switch_frequency=0.0,
                    created_at="2026-07-16T12:00:00+00:00",
                )
            )
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )

        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        persisted = await report_repo.get_by_date(1, date(2026, 7, 17))

        assert report["focus_score"] > 0
        assert report["id"] == "provisional"
        assert persisted is not None
        assert persisted["id"] == report["id"]
        assert persisted["created_at"] != "2026-07-16T12:00:00+00:00"

    async def test_refresh_recomputes_report_created_after_business_day_completed(
        self,
        seeded_repos,
        engine,
    ):
        """The final scheduled pass must replace a report viewed just after midnight."""
        activity_repo, focus_repo, report_repo = seeded_repos
        async with engine.begin() as conn:
            await conn.execute(
                daily_reports.insert().values(
                    id="early-view",
                    user_id=1,
                    date="2026-07-17",
                    total_focus_min=0.0,
                    total_distraction_min=0.0,
                    focus_score=0.0,
                    switch_frequency=0.0,
                    created_at="2026-07-18T00:01:00+00:00",
                )
            )
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )

        report = await svc.generate_daily_report(
            1,
            date(2026, 7, 17),
            refresh=True,
        )

        assert report["id"] == "early-view"
        assert report["focus_score"] > 0
        assert report["created_at"] != "2026-07-18T00:01:00+00:00"

    async def test_invalid_created_at_is_recomputed(
        self,
        seeded_repos,
        engine,
    ):
        """Malformed legacy timestamps must not freeze a provisional report."""
        activity_repo, focus_repo, report_repo = seeded_repos
        async with engine.begin() as conn:
            await conn.execute(
                daily_reports.insert().values(
                    id="invalid-created-at",
                    user_id=1,
                    date="2026-07-17",
                    total_focus_min=0.0,
                    total_distraction_min=0.0,
                    focus_score=0.0,
                    switch_frequency=0.0,
                    created_at="not-a-timestamp",
                )
            )
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )

        report = await svc.generate_daily_report(1, date(2026, 7, 17))

        assert report["id"] == "invalid-created-at"
        assert report["focus_score"] > 0
        assert report["created_at"] != "not-a-timestamp"


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

    async def test_current_week_does_not_generate_future_reports(
        self,
        repos,
        monkeypatch,
    ):
        """Viewing a current week must not persist placeholder rows for future days."""
        activity_repo, focus_repo, report_repo = repos
        monkeypatch.setattr(
            report_service_module,
            "business_today",
            lambda _timezone: date(2026, 7, 24),
        )
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        report = await svc.weekly_report(1, date(2026, 7, 20))

        assert [row["date"] for row in report["daily_reports"]] == [
            "2026-07-20",
            "2026-07-21",
            "2026-07-22",
            "2026-07-23",
            "2026-07-24",
        ]
        assert await report_repo.get_by_date(1, date(2026, 7, 25)) is None
        assert await report_repo.get_by_date(1, date(2026, 7, 26)) is None

    # ── P2-2: asyncio.gather concurrency ─────────────────────────────

    async def test_all_days_enter_before_any_completes(
        self,
        seeded_repos,
    ):
        """RED→GREEN: gather must schedule all daily_report calls before any awaits.

        Uses asyncio.Event barriers: each mock call signals ``entered`` and
        (on the last expected entry) fires an ``all_entered`` event, **then**
        awaits a single ``release`` event.  The test awaits ``all_entered``
        with a bounded ``asyncio.wait_for`` timeout so serial code fails
        promptly rather than hanging.

        After release, asserts output date order, week bounds, and
        aggregation presence.
        """
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        week_start = date(2026, 7, 13)
        expected_days = 7

        entered: list[date] = []
        all_entered = asyncio.Event()
        release = asyncio.Event()

        _original = svc.generate_daily_report

        async def _mock_generate(user_id: int, target_date: date) -> dict[str, Any]:
            entered.append(target_date)
            if len(entered) == expected_days:
                all_entered.set()
            await release.wait()
            return await _original(user_id, target_date)

        svc.generate_daily_report = _mock_generate  # type: ignore[method-assign]

        task: asyncio.Task[dict[str, Any]] | None = None
        try:
            task = asyncio.create_task(svc.weekly_report(1, week_start))

            # Bounded wait: serial code only enters day 0 then stalls →
            # TimeoutError after 2 s.  Gather code sets all_entered before
            # any coroutine yields at the release barrier.
            try:
                await asyncio.wait_for(all_entered.wait(), timeout=2.0)
            except TimeoutError:
                pytest.fail(
                    f"Only {len(entered)}/{expected_days} days entered before "
                    "timeout — serial execution still active (gather missing "
                    "or not working)"
                )

            # All coroutines scheduled — prove ordering
            iso_expected = [(week_start + timedelta(days=i)).isoformat() for i in range(7)]
            assert [d.isoformat() for d in entered] == iso_expected, (
                f"Entry order {entered} ≠ {iso_expected}"
            )

            # Release the barrier; all coroutines resume concurrently
            release.set()
            result = await task

            # Output order must equal input date order
            assert [r["date"] for r in result["daily_reports"]] == iso_expected
            assert result["week_start"] == "2026-07-13"
            assert result["week_end"] == "2026-07-19"
            assert len(result["daily_reports"]) == expected_days
            assert "averages" in result
            assert "trend" in result
        finally:
            release.set()  # ensure no leaked tasks on failure
            if task is not None and not task.done():
                task.cancel()
                # Let cancellation propagate before restoring the original
                with suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(task, timeout=1.0)
            svc.generate_daily_report = _original

    async def test_one_day_failure_propagates_no_skipping(
        self,
        seeded_repos,
    ):
        """When one daily_report raises, weekly_report must raise the same
        error without skipping the failed day or logging a fallback.

        Uses an Event barrier so the other 6 coroutines stay alive until
        release; gather cancels them automatically on first exception.
        The mock returns a pure dict (no DB touch) so no aiosqlite worker
        threads survive test teardown.
        """
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        week_start = date(2026, 7, 13)
        release = asyncio.Event()
        entered: list[date] = []

        _original = svc.generate_daily_report

        class _BoomError(Exception):
            pass

        _mock_return: dict[str, Any] = {"date": "", "user_id": 1}

        async def _failing_mock(user_id: int, target_date: date) -> dict[str, Any]:
            entered.append(target_date)
            day_idx = (target_date - week_start).days
            if day_idx == 3:  # fourth day (Thursday)
                raise _BoomError("simulated daily-report failure")
            await release.wait()
            # Return pure dict — never touches DB, no aiosqlite workers
            return {**_mock_return, "date": target_date.isoformat()}

        svc.generate_daily_report = _failing_mock  # type: ignore[method-assign]

        try:
            with pytest.raises(_BoomError, match="simulated daily-report failure"):
                await svc.weekly_report(1, week_start)

            # All days up to and including the failing day must have been
            # entered (days 0-3); days after may or may not depending on
            # scheduling interleaving.
            assert len(entered) >= 4, (
                f"Expected at least 4 entries (up to failing day), got {len(entered)}"
            )
            # The failing day must be present
            assert week_start + timedelta(days=3) in entered
        finally:
            # Release barrier so cancelled tasks' release.wait() unblocks
            # and CancelledError propagates cleanly before method restore.
            release.set()
            # Yield to let cancelled gather tasks finalize
            await asyncio.sleep(0)
            svc.generate_daily_report = _original

    async def test_real_driver_max_concurrent_gt_one(
        self,
        seeded_repos,
    ):
        """Real implementation with DB: concurrent-active count must exceed 1.

        Wraps the actual ``generate_daily_report`` with an atomic counter
        that tracks peak active calls.  The factory repos and the SQLite
        in-memory DB are the same as every other test — no mocks in the
        data path.
        """
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        _original = svc.generate_daily_report

        async def _counting_generate(user_id: int, target_date: date) -> dict[str, Any]:
            nonlocal active_count, max_active
            async with lock:
                active_count += 1
                if active_count > max_active:
                    max_active = active_count
            try:
                return await _original(user_id, target_date)
            finally:
                async with lock:
                    active_count -= 1

        svc.generate_daily_report = _counting_generate  # type: ignore[method-assign]

        try:
            week_start = date(2026, 7, 13)
            result = await svc.weekly_report(1, week_start)
        finally:
            svc.generate_daily_report = _original
            # Yield once so any in-flight counting_generate coroutine
            # that was about to decrement can finalize its lock scope.
            await asyncio.sleep(0)

        assert max_active > 1, (
            f"Expected concurrent active > 1 (gather), got {max_active}"
        )
        assert len(result["daily_reports"]) == 7
        iso_expected = [(week_start + timedelta(days=i)).isoformat() for i in range(7)]
        assert [r["date"] for r in result["daily_reports"]] == iso_expected

    async def test_output_order_equals_input_dates(
        self,
        seeded_repos,
    ):
        """daily_reports date order must exactly match the week's date order."""
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

        week_start = date(2026, 7, 13)
        result = await svc.weekly_report(1, week_start)

        iso_expected = [(week_start + timedelta(days=i)).isoformat() for i in range(7)]
        assert [r["date"] for r in result["daily_reports"]] == iso_expected
