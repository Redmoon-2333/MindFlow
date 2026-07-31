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
from mindflow.time_utils import business_day_bounds_utc


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


class TestDailyReportOutputContract:
    """Pin the daily report dict contract (Todo 14/15).

    Both the generated and cached paths expose the same transient frontend
    fields — ``total_focus_minutes`` / ``total_sessions`` /
    ``total_distractions`` / ``hourly_distribution`` — plus ``data_state``
    (Todo 14 finding: the generated path must not omit the aliases the
    cached path already decorates).  An empty day yields a zero-filled row
    with ``top_apps`` normalized to an empty list and
    ``data_state == "no_activity"``.
    """

    GENERATED_KEYS = (
        "id",
        "user_id",
        "date",
        "total_focus_min",
        "total_distraction_min",
        "focus_score",
        "top_apps",
        "switch_frequency",
        "pattern_summary",
        "created_at",
    )
    TRANSIENT_KEYS = (
        "total_focus_minutes",
        "total_sessions",
        "total_distractions",
        "hourly_distribution",
        "data_state",
    )

    async def test_generated_report_key_contract(self, seeded_repos) -> None:
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert set(report) == set(self.GENERATED_KEYS) | set(self.TRANSIENT_KEYS)
        assert report["data_state"] == "ready"

    async def test_generated_and_cached_paths_share_transient_fields(
        self,
        seeded_repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )
        first = await svc.generate_daily_report(1, date(2026, 7, 17))
        cached = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert first["id"] == cached["id"]
        # Todo 14 finding: the generated path must attach the same aliases
        # and state as the cached path, from the same live single-day read.
        for key in self.TRANSIENT_KEYS:
            assert first[key] == cached[key]
        assert cached["total_focus_minutes"] == cached["total_focus_min"]
        assert cached["total_sessions"] == 2
        assert cached["total_distractions"] == 1
        assert list(cached["hourly_distribution"]) == [str(h) for h in range(24)]
        assert cached["hourly_distribution"] == first["hourly_distribution"]

    async def test_empty_day_returns_zero_filled_report(self, repos, monkeypatch) -> None:
        monkeypatch.setattr(
            report_service_module,
            "business_today",
            lambda _timezone: date(2026, 7, 17),
        )
        activity_repo, focus_repo, report_repo = repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert report["total_focus_min"] == 0.0
        assert report["total_distraction_min"] == 0.0
        assert report["focus_score"] == 0.0
        assert report["switch_frequency"] == 0.0
        assert report["top_apps"] == []
        assert report["pattern_summary"] not in (None, "")
        assert report["data_state"] == "no_activity"

    async def test_generate_daily_for_all_defaults_to_business_today(
        self,
        repos,
        monkeypatch,
    ) -> None:
        monkeypatch.setattr(
            report_service_module,
            "business_today",
            lambda _timezone: date(2026, 7, 24),
        )
        activity_repo, focus_repo, report_repo = repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )
        report = await svc.generate_daily_for_all(user_id=1)
        assert report["date"] == "2026-07-24"


class TestWeeklyReportOutputContract:
    """Characterization: pin the current weekly report dict contract."""

    async def test_weekly_key_contract(self, seeded_repos, monkeypatch) -> None:
        monkeypatch.setattr(
            report_service_module,
            "business_today",
            lambda _timezone: date(2026, 7, 31),
        )
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )
        report = await svc.weekly_report(1, date(2026, 7, 13))
        assert report["week_start"] == "2026-07-13"
        assert report["week_end"] == "2026-07-19"
        assert len(report["daily_reports"]) == 7
        for key in (
            "averages",
            "trend",
            "week_number",
            "total_focus_minutes",
            "total_sessions",
            "total_distractions",
            "avg_focus_score",
            "daily_summary",
            "data_state",
        ):
            assert key in report
        assert report["intervention_effectiveness"] is None
        assert len(report["daily_summary"]) == 7


class TestHourlyDistribution:
    """Todo 14: timezone-correct 24-hour focus distribution.

    Covers the generated and cached report paths, same-hour / cross-hour /
    cross-midnight clipping, the UTC-16:00 -> Asia/Shanghai next-day
    boundary, one DST-observing IANA timezone, malformed session
    timestamps, repeated-read idempotency, the sum-of-buckets invariant,
    and the unchanged persistence schema.
    """

    ALL_HOUR_KEYS = [str(h) for h in range(24)]

    @staticmethod
    def _zero_buckets() -> dict[str, float]:
        return {key: 0.0 for key in TestHourlyDistribution.ALL_HOUR_KEYS}

    async def test_generated_report_has_all_24_keys_and_stays_transient(
        self,
        seeded_repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert list(report["hourly_distribution"]) == self.ALL_HOUR_KEYS
        # Focus session 08:00-09:00 UTC == 16:00-17:00 local; the
        # distraction session is excluded from the distribution.
        assert report["hourly_distribution"]["16"] == 60.0
        assert sum(v for k, v in report["hourly_distribution"].items() if k != "16") == 0.0
        # Transient: the persisted row must not carry the distribution.
        persisted = await report_repo.get_by_date(1, date(2026, 7, 17))
        assert persisted is not None
        assert "hourly_distribution" not in persisted

    async def test_same_hour_session_bucketed_to_single_local_hour(
        self,
        repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T08:00:00").isoformat(),
                "end_time": _utc("2026-07-17T09:00:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        expected = self._zero_buckets()
        expected["16"] = 60.0
        assert report["hourly_distribution"] == expected

    async def test_cross_hour_session_split_across_local_hours(
        self,
        repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T09:30:00").isoformat(),
                "end_time": _utc("2026-07-17T10:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        # 09:30-10:30 UTC == 17:30-18:30 local, split at the 18:00 boundary.
        expected = self._zero_buckets()
        expected["17"] = 30.0
        expected["18"] = 30.0
        assert report["hourly_distribution"] == expected

    async def test_session_crossing_local_midnight_is_clipped_to_day(
        self,
        repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T15:30:00").isoformat(),
                "end_time": _utc("2026-07-18T00:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        # 23:30-00:30 local is clipped to 23:59:59.999999 local -> 30 minutes.
        expected = self._zero_buckets()
        expected["23"] = 30.0
        assert report["hourly_distribution"] == expected

    async def test_utc16_shanghai_maps_to_next_local_day(
        self,
        repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = repos
        # 2026-07-16T16:00Z is local midnight of 2026-07-17 in Asia/Shanghai.
        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-16T16:00:00").isoformat(),
                "end_time": _utc("2026-07-16T17:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        expected = self._zero_buckets()
        expected["0"] = 60.0
        expected["1"] = 30.0
        assert report["hourly_distribution"] == expected

    async def test_dst_spring_forward_day_uses_local_wall_clock(
        self,
        repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = repos
        # US spring-forward 2026-03-08: 02:00 EST jumps to 03:00 EDT, so
        # local hour 2 does not exist.  06:00-07:00Z is 01:00-02:00 EST
        # (bucket "1"); 07:00-08:00Z is 03:00-04:00 EDT (bucket "3").
        await focus_repo.save_sessions(1, [
            {
                "date": "2026-03-08",
                "start_time": _utc("2026-03-08T06:00:00").isoformat(),
                "end_time": _utc("2026-03-08T08:00:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="America/New_York",
        )
        report = await svc.generate_daily_report(1, date(2026, 3, 8))
        expected = self._zero_buckets()
        expected["1"] = 60.0
        expected["3"] = 60.0
        assert report["hourly_distribution"] == expected

    async def test_malformed_session_timestamps_contribute_zero(
        self,
        repos,
        engine,
    ) -> None:
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T08:00:00").isoformat(),
                "end_time": _utc("2026-07-17T09:00:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        async with engine.begin() as conn:
            await conn.execute(focus_sessions.insert(), [
                {
                    "id": "bad-start",
                    "user_id": 1,
                    "date": "2026-07-17",
                    "start_time": "not-a-timestamp",
                    "end_time": _utc("2026-07-17T08:30:00").isoformat(),
                    "session_type": "focus",
                },
                {
                    "id": "empty-times",
                    "user_id": 1,
                    "date": "2026-07-17",
                    "start_time": "",
                    "end_time": "",
                    "session_type": "focus",
                },
                {
                    "id": "inverted",
                    "user_id": 1,
                    "date": "2026-07-17",
                    "start_time": _utc("2026-07-17T09:00:00").isoformat(),
                    "end_time": _utc("2026-07-17T08:30:00").isoformat(),
                    "session_type": "focus",
                },
            ])
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        expected = self._zero_buckets()
        expected["16"] = 60.0
        assert report["hourly_distribution"] == expected

    async def test_repeated_reads_identical_distribution(
        self,
        seeded_repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        first = await svc.generate_daily_report(1, date(2026, 7, 17))
        cached = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert cached["id"] == first["id"]
        assert cached["hourly_distribution"] == first["hourly_distribution"]
        assert list(cached["hourly_distribution"]) == self.ALL_HOUR_KEYS

    async def test_sum_of_buckets_equals_clipped_focus_minutes(
        self,
        repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T09:30:00").isoformat(),
                "end_time": _utc("2026-07-17T10:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T15:30:00").isoformat(),
                "end_time": _utc("2026-07-18T00:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 80.0,
                "switch_count": 0,
            },
        ])
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        report = await svc.generate_daily_report(1, date(2026, 7, 17))
        day_start, day_end = business_day_bounds_utc(date(2026, 7, 17), "Asia/Shanghai")
        clipped_total = 0.0
        for start, end in (
            (_utc("2026-07-17T09:30:00"), _utc("2026-07-17T10:30:00")),
            (_utc("2026-07-17T15:30:00"), _utc("2026-07-18T00:30:00")),
        ):
            clipped_start = max(start, day_start)
            clipped_end = min(end, day_end)
            clipped_total += (clipped_end - clipped_start).total_seconds() / 60.0
        assert sum(report["hourly_distribution"].values()) == round(clipped_total, 1)

    async def test_cached_path_decorates_from_live_single_day_data(
        self,
        seeded_repos,
    ) -> None:
        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        first = await svc.generate_daily_report(1, date(2026, 7, 17))
        cached = await svc.generate_daily_report(1, date(2026, 7, 17))
        assert cached["total_sessions"] == 2
        assert cached["total_distractions"] == 1
        assert cached["hourly_distribution"] == first["hourly_distribution"]
        persisted = await report_repo.get_by_date(1, date(2026, 7, 17))
        assert persisted is not None
        assert "hourly_distribution" not in persisted

    async def test_persistence_schema_unchanged(
        self,
        seeded_repos,
        engine,
    ) -> None:
        import sqlalchemy as sa

        activity_repo, focus_repo, report_repo = seeded_repos
        svc = ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="Asia/Shanghai",
        )
        await svc.generate_daily_report(1, date(2026, 7, 17))
        persisted = await report_repo.get_by_date(1, date(2026, 7, 17))
        assert persisted is not None
        assert set(persisted) == set(TestDailyReportOutputContract.GENERATED_KEYS)
        async with engine.begin() as conn:
            columns = await conn.run_sync(
                lambda sync_conn: [
                    c["name"] for c in sa.inspect(sync_conn).get_columns("daily_reports")
                ]
            )
        assert "hourly_distribution" not in columns


class TestDailyDataState:
    """Todo 15: explicit daily data states.

    States: ``future`` (target after business today), ``no_activity`` (no
    sessions and no events), ``events_only`` (events without sessions),
    ``neutral_only`` (sessions with neither focus nor distraction),
    ``no_focus`` (distraction without focus), else ``ready``.
    """

    TODAY = date(2026, 7, 17)

    @staticmethod
    def _fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            report_service_module,
            "business_today",
            lambda _timezone: TestDailyDataState.TODAY,
        )

    @staticmethod
    def _svc(repos) -> ReportService:
        activity_repo, focus_repo, report_repo = repos
        return ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )

    async def test_future_day_returns_future_state_without_persisting(
        self,
        repos,
        monkeypatch,
    ) -> None:
        self._fixed_today(monkeypatch)
        activity_repo, focus_repo, report_repo = repos
        svc = self._svc(repos)

        report = await svc.generate_daily_report(1, date(2026, 7, 18))

        assert report["data_state"] == "future"
        # Empty arrays/objects, never fabricated nonzero bars or null lists.
        assert report["top_apps"] == []
        assert report["total_sessions"] == 0
        assert report["total_distractions"] == 0
        assert report["hourly_distribution"] == {str(h): 0.0 for h in range(24)}
        # A future report must never be persisted.
        assert await report_repo.get_by_date(1, date(2026, 7, 18)) is None

    async def test_no_activity_day(self, repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        svc = self._svc(repos)

        report = await svc.generate_daily_report(1, date(2026, 7, 17))

        assert report["data_state"] == "no_activity"
        assert report["total_sessions"] == 0
        assert report["total_distractions"] == 0

    async def test_events_only_day(self, repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        activity_repo, focus_repo, report_repo = repos
        for i in range(3):
            await activity_repo.append_event(
                make_event(
                    user_id=1,
                    timestamp_utc=_utc("2026-07-17T08:00:00") + timedelta(minutes=i * 10),
                    duration_s=300.0,
                    process_name="Code.exe",
                    app_name="VS Code",
                )
            )
        svc = self._svc(repos)

        report = await svc.generate_daily_report(1, date(2026, 7, 17))

        assert report["data_state"] == "events_only"
        assert report["total_sessions"] == 0

    async def test_neutral_only_day(self, repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [
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
        svc = self._svc(repos)

        report = await svc.generate_daily_report(1, date(2026, 7, 17))

        assert report["data_state"] == "neutral_only"
        assert report["total_sessions"] == 1

    async def test_no_focus_day(self, repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [
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
        svc = self._svc(repos)

        report = await svc.generate_daily_report(1, date(2026, 7, 17))

        assert report["data_state"] == "no_focus"
        assert report["total_sessions"] == 1
        assert report["total_distractions"] == 1

    async def test_ready_day_with_focus_session(self, seeded_repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        svc = self._svc(seeded_repos)

        report = await svc.generate_daily_report(1, date(2026, 7, 17))

        assert report["data_state"] == "ready"
        assert report["total_sessions"] == 2

    async def test_generated_and_cached_paths_share_state_and_aliases(
        self,
        seeded_repos,
        monkeypatch,
    ) -> None:
        self._fixed_today(monkeypatch)
        svc = self._svc(seeded_repos)

        generated = await svc.generate_daily_report(1, date(2026, 7, 17))
        cached = await svc.generate_daily_report(1, date(2026, 7, 17))

        assert generated["id"] == cached["id"]
        for key in (
            "total_focus_minutes",
            "total_sessions",
            "total_distractions",
            "hourly_distribution",
            "data_state",
        ):
            assert generated[key] == cached[key]
        assert cached["data_state"] == "ready"


class TestWeeklyDataState:
    """Todo 15: explicit weekly data states.

    States: ``future`` (week starts after today), ``no_activity`` (all
    produced days are no_activity), ``partial`` (current/incomplete week or
    any non-ready day), else ``ready``.
    """

    TODAY = date(2026, 7, 17)

    @staticmethod
    def _fixed_today(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            report_service_module,
            "business_today",
            lambda _timezone: TestWeeklyDataState.TODAY,
        )

    @staticmethod
    def _svc(repos) -> ReportService:
        activity_repo, focus_repo, report_repo = repos
        return ReportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
            timezone="UTC",
        )

    @staticmethod
    def _focus_session(day: date) -> dict[str, Any]:
        return {
            "date": day.isoformat(),
            "start_time": _utc(f"{day.isoformat()}T08:00:00").isoformat(),
            "end_time": _utc(f"{day.isoformat()}T09:00:00").isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 80.0,
            "switch_count": 0,
        }

    async def test_future_week(self, repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        svc = self._svc(repos)

        report = await svc.weekly_report(1, date(2026, 7, 20))

        assert report["data_state"] == "future"
        assert report["daily_reports"] == []
        assert report["daily_summary"] == []
        assert report["averages"] == {}
        assert report["trend"] == {}

    async def test_past_week_all_no_activity(self, repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        svc = self._svc(repos)

        report = await svc.weekly_report(1, date(2026, 7, 6))

        assert report["data_state"] == "no_activity"
        assert len(report["daily_reports"]) == 7
        assert all(d["data_state"] == "no_activity" for d in report["daily_reports"])

    async def test_current_week_all_no_activity_is_no_activity(
        self,
        repos,
        monkeypatch,
    ) -> None:
        """no_activity precedence: an all-empty current week stays no_activity."""
        self._fixed_today(monkeypatch)
        svc = self._svc(repos)

        report = await svc.weekly_report(1, date(2026, 7, 13))

        assert report["data_state"] == "no_activity"

    async def test_current_week_with_data_is_partial(self, repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [self._focus_session(date(2026, 7, 14))])
        svc = self._svc(repos)

        report = await svc.weekly_report(1, date(2026, 7, 13))

        assert report["data_state"] == "partial"
        assert [d["date"] for d in report["daily_reports"]] == [
            "2026-07-13",
            "2026-07-14",
            "2026-07-15",
            "2026-07-16",
            "2026-07-17",
        ]

    async def test_past_week_with_non_ready_day_is_partial(
        self,
        repos,
        monkeypatch,
    ) -> None:
        self._fixed_today(monkeypatch)
        activity_repo, focus_repo, report_repo = repos
        await focus_repo.save_sessions(1, [self._focus_session(date(2026, 7, 8))])
        svc = self._svc(repos)

        report = await svc.weekly_report(1, date(2026, 7, 6))

        assert report["data_state"] == "partial"
        assert any(d["data_state"] == "ready" for d in report["daily_reports"])
        assert any(d["data_state"] == "no_activity" for d in report["daily_reports"])

    async def test_past_week_all_ready(self, repos, monkeypatch) -> None:
        self._fixed_today(monkeypatch)
        activity_repo, focus_repo, report_repo = repos
        for offset in range(7):
            await focus_repo.save_sessions(
                1, [self._focus_session(date(2026, 7, 6) + timedelta(days=offset))]
            )
        svc = self._svc(repos)

        report = await svc.weekly_report(1, date(2026, 7, 6))

        assert report["data_state"] == "ready"
        assert all(d["data_state"] == "ready" for d in report["daily_reports"])
