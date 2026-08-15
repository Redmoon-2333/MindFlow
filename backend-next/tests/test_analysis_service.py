"""Tests for AnalysisService.

Covers:
  - identify_focus_sessions: positive (creates sessions), boundary (too few events),
    idempotent (skips when sessions exist)
  - detect_patterns: empty history, populated history
  - behavioral_profile: empty events, populated history
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
from mindflow.services import analysis_service as analysis_service_module
from mindflow.services.analysis_service import AnalysisService


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


@pytest.fixture
async def repos(engine, session_factory):
    """Create repositories with all needed tables."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)

    activity_repo = SQLAlchemyActivityRepository(
        session_factory=session_factory, pulsetime_s=10
    )
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    return activity_repo, focus_repo


@pytest.fixture
async def service(repos) -> AnalysisService:
    activity_repo, focus_repo = repos
    return AnalysisService(activity_repo=activity_repo, focus_repo=focus_repo)


class TestIdentifyFocusSessions:
    """Session identification algorithm tests."""

    async def test_creates_sessions(self, repos, service):
        """Events >= threshold on same app should create a focus session."""
        activity_repo, _ = repos
        target = date(2026, 7, 17)

        # Insert 60 events for the same app over 30 minutes (5s each, 2s apart)
        for i in range(60):
            ev = make_event(
                user_id=1,
                timestamp_utc=_BASE + timedelta(seconds=i * 2),
                duration_s=5.0,
                process_name="Code.exe",
                app_name="VS Code",
            )
            await activity_repo.append_event(ev)

        sessions = await service.identify_focus_sessions(1, target)
        assert len(sessions) >= 1
        assert all(s["session_type"] == "focus" for s in sessions)
        assert all(s["dominant_app"] == "Code.exe" for s in sessions)

    async def test_skip_too_few_events(self, repos, service):
        """Fewer than 2 events should return empty."""
        activity_repo, _ = repos
        target = date(2026, 7, 17)

        ev = make_event(
            user_id=1,
            timestamp_utc=_BASE,
            duration_s=5.0,
            process_name="Code.exe",
        )
        await activity_repo.append_event(ev)

        sessions = await service.identify_focus_sessions(1, target)
        assert sessions == []

    async def test_idempotent_skips_existing(self, repos, service):
        """If sessions already exist for a date, should skip."""
        activity_repo, focus_repo = repos
        target = date(2026, 7, 17)

        # Pre-save a session for this date
        await focus_repo.save_sessions(1, [{
            "date": target.isoformat(),
            "start_time": _utc("2026-07-17T10:00:00").isoformat(),
            "end_time": _utc("2026-07-17T10:30:00").isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 80.0,
            "switch_count": 0,
        }])

        # Insert events
        for i in range(60):
            ev = make_event(
                user_id=1,
                timestamp_utc=_BASE + timedelta(seconds=i * 2),
                duration_s=5.0,
                process_name="Code.exe",
            )
            await activity_repo.append_event(ev)

        sessions = await service.identify_focus_sessions(1, target)
        assert sessions == []

    async def test_refresh_replaces_existing_sessions(self, repos, service):
        """The final daily analysis must replace a provisional session."""
        activity_repo, focus_repo = repos
        target = date(2026, 7, 17)
        provisional = await focus_repo.save_sessions(1, [{
            "date": target.isoformat(),
            "start_time": _BASE.isoformat(),
            "end_time": _BASE.isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 100.0,
            "switch_count": 0,
        }])
        provisional_id = provisional[0]["id"]
        for i in range(60):
            await activity_repo.append_event(
                make_event(
                    user_id=1,
                    timestamp_utc=_BASE + timedelta(seconds=i * 5),
                    duration_s=5.0,
                    process_name="Code.exe",
                    window_title=f"window-{i}",
                )
            )

        sessions = await service.identify_focus_sessions(1, target, refresh=True)
        persisted = await focus_repo.get_by_date(1, target)

        assert len(sessions) == 1
        assert persisted[0]["id"] == sessions[0]["id"]
        # Same (date, start_time) as the provisional row → its id is reused,
        # so existing feedback stays linked while end_time is refreshed.
        assert persisted[0]["id"] == provisional_id
        assert persisted[0]["end_time"] != persisted[0]["start_time"]

    async def test_refresh_removes_existing_sessions_when_no_block_remains(
        self,
        repos,
        service,
    ):
        """A refresh must replace stale sessions even when the new result is empty."""
        activity_repo, focus_repo = repos
        target = date(2026, 7, 17)
        await focus_repo.save_sessions(1, [{
            "date": target.isoformat(),
            "start_time": _BASE.isoformat(),
            "end_time": (_BASE + timedelta(minutes=30)).isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 100.0,
            "switch_count": 0,
        }])
        for i in range(2):
            await activity_repo.append_event(
                make_event(
                    user_id=1,
                    timestamp_utc=_BASE + timedelta(seconds=i * 5),
                    duration_s=5.0,
                    process_name="Code.exe",
                    window_title=f"short-window-{i}",
                )
            )

        sessions = await service.identify_focus_sessions(1, target, refresh=True)

        assert sessions == []
        assert await focus_repo.get_by_date(1, target) == []

    async def test_session_end_includes_last_event_duration(self, repos, service):
        """Session duration must include the final event's covered time."""
        activity_repo, _ = repos
        target = date(2026, 7, 17)
        for i in range(60):
            await activity_repo.append_event(
                make_event(
                    user_id=1,
                    timestamp_utc=_BASE + timedelta(seconds=i * 5),
                    duration_s=5.0,
                    process_name="Code.exe",
                    window_title=f"window-{i}",
                )
            )

        sessions = await service.identify_focus_sessions(1, target)

        assert len(sessions) == 1
        assert sessions[0]["start_time"] == _BASE.isoformat()
        assert sessions[0]["end_time"] == (_BASE + timedelta(minutes=5)).isoformat()

    async def test_switching_active_block_can_be_classified_as_distraction(
        self,
        repos,
        service,
    ):
        """Switch-rate classification must see transitions inside one active block."""
        activity_repo, _ = repos
        target = date(2026, 7, 17)
        for i in range(13):
            process = "Code.exe" if i % 2 == 0 else "Chrome.exe"
            await activity_repo.append_event(
                make_event(
                    user_id=1,
                    timestamp_utc=_BASE + timedelta(seconds=i * 30),
                    duration_s=30.0,
                    process_name=process,
                    app_name=process,
                )
            )

        sessions = await service.identify_focus_sessions(1, target)

        assert len(sessions) == 1
        assert sessions[0]["session_type"] == "distraction"
        assert sessions[0]["switch_count"] == 12
        assert sessions[0]["dominant_app"] == "Code.exe"
        assert sessions[0]["focus_score"] < 100.0

    async def test_overlapping_events_keep_the_furthest_covered_end(
        self,
        repos,
        service,
    ):
        """A short nested event must not truncate an active block's covered range."""
        activity_repo, _ = repos
        target = date(2026, 7, 17)
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_BASE,
                duration_s=600.0,
                process_name="Code.exe",
                window_title="long-code-window",
            ),
            make_event(
                user_id=1,
                timestamp_utc=_BASE + timedelta(seconds=100),
                duration_s=1.0,
                process_name="Chrome.exe",
                window_title="nested-browser-window",
            ),
            make_event(
                user_id=1,
                timestamp_utc=_BASE + timedelta(seconds=450),
                duration_s=1.0,
                process_name="Code.exe",
                window_title="covered-code-window",
            ),
        ]
        for event in events:
            await activity_repo.append_event(event)

        sessions = await service.identify_focus_sessions(1, target)

        assert len(sessions) == 1
        assert sessions[0]["switch_count"] == 0

    async def test_skips_idle_events(self, repos, service):
        """Idle events should not contribute to sessions."""
        activity_repo, _ = repos
        target = date(2026, 7, 17)

        # Only idle events
        for i in range(10):
            ev = make_event(
                user_id=1,
                timestamp_utc=_BASE + timedelta(seconds=i * 5),
                duration_s=5.0,
                process_name="",
                is_idle=True,
                event_type="idle_change",
            )
            await activity_repo.append_event(ev)

        sessions = await service.identify_focus_sessions(1, target)
        assert sessions == []


class TestDetectPatterns:
    """Pattern detection tests."""

    async def test_empty_history(self, service):
        """No sessions should return empty pattern data."""
        patterns = await service.detect_patterns(1, days=14)
        assert patterns["total_sessions"] == 0
        assert patterns["high_switch_periods"] == []
        assert patterns["trigger_apps"] == []
        assert patterns["distraction_ratio"] == 0.0

    async def test_with_sessions(self, repos, service, monkeypatch):
        """Sessions should produce pattern data."""
        _, focus_repo = repos

        monkeypatch.setattr(
            analysis_service_module,
            "business_today",
            lambda _timezone: date(2026, 7, 17),
        )

        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T10:00:00").isoformat(),
                "end_time": _utc("2026-07-17T10:30:00").isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 85.0,
                "switch_count": 0,
            },
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T14:00:00").isoformat(),
                "end_time": _utc("2026-07-17T14:15:00").isoformat(),
                "session_type": "distraction",
                "dominant_app": "Chrome.exe",
                "focus_score": 30.0,
                "switch_count": 5,
            },
        ])

        patterns = await service.detect_patterns(1, days=14)
        assert patterns["total_sessions"] == 2
        assert patterns["distraction_ratio"] == 0.5
        assert len(patterns["high_switch_periods"]) > 0
        assert len(patterns["trigger_apps"]) > 0
        assert patterns["trigger_apps"][0]["app"] == "Chrome.exe"

    async def test_uses_business_timezone_for_hour_and_weekday(
        self,
        repos,
        monkeypatch,
    ):
        """UTC session timestamps must be aggregated in the configured timezone."""
        _, focus_repo = repos
        target = date(2026, 7, 20)
        await focus_repo.save_sessions(1, [{
            "date": target.isoformat(),
            "start_time": _utc("2026-07-19T23:30:00").isoformat(),
            "end_time": _utc("2026-07-20T00:00:00").isoformat(),
            "session_type": "distraction",
            "dominant_app": "Chrome.exe",
            "focus_score": 25.0,
            "switch_count": 7,
        }])
        monkeypatch.setattr(
            analysis_service_module,
            "business_today",
            lambda _timezone: target,
        )
        service = AnalysisService(
            activity_repo=repos[0],
            focus_repo=focus_repo,
            timezone="Asia/Shanghai",
        )

        patterns = await service.detect_patterns(1)

        assert patterns["heatmap"][7][0] == 7
        assert patterns["heatmap"][23][6] == 0
        assert patterns["high_switch_periods"][0]["hour"] == 7


class TestBehavioralProfile:
    """Behavioural profile tests."""

    async def test_empty_events(self, service):
        """No events should return empty profile."""
        profile = await service.behavioral_profile(1, days=30)
        assert profile["total_events_analysed"] == 0
        assert profile["peak_focus_hours"] == []
        assert profile["top_apps"] == []

    async def test_with_events_and_sessions(self, repos, service):
        """Events and sessions should produce a profile."""
        activity_repo, focus_repo = repos

        # Seed against a recent clock so the default 30-day profile window
        # always covers the fixtures (a fixed date would age out of the
        # window and silently return an empty profile).
        base = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(
            hours=12
        )

        # Insert events
        for i in range(10):
            ev = make_event(
                user_id=1,
                timestamp_utc=base + timedelta(seconds=i * 5),
                duration_s=5.0,
                process_name="Code.exe",
            )
            await activity_repo.append_event(ev)

        # Add sessions
        await focus_repo.save_sessions(1, [
            {
                "date": base.date().isoformat(),
                "start_time": base.isoformat(),
                "end_time": (base + timedelta(minutes=30)).isoformat(),
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 90.0,
                "switch_count": 0,
            },
        ])

        profile = await service.behavioral_profile(1, days=30)
        # Events are heartbeart-merged, so fewer rows than raw inserts
        assert profile["total_events_analysed"] > 0
        assert len(profile["top_apps"]) > 0
        assert profile["avg_focus_block_min"] > 0
        assert profile["profile_date"] is not None

    async def test_peak_focus_hour_uses_business_timezone(
        self,
        repos,
        monkeypatch,
    ):
        """Peak-hour labels should reflect local wall-clock time, not UTC."""
        activity_repo, focus_repo = repos
        target = date(2026, 7, 20)
        await activity_repo.append_event(
            make_event(
                user_id=1,
                timestamp_utc=_utc("2026-07-19T23:30:00"),
                duration_s=300.0,
                process_name="Code.exe",
            )
        )
        await focus_repo.save_sessions(1, [{
            "date": target.isoformat(),
            "start_time": _utc("2026-07-19T23:30:00").isoformat(),
            "end_time": _utc("2026-07-20T00:00:00").isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 80.0,
            "switch_count": 0,
        }])
        monkeypatch.setattr(
            analysis_service_module,
            "business_today",
            lambda _timezone: target,
        )
        service = AnalysisService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            timezone="Asia/Shanghai",
        )

        profile = await service.behavioral_profile(1)

        assert profile["peak_focus_hours"][0]["hour"] == 7
