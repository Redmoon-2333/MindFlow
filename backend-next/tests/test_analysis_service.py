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

    async def test_with_sessions(self, repos, service):
        """Sessions should produce pattern data."""
        _, focus_repo = repos

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

        # Insert events
        for i in range(10):
            ev = make_event(
                user_id=1,
                timestamp_utc=_BASE + timedelta(seconds=i * 5),
                duration_s=5.0,
                process_name="Code.exe",
            )
            await activity_repo.append_event(ev)

        # Add sessions
        await focus_repo.save_sessions(1, [
            {
                "date": "2026-07-17",
                "start_time": _utc("2026-07-17T08:00:00").isoformat(),
                "end_time": _utc("2026-07-17T08:30:00").isoformat(),
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
