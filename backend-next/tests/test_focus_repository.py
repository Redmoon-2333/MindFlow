"""Tests for SQLAlchemyFocusSessionRepository.

Covers:
  - save_sessions: batch insert, generated IDs
  - query_range: date filtering, ordering
  - exists_for_date: idempotency check
  - get_by_date: single-day retrieval
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import text

from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
    focus_sessions,
)


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def _session_dict(
    date_str: str = "2026-07-17",
    start_time: str = "2026-07-17T10:00:00",
    end_time: str = "2026-07-17T10:30:00",
    session_type: str = "focus",
    dominant_app: str = "Code.exe",
    focus_score: float = 80.0,
    switch_count: int = 0,
) -> dict:
    return {
        "date": date_str,
        "start_time": _utc(start_time).isoformat(),
        "end_time": _utc(end_time).isoformat(),
        "session_type": session_type,
        "dominant_app": dominant_app,
        "focus_score": focus_score,
        "switch_count": switch_count,
    }


@pytest.fixture
async def repo(engine, session_factory):
    """Create a repository with focus_sessions table created."""
    async with engine.begin() as conn:
        await conn.run_sync(focus_sessions.metadata.create_all)
    return SQLAlchemyFocusSessionRepository(session_factory=session_factory)


class TestSaveSessions:
    """Batch insert behaviour."""

    async def test_save_sessions_returns_with_ids(self, repo):
        """Saved sessions should have generated UUIDv7 ids."""
        sessions = [_session_dict()]
        result = await repo.save_sessions(1, sessions)
        assert len(result) == 1
        assert "id" in result[0]
        assert len(result[0]["id"]) > 20  # UUIDv7

    async def test_save_multiple_sessions(self, repo, engine):
        """Multiple sessions should all be persisted."""
        sessions = [
            _session_dict(start_time="2026-07-17T10:00:00"),
            _session_dict(start_time="2026-07-17T14:00:00"),
        ]
        await repo.save_sessions(1, sessions)

        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT count(*) FROM focus_sessions"))
            assert result.scalar() == 2

    async def test_save_different_users(self, repo, engine):
        """Sessions for different users should be stored separately."""
        s1 = _session_dict()
        s2 = _session_dict(start_time="2026-07-17T12:00:00")
        await repo.save_sessions(1, [s1])
        await repo.save_sessions(2, [s2])

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM focus_sessions WHERE user_id = 1")
            )
            assert result.scalar() == 1
            result = await conn.execute(
                text("SELECT count(*) FROM focus_sessions WHERE user_id = 2")
            )
            assert result.scalar() == 1


class TestQueryRange:
    """Time-range query semantics."""

    async def test_query_returns_sessions_in_range(self, repo):
        """Sessions within the date range should be returned."""
        await repo.save_sessions(1, [
            _session_dict(date_str="2026-07-17"),
            _session_dict(date_str="2026-07-18"),
            _session_dict(date_str="2026-07-19"),
        ])

        result = await repo.query_range(1, date(2026, 7, 17), date(2026, 7, 18))
        assert len(result) == 2
        assert result[0]["date"] == "2026-07-17"
        assert result[1]["date"] == "2026-07-18"

    async def test_query_empty_range(self, repo):
        """No sessions in range should return empty list."""
        result = await repo.query_range(1, date(2026, 7, 17), date(2026, 7, 17))
        assert result == []

    async def test_query_filters_user_id(self, repo):
        """Only sessions for the requested user should be returned."""
        await repo.save_sessions(1, [_session_dict()])
        await repo.save_sessions(2, [_session_dict(start_time="2026-07-17T12:00:00")])

        result = await repo.query_range(1, date(2026, 7, 17), date(2026, 7, 17))
        assert len(result) == 1
        assert result[0]["user_id"] == 1


class TestExistsForDate:
    """Idempotency check semantics."""

    async def test_exists_after_save(self, repo):
        """exists_for_date should return True after saving sessions."""
        assert not await repo.exists_for_date(1, date(2026, 7, 17))
        await repo.save_sessions(1, [_session_dict()])
        assert await repo.exists_for_date(1, date(2026, 7, 17))

    async def test_exists_for_different_user(self, repo):
        """Other user's sessions should not affect check."""
        await repo.save_sessions(1, [_session_dict()])
        assert not await repo.exists_for_date(2, date(2026, 7, 17))

    async def test_exists_for_different_date(self, repo):
        """Sessions on other dates should not affect check."""
        await repo.save_sessions(1, [_session_dict(date_str="2026-07-17")])
        assert not await repo.exists_for_date(1, date(2026, 7, 18))


class TestGetByDate:
    """Single-day retrieval."""

    async def test_get_by_date_returns_sessions(self, repo):
        """Sessions for the requested date should be returned."""
        await repo.save_sessions(1, [
            _session_dict(date_str="2026-07-17", start_time="2026-07-17T10:00:00"),
            _session_dict(date_str="2026-07-17", start_time="2026-07-17T14:00:00"),
        ])
        result = await repo.get_by_date(1, date(2026, 7, 17))
        assert len(result) == 2

    async def test_get_by_date_nonexistent(self, repo):
        """Non-existent date should return empty list."""
        result = await repo.get_by_date(1, date(2026, 7, 17))
        assert result == []
