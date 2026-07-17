"""Tests for SQLAlchemyDailyReportRepository.

Covers:
  - upsert: idempotent insert/update with UNIQUE constraint
  - get_by_date: single-report retrieval
  - query_range: date range retrieval
  - exists_for_date: idempotency check
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text

from mindflow.infrastructure.repositories.report import (
    SQLAlchemyDailyReportRepository,
    daily_reports,
)


def _report_dict(
    user_id: int = 1,
    date_str: str = "2026-07-17",
    focus_min: float = 120.0,
    distraction_min: float = 30.0,
    score: float = 75.0,
    switch_freq: float = 15.0,
    summary: str | None = None,
) -> dict:
    return {
        "user_id": user_id,
        "date": date_str,
        "total_focus_min": focus_min,
        "total_distraction_min": distraction_min,
        "focus_score": score,
        "top_apps": [{"app": "Code.exe", "minutes": 60.0}],
        "switch_frequency": switch_freq,
        "pattern_summary": summary,
    }


@pytest.fixture
async def repo(engine, session_factory):
    """Create a repository with daily_reports table created."""
    async with engine.begin() as conn:
        await conn.run_sync(daily_reports.metadata.create_all)
    return SQLAlchemyDailyReportRepository(session_factory=session_factory)


class TestUpsert:
    """Idempotent insert/update behaviour."""

    async def test_upsert_inserts_new(self, repo, engine):
        """Upsert should insert a new report."""
        result = await repo.upsert(_report_dict())
        assert result["date"] == "2026-07-17"
        assert result["focus_score"] == 75.0

        async with engine.connect() as conn:
            cnt = await conn.execute(text("SELECT count(*) FROM daily_reports"))
            assert cnt.scalar() == 1

    async def test_upsert_idempotent(self, repo, engine):
        """Upserting the same user+date twice should update, not duplicate."""
        await repo.upsert(_report_dict(score=75.0))
        r2 = await repo.upsert(_report_dict(score=80.0))
        assert r2["focus_score"] == 80.0

        async with engine.connect() as conn:
            cnt = await conn.execute(text("SELECT count(*) FROM daily_reports"))
            assert cnt.scalar() == 1

    async def test_upsert_different_dates(self, repo, engine):
        """Different dates should create separate rows."""
        await repo.upsert(_report_dict(date_str="2026-07-17"))
        await repo.upsert(_report_dict(date_str="2026-07-18"))

        async with engine.connect() as conn:
            cnt = await conn.execute(text("SELECT count(*) FROM daily_reports"))
            assert cnt.scalar() == 2

    async def test_upsert_different_users(self, repo, engine):
        """Different users on the same date should create separate rows."""
        await repo.upsert(_report_dict(user_id=1, date_str="2026-07-17"))
        await repo.upsert(_report_dict(user_id=2, date_str="2026-07-17"))

        async with engine.connect() as conn:
            cnt = await conn.execute(text("SELECT count(*) FROM daily_reports"))
            assert cnt.scalar() == 2


class TestGetByDate:
    """Single-report retrieval."""

    async def test_get_existing(self, repo):
        """Existing report should be returned."""
        await repo.upsert(_report_dict())
        result = await repo.get_by_date(1, date(2026, 7, 17))
        assert result is not None
        assert result["date"] == "2026-07-17"

    async def test_get_nonexistent(self, repo):
        """Non-existent report should return None."""
        result = await repo.get_by_date(1, date(2026, 7, 17))
        assert result is None


class TestQueryRange:
    """Date-range retrieval."""

    async def test_query_returns_reports_in_range(self, repo):
        """Reports within the range should be returned."""
        await repo.upsert(_report_dict(date_str="2026-07-17"))
        await repo.upsert(_report_dict(date_str="2026-07-18"))

        result = await repo.query_range(1, date(2026, 7, 17), date(2026, 7, 17))
        assert len(result) == 1

    async def test_query_empty(self, repo):
        """No reports in range should return empty list."""
        result = await repo.query_range(1, date(2026, 7, 17), date(2026, 7, 17))
        assert result == []


class TestExistsForDate:
    """Idempotency check."""

    async def test_exists_true(self, repo):
        """Should return True when report exists."""
        await repo.upsert(_report_dict())
        assert await repo.exists_for_date(1, date(2026, 7, 17))

    async def test_exists_false(self, repo):
        """Should return False when no report exists."""
        assert not await repo.exists_for_date(1, date(2026, 7, 17))
