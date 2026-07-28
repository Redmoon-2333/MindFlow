"""Persistent scheduled job run claim tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

import sqlalchemy as sa

from mindflow.infrastructure.repositories.scheduled_jobs import (
    SCHEDULED_JOB_RUN_LEASE,
    ScheduledJobRunsRepository,
    scheduled_job_runs,
)


async def _create_table(engine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(scheduled_job_runs.metadata.create_all)


async def test_claim_is_atomic_and_failed_run_requires_explicit_retry(
    engine, session_factory
) -> None:
    await _create_table(engine)
    repository = ScheduledJobRunsRepository(session_factory)
    local_date = date(2026, 7, 26)
    claims = await asyncio.gather(
        repository.claim("daily_report", local_date), repository.claim("daily_report", local_date)
    )
    assert claims.count(1) == 1
    assert claims.count(None) == 1
    assert await repository.mark_failed(
        "daily_report", local_date, attempt_count=1, error="boom"
    )
    assert await repository.claim("daily_report", local_date) is None
    retry_claims = await asyncio.gather(
        repository.claim("daily_report", local_date, retry_failed=True),
        repository.claim("daily_report", local_date, retry_failed=True),
    )
    assert retry_claims.count(2) == 1
    assert retry_claims.count(None) == 1
    async with session_factory() as session:
        row = (await session.execute(sa.select(scheduled_job_runs))).one()
    assert row.status == "running"
    assert row.attempt_count == 2
    assert row.last_error is None


async def test_heartbeat_renews_lease_and_expired_claim_is_atomically_fenced(
    engine, session_factory
) -> None:
    await _create_table(engine)
    repository = ScheduledJobRunsRepository(session_factory)
    local_date = date(2026, 7, 26)
    assert await repository.claim("daily_backup", local_date) == 1

    original_lease_expired_at = (
        datetime.now(UTC) - SCHEDULED_JOB_RUN_LEASE - timedelta(seconds=1)
    )
    async with session_factory() as session, session.begin():
        await session.execute(
            sa.update(scheduled_job_runs)
            .where(
                scheduled_job_runs.c.job_name == "daily_backup",
                scheduled_job_runs.c.local_date == local_date.isoformat(),
            )
            .values(started_at=original_lease_expired_at.isoformat())
        )

    assert await repository.heartbeat("daily_backup", local_date, attempt_count=1)
    assert await repository.claim("daily_backup", local_date) is None

    heartbeat_expired_at = datetime.now(UTC) - SCHEDULED_JOB_RUN_LEASE - timedelta(seconds=1)
    async with session_factory() as session, session.begin():
        await session.execute(
            sa.update(scheduled_job_runs)
            .where(
                scheduled_job_runs.c.job_name == "daily_backup",
                scheduled_job_runs.c.local_date == local_date.isoformat(),
            )
            .values(heartbeat_at=heartbeat_expired_at.isoformat())
        )

    claims = await asyncio.gather(
        repository.claim("daily_backup", local_date),
        repository.claim("daily_backup", local_date),
    )
    assert claims.count(2) == 1
    assert claims.count(None) == 1

    assert not await repository.mark_succeeded(
        "daily_backup", local_date, attempt_count=1
    )
    async with session_factory() as session:
        row = (await session.execute(sa.select(scheduled_job_runs))).one()
    assert row.status == "running"
    assert row.attempt_count == 2

    assert await repository.mark_succeeded(
        "daily_backup", local_date, attempt_count=2
    )
    assert await repository.claim("daily_backup", local_date, retry_failed=True) is None


async def test_cancelled_run_is_explicitly_retryable(engine, session_factory) -> None:
    await _create_table(engine)
    repository = ScheduledJobRunsRepository(session_factory)
    local_date = date(2026, 7, 26)
    assert await repository.claim("daily_panel", local_date) == 1

    assert await repository.mark_cancelled(
        "daily_panel", local_date, attempt_count=1
    )

    assert await repository.claim("daily_panel", local_date) is None
    assert await repository.claim("daily_panel", local_date, retry_failed=True) == 2
