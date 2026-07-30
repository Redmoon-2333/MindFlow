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


# ── Budget reservation concurrent tests ────────────────────────────────


async def _create_workflow_tables(engine) -> None:
    """Create workflow tables for reservation tests."""
    from mindflow.infrastructure.schema import (
        workflow_runs,
    )

    async with engine.begin() as connection:
        await connection.run_sync(workflow_runs.metadata.create_all)


async def test_budget_reservation_atomic_one_winner(
    engine, session_factory,
) -> None:
    """Concurrent budget reservations permit exactly one winner."""
    await _create_workflow_tables(engine)

    from mindflow.infrastructure.repositories.workflow_runs import (
        BudgetReservationRepository,
    )

    repo = BudgetReservationRepository(session_factory)
    key = "scheduler:1:2026-07-26:daily_attribution"

    results = await asyncio.gather(
        repo.try_reserve(key, cost_estimate=1.0),
        repo.try_reserve(key, cost_estimate=1.0),
        repo.try_reserve(key, cost_estimate=1.0),
    )
    assert results.count(True) == 1, f"Expected exactly 1 winner, got {results.count(True)}"
    assert results.count(False) == 2, f"Expected 2 losers, got {results.count(False)}"


async def test_budget_reservation_different_keys_both_succeed(
    engine, session_factory,
) -> None:
    """Different idempotency keys can both reserve."""
    await _create_workflow_tables(engine)

    from mindflow.infrastructure.repositories.workflow_runs import (
        BudgetReservationRepository,
    )

    repo = BudgetReservationRepository(session_factory)

    results = await asyncio.gather(
        repo.try_reserve("scheduler:1:2026-07-26:daily_panel"),
        repo.try_reserve("scheduler:1:2026-07-26:daily_attribution"),
    )
    assert results == [True, True]


async def test_budget_reservation_release_allows_re_reserve(
    engine, session_factory,
) -> None:
    """After release, a new reservation for the same key succeeds."""
    await _create_workflow_tables(engine)

    from mindflow.infrastructure.repositories.workflow_runs import (
        BudgetReservationRepository,
    )

    repo = BudgetReservationRepository(session_factory)
    key = "scheduler:2:2026-07-27:daily_attribution"

    assert await repo.try_reserve(key) is True
    assert await repo.try_reserve(key) is False

    await repo.release(key)

    assert await repo.try_reserve(key) is True


async def test_budget_reservation_idempotency_key_unique_insert_fails(
    engine, session_factory,
) -> None:
    """Direct INSERT with duplicate idempotency_key raises IntegrityError."""
    await _create_workflow_tables(engine)

    import sqlalchemy as sa

    from mindflow.infrastructure.schema import workflow_budget_reservations

    values = {
        "id": "id-1",
        "workflow_name": "daily_analysis",
        "run_id": "run-1",
        "origin": "scheduler",
        "idempotency_key": "dup-key",
        "user_id": 1,
        "target_date": "2026-07-26",
        "budget_type": "cost_1.0",
        "reserved_at": "2026-07-26T00:00:00Z",
    }
    async with session_factory() as session, session.begin():
        await session.execute(sa.insert(workflow_budget_reservations).values(**values))

        with __import__("pytest").raises(Exception):
            await session.execute(sa.insert(workflow_budget_reservations).values(**{**values, "id": "id-2"}))
            await session.commit()
