"""Deterministic concurrency tests for the intervention reliability dispatch
transaction (safety guard + atomic daily-slot reservation + release-on-failure)
against a real SQLite database.

Drive with ``asyncio.gather`` / ``asyncio.Event`` (no sleeps) on a WAL-mode
SQLite file with a 5s busy timeout so the atomic ``INSERT … ON CONFLICT
DO NOTHING`` reservation arbitrates exactly one winner.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import async_sessionmaker

from mindflow.domain.procrastination import (
    CBTTechnique,
    ProcrastinationAssessment,
    ProcrastinationType,
)
from mindflow.infrastructure.database import create_engine
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
    intervention_logs,
)
from mindflow.infrastructure.schema import intervention_slot_reservations
from mindflow.services.intervention_service import InterventionService
from mindflow.services.intervention_throttle import InterventionThrottle


def _assessment() -> ProcrastinationAssessment:
    return ProcrastinationAssessment(
        types=(ProcrastinationType.IMPULSIVITY,),
        confidence={ProcrastinationType.IMPULSIVITY: 0.8},
        recommended_technique=CBTTechnique.STIMULUS_CONTROL,
        rationale="检测到冲动分心模式",
        source="rule_engine",
    )


def _throttle(repo: InterventionLogRepository) -> InterventionThrottle:
    return InterventionThrottle(
        repo=repo,
        daily_limit=3,
        type_limit=2,
        cooldown_h=2.0,
        ignore_rate_threshold=0.6,
        fatigue_daily_limit=1,
    )


@pytest.fixture
async def engine(tmp_path):
    """Isolated WAL-mode SQLite engine with intervention + reservation tables."""
    engine = create_engine(f"sqlite+aiosqlite:///{tmp_path / 'concurrency.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(intervention_slot_reservations.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


async def _count_rows(session_factory, table) -> int:
    async with session_factory() as session:
        result = await session.execute(sa.select(table.c.id))
        return len(result.fetchall())


async def _slot_indices(session_factory, table) -> list[int]:
    async with session_factory() as session:
        result = await session.execute(
            sa.select(table.c.slot_index).order_by(table.c.slot_index)
        )
        return [row[0] for row in result.fetchall()]


async def test_concurrent_identical_calls_exactly_one_wins(engine, session_factory) -> None:
    """Two identical eligible concurrent calls → exactly one reserves, persists,
    and dispatches; the loser skips with no side effects."""
    assessment = _assessment()
    notifier = AsyncMock()
    broadcast = AsyncMock(return_value=1)

    # Two repo instances over the SAME session factory so both calls share one
    # reservation table while staying independent services.
    repo_a = InterventionLogRepository(session_factory=session_factory)
    repo_b = InterventionLogRepository(session_factory=session_factory)

    service_a = InterventionService(
        intervention_repo=repo_a,
        throttle=_throttle(repo_a),
        notifier=notifier,
        broadcast_fn=broadcast,
    )
    service_b = InterventionService(
        intervention_repo=repo_b,
        throttle=_throttle(repo_b),
        notifier=notifier,
        broadcast_fn=broadcast,
    )

    # Gate call A's persistence on call B finishing. Call B is never gated, so
    # it always completes and can only contend for the SAME slot (today_count
    # stays 0 while A is gated) — making the single-winner outcome deterministic
    # regardless of coroutine interleaving.
    second_call_finished = asyncio.Event()
    real_log_a = repo_a.log_triggered

    async def gated_log_triggered(**kwargs: object) -> dict[str, object]:
        await second_call_finished.wait()
        return await real_log_a(**kwargs)

    repo_a.log_triggered = gated_log_triggered  # type: ignore[method-assign]

    async def runner_b():
        try:
            return await service_b.maybe_intervene(assessment=assessment, user_id=1)
        finally:
            second_call_finished.set()

    results = await asyncio.gather(
        service_a.maybe_intervene(assessment=assessment, user_id=1),
        runner_b(),
    )

    dispatched = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]
    assert len(dispatched) == 1, f"expected exactly one winner, got {results}"
    assert len(skipped) == 1, f"expected exactly one loser, got {results}"
    assert skipped[0].skip_reason, "loser must carry an observable skip reason"

    assert await _count_rows(session_factory, intervention_slot_reservations) == 1
    assert await _count_rows(session_factory, intervention_logs) == 1
    assert broadcast.await_count == 1
    assert notifier.send.await_count == 1


async def test_cancellation_during_persistence_releases_slot_and_reraises(
    engine, session_factory,
) -> None:
    """A runtime cancellation after reservation but during persistence frees the
    exact slot, creates no log/broadcast/notification, re-raises the
    cancellation, and a same-day retry can reserve and dispatch."""
    assessment = _assessment()
    notifier = AsyncMock()
    broadcast = AsyncMock(return_value=1)
    repo = InterventionLogRepository(session_factory=session_factory)
    service = InterventionService(
        intervention_repo=repo,
        throttle=_throttle(repo),
        notifier=notifier,
        broadcast_fn=broadcast,
    )

    persistence_entered = asyncio.Event()
    resume = asyncio.Event()
    real_log = repo.log_triggered

    async def cancel_during_persistence(**kwargs: object) -> dict[str, object]:
        persistence_entered.set()
        await resume.wait()
        raise asyncio.CancelledError()

    repo.log_triggered = cancel_during_persistence  # type: ignore[method-assign]

    task = asyncio.create_task(
        service.maybe_intervene(assessment=assessment, user_id=1)
    )
    await persistence_entered.wait()
    # The reservation is owned and persistence is blocked — the cancellation
    # lands after ownership, before persistence completes.
    assert await _count_rows(session_factory, intervention_slot_reservations) == 1

    # A sibling slot (2) held by another caller must be preserved when slot 1
    # is released by the cancellation cleanup.
    assert await repo.try_reserve_daily_slot(1, 2, "nudge") is True

    resume.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Cancellation-safe cleanup: exact slot released (sibling slot 2 intact),
    # nothing persisted or dispatched.
    assert await _slot_indices(session_factory, intervention_slot_reservations) == [2]
    assert await _count_rows(session_factory, intervention_logs) == 0
    assert broadcast.await_count == 0
    assert notifier.send.await_count == 0

    # A same-day retry wins the freed slot and dispatches normally.
    repo.log_triggered = real_log
    retry = await service.maybe_intervene(assessment=assessment, user_id=1)
    assert not retry.skipped
    assert retry.intervention is not None
    assert await _slot_indices(session_factory, intervention_slot_reservations) == [1, 2]
    assert await _count_rows(session_factory, intervention_logs) == 1
    assert broadcast.await_count == 1
    assert notifier.send.await_count == 1


async def test_persistence_failure_releases_slot_so_retry_wins(engine, session_factory) -> None:
    """A persistence failure releases the reserved slot and dispatches nothing;
    a retry then wins the same slot and dispatches normally."""
    assessment = _assessment()
    notifier = AsyncMock()
    broadcast = AsyncMock(return_value=1)
    repo = InterventionLogRepository(session_factory=session_factory)
    service = InterventionService(
        intervention_repo=repo,
        throttle=_throttle(repo),
        notifier=notifier,
        broadcast_fn=broadcast,
    )

    real_log = repo.log_triggered

    async def failing_log_triggered(**kwargs: object) -> dict[str, object]:
        raise RuntimeError("simulated persistence failure")

    repo.log_triggered = failing_log_triggered  # type: ignore[method-assign]

    # First attempt: reservation is taken, persistence fails → slot released,
    # nothing dispatched.
    result = await service.maybe_intervene(assessment=assessment, user_id=1)
    assert result.skipped
    assert "持久化" in result.skip_reason
    assert await _count_rows(session_factory, intervention_slot_reservations) == 0
    assert await _count_rows(session_factory, intervention_logs) == 0
    assert broadcast.await_count == 0
    assert notifier.send.await_count == 0

    # Retry: same user/date slot is free again → wins, persists, dispatches.
    repo.log_triggered = real_log
    result2 = await service.maybe_intervene(assessment=assessment, user_id=1)
    assert not result2.skipped
    assert result2.intervention is not None
    assert await _count_rows(session_factory, intervention_slot_reservations) == 1
    assert await _count_rows(session_factory, intervention_logs) == 1
    assert broadcast.await_count == 1
    assert notifier.send.await_count == 1
