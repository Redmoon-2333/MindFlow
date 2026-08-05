"""CollectorService self-healing supervisor — recovery state + backoff retries.

Covers the state-transition contract of the single long-lived ``_task``
supervisor (``_run``):

  - Every 10 consecutive failures close the current run interval with
    ``failure=True``, set status ``degraded``, schedule the exact retry
    delay on ``RecoveryState``, and sleep it out inside a ``sleep=True``
    backoff interval — then the same supervisor opens a fresh run cycle.
  - The retry-delay sequence across repeated degradations is
    5 / 15 / 30 / 60 / 60 seconds (no real waiting).
  - The first successful tick sets ``running``, zeroes the tick-failure
    counter and clears recovery state via ``record_success``.
  - Manual stop during a run or during backoff prevents every later retry
    and ends ``stopped``; the sleep interval closes exactly once and is
    never marked manual without an actual stop request.
  - Concurrent start / stop / recovery still yields exactly one supervisor.
  - ``next_retry_at`` / ``last_error`` / ``recovery_attempts`` expose the
    recovery state.

All time is injected: ``sleep`` is a recording fake (returns instantly,
optionally gated to hold the supervisor inside a backoff wait) and ``now``
is a fixed UTC clock — no real backoff, no wall-clock dependence.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.domain.events import WindowSnapshot
from mindflow.infrastructure.collectors.base import EventCollector
from mindflow.infrastructure.repositories.base import ActivityRepository
from mindflow.ports import CollectorIntervalRecord
from mindflow.services.collector_service import CollectorService

_NOW = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)
"""Fixed UTC clock shared by every test — reproducible transitions."""


def _snapshot() -> WindowSnapshot:
    """Return a minimal valid WindowSnapshot."""
    return WindowSnapshot(
        app_name="Code",
        window_title="main.py",
        process_name="code.exe",
        is_idle=False,
        timestamp_utc=_NOW,
    )


class RecordingIntervalsPort:
    """In-memory CollectorIntervalsPort recording every open/close call."""

    def __init__(self) -> None:
        self.opens: list[CollectorIntervalRecord] = []
        self.closes: list[dict[str, Any]] = []
        self._seq = 0

    async def open(
        self,
        user_id: int,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord:
        self._seq += 1
        record = CollectorIntervalRecord(
            id=f"iv-{self._seq}",
            user_id=user_id,
            started_at=(now or _NOW).isoformat(),
            ended_at=None,
            reason=reason,
            manual_stop=manual_stop,
            failure=failure,
            sleep=sleep,
            last_error=None,
        )
        self.opens.append(record)
        return record

    async def close(
        self,
        interval_id: str,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord | None:
        self.closes.append(
            {
                "interval_id": interval_id,
                "reason": reason,
                "manual_stop": manual_stop,
                "failure": failure,
                "sleep": sleep,
                "last_error": last_error,
            }
        )
        return None

    async def list_by_user(
        self, user_id: int, *, limit: int = 100
    ) -> list[CollectorIntervalRecord]:
        return []


class FailingAuditIntervalsPort(RecordingIntervalsPort):
    """Recording port that raises once at one lifecycle audit operation."""

    def __init__(self, failure_point: str) -> None:
        super().__init__()
        self._failure_point = failure_point
        self._failed = False

    def _raise_once(self, operation: str) -> None:
        if operation == self._failure_point and not self._failed:
            self._failed = True
            raise RuntimeError(f"audit {operation} failed")

    async def open(
        self,
        user_id: int,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord:
        self._raise_once("open_sleep" if sleep else "open")
        return await super().open(
            user_id,
            reason=reason,
            manual_stop=manual_stop,
            failure=failure,
            sleep=sleep,
            now=now,
        )

    async def close(
        self,
        interval_id: str,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord | None:
        self._raise_once("close_sleep" if sleep else "close")
        return await super().close(
            interval_id,
            reason=reason,
            manual_stop=manual_stop,
            failure=failure,
            sleep=sleep,
            last_error=last_error,
            now=now,
        )


class AdvancingClock:
    """UTC clock advanced only by the injected sleep seam."""

    def __init__(self) -> None:
        self.current = _NOW

    def __call__(self) -> datetime:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)
        await asyncio.sleep(0)


class FakeSleep:
    """Recording fake sleep — never waits, optionally gated on backoff.

    Every requested delay is appended to ``calls``.  When a ``gate`` is
    set, backoff waits (delays >= 5s) block on it until released so tests
    can hold the supervisor inside a backoff sleep and then drive
    stop() / cancellation deterministically.
    """

    def __init__(self) -> None:
        self.calls: list[float] = []
        self._gate: asyncio.Event | None = None

    def gate(self, event: asyncio.Event) -> None:
        """Gate subsequent backoff waits on *event*."""
        self._gate = event

    @property
    def backoff_calls(self) -> list[float]:
        """Only the backoff delays (inter-tick sleeps are always < 5s)."""
        return [c for c in self.calls if c >= 5.0]

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if seconds >= 5.0 and self._gate is not None:
            await self._gate.wait()


@pytest.fixture
def failing_collector() -> MagicMock:
    """EventCollector whose snapshot always raises."""
    collector = MagicMock(spec=EventCollector)
    collector.snapshot = AsyncMock(side_effect=RuntimeError("Persistent failure"))
    collector.idle_seconds = AsyncMock(return_value=0.0)
    return collector


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock ActivityRepository (Protocol-based spec)."""
    repo = MagicMock(spec=ActivityRepository)
    repo.append_event = AsyncMock()
    return repo


def _make_service(
    collector: MagicMock,
    repository: MagicMock,
    *,
    interval_repository: RecordingIntervalsPort | None = None,
    sleep: FakeSleep | None = None,
    now: Any | None = None,
    interval_s: float = 0.01,
) -> CollectorService:
    """Construct a CollectorService with the deterministic seams."""
    return CollectorService(
        collector=collector,
        repository=repository,
        user_id=1,
        interval_s=interval_s,
        idle_threshold_s=60,
        interval_repository=interval_repository,
        sleep=sleep,
        now=now,
    )


async def _wait_until(predicate: Any, *, timeout: float = 5.0) -> None:
    """Poll *predicate* with tiny real sleeps; fail on timeout."""
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(timeout / 1000)
    raise AssertionError("condition not reached within timeout")


# ── Backoff sequence across repeated degradations ────────────────────────


async def test_repeated_degradations_backoff_sequence_5_15_30_60_60(
    failing_collector, mock_repository
) -> None:
    """Five consecutive degradations wait exactly 5/15/30/60/60 seconds."""
    fake_sleep = FakeSleep()
    service = _make_service(failing_collector, mock_repository, sleep=fake_sleep)

    await service.start()
    await _wait_until(lambda: len(fake_sleep.backoff_calls) >= 5)
    await service.stop()

    assert fake_sleep.backoff_calls[:5] == [5.0, 15.0, 30.0, 60.0, 60.0]
    assert service.recovery_attempts >= 5
    # The supervisor survived — retries happened on the same _task.
    assert service._task is None


@pytest.mark.parametrize(
    "failure_point",
    ["open", "close", "open_sleep", "close_sleep"],
)
async def test_interval_audit_failure_does_not_kill_supervisor_and_recovers(
    failure_point, mock_repository
) -> None:
    """Interval audit persistence is best-effort and never blocks collection."""
    collector = MagicMock(spec=EventCollector)
    collector.idle_seconds = AsyncMock(return_value=0.0)
    failures = {"remaining": 10}

    async def fail_then_recover():
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise RuntimeError("collector unavailable")
        return _snapshot()

    collector.snapshot = AsyncMock(side_effect=fail_then_recover)
    audit_repo = FailingAuditIntervalsPort(failure_point)
    fake_sleep = FakeSleep()
    service = _make_service(
        collector,
        mock_repository,
        interval_repository=audit_repo,
        sleep=fake_sleep,
        now=lambda: _NOW,
    )

    await service.start()
    try:
        await _wait_until(
            lambda: (
                mock_repository.append_event.await_count >= 1
                and service.status == "running"
            )
            or (service._task is not None and service._task.done())
        )
        assert mock_repository.append_event.await_count >= 1
        assert service.status == "running"
        assert service._task is not None
        assert not service._task.done()
    finally:
        with contextlib.suppress(RuntimeError):
            await service.stop()


async def test_failed_run_interval_open_is_retried_before_stop(
    mock_repository,
) -> None:
    """A transient open failure still produces one closed run interval."""
    collector = MagicMock(spec=EventCollector)
    collector.snapshot = AsyncMock(return_value=_snapshot())
    collector.idle_seconds = AsyncMock(return_value=0.0)
    audit_repo = FailingAuditIntervalsPort("open")
    service = _make_service(
        collector,
        mock_repository,
        interval_repository=audit_repo,
        now=lambda: _NOW,
    )

    await service.start()
    await _wait_until(lambda: mock_repository.append_event.await_count >= 1)
    await service.stop()

    assert len(audit_repo.opens) == 1
    assert len(audit_repo.closes) == 1
    assert audit_repo.closes[0]["interval_id"] == audit_repo.opens[0].id
    assert audit_repo.closes[0]["manual_stop"] is True


async def test_failed_run_interval_close_is_retried_before_stop_returns(
    mock_repository,
) -> None:
    """A transient close failure is reconciled with terminal facts."""
    collector = MagicMock(spec=EventCollector)
    collector.snapshot = AsyncMock(return_value=_snapshot())
    collector.idle_seconds = AsyncMock(return_value=0.0)
    audit_repo = FailingAuditIntervalsPort("close")
    service = _make_service(
        collector,
        mock_repository,
        interval_repository=audit_repo,
        now=lambda: _NOW,
    )

    await service.start()
    await _wait_until(lambda: mock_repository.append_event.await_count >= 1)
    await service.stop()

    assert len(audit_repo.opens) == 1
    assert len(audit_repo.closes) == 1
    assert audit_repo.closes[0]["interval_id"] == audit_repo.opens[0].id
    assert audit_repo.closes[0]["manual_stop"] is True


async def test_recovered_run_resets_ticker_duration_after_backoff(
    mock_repository,
) -> None:
    """The first event in a recovered run excludes degraded/backoff time."""
    collector = MagicMock(spec=EventCollector)
    collector.idle_seconds = AsyncMock(return_value=0.0)
    failures = {"remaining": 10}

    async def fail_then_recover():
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise RuntimeError("collector unavailable")
        return _snapshot()

    collector.snapshot = AsyncMock(side_effect=fail_then_recover)
    clock = AdvancingClock()
    service = _make_service(
        collector,
        mock_repository,
        sleep=clock.sleep,
        now=clock,
        interval_s=1.0,
    )

    await service.start()
    await _wait_until(lambda: mock_repository.append_event.await_count >= 1)
    first_recovered_event = mock_repository.append_event.await_args_list[0].args[0]
    await service.stop()

    assert first_recovered_event.duration_s == 1.0


# ── Degraded cycle: run interval closes failure, sleep interval spans wait ──


async def test_degraded_cycle_closes_run_failure_and_opens_sleep_interval(
    failing_collector, mock_repository
) -> None:
    """One degradation closes the run failure=True and opens a sleep=True
    backoff interval; the run interval is never marked manual."""
    gate = asyncio.Event()
    fake_sleep = FakeSleep()
    fake_sleep.gate(gate)
    repo = RecordingIntervalsPort()
    service = _make_service(
        failing_collector,
        mock_repository,
        interval_repository=repo,
        sleep=fake_sleep,
    )

    await service.start()
    try:
        # Supervisor is now inside the gated backoff sleep.
        await _wait_until(lambda: len(fake_sleep.backoff_calls) >= 1)

        assert service.status == "degraded"
        assert len(repo.opens) == 2  # run + sleep
        assert len(repo.closes) == 1  # only the run interval so far
        run_open, sleep_open = repo.opens
        assert run_open.sleep is False
        assert sleep_open.sleep is True
        assert sleep_open.reason == "collector backoff (system sleep)"

        run_close = repo.closes[0]
        assert run_close["interval_id"] == run_open.id
        assert run_close["failure"] is True
        assert run_close["manual_stop"] is False
        assert run_close["reason"] == "collector degraded (10 consecutive failures)"
        assert run_close["last_error"] == "RuntimeError: Persistent failure"
    finally:
        await service.stop()

    # Stop closes the sleep interval exactly once, truthfully manual.
    assert len(repo.closes) == 2
    sleep_close = repo.closes[1]
    assert sleep_close["interval_id"] == sleep_open.id
    assert sleep_close["sleep"] is True
    assert sleep_close["manual_stop"] is True
    assert sleep_close["failure"] is False
    assert sleep_close["reason"] == "manual stop"


# ── State transition table + recovery properties ─────────────────────────


async def test_state_transitions_and_recovery_properties(mock_repository) -> None:
    """running → degraded → running → stopped, with recovery state
    populated on failure and cleared on the first successful tick."""
    collector = MagicMock(spec=EventCollector)
    collector.idle_seconds = AsyncMock(return_value=0.0)
    failures = {"n": 0}

    async def flaky():
        if failures["n"] < 10:
            failures["n"] += 1
            raise RuntimeError("flaky")
        return _snapshot()

    collector.snapshot = AsyncMock(side_effect=flaky)
    gate = asyncio.Event()
    fake_sleep = FakeSleep()
    fake_sleep.gate(gate)
    service = _make_service(
        collector, mock_repository, sleep=fake_sleep, now=lambda: _NOW
    )

    await service.start()
    assert service.status == "running"
    assert service.recovery_attempts == 0
    assert service.next_retry_at is None
    assert service.last_error is None

    # 10 failures → degraded; the backoff sleep is gated so the recovery
    # state is stable to inspect.
    await _wait_until(lambda: service.status == "degraded")
    assert service.recovery_attempts == 1
    assert service.next_retry_at == _NOW + timedelta(seconds=5.0)
    assert service.last_error == "RuntimeError: flaky"

    # Release the backoff: the fresh cycle's first successful tick
    # restores running and clears recovery state.
    gate.set()
    await _wait_until(lambda: service.recovery_attempts == 0)
    assert service.status == "running"
    assert service._consecutive_failures == 0
    assert service.next_retry_at is None
    assert service.last_error is None

    await service.stop()
    assert service.status == "stopped"


# ── Manual stop during backoff prevents all later retries ────────────────


async def test_manual_stop_during_backoff_prevents_later_retries(
    failing_collector, mock_repository
) -> None:
    """stop() during a gated backoff cancels the wait, closes the sleep
    interval once as manual, and prevents every later retry."""
    gate = asyncio.Event()
    fake_sleep = FakeSleep()
    fake_sleep.gate(gate)
    repo = RecordingIntervalsPort()
    service = _make_service(
        failing_collector,
        mock_repository,
        interval_repository=repo,
        sleep=fake_sleep,
    )

    await service.start()
    await _wait_until(lambda: service.recovery_attempts == 1)
    assert service.status == "degraded"

    ticks_before = failing_collector.snapshot.await_count
    await service.stop()

    assert service.status == "stopped"
    assert service._task is None
    # No fresh run cycle began — no tick ran after the backoff stop.
    assert failing_collector.snapshot.await_count == ticks_before
    # Run + sleep intervals, both closed exactly once.
    assert len(repo.opens) == 2
    assert len(repo.closes) == 2
    sleep_close = repo.closes[1]
    assert sleep_close["sleep"] is True
    assert sleep_close["manual_stop"] is True
    assert sleep_close["failure"] is False
    assert sleep_close["reason"] == "manual stop"
    # Recovery never advanced to a second retry.
    assert service.recovery_attempts == 1


# ── Cancellation during backoff: sleep interval closes, restart reopens ──


async def test_cancelled_stop_during_backoff_closes_sleep_once_and_restart_reopens(
    failing_collector, mock_repository
) -> None:
    """Cancelling stop() mid-backoff still closes the sleep interval once
    and a restart opens a fresh run interval (no stale backoff retry)."""
    # Wide interval gives stop()'s own timeout a large margin, so the
    # explicit cancel below deterministically lands first.
    gate = asyncio.Event()
    fake_sleep = FakeSleep()
    fake_sleep.gate(gate)
    repo = RecordingIntervalsPort()
    service = _make_service(
        failing_collector,
        mock_repository,
        interval_repository=repo,
        sleep=fake_sleep,
        interval_s=1.0,
    )

    await service.start()
    await _wait_until(lambda: service.recovery_attempts == 1)
    assert len(repo.opens) == 2

    stop_task = asyncio.create_task(service.stop())
    # Marker task scheduled after stop_task runs only once stop() has
    # yielded at its await (tasks run in FIFO creation order).
    await _wait_until(lambda: service._status == "stopping")
    background_task = service._task
    assert background_task is not None

    stop_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stop_task

    # stop()'s cancel handler cancelled the supervisor mid-backoff; the
    # sleep interval closes exactly once — manual, because a stop WAS
    # requested — and lifecycle state is fully cleaned up.
    assert len(repo.opens) == 2
    assert len(repo.closes) == 2
    sleep_close = repo.closes[1]
    assert sleep_close["sleep"] is True
    assert sleep_close["manual_stop"] is True
    assert sleep_close["failure"] is False
    assert sleep_close["reason"] == "manual stop"
    assert service.status == "stopped"
    assert service._task is None

    # Restart opens a fresh run interval, never a stale backoff retry.
    failing_collector.snapshot = AsyncMock(return_value=_snapshot())
    await service.start()
    assert service.status == "running"
    for _ in range(100):
        if len(repo.opens) >= 3:
            break
        await asyncio.sleep(0.01)
    assert len(repo.opens) == 3
    assert repo.opens[2].sleep is False
    await service.stop()
    assert len(repo.closes) == 3


# ── Concurrent start/stop/recovery yields exactly one supervisor ─────────


async def test_start_during_backoff_stop_is_noop_single_supervisor(
    failing_collector, mock_repository
) -> None:
    """start() raced against a stop that is cancelling a backoff is a
    no-op — concurrent lifecycle calls never spawn a second supervisor."""
    gate = asyncio.Event()
    fake_sleep = FakeSleep()
    fake_sleep.gate(gate)
    service = _make_service(failing_collector, mock_repository, sleep=fake_sleep)

    await service.start()
    supervisor_task = service._task
    assert supervisor_task is not None
    await _wait_until(lambda: service.recovery_attempts == 1)

    stop_task = asyncio.create_task(service.stop())
    await _wait_until(lambda: service._status == "stopping")

    # start() while stop is mid-flight must be a no-op.
    await service.start()
    assert service._task is supervisor_task
    await stop_task

    assert service.status == "stopped"
    assert service._task is None
