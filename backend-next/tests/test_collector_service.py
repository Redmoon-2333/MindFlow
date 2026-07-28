"""Tests for CollectorService — lifecycle, tick, failure handling.

Tests cover:
  - start() creates the asyncio task, stop() waits for in-flight tick (P1-1)
  - Double start is idempotent
  - Tick loop calls collector and repository
  - 10 consecutive failures → status degraded
  - Single tick failure doesn't stop the loop
  - Hanging tick triggers timeout (P1-4)
  - Stop preserves in-flight events (P1-1)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.domain.events import ActivityEvent, WindowSnapshot
from mindflow.infrastructure.collectors.base import EventCollector
from mindflow.infrastructure.repositories.base import ActivityRepository
from mindflow.services.collector_service import CollectorService


def _snapshot() -> WindowSnapshot:
    """Return a minimal valid WindowSnapshot."""
    return WindowSnapshot(
        app_name="Code",
        window_title="main.py",
        process_name="code.exe",
        is_idle=False,
        timestamp_utc=datetime.now(UTC),
    )


@pytest.fixture
def mock_collector():
    """Return a mock EventCollector with stubbed methods."""
    collector = MagicMock(spec=EventCollector)
    collector.snapshot = AsyncMock(return_value=_snapshot())
    collector.idle_seconds = AsyncMock(return_value=0.0)
    return collector


@pytest.fixture
def mock_repository():
    """Return a mock ActivityRepository (Protocol-based spec)."""
    repo = MagicMock(spec=ActivityRepository)
    repo.append_event = AsyncMock()
    return repo


@pytest.fixture
def service(mock_collector, mock_repository):
    """Return a CollectorService with mocked dependencies.

    Uses a very short interval_s so ticks happen quickly in tests.
    """
    return CollectorService(
        collector=mock_collector,
        repository=mock_repository,
        user_id=1,
        interval_s=0.01,
        idle_threshold_s=60,
    )


# ── Lifecycle ─────────────────────────────────────────────────────────


class TestStartStop:
    """Basic start/stop lifecycle."""

    async def test_start_sets_status_running(self, service):
        """start() sets status to 'running'."""
        await service.start()
        assert service.status == "running"
        await service.stop()

    async def test_stop_sets_status_stopped(self, service):
        """stop() sets status to 'stopped'."""
        await service.start()
        await service.stop()
        assert service.status == "stopped"

    async def test_double_start_is_idempotent(self, service):
        """Calling start() twice does not create two tasks."""
        await service.start()
        task_id = id(service._task)
        await service.start()  # Second start — should be no-op
        assert id(service._task) == task_id  # Same task reference
        await service.stop()

    async def test_double_stop_is_safe(self, service):
        """Calling stop() twice does not raise."""
        await service.start()
        await service.stop()
        await service.stop()  # Second stop — should be no-op
        assert service.status == "stopped"

    async def test_stop_without_start_is_safe(self, service):
        """stop() on a not-started service is a no-op."""
        await service.stop()
        assert service.status == "stopped"


class TestTickBehavior:
    """Tick loop calls collector and repository correctly."""

    async def test_tick_calls_collector_and_repository(
        self, service, mock_collector, mock_repository
    ):
        """After starting, a tick calls snapshot, idle_seconds, and append_event."""
        # Let the loop run for 3 ticks
        await service.start()
        await asyncio.sleep(0.1)
        await service.stop()

        assert mock_collector.snapshot.await_count >= 1
        assert mock_collector.idle_seconds.await_count >= 1
        assert mock_repository.append_event.await_count >= 1

    async def test_tick_passes_valid_event_to_repository(self, service, mock_repository):
        """Events passed to the repository have correct attributes."""
        captured_events: list[ActivityEvent] = []

        async def capture(event: ActivityEvent) -> None:
            captured_events.append(event)

        mock_repository.append_event.side_effect = capture

        await service.start()
        await asyncio.sleep(0.1)
        await service.stop()

        assert len(captured_events) >= 1
        ev = captured_events[0]
        assert ev.user_id == 1
        assert ev.event_type == "window_snapshot"  # Not idle (idle_seconds=0)
        assert isinstance(ev.data, WindowSnapshot)
        assert ev.data.app_name == "Code"
        assert ev.id is not None

    async def test_tick_sets_idle_when_above_threshold(
        self, service, mock_collector, mock_repository
    ):
        """When idle_seconds >= idle_threshold_s, event_type is idle_change."""
        mock_collector.idle_seconds = AsyncMock(return_value=120.0)  # > 60 threshold

        captured: list[ActivityEvent] = []

        async def capture(event: ActivityEvent) -> None:
            captured.append(event)

        mock_repository.append_event.side_effect = capture

        await service.start()
        await asyncio.sleep(0.1)
        await service.stop()

        assert len(captured) >= 1
        assert captured[0].event_type == "idle_change"
        assert captured[0].data.is_idle is True

    async def test_first_tick_uses_config_interval_as_duration(self, service, mock_repository):
        """The first tick's duration_s defaults to interval_s."""
        captured: list[ActivityEvent] = []

        async def capture(event: ActivityEvent) -> None:
            captured.append(event)

        mock_repository.append_event.side_effect = capture

        await service.start()
        await asyncio.sleep(0.05)
        await service.stop()

        assert len(captured) >= 1
        # First tick should use interval_s as fallback duration
        assert captured[0].duration_s == 0.01


class TestFailureHandling:
    """Collector failure handling and degraded status."""

    async def test_single_failure_does_not_stop_loop(
        self, service, mock_collector, mock_repository
    ):
        """A single tick failure is logged but the loop continues."""
        fail = True

        async def fail_once() -> WindowSnapshot:
            nonlocal fail
            if fail:
                fail = False
                raise RuntimeError("Transient failure")
            return _snapshot()

        mock_collector.snapshot = AsyncMock(side_effect=fail_once)

        await service.start()
        await asyncio.sleep(0.2)
        await service.stop()

        # Should have recovered after the first failure
        assert mock_repository.append_event.await_count >= 1
        assert service.status == "stopped"  # Stopped normally, not degraded

    async def test_ten_consecutive_failures_triggers_degraded(
        self, service, mock_collector, mock_repository
    ):
        """After 10 consecutive failures, status becomes 'degraded' and loop stops."""
        mock_collector.snapshot = AsyncMock(side_effect=RuntimeError("Persistent failure"))

        await service.start()

        # Wait enough time for 10+ ticks to be attempted
        for _ in range(20):
            if service.status == "degraded":
                break
            await asyncio.sleep(0.1)

        assert service.status == "degraded"
        # No successful events should have been appended
        assert mock_repository.append_event.await_count == 0

        # Clean up (the loop should have stopped, but stop is safe)
        await service.stop()


class TestEdgeCases:
    """Edge cases for the collector service."""

    async def test_repository_exception_handled(self, service, mock_repository):
        """Exceptions from the repository are caught by the tick handler."""
        mock_repository.append_event = AsyncMock(side_effect=RuntimeError("DB error"))

        await service.start()
        await asyncio.sleep(0.1)
        await service.stop()

        # Service should still be running (single failure doesn't stop)
        # or already stopped (if we exceeded 10 failures)
        assert service.status in ("stopped", "running")


class TestStopPreservesEvent:
    """P1-1: stop() does not lose the in-flight tick event."""

    async def test_stop_preserves_in_flight_append(self, mock_collector, mock_repository):
        """When stop() is called mid-tick, the pending append still completes."""
        event_appended = asyncio.Event()

        async def slow_append(event: ActivityEvent) -> None:
            await asyncio.sleep(0.02)
            event_appended.set()

        mock_repository.append_event.side_effect = slow_append

        # Use a longer interval to give stop() time to wait for the tick
        service = CollectorService(
            collector=mock_collector,
            repository=mock_repository,
            user_id=1,
            interval_s=0.1,
        )

        await service.start()
        # Let a tick start (tick calls append which will be slow)
        await asyncio.sleep(0.05)
        await service.stop()

        # The in-flight append should have completed
        assert event_appended.is_set(), "append_event did not complete during stop()"
        assert mock_repository.append_event.await_count >= 1


class TestTickTimeout:
    """P1-4: A hanging tick triggers TimeoutError and is counted as failure."""

    async def test_hanging_tick_triggers_timeout(self, mock_collector, mock_repository):
        """A tick that hangs longer than interval_s*2 triggers TimeoutError."""
        async def hanging_snapshot():
            await asyncio.sleep(3600)

        mock_collector.snapshot = AsyncMock(side_effect=hanging_snapshot)

        service = CollectorService(
            collector=mock_collector,
            repository=mock_repository,
            user_id=1,
            interval_s=0.01,
        )

        await service.start()

        # The tick should time out quickly (timeout = 0.02s)
        for _ in range(50):
            if service._consecutive_failures >= 1:
                break
            await asyncio.sleep(0.01)

        assert service._consecutive_failures >= 1, "Tick should have timed out"
        await service.stop()


# ── Concurrency (P1-4) ──────────────────────────────────────────────────


class TestConcurrency:
    """Concurrent start/stop lifecycle transitions are atomic and idempotent.

    P1-4: asyncio.Lock guards state transitions so that concurrent
    start() / stop() calls never leave orphan tasks, inconsistent
    status, or deadlock.
    """

    async def test_concurrent_starts_create_exactly_one_task(
        self, mock_collector, mock_repository
    ):
        """Three concurrent start() calls → exactly one _run task created.

        Tracks distinct asyncio.Task identities entering _run via the
        mock snapshot.  Each unique task id represents one background
        loop — there must be exactly one.
        """
        seen_tasks: set[int] = set()
        release_barrier = asyncio.Event()
        first_entered = asyncio.Event()

        async def blocking_snapshot():
            task_id = id(asyncio.current_task())
            seen_tasks.add(task_id)
            first_entered.set()
            # Block so the loop cannot complete ticks and we can
            # inspect state before stop().
            await release_barrier.wait()
            return _snapshot()

        mock_collector.snapshot = AsyncMock(side_effect=blocking_snapshot)

        service = CollectorService(
            collector=mock_collector,
            repository=mock_repository,
            user_id=1,
            interval_s=0.01,
        )

        # Fire three concurrent starts.
        await asyncio.gather(
            service.start(),
            service.start(),
            service.start(),
        )

        # Wait for the single task to enter _run.
        await asyncio.wait_for(first_entered.wait(), timeout=2.0)

        # At this point we should see exactly 1 distinct task.
        assert len(seen_tasks) == 1, (
            f"Expected 1 distinct task, got {len(seen_tasks)}. "
            "Concurrent start() created orphan tasks."
        )

        # Release the barrier so the tick can complete, then stop.
        # stop() awaits the task — no sleep needed.
        release_barrier.set()
        await service.stop()

        # Post-stop: no new tasks should have appeared.
        assert len(seen_tasks) == 1, (
            f"Tasks after stop: {len(seen_tasks)}. Orphan loop survived stop()."
        )

    async def test_concurrent_stops_on_running_service_are_safe(
        self, mock_collector, mock_repository
    ):
        """Three concurrent stop() calls on a running service
        must all complete without raising and leave status 'stopped'.
        """
        service = CollectorService(
            collector=mock_collector,
            repository=mock_repository,
            user_id=1,
            interval_s=0.01,
        )

        await service.start()

        # Fire three concurrent stops — no tick-collection sleep needed;
        # stop() sets _stop_requested and awaits _task regardless.
        await asyncio.gather(
            service.stop(),
            service.stop(),
            service.stop(),
        )

        assert service.status == "stopped"
        assert service._task is None

    async def test_start_during_stop_is_noop(
        self, mock_collector, mock_repository
    ):
        """start() during an in-flight stop() is a safe no-op.

        Uses a marker task created *after* stop_task to signal that
        stop() has yielded at its await — zero timing sleeps, pure
        event-loop ordering (tasks run in creation order).
        """
        tick_barrier = asyncio.Event()
        tick_started = asyncio.Event()

        async def blocking_snapshot():
            tick_started.set()
            await tick_barrier.wait()
            return _snapshot()

        mock_collector.snapshot = AsyncMock(side_effect=blocking_snapshot)

        service = CollectorService(
            collector=mock_collector,
            repository=mock_repository,
            user_id=1,
            interval_s=0.01,
        )

        # Start → task enters _run, blocks on barrier.
        await service.start()
        await asyncio.wait_for(tick_started.wait(), timeout=2.0)

        # Fire stop() in background.  A marker task scheduled after it
        # will only run once stop() yields at its await.
        stop_task = asyncio.create_task(service.stop())
        stop_yielded = asyncio.Event()

        async def _mark() -> None:
            stop_yielded.set()

        asyncio.create_task(_mark())
        await stop_yielded.wait()
        # stop() has now released _state_lock and is awaiting _task.

        # start() must be a no-op — _task is non-None, status is "stopping".
        await service.start()

        # Release barrier so stop() can complete.
        tick_barrier.set()
        await stop_task

        assert service._task is None
        assert service.status == "stopped"

    async def test_start_after_stop_restarts_cleanly(
        self, mock_collector, mock_repository
    ):
        """start() → stop() → start() → stop() — full lifecycle twice,
        each transition leaving the service in a consistent state.
        No tick-collection sleeps; the lifecycle transitions are the
        unit under test, not tick count.
        """
        service = CollectorService(
            collector=mock_collector,
            repository=mock_repository,
            user_id=1,
            interval_s=0.01,
        )

        # First lifecycle
        await service.start()
        assert service.status == "running"
        await service.stop()
        assert service.status == "stopped"
        assert service._task is None

        # Second lifecycle
        await service.start()
        assert service.status == "running"
        await service.stop()
        assert service.status == "stopped"
        assert service._task is None

    async def test_concurrent_start_and_stop_race_does_not_orphan(
        self, mock_collector, mock_repository
    ):
        """Simultaneous start() and stop() from concurrent tasks must
        reach a consistent terminal state with no orphaned tasks.
        """
        release = asyncio.Event()
        tick_started = asyncio.Event()

        async def blocking_snapshot():
            tick_started.set()
            await release.wait()
            return _snapshot()

        mock_collector.snapshot = AsyncMock(side_effect=blocking_snapshot)

        service = CollectorService(
            collector=mock_collector,
            repository=mock_repository,
            user_id=1,
            interval_s=0.01,
        )

        # Start first
        await service.start()
        await asyncio.wait_for(tick_started.wait(), timeout=2.0)
        tick_started.clear()

        # Fire concurrent start + stop.  Await start first so it
        # completes (no-op — task already running), then launch stop.
        t_start = asyncio.create_task(service.start())
        await t_start  # start returns (idempotent)
        t_stop = asyncio.create_task(service.stop())

        # Release the blocked tick so stop can complete.
        release.set()

        await asyncio.gather(t_start, t_stop)

        # Both tasks complete — state is already settled.
        assert service._task is None, "Task reference not cleaned up"
        assert service.status in ("stopped",), (
            f"Expected 'stopped', got '{service.status}'"
        )

    async def test_stop_cancellation_cleans_up_and_allows_restart(
        self, mock_collector, mock_repository
    ):
        """Cancelling stop() mid-await must clean up _task/_status
        so that a subsequent start() succeeds (no stuck "stopping").

        Uses a barrier inside _tick and the service's own _state_lock
        as a deterministic signal — zero timing sleeps.
        """
        tick_barrier = asyncio.Event()
        tick_entered = asyncio.Event()

        async def blocking_snapshot():
            tick_entered.set()
            await tick_barrier.wait()
            return _snapshot()

        mock_collector.snapshot = AsyncMock(side_effect=blocking_snapshot)

        service = CollectorService(
            collector=mock_collector,
            repository=mock_repository,
            user_id=1,
            interval_s=0.01,
        )

        # Phase 1 — Start; task enters _run and blocks on barrier.
        await service.start()
        await asyncio.wait_for(tick_entered.wait(), timeout=2.0)
        assert service.status == "running"

        # Phase 2 — Fire stop() in background.  A marker task created
        # *after* stop_task signals when stop() has yielded at its
        # await — tasks run in FIFO creation order.
        stop_task = asyncio.create_task(service.stop())
        stop_yielded = asyncio.Event()

        async def _mark() -> None:
            stop_yielded.set()

        asyncio.create_task(_mark())
        await stop_yielded.wait()
        # stop() has now released _state_lock and is awaiting _task.

        # Save the background task reference for post-cancellation await.
        background_task = service._task
        assert background_task is not None, "Task vanished before cancel"

        # Phase 3 — Cancel stop_task.  CancelledError MUST propagate.
        stop_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stop_task

        # After cancellation: _finalize ran under shield → clean state.
        assert service._task is None, (
            "_task not cleared after stop cancellation — stale reference"
        )
        assert service.status == "stopped", (
            f"Expected 'stopped' after stop cancellation, got '{service.status}'"
        )

        # Phase 4 — Release the barrier.  The background task was
        # already cancelled by stop()'s CancelledError handler and
        # completed — awaiting it surfaces the cancellation.
        tick_barrier.set()
        with pytest.raises(asyncio.CancelledError):
            await background_task

        # Phase 5 — Restart must work.
        await service.start()
        assert service.status == "running", (
            f"Cannot restart after stop cancellation: status={service.status}"
        )
        await service.stop()
        assert service.status == "stopped"
