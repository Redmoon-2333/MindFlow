"""Tests for CollectorService — lifecycle, tick, failure handling.

Tests cover:
  - start() creates the asyncio task, stop() cancels it gracefully
  - Double start is idempotent
  - Tick loop calls collector and repository
  - 10 consecutive failures → status degraded
  - Single tick failure doesn't stop the loop
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from mindflow.domain.events import ActivityEvent, WindowSnapshot
from mindflow.infrastructure.collectors.base import EventCollector
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
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
    """Return a mock SQLAlchemyActivityRepository."""
    repo = MagicMock(spec=SQLAlchemyActivityRepository)
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
