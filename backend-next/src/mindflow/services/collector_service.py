"""Background collector service — 5-second tick loop for window tracking.

CollectorService owns the active collection loop: it polls the active
window at a fixed interval, constructs ActivityEvents, and persists
them through the ActivityRepository.

Key design decisions (ADR-007, ADR-002):
  - Own asyncio task per instance (not a global singleton).
  - Bare asyncio loop (no APScheduler) — matches the "no framework"
    spirit of the new architecture.
  - Single tick failure does not kill the loop; 10 consecutive failures
    transitions to ``degraded`` status and stops.
  - ``stop()`` is graceful — cancels the task and waits for the
    current tick to finish.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime

from loguru import logger

from mindflow.config import get_settings
from mindflow.domain.events import ActivityEvent, EventType, WindowSnapshot
from mindflow.domain.ids import new_id
from mindflow.infrastructure.collectors.base import EventCollector
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)

_IDLE_THRESHOLD_S: int = 60
"""Seconds of inactivity before marking a snapshot as idle."""


class CollectorService:
    """Background collector service — polls the active window on a tick.

    Not a singleton — each instance manages its own lifecycle. The caller
    is responsible for holding a reference and calling ``stop()`` during
    application shutdown.

    Args:
        collector: Platform-specific ``EventCollector``.
        repository: ``ActivityRepository`` for persisting events.
        user_id: User identifier to attach to collected events (default 1).
        interval_s: Tick interval in seconds (defaults to settings).
        idle_threshold_s: Seconds of no input before marking idle
            (default 60).
    """

    def __init__(
        self,
        collector: EventCollector,
        repository: SQLAlchemyActivityRepository,
        user_id: int = 1,
        interval_s: float | None = None,
        idle_threshold_s: int = _IDLE_THRESHOLD_S,
    ) -> None:
        self._collector = collector
        self._repository = repository
        self._user_id = user_id
        self._interval_s = (
            interval_s if interval_s is not None else float(get_settings().collect_interval_s)
        )
        self._idle_threshold_s = idle_threshold_s

        self._task: asyncio.Task[None] | None = None
        self._status: str = "stopped"
        self._consecutive_failures: int = 0
        self._last_tick_time: datetime | None = None

    @property
    def status(self) -> str:
        """Return current status: stopped, running, stopping, or degraded."""
        return self._status

    async def start(self) -> None:
        """Start the collection loop.

        If the service is already running this is a no-op (idempotent).
        """
        if self._task is not None:
            logger.warning("CollectorService already running (start ignored)")
            return

        self._status = "running"
        self._consecutive_failures = 0
        self._last_tick_time = None
        self._task = asyncio.create_task(self._run())
        logger.info("CollectorService started (interval={}s)", self._interval_s)

    async def stop(self) -> None:
        """Stop the collection loop gracefully.

        Cancels the asyncio task and waits for it to complete. Safe to
        call even if the service is not running.
        """
        if self._task is None:
            return

        self._status = "stopping"
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self._status = "stopped"
        logger.info("CollectorService stopped")

    # ── Internal: tick loop ──────────────────────────────────────────

    async def _run(self) -> None:
        """Main collection loop — runs until cancelled or degraded."""
        while True:
            tick_start = datetime.now(UTC)
            try:
                await self._tick()
                self._consecutive_failures = 0
            except Exception:
                self._consecutive_failures += 1
                logger.opt(exception=True).warning(
                    "Collector tick failed ({}/{})",
                    self._consecutive_failures,
                    10,
                )
                if self._consecutive_failures >= 10:
                    logger.error(
                        "10 consecutive failures — CollectorService degraded"
                    )
                    self._status = "degraded"
                    break

            # Sleep until the next tick (account for tick duration)
            elapsed = (datetime.now(UTC) - tick_start).total_seconds()
            sleep_time = max(0.0, self._interval_s - elapsed)
            await asyncio.sleep(sleep_time)

    async def _tick(self) -> None:
        """Execute a single collection tick."""
        now = datetime.now(UTC)

        # Duration since last tick (measured, not config-based)
        if self._last_tick_time is not None:
            actual_duration = (now - self._last_tick_time).total_seconds()
        else:
            actual_duration = float(self._interval_s)
        self._last_tick_time = now

        # Collect window and idle info
        snapshot = await self._collector.snapshot()
        idle_secs = await self._collector.idle_seconds()

        is_idle = idle_secs >= self._idle_threshold_s
        event_type: EventType = "idle_change" if is_idle else "window_snapshot"

        event = ActivityEvent(
            id=new_id(),
            user_id=self._user_id,
            timestamp_utc=now,
            duration_s=actual_duration,
            event_type=event_type,
            data=WindowSnapshot(
                app_name=snapshot.app_name,
                window_title=snapshot.window_title,
                process_name=snapshot.process_name,
                is_idle=is_idle,
                timestamp_utc=now,
            ),
        )

        await self._repository.append_event(event)
