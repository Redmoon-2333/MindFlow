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
  - ``stop()`` is graceful — sets a sentinel flag and waits for the
    current tick to complete naturally, with a timeout-based cancel
    fallback. This ensures in-flight events are persisted before
    shutdown (addresses P1-1).
  - Each ``_tick()`` call is wrapped in ``asyncio.wait_for`` so that
    a hung collector never blocks the loop indefinitely (addresses P1-4).
  - ``_state_lock`` (``asyncio.Lock``) makes start/stop lifecycle
    transitions atomic.  Task completion is NEVER awaited under the
    lock — the lock gates only state reads/writes, not I/O-bound
    awaits (addresses P1-4).
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
from mindflow.infrastructure.repositories.base import ActivityRepository

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
        repository: ActivityRepository,
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

        self._state_lock = asyncio.Lock()
        """Serialises start/stop transitions so that concurrent callers
        never create orphan tasks, double-stop the same task, or leave
        ``_task`` / ``_status`` / ``_stop_requested`` inconsistent."""

        self._task: asyncio.Task[None] | None = None
        self._status: str = "stopped"
        self._stop_requested: bool = False
        self._consecutive_failures: int = 0
        self._last_tick_time: datetime | None = None

    @property
    def status(self) -> str:
        """Return current status: stopped, running, stopping, or degraded."""
        return self._status

    async def start(self) -> None:
        """Start the collection loop.

        Idempotent — if a task is already active (or a stop is in
        progress) this is a safe no-op.  The ``_state_lock`` ensures
        that the check-then-create sequence is atomic so concurrent
        callers never spawn duplicate background tasks.
        """
        async with self._state_lock:
            if self._task is not None or self._status == "stopping":
                logger.warning("CollectorService already running (start ignored)")
                return

            self._status = "running"
            self._stop_requested = False
            self._consecutive_failures = 0
            self._last_tick_time = None
            self._task = asyncio.create_task(self._run())

        logger.info("CollectorService started (interval={}s)", self._interval_s)

    async def stop(self) -> None:
        """Stop the collection loop gracefully.

        Sets a sentinel flag so the current tick finishes naturally
        (preserving in-flight events), then waits for the background
        task to exit. If the tick takes longer than ``interval_s * 2``,
        a cancel() fallback kicks in to avoid hanging on shutdown.

        The ``_state_lock`` is held ONLY for the state reads/writes —
        the actual ``await`` of the task happens *outside* the lock so
        that (a) the lock is never held across I/O, and (b) concurrent
        ``stop()`` callers see a consistent ``_status == "stopping"``
        gate and return early.

        Cancellation-safe: if the caller's task is cancelled while
        awaiting the background task, the background task is also
        cancelled and ``_finalize`` runs under ``asyncio.shield`` so
        ``_task`` / ``_status`` are always cleaned up.  CancelledError
        propagates after cleanup — never swallowed.
        """
        task: asyncio.Task[None] | None = None

        async with self._state_lock:
            if self._task is None or self._status == "stopping":
                return
            self._status = "stopping"
            self._stop_requested = True
            task = self._task
            # Keep _task alive so concurrent start() sees it is busy;
            # _finalize clears it after the task completes / is cancelled.

        # ── Await / cancel OUTSIDE the lock ──────────────────────────
        try:
            try:
                await asyncio.wait_for(task, timeout=self._interval_s * 2)
            except asyncio.CancelledError:
                # We were cancelled — cancel the background task too,
                # then re-raise so the caller knows.  The finally block
                # below runs *before* the re-raise.
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                raise
            except TimeoutError:
                logger.warning(
                    "CollectorService stop timeout — cancelling task (interval={}s)",
                    self._interval_s,
                )
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            # Shielded: _finalize always completes even if this
            # coroutine was cancelled — no stale _task or stuck
            # _status="stopping" left behind.
            await asyncio.shield(self._finalize(task))

        logger.info("CollectorService stopped")

    async def _finalize(self, task: asyncio.Task[None]) -> None:
        """Clean up lifecycle state under ``_state_lock``.

        Called from ``stop()`` via ``asyncio.shield`` so that
        cancellation of the caller never skips this step.
        """
        async with self._state_lock:
            if self._task is task:
                self._task = None
            self._status = "stopped"

    # ── Internal: tick loop ──────────────────────────────────────────

    async def _run(self) -> None:
        """Main collection loop — runs until stop_requested or degraded."""
        while not self._stop_requested:
            tick_start = datetime.now(UTC)
            try:
                await asyncio.wait_for(self._tick(), timeout=self._interval_s * 2)
                self._consecutive_failures = 0
            except TimeoutError:
                self._consecutive_failures += 1
                logger.warning(
                    "Collector tick timed out ({}/{})",
                    self._consecutive_failures,
                    10,
                )
                if self._consecutive_failures >= 10:
                    logger.error(
                        "10 consecutive failures — CollectorService degraded"
                    )
                    self._status = "degraded"
                    break
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
            if not self._stop_requested:
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
