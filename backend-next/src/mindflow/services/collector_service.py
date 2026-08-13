"""Background collector service — 5-second tick loop for window tracking.

CollectorService owns the active collection loop: it polls the active
window at a fixed interval, builds ActivityEvents, and persists them
through the ActivityRepository.

Key design decisions (ADR-007, ADR-002):
  - Own asyncio task per instance (not a global singleton).
  - Bare asyncio loop (no APScheduler) — matches the "no framework"
    spirit of the new architecture.
  - One failure does not kill the loop; 10 consecutive failures
    transition to ``degraded``, close the run interval ``failure=True``
    and enter a backoff retry (5/15/30/60s) before the same supervisor
    opens a fresh run cycle (self-healing).
  - ``stop()`` is graceful — sets a sentinel flag and waits for the
    current tick to finish naturally, with a timeout-based cancel
    fallback so in-flight events persist before shutdown (P1-1).
  - Each tick runs under ``asyncio.wait_for`` so a hung collector can
    never block the loop indefinitely (P1-4).
  - ``_state_lock`` (``asyncio.Lock``) makes start/stop transitions
    atomic; task completion is NEVER awaited under the lock — the lock
    gates only state reads/writes, not I/O-bound awaits (P1-4).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger

from mindflow.config import get_settings
from mindflow.infrastructure.collectors.base import EventCollector
from mindflow.infrastructure.repositories.base import ActivityRepository
from mindflow.ports import CollectorIntervalsPort
from mindflow.services.collector_interval_lifecycle import (
    CollectorIntervalLifecycle,
    safe_error_text,
)
from mindflow.services.collector_recovery import RecoveryState
from mindflow.services.collector_ticker import CollectorTicker

_IDLE_THRESHOLD_S: int = 60
"""Seconds of inactivity before marking a snapshot as idle."""


class CollectorService:
    """Background collector service — polls the active window on a tick.

    Not a singleton — each instance manages its own lifecycle; the caller
    holds a reference and calls ``stop()`` on shutdown.  The single
    long-lived ``_task`` supervisor runs repeated run cycles: every 10
    consecutive failures close the current run interval ``failure=True``,
    schedule the exact retry delay on ``RecoveryState``, sleep it out
    inside a ``sleep=True`` backoff interval, then open a fresh run cycle.
    A manual stop — during a run or during backoff — is the only exit.

    Args:
        collector: Platform-specific ``EventCollector``.
        repository: ``ActivityRepository`` for persisting events.
        user_id: User identifier to attach to collected events (default 1).
        interval_s: Tick interval in seconds (defaults to settings).
        idle_threshold_s: Seconds of no input before marking idle
            (default 60).
        interval_repository: Optional ``CollectorIntervalsPort`` — when
            provided, each run and backoff persists one interval row,
            opened/closed from single lifecycle seams.
        sleep: Injectable async sleep (default ``asyncio.sleep``) so
            tests never wait out real backoff.
        now: Injectable UTC clock (default ``datetime.now(UTC)``).
    """

    def __init__(
        self,
        collector: EventCollector,
        repository: ActivityRepository,
        user_id: int = 1,
        interval_s: float | None = None,
        idle_threshold_s: int = _IDLE_THRESHOLD_S,
        interval_repository: CollectorIntervalsPort | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        now: Callable[[], datetime] | None = None,
        idle_collect_interval_s: float | None = None,
    ) -> None:
        self._collector = collector
        self._repository = repository
        self._user_id = user_id
        self._interval_s = (
            interval_s if interval_s is not None else float(get_settings().collect_interval_s)
        )
        self._idle_threshold_s = idle_threshold_s
        # Widened tick gap while the machine is idle (architecture plan H):
        # saves battery and avoids a flood of idle_change rows when nobody
        # is at the keyboard.
        self._idle_collect_interval_s = float(
            idle_collect_interval_s
            if idle_collect_interval_s is not None
            else get_settings().idle_collect_interval_s,
        )
        self._intervals = CollectorIntervalLifecycle(interval_repository, user_id)
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self._now = now if now is not None else lambda: datetime.now(UTC)
        self._recovery = RecoveryState()
        self._ticker = CollectorTicker(
            collector=collector,
            repository=repository,
            user_id=user_id,
            interval_s=self._interval_s,
            idle_threshold_s=idle_threshold_s,
            now=self._now,
        )

        self._state_lock = asyncio.Lock()
        """Serialises start/stop transitions so that concurrent callers
        never create orphan tasks, double-stop the same task, or leave
        ``_task`` / ``_status`` / ``_stop_requested`` inconsistent."""

        self._task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._status: str = "stopped"
        self._stop_requested: bool = False
        self._consecutive_failures: int = 0

    @property
    def status(self) -> str:
        """Return current status: stopped, running, stopping, or degraded."""
        return self._status

    @property
    def next_retry_at(self) -> datetime | None:
        """UTC time of the next permitted retry, or None while healthy."""
        return self._recovery.next_retry_at

    @property
    def last_error(self) -> str | None:
        """Message of the most recent failure, or None after a success."""
        return self._recovery.last_error

    @property
    def recovery_attempts(self) -> int:
        """Consecutive recovery-attempt count (reset to 0 on success)."""
        return self._recovery.recovery_attempts

    async def health_summary(self) -> dict[str, Any]:
        """Aggregate collector health for observability (architecture plan B).

        Combines live service state with the persisted ``collector_intervals``
        audit trail so the UI can show *why* data may be missing (crashes,
        system sleep, recovery backoff) instead of a bare running/stopped
        flag.
        """
        summary: dict[str, Any] = {
            "status": self._status,
            "recovery_attempts": self.recovery_attempts,
            "last_error": self.last_error,
            "next_retry_at": self.next_retry_at.isoformat()
            if self.next_retry_at is not None
            else None,
        }
        # Persisted audit trail (only when an interval repository is wired).
        interval_repo = getattr(self._intervals, "_repository", None)
        if interval_repo is not None:
            try:
                cutoff = self._now() - timedelta(days=7)
                records = await interval_repo.list_by_user_range(
                    self._user_id, cutoff, self._now()
                )
                failures = [
                    r for r in records if r.failure or (r.last_error is not None)
                ]
                summary["failure_count_7d"] = len(failures)
                summary["last_failure_at"] = (
                    failures[-1].ended_at or failures[-1].started_at
                    if failures
                    else None
                )
                summary["last_failure_reason"] = (
                    failures[-1].last_error or failures[-1].reason
                    if failures
                    else None
                )
                sleep_count = sum(1 for r in records if r.sleep)
                summary["sleep_count_7d"] = sleep_count
            except Exception as exc:
                logger.warning(
                    "Collector health_summary interval query failed: {}",
                    safe_error_text(exc),
                )
        return summary

    async def start(self) -> None:
        """Start the collection loop.

        Idempotent — if a task is already active (or a stop is in
        progress) this is a safe no-op.  The ``_state_lock`` ensures
        that the check-then-create sequence is atomic so concurrent
        callers never spawn duplicate background tasks.
        """
        async with self._state_lock:
            if (
                self._task is not None
                or self._cleanup_task is not None
                or self._status == "stopping"
            ):
                logger.warning("CollectorService running or stopping (start ignored)")
                return

            self._status = "running"
            self._stop_requested = False
            self._consecutive_failures = 0
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
        cancelled.  When cancellation cleanup outlives the stop timeout,
        it remains tracked with status ``stopping`` so ``start()`` stays
        a no-op until the old supervisor truly exits.  CancelledError
        still propagates to the caller — never swallowed.
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
                done, _ = await asyncio.wait(
                    {task}, timeout=self._interval_s * 2
                )
                if task in done:
                    await task
                else:
                    logger.warning(
                        "CollectorService stop timeout — cancelling task (interval={}s)",
                        self._interval_s,
                    )
                    task.cancel()
            except asyncio.CancelledError:
                # We were cancelled — cancel the background task too,
                # then give its audit finalizer the same bounded grace
                # period before re-raising to the caller.
                task.cancel()
                await asyncio.wait({task}, timeout=self._interval_s * 2)
                raise
        finally:
            if task.done():
                await asyncio.shield(self._finalize(task))
            else:
                await asyncio.shield(self._track_cleanup(task))

        logger.info(
            "CollectorService {}",
            "stopped" if task.done() else "stop returned (cleanup pending)",
        )

    async def _track_cleanup(self, task: asyncio.Task[None]) -> None:
        """Keep a timed-out supervisor registered until it really exits."""
        async with self._state_lock:
            if self._task is task and self._cleanup_task is None:
                self._cleanup_task = asyncio.create_task(self._finish_cleanup(task))

    async def _finish_cleanup(self, task: asyncio.Task[None]) -> None:
        """Await detached supervisor cleanup, then release lifecycle state."""
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "CollectorService supervisor cleanup failed: {}", safe_error_text(exc)
            )
        finally:
            await self._finalize(task)

    async def _finalize(self, task: asyncio.Task[None]) -> None:
        """Clean up lifecycle state under ``_state_lock``.

        Called from ``stop()`` via ``asyncio.shield`` so that
        cancellation of the caller never skips this step.
        """
        async with self._state_lock:
            if self._task is task:
                self._task = None
            if self._cleanup_task is asyncio.current_task():
                self._cleanup_task = None
            self._status = "stopped"

    # ── Internal: supervisor loop ─────────────────────────────────────

    async def _run(self) -> None:
        """Single long-lived supervisor — run cycles and backoff retries.

        Never dies on degraded exits: after 10 consecutive failures the
        current run interval closes ``failure=True``, ``RecoveryState``
        schedules the exact retry delay, a ``sleep=True`` backoff interval
        spans the wait, then the same supervisor opens a fresh run cycle.
        A manual stop request is the only exit — during a run or during
        backoff it prevents every later retry.
        """
        while not self._stop_requested:
            self._consecutive_failures = 0
            self._ticker.reset()
            interval_id: str | None = None
            degraded = False
            last_error: str | None = None
            try:
                interval_id = await self._intervals.open()

                while not self._stop_requested:
                    tick_start = self._now()
                    last_tick_was_idle = False
                    try:
                        last_tick_was_idle = await asyncio.wait_for(
                            self._ticker.tick(), timeout=self._interval_s * 2
                        )
                        self._consecutive_failures = 0
                        if self._status == "degraded":
                            self._status = "running"
                        self._recovery.record_success()
                    except TimeoutError:
                        last_error = safe_error_text(TimeoutError("collector tick timed out"))
                        if self._degrade_on_failure("timed out"):
                            degraded = True
                            break
                    except Exception as exc:
                        last_error = safe_error_text(exc)
                        if self._degrade_on_failure("failed"):
                            degraded = True
                            break

                    if interval_id is None:
                        interval_id = await self._intervals.open()

                    # Adaptive frequency (architecture plan H): while the
                    # machine is idle, widen the gap to save battery.
                    current_interval = (
                        self._idle_collect_interval_s
                        if last_tick_was_idle
                        else self._interval_s
                    )
                    elapsed = (self._now() - tick_start).total_seconds()
                    sleep_time = max(0.0, current_interval - elapsed)
                    if not self._stop_requested:
                        await self._sleep(sleep_time)
                        # Keep injected no-op sleeps cooperative.  Production
                        # ``asyncio.sleep`` already yields, but deterministic
                        # test clocks and shutdown hooks may return immediately;
                        # an explicit checkpoint prevents a tight supervisor
                        # loop from starving lifecycle callers.
                        await asyncio.sleep(0)
            finally:
                # Single lifecycle seam per run cycle: every exit path
                # (manual stop, degraded failure, cancellation/shutdown)
                # closes the run interval exactly once with truthful facts.
                if interval_id is None:
                    interval_id = await self._intervals.open()
                close_task = asyncio.create_task(
                    self._intervals.close(
                        interval_id,
                        degraded=degraded,
                        stop_requested=self._stop_requested,
                        last_error=last_error,
                    )
                )
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    # ``stop()`` may cancel the supervisor after its short
                    # tick timeout.  Audit persistence must still finish
                    # before cancellation propagates, otherwise an interval
                    # can remain open forever in SQLite.
                    await asyncio.shield(close_task)
                    raise

            if not degraded:
                break  # manual stop / shutdown — no further retries

            # Backoff: schedule the exact delay, then sleep it out inside
            # a sleep=True interval.  The same supervisor retries after.
            delay = self._recovery.record_failure(self._now(), last_error)
            # Publish the degraded state only after recovery metadata is
            # complete, so observers never see a half-updated snapshot.
            self._status = "degraded"
            logger.error("CollectorService degraded — retrying in {}s", delay)
            sleep_id = await self._intervals.open_sleep()
            try:
                await self._sleep(delay)
            finally:
                if sleep_id is None:
                    sleep_id = await self._intervals.open_sleep()
                await self._intervals.close_sleep(sleep_id, stop_requested=self._stop_requested)
            # The injected sleep may complete synchronously; yield before
            # opening the next run cycle so stop/start callers get scheduled.
            await asyncio.sleep(0)

    def _degrade_on_failure(self, kind: str) -> bool:
        """Count one consecutive failure; degrade at the 10th."""
        self._consecutive_failures += 1
        logger.warning("Collector tick {} ({}/{})", kind, self._consecutive_failures, 10)
        if self._consecutive_failures < 10:
            return False
        logger.error("10 consecutive failures — CollectorService degraded")
        return True
