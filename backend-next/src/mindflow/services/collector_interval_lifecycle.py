"""Collector interval lifecycle — open / classify / close bookkeeping.

Owns the persistence seam for one collector run's interval row: opens
exactly one row per run, classifies the terminal exit (degraded failure,
manual stop, or service shutdown), and closes the row exactly once with
truthful terminal facts.  Kept out of :class:`CollectorService` so the
service stays focused on the tick loop while the interval wiring stays in
one collaborator.

Behavior contract (unchanged from the original seam):
  - Unwired (``repository is None``) — every method is a no-op; the
    service behaves exactly as before interval wiring.
  - ``open()`` returns the interval id, ``None`` when unwired.
  - ``close()`` is called from a single lifecycle seam (the service's
    ``finally``) so every exit path — manual stop, degraded exit, and
    cancellation/shutdown — persists terminal facts exactly once.
  - Terminal classification priority: a degraded failure is recorded even
    when a manual stop was also requested; otherwise a requested stop is
    a manual stop; any other exit is a service shutdown.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from mindflow.ports import CollectorIntervalsPort

_MAX_ERROR_LEN: int = 200
"""Cap for the persisted ``last_error`` string — never store full traces."""


def safe_error_text(exc: Exception) -> str:
    """Return a short, sanitised error summary for interval persistence.

    Records the exception class and its message only — never event payload
    or window content.  Truncated to ``_MAX_ERROR_LEN`` characters so a
    verbose exception cannot bloat the interval row.
    """
    return f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_LEN]


class CollectorIntervalLifecycle:
    """Bookkeeping for one run's collector-interval row.

    Args:
        repository: Optional ``CollectorIntervalsPort`` — when ``None``
            every method is a no-op (un-wired behaviour unchanged).
        user_id: User identifier attached to the interval row.
    """

    def __init__(
        self,
        repository: CollectorIntervalsPort | None,
        user_id: int,
    ) -> None:
        self._repository = repository
        self._user_id = user_id

    async def open(self) -> str | None:
        """Open exactly one interval row for the run; ``None`` when unwired."""
        if self._repository is None:
            return None
        try:
            record = await self._repository.open(
                self._user_id, reason="collector run started"
            )
            return record.id
        except Exception as exc:
            logger.warning("Collector interval open audit failed: {}", safe_error_text(exc))
            return None

    async def open_sleep(self) -> str | None:
        """Open a sleep/backoff interval row; ``None`` when unwired."""
        if self._repository is None:
            return None
        try:
            record = await self._repository.open(
                self._user_id,
                reason="collector backoff (system sleep)",
                sleep=True,
            )
            return record.id
        except Exception as exc:
            logger.warning(
                "Collector sleep interval open audit failed: {}", safe_error_text(exc)
            )
            return None

    async def close(
        self,
        interval_id: str | None,
        *,
        degraded: bool,
        stop_requested: bool,
        last_error: str | None,
    ) -> None:
        """Close the interval exactly once with truthful terminal facts.

        The repository's guarded close makes a second close a no-op
        rewrite — terminal facts are never duplicated.
        """
        if interval_id is None or self._repository is None:
            return
        for attempt in range(2):
            try:
                if degraded:
                    await self._repository.close(
                        interval_id,
                        reason="collector degraded (10 consecutive failures)",
                        manual_stop=False,
                        failure=True,
                        last_error=last_error,
                    )
                elif stop_requested:
                    await self._repository.close(
                        interval_id,
                        reason="manual stop",
                        manual_stop=True,
                        failure=False,
                    )
                else:
                    await self._repository.close(
                        interval_id,
                        reason="service shutdown",
                        manual_stop=False,
                        failure=False,
                    )
                return
            except Exception as exc:
                logger.warning(
                    "Collector interval close audit failed: {}", safe_error_text(exc)
                )
                if attempt == 0:
                    await asyncio.sleep(0)

    async def close_sleep(
        self,
        interval_id: str | None,
        *,
        stop_requested: bool,
    ) -> None:
        """Close a sleep interval exactly once, keeping ``sleep=True``.

        ``manual_stop`` is recorded only when a stop was actually
        requested; any other exit is recorded as a system-sleep close.
        The repository's guarded close makes a second close a no-op
        rewrite — the sleep interval's terminal facts are never
        duplicated.
        """
        if interval_id is None or self._repository is None:
            return
        for attempt in range(2):
            try:
                await self._repository.close(
                    interval_id,
                    reason="manual stop" if stop_requested else "system sleep",
                    manual_stop=stop_requested,
                    failure=False,
                    sleep=True,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Collector sleep interval close audit failed: {}",
                    safe_error_text(exc),
                )
                if attempt == 0:
                    await asyncio.sleep(0)
