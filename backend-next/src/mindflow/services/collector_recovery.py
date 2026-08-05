"""Collector recovery state and backoff — pure, clock-injected retry schedule.

Exact retry sequence for collector failures (1-based attempt -> delay in
seconds)::

    1 -> 5, 2 -> 15, 3 -> 30, 4+ -> 60    (capped at 60s)

``RecoveryState`` is a small mutable state object tracking the next retry
time, the last error message, and the consecutive recovery-attempt count.
All transitions are deterministic and take an explicit ``now`` argument —
there is no wall-clock dependence, so tests are reproducible.  Production
callers inject ``datetime.now(timezone.utc)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

#: Retry delay schedule in seconds, indexed by 1-based attempt number.
#: The last entry is the cap — attempts 4+ all wait 60s.
RETRY_DELAY_SECONDS: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0)

#: Upper bound for any single retry delay (seconds).
MAX_RETRY_DELAY_SECONDS: float = RETRY_DELAY_SECONDS[-1]


def retry_delay_seconds(attempt: int) -> float:
    """Return the backoff delay for a 1-based *attempt*.

    Attempts below 1 clamp to the first delay; attempts beyond the
    schedule length clamp to the 60s cap.
    """
    idx = min(max(attempt, 1), len(RETRY_DELAY_SECONDS)) - 1
    return RETRY_DELAY_SECONDS[idx]


@dataclass
class RecoveryState:
    """Mutable recovery state for a collector.

    Attributes:
        next_retry_at: UTC datetime of the next permitted retry, or None
            while no failure has been recorded.
        last_error: Message of the most recent failure, or None after a
            successful recovery.
        recovery_attempts: Consecutive failure count (reset to 0 on success).
    """

    next_retry_at: datetime | None = None
    last_error: str | None = None
    recovery_attempts: int = 0

    def record_failure(self, now: datetime, error: str) -> float:
        """Record a failure and advance the backoff schedule.

        Args:
            now: Current UTC time (injected; never read from the clock).
            error: Human-readable error message retained as ``last_error``.

        Returns:
            The delay in seconds before the next retry.
        """
        self.recovery_attempts += 1
        delay = retry_delay_seconds(self.recovery_attempts)
        self.next_retry_at = now + timedelta(seconds=delay)
        self.last_error = error
        return delay

    def record_success(self) -> None:
        """Reset recovery state after a successful run.

        Returns attempts, next retry time and last error to their idle
        defaults so the next failure restarts at the 5s delay.
        """
        self.recovery_attempts = 0
        self.next_retry_at = None
        self.last_error = None
