"""CollectorService single-tick executor — event build and persistence.

Owns the per-tick work that would otherwise bloat
:class:`mindflow.services.collector_service.CollectorService`: polling the
collector, measuring the duration since the previous tick, building the
``ActivityEvent`` and persisting it.  Keeping it in a focused collaborator
lets the service stay a single-responsibility supervisor loop (recovery
state + interval lifecycle) while every clock read still goes through the
injected ``now`` seam.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from mindflow.domain.events import ActivityEvent, EventType, WindowSnapshot
from mindflow.domain.ids import new_id
from mindflow.infrastructure.collectors.base import EventCollector
from mindflow.infrastructure.repositories.base import ActivityRepository


class CollectorTicker:
    """Execute one collector tick against injected dependencies.

    Args:
        collector: Platform-specific ``EventCollector``.
        repository: ``ActivityRepository`` for persisting events.
        user_id: User identifier to attach to collected events.
        interval_s: Tick interval in seconds — the first tick's measured
            duration falls back to this.
        idle_threshold_s: Seconds of no input before marking idle.
        now: Injectable UTC clock; never read from the wall clock.
    """

    def __init__(
        self,
        collector: EventCollector,
        repository: ActivityRepository,
        user_id: int,
        interval_s: float,
        idle_threshold_s: int,
        now: Callable[[], datetime],
    ) -> None:
        self._collector = collector
        self._repository = repository
        self._user_id = user_id
        self._interval_s = interval_s
        self._idle_threshold_s = idle_threshold_s
        self._now = now
        self._last_tick_time: datetime | None = None

    def reset(self) -> None:
        """Start a fresh run cycle without carrying prior backoff time."""
        self._last_tick_time = None

    async def tick(self) -> None:
        """Execute a single collection tick."""
        now = self._now()

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
