"""Protocol definitions for repository interfaces.

Defines the ``ActivityRepository`` protocol used throughout the application.
Concrete implementations (``SQLAlchemyActivityRepository``) satisfy this
protocol structurally — no explicit subclassing required.

Design decisions:
  - Protocol (not ABC) for structural typing — mypy --strict catches
    missing methods at compile time without requiring explicit subclassing.
  - ``@runtime_checkable`` enables ``isinstance`` checks and
    ``unittest.mock.MagicMock(spec=...)`` in tests.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from mindflow.domain.events import ActivityEvent


@runtime_checkable
class ActivityRepository(Protocol):
    """Protocol for activity event persistence.

    Implementations manage the append-mostly event stream with heartbeat
    merge semantics (ADR-002, ADR-007).
    """

    async def append_event(self, event: ActivityEvent) -> None:
        """Persist an activity event (with heartbeat merge)."""
        ...

    async def query_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
    ) -> list[ActivityEvent]:
        """Return events for *user_id* in [*start*, *end*], ordered by time.

        Args:
            user_id: User identifier.
            start: Inclusive start of the time range (timezone-aware UTC).
            end: Inclusive end of the time range (timezone-aware UTC).

        Returns:
            A list of ActivityEvents sorted by timestamp ascending.
        """
        ...

    async def last_event(self, user_id: int) -> ActivityEvent | None:
        """Return the most recent event for *user_id*, or None.

        Args:
            user_id: User identifier.

        Returns:
            The latest ActivityEvent by timestamp, or None if no events exist.
        """
        ...
