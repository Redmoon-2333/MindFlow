"""Task domain models for the smart-prioritization intervention.

Pure data, zero framework dependencies — matching domain/events.py and
domain/intervention.py.  Tasks are the first real data source for the
``smart_prioritization`` intervention: instead of a static suggestion,
the intervention service can now rank the user's actual pending tasks
by deadline proximity and priority.

Design constraints:
  - Frozen dataclass (no pydantic, no SQLAlchemy, no I/O).
  - Timestamps are timezone-aware UTC datetimes (domain convention).
  - ``deadline_utc`` is optional: tasks without a deadline still rank,
    just below deadline-bearing ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

TaskStatus = Literal["pending", "in_progress", "done"]

# Priority scale: 1 (lowest) .. 5 (highest).  Integer instead of an enum
# so the API can expose simple number semantics to the frontend.
MIN_PRIORITY: int = 1
MAX_PRIORITY: int = 5


@dataclass(frozen=True)
class Task:
    """One user task tracked for the smart-prioritization intervention.

    Attributes:
        id: Stable unique identifier.
        user_id: Owning user (single-user mode defaults to 1).
        title: Short task title (required).
        description: Optional longer description.
        priority: 1..5 urgency weight.
        status: ``pending`` / ``in_progress`` / ``done``.
        deadline_utc: Optional UTC deadline.
        estimated_minutes: Optional effort estimate.
        created_at: Row creation time (UTC).
        updated_at: Last update time (UTC).
    """

    id: str
    user_id: int
    title: str
    description: str
    priority: int
    status: TaskStatus
    deadline_utc: datetime | None
    estimated_minutes: int | None
    created_at: datetime
    updated_at: datetime
