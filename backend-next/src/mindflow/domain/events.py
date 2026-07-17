"""Domain models for activity event sourcing.

WindowSnapshot represents a point-in-time observation of the active window.
ActivityEvent wraps a snapshot with metadata and is the fundamental unit of
the append-mostly event stream.

Design decisions:
  - Frozen dataclasses enforce immutability at runtime (ADR-009).
  - to_dict / from_dict provide JSON-safe serialization (ISO8601 for datetimes).
  - Naive datetimes are rejected — all timestamps must be timezone-aware UTC
    (overturns old code's naive-UTC policy, see architecture doc §2.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from mindflow.domain.ids import new_id

EventType = Literal["window_snapshot", "idle_change", "manual_tag"]

_VALID_EVENT_TYPES: frozenset[str] = frozenset({"window_snapshot", "idle_change", "manual_tag"})


def _check_aware(dt: datetime, field_name: str) -> None:
    """Validate that a datetime is timezone-aware (not naive)."""
    if dt.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware (got naive datetime)")


# ── WindowSnapshot ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WindowSnapshot:
    """A point-in-time observation of the active desktop window.

    Attributes:
        app_name: Display name of the active application.
        window_title: Title text of the active window.
        process_name: Executable/process name.
        is_idle: Whether the user was idle at this snapshot.
        timestamp_utc: When the snapshot was taken (timezone-aware UTC).
    """

    app_name: str
    window_title: str
    process_name: str
    is_idle: bool
    timestamp_utc: datetime

    def __post_init__(self) -> None:
        _check_aware(self.timestamp_utc, "WindowSnapshot.timestamp_utc")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (datetime -> ISO8601 string)."""
        return {
            "app_name": self.app_name,
            "window_title": self.window_title,
            "process_name": self.process_name,
            "is_idle": self.is_idle,
            "timestamp_utc": self.timestamp_utc.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowSnapshot:
        """Deserialize from a dict (ISO8601 string -> datetime).

        Args:
            data: A dict with keys matching the constructor,
                  where ``timestamp_utc`` may be an ISO8601 string.
        """
        ts = data["timestamp_utc"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            app_name=data["app_name"],
            window_title=data["window_title"],
            process_name=data["process_name"],
            is_idle=bool(data["is_idle"]),
            timestamp_utc=ts,
        )


# ── ActivityEvent ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActivityEvent:
    """An immutable activity event in the append-mostly event stream.

    Attributes:
        id: UUIDv7 string (time-sortable).
        user_id: User identifier.
        timestamp_utc: When this event occurred (timezone-aware UTC).
        duration_s: Estimated duration (seconds) since the previous event.
        event_type: Categorisation of the event.
        data: The event payload — the window observation.
    """

    id: str
    user_id: int
    timestamp_utc: datetime
    duration_s: float
    event_type: EventType
    data: WindowSnapshot

    def __post_init__(self) -> None:
        _check_aware(self.timestamp_utc, "ActivityEvent.timestamp_utc")
        if self.event_type not in _VALID_EVENT_TYPES:
            valid = ", ".join(sorted(_VALID_EVENT_TYPES))
            raise ValueError(f"Invalid event_type: {self.event_type!r}. Must be one of: {valid}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for storage or transport."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "duration_s": self.duration_s,
            "event_type": self.event_type,
            "data": self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActivityEvent:
        """Deserialize from a dict (ISO8601 string -> datetime).

        Args:
            data: A dict with keys matching the constructor.
                  Nested ``data`` may be a dict or existing WindowSnapshot.
        """
        ts = data["timestamp_utc"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        raw_data = data["data"]
        if isinstance(raw_data, dict):
            snapshot = WindowSnapshot.from_dict(raw_data)
        elif isinstance(raw_data, WindowSnapshot):
            snapshot = raw_data
        else:
            raise TypeError(
                f"Expected dict or WindowSnapshot for 'data', got {type(raw_data).__name__}"
            )

        return cls(
            id=str(data["id"]),
            user_id=int(data["user_id"]),
            timestamp_utc=ts,
            duration_s=float(data["duration_s"]),
            event_type=data["event_type"],
            data=snapshot,
        )


# ── Factory helper ──────────────────────────────────────────────────────────


def make_event(
    *,
    user_id: int,
    timestamp_utc: datetime,
    duration_s: float = 0.0,
    event_type: EventType = "window_snapshot",
    app_name: str = "",
    window_title: str = "",
    process_name: str = "",
    is_idle: bool = False,
) -> ActivityEvent:
    """Convenience factory: creates an ActivityEvent with a WindowSnapshot.

    All arguments after ``duration_s`` are flattened — no need to construct a
    WindowSnapshot manually when writing tests or quick inline code.
    """
    snapshot = WindowSnapshot(
        app_name=app_name,
        window_title=window_title,
        process_name=process_name,
        is_idle=is_idle,
        timestamp_utc=timestamp_utc,
    )
    return ActivityEvent(
        id=new_id(),
        user_id=user_id,
        timestamp_utc=timestamp_utc,
        duration_s=duration_s,
        event_type=event_type,
        data=snapshot,
    )
