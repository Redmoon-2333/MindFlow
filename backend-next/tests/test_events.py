"""Tests for mindflow.domain.events — ActivityEvent + WindowSnapshot.

Tests cover:
  - Serialisation roundtrip (to_dict -> from_dict -> equality)
  - Frozen dataclass immutability (FrozenInstanceError)
  - Invalid event_type rejection
  - Naive datetime rejection (timezone-aware requirement)
  - make_event factory helper
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from mindflow.domain.events import (
    ActivityEvent,
    EventType,
    WindowSnapshot,
    make_event,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _utc(iso: str) -> datetime:
    """Parse an ISO string and attach UTC timezone."""
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_TS = _utc("2026-07-17T10:00:00")
_TS_NAIVE = datetime(2026, 7, 17, 10, 0, 0)


def _snapshot(
    app_name: str = "Code",
    window_title: str = "main.py",
    process_name: str = "code.exe",
    is_idle: bool = False,
    ts: datetime = _TS,
) -> WindowSnapshot:
    return WindowSnapshot(
        app_name=app_name,
        window_title=window_title,
        process_name=process_name,
        is_idle=is_idle,
        timestamp_utc=ts,
    )


def _event(
    id_str: str | None = None,
    user_id: int = 1,
    ts: datetime = _TS,
    duration_s: float = 5.0,
    event_type: EventType = "window_snapshot",
    snapshot: WindowSnapshot | None = None,
) -> ActivityEvent:
    return ActivityEvent(
        id=id_str or "018f3a6b-7c8d-7e9f-a012-3456789abcde",
        user_id=user_id,
        timestamp_utc=ts,
        duration_s=duration_s,
        event_type=event_type,
        data=snapshot or _snapshot(),
    )


# ── WindowSnapshot ───────────────────────────────────────────────────────────


class TestWindowSnapshot:
    """Frozen dataclass + serialisation."""

    def test_immutable(self):
        """WindowSnapshot is frozen — attribute assignment raises."""
        snap = _snapshot()
        with pytest.raises(FrozenInstanceError):
            snap.app_name = "Other"  # type: ignore[misc]

    def test_to_dict_roundtrip(self):
        """to_dict -> from_dict yields an equal instance."""
        snap = _snapshot()
        d = snap.to_dict()
        restored = WindowSnapshot.from_dict(d)
        assert restored == snap
        assert restored is not snap

    def test_to_dict_datetime_is_string(self):
        """timestamp_utc is serialised to an ISO8601 string."""
        snap = _snapshot()
        d = snap.to_dict()
        assert isinstance(d["timestamp_utc"], str)

    def test_from_dict_accepts_datetime_instance(self):
        """from_dict can receive an already-parsed datetime."""
        snap = _snapshot()
        d = snap.to_dict()
        d["timestamp_utc"] = _TS  # already a datetime
        restored = WindowSnapshot.from_dict(d)
        assert restored == snap

    def test_naive_datetime_rejected(self):
        """WindowSnapshot raises ValueError for naive datetime."""
        with pytest.raises(ValueError, match="timezone-aware"):
            WindowSnapshot(
                app_name="X",
                window_title="Y",
                process_name="Z",
                is_idle=False,
                timestamp_utc=_TS_NAIVE,
            )


# ── ActivityEvent ────────────────────────────────────────────────────────────


class TestActivityEvent:
    """Frozen dataclass + serialisation + validation."""

    def test_immutable(self):
        """ActivityEvent is frozen — attribute assignment raises."""
        ev = _event()
        with pytest.raises(FrozenInstanceError):
            ev.id = "other"  # type: ignore[misc]

    def test_to_dict_roundtrip(self):
        """to_dict -> from_dict yields an equal event."""
        ev = _event()
        d = ev.to_dict()
        restored = ActivityEvent.from_dict(d)
        assert restored == ev
        assert restored is not ev

    def test_to_dict_nested_data(self):
        """data dict contains nested WindowSnapshot fields."""
        ev = _event()
        d = ev.to_dict()
        assert isinstance(d["data"], dict)
        assert d["data"]["app_name"] == "Code"
        assert d["data"]["timestamp_utc"] == _TS.isoformat()

    def test_from_dict_accepts_window_snapshot_instance(self):
        """from_dict works when 'data' is already a WindowSnapshot."""
        ev = _event()
        d = ev.to_dict()
        d["data"] = _snapshot()  # already a WindowSnapshot
        restored = ActivityEvent.from_dict(d)
        assert restored == ev

    def test_from_dict_rejects_bad_data_type(self):
        """from_dict raises TypeError when 'data' is neither dict nor WindowSnapshot."""
        with pytest.raises(TypeError, match="dict or WindowSnapshot"):
            ActivityEvent.from_dict(
                {
                    "id": "x",
                    "user_id": 1,
                    "timestamp_utc": _TS.isoformat(),
                    "duration_s": 5.0,
                    "event_type": "window_snapshot",
                    "data": "not-a-snapshot",
                }
            )

    def test_invalid_event_type(self):
        """Constructing with an invalid event_type raises ValueError."""
        with pytest.raises(ValueError, match="event_type"):
            _event(event_type="invalid_type")  # type: ignore[arg-type]

    def test_all_valid_event_types(self):
        """Every EventType literal is accepted."""
        for et in ("window_snapshot", "idle_change", "manual_tag"):
            ev = _event(event_type=et)  # type: ignore[arg-type]
            assert ev.event_type == et

    def test_naive_datetime_rejected(self):
        """ActivityEvent raises ValueError for naive datetime."""
        with pytest.raises(ValueError, match="timezone-aware"):
            _event(ts=_TS_NAIVE)

    def test_id_preserved(self):
        """String id is stored and retrieved as-is."""
        uid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        ev = _event(id_str=uid)
        assert ev.id == uid

    def test_user_id_int_coercion(self):
        """from_dict coerces user_id from any compatible type."""
        d = _event().to_dict()
        d["user_id"] = "42"  # string
        restored = ActivityEvent.from_dict(d)
        assert restored.user_id == 42

    def test_duration_s_float_coercion(self):
        """from_dict coerces duration_s from any compatible type."""
        d = _event().to_dict()
        d["duration_s"] = "3.5"  # string
        restored = ActivityEvent.from_dict(d)
        assert restored.duration_s == 3.5


# ── make_event factory ──────────────────────────────────────────────────────


class TestMakeEvent:
    """Convenience factory produces valid ActivityEvents."""

    def test_basic(self):
        """make_event returns a valid ActivityEvent with flattened args."""
        ev = make_event(
            user_id=42,
            timestamp_utc=_TS,
            duration_s=5.0,
            event_type="window_snapshot",
            app_name="Chrome",
            window_title="GitHub",
            process_name="chrome.exe",
            is_idle=False,
        )
        assert ev.user_id == 42
        assert ev.duration_s == 5.0
        assert ev.event_type == "window_snapshot"
        assert ev.data.app_name == "Chrome"
        assert ev.data.window_title == "GitHub"
        assert ev.data.process_name == "chrome.exe"
        assert ev.data.is_idle is False
        assert isinstance(ev.id, str)
        assert len(ev.id) == 36  # UUID

    def test_defaults(self):
        """make_event supplies sensible defaults for optional args."""
        ev = make_event(user_id=1, timestamp_utc=_TS)
        assert ev.duration_s == 0.0
        assert ev.event_type == "window_snapshot"
        assert ev.data.app_name == ""
        assert ev.data.is_idle is False

    def test_generates_unique_ids(self):
        """Each call to make_event generates a unique UUIDv7."""
        ids = {make_event(user_id=1, timestamp_utc=_TS).id for _ in range(10)}
        assert len(ids) == 10
