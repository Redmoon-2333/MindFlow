"""Tests for SQLAlchemyActivityRepository — real SQLite, heartbeat merge.

Tests cover:
  - INSERT a new event and read it back
  - Heartbeat merge: same app within pulsetime → UPDATE duration_s
  - Heartbeat merge: same app but outside pulsetime → INSERT new row
  - Heartbeat merge: different app → INSERT new row
  - query_range: inclusive boundaries, empty range, ordering
  - last_event: returns most recent, returns None for empty DB
  - Serialisation roundtrip via to_dict/from_dict
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from mindflow.domain.events import ActivityEvent, WindowSnapshot
from mindflow.domain.ids import new_id
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    metadata,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _utc(iso: str) -> datetime:
    """Parse ISO8601 string and attach UTC timezone."""
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE_TS = _utc("2026-07-17T10:00:00")


def _event(
    app_name: str = "Code",
    window_title: str = "test",
    process_name: str = "code.exe",
    duration_s: float = 5.0,
    user_id: int = 1,
    ts: datetime = _BASE_TS,
    is_idle: bool = False,
    event_type: str = "window_snapshot",
) -> ActivityEvent:
    """Factory helper for test events."""
    return ActivityEvent(
        id=new_id(),
        user_id=user_id,
        timestamp_utc=ts,
        duration_s=duration_s,
        event_type=event_type,
        data=WindowSnapshot(
            app_name=app_name,
            window_title=window_title,
            process_name=process_name,
            is_idle=is_idle,
            timestamp_utc=ts,
        ),
    )


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
async def repo(engine, session_factory):
    """Create a repository with a fully initialised table."""
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    return SQLAlchemyActivityRepository(
        session_factory=session_factory,
        pulsetime_s=10,
    )


# ── Test: INSERT ──────────────────────────────────────────────────────


class TestAppendEvent:
    """Basic INSERT behaviour."""

    async def test_insert_and_read_back(self, repo, engine):
        """A single event is persisted and can be read back."""
        ev = _event()
        await repo.append_event(ev)

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT id, user_id, duration_s FROM activity_events")
            )
            row = result.fetchone()
            assert row is not None
            assert row.id == ev.id
            assert row.user_id == ev.user_id
            assert row.duration_s == ev.duration_s

    async def test_insert_stores_data_json(self, repo, engine):
        """The WindowSnapshot is stored as JSON in data_json."""
        ev = _event(app_name="Chrome")
        await repo.append_event(ev)

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT data_json FROM activity_events WHERE id = :id"),
                {"id": ev.id},
            )
            row = result.fetchone()
            assert row is not None
            data = json.loads(row.data_json)
            assert data["app_name"] == "Chrome"
            assert data["window_title"] == "test"

    async def test_insert_with_idle_event(self, repo, engine):
        """Idle events are inserted normally (no merge with window_snapshot)."""
        ev = _event(is_idle=True, event_type="idle_change")
        await repo.append_event(ev)

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM activity_events")
            )
            assert result.scalar() == 1

    async def test_insert_multiple_events(self, repo, engine):
        """Multiple events are stored as separate rows."""
        for i in range(5):
            await repo.append_event(
                _event(app_name=f"App{i}", ts=_BASE_TS + timedelta(seconds=i))
            )

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM activity_events")
            )
            assert result.scalar() == 5


# ── Test: Heartbeat merge ─────────────────────────────────────────────


class TestHeartbeatMerge:
    """Heartbeat merge behaviour (ADR-002, ADR-007)."""

    async def test_same_app_within_pulsetime_merges(self, repo, engine):
        """Same app, 4s < 10s pulsetime → duration_s is summed, 1 row."""
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS)
        )
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS + timedelta(seconds=4))
        )

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*), sum(duration_s) FROM activity_events")
            )
            row = result.fetchone()
            assert row[0] == 1, "Expected 1 row (merged)"
            assert row[1] == 10.0, "Expected combined duration of 10.0"

    async def test_same_app_outside_pulsetime_inserts(self, repo, engine):
        """Same app but 11s > 10s pulsetime → 2 separate rows."""
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS)
        )
        await repo.append_event(
            _event(
                app_name="Code",
                duration_s=5.0,
                ts=_BASE_TS + timedelta(seconds=11),
            )
        )

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM activity_events")
            )
            assert result.scalar() == 2

    async def test_different_app_does_not_merge(self, repo, engine):
        """Different app_name within pulsetime → 2 separate rows."""
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS)
        )
        await repo.append_event(
            _event(
                app_name="Chrome",
                duration_s=5.0,
                ts=_BASE_TS + timedelta(seconds=4),
            )
        )

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM activity_events")
            )
            assert result.scalar() == 2

    async def test_merge_preserves_original_timestamp(self, repo, engine):
        """After merge, the row retains the original timestamp."""
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS)
        )
        await repo.append_event(
            _event(app_name="Code", duration_s=3.0, ts=_BASE_TS + timedelta(seconds=4))
        )

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT timestamp FROM activity_events")
            )
            row = result.fetchone()
            # The original timestamp is preserved (only duration_s updated)
            assert row.timestamp == _BASE_TS.isoformat()

    async def test_merge_only_last_event(self, repo, engine):
        """Merge checks only the most recent event, not all history."""
        # Three events: Code(5s), Chrome(5s), Code(4s)
        # The 3rd event (Code 4s later) should NOT merge with the first
        # because the immediate predecessor is Chrome (different app)
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS)
        )
        await repo.append_event(
            _event(app_name="Chrome", duration_s=5.0, ts=_BASE_TS + timedelta(seconds=4))
        )
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS + timedelta(seconds=8))
        )

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM activity_events")
            )
            assert result.scalar() == 3

    async def test_idle_event_does_not_block_merge(self, repo, engine):
        """An idle_change between two window_snapshots doesn't prevent merge.

        The merge only considers window_snapshot events, so idle events
        are ignored when looking for the last mergeable predecessor.
        """
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS)
        )
        # Insert an idle event
        idle_ev = _event(
            app_name="Code",
            duration_s=2.0,
            ts=_BASE_TS + timedelta(seconds=3),
            is_idle=True,
            event_type="idle_change",
        )
        await repo.append_event(idle_ev)
        # Another Code event within pulsetime of the FIRST window_snapshot
        await repo.append_event(
            _event(app_name="Code", duration_s=5.0, ts=_BASE_TS + timedelta(seconds=6))
        )

        # Total rows = 2 (idle event + merged Code).
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*) FROM activity_events")
            )
            assert result.scalar() == 2

    async def test_consecutive_merges(self, repo, engine):
        """Multiple consecutive same-app events all merge into one row."""
        base = _BASE_TS
        for i in range(10):
            await repo.append_event(
                _event(app_name="Code", duration_s=5.0, ts=base + timedelta(seconds=i * 2))
            )

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT count(*), sum(duration_s) FROM activity_events")
            )
            row = result.fetchone()
            assert row[0] == 2, (
                "10 events with 2s spacing span 18s > pulsetime 10s, split into 2 rows"
            )
            assert row[1] == 50.0, "Total duration should be 50.0"


# ── Test: query_range ─────────────────────────────────────────────────


class TestQueryRange:
    """Time-range query semantics."""

    async def test_query_returns_events_in_range(self, repo):
        """Events within the time range are returned."""
        ev1 = _event(app_name="A", ts=_BASE_TS)
        ev2 = _event(app_name="B", ts=_BASE_TS + timedelta(seconds=5))
        ev3 = _event(app_name="C", ts=_BASE_TS + timedelta(seconds=10))
        await repo.append_event(ev1)
        await repo.append_event(ev2)
        await repo.append_event(ev3)

        events = await repo.query_range(
            user_id=1,
            start=_BASE_TS,
            end=_BASE_TS + timedelta(seconds=8),
        )
        assert len(events) == 2
        assert events[0].data.app_name == "A"
        assert events[1].data.app_name == "B"

    async def test_query_inclusive_boundaries(self, repo):
        """Boundary timestamps are inclusive."""
        ev = _event(app_name="X", ts=_BASE_TS)
        await repo.append_event(ev)

        events = await repo.query_range(
            user_id=1,
            start=_BASE_TS,
            end=_BASE_TS,
        )
        assert len(events) == 1

    async def test_query_returns_empty_for_no_match(self, repo):
        """Empty range returns an empty list."""
        events = await repo.query_range(
            user_id=1,
            start=_BASE_TS,
            end=_BASE_TS + timedelta(seconds=10),
        )
        assert events == []

    async def test_query_ordered_by_timestamp(self, repo):
        """Events are returned in ascending timestamp order."""
        ev1 = _event(app_name="Z", ts=_BASE_TS)
        ev2 = _event(app_name="A", ts=_BASE_TS + timedelta(seconds=3))
        ev3 = _event(app_name="M", ts=_BASE_TS + timedelta(seconds=1))
        # Insert out of order
        await repo.append_event(ev1)
        await repo.append_event(ev3)
        await repo.append_event(ev2)

        events = await repo.query_range(
            user_id=1,
            start=_BASE_TS,
            end=_BASE_TS + timedelta(seconds=10),
        )
        assert len(events) == 3
        assert events[0].data.app_name == "Z"
        assert events[1].data.app_name == "M"
        assert events[2].data.app_name == "A"

    async def test_query_filters_by_user_id(self, repo):
        """Only events for the requested user_id are returned."""
        await repo.append_event(_event(app_name="User1", ts=_BASE_TS, user_id=1))
        await repo.append_event(_event(app_name="User2", ts=_BASE_TS, user_id=2))

        events = await repo.query_range(
            user_id=1,
            start=_BASE_TS,
            end=_BASE_TS + timedelta(seconds=1),
        )
        assert len(events) == 1
        assert events[0].data.app_name == "User1"


# ── Test: last_event ─────────────────────────────────────────────────


class TestLastEvent:
    """Latest-event query semantics."""

    async def test_last_event_returns_most_recent(self, repo):
        """last_event returns the latest event by timestamp."""
        ev1 = _event(app_name="First", ts=_BASE_TS)
        ev2 = _event(app_name="Last", ts=_BASE_TS + timedelta(seconds=5))
        await repo.append_event(ev1)
        await repo.append_event(ev2)

        last = await repo.last_event(1)
        assert last is not None
        assert last.data.app_name == "Last"
        assert last.id == ev2.id

    async def test_last_event_returns_none_for_empty(self, repo):
        """last_event returns None when no events exist."""
        last = await repo.last_event(1)
        assert last is None

    async def test_last_event_filters_by_user_id(self, repo):
        """last_event only considers the given user_id."""
        await repo.append_event(
            _event(app_name="User2", ts=_BASE_TS, user_id=2)
        )
        await repo.append_event(
            _event(app_name="User1", ts=_BASE_TS + timedelta(seconds=5), user_id=1)
        )

        last = await repo.last_event(2)
        assert last is not None
        assert last.data.app_name == "User2"


# ── Test: Roundtrip ──────────────────────────────────────────────────


class TestRoundtrip:
    """Serialisation roundtrip via domain to_dict/from_dict."""

    async def test_event_roundtrip(self, repo):
        """Event stored and retrieved yields equal domain objects."""
        ev = _event(
            app_name="VS Code",
            process_name="Code.exe",
            duration_s=7.5,
            is_idle=False,
            ts=_BASE_TS,
        )
        await repo.append_event(ev)

        retrieved = await repo.query_range(
            user_id=1,
            start=_BASE_TS - timedelta(seconds=1),
            end=_BASE_TS + timedelta(seconds=1),
        )
        assert len(retrieved) == 1
        assert retrieved[0].id == ev.id
        assert retrieved[0].user_id == ev.user_id
        assert retrieved[0].duration_s == ev.duration_s
        assert retrieved[0].data.app_name == ev.data.app_name
        assert retrieved[0].data.window_title == ev.data.window_title
        assert retrieved[0].data.is_idle == ev.data.is_idle

    async def test_window_snapshot_json_roundtrip(self, repo, engine):
        """WindowSnapshot stored as JSON can be deserialised correctly."""
        ev = _event(app_name="Chrome", window_title="GitHub - Pull Requests")
        await repo.append_event(ev)

        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT data_json FROM activity_events WHERE id = :id"),
                {"id": ev.id},
            )
            row = result.fetchone()
            data = json.loads(row.data_json)
            restored = WindowSnapshot.from_dict(data)

        assert restored.app_name == "Chrome"
        assert restored.window_title == "GitHub - Pull Requests"
        assert restored == ev.data
