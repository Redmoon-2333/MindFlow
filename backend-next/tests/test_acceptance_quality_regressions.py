"""Acceptance regressions for window coverage and collector provenance."""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from mindflow.domain.events import ActivityEvent, WindowSnapshot, make_event
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.collector_intervals import (
    CollectorIntervalsRepository,
)
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import metadata
from mindflow.ports import CollectorIntervalRecord
from mindflow.services.telemetry_service import TelemetryService
from mindflow.services.window_quality import build_window_quality

START = datetime(2026, 9, 19, 10, tzinfo=UTC)
END = START + timedelta(minutes=5)


def test_coverage_clips_and_unions_activity_intervals() -> None:
    quality = build_window_quality(
        window_start=START,
        window_end=END,
        events=[
            SimpleNamespace(timestamp_utc=START - timedelta(seconds=290), duration_s=300),
            SimpleNamespace(timestamp_utc=START + timedelta(seconds=5), duration_s=15),
            SimpleNamespace(timestamp_utc=START + timedelta(seconds=15), duration_s=10),
        ],
        interaction_buckets=[],
        browser_segments=[],
    )
    assert quality.observed_seconds["activity"] == 25
    assert quality.coverage["activity"] == pytest.approx(25 / 300)
    assert quality.uncovered_seconds["activity"] == 275
    assert quality.gap_seconds["activity"] is None


def test_input_and_browser_coverage_use_real_bounds_without_double_counting() -> None:
    bucket = {
        "window_start_utc": (START - timedelta(seconds=10)).isoformat(),
        "window_end_utc": (START + timedelta(seconds=10)).isoformat(),
        "keypress_count": 0,
    }
    segment = {"timestamp": START.isoformat(), "duration_s": 10}
    quality = build_window_quality(
        window_start=START, window_end=END, events=[],
        interaction_buckets=[bucket, bucket], browser_segments=[segment, segment],
    )
    assert quality.observed_seconds["input"] == 10
    assert quality.observed_seconds["browser"] == 10
    assert quality.feature_observed("keypress_rate_per_min")
    assert quality.quality_tier != "full"


def test_missing_collector_state_stays_unknown() -> None:
    quality = build_window_quality(
        window_start=START, window_end=END, events=[],
        interaction_buckets=[], browser_segments=[],
    )
    assert quality.sources_enabled["browser"] is None
    assert quality.sources_available["input"] is None
    assert not quality.feature_observed("browser_ratio")
    assert quality.gap_seconds["browser"] is None


def test_malformed_or_unlocated_payload_does_not_claim_coverage() -> None:
    quality = build_window_quality(
        window_start=START, window_end=END,
        events=[SimpleNamespace(duration_s=300)],
        interaction_buckets=[{"duration_s": 300}],
        browser_segments=[
            {"timestamp": "bad", "duration_s": 300},
            {"timestamp": START.isoformat(), "duration_s": float("inf")},
        ],
    )
    assert all(value == 0 for value in quality.observed_seconds.values())
    assert quality.quality_tier == "empty"


async def test_overlap_query_includes_running_interval_and_excludes_touching(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    repository = CollectorIntervalsRepository(session_factory)
    old = await repository.open(1, now=START - timedelta(hours=1))
    ended = await repository.open(1, now=START - timedelta(hours=2))
    await repository.close(ended.id, now=START)
    await repository.open(2, now=START)
    await repository.open(1, now=END)
    rows = await repository.list_overlapping_range(1, START, END)
    assert [row.id for row in rows] == [old.id]
    tz = timezone(timedelta(hours=8))
    assert await repository.list_overlapping_range(
        1, START.astimezone(tz), END.astimezone(tz),
    ) == rows


async def test_collector_lookup_failure_is_unknown(tmp_path: Path) -> None:
    interval_repo = SimpleNamespace(
        list_overlapping_range=AsyncMock(side_effect=OSError("offline")),
    )
    service = TelemetryService(
        repository=AsyncMock(), preferences_repository=AsyncMock(),
        data_dir=tmp_path, interval_repository=interval_repo,
    )
    assert await service._collector_intervals_for_range(START, END, 1) is None


def test_state_is_window_local_and_does_not_invent_auxiliary_switches() -> None:
    interval = CollectorIntervalRecord(
        id="interval", user_id=1, manual_stop=False, sleep=False,
        started_at=START.isoformat(), ended_at=END.isoformat(),
        failure=False, last_error=None, reason=None,
    )
    first = TelemetryService._collector_state_for_window(START, END, [interval])
    later = TelemetryService._collector_state_for_window(
        END, END + timedelta(minutes=5), [interval],
    )
    assert first["enabled"]["activity"] is True
    assert later["enabled"]["activity"] is None
    assert first["enabled"]["input"] is None
    assert first["enabled"]["browser"] is None
    assert first["available"]["activity"] is None


async def test_rollup_persists_clipped_quality_without_inventing_switches(tmp_path: Path) -> None:
    previous_start = START - timedelta(seconds=290)
    event = ActivityEvent(
        id="audit-event", user_id=1, timestamp_utc=previous_start,
        duration_s=300, event_type="window_snapshot",
        data=WindowSnapshot(
            app_name="editor", window_title="", process_name="editor.exe",
            is_idle=False, timestamp_utc=previous_start,
        ),
    )
    activity = AsyncMock()
    activity.query_range.return_value = []
    activity.last_event_before.return_value = event
    repository = AsyncMock()
    repository.list_interaction_buckets.return_value = []
    repository.list_browser_segments.return_value = []
    repository.last_browser_segment_before.return_value = None
    intervals = SimpleNamespace(list_overlapping_range=AsyncMock(return_value=[]))
    service = TelemetryService(
        repository=repository, preferences_repository=AsyncMock(),
        activity_repository=activity, data_dir=tmp_path,
        interval_repository=intervals,
    )
    assert await service.rollup_feature_windows(START, END) == 1
    rows = repository.upsert_feature_windows.call_args.args[0]
    quality = json.loads(rows[0]["quality_json"])
    assert quality["observed_seconds"]["activity"] == 10
    assert quality["sources_enabled"]["browser"] is None
    assert quality["sources_available"]["input"] is None
    assert quality["quality_tier"] == "activity_only"


async def test_real_rollup_carries_bucket_coverage_across_boundary(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession], tmp_path: Path,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    repository = TelemetryRepository(session_factory)
    await repository.save_interaction_bucket(
        user_id=1, window_start_utc=END - timedelta(seconds=10), duration_s=30,
        context_key="editor", keypress_count=0, mouse_click_count=0, scroll_delta=0,
        mouse_distance_px=0, input_active_s=0, interaction_burst_count=0,
    )
    service = TelemetryService(
        repository, PreferencesRepository(session_factory), tmp_path,
        activity_repository=SQLAlchemyActivityRepository(session_factory),
    )
    assert await service.rollup_feature_windows(START, END + timedelta(minutes=5)) == 2
    rows = await repository.list_feature_windows(1, feature_schema_version=3)
    rows.sort(key=lambda row: row["window_start_utc"])
    assert [
        json.loads(row["quality_json"])["observed_seconds"]["input"] for row in rows
    ] == [10, 20]
    assert await service.rollup_feature_windows(END, END + timedelta(minutes=5)) == 1
    rows = await repository.list_feature_windows(1, feature_schema_version=3)
    rows.sort(key=lambda row: row["window_start_utc"])
    assert json.loads(rows[-1]["quality_json"])["observed_seconds"]["input"] == 20


async def test_non_aligned_reroll_preserves_complete_window(
    engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession], tmp_path: Path,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    activity = SQLAlchemyActivityRepository(session_factory)
    repository = TelemetryRepository(session_factory)
    for i in range(10):
        await activity.append_event(make_event(
            user_id=1, timestamp_utc=START + timedelta(seconds=30 * i),
            duration_s=30, process_name="editor.exe",
        ))
    service = TelemetryService(
        repository, PreferencesRepository(session_factory), tmp_path, activity_repository=activity,
    )
    assert await service.rollup_feature_windows(START, END) == 1
    assert await service.rollup_feature_windows(START + timedelta(minutes=2), END) == 1
    rows = await repository.list_feature_windows(1, feature_schema_version=3)
    assert len(rows) == 1
    assert json.loads(rows[0]["quality_json"])["observed_seconds"]["activity"] == 300


def test_disabled_collection_does_not_report_downtime() -> None:
    quality = build_window_quality(
        window_start=START, window_end=END, events=[],
        interaction_buckets=[], browser_segments=[],
        enabled={"browser": False}, available={"browser": False},
    )
    assert quality.gap_seconds["browser"] == 0
    assert quality.uncovered_seconds["browser"] == 300
