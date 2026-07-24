from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from mindflow.domain.events import make_event
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.preferences import PreferencesRepository, user_preferences
from mindflow.infrastructure.repositories.telemetry import (
    TelemetryRepository,
    metadata,
)
from mindflow.services.telemetry_service import TelemetryService


@pytest.fixture
async def telemetry_repo(engine, session_factory):
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    return TelemetryRepository(session_factory)


async def test_interaction_bucket_roundtrip(telemetry_repo: TelemetryRepository) -> None:
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    await telemetry_repo.save_interaction_bucket(
        user_id=1,
        window_start_utc=start,
        duration_s=30.0,
        context_key="code.exe:abc",
        keypress_count=42,
        mouse_click_count=8,
        scroll_delta=3,
        mouse_distance_px=1200.0,
        input_active_s=18.0,
        interaction_burst_count=4,
    )

    status = await telemetry_repo.get_status(1, start.date())

    assert status["interaction_bucket_count"] == 1
    assert status["last_interaction_at"] == start.isoformat()


async def test_browser_heartbeats_merge_by_domain(telemetry_repo: TelemetryRepository) -> None:
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    await telemetry_repo.save_browser_heartbeat(
        user_id=1,
        timestamp_utc=start,
        duration_s=5.0,
        browser_name="edge",
        domain="docs.python.org",
        audible=False,
        context_key="edge:docs.python.org",
    )
    await telemetry_repo.save_browser_heartbeat(
        user_id=1,
        timestamp_utc=start + timedelta(seconds=5),
        duration_s=5.0,
        browser_name="edge",
        domain="docs.python.org",
        audible=False,
        context_key="edge:docs.python.org",
    )
    await telemetry_repo.save_browser_heartbeat(
        user_id=1,
        timestamp_utc=start + timedelta(seconds=10),
        duration_s=5.0,
        browser_name="edge",
        domain="youtube.com",
        audible=True,
        context_key="edge:youtube.com",
    )

    segments = await telemetry_repo.list_browser_segments(1, start, start + timedelta(minutes=1))

    assert len(segments) == 2
    assert segments[0]["duration_s"] == 10.0
    assert segments[1]["domain"] == "youtube.com"


async def test_focus_feedback_roundtrip(telemetry_repo: TelemetryRepository) -> None:
    feedback = await telemetry_repo.save_focus_feedback(
        user_id=1,
        session_id="session-1",
        label="focus",
        score=5,
        task_type="coding",
    )

    stored = await telemetry_repo.list_focus_feedback(1)

    assert feedback["session_id"] == "session-1"
    assert stored[0]["label"] == "focus"
    assert stored[0]["score"] == 5


async def test_delete_scope_only_removes_selected_data(
    telemetry_repo: TelemetryRepository,
) -> None:
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    await telemetry_repo.save_interaction_bucket(
        user_id=1,
        window_start_utc=start,
        duration_s=30.0,
        context_key="code.exe:abc",
        keypress_count=1,
        mouse_click_count=1,
        scroll_delta=0,
        mouse_distance_px=0.0,
        input_active_s=1.0,
        interaction_burst_count=1,
    )
    await telemetry_repo.save_browser_heartbeat(
        user_id=1,
        timestamp_utc=start,
        duration_s=5.0,
        browser_name="edge",
        domain="example.com",
        audible=False,
        context_key="edge:example.com",
    )

    deleted = await telemetry_repo.delete_scope(1, "interaction")
    status = await telemetry_repo.get_status(1, start.date())

    assert deleted == 1
    assert status["interaction_bucket_count"] == 0
    assert status["browser_segment_count"] == 1


async def test_rollup_carries_long_segments_across_five_minute_windows(
    engine, session_factory, tmp_path
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.run_sync(activity_events.metadata.create_all)
        await connection.run_sync(user_preferences.metadata.create_all)
    activity_repository = SQLAlchemyActivityRepository(session_factory)
    telemetry_repository = TelemetryRepository(session_factory)
    service = TelemetryService(
        telemetry_repository,
        PreferencesRepository(session_factory),
        data_dir=tmp_path,
        activity_repository=activity_repository,
    )
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    await activity_repository.append_event(
        make_event(
            user_id=1,
            timestamp_utc=start,
            duration_s=600,
            process_name="code.exe",
            is_idle=False,
        )
    )
    await telemetry_repository.save_browser_heartbeat(
        user_id=1,
        timestamp_utc=start,
        duration_s=600,
        browser_name="edge",
        domain="docs.example.com",
        audible=False,
        context_key="edge:docs.example.com",
    )

    count = await service.rollup_feature_windows(start, start + timedelta(minutes=10))
    windows = await telemetry_repository.list_feature_windows(1)

    assert count == 2
    assert len(windows) == 2
    features = [json.loads(window["features_json"]) for window in windows]
    assert all(window["active_seconds_ratio"] == 1.0 for window in features)
    assert all(window["browser_ratio"] == 1.0 for window in features)


async def test_rollup_includes_segments_started_before_range(
    engine, session_factory, tmp_path
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.run_sync(activity_events.metadata.create_all)
        await connection.run_sync(user_preferences.metadata.create_all)
    activity_repository = SQLAlchemyActivityRepository(session_factory)
    telemetry_repository = TelemetryRepository(session_factory)
    service = TelemetryService(
        telemetry_repository,
        PreferencesRepository(session_factory),
        data_dir=tmp_path,
        activity_repository=activity_repository,
    )
    range_start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    segment_start = range_start - timedelta(minutes=5)
    await activity_repository.append_event(
        make_event(
            user_id=1,
            timestamp_utc=segment_start,
            duration_s=600,
            process_name="code.exe",
            is_idle=False,
        )
    )
    await telemetry_repository.save_browser_heartbeat(
        user_id=1,
        timestamp_utc=segment_start,
        duration_s=600,
        browser_name="edge",
        domain="docs.example.com",
        audible=False,
        context_key="edge:docs.example.com",
    )

    count = await service.rollup_feature_windows(
        range_start,
        range_start + timedelta(minutes=5),
    )
    windows = await telemetry_repository.list_feature_windows(1)

    assert count == 1
    features = json.loads(windows[0]["features_json"])
    assert features["active_seconds_ratio"] == 1.0
    assert features["browser_ratio"] == 1.0


async def test_cleanup_retains_only_recent_feature_windows(
    telemetry_repo: TelemetryRepository,
) -> None:
    now = datetime(2026, 7, 24, 8, tzinfo=UTC)
    await telemetry_repo.save_feature_window(
        user_id=1,
        window_start_utc=now - timedelta(days=181),
        window_end_utc=now - timedelta(days=181, minutes=-5),
        feature_schema_version=2,
        features_json="{}",
    )
    await telemetry_repo.save_feature_window(
        user_id=1,
        window_start_utc=now - timedelta(days=10),
        window_end_utc=now - timedelta(days=10, minutes=-5),
        feature_schema_version=2,
        features_json="{}",
    )

    deleted = await telemetry_repo.cleanup_old_telemetry(
        interaction_cutoff=now - timedelta(days=7),
        activity_cutoff=now - timedelta(days=30),
        feature_cutoff=now - timedelta(days=180),
    )
    windows = await telemetry_repo.list_feature_windows(1)

    assert deleted == 1
    assert len(windows) == 1


async def test_delete_input_scope_removes_derived_feature_windows(
    telemetry_repo: TelemetryRepository,
) -> None:
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    await telemetry_repo.save_interaction_bucket(
        user_id=1,
        window_start_utc=start,
        duration_s=30.0,
        context_key="code.exe:abc",
        keypress_count=1,
        mouse_click_count=1,
        scroll_delta=0,
        mouse_distance_px=0.0,
        input_active_s=1.0,
        interaction_burst_count=1,
    )
    await telemetry_repo.save_feature_window(
        user_id=1,
        window_start_utc=start,
        window_end_utc=start + timedelta(minutes=5),
        feature_schema_version=2,
        features_json="{}",
    )

    deleted = await telemetry_repo.delete_scope(1, "interaction")

    assert deleted == 2
    assert await telemetry_repo.list_feature_windows(1) == []
