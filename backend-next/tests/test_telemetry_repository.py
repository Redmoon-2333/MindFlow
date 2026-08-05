from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.events import make_event
from mindflow.domain.feature_schema import V2_FEATURE_NAMES
from mindflow.domain.ids import new_id
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.baseline import (
    BaselineRepository,
    baseline_models,
)
from mindflow.infrastructure.repositories.focus import focus_sessions
from mindflow.infrastructure.repositories.preferences import PreferencesRepository, user_preferences
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import behavior_feature_windows, metadata
from mindflow.services.telemetry_service import TelemetryService


@pytest.fixture
async def telemetry_repo(engine, session_factory):
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.run_sync(activity_events.metadata.create_all)
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


async def test_feedback_snapshots_and_intervention_checks(
    telemetry_repo: TelemetryRepository, engine, session_factory,
) -> None:
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    async with engine.begin() as connection:
        await connection.run_sync(focus_sessions.metadata.create_all)
    async with session_factory() as session, session.begin():
        await session.execute(
            focus_sessions.insert().values(
                id="session-1",
                user_id=1,
                date="2026-07-24",
                start_time=start.isoformat(),
                end_time=(start + timedelta(minutes=30)).isoformat(),
                session_type="focus",
                created_at=datetime.now(UTC).isoformat(),
            )
        )
    saved = await telemetry_repo.save_focus_feedback(
        user_id=1, session_id="session-1", label="focus", score=5, task_type="coding"
    )
    assert saved["session_start_utc"] == start.isoformat()
    assert saved["session_end_utc"] == (start + timedelta(minutes=30)).isoformat()

    await telemetry_repo.save_intervention_check(
        user_id=1,
        checked_at=datetime.now(UTC).isoformat(),
        reason="low_confidence",
        confidence=0.4,
        ml_status="ready",
    )
    async with session_factory() as session:
        from mindflow.infrastructure.schema import intervention_checks
        count = await session.scalar(
            sa.select(sa.func.count()).select_from(intervention_checks)
        )
    assert count == 1


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


async def test_focus_feedback_update_reuses_existing_row(
    telemetry_repo: TelemetryRepository,
) -> None:
    original = await telemetry_repo.save_focus_feedback(
        user_id=1,
        session_id="session-1",
        label="focus",
        score=5,
        task_type="coding",
    )

    updated = await telemetry_repo.save_focus_feedback(
        user_id=1,
        session_id="session-1",
        label="distracted",
        score=2,
        task_type=None,
    )
    stored = await telemetry_repo.list_focus_feedback(1)

    assert updated["id"] == original["id"]
    assert len(stored) == 1
    assert stored[0]["label"] == "distracted"
    assert stored[0]["score"] == 2
    assert stored[0]["task_type"] is None


async def test_revoke_browser_tokens_returns_exact_count(
    telemetry_repo: TelemetryRepository,
) -> None:
    await telemetry_repo.save_browser_token(1, "token-1")
    await telemetry_repo.save_browser_token(1, "token-2")
    await telemetry_repo.save_browser_token(2, "token-other-user")

    revoked = await telemetry_repo.revoke_browser_tokens(1)

    assert revoked == 2
    assert not await telemetry_repo.verify_browser_token("token-1")
    assert not await telemetry_repo.verify_browser_token("token-2")
    assert await telemetry_repo.verify_browser_token("token-other-user")
    assert await telemetry_repo.revoke_browser_tokens(1) == 0


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
        feature_schema_version=3,
        features_json="{}",
    )
    await telemetry_repo.save_feature_window(
        user_id=1,
        window_start_utc=now - timedelta(days=10),
        window_end_utc=now - timedelta(days=10, minutes=-5),
        feature_schema_version=3,
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


async def test_cleanup_old_telemetry_deletes_raw_activity_events(
    telemetry_repo: TelemetryRepository,
    session_factory,
) -> None:
    """Raw ``activity_events`` older than the activity cutoff are deleted."""
    now = datetime(2026, 7, 24, 8, tzinfo=UTC)
    activity = SQLAlchemyActivityRepository(session_factory)
    await activity.append_event(
        make_event(
            user_id=1,
            timestamp_utc=now - timedelta(days=40),
            app_name="Old App",
            process_name="old.exe",
        )
    )
    await activity.append_event(
        make_event(
            user_id=1,
            timestamp_utc=now - timedelta(days=10),
            app_name="Recent App",
            process_name="recent.exe",
        )
    )

    deleted = await telemetry_repo.cleanup_old_telemetry(
        interaction_cutoff=now - timedelta(days=7),
        activity_cutoff=now - timedelta(days=30),
        feature_cutoff=now - timedelta(days=180),
    )

    remaining = await activity.query_range(
        1, now - timedelta(days=400), now + timedelta(days=1)
    )
    assert deleted >= 1
    assert [event.data.app_name for event in remaining] == ["Recent App"]


async def test_cleanup_old_telemetry_activity_cutoff_boundary_preserved(
    telemetry_repo: TelemetryRepository,
    session_factory,
) -> None:
    """An event exactly at the activity cutoff survives (strict ``<``)."""
    now = datetime(2026, 7, 24, 8, tzinfo=UTC)
    activity = SQLAlchemyActivityRepository(session_factory)
    await activity.append_event(
        make_event(
            user_id=1,
            timestamp_utc=now - timedelta(days=30),
            app_name="Boundary App",
            process_name="boundary.exe",
        )
    )

    deleted = await telemetry_repo.cleanup_old_telemetry(
        interaction_cutoff=now - timedelta(days=7),
        activity_cutoff=now - timedelta(days=30),
        feature_cutoff=now - timedelta(days=180),
    )

    remaining = await activity.query_range(
        1, now - timedelta(days=400), now + timedelta(days=1)
    )
    assert deleted == 0
    assert len(remaining) == 1


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
        feature_schema_version=3,
        features_json="{}",
    )

    deleted = await telemetry_repo.delete_scope(1, "interaction")

    assert deleted == 2
    assert await telemetry_repo.list_feature_windows(1) == []


# ── Todo 8: baseline updated only from newly inserted windows ──────────────
#
# The rollup must fold feature windows into the persisted personal baseline
# exactly once: re-rolling the same range is a no-op for the baseline, while
# one later new window increments it exactly once. Windows and the baseline
# row are persisted in one transaction so a baseline failure rolls the whole
# write back and a retry is fully recoverable (no partial sync, no double
# count through Welford's algorithm).


async def _wire_baseline_service(engine: Any, session_factory: Any, tmp_path: Any):
    """Build a real TelemetryService whose rollup also refreshes the baseline.

    Returns (service, activity_repository, telemetry_repository,
    baseline_repository) against one temp-SQLite database.
    """
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.run_sync(activity_events.metadata.create_all)
        await connection.run_sync(user_preferences.metadata.create_all)
    activity_repository = SQLAlchemyActivityRepository(session_factory)
    telemetry_repository = TelemetryRepository(session_factory)
    baseline_repository = BaselineRepository(session_factory)
    service = TelemetryService(
        telemetry_repository,
        PreferencesRepository(session_factory),
        data_dir=tmp_path,
        activity_repository=activity_repository,
        baseline_repository=baseline_repository,
        session_factory=session_factory,
    )
    return service, activity_repository, telemetry_repository, baseline_repository


async def _baseline_row_json(session_factory: Any) -> str | None:
    stmt = (
        sa.select(baseline_models.c.model_json)
        .where(baseline_models.c.user_id == 1)
    )
    async with session_factory() as session:
        return (await session.execute(stmt)).scalar_one_or_none()


def _canonical_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Idempotent projection of a feature window (excludes id/created_at)."""
    return [
        {
            key: window[key]
            for key in (
                "window_start_utc",
                "window_end_utc",
                "feature_schema_version",
                "features_json",
                "label",
            )
        }
        for window in windows
    ]


async def test_rollup_twice_leaves_windows_and_baseline_byte_equivalent(
    engine, session_factory, tmp_path,
) -> None:
    """Given: one event spanning a 10-minute range and a baseline-wired rollup.
    When: the same range is rolled up twice.
    Then: window content and the persisted baseline row stay byte-equivalent and
    Welford counts reflect the windows exactly once (no double counting)."""
    service, activity_repository, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    end = start + timedelta(minutes=10)
    await activity_repository.append_event(
        make_event(
            user_id=1,
            timestamp_utc=start,
            duration_s=600,
            process_name="code.exe",
            is_idle=False,
        )
    )

    assert await service.rollup_feature_windows(start, end) == 2
    windows_after_first = await telemetry_repository.list_feature_windows(1)
    baseline_row_after_first = await _baseline_row_json(session_factory)
    assert baseline_row_after_first is not None
    baseline_after_first = await baseline_repository.get_latest(1)
    assert baseline_after_first is not None
    assert baseline_after_first.total_samples() == 2 * len(V2_FEATURE_NAMES)

    assert await service.rollup_feature_windows(start, end) == 2
    windows_after_second = await telemetry_repository.list_feature_windows(1)
    baseline_row_after_second = await _baseline_row_json(session_factory)

    assert _canonical_windows(windows_after_second) == _canonical_windows(
        windows_after_first
    )
    assert baseline_row_after_second == baseline_row_after_first

    reloaded = await baseline_repository.get_latest(1)
    assert reloaded is not None
    assert reloaded.total_samples() == 2 * len(V2_FEATURE_NAMES)
    assert reloaded.total_days == 1
    assert reloaded.overall_mean("active_seconds_ratio") == pytest.approx(1.0)


async def test_rollup_baseline_increments_once_for_a_later_new_window(
    engine, session_factory, tmp_path,
) -> None:
    """Given: an early range rolled once, then one new event in a disjoint range.
    When: the later range is rolled up (twice).
    Then: the baseline increments exactly once per genuinely new window."""
    service, activity_repository, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    early = datetime(2026, 7, 24, 8, tzinfo=UTC)
    await activity_repository.append_event(
        make_event(
            user_id=1,
            timestamp_utc=early,
            duration_s=600,
            process_name="code.exe",
            is_idle=False,
        )
    )
    await service.rollup_feature_windows(early, early + timedelta(minutes=10))

    later = datetime(2026, 7, 24, 14, 0, tzinfo=UTC)
    await activity_repository.append_event(
        make_event(
            user_id=1,
            timestamp_utc=later,
            duration_s=600,
            process_name="code.exe",
            is_idle=False,
        )
    )
    assert await service.rollup_feature_windows(later, later + timedelta(minutes=5)) == 1
    after_first_new = await baseline_repository.get_latest(1)
    assert after_first_new is not None
    assert after_first_new.total_samples() == 3 * len(V2_FEATURE_NAMES)
    assert len(await telemetry_repository.list_feature_windows(1)) == 3

    assert await service.rollup_feature_windows(later, later + timedelta(minutes=5)) == 1
    after_second_new = await baseline_repository.get_latest(1)
    assert after_second_new is not None
    assert after_second_new.total_samples() == 3 * len(V2_FEATURE_NAMES)
    assert after_second_new.to_dict() == after_first_new.to_dict()


async def test_rollup_baseline_failure_rolls_back_windows_and_retry_recovers(
    engine, session_factory, tmp_path, monkeypatch,
) -> None:
    """Given: a baseline-wired rollup whose baseline write is about to fail.
    When: the rollup runs once (failing) and then again with a healthy repo.
    Then: the failed run persists nothing (explicit transaction boundary — no
    sync is claimed), and the retry persists windows + baseline exactly once."""
    service, activity_repository, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    end = start + timedelta(minutes=10)
    await activity_repository.append_event(
        make_event(
            user_id=1,
            timestamp_utc=start,
            duration_s=600,
            process_name="code.exe",
            is_idle=False,
        )
    )
    real_upsert = baseline_repository.upsert

    async def broken_upsert(model, *, session=None):
        raise RuntimeError("baseline write failed")

    monkeypatch.setattr(baseline_repository, "upsert", broken_upsert)
    with pytest.raises(RuntimeError):
        await service.rollup_feature_windows(start, end)

    assert await telemetry_repository.list_feature_windows(1) == []
    assert await baseline_repository.get_latest(1) is None

    monkeypatch.setattr(baseline_repository, "upsert", real_upsert)
    assert await service.rollup_feature_windows(start, end) == 2
    assert len(await telemetry_repository.list_feature_windows(1)) == 2
    reloaded = await baseline_repository.get_latest(1)
    assert reloaded is not None
    assert reloaded.total_samples() == 2 * len(V2_FEATURE_NAMES)


# ── Todo 9: conditional bounded V2 baseline backfill (seam) ────────────────
#
# ``TelemetryService.rebuild_baseline_if_needed`` is the startup seam (wired by
# Todo 12, not here). It must:
#   - rebuild only when the baseline row is missing OR its stored
#     ``feature_schema_version`` is not 2; a V1 payload is never upgraded in
#     memory — it is replaced only after a complete V2 rebuild
#   - load at most the prior 180 business days of existing V2 windows via ONE
#     bounded range query (no per-row lookups), the cutoff being a 180-day
#     shift of ``now`` in the configured business timezone
#   - build the fresh model entirely, then persist with a single upsert inside
#     one caller-owned transaction, so any interruption before that upsert
#     leaves the prior baseline intact
#   - be rerunnable: an existing V2 baseline is skipped byte-identically, and a
#     repeated rebuild reproduces identical stats/counts/dates (Welford + local
#     dates accumulate identically when the same windows are processed in the
#     same ascending order, all at once or one at a time)


def _make_v2_window(
    start_utc: datetime,
    *,
    user_id: int = 1,
    app_switch_count: float = 10.0,
    active_seconds_ratio: float = 0.5,
    feature_schema_version: int = 3,
) -> dict[str, Any]:
    """One feature-window row carrying the flat v2 feature vocabulary."""
    features = {name: 0.0 for name in V2_FEATURE_NAMES}
    features["app_switch_count"] = app_switch_count
    features["active_seconds_ratio"] = active_seconds_ratio
    return {
        "user_id": user_id,
        "window_start_utc": start_utc,
        "window_end_utc": start_utc + timedelta(minutes=5),
        "feature_schema_version": feature_schema_version,
        "features_json": json.dumps(features, ensure_ascii=False),
        "label": None,
    }


async def _seed_raw_window(
    session_factory: Any,
    *,
    window_start_utc: str,
    features_json: str = "{}",
) -> None:
    """Insert one feature-window row verbatim (bypasses timestamp parsing)."""
    async with session_factory() as session:
        await session.execute(
            sa.insert(behavior_feature_windows).values(
                id=new_id(),
                user_id=1,
                window_start_utc=window_start_utc,
                window_end_utc="2026-07-25T00:05:00+00:00",
                feature_schema_version=3,
                features_json=features_json,
                label=None,
                created_at="2026-07-24T00:00:00+00:00",
            )
        )
        await session.commit()


def _v1_baseline_payload(user_id: int = 1) -> dict[str, Any]:
    """A legacy V1 baseline payload (no feature_schema_version field)."""
    return {
        "user_id": user_id,
        "timezone": "Asia/Shanghai",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "total_days": 3,
        "stats": {},
        "top_apps": {},
    }


class _InterruptedBaselineWriteError(RuntimeError):
    """Test-only failure injection for interruption atomicity probes."""


def _canonical_baseline(model: BaselineModel) -> dict[str, Any]:
    """Semantic projection of a baseline — the fields a rebuild must reproduce."""
    payload = model.to_dict()
    return {
        "feature_schema_version": payload["feature_schema_version"],
        "timezone": payload["timezone"],
        "total_days": payload["total_days"],
        "local_dates": payload["local_dates"],
        "stats": payload["stats"],
        "top_apps": payload["top_apps"],
    }


async def test_backfill_rebuilds_missing_baseline_from_v2_windows(
    engine, session_factory, tmp_path,
) -> None:
    """Given: V2 windows and no baseline row. When: the seam runs once.
    Then: a V2 baseline is created carrying exactly those windows' stats,
    sample counts and local days."""
    service, _, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    start = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
    await telemetry_repository.upsert_feature_windows([
        _make_v2_window(start, app_switch_count=10.0, active_seconds_ratio=0.5),
        _make_v2_window(
            start + timedelta(hours=6),
            app_switch_count=14.0,
            active_seconds_ratio=0.8,
        ),
    ])

    result = await service.rebuild_baseline_if_needed(
        1,
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    assert result.rebuilt is True
    assert result.reason == "missing"
    assert result.windows_loaded == 2
    assert result.samples == 2 * len(V2_FEATURE_NAMES)
    baseline = await baseline_repository.get_latest(1)
    assert baseline is not None
    assert baseline.FEATURE_SCHEMA_VERSION == 3
    assert baseline.total_samples() == 2 * len(V2_FEATURE_NAMES)
    assert baseline.total_days == 1
    assert baseline.overall_mean("app_switch_count") == pytest.approx(12.0)
    assert baseline.overall_mean("active_seconds_ratio") == pytest.approx(0.65)


async def test_backfill_replaces_v1_baseline_only_after_full_v2_rebuild(
    engine, session_factory, tmp_path,
) -> None:
    """Given: a persisted legacy V1 baseline and V2 windows. When: the seam
    runs. Then: the V1 row is replaced by a complete fresh V2 baseline — the
    V1 payload is discarded, never upgraded in memory, so no V1 remnant
    (old total_days, empty stats) survives in the stored row."""
    service, _, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    v1 = BaselineModel.from_dict(_v1_baseline_payload())
    await baseline_repository.upsert(v1)
    await telemetry_repository.upsert_feature_windows([
        _make_v2_window(datetime(2026, 7, 24, 8, 0, tzinfo=UTC), app_switch_count=10.0),
    ])

    result = await service.rebuild_baseline_if_needed(
        1,
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    assert result.rebuilt is True
    assert result.reason == "schema_mismatch"
    baseline = await baseline_repository.get_latest(1)
    assert baseline is not None
    assert baseline.FEATURE_SCHEMA_VERSION == 3
    assert baseline.total_samples() == len(V2_FEATURE_NAMES)
    # The V1 payload's stored 3-day count was discarded, not carried over.
    assert baseline.total_days == 1
    assert baseline.overall_mean("app_switch_count") == pytest.approx(10.0)
    stored = await _baseline_row_json(session_factory)
    assert stored is not None
    assert json.loads(stored)["feature_schema_version"] == 3
    assert await _count_baseline_rows(session_factory) == 1


async def _count_baseline_rows(session_factory: Any) -> int:
    stmt = sa.select(sa.func.count()).select_from(baseline_models)
    async with session_factory() as session:
        return int((await session.execute(stmt)).scalar() or 0)


async def test_backfill_skips_existing_v2_baseline_without_writing(
    engine, session_factory, tmp_path,
) -> None:
    """Given: an existing V2 baseline. When: the seam runs again. Then: it
    skips (no rebuild claimed, zero windows loaded) and the stored row is
    byte-identical — the seam is a no-op for current-state V2 data."""
    service, _, telemetry_repository, _ = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    await telemetry_repository.upsert_feature_windows([
        _make_v2_window(datetime(2026, 7, 24, 8, 0, tzinfo=UTC), app_switch_count=10.0),
    ])
    first = await service.rebuild_baseline_if_needed(
        1,
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )
    assert first.rebuilt is True
    stored_before = await _baseline_row_json(session_factory)

    second = await service.rebuild_baseline_if_needed(
        1,
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    assert second.rebuilt is False
    assert second.reason == "skipped_v2"
    assert second.windows_loaded == 0
    assert second.samples == 0
    stored_after = await _baseline_row_json(session_factory)
    assert stored_after == stored_before


async def test_backfill_cuts_at_exact_180_business_day_bound_in_asia_shanghai(
    engine, session_factory, tmp_path,
) -> None:
    """Given: V2 windows straddling the 180-day business-timezone cutoff and
    the ``now`` upper bound. When: the seam rebuilds with a fixed now in
    Asia/Shanghai. Then: exactly the windows inside [cutoff, now) load — one
    minute below cutoff and a window exactly at ``now`` are excluded, and the
    cutoff is the precise 180-day shift of now in the business timezone
    (2026-02-01T12:00Z)."""
    service, _, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    now_utc = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    cutoff_utc = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    await telemetry_repository.upsert_feature_windows([
        _make_v2_window(datetime(2026, 1, 31, 12, 0, tzinfo=UTC), app_switch_count=0.0),
        _make_v2_window(datetime(2026, 2, 1, 11, 59, tzinfo=UTC), app_switch_count=0.0),
        _make_v2_window(cutoff_utc, app_switch_count=1.0),
        _make_v2_window(datetime(2026, 2, 1, 12, 1, tzinfo=UTC), app_switch_count=2.0),
        _make_v2_window(now_utc, app_switch_count=3.0),
    ])

    result = await service.rebuild_baseline_if_needed(
        1, timezone="Asia/Shanghai", now_utc=now_utc,
    )

    assert result.rebuilt is True
    assert result.cutoff_utc == cutoff_utc
    assert result.windows_loaded == 2
    baseline = await baseline_repository.get_latest(1)
    assert baseline is not None
    assert baseline.total_samples() == 2 * len(V2_FEATURE_NAMES)
    assert baseline.overall_mean("app_switch_count") == pytest.approx(1.5)


async def test_backfill_rebuild_after_wipe_is_identical_to_first_rebuild(
    engine, session_factory, tmp_path,
) -> None:
    """Given: V2 windows and a rebuilt baseline. When: the baseline row is
    removed and the seam runs again. Then: the rerun reproduces the exact same
    stats, counts and local dates — the backfill is deterministic."""
    service, _, telemetry_repository, _ = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    start = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
    await telemetry_repository.upsert_feature_windows([
        _make_v2_window(start, app_switch_count=10.0),
        _make_v2_window(start + timedelta(hours=5), app_switch_count=14.0),
    ])
    kwargs = {
        "timezone": "Asia/Shanghai",
        "now_utc": datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    }
    first = await service.rebuild_baseline_if_needed(1, **kwargs)
    assert first.rebuilt is True
    first_row = await _baseline_row_json(session_factory)
    assert first_row is not None
    first_payload = json.loads(first_row)

    async with session_factory() as session:
        await session.execute(
            sa.delete(baseline_models).where(baseline_models.c.user_id == 1)
        )
        await session.commit()

    second = await service.rebuild_baseline_if_needed(1, **kwargs)
    assert second.rebuilt is True
    second_row = await _baseline_row_json(session_factory)
    assert second_row is not None
    second_payload = json.loads(second_row)
    for key in ("feature_schema_version", "total_days", "local_dates", "stats", "top_apps"):
        assert second_payload[key] == first_payload[key]


async def test_backfill_equals_ordered_incremental_processing_of_same_windows(
    engine, session_factory, tmp_path,
) -> None:
    """Given: V2 windows across several local days and hours. When: the seam
    backfills (all windows at once, ascending) vs a fresh model updated one
    window at a time in the same order (incremental). Then: baseline stats,
    sample counts and local dates are equivalent."""
    service, _, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    windows = [
        _make_v2_window(datetime(2026, 7, 24, 8, 0, tzinfo=UTC), app_switch_count=10.0),
        _make_v2_window(datetime(2026, 7, 24, 8, 5, tzinfo=UTC), app_switch_count=12.0),
        _make_v2_window(datetime(2026, 7, 25, 2, 0, tzinfo=UTC), app_switch_count=14.0),
        _make_v2_window(datetime(2026, 7, 25, 14, 30, tzinfo=UTC), app_switch_count=18.0),
    ]
    await telemetry_repository.upsert_feature_windows(windows)

    result = await service.rebuild_baseline_if_needed(
        1,
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )
    assert result.rebuilt is True
    backfilled = await baseline_repository.get_latest(1)
    assert backfilled is not None

    incremental = BaselineModel(user_id=1, timezone="Asia/Shanghai")
    for window in windows:
        incremental.update([window])

    assert _canonical_baseline(backfilled) == _canonical_baseline(incremental)


async def test_backfill_interruption_before_upsert_leaves_prior_baseline_intact(
    engine, session_factory, tmp_path, monkeypatch,
) -> None:
    """Given: a persisted V1 baseline and V2 windows. When: the seam's single
    upsert is interrupted (raises) and then reruns healthy. Then: the failed
    run leaves the V1 row byte-identical — the fresh model is built before any
    write, so nothing half-replaced is persisted — and the retry replaces it
    with a complete V2 baseline."""
    service, _, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    await baseline_repository.upsert(BaselineModel.from_dict(_v1_baseline_payload()))
    await telemetry_repository.upsert_feature_windows([
        _make_v2_window(datetime(2026, 7, 24, 8, 0, tzinfo=UTC), app_switch_count=10.0),
    ])
    before = await _baseline_row_json(session_factory)
    real_upsert = baseline_repository.upsert

    async def broken_upsert(model, *, session=None):
        # Deliberate failure injection to prove interruption atomicity; matches
        # the identical Todo 8 fixture pattern above.
        raise _InterruptedBaselineWriteError("interrupted before/at upsert")

    monkeypatch.setattr(baseline_repository, "upsert", broken_upsert)
    with pytest.raises(RuntimeError):
        await service.rebuild_baseline_if_needed(
            1,
            timezone="Asia/Shanghai",
            now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
        )

    assert await _baseline_row_json(session_factory) == before
    assert await _count_baseline_rows(session_factory) == 1

    monkeypatch.setattr(baseline_repository, "upsert", real_upsert)
    result = await service.rebuild_baseline_if_needed(
        1,
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )
    assert result.rebuilt is True
    baseline = await baseline_repository.get_latest(1)
    assert baseline is not None
    assert baseline.FEATURE_SCHEMA_VERSION == 3
    assert baseline.total_days == 1
    assert baseline.total_samples() == len(V2_FEATURE_NAMES)
    assert await _count_baseline_rows(session_factory) == 1


async def test_feature_window_range_query_is_bounded_filtered_and_ordered(
    engine, session_factory,
) -> None:
    """Given: V1 and V2 windows on both sides of [start, end). When: the range
    query runs. Then: only V2 windows inside the half-open bounds come back in
    ascending order from a single query — older/newer windows and V1 rows are
    excluded, so the seam never loads the full history."""
    telemetry_repository = TelemetryRepository(session_factory)
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    start = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    end = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    await telemetry_repository.upsert_feature_windows([
        _make_v2_window(datetime(2026, 2, 1, 11, 59, tzinfo=UTC), app_switch_count=1.0),
        _make_v2_window(start, app_switch_count=2.0),
        _make_v2_window(datetime(2026, 6, 1, 0, 0, tzinfo=UTC), app_switch_count=3.0),
        _make_v2_window(end, app_switch_count=4.0),
        _make_v2_window(
            datetime(2026, 3, 1, 0, 0, tzinfo=UTC),
            app_switch_count=5.0,
            feature_schema_version=1,
        ),
    ])

    found = await telemetry_repository.list_feature_windows_in_range(1, start, end)

    assert [row["window_start_utc"] for row in found] == [
        "2026-02-01T12:00:00+00:00",
        "2026-06-01T00:00:00+00:00",
    ]


async def test_backfill_tolerates_malformed_stored_windows(
    engine, session_factory, tmp_path,
) -> None:
    """Given: stored V2 windows where one has an unparseable timestamp and one
    has unparseable feature JSON, alongside valid windows. When: the seam
    rebuilds. Then: it does not crash — the malformed rows are loaded but
    skipped by the model, and the baseline reflects only the valid windows."""
    service, _, telemetry_repository, baseline_repository = (
        await _wire_baseline_service(engine, session_factory, tmp_path)
    )
    start = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
    await telemetry_repository.upsert_feature_windows([
        _make_v2_window(start, app_switch_count=10.0),
        _make_v2_window(start + timedelta(minutes=5), app_switch_count=12.0),
    ])
    await _seed_raw_window(session_factory, window_start_utc="2026-07-25T08:00:00+99:99")
    await _seed_raw_window(
        session_factory,
        window_start_utc="2026-07-25T09:00:00+00:00",
        features_json="{not json",
    )

    result = await service.rebuild_baseline_if_needed(
        1,
        timezone="Asia/Shanghai",
        now_utc=datetime(2026, 7, 31, 12, 0, tzinfo=UTC),
    )

    assert result.rebuilt is True
    assert result.windows_loaded == 4
    baseline = await baseline_repository.get_latest(1)
    assert baseline is not None
    # Samples come only from the two valid windows; the malformed rows load
    # but contribute none. total_days counts one local date per row with a
    # parseable timestamp (domain contract), and the features-malformed row
    # on 2026-07-25 still has a valid timestamp — hence 2, not 1.
    assert baseline.total_samples() == 2 * len(V2_FEATURE_NAMES)
    assert baseline.total_days == 2
    assert baseline.overall_mean("app_switch_count") == pytest.approx(11.0)
