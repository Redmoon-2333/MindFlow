from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from mindflow.domain.events import make_event
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.app_classification import (
    AppClassificationRulesRepository,
    app_classification_rules,
)
from mindflow.infrastructure.repositories.telemetry import (
    TelemetryRepository,
    behavior_feature_windows,
    browser_segments,
    browser_tokens,
)
from mindflow.infrastructure.schema import (
    metadata as telemetry_metadata,
)
from mindflow.services import maintenance_service as maintenance_module
from mindflow.services.export_service import ExportService
from mindflow.services.maintenance_service import MaintenanceService
from mindflow.services.telemetry_service import TelemetryService
from mindflow.train.v2 import V2_FEATURE_NAMES


async def test_cleanup_deletes_in_bounded_batches(
    engine: Any,
    session_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(activity_events.metadata.create_all)
        await connection.execute(
            activity_events.insert(),
            [
                {
                    "id": f"old-{index}",
                    "user_id": 1,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "duration_s": 1.0,
                    "data_json": json.dumps({
                        "app_name": "Test",
                        "window_title": "",
                        "process_name": "test.exe",
                        "is_idle": False,
                        "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    }),
                    "event_type": "window_snapshot",
                }
                for index in range(5)
            ],
        )

    monkeypatch.setattr(maintenance_module, "_BATCH_SIZE", 2)
    service = MaintenanceService(
        engine=engine,
        session_factory=session_factory,
        notifier=AsyncMock(),
        clock=lambda: datetime(2026, 7, 26, tzinfo=UTC),
    )

    assert await service.cleanup_old_events(retention_days=30) == 5


async def test_activity_repository_cursor_page_avoids_overlap(
    engine: Any,
    session_factory: Any,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(activity_events.metadata.create_all)
    repository = SQLAlchemyActivityRepository(session_factory)
    start = datetime(2026, 7, 26, 8, tzinfo=UTC)
    for index in range(5):
        await repository.append_event(
            make_event(
                user_id=1,
                timestamp_utc=start + timedelta(minutes=index),
                duration_s=1.0,
                app_name=f"App {index}",
                process_name=f"app-{index}.exe",
            )
        )

    first = await repository.query_range(
        1,
        start - timedelta(minutes=1),
        start + timedelta(minutes=10),
        limit=2,
        descending=True,
    )
    cursor = (first[-1].timestamp_utc.isoformat(), first[-1].id)
    second = await repository.query_range(
        1,
        start - timedelta(minutes=1),
        start + timedelta(minutes=10),
        limit=2,
        descending=True,
        cursor=cursor,
    )

    assert len(first) == 2
    assert len(second) == 2
    assert {event.id for event in first}.isdisjoint(event.id for event in second)
    assert second[0].timestamp_utc < first[-1].timestamp_utc


async def test_feature_windows_bulk_upsert_and_latest_limit_one(
    engine: Any,
    session_factory: Any,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(telemetry_metadata.create_all)
    repository = TelemetryRepository(session_factory)
    start = datetime(2026, 7, 26, 8, tzinfo=UTC)

    await repository.upsert_feature_windows([
        {
            "user_id": 1,
            "window_start_utc": start,
            "window_end_utc": start + timedelta(minutes=5),
            "feature_schema_version": 2,
            "features_json": '{"value": 1}',
            "label": None,
        },
        {
            "user_id": 1,
            "window_start_utc": start + timedelta(minutes=5),
            "window_end_utc": start + timedelta(minutes=10),
            "feature_schema_version": 2,
            "features_json": '{"value": 2}',
            "label": None,
        },
    ])
    await repository.upsert_feature_windows([
        {
            "user_id": 1,
            "window_start_utc": start + timedelta(minutes=5),
            "window_end_utc": start + timedelta(minutes=10),
            "feature_schema_version": 2,
            "features_json": '{"value": 3}',
            "label": "focus",
        }
    ])

    latest = await repository.latest_feature_window(1, feature_schema_version=2)
    async with engine.connect() as connection:
        count = await connection.scalar(
            sa.select(sa.func.count()).select_from(behavior_feature_windows)
        )

    assert count == 2
    assert latest is not None
    assert latest["features_json"] == '{"value": 3}'
    assert latest["label"] == "focus"


async def test_authenticated_heartbeat_updates_last_used_and_segment_atomically(
    engine: Any,
    session_factory: Any,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(telemetry_metadata.create_all)
    repository = TelemetryRepository(session_factory)
    token_hash = hashlib.sha256(b"secret").hexdigest()
    await repository.save_browser_token(1, token_hash)
    timestamp = datetime(2026, 7, 26, 8, tzinfo=UTC)

    authorized, segment = await repository.save_authenticated_browser_heartbeat(
        token_hash,
        heartbeat={
            "user_id": 1,
            "timestamp_utc": timestamp,
            "duration_s": 5.0,
            "browser_name": "edge",
            "domain": "docs.python.org",
            "audible": False,
            "context_key": "edge:docs.python.org",
        },
    )

    async with engine.connect() as connection:
        last_used = await connection.scalar(
            sa.select(browser_tokens.c.last_used_at).where(
                browser_tokens.c.token_hash == token_hash
            )
        )
        segment_count = await connection.scalar(
            sa.select(sa.func.count()).select_from(browser_segments)
        )

    assert authorized is True
    assert segment is not None
    assert last_used is not None
    assert segment_count == 1


async def test_rollup_uses_one_bulk_upsert_call(tmp_path: Any) -> None:
    start = datetime(2026, 7, 26, 8, tzinfo=UTC)
    activity_repository = AsyncMock()
    activity_repository.query_range.return_value = [
        make_event(
            user_id=1,
            timestamp_utc=start,
            duration_s=600.0,
            app_name="Code",
            process_name="code.exe",
        )
    ]
    activity_repository.last_event_before.return_value = None
    repository = AsyncMock()
    repository.list_interaction_buckets.return_value = []
    repository.list_browser_segments.return_value = []
    repository.last_browser_segment_before.return_value = None
    repository.upsert_feature_windows.return_value = None
    service = TelemetryService(
        repository=repository,
        preferences_repository=AsyncMock(),
        data_dir=tmp_path,
        activity_repository=activity_repository,
    )

    count = await service.rollup_feature_windows(start, start + timedelta(minutes=10))

    assert count == 2
    repository.upsert_feature_windows.assert_awaited_once()
    rows = repository.upsert_feature_windows.await_args.args[0]
    assert len(rows) == 2
    repository.save_feature_window.assert_not_awaited()


class _Classifier:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert features.shape == (1, len(V2_FEATURE_NAMES))
        return np.array([[0.25, 0.75]])

    def get_feature_importance(self) -> dict[str, float]:
        return {name: 0.1 for name in V2_FEATURE_NAMES}


class _Manager:
    classifier = _Classifier()
    current_version_tag = "test"


async def test_prediction_reads_only_latest_feature_window(tmp_path: Any) -> None:
    repository = AsyncMock()
    repository.latest_feature_window.return_value = {
        "window_start_utc": "2026-07-26T08:00:00+00:00",
        "features_json": "{}",
    }
    service = TelemetryService(repository, AsyncMock(), tmp_path)
    service.attach_model_manager(_Manager())

    prediction = await service.predict_latest_focus()

    assert prediction["focus_probability"] == 0.75
    repository.latest_feature_window.assert_awaited_once_with(
        1,
        feature_schema_version=2,
    )
    repository.list_feature_windows.assert_not_awaited()


async def test_replace_all_rolls_back_on_bulk_insert_failure(
    engine: Any,
    session_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(app_classification_rules.metadata.create_all)
    repository = AppClassificationRulesRepository(session_factory)
    original = await repository.add(
        1,
        {"process_name": "old.exe", "category": "other"},
    )

    monkeypatch.setattr(
        "mindflow.infrastructure.repositories.app_classification.uuid6.uuid7",
        lambda: "duplicate-id",
    )
    with pytest.raises(IntegrityError):
        await repository.replace_all(1, [
            {"process_name": "a.exe", "category": "code"},
            {"process_name": "b.exe", "category": "document"},
        ])

    assert await repository.get_all(1) == [original]


class _ChunkedActivityRepository:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.calls = 0

    async def iter_range_chunks(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
        *,
        chunk_size: int,
    ):
        self.calls += 1
        for index in range(0, len(self.events), chunk_size):
            yield self.events[index:index + chunk_size]

    async def query_range(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        raise AssertionError("streaming export must not materialize all events")


@pytest.mark.parametrize("fmt", ["csv", "json"])
async def test_export_stream_yields_multiple_chunks(fmt: str) -> None:
    start = datetime(2026, 7, 26, 8, tzinfo=UTC)
    activity_repository = _ChunkedActivityRepository([
        make_event(
            user_id=1,
            timestamp_utc=start + timedelta(seconds=index),
            duration_s=1.0,
            app_name=f"App {index}",
            process_name=f"app-{index}.exe",
        )
        for index in range(5)
    ])
    focus_repository = AsyncMock()
    focus_repository.query_range.return_value = []
    report_repository = AsyncMock()
    report_repository.query_range.return_value = []
    service = ExportService(
        activity_repository,
        focus_repository,
        report_repository,
    )

    result = await service.stream_events(start, start + timedelta(minutes=1), fmt=fmt, chunk_size=2)
    chunks = [chunk async for chunk in result.content]
    body = b"".join(chunks)

    assert activity_repository.calls == 1
    assert len(chunks) >= 3
    if fmt == "csv":
        assert body.decode("utf-8-sig").count("App ") == 5
    else:
        assert len(json.loads(body)["events"]) == 5
