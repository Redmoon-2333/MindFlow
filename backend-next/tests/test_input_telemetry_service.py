"""Privacy regression tests for interaction-bucket ``context_key``.

The stored ``interaction_buckets.context_key`` must derive from
``process_name`` alone. Two otherwise-identical buckets for the same
process with different window titles must produce the same key, and
neither the raw title nor any hash that includes it may appear in the
stored key. The ``unknown`` fallback for a missing last event and the
deterministic per-process separation are preserved.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

from mindflow.domain.events import ActivityEvent, make_event
from mindflow.services.input_telemetry_service import InputTelemetryService

_PROCESS = "Code.exe"
_TITLE_A = "Project Alpha - main.py"
_TITLE_B = "SECRET-TITLE-勿泄露 review"


def _expected_process_only_key(process_name: str) -> str:
    digest = hashlib.sha256(process_name.encode()).hexdigest()[:16]
    return f"{process_name.lower()}:{digest}"


def _legacy_title_aware_key(process_name: str, window_title: str) -> str:
    source = f"{process_name}\0{window_title}"
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    return f"{process_name.lower()}:{digest}"


def _make_service(last_event: ActivityEvent | None) -> InputTelemetryService:
    activity = AsyncMock()
    activity.last_event.return_value = last_event
    telemetry = AsyncMock()
    return InputTelemetryService(
        telemetry_repository=telemetry,
        activity_repository=activity,
        user_id=1,
    )


def _persisted_key(service: InputTelemetryService) -> str:
    return service._telemetry_repository.save_interaction_bucket.call_args.kwargs["context_key"]


def _bucket_message(timestamp: datetime) -> dict[str, Any]:
    return {
        "window_start_utc": timestamp.isoformat(),
        "duration_s": 30.0,
        "keypress_count": 5,
        "mouse_click_count": 2,
        "scroll_delta": 120,
        "mouse_distance_px": 12.5,
        "input_active_s": 30.0,
        "interaction_burst_count": 1,
    }


async def test_context_key_unchanged_when_only_window_title_differs() -> None:
    base = datetime(2026, 8, 5, 10, 0, 0, tzinfo=UTC)
    event_a = make_event(
        user_id=1,
        timestamp_utc=base,
        process_name=_PROCESS,
        window_title=_TITLE_A,
    )
    event_b = make_event(
        user_id=1,
        timestamp_utc=base,
        process_name=_PROCESS,
        window_title=_TITLE_B,
    )

    service_a = _make_service(event_a)
    service_b = _make_service(event_b)
    await service_a._persist_bucket(_bucket_message(base))
    await service_b._persist_bucket(_bucket_message(base))

    key_a = _persisted_key(service_a)
    key_b = _persisted_key(service_b)
    assert key_a == key_b
    assert key_a == _expected_process_only_key(_PROCESS)


async def test_context_key_contains_no_title_or_combined_hash() -> None:
    base = datetime(2026, 8, 5, 10, 0, 0, tzinfo=UTC)
    event = make_event(
        user_id=1,
        timestamp_utc=base,
        process_name=_PROCESS,
        window_title=_TITLE_B,
    )
    service = _make_service(event)
    await service._persist_bucket(_bucket_message(base))

    key = _persisted_key(service)
    assert key == _expected_process_only_key(_PROCESS)
    assert _TITLE_B not in key
    assert _TITLE_B.lower() not in key
    assert key != _legacy_title_aware_key(_PROCESS, _TITLE_B)


async def test_missing_last_event_still_falls_back_to_unknown() -> None:
    base = datetime(2026, 8, 5, 10, 0, 0, tzinfo=UTC)
    service = _make_service(None)
    await service._persist_bucket(_bucket_message(base))

    key = _persisted_key(service)
    assert key == "unknown"


async def test_distinct_processes_produce_distinct_keys() -> None:
    base = datetime(2026, 8, 5, 10, 0, 0, tzinfo=UTC)
    event_code = make_event(
        user_id=1, timestamp_utc=base, process_name="Code.exe", window_title=_TITLE_A
    )
    event_browser = make_event(
        user_id=1, timestamp_utc=base, process_name="browser.exe", window_title=_TITLE_B
    )

    service_code = _make_service(event_code)
    service_browser = _make_service(event_browser)
    await service_code._persist_bucket(_bucket_message(base))
    await service_browser._persist_bucket(_bucket_message(base))

    key_code = _persisted_key(service_code)
    key_browser = _persisted_key(service_browser)
    assert key_code == _expected_process_only_key("Code.exe")
    assert key_browser == _expected_process_only_key("browser.exe")
    assert key_code != key_browser
