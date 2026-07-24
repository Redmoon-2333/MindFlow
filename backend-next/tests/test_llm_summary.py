from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mindflow.domain.events import make_event
from mindflow.infrastructure.llm.summary import build_behavior_summary


def _event(
    timestamp: datetime,
    duration_s: float,
    process_name: str,
    *,
    is_idle: bool = False,
):
    return make_event(
        user_id=1,
        timestamp_utc=timestamp,
        duration_s=duration_s,
        app_name=process_name,
        process_name=process_name,
        is_idle=is_idle,
    )


def test_summary_uses_recorded_duration_instead_of_wall_clock_span() -> None:
    start = datetime(2026, 7, 24, tzinfo=UTC)
    events = [
        _event(start, 30.0, "code.exe"),
        _event(start + timedelta(hours=12), 30.0, "code.exe"),
    ]

    summary = build_behavior_summary(events)

    assert summary.duration_min == pytest.approx(1.0)


def test_social_ratio_uses_non_idle_duration() -> None:
    start = datetime(2026, 7, 24, tzinfo=UTC)
    events = [
        _event(start, 30.0, "msedge.exe"),
        _event(start + timedelta(seconds=30), 30.0, "code.exe"),
        _event(start + timedelta(seconds=60), 60.0, "code.exe", is_idle=True),
    ]

    summary = build_behavior_summary(events)

    assert summary.social_media_ratio == pytest.approx(0.5)
