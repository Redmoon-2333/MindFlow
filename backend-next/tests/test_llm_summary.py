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
    window_title: str = "",
    is_idle: bool = False,
):
    return make_event(
        user_id=1,
        timestamp_utc=timestamp,
        duration_s=duration_s,
        app_name=process_name,
        process_name=process_name,
        window_title=window_title,
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
        _event(start, 30.0, "douyin.exe"),
        _event(start + timedelta(seconds=30), 30.0, "code.exe"),
        _event(start + timedelta(seconds=60), 60.0, "code.exe", is_idle=True),
    ]

    summary = build_behavior_summary(events)

    assert summary.social_media_ratio == pytest.approx(0.5)


def test_browser_work_not_blanket_entertainment() -> None:
    """Chrome/Edge reading a document must NOT count as social media."""
    start = datetime(2026, 7, 24, tzinfo=UTC)
    events = [
        _event(
            start,
            30.0,
            "msedge.exe",
            window_title="论文.pdf - Microsoft Edge",
        ),
        _event(start + timedelta(seconds=30), 30.0, "code.exe"),
    ]

    summary = build_behavior_summary(events)

    assert summary.social_media_ratio == pytest.approx(0.0)


def test_browser_entertainment_domain_counts() -> None:
    """Browser time on a known entertainment domain still counts as social."""
    start = datetime(2026, 7, 24, tzinfo=UTC)
    events = [
        _event(
            start,
            30.0,
            "chrome.exe",
            window_title="youtube.com/watch?v=dQw4w9WgXcQ",
        ),
        _event(start + timedelta(seconds=30), 30.0, "code.exe"),
    ]

    summary = build_behavior_summary(events)

    assert summary.social_media_ratio == pytest.approx(0.5)


def test_browser_productive_learning_not_entertainment() -> None:
    """Bilibili lecture content in a browser is work, not entertainment."""
    start = datetime(2026, 7, 24, tzinfo=UTC)
    events = [
        _event(
            start,
            30.0,
            "chrome.exe",
            window_title="高等数学第3讲 - bilibili",
        ),
        _event(start + timedelta(seconds=30), 30.0, "code.exe"),
    ]

    summary = build_behavior_summary(events)

    assert summary.social_media_ratio == pytest.approx(0.0)
