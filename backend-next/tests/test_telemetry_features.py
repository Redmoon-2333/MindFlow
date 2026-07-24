from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mindflow.domain.events import make_event
from mindflow.services.telemetry_features import build_v2_feature_window


def test_build_v2_feature_window_combines_activity_input_and_browser() -> None:
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    events = [
        make_event(
            user_id=1,
            timestamp_utc=start,
            duration_s=120,
            process_name="code.exe",
            is_idle=False,
        ),
        make_event(
            user_id=1,
            timestamp_utc=start + timedelta(minutes=2),
            duration_s=60,
            process_name="code.exe",
            is_idle=True,
            event_type="idle_change",
        ),
        make_event(
            user_id=1,
            timestamp_utc=start + timedelta(minutes=3),
            duration_s=120,
            process_name="msedge.exe",
            is_idle=False,
        ),
    ]
    buckets = [
        {
            "duration_s": 30,
            "keypress_count": 30,
            "mouse_click_count": 10,
            "scroll_delta": 120,
            "mouse_distance_px": 600,
            "input_active_s": 15,
            "interaction_burst_count": 3,
        }
    ]
    browser = [
        {
            "duration_s": 120,
            "domain": "docs.python.org",
            "audible": False,
        }
    ]

    features = build_v2_feature_window(events, buckets, browser, start, end)

    assert features["feature_schema_version"] == 2
    assert features["idle_ratio"] == pytest.approx(0.2)
    assert features["keypress_rate_per_min"] == pytest.approx(6.0)
    assert features["mouse_click_rate_per_min"] == pytest.approx(2.0)
    assert features["click_key_ratio"] == pytest.approx(1 / 3)
    assert features["browser_ratio"] == pytest.approx(0.4)
    assert all(value == value for value in features.values() if isinstance(value, float))


def test_v2_feature_window_uses_overlap_ratios_and_bucket_intervals() -> None:
    start = datetime(2026, 7, 24, 8, tzinfo=UTC)
    end = start + timedelta(minutes=5)
    events = [
        make_event(
            user_id=1,
            timestamp_utc=start - timedelta(minutes=1),
            duration_s=180,
            process_name="code.exe",
            is_idle=False,
        ),
        make_event(
            user_id=1,
            timestamp_utc=start + timedelta(minutes=2),
            duration_s=180,
            process_name="browser.exe",
            is_idle=False,
        ),
    ]
    buckets = [
        {
            "window_start_utc": (start + timedelta(seconds=30)).isoformat(),
            "duration_s": 30,
            "keypress_count": 2,
            "mouse_click_count": 0,
            "scroll_delta": 0,
            "mouse_distance_px": 0,
            "input_active_s": 5,
            "interaction_burst_count": 1,
        },
        {
            "window_start_utc": (start + timedelta(seconds=90)).isoformat(),
            "duration_s": 30,
            "keypress_count": 2,
            "mouse_click_count": 0,
            "scroll_delta": 0,
            "mouse_distance_px": 0,
            "input_active_s": 5,
            "interaction_burst_count": 1,
        },
        {
            "window_start_utc": (start + timedelta(seconds=210)).isoformat(),
            "duration_s": 30,
            "keypress_count": 2,
            "mouse_click_count": 0,
            "scroll_delta": 0,
            "mouse_distance_px": 0,
            "input_active_s": 5,
            "interaction_burst_count": 1,
        },
    ]
    browser = [
        {
            "timestamp": start.isoformat(),
            "duration_s": 240,
            "domain": "docs.example.com",
            "audible": False,
        },
        {
            "timestamp": (start + timedelta(minutes=4)).isoformat(),
            "duration_s": 120,
            "domain": "chat.example.com",
            "audible": True,
        },
    ]

    features = build_v2_feature_window(events, buckets, browser, start, end)

    assert features["top_app_ratio"] == pytest.approx(0.6)
    assert features["top_domain_ratio"] == pytest.approx(0.8)
    assert features["longest_segment_ratio"] == pytest.approx(0.6)
    assert features["active_seconds_ratio"] == pytest.approx(1.0)
    assert features["interaction_interval_mean_s"] == pytest.approx(90.0)
    assert features["interaction_interval_std_s"] == pytest.approx(30.0)
    assert features["interaction_interval_cv"] == pytest.approx(1 / 3)
    for name in (
        "idle_ratio",
        "longest_segment_ratio",
        "input_active_ratio",
        "browser_ratio",
        "audible_browser_ratio",
        "active_seconds_ratio",
        "top_app_ratio",
        "top_domain_ratio",
    ):
        assert 0.0 <= float(features[name]) <= 1.0
