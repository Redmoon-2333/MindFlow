"""Privacy-preserving feature schema v2."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from mindflow.domain.events import ActivityEvent

FEATURE_SCHEMA_VERSION = 2


def build_v2_feature_window(
    events: list[ActivityEvent],
    interaction_buckets: list[dict[str, Any]],
    browser_segments: list[dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
) -> dict[str, int | float]:
    window_seconds = max(1.0, (window_end - window_start).total_seconds())
    window_minutes = window_seconds / 60.0
    event_durations = [
        (event, _event_overlap_seconds(event, window_start, window_end))
        for event in events
    ]
    event_durations = [(event, duration) for event, duration in event_durations if duration > 0]
    total_event_seconds = sum(duration for _, duration in event_durations)
    idle_seconds = sum(duration for event, duration in event_durations if event.data.is_idle)
    active_events = sorted(
        ((event, duration) for event, duration in event_durations if not event.data.is_idle),
        key=lambda item: item[0].timestamp_utc,
    )
    active_seconds = sum(duration for _, duration in active_events)
    app_switch_count = sum(
        current[0].data.process_name != previous[0].data.process_name
        for previous, current in zip(active_events, active_events[1:], strict=False)
    )
    longest_segment_s = max((duration for _, duration in active_events), default=0.0)
    app_durations: dict[str, float] = {}
    for event, duration in active_events:
        process_name = event.data.process_name
        app_durations[process_name] = app_durations.get(process_name, 0.0) + duration

    browser_rows = [
        (segment, _segment_overlap_seconds(segment, window_start, window_end))
        for segment in browser_segments
    ]
    browser_rows = [(segment, duration) for segment, duration in browser_rows if duration > 0]
    domains = [str(segment.get("domain", "")) for segment, _ in browser_rows]
    domain_switch_count = sum(
        current != previous for previous, current in zip(domains, domains[1:], strict=False)
    )
    browser_seconds = sum(duration for _, duration in browser_rows)
    audible_seconds = sum(
        duration for segment, duration in browser_rows if bool(segment.get("audible"))
    )
    domain_durations: dict[str, float] = {}
    for segment, duration in browser_rows:
        domain = str(segment.get("domain", ""))
        domain_durations[domain] = domain_durations.get(domain, 0.0) + duration

    keypress_count = sum(
        max(0, int(bucket.get("keypress_count", 0))) for bucket in interaction_buckets
    )
    mouse_click_count = sum(
        max(0, int(bucket.get("mouse_click_count", 0))) for bucket in interaction_buckets
    )
    scroll_delta = sum(
        abs(int(bucket.get("scroll_delta", 0))) for bucket in interaction_buckets
    )
    mouse_distance_px = sum(
        max(0.0, float(bucket.get("mouse_distance_px", 0.0)))
        for bucket in interaction_buckets
    )
    input_active_s = sum(
        max(0.0, float(bucket.get("input_active_s", 0.0)))
        for bucket in interaction_buckets
    )
    burst_count = sum(
        max(0, int(bucket.get("interaction_burst_count", 0)))
        for bucket in interaction_buckets
    )
    interval_mean, interval_std, interval_cv = _interaction_interval_stats(
        interaction_buckets
    )
    hour_angle = 2.0 * math.pi * (
        window_start.hour + window_start.minute / 60.0
    ) / 24.0
    weekday_angle = 2.0 * math.pi * window_start.weekday() / 7.0

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "app_switch_count": int(app_switch_count),
        "domain_switch_count": int(domain_switch_count),
        "longest_segment_s": round(longest_segment_s, 4),
        "longest_segment_ratio": _ratio(longest_segment_s, window_seconds),
        "idle_ratio": _ratio(idle_seconds, max(total_event_seconds, 0.01)),
        "keypress_rate_per_min": round(keypress_count / window_minutes, 6),
        "mouse_click_rate_per_min": round(mouse_click_count / window_minutes, 6),
        "scroll_rate_per_min": round(scroll_delta / window_minutes, 6),
        "mouse_distance_per_min": round(mouse_distance_px / window_minutes, 6),
        "input_active_ratio": _ratio(input_active_s, window_seconds),
        "interaction_bursts_per_min": round(burst_count / window_minutes, 6),
        "click_key_ratio": round(mouse_click_count / max(keypress_count, 1), 6),
        "browser_ratio": _ratio(browser_seconds, window_seconds),
        "audible_browser_ratio": _ratio(audible_seconds, max(browser_seconds, 0.01)),
        "active_seconds_ratio": _ratio(active_seconds, window_seconds),
        "top_app_ratio": _top_ratio(app_durations, max(active_seconds, 0.01)),
        "top_domain_ratio": _top_ratio(domain_durations, max(browser_seconds, 0.01)),
        "interaction_interval_mean_s": round(interval_mean, 6),
        "interaction_interval_std_s": round(interval_std, 6),
        "interaction_interval_cv": round(interval_cv, 6),
        "hour_of_day": window_start.hour,
        "day_of_week": window_start.weekday(),
        "hour_sin": round((math.sin(hour_angle) + 1.0) / 2.0, 6),
        "hour_cos": round((math.cos(hour_angle) + 1.0) / 2.0, 6),
        "weekday_sin": round((math.sin(weekday_angle) + 1.0) / 2.0, 6),
        "weekday_cos": round((math.cos(weekday_angle) + 1.0) / 2.0, 6),
        "task_type_code": 0.0,
    }


def _event_overlap_seconds(
    event: ActivityEvent,
    window_start: datetime,
    window_end: datetime,
) -> float:
    event_start = event.timestamp_utc
    event_end = event_start + timedelta(seconds=max(0.0, event.duration_s))
    return _overlap_seconds(event_start, event_end, window_start, window_end)


def _segment_overlap_seconds(
    segment: dict[str, Any],
    window_start: datetime,
    window_end: datetime,
) -> float:
    duration = max(0.0, float(segment.get("duration_s", 0.0)))
    timestamp = segment.get("timestamp")
    if not timestamp:
        return min(duration, max(0.0, (window_end - window_start).total_seconds()))
    try:
        segment_start = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if segment_start.tzinfo is None:
        segment_start = segment_start.replace(tzinfo=UTC)
    segment_start = segment_start.astimezone(UTC)
    segment_end = segment_start + timedelta(seconds=duration)
    return _overlap_seconds(segment_start, segment_end, window_start, window_end)


def _overlap_seconds(
    first_start: datetime,
    first_end: datetime,
    second_start: datetime,
    second_end: datetime,
) -> float:
    return max(0.0, (min(first_end, second_end) - max(first_start, second_start)).total_seconds())


def _interaction_interval_stats(
    buckets: list[dict[str, Any]],
) -> tuple[float, float, float]:
    timestamps: list[datetime] = []
    for bucket in buckets:
        if not _has_interaction(bucket):
            continue
        raw_timestamp = bucket.get("window_start_utc")
        if not raw_timestamp:
            continue
        try:
            timestamp = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
        except ValueError:
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        timestamps.append(timestamp.astimezone(UTC))
    timestamps.sort()
    if len(timestamps) < 2:
        return 0.0, 0.0, 0.0
    intervals = np.diff([timestamp.timestamp() for timestamp in timestamps])
    mean = float(np.mean(intervals))
    std = float(np.std(intervals))
    return mean, std, std / mean if mean > 0 else 0.0


def _has_interaction(bucket: dict[str, Any]) -> bool:
    return any(
        float(bucket.get(name, 0.0)) != 0.0
        for name in (
            "keypress_count",
            "mouse_click_count",
            "scroll_delta",
            "mouse_distance_px",
            "input_active_s",
            "interaction_burst_count",
        )
    )


def _ratio(numerator: float, denominator: float) -> float:
    return round(min(max(numerator / denominator, 0.0), 1.0), 6)


def _top_ratio(durations: dict[str, float], total: float) -> float:
    return _ratio(max(durations.values(), default=0.0), total)
