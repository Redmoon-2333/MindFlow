"""Per-window data-quality record: separate "zero" from "not observed".

A window built while the browser collector was off has ``browser_ratio == 0``
for the same reason a window with no browsing has it: the feature cannot tell
the two apart. On the real snapshot this mattered — ``browser_segments`` is
empty for every stored window, so three of the 24 V3 features are structurally
constant zero, and the model reads that constant as behaviour.

This module defines the quality record attached to each window and the helper
that decides whether a feature is *observed*. It does not change the V3
feature vector: the record travels alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

#: Feature-name → the data source that must be available for it to be
#: meaningful. Features not listed are always derivable from activity events.
FEATURE_SOURCE: dict[str, str] = {
    "browser_ratio": "browser",
    "audible_browser_ratio": "browser",
    "top_domain_ratio": "browser",
    "domain_switch_count": "browser",
    "keypress_rate_per_min": "input",
    "mouse_click_rate_per_min": "input",
    "scroll_rate_per_min": "input",
    "mouse_distance_per_min": "input",
    "input_active_ratio": "input",
    "interaction_bursts_per_min": "input",
    "click_key_ratio": "input",
    "interaction_interval_mean_s": "input",
    "interaction_interval_std_s": "input",
    "interaction_interval_cv": "input",
}

SOURCE_ACTIVITY = "activity"
SOURCE_BROWSER = "browser"
SOURCE_INPUT = "input"

ALL_SOURCES: tuple[str, ...] = (SOURCE_ACTIVITY, SOURCE_BROWSER, SOURCE_INPUT)


@dataclass
class WindowQuality:
    """What was actually observed while building one feature window.

    Attributes:
        window_start: window start (UTC).
        window_end: window end (UTC).
        sources_enabled: known switch state, or None for unknown history.
        sources_available: delivery evidence, or None when unknown.
        observed_seconds: per source, seconds of data seen in the window.
        coverage: per source, ``observed_seconds / window_seconds`` (0-1).
        uncovered_seconds: per source, uncovered payload seconds (not downtime).
        gap_seconds: known collection downtime, None without heartbeat evidence.
        recovered: per source, whether the collector recovered mid-window.
        event_count: activity events overlapping the window.
        bucket_count: interaction buckets overlapping the window.
        browser_segment_count: browser segments overlapping the window.
    """

    window_start: datetime
    window_end: datetime
    sources_enabled: dict[str, bool | None] = field(default_factory=dict)
    sources_available: dict[str, bool | None] = field(default_factory=dict)
    observed_seconds: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)
    uncovered_seconds: dict[str, float] = field(default_factory=dict)
    gap_seconds: dict[str, float | None] = field(default_factory=dict)
    recovered: dict[str, bool] = field(default_factory=dict)
    event_count: int = 0
    bucket_count: int = 0
    browser_segment_count: int = 0

    @property
    def window_seconds(self) -> float:
        return max(0.0, (self.window_end - self.window_start).total_seconds())

    def is_observed(self, source: str) -> bool:
        """True when a source has positive observation evidence.

        A source that was disabled is never "observed" — its zero is an
        absence of measurement, not a measurement of zero. This is not a
        full-coverage claim: consult ``coverage`` for the measured fraction.
        """
        return bool(self.sources_enabled.get(source)) and bool(
            self.sources_available.get(source)
        )

    def feature_observed(self, feature_name: str) -> bool:
        """Whether *feature_name* had a real chance to be non-zero."""
        source = FEATURE_SOURCE.get(feature_name)
        if source is None:
            return self.is_observed(SOURCE_ACTIVITY)
        return self.is_observed(source)

    @property
    def quality_tier(self) -> str:
        """Coarse label used in reports and coverage counts.

        ``full`` — activity observed and every *enabled* auxiliary source
        observed too.
        ``partial`` — activity observed, but an enabled source delivered
        nothing.
        ``activity_only`` — only activity has positive observation evidence.
        ``empty`` — nothing observed; the feature values are not usable.
        """
        if not any(self.is_observed(source) for source in ALL_SOURCES):
            return "empty"
        if not self.is_observed(SOURCE_ACTIVITY):
            return "partial"

        enabled_aux = [
            src for src in (SOURCE_BROWSER, SOURCE_INPUT)
            if self.sources_enabled.get(src)
        ]
        if not enabled_aux:
            return "activity_only"
        missing = [src for src in enabled_aux if not self.is_observed(src)]
        return "partial" if missing else "full"

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start_utc": self.window_start.isoformat(),
            "window_end_utc": self.window_end.isoformat(),
            "sources_enabled": dict(self.sources_enabled),
            "sources_available": dict(self.sources_available),
            "observed_seconds": {
                k: round(v, 3) for k, v in self.observed_seconds.items()
            },
            "coverage": {k: round(v, 4) for k, v in self.coverage.items()},
            "uncovered_seconds": {k: round(v, 3) for k, v in self.uncovered_seconds.items()},
            "gap_seconds": {
                k: round(v, 3) if v is not None else None for k, v in self.gap_seconds.items()
            },
            "recovered": dict(self.recovered),
            "event_count": self.event_count,
            "bucket_count": self.bucket_count,
            "browser_segment_count": self.browser_segment_count,
            "quality_tier": self.quality_tier,
        }


def build_window_quality(
    *,
    window_start: datetime,
    window_end: datetime,
    events: list[Any],
    interaction_buckets: list[dict[str, Any]],
    browser_segments: list[dict[str, Any]],
    enabled: dict[str, bool | None] | None = None,
    available: dict[str, bool | None] | None = None,
    recovered: dict[str, bool] | None = None,
) -> WindowQuality:
    """Derive a :class:`WindowQuality` from the inputs of one window build.

    ``enabled``/``available`` describe the *collectors*; the counts come from
    the data itself. Coverage is computed against the window duration, so a
    window that is half unobserved and half collected reports 0.5 for that source
    rather than rounding to "has data".
    """
    window_seconds = max(0.0, (window_end - window_start).total_seconds())
    source_enabled = dict(enabled or {})
    source_available = dict(available or {})

    # Payload coverage is not collector uptime. Clip to this window and union
    # intervals so long events and duplicate deliveries cannot inflate it.
    observed = {
        SOURCE_ACTIVITY: _covered_seconds([
            (getattr(event, "timestamp_utc", None), None, getattr(event, "duration_s", 0))
            for event in events
        ], window_start, window_end),
        SOURCE_INPUT: _covered_seconds([
            (bucket.get("window_start_utc"), bucket.get("window_end_utc"),
             bucket.get("duration_s", 0))
            for bucket in interaction_buckets
        ], window_start, window_end),
        SOURCE_BROWSER: _covered_seconds([
            (segment.get("timestamp"), None, segment.get("duration_s", 0))
            for segment in browser_segments
        ], window_start, window_end),
    }
    coverage = {
        src: (seconds / window_seconds if window_seconds > 0 else 0.0)
        for src, seconds in observed.items()
    }
    uncovered = {
        src: max(0.0, window_seconds - seconds) for src, seconds in observed.items()
    }

    # Missing historical evidence is unknown, not proof that a switch was off.
    # An explicitly supplied False still wins over stale/contradictory payloads.
    for src in ALL_SOURCES:
        if source_available.get(src) is None:
            source_available[src] = True if observed[src] > 0 else None
        if source_enabled.get(src) is None:
            source_enabled[src] = True if observed[src] > 0 else None

    return WindowQuality(
        window_start=window_start,
        window_end=window_end,
        sources_enabled=source_enabled,
        sources_available=source_available,
        observed_seconds=observed,
        coverage=coverage,
        uncovered_seconds=uncovered,
        gap_seconds={
            src: 0.0 if source_enabled.get(src) is False else None for src in ALL_SOURCES
        },
        recovered=dict(recovered or {}),
        event_count=len(events or []),
        bucket_count=len(interaction_buckets or []),
        browser_segment_count=len(browser_segments or []),
    )


def _covered_seconds(
    rows: list[tuple[Any, Any, Any]], start: datetime, end: datetime,
) -> float:
    intervals: list[tuple[datetime, datetime]] = []
    for raw_start, raw_end, raw_duration in rows:
        try:
            if raw_start is None:
                continue
            left = _utc_timestamp(raw_start)
            if raw_end is not None:
                right = _utc_timestamp(raw_end)
            else:
                duration = float(raw_duration or 0)
                if not isfinite(duration):
                    continue
                right = left + timedelta(seconds=max(0.0, duration))
            left, right = max(left, start), min(right, end)
            if right > left:
                intervals.append((left, right))
        except (TypeError, ValueError, OverflowError):
            continue
    total = 0.0
    cursor = start
    for left, right in sorted(intervals):
        left = max(left, cursor)
        if right > left:
            total += (right - left).total_seconds()
            cursor = right
    return total


def _utc_timestamp(value: Any) -> datetime:
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


__all__ = [
    "ALL_SOURCES",
    "FEATURE_SOURCE",
    "SOURCE_ACTIVITY",
    "SOURCE_BROWSER",
    "SOURCE_INPUT",
    "WindowQuality",
    "build_window_quality",
]
