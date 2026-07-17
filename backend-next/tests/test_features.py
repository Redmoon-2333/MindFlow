"""Tests for mindflow.domain.features — feature-computation functions.

Tests cover:
  - Known inputs produce expected outputs (regression against old code)
  - Edge cases: empty list, all idle, single event
  - Hypothesis property test: focus_score is always in [0, 100]
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mindflow.domain.events import ActivityEvent, make_event
from mindflow.domain.features import (
    AppUsage,
    TitleFeatures,
    app_usage_ranking,
    focus_score,
    longest_focus_block_s,
    switch_rate_per_hour,
    title_features,
)

# ── Hypothesis strategies ────────────────────────────────────────────────────


def _aware_dt(year: int = 2026, month: int = 7, day: int = 17) -> datetime:
    """Convenience: timeline-aware UTC datetime builder."""
    return datetime(year, month, day, tzinfo=UTC)


def _ts(offset_minutes: int = 0) -> datetime:
    """Base timestamp + offset in minutes."""
    return _aware_dt() + timedelta(minutes=offset_minutes)


# Hypothesis composite strategy for generating lists of events
# Hypothesis datetimes strategy requires naive bounds + separate timezones kwarg.
_MIN_DATE = datetime(2020, 1, 1)
_MAX_DATE = datetime(2030, 1, 1)


@st.composite
def event_lists(draw: st.DrawFn) -> list[ActivityEvent]:
    """Generate a list of 0-30 ActivityEvents with random parameters."""
    n = draw(st.integers(min_value=0, max_value=30))

    # Generate one base datetime
    base = draw(
        st.datetimes(
            min_value=_MIN_DATE,
            max_value=_MAX_DATE,
            timezones=st.just(UTC),
        )
    )
    events: list[ActivityEvent] = []
    for i in range(n):
        ts = base + timedelta(seconds=i * draw(st.integers(1, 60)))
        ev = make_event(
            user_id=1,
            timestamp_utc=ts,
            duration_s=draw(
                st.floats(min_value=0.0, max_value=3600.0, allow_nan=False, allow_infinity=False)
            ),
            event_type="window_snapshot",
            app_name=draw(st.sampled_from(["Code", "Chrome", "Terminal", "Slack", "Spotify"])),
            window_title="",
            process_name=draw(
                st.sampled_from(
                    ["code.exe", "chrome.exe", "terminal.exe", "slack.exe", "spotify.exe"]
                )
            ),
            is_idle=draw(st.booleans()),
        )
        events.append(ev)

    return events


# ── focus_score ──────────────────────────────────────────────────────────────


class TestFocusScore:
    """Regression tests against old ``calculate_focus_score`` behaviour."""

    def test_fewer_than_min_threshold_returns_zero(self):
        """Fewer than MIN_ACTIVITY_THRESHOLD (10) non-idle events -> 0.0."""
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(i),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=False,
            )
            for i in range(5)
        ]
        assert focus_score(events) == 0.0

    def test_all_idle_returns_zero(self):
        """All idle events -> 0.0 (no non-idle events to score)."""
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(i),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=True,
            )
            for i in range(10)
        ]
        assert focus_score(events) == 0.0

    def test_empty_list_returns_zero(self):
        """Empty list -> 0.0."""
        assert focus_score([]) == 0.0

    def test_single_app_perfect_focus(self):
        """Single app, no switches -> high score (top_app_ratio ~1, switch_penalty 0)."""
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(i),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=False,
            )
            for i in range(11)
        ]
        score = focus_score(events)
        # top_app_ratio = 1.0, switch_freq = 0 -> (1.0 * 60) + (1.0 * 40) = 100
        assert score == 100.0

    def test_two_apps_equal_time(self):
        """Two apps with equal time and a single switch -> score around 62.7."""
        events: list[ActivityEvent] = []
        for i in range(6):
            events.append(
                make_event(
                    user_id=1,
                    timestamp_utc=_ts(i),
                    duration_s=5.0,
                    process_name="code.exe",
                    is_idle=False,
                )
            )
        for i in range(6, 12):
            events.append(
                make_event(
                    user_id=1,
                    timestamp_utc=_ts(i),
                    duration_s=5.0,
                    process_name="chrome.exe",
                    is_idle=False,
                )
            )
        score = focus_score(events)
        # top_app_ratio = 0.5, 1 switch in 11 min -> switch_freq ~5.45/h
        # (0.5 * 60) + (1 - 5.45/30) * 40 = 30 + 32.73 = 62.73
        assert score == pytest.approx(62.7, abs=0.5)

    def test_custom_weights(self):
        """Custom weights are reflected in the output."""
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(i),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=False,
            )
            for i in range(11)
        ]
        # top_app_weight=100, switch_weight=0 — score should be top_app_ratio * 100
        score = focus_score(events, weights={"top_app_weight": 100.0, "switch_weight": 0.0})
        assert score == 100.0

    def test_frequent_switching_reduces_score(self):
        """High switch frequency reduces the score."""
        # Alternating apps every event = many switches
        events: list[ActivityEvent] = []
        for i in range(11):
            proc = "code.exe" if i % 2 == 0 else "chrome.exe"
            events.append(
                make_event(
                    user_id=1,
                    timestamp_utc=_ts(i),
                    duration_s=5.0,
                    process_name=proc,
                    is_idle=False,
                )
            )
        score = focus_score(events)
        assert 0.0 < score < 100.0  # Something in between

    # ── Hypothesis property test ──

    @given(events=event_lists())
    def test_score_in_range(self, events: list[ActivityEvent]):
        """focus_score is always in [0, 100], regardless of input."""
        score = focus_score(events)
        assert 0.0 <= score <= 100.0

    @given(events=event_lists())
    def test_score_in_range_custom_weights(self, events: list[ActivityEvent]):
        """focus_score stays in [0, 100] even with extreme custom weights."""
        extreme_weights = {"top_app_weight": 200.0, "switch_weight": -100.0}
        score = focus_score(events, weights=extreme_weights)
        assert 0.0 <= score <= 100.0


# ── switch_rate_per_hour ────────────────────────────────────────────────────


class TestSwitchRatePerHour:
    """Regression tests against old ``calculate_switch_frequency``."""

    def test_empty_returns_zero(self):
        assert switch_rate_per_hour([]) == 0.0

    def test_single_event_returns_zero(self):
        ev = make_event(
            user_id=1, timestamp_utc=_ts(), duration_s=5.0, process_name="code.exe", is_idle=False
        )
        assert switch_rate_per_hour([ev]) == 0.0

    def test_two_events_same_app_zero(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=False,
            ),
        ]
        assert switch_rate_per_hour(events) == 0.0

    def test_two_events_different_app(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=5.0,
                process_name="chrome.exe",
                is_idle=False,
            ),
        ]
        # 1 switch in 1 minute = 60 switches/hour
        rate = switch_rate_per_hour(events)
        assert rate == pytest.approx(60.0, rel=0.01)

    def test_idle_events_ignored(self):
        """Idle events don't count as switches."""
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=True,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(2),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=False,
            ),
        ]
        assert switch_rate_per_hour(events) == 0.0

    def test_zero_time_span_returns_zero(self):
        """Events with identical timestamps -> 0 (no time span)."""
        ts = _ts()
        events = [
            make_event(
                user_id=1, timestamp_utc=ts, duration_s=5.0, process_name="code.exe", is_idle=False
            ),
            make_event(
                user_id=1,
                timestamp_utc=ts,
                duration_s=5.0,
                process_name="chrome.exe",
                is_idle=False,
            ),
        ]
        assert switch_rate_per_hour(events) == 0.0


# ── longest_focus_block_s ────────────────────────────────────────────────────


class TestLongestFocusBlock:
    """Focus block detection."""

    def test_empty_returns_zero(self):
        assert longest_focus_block_s([]) == 0.0

    def test_single_event(self):
        ev = make_event(
            user_id=1, timestamp_utc=_ts(), duration_s=10.0, process_name="code.exe", is_idle=False
        )
        assert longest_focus_block_s([ev]) == 10.0

    def test_continuous_same_app(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=10.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=20.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(2),
                duration_s=30.0,
                process_name="code.exe",
                is_idle=False,
            ),
        ]
        assert longest_focus_block_s(events) == 60.0  # 10 + 20 + 30

    def test_idle_breaks_block(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=10.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=5.0,
                process_name="code.exe",
                is_idle=True,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(2),
                duration_s=20.0,
                process_name="code.exe",
                is_idle=False,
            ),
        ]
        assert longest_focus_block_s(events) == 20.0  # block after idle

    def test_app_switch_breaks_block(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=10.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=20.0,
                process_name="chrome.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(2),
                duration_s=30.0,
                process_name="chrome.exe",
                is_idle=False,
            ),
        ]
        assert longest_focus_block_s(events) == 50.0  # chrome block: 20 + 30

    def test_multiple_blocks_finds_longest(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=10.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=5.0,
                process_name="chrome.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(2),
                duration_s=5.0,
                process_name="chrome.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(3),
                duration_s=50.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(4),
                duration_s=50.0,
                process_name="code.exe",
                is_idle=False,
            ),
        ]
        assert longest_focus_block_s(events) == 100.0  # code block at end: 50 + 50

    def test_all_idle_returns_zero(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=10.0,
                process_name="code.exe",
                is_idle=True,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=20.0,
                process_name="chrome.exe",
                is_idle=True,
            ),
        ]
        assert longest_focus_block_s(events) == 0.0


# ── app_usage_ranking ────────────────────────────────────────────────────────


class TestAppUsageRanking:
    """App usage ranking."""

    def test_empty_returns_empty(self):
        assert app_usage_ranking([]) == []

    def test_single_app(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(),
                duration_s=10.0,
                process_name="code.exe",
                is_idle=False,
            ),
        ]
        ranking = app_usage_ranking(events)
        assert ranking == [AppUsage(app_name="code.exe", total_duration_s=10.0)]

    def test_multiple_apps_sorted(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=10.0,
                process_name="chrome.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=30.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(2),
                duration_s=20.0,
                process_name="spotify.exe",
                is_idle=False,
            ),
        ]
        ranking = app_usage_ranking(events)
        assert ranking == [
            AppUsage(app_name="code.exe", total_duration_s=30.0),
            AppUsage(app_name="spotify.exe", total_duration_s=20.0),
            AppUsage(app_name="chrome.exe", total_duration_s=10.0),
        ]

    def test_idle_events_excluded(self):
        events = [
            make_event(
                user_id=1,
                timestamp_utc=_ts(0),
                duration_s=10.0,
                process_name="code.exe",
                is_idle=False,
            ),
            make_event(
                user_id=1,
                timestamp_utc=_ts(1),
                duration_s=50.0,
                process_name="code.exe",
                is_idle=True,
            ),
        ]
        ranking = app_usage_ranking(events)
        assert ranking == [AppUsage(app_name="code.exe", total_duration_s=10.0)]


# ── title_features ──────────────────────────────────────────────────────────


class TestTitleFeatures:
    """Title feature extraction (ported from TitleAnalyzer.analyze())."""

    def test_empty_title(self):
        assert title_features("") == TitleFeatures()

    def test_whitespace_title(self):
        assert title_features("   ") == TitleFeatures()

    def test_browser_url_detected(self):
        result = title_features("GitHub - https://github.com/user/repo")
        assert result.is_browser is True
        assert result.url_domain == "github.com"

    def test_browser_url_without_scheme(self):
        result = title_features("www.example.com/page")
        assert result.is_browser is True
        assert result.url_domain == "example.com"

    def test_code_extension_python(self):
        result = title_features("main.py - VS Code")
        assert result.is_code_editor is True
        assert result.file_extension in {".py", ".ipynb"}

    def test_document_extension(self):
        result = title_features("report.pdf - Adobe Acrobat")
        assert result.is_document is True
        assert result.file_extension == ".pdf"

    def test_code_takes_precedence_over_document(self):
        """If a title matches both code and doc extensions, code wins."""
        result = title_features("main.py.pdf")
        # ".py" is checked first
        assert result.is_code_editor is True

    def test_meeting_keyword(self):
        result = title_features("Zoom Meeting - Team Standup")
        assert result.is_meeting is True

    def test_entertainment_anime(self):
        result = title_features("Watch Episode 12 - Anime Site")
        assert result.is_likely_entertainment is True

    def test_entertainment_bilibili(self):
        result = title_features("[Bilibili] 番剧推荐")
        assert result.is_likely_entertainment is True

    def test_no_features_for_unknown_title(self):
        result = title_features("Untitled - Notepad")
        assert result == TitleFeatures()
