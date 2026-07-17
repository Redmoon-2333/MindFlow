"""Behavior summary builder: ActivityEvent list → BehaviorSummary.

This is the context-engineering step of the LLM pipeline (llm-cbt.md §1).
Raw event data is aggregated into a compact JSON summary before reaching
any LLM, achieving two goals:

  1. **Token efficiency** — a day of 5-second ticks (~17k events) compresses
     into one 500-byte JSON summary.
  2. **Privacy (NF-S3a)** — window titles and file paths are NEVER included
     in the summary. Only objective metrics (durations, counts, ratios) and
     anonymous labels (app categories) are passed to the LLM.

All computation reuses pure functions from ``domain/features``.
"""

from __future__ import annotations

import json
from typing import Any

from mindflow.domain.events import ActivityEvent
from mindflow.domain.features import (
    longest_focus_block_s,
    switch_rate_per_hour,
    title_features,
)
from mindflow.domain.procrastination import BehaviorSummary

# ── App categories for social-media detection ──────────────────────────────────

_ENTERTAINMENT_APPS: frozenset[str] = frozenset({
    "wechat",
    "微信",
    "qq",
    "tim",
    "chrome.exe",
    "firefox.exe",
    "msedge.exe",
    "bilibili",
    "哔哩哔哩",
    "douyin",
    "抖音",
    "tiktok",
    "weibo",
    "微博",
    "zhihu",
    "知乎",
    "xiaohongshu",
    "小红书",
    "kuaishou",
    "快手",
    "netflix",
    "youtube",
    "spotify",
    "netease",
    "网易云音乐",
})

_FOCUS_APPS: frozenset[str] = frozenset({
    "code.exe",
    "cursor.exe",
    "windsurf.exe",
    "vscode.exe",
    "code",
    "intellij",
    "pycharm",
    "webstorm",
    "clion",
    "goland",
    "sublime_text.exe",
    "notepad++.exe",
    "vim",
    "neovim",
    "terminal",
    "windows terminal",
    "powershell",
    "cmd.exe",
    "wsl",
})

_PERFECTIONISM_PATTERNS: frozenset[str] = frozenset({
    "不够好",
    "失败",
    "重来",
    "做不好",
    "不够完美",
    "重新开始",
    "又错了",
    "太差了",
})


# ── Public API ─────────────────────────────────────────────────────────────────


def build_behavior_summary(
    events: list[ActivityEvent],
    baseline_deviation: float | None = None,
) -> BehaviorSummary:
    """Aggregate activity events into a BehaviorSummary for LLM analysis.

    Args:
        events: Activity events for the analysis window (typically one day).
            Events should be ordered by time (will be sorted if not).
        baseline_deviation: Optional Z-score deviation from the user's
            baseline focus pattern.  Pass None if baseline is not available.

    Returns:
        A ``BehaviorSummary`` ready for LLM consumption or rule-engine analysis.

    Raises:
        ValueError: If *events* is empty.
    """
    if not events:
        raise ValueError("Cannot build BehaviorSummary from empty event list")

    sorted_events = sorted(events, key=lambda e: e.timestamp_utc)

    # Core metrics from domain/features
    switches_per_hour = switch_rate_per_hour(sorted_events)
    longest_block_s = longest_focus_block_s(sorted_events)

    # Total duration
    first_ts = sorted_events[0].timestamp_utc
    last_ts = sorted_events[-1].timestamp_utc
    total_duration_s = (last_ts - first_ts).total_seconds()
    total_duration_min = total_duration_s / 60.0 if total_duration_s > 0 else 1.0

    # Social media / entertainment ratio
    entertainment_duration_s = _entertainment_duration(sorted_events)
    social_media_ratio = (
        entertainment_duration_s / total_duration_s if total_duration_s > 0 else 0.0
    )

    # Focus duration estimation
    non_idle_events = [e for e in sorted_events if not e.data.is_idle]
    total_non_idle_s = sum(e.duration_s for e in non_idle_events)
    actual_focus_min = _estimate_focus_minutes(non_idle_events, total_non_idle_s)

    # Keyword flags from window titles
    keyword_flags = _extract_keyword_flags(sorted_events)

    # Start delay: time from first event to first non-entertainment activity
    start_delay_min = _estimate_start_delay(sorted_events)

    # Intended task from manual_tag events
    intended_task = _find_intended_task(sorted_events)

    return BehaviorSummary(
        intended_task=intended_task,
        duration_min=round(total_duration_min, 1),
        actual_focus_min=round(actual_focus_min, 1),
        context_switches_per_hour=round(switches_per_hour, 1),
        longest_focus_block_s=round(longest_block_s, 1),
        social_media_ratio=round(social_media_ratio, 3),
        start_delay_min=round(start_delay_min, 1),
        keyword_flags=keyword_flags,
        baseline_deviation=baseline_deviation,
    )


def serialize_summary(summary: BehaviorSummary) -> str:
    """Serialize a BehaviorSummary to a JSON string for LLM input.

    The JSON structure follows the schema defined in llm-cbt.md §1:

    .. code-block:: json

        {
          "session": {"intended_task": "...", "duration_min": 120, ...},
          "metrics": {"context_switches_per_hour": 18, ...},
          "pattern_summary": "..."
        }

    Args:
        summary: The behavior summary to serialize.

    Returns:
        A JSON string with no PII (NF-S3a).
    """
    pattern_parts: list[str] = []
    if summary.context_switches_per_hour > 12:
        pattern_parts.append(f"切换频繁({summary.context_switches_per_hour:.0f}次/小时)")
    if summary.longest_focus_block_s < 300:
        pattern_parts.append("专注时间短")
    if summary.social_media_ratio > 0.4:
        pattern_parts.append("社交媒体占比高")
    if summary.start_delay_min > 10:
        pattern_parts.append("启动延迟明显")
    pattern_summary = "；".join(pattern_parts) if pattern_parts else "无显著行为模式异常"

    data: dict[str, Any] = {
        "session": {
            "intended_task": summary.intended_task or "未设定",
            "duration_min": summary.duration_min,
            "actual_focus_min": summary.actual_focus_min,
        },
        "metrics": {
            "context_switches_per_hour": summary.context_switches_per_hour,
            "longest_focus_block_sec": summary.longest_focus_block_s,
            "social_media_ratio": summary.social_media_ratio,
            "start_delay_min": summary.start_delay_min,
        },
        "pattern_summary": pattern_summary,
    }
    if summary.baseline_deviation is not None:
        data["baseline_deviation"] = {"focus_vs_typical": summary.baseline_deviation}

    return json.dumps(data, ensure_ascii=False)


# ── Internal helpers ───────────────────────────────────────────────────────────


def _entertainment_duration(events: list[ActivityEvent]) -> float:
    """Compute total seconds spent on entertainment/social-media apps."""
    total = 0.0
    for ev in events:
        if ev.data.is_idle:
            continue
        app_lower = ev.data.process_name.lower()
        if app_lower in _ENTERTAINMENT_APPS:
            total += ev.duration_s
            continue
        # Check window title for entertainment patterns
        features = title_features(ev.data.window_title)
        if features.is_likely_entertainment:
            total += ev.duration_s
    return total


def _estimate_focus_minutes(events: list[ActivityEvent], total_non_idle_s: float) -> float:
    """Estimate actual focus minutes from the event stream.

    Focus is estimated as time spent in non-entertainment, non-idle apps
    multiplied by a heuristic focus-quality factor based on switch rate.
    """
    if not events or total_non_idle_s <= 0:
        return 0.0

    entertainment_s = _entertainment_duration(events)
    focus_candidate_s = max(0.0, total_non_idle_s - entertainment_s)

    # Apply a penalty based on switch frequency
    rate = switch_rate_per_hour(events)
    if rate > 30:
        factor = 0.5
    elif rate > 15:
        factor = 0.7
    else:
        factor = 0.9

    return (focus_candidate_s * factor) / 60.0


def _extract_keyword_flags(events: list[ActivityEvent]) -> frozenset[str]:
    """Extract perfectionism-related keyword flags from window titles."""
    flags: set[str] = set()
    for ev in events:
        if ev.data.is_idle:
            continue
        title_lower = ev.data.window_title.lower()
        for pattern in _PERFECTIONISM_PATTERNS:
            if pattern in title_lower:
                flags.add("self_criticism")
            # Check for redo patterns
            if "重新" in title_lower and ("开始" in title_lower or "写" in title_lower):
                flags.add("redo_pattern")
    return frozenset(flags)


def _estimate_start_delay(events: list[ActivityEvent]) -> float:
    """Estimate how long the user spent before starting productive work.

    The "start" is defined as the first non-entertainment, non-idle event
    after the session begins.  The delay is the time between the first event
    and that start event.
    """
    if not events:
        return 0.0

    first_ts = events[0].timestamp_utc

    for ev in events:
        if ev.data.is_idle:
            continue
        app_lower = ev.data.process_name.lower()
        is_entertainment = title_features(ev.data.window_title).is_likely_entertainment
        if app_lower not in _ENTERTAINMENT_APPS and not is_entertainment:
            delay_s = (ev.timestamp_utc - first_ts).total_seconds()
            return max(0.0, delay_s / 60.0)

    return 0.0


def _find_intended_task(events: list[ActivityEvent]) -> str | None:
    """Find the intended task from manual_tag events."""
    for ev in events:
        if ev.event_type == "manual_tag" and ev.data.window_title.strip():
            return ev.data.window_title.strip()
    return None
