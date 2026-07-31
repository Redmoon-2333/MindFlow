"""Pure feature-computation functions for activity-event analysis.

All functions are stateless and operate on lists of ActivityEvents — no
framework or I/O dependencies.

Ported from ``backend/mindflow/analyzer/features.py`` and
``backend/mindflow/analyzer/title_analyzer.py`` with these adaptations:
  - DB queries replaced by in-memory event iteration.
  - str-based event filtering replaces SQL WHERE.
  - TitleFeatures returned as a frozen dataclass instead of a plain dict.
  - Weights exposed as overridable parameters (old code used module-level
    globals).
  - URL/file-extension/meeting/entertainment heuristics preserved verbatim.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from mindflow.domain.events import ActivityEvent

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (preserved from old features.py)
# ═══════════════════════════════════════════════════════════════════════════════

MIN_ACTIVITY_THRESHOLD: int = 10
"""Legacy heartbeat-count threshold retained for compatibility."""

MIN_ACTIVITY_DURATION_S: float = 50.0
"""Minimum non-idle duration required to compute a meaningful score."""

MIN_SWITCH_SAMPLES: int = 2
"""Fewer events than this yields a switch rate of 0 (not enough data)."""

MAX_ACCEPTABLE_SWITCHES_PER_HOUR: float = 30.0
MAX_COLLECTION_GAP_S: float = 60.0

DEFAULT_SWITCH_MIN_DWELL_S: float = 10.0
"""Minimum time a new process must stay foreground before counting a switch."""

TRANSIENT_PROCESSES: frozenset[str] = frozenset({
    "explorer.exe",
    "ApplicationFrameHost.exe",
    "ShellHost.exe",
    "ShellExperienceHost.exe",
    "DesktopMgr64.exe",
    "SearchHost.exe",
    "TextInputHost.exe",
    "StartMenuExperienceHost.exe",
})
"""Switches beyond this threshold incur maximum penalty."""

DEFAULT_FOCUS_WEIGHTS: Mapping[str, float] = {
    "top_app_weight": 60.0,
    "switch_weight": 40.0,
}
"""Weight distribution for the two focus-score components (must sum to 100)."""

# ═══════════════════════════════════════════════════════════════════════════════
# AppUsage
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AppUsage:
    """Aggregated usage statistics for a single application."""

    app_name: str
    total_duration_s: float


# ═══════════════════════════════════════════════════════════════════════════════
# TitleFeatures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TitleFeatures:
    """Objective features extracted from a window title string.

    Ported from ``TitleAnalyzer.analyze()`` in the old codebase.  No app
    classification — purely pattern-based.
    """

    url_domain: str | None = None
    is_browser: bool = False
    is_code_editor: bool = False
    is_document: bool = False
    is_meeting: bool = False
    is_likely_entertainment: bool = False
    is_likely_productive_learning: bool = False
    file_extension: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# URL / file-extension / keyword patterns (ported from title_analyzer.py)
# ═══════════════════════════════════════════════════════════════════════════════

_URL_PATTERN = re.compile(
    r"(?:https?://|www\.|[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}/)[^\s]*",
    re.IGNORECASE,
)

_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".h",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".r",
        ".ipynb",
        ".sql",
        ".sh",
        ".bash",
        ".yml",
        ".yaml",
        ".toml",
        ".json",
        ".xml",
        ".html",
        ".css",
        ".scss",
        ".vue",
        ".svelte",
    }
)

_DOC_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".md",
        ".tex",
        ".txt",
        ".csv",
        ".rtf",
        ".odt",
    }
)

_MEETING_KEYWORDS: frozenset[str] = frozenset(
    {
        "zoom",
        "meet",
        "teams",
        "meeting",
        "会议",
        "腾讯会议",
        "dingtalk",
        "飞书",
        "feishu",
        "slack",
        "discord",
    }
)

_ENTERTAINMENT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"番剧|动漫|anime|episode\s*\d+",
        r"第\d+集|第\d+话",
        r"直播间|live\s*room|直播",
        r"短视频|short\s*video",
        r"steam\s*(library|store|community)",
        r"游戏|game\s*(play|store|library)",
    ]
)

_PRODUCTIVE_LEARNING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        # Chinese course/lecture detection
        r"第\d+[讲课节章]",            # 第3讲, 第5课
        r"高等数学|线性代数|概率论|离散数学|数据结构|操作系统|计算机网络",
        r"考研|四六级|托福|雅思|GRE|考公|考编",
        r"\b(课程|教程|入门|实战|进阶|精通)\b",
        r"慕课|mooc|公开课|网课|在线课程",
        r"笔记|讲义|课件|习题|作业|考试",
        r"[Bb][Vv]1[A-Za-z0-9]{9}",     # bilibili BV video ID
        r"大学物理|大学英语|复变函数|数理方程|模拟电子|数字电路",
        # English learning keywords
        r"\b(lecture|tutorial|course|lesson|workshop|seminar)\b",
        r"\b(learn|learning|study|practice|exercise)\b",
        r"\b(algorithm|programming|coding|computer science)\b",
    ]
)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature functions
# ═══════════════════════════════════════════════════════════════════════════════


def _non_idle_events(events: list[ActivityEvent]) -> list[ActivityEvent]:
    """Filter out idle events, preserving original order."""
    return [e for e in events if not e.data.is_idle]


def _sorted_events(events: list[ActivityEvent]) -> list[ActivityEvent]:
    """Return events sorted by timestamp (defensive copy)."""
    return sorted(events, key=lambda e: e.timestamp_utc)


def count_confirmed_switches(
    events: list[ActivityEvent],
    *,
    min_dwell_s: float = DEFAULT_SWITCH_MIN_DWELL_S,
    transient_processes: frozenset[str] = TRANSIENT_PROCESSES,
) -> int:
    """Count foreground switches that persist long enough to be meaningful.

    Short excursions (under *min_dwell_s*) and system-transient windows are
    ignored so rapid clicks inside one app cannot inflate the switch count.
    """
    active = _sorted_events(_non_idle_events(events))
    if len(active) < 2:
        return 0

    current: str | None = None
    current_dwell = 0.0
    candidate: str | None = None
    candidate_dwell = 0.0
    switches = 0

    for event in active:
        process = event.data.process_name
        if not process or process in transient_processes:
            continue
        duration = max(0.0, event.duration_s)
        if current is None:
            current = process
            current_dwell = duration
            continue
        if process == current:
            if candidate is not None and candidate_dwell >= min_dwell_s:
                switches += 1
                current = candidate
                current_dwell = candidate_dwell
                candidate = process
                candidate_dwell = duration
            else:
                candidate = None
                candidate_dwell = 0.0
                current_dwell += duration
        elif process == candidate:
            candidate_dwell += duration
            if candidate_dwell >= min_dwell_s:
                switches += 1
                current = candidate
                current_dwell = candidate_dwell
                candidate = None
                candidate_dwell = 0.0
        else:
            if candidate is not None and candidate_dwell >= min_dwell_s:
                switches += 1
                current = candidate
                current_dwell = candidate_dwell
            candidate = process
            candidate_dwell = duration

    if candidate is not None and candidate_dwell >= min_dwell_s:
        switches += 1
    return switches


def focus_score(
    events: list[ActivityEvent],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute a focus score in [0, 100] from a list of activity events.

    Uses the same two-factor formula as the old ``calculate_focus_score``:
      1. **Top-app ratio**: fraction of total time spent in the single most
         used application.
      2. **Switch penalty**: how many process-name changes occur per hour
         (capped at ``MAX_ACCEPTABLE_SWITCHES_PER_HOUR``).

    Args:
        events: Activity events (idle events are ignored).
        weights: Optional overrides for ``top_app_weight`` and
                 ``switch_weight``.  Must sum to 100.

    Returns:
        A float in [0, 100], or 0.0 when there are too few non-idle events.
    """
    w = weights if weights is not None else DEFAULT_FOCUS_WEIGHTS
    top_app_weight = w.get("top_app_weight", 60.0)
    switch_weight = w.get("switch_weight", 40.0)

    active = _non_idle_events(events)
    if sum(max(0.0, event.duration_s) for event in active) < MIN_ACTIVITY_DURATION_S:
        return 0.0

    # App durations
    app_durations: dict[str, float] = {}
    for ev in active:
        app_durations[ev.data.process_name] = (
            app_durations.get(ev.data.process_name, 0.0) + ev.duration_s
        )

    if not app_durations:
        return 0.0

    total_duration = sum(app_durations.values())
    top_app_ratio = max(app_durations.values()) / total_duration if total_duration > 0 else 0.0

    switch_freq = switch_rate_per_hour(active)
    switch_penalty = min(switch_freq / MAX_ACCEPTABLE_SWITCHES_PER_HOUR, 1.0)

    raw_score = (top_app_ratio * top_app_weight) + ((1.0 - switch_penalty) * switch_weight)
    return round(min(max(raw_score, 0.0), 100.0), 1)


def app_usage_ranking(
    events: list[ActivityEvent],
) -> list[AppUsage]:
    """Rank applications by total active duration, descending.

    Idle events are excluded.  Returns an empty list when there are no
    non-idle events.

    Ported from ``get_top_apps()`` in the old codebase (in-memory version).
    """
    app_durations: dict[str, float] = {}
    for ev in _non_idle_events(events):
        app_durations[ev.data.process_name] = (
            app_durations.get(ev.data.process_name, 0.0) + ev.duration_s
        )

    sorted_apps = sorted(
        app_durations.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    return [AppUsage(app_name=name, total_duration_s=dur) for name, dur in sorted_apps]


def switch_rate_per_hour(events: list[ActivityEvent]) -> float:
    """Compute confirmed process switches per observed collection hour."""
    active = _sorted_events(_non_idle_events(events))
    if len(active) < MIN_SWITCH_SAMPLES:
        return 0.0

    switches = count_confirmed_switches(active)
    observed_seconds = 0.0
    previous = active[0]
    for event in active[1:]:
        gap_s = (event.timestamp_utc - previous.timestamp_utc).total_seconds()
        if 0 < gap_s <= MAX_COLLECTION_GAP_S:
            observed_seconds += gap_s
        previous = event

    if observed_seconds <= 0:
        return 0.0

    return switches / (observed_seconds / 3600.0)


def longest_focus_block_s(events: list[ActivityEvent]) -> float:
    """Return the longest same-app block without idle or collection gaps."""
    sorted_events = _sorted_events(events)
    sorted_events = [
        event for event in _sorted_events(events)
        if event.data.process_name not in TRANSIENT_PROCESSES
    ]
    longest = 0.0
    current_block = 0.0
    current_app: str | None = None
    previous_timestamp = None

    for event in sorted_events:
        has_collection_gap = (
            previous_timestamp is not None
            and (event.timestamp_utc - previous_timestamp).total_seconds()
            > MAX_COLLECTION_GAP_S
        )
        if event.data.is_idle or has_collection_gap:
            longest = max(longest, current_block)
            current_block = 0.0
            current_app = None
        if not event.data.is_idle:
            if current_app is None or event.data.process_name != current_app:
                longest = max(longest, current_block)
                current_app = event.data.process_name
                current_block = max(0.0, event.duration_s)
            else:
                current_block += max(0.0, event.duration_s)
        previous_timestamp = event.timestamp_utc

    return max(longest, current_block)


def title_features(title: str) -> TitleFeatures:
    """Extract objective features from a window title string.

    Ported from ``TitleAnalyzer.analyze()``.  No app classification — purely
    pattern matching on URL schemes, file extensions, and structural keywords.
    """
    raw = title.strip() if title else ""

    if not raw:
        return TitleFeatures()

    # Build values dict incrementally, create TitleFeatures only once.
    vals: dict[str, object] = {}
    title_lower = raw.lower()

    # URL / browser detection
    url_match = _URL_PATTERN.search(raw)
    if url_match:
        vals["is_browser"] = True
        try:
            raw_url = (
                url_match.group() if "://" in url_match.group() else f"https://{url_match.group()}"
            )
            parsed = urlparse(raw_url)
            domain = parsed.netloc or parsed.path.split("/")[0]
            domain = domain.replace("www.", "").lower()
            vals["url_domain"] = domain
        except Exception:  # noqa: BLE001 — urlparse can raise on malformed URLs
            pass

    # Code file extensions
    for ext in _CODE_EXTENSIONS:
        ext_clean = ext.lstrip(".")
        if f".{ext_clean}" in title_lower:
            vals["file_extension"] = ext
            vals["is_code_editor"] = True
            break

    # Document extensions (only if not already code)
    if not vals.get("is_code_editor"):
        for ext in _DOC_EXTENSIONS:
            if ext in title_lower:
                vals["file_extension"] = ext
                vals["is_document"] = True
                break

    # Meeting / communication keywords
    if any(kw in title_lower for kw in _MEETING_KEYWORDS):
        vals["is_meeting"] = True

    # Entertainment patterns
    if any(p.search(raw) for p in _ENTERTAINMENT_PATTERNS):
        vals["is_likely_entertainment"] = True

    # Productive learning patterns (can coexist with entertainment patterns
    # — e.g., bilibili lecture matches both)
    if any(p.search(raw) for p in _PRODUCTIVE_LEARNING_PATTERNS):
        vals["is_likely_productive_learning"] = True

    return TitleFeatures(**vals)  # type: ignore[arg-type]
