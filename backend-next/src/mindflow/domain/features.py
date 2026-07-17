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
"""Minimum number of non-idle events required to compute a meaningful score."""

MIN_SWITCH_SAMPLES: int = 2
"""Fewer events than this yields a switch rate of 0 (not enough data)."""

MAX_ACCEPTABLE_SWITCHES_PER_HOUR: float = 30.0
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


# ═══════════════════════════════════════════════════════════════════════════════
# Feature functions
# ═══════════════════════════════════════════════════════════════════════════════


def _non_idle_events(events: list[ActivityEvent]) -> list[ActivityEvent]:
    """Filter out idle events, preserving original order."""
    return [e for e in events if not e.data.is_idle]


def _sorted_events(events: list[ActivityEvent]) -> list[ActivityEvent]:
    """Return events sorted by timestamp (defensive copy)."""
    return sorted(events, key=lambda e: e.timestamp_utc)


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
    if len(active) < MIN_ACTIVITY_THRESHOLD:
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
    """Compute how many process-name switches occur per hour.

    Only non-idle events are considered.  Returns 0.0 when there are fewer
    than ``MIN_SWITCH_SAMPLES`` events.

    Ported from ``calculate_switch_frequency()`` (in-memory version, no DB).
    """
    active = _non_idle_events(events)
    if len(active) < MIN_SWITCH_SAMPLES:
        return 0.0

    switches = 0
    prev_proc = active[0].data.process_name
    for ev in active[1:]:
        if ev.data.process_name != prev_proc:
            switches += 1
        prev_proc = ev.data.process_name

    first_ts = active[0].timestamp_utc
    last_ts = active[-1].timestamp_utc
    total_hours = (last_ts - first_ts).total_seconds() / 3600.0
    if total_hours <= 0:
        return 0.0

    return switches / total_hours


def longest_focus_block_s(events: list[ActivityEvent]) -> float:
    """Find the longest continuous same-app focus block in seconds.

    A focus block is a sequence of consecutive non-idle events on the same
    ``process_name``.  The block duration is the sum of ``duration_s`` for
    those events.  Idle events and process-name changes end the current
    block.

    Returns 0.0 for empty input or when there are no focus blocks.
    """
    sorted_evs = _sorted_events(events)
    longest = 0.0
    current_block = 0.0
    current_app: str | None = None

    for ev in sorted_evs:
        if ev.data.is_idle:
            # Idle breaks the block
            if current_block > longest:
                longest = current_block
            current_block = 0.0
            current_app = None
        elif current_app is None:
            # Start of a new block
            current_app = ev.data.process_name
            current_block = ev.duration_s
        elif ev.data.process_name != current_app:
            # App switch — finalise old block, start new
            if current_block > longest:
                longest = current_block
            current_app = ev.data.process_name
            current_block = ev.duration_s
        else:
            # Same app, continue block
            current_block += ev.duration_s

    # Flush last block
    if current_block > longest:
        longest = current_block

    return longest


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

    return TitleFeatures(**vals)  # type: ignore[arg-type]
