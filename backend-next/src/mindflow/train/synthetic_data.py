"""Synthetic activity data generator for MindFlow training pipeline.

Generates realistic multi-day activity logs simulating Chinese application
ecosystem usage patterns across weekday/weekend schedules. Outputs
``list[dict]`` compatible with ``BaselineModel.update()`` input format.

Ported from ``backend/mindflow/analyzer/data_pipeline.py`` (lines 316-428).
Key differences vs. the original:
  - Returns ``list[dict]`` instead of ``pandas.DataFrame`` (pandas is only
    used internally within this function for sorting).
  - Column names match the new domain's feature dict convention (snake_case).
  - Fixed seed (42) ensures deterministic, reproducible output.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd


def generate_synthetic_data(
    days: int = 14,
    samples_per_hour: int = 12,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate realistic synthetic activity data.

    Simulates Chinese application ecosystem usage across 5 daily patterns:
      - early_morning (6-9):  planning, email, messaging
      - morning_focus (9-12):  code, document editing
      - afternoon_mixed (13-17):  work + communication + browsing
      - evening_leisure (19-22):  entertainment, social media
      - late_night (22-6):  entertainment dominant

    Weekend patterns skew heavily toward leisure.

    Args:
        days: Number of days to generate (default 14 for a two-week cycle).
        samples_per_hour: Discrete samples per hour controlling time resolution
            (default 12 → one sample every 5 minutes).
        seed: Random seed for deterministic reproducibility (default 42).

    Returns:
        List of dicts with keys:
          ``timestamp`` (datetime), ``process_name`` (str),
          ``window_title`` (str), ``duration_seconds`` (float),
          ``is_idle`` (int 0|1).

    Example:
        >>> rows = generate_synthetic_data(days=3)
        >>> len(rows)
        864
        >>> rows[0]["timestamp"].hour
        0
    """
    rng = np.random.default_rng(seed)
    start_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    interval_seconds = 3600 // samples_per_hour

    # ── Pattern definitions ──────────────────────────────────────────────
    apps_by_pattern: dict[str, dict[str, Any]] = {
        "morning_focus": {
            "apps": ["vscode", "pycharm", "notion", "typora", "terminal"],
            "titles": [
                "main.py - MindFlow",
                "analysis.ipynb - Jupyter",
                "notes - Obsidian",
                "paper_draft.md - Typora",
                "terminal - zsh",
            ],
            "weights": [0.35, 0.25, 0.15, 0.15, 0.10],
        },
        "afternoon_mixed": {
            "apps": ["chrome", "teams", "vscode", "wechat", "excel"],
            "titles": [
                "github.com/MindFlow - Chrome",
                "Teams Meeting",
                "app.py - VSCode",
                "WeChat",
                "quarterly_report.xlsx - Excel",
            ],
            "weights": [0.25, 0.20, 0.20, 0.20, 0.15],
        },
        "evening_leisure": {
            "apps": ["bilibili", "youtube", "steam", "wechat", "chrome"],
            "titles": [
                "B站 - Anime",
                "YouTube - Music",
                "Steam",
                "WeChat Moments",
                "Zhihu - Chrome",
            ],
            "weights": [0.30, 0.20, 0.15, 0.20, 0.15],
        },
        "late_night": {
            "apps": ["bilibili", "douyin", "weibo", "chrome", "wechat"],
            "titles": [
                "B站 - Late Night",
                "Douyin",
                "Weibo Hot Search",
                "reddit - Chrome",
                "WeChat",
            ],
            "weights": [0.30, 0.25, 0.15, 0.15, 0.15],
        },
        "early_morning": {
            "apps": ["chrome", "notion", "wechat", "calendar", "mail"],
            "titles": [
                "Gmail - Chrome",
                "Today Plan - Notion",
                "WeChat Messages",
                "Calendar",
                "Mail",
            ],
            "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
        },
    }

    idle_apps = ["", "lock_screen", "screensaver"]
    idle_titles = ["", "Locked", "Screensaver"]

    rows: list[dict[str, Any]] = []
    for day_offset in range(days):
        day_start = start_date + timedelta(days=day_offset)
        is_weekend = day_start.weekday() >= 5

        for hour in range(24):
            for sample in range(samples_per_hour):
                ts = day_start + timedelta(hours=hour, seconds=sample * interval_seconds)

                if is_weekend:
                    pattern_key = _weekend_pattern(hour, rng)
                else:
                    pattern_key = _weekday_pattern(hour, rng)

                # Idle probability by time segment
                if hour < 2 or hour >= 23:
                    idle_chance = 0.85
                elif pattern_key == "early_morning":
                    idle_chance = 0.25
                elif pattern_key == "evening_leisure":
                    idle_chance = 0.03
                elif pattern_key == "late_night":
                    idle_chance = 0.15
                else:
                    idle_chance = 0.05

                if rng.random() < idle_chance:
                    idx = int(rng.integers(0, len(idle_apps)))
                    is_idle = 1
                    proc_name = idle_apps[idx]
                    win_title = idle_titles[min(idx, len(idle_titles) - 1)]
                    duration = max(1, int(rng.normal(120, 30)))
                else:
                    pattern = apps_by_pattern[pattern_key]
                    apps: list[str] = pattern["apps"]
                    titles: list[str] = pattern["titles"]
                    weights = np.array(pattern["weights"], dtype=float)
                    weights = weights / weights.sum()

                    idx = int(rng.choice(len(apps), p=weights))
                    is_idle = 0
                    proc_name = apps[idx]
                    win_title = titles[idx]
                    base_duration = 3600 // samples_per_hour
                    duration = max(1, int(rng.normal(base_duration, base_duration * 0.2)))

                rows.append(
                    {
                        "timestamp": ts,
                        "process_name": proc_name,
                        "window_title": win_title,
                        "duration_seconds": float(duration),
                        "is_idle": is_idle,
                    }
                )

    # Sort and return as list[dict] (pandas used only internally for sorting)
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return cast(list[dict[str, Any]], df.to_dict(orient="records"))


def _weekday_pattern(hour: int, rng: np.random.Generator) -> str:
    """Determine behavior pattern for a weekday hour."""
    if 0 <= hour < 6:
        return "late_night"
    if 6 <= hour < 9:
        return "early_morning"
    if 9 <= hour < 12:
        return "morning_focus" if rng.random() < 0.85 else "afternoon_mixed"
    if 12 <= hour < 13:
        return "afternoon_mixed" if rng.random() < 0.5 else "evening_leisure"
    if 13 <= hour < 18:
        return "afternoon_mixed" if rng.random() < 0.80 else "morning_focus"
    if 18 <= hour < 19:
        return "afternoon_mixed"
    if 19 <= hour < 22:
        return "evening_leisure" if rng.random() >= 0.10 else "morning_focus"
    return "evening_leisure" if rng.random() < 0.60 else "late_night"


def _weekend_pattern(hour: int, rng: np.random.Generator) -> str:
    """Determine behavior pattern for a weekend hour."""
    if 0 <= hour < 7:
        return "late_night"
    if 7 <= hour < 10:
        return "early_morning"
    if 10 <= hour < 13:
        return "evening_leisure" if rng.random() >= 0.30 else "morning_focus"
    if 13 <= hour < 18:
        return "evening_leisure" if rng.random() >= 0.45 else "afternoon_mixed"
    if 18 <= hour < 22:
        return "evening_leisure"
    return "evening_leisure" if rng.random() < 0.70 else "late_night"
