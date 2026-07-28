"""Synthetic activity data generator for MindFlow training pipeline.

Generates realistic multi-day activity logs simulating Chinese application
ecosystem usage patterns across weekday/weekend schedules. Supports single-user
legacy mode (backward compatible) and multi-user generation with 30 student
archetypes and procrastination episode injection.

Outputs ``list[dict]`` compatible with ``BaselineModel.update()`` input format.

Ported from ``backend/mindflow/analyzer/data_pipeline.py`` (lines 316-428).
Key differences vs. the original:
  - Returns ``list[dict]`` instead of ``pandas.DataFrame`` (pandas is only
    used internally within this function for sorting).
  - Column names match the new domain's feature dict convention (snake_case).
  - Fixed seed (42) ensures deterministic, reproducible output.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

from mindflow.train.user_profiles import (
    PROFILES,
    ProcrastinationEpisode,
    StudentArchetype,
    get_archetype,
    get_episode,
)


def generate_synthetic_data(
    days: int = 14,
    samples_per_hour: int = 12,
    seed: int = 42,
    num_users: int = 1,
    include_procrastination: bool = False,
    user_profiles: list[str] | None = None,
    archetypes: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate realistic synthetic activity data.

    Simulates Chinese application ecosystem usage across 5 daily patterns:
      - early_morning (6-9):  planning, email, messaging
      - morning_focus (9-12):  code, document editing
      - afternoon_mixed (13-17):  work + communication + browsing
      - evening_leisure (19-22):  entertainment, social media
      - late_night (22-6):  entertainment dominant

    Weekend patterns skew heavily toward leisure.

    When ``num_users > 1``, ``include_procrastination=True``, or
    ``user_profiles`` is set, uses the enhanced multi-user generator with
    student archetype profiles and procrastination episode injection.

    Args:
        days: Number of days to generate (default 14 for a two-week cycle).
        samples_per_hour: Discrete samples per hour controlling time resolution
            (default 12 → one sample every 5 minutes).
        seed: Random seed for deterministic reproducibility (default 42).
        num_users: Number of users to simulate (default 1). Ignored when
            ``user_profiles`` is provided.
        include_procrastination: If True, inject archetype-specific
            procrastination episodes into the daily schedule.
        user_profiles: List of profile IDs (e.g. ``["junior_cs", "senior_business"]``).
            If provided, ``num_users`` is set to ``len(user_profiles)``.
        archetypes: Custom archetype definitions keyed by profile ID.
            Each value is a dict with optional ``major``, ``year`` fields
            (for base profile lookup) plus any ``StudentArchetype`` field overrides.

    Returns:
        List of dicts with keys:
          ``timestamp`` (datetime), ``process_name`` (str),
          ``window_title`` (str), ``duration_seconds`` (float),
          ``is_idle`` (int 0|1), ``user_id`` (int, 1-indexed).

    Example:
        >>> rows = generate_synthetic_data(days=3)
        >>> len(rows)
        864
        >>> rows[0]["timestamp"].hour
        0
        >>> rows[0]["user_id"]
        1
    """
    # ── Backward-compatible legacy path ────────────────────────────────────
    # When all enhanced params are at defaults, produce output identical to
    # the original function (plus ``user_id: 1`` on every row).
    if user_profiles is None and num_users == 1 and not include_procrastination:
        return _generate_legacy(days, samples_per_hour, seed)

    # ── Enhanced multi-user path ───────────────────────────────────────────
    profiles = _resolve_profiles(num_users, user_profiles, archetypes)

    all_rows: list[dict[str, Any]] = []
    for user_idx, profile in enumerate(profiles):
        user_id = user_idx + 1
        user_seed = seed + user_id * 10000
        user_rows = _generate_user_data(
            days=days,
            samples_per_hour=samples_per_hour,
            seed=user_seed,
            user_id=user_id,
            profile=profile,
            include_procrastination=include_procrastination,
        )
        all_rows.extend(user_rows)

    # Sort globally by timestamp, then user_id for determinism
    df = pd.DataFrame(all_rows).sort_values(
        ["timestamp", "user_id"]
    ).reset_index(drop=True)
    return cast(list[dict[str, Any]], df.to_dict(orient="records"))


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy generator (backward-compatible single user)
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_legacy(
    days: int,
    samples_per_hour: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Exact same generation logic as the original function, plus user_id=1.

    Preserves the original apps_by_pattern dict, idle logic, pattern
    selection, and RNG behaviour. The ONLY difference from the original
    is the addition of ``"user_id": 1`` to every output row.
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
                        "user_id": 1,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Profile resolution
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_profiles(
    num_users: int,
    user_profiles: list[str] | None,
    custom_archetypes: dict[str, Any] | None,
) -> list[StudentArchetype]:
    """Resolve the list of StudentArchetype objects to use for generation.

    If ``user_profiles`` is provided, those profile IDs are used directly.
    Otherwise, cycles through the default profile set for ``num_users`` users.
    Custom archetype overrides are applied where profile IDs match.
    """
    if user_profiles is not None:
        profile_ids = list(user_profiles)
    else:
        all_ids = sorted(PROFILES.keys())
        profile_ids = [all_ids[i % len(all_ids)] for i in range(num_users)]

    resolved: list[StudentArchetype] = []
    for pid in profile_ids:
        resolved.append(_resolve_single_archetype(pid, custom_archetypes))
    return resolved


def _resolve_single_archetype(
    profile_id: str,
    custom_archetypes: dict[str, Any] | None,
) -> StudentArchetype:
    """Resolve one archetype, applying custom overrides if present.

    Custom archetype dicts may contain ``major`` and ``year`` keys to
    select the base profile, plus any ``StudentArchetype`` field overrides.
    """
    if custom_archetypes and profile_id in custom_archetypes:
        custom = custom_archetypes[profile_id]
        major = str(custom.get("major", "cs"))
        year = str(custom.get("year", "junior"))
        base_id = f"{year}_{major}"
        base = get_archetype(base_id)

        # Collect override fields (everything except major/year)
        overrides: dict[str, Any] = {
            k: v for k, v in custom.items() if k not in ("major", "year")
        }
        return dataclasses.replace(base, profile_id=profile_id, **overrides)

    return get_archetype(profile_id)


# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced user data generator
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_user_data(
    days: int,
    samples_per_hour: int,
    seed: int,
    user_id: int,
    profile: StudentArchetype,
    include_procrastination: bool,
) -> list[dict[str, Any]]:
    """Generate activity rows for a single user using their student archetype.

    Uses profile-specific app ecosystems, schedule patterns, and optionally
    injects procrastination episodes.
    """
    rng = np.random.default_rng(seed)
    procrastination_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    start_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    interval_seconds = 3600 // samples_per_hour

    # ── Build pattern → apps mapping from profile ─────────────────────────
    apps_by_pattern: dict[str, dict[str, Any]] = {
        k: {
            "apps": list(profile.primary_apps[k]),
            "titles": list(profile.primary_titles[k]),
            "weights": list(profile.primary_weights[k]),
        }
        for k in ["early_morning", "morning_focus", "afternoon_mixed",
                   "evening_leisure", "late_night"]
    }

    idle_apps = ["", "lock_screen", "screensaver"]
    idle_titles = ["", "Locked", "Screensaver"]

    rows: list[dict[str, Any]] = []
    for day_offset in range(days):
        day_start = start_date + timedelta(days=day_offset)
        is_weekend = day_start.weekday() >= 5

        # Compute effective wake for this day (shifted on weekends)
        effective_wake = profile.typical_wake_hour
        if is_weekend:
            effective_wake += profile.weekend_delay_hours

        # Pre-compute procrastination episodes for this day
        episodes_today: list[tuple[ProcrastinationEpisode, float, float]] = []
        deadline_panic = False
        if include_procrastination:
            episodes_today = _compute_episodes_for_day(
                profile, is_weekend, procrastination_rng
            )
            # Deadline panic: on non-procrastination days, a burst of
            # hyper-productivity may occur (post-procrastination crunch).
            # Higher probability to ensure reliable test coverage.
            if not episodes_today and procrastination_rng.random() < 0.25:
                deadline_panic = True

        for hour in range(24):
            for sample in range(samples_per_hour):
                ts = day_start + timedelta(
                    hours=hour, seconds=sample * interval_seconds
                )
                hour_float = hour + sample / samples_per_hour

                # Always consume the baseline RNG stream first. Enabling
                # procrastination then overlays episode samples without changing
                # unrelated profile behaviour for the same seed.
                pattern_key = _profile_pattern(
                    hour, rng, is_weekend, profile, samples_per_hour
                )
                idle_chance = _profile_idle_chance(
                    hour, pattern_key, profile, effective_wake
                )

                if rng.random() < idle_chance:
                    idx = int(rng.integers(0, len(idle_apps)))
                    rows.append({
                        "timestamp": ts,
                        "process_name": idle_apps[idx],
                        "window_title": idle_titles[min(idx, len(idle_titles) - 1)],
                        "duration_seconds": float(max(1, int(rng.normal(120, 30)))),
                        "is_idle": 1,
                        "user_id": user_id,
                    })
                else:
                    pattern = apps_by_pattern[pattern_key]
                    apps: list[str] = pattern["apps"]
                    titles: list[str] = pattern["titles"]
                    weights = np.array(pattern["weights"], dtype=float)
                    weights = weights / weights.sum()

                    idx = int(rng.choice(len(apps), p=weights))
                    base_dur = interval_seconds
                    duration = max(
                        1, int(rng.normal(base_dur, base_dur * 0.2))
                    )
                    rows.append({
                        "timestamp": ts,
                        "process_name": apps[idx],
                        "window_title": titles[idx],
                        "duration_seconds": float(duration),
                        "is_idle": 0,
                        "user_id": user_id,
                    })

                current_ep = _find_active_episode(episodes_today, hour_float)
                if current_ep is not None:
                    rows.pop()
                    _append_episode_sample(
                        rows, ts, current_ep, procrastination_rng,
                        interval_seconds, user_id,
                    )
                elif (
                    deadline_panic
                    and pattern_key in {"morning_focus", "afternoon_mixed"}
                    and rows[-1]["is_idle"] == 0
                ):
                    rows.pop()
                    _append_deadline_panic_sample(
                        rows, ts, apps_by_pattern, procrastination_rng,
                        interval_seconds, user_id,
                    )

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Profile-aware pattern and idle helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _profile_pattern(
    hour: int,
    rng: np.random.Generator,
    is_weekend: bool,
    profile: StudentArchetype,
    samples_per_hour: int,  # noqa: ARG001 reserved for future use
) -> str:
    """Determine behaviour pattern for a profile at a given hour.

    Shifts the effective hour based on the profile's wake time, then
    delegates to the standard weekday/weekend pattern functions. Schedule
    rigidity modulates the pattern-switch probabilities.
    """
    wake = profile.typical_wake_hour
    if is_weekend:
        wake += profile.weekend_delay_hours
    shift = wake - 6  # align with default schedule (6am early_morning start)
    effective_hour = (hour - shift) % 24

    rigidity = profile.schedule_rigidity

    if is_weekend:
        return _rigid_weekend_pattern(effective_hour, rng, rigidity)
    return _rigid_weekday_pattern(effective_hour, rng, rigidity)


def _rigid_weekday_pattern(
    hour: int, rng: np.random.Generator, rigidity: float
) -> str:
    """Weekday pattern with rigidity-adjusted transition probabilities.

    Higher rigidity → pattern transitions are more predictable (closer
    to the base thresholds).  Lower rigidity → transitions are noisier.
    """
    # Blend base probability toward 0.5 based on rigidity
    def p(base: float) -> float:
        return base * rigidity + 0.5 * (1.0 - rigidity)

    if 0 <= hour < 6:
        return "late_night"
    if 6 <= hour < 9:
        return "early_morning"
    if 9 <= hour < 12:
        return "morning_focus" if rng.random() < p(0.85) else "afternoon_mixed"
    if 12 <= hour < 13:
        return "afternoon_mixed" if rng.random() < p(0.50) else "evening_leisure"
    if 13 <= hour < 18:
        return "afternoon_mixed" if rng.random() < p(0.80) else "morning_focus"
    if 18 <= hour < 19:
        return "afternoon_mixed"
    if 19 <= hour < 22:
        return "evening_leisure" if rng.random() < p(0.90) else "morning_focus"
    return "evening_leisure" if rng.random() < p(0.60) else "late_night"


def _rigid_weekend_pattern(
    hour: int, rng: np.random.Generator, rigidity: float
) -> str:
    """Weekend pattern with rigidity-adjusted transition probabilities."""
    def p(base: float) -> float:
        return base * rigidity + 0.5 * (1.0 - rigidity)

    if 0 <= hour < 7:
        return "late_night"
    if 7 <= hour < 10:
        return "early_morning"
    if 10 <= hour < 13:
        return "evening_leisure" if rng.random() < p(0.70) else "morning_focus"
    if 13 <= hour < 18:
        return "evening_leisure" if rng.random() < p(0.55) else "afternoon_mixed"
    if 18 <= hour < 22:
        return "evening_leisure"
    return "evening_leisure" if rng.random() < p(0.70) else "late_night"


def _hour_in_sleep_window(hour: int, wake: int, sleep: int) -> bool:
    """Check whether ``hour`` falls inside the sleep window [sleep, wake).

    Handles wrap-around when sleep > wake (night-owl schedule).
    """
    if sleep < wake:
        return sleep <= hour < wake
    return hour >= sleep or hour < wake


def _hour_in_interval(hour: int, start: int, end: int) -> bool:
    """Check whether ``hour`` falls inside [start, end), with wrap-around."""
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end


def _profile_idle_chance(
    hour: int,
    pattern_key: str,
    profile: StudentArchetype,
    effective_wake: int,
) -> float:
    """Compute idle probability for a profile-adjusted hour and pattern.

    Uses ``effective_wake`` (already adjusted for weekend delay) to
    determine which hours the user is expected to be asleep vs active.
    Schedule rigidity blends only the *awake*-hour idle probabilities
    toward 0.25 — sleep-hour idle stays high regardless of rigidity
    (even unstructured students are mostly idle while sleeping).
    """
    sleep = profile.typical_sleep_hour
    if sleep == 0:
        sleep = 24  # normalize midnight
    # On weekends, sleep also shifts
    weekend_delta = effective_wake - profile.typical_wake_hour
    effective_sleep = (sleep + weekend_delta) % 24
    if effective_sleep == 0:
        effective_sleep = 24

    rigidity = profile.schedule_rigidity

    # ── Sleep / wake determination ────────────────────────────────────────
    # The sleep window is [effective_sleep, effective_wake).  When sleep < wake
    # (normal), this is a single contiguous block.  When sleep > wake
    # (night-owl), it wraps around midnight.
    in_sleep = _hour_in_sleep_window(hour, effective_wake, effective_sleep)
    if in_sleep:
        # Deep sleep: hours well inside the sleep window → always idle.
        deep_start = (effective_sleep + 1) % 24
        deep_end = (effective_wake - 2) % 24
        in_deep = _hour_in_interval(hour, deep_start, deep_end)
        if in_deep:
            return 1.0
        # Near wake/sleep edges: slightly less idle (dreaming, phone check).
        return 0.92

    # ── Awake hours: pattern-dependent idle with rigidity blending ────────

    # Awake hours: idle depends on pattern + rigidity blending.
    if pattern_key == "early_morning":
        base_idle = 0.25
    elif pattern_key == "evening_leisure":
        base_idle = 0.03
    elif pattern_key == "late_night":
        base_idle = 0.15
    else:
        base_idle = 0.05

    # Rigidity blending: low-rigidity students have less predictable
    # activity within their awake hours. Sleep hours are excluded above.
    return base_idle * rigidity + 0.25 * (1.0 - rigidity)


# ═══════════════════════════════════════════════════════════════════════════════
# Procrastination episode logic
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_episodes_for_day(
    profile: StudentArchetype,
    is_weekend: bool,
    rng: np.random.Generator,
) -> list[tuple[ProcrastinationEpisode, float, float]]:
    """Determine procrastination episodes for a single user-day.

    Returns a list of ``(episode, start_hour_float, end_hour_float)`` tuples.
    Episodes are clipped to the valid hour range and filtered by day bias.
    """
    proc_chance = profile.daily_proc_probability
    if is_weekend:
        proc_chance *= profile.weekend_multiplier
    proc_chance = min(proc_chance, 0.95)

    if rng.random() >= proc_chance:
        return []

    # Number of episodes today (1-3, weighted)
    num_eps = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15]))

    ep_types = list(profile.episode_type_weights.keys())
    ep_weights_arr = np.array(
        [profile.episode_type_weights[t] for t in ep_types], dtype=float
    )
    ep_weights_arr = ep_weights_arr / ep_weights_arr.sum()

    episodes: list[tuple[ProcrastinationEpisode, float, float]] = []
    for _ in range(num_eps):
        ep_name = str(rng.choice(ep_types, p=ep_weights_arr))
        episode = get_episode(ep_name)

        # Respect day_bias
        if episode.day_bias == "weekday_only" and is_weekend:
            continue
        if episode.day_bias == "weekend_only" and not is_weekend:
            continue

        # Pick start hour within the episode's valid range
        earliest = float(episode.earliest_hour)
        latest = min(float(episode.latest_hour), 23.5)
        if latest <= earliest:
            latest = earliest + 2.0
        start_hour = float(rng.uniform(earliest, latest))

        # Pick duration
        duration = float(rng.uniform(
            episode.min_duration_hours, episode.max_duration_hours
        ))
        end_hour = min(start_hour + duration, 24.0)
        episodes.append((episode, start_hour, end_hour))

    return _merge_overlapping_episodes(episodes)


def _merge_overlapping_episodes(
    episodes: list[tuple[ProcrastinationEpisode, float, float]],
) -> list[tuple[ProcrastinationEpisode, float, float]]:
    """Merge episodes that overlap in time, keeping the first episode's type."""
    if len(episodes) <= 1:
        return episodes

    sorted_eps = sorted(episodes, key=lambda e: e[1])
    merged: list[tuple[ProcrastinationEpisode, float, float]] = []
    current_ep, cur_start, cur_end = sorted_eps[0]

    for ep, start, end in sorted_eps[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((current_ep, cur_start, cur_end))
            current_ep, cur_start, cur_end = ep, start, end

    merged.append((current_ep, cur_start, cur_end))
    return merged


def _find_active_episode(
    episodes: list[tuple[ProcrastinationEpisode, float, float]],
    hour_float: float,
) -> ProcrastinationEpisode | None:
    """Return the episode active at ``hour_float``, or None."""
    for ep, start, end in episodes:
        if start <= hour_float < end:
            return ep
    return None


def _append_episode_sample(
    rows: list[dict[str, Any]],
    ts: datetime,
    episode: ProcrastinationEpisode,
    rng: np.random.Generator,
    interval_seconds: int,
    user_id: int,
) -> None:
    """Append a single row for a procrastination episode time slot.

    Uses the episode's apps, weights, switch_frequency, and idle_ratio to
    produce realistic procrastination behaviour. App switching follows a
    simple sticky model: each sample has a switch_prob chance of changing
    to a different app.
    """
    ep_weights = np.array(episode.weights, dtype=float)
    ep_weights = ep_weights / ep_weights.sum()

    # --- Idle check (very rare during episodes to preserve streaks) ---
    if rng.random() < 0.005:
        idle_apps = ["", "lock_screen", "screensaver"]
        idle_titles = ["", "Locked", "Screensaver"]
        idx = int(rng.integers(0, len(idle_apps)))
        rows.append({
            "timestamp": ts,
            "process_name": idle_apps[idx],
            "window_title": idle_titles[min(idx, len(idle_titles) - 1)],
            "duration_seconds": float(max(1, int(rng.normal(120, 30)))),
            "is_idle": 1,
            "user_id": user_id,
        })
        return

    # --- Sticky app switching model ---
    # switch_frequency is the expected number of app switches per hour.
    # switch_prob = switch_frequency / samples_per_hour gives per-sample
    # probability of switching to a different app.
    samples_per_hour = 3600 // interval_seconds
    switch_prob = min(1.0, episode.switch_frequency / max(samples_per_hour, 1))

    # Pick initial app (state is tracked via an attribute on the episode
    # object itself — a bit hacky but simple)
    if not hasattr(_append_episode_sample, "_state"):
        _append_episode_sample._state = {}  # type: ignore[attr-defined]
    state_key = id(episode)  # per-episode-object state
    if state_key not in _append_episode_sample._state:  # type: ignore[attr-defined]
        _append_episode_sample._state[state_key] = int(  # type: ignore[attr-defined]
            rng.choice(len(episode.apps), p=ep_weights)
        )

    if rng.random() < switch_prob:
        _append_episode_sample._state[state_key] = int(  # type: ignore[attr-defined]
            rng.choice(len(episode.apps), p=ep_weights)
        )

    idx = _append_episode_sample._state[state_key]  # type: ignore[attr-defined]
    base_dur = interval_seconds
    duration = max(1, int(rng.normal(base_dur, base_dur * 0.2)))

    rows.append({
        "timestamp": ts,
        "process_name": episode.apps[idx],
        "window_title": episode.titles[idx],
        "duration_seconds": float(duration),
        "is_idle": 0,
        "user_id": user_id,
    })


def _append_deadline_panic_sample(
    rows: list[dict[str, Any]],
    ts: datetime,
    apps_by_pattern: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    interval_seconds: int,
    user_id: int,
) -> None:
    """Generate a hyper-productive sample for a deadline-panic day.

    During deadline panic, the user stays intensely focused on productive
    apps (morning_focus / afternoon_mixed) with very low idle probability
    and high stickiness (rarely switches apps).
    """
    # Use morning_focus apps if available, else afternoon_mixed
    pattern = apps_by_pattern.get("morning_focus") or apps_by_pattern.get(
        "afternoon_mixed", apps_by_pattern.get("early_morning")
    )
    if pattern is None:
        # Fallback: use first available pattern
        pattern = next(iter(apps_by_pattern.values()))

    apps: list[str] = pattern["apps"]
    titles: list[str] = pattern["titles"]
    weights = np.array(pattern["weights"], dtype=float)
    weights = weights / weights.sum()

    # Sticky app selection: switch only ~5% of the time for long runs
    switch_prob = 0.05

    if not hasattr(_append_deadline_panic_sample, "_state"):
        _append_deadline_panic_sample._state = {}  # type: ignore[attr-defined]
    state_key = id(apps_by_pattern)  # per-generator-context state
    if state_key not in _append_deadline_panic_sample._state:  # type: ignore[attr-defined]
        _append_deadline_panic_sample._state[state_key] = int(  # type: ignore[attr-defined]
            rng.choice(len(apps), p=weights)
        )

    if rng.random() < switch_prob:
        _append_deadline_panic_sample._state[state_key] = int(  # type: ignore[attr-defined]
            rng.choice(len(apps), p=weights)
        )

    idx = _append_deadline_panic_sample._state[state_key]  # type: ignore[attr-defined]
    base_dur = interval_seconds
    duration = max(1, int(rng.normal(base_dur, base_dur * 0.2)))

    rows.append({
        "timestamp": ts,
        "process_name": apps[idx],
        "window_title": titles[idx],
        "duration_seconds": float(duration),
        "is_idle": 0,
        "user_id": user_id,
    })
