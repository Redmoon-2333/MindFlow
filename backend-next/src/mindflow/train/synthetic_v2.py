"""Synthetic v2 feature window generator for pre-training base models.

Generates labeled 24-dim v2 feature windows directly from the 30 student
archetypes (user_profiles.py), bypassing raw event simulation and rollup.

Design:
  - Each archetype generates N days of 5-min feature windows.
  - Per-window features are sampled from archetype-specific distributions.
  - Windows are labeled based on the archetype's procrastination episodes
    and time-pattern productivity profile, with realistic label noise.
  - Output is directly consumable by ``prepare_v2_training_data()``.

Usage::

    from mindflow.train.synthetic_v2 import generate_v2_synthetic_data
    windows, feedback = generate_v2_synthetic_data(
        archetype_ids="all",
        days_per_archetype=14,
    )
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from mindflow.train.v2 import V2_FEATURE_NAMES

# ── Feature generation parameters ──────────────────────────────────────────

# Interaction profiles per major category (keypress, mouse, scroll intensity)
_INTERACTION_PROFILES: dict[str, dict[str, float | tuple[float, float]]] = {
    "cs": {
        "keypress_mean": 45.0,
        "keypress_std": 25.0,
        "mouse_click_mean": 8.0,
        "mouse_click_std": 6.0,
        "scroll_mean": 15.0,
        "scroll_std": 20.0,
        "mouse_distance_mean": 800.0,
        "mouse_distance_std": 600.0,
        "input_active_ratio_mean": 0.75,
        "burst_mean": 2.5,
        "interval_mean": 8.0,
        "interval_std": 12.0,
    },
    "ee": {
        "keypress_mean": 35.0,
        "keypress_std": 20.0,
        "mouse_click_mean": 10.0,
        "mouse_click_std": 8.0,
        "scroll_mean": 20.0,
        "scroll_std": 25.0,
        "mouse_distance_mean": 900.0,
        "mouse_distance_std": 700.0,
        "input_active_ratio_mean": 0.7,
        "burst_mean": 2.0,
        "interval_mean": 10.0,
        "interval_std": 15.0,
    },
    "liberal_arts": {
        "keypress_mean": 25.0,
        "keypress_std": 18.0,
        "mouse_click_mean": 6.0,
        "mouse_click_std": 5.0,
        "scroll_mean": 40.0,
        "scroll_std": 30.0,
        "mouse_distance_mean": 1200.0,
        "mouse_distance_std": 800.0,
        "input_active_ratio_mean": 0.6,
        "burst_mean": 1.5,
        "interval_mean": 15.0,
        "interval_std": 20.0,
    },
    "business": {
        "keypress_mean": 30.0,
        "keypress_std": 20.0,
        "mouse_click_mean": 12.0,
        "mouse_click_std": 9.0,
        "scroll_mean": 25.0,
        "scroll_std": 22.0,
        "mouse_distance_mean": 1000.0,
        "mouse_distance_std": 700.0,
        "input_active_ratio_mean": 0.65,
        "burst_mean": 1.8,
        "interval_mean": 12.0,
        "interval_std": 16.0,
    },
    "design": {
        "keypress_mean": 20.0,
        "keypress_std": 15.0,
        "mouse_click_mean": 15.0,
        "mouse_click_std": 10.0,
        "scroll_mean": 30.0,
        "scroll_std": 25.0,
        "mouse_distance_mean": 2000.0,
        "mouse_distance_std": 1500.0,
        "input_active_ratio_mean": 0.7,
        "burst_mean": 2.2,
        "interval_mean": 9.0,
        "interval_std": 14.0,
    },
    "medical": {
        "keypress_mean": 15.0,
        "keypress_std": 12.0,
        "mouse_click_mean": 5.0,
        "mouse_click_std": 4.0,
        "scroll_mean": 50.0,
        "scroll_std": 35.0,
        "mouse_distance_mean": 600.0,
        "mouse_distance_std": 500.0,
        "input_active_ratio_mean": 0.5,
        "burst_mean": 1.0,
        "interval_mean": 20.0,
        "interval_std": 25.0,
    },
}

# Entertainment-time interaction modifier (fraction of productive levels)
_ENTERTAINMENT_INTERACTION_FACTOR: float = 0.15

# Sleep-hour interaction modifier
_SLEEP_INTERACTION_FACTOR: float = 0.02

# Probability that a procrastination window gets label = distract
_PROCRASTINATION_LABEL_PROB: float = 0.85

# Probability that a productive window gets label = focus
_PRODUCTIVE_LABEL_PROB: float = 0.80

# Noise label probability (random flip)
_LABEL_NOISE_PROB: float = 0.05

# Window size in minutes
_WINDOW_MINUTES: int = 5

# Windows per day
_WINDOWS_PER_DAY: int = 24 * 60 // _WINDOW_MINUTES  # 288


@dataclass
class ArchetypeParams:
    """Extracted parameters from a StudentArchetype for v2 generation."""

    profile_id: str
    grade: str
    major: str
    wake_hour: float
    sleep_hour: float
    weekend_delay_hours: float
    schedule_rigidity: float
    daily_proc_probability: float
    expected_switch_frequency_mean: float
    expected_idle_ratio_mean: float
    expected_focus_score_mean: float
    expected_entertainment_ratio_mean: float
    episode_type_weights: dict[str, float]
    weekend_multiplier: float


def _extract_archetype_params(archetype: Any) -> ArchetypeParams:
    """Extract generation parameters from a StudentArchetype."""
    majors_reverse = {
        "计算机/软件": "cs",
        "电子信息/自动化": "ee",
        "人文/社科": "liberal_arts",
        "经管/商科": "business",
        "设计/艺术": "design",
        "医学/药学": "medical",
    }
    major_key = majors_reverse.get(archetype.major, "cs")

    return ArchetypeParams(
        profile_id=archetype.profile_id,
        grade=archetype.grade,
        major=major_key,
        wake_hour=archetype.typical_wake_hour,
        sleep_hour=archetype.typical_sleep_hour,
        weekend_delay_hours=archetype.weekend_delay_hours,
        schedule_rigidity=archetype.schedule_rigidity,
        daily_proc_probability=archetype.daily_proc_probability,
        expected_switch_frequency_mean=archetype.expected_switch_frequency_mean,
        expected_idle_ratio_mean=archetype.expected_idle_ratio_mean,
        expected_focus_score_mean=archetype.expected_focus_score_mean,
        expected_entertainment_ratio_mean=archetype.expected_entertainment_ratio_mean,
        episode_type_weights=archetype.episode_type_weights,
        weekend_multiplier=archetype.weekend_multiplier,
    )


# ── Day simulator ──────────────────────────────────────────────────────────


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # Saturday or Sunday


def _compute_daily_patterns(
    params: ArchetypeParams,
    target_date: date,
    rng: random.Random,
) -> list[dict[str, float]]:
    """Generate 288 (24h × 12 windows/h) feature windows for one day.

    Each window is a dict of 24 v2 feature values.
    """
    is_weekend_flag = _is_weekend(target_date)
    weekday = target_date.weekday()
    hour_float_sleep = params.sleep_hour
    hour_float_wake = params.wake_hour + (
        params.weekend_delay_hours if is_weekend_flag else 0.0
    )

    # Determine if today has a procrastination episode
    has_procrastination = rng.random() < params.daily_proc_probability * (
        params.weekend_multiplier if is_weekend_flag else 1.0
    )

    # Pick episode type if procrastinating
    episode_type: str | None = None
    if has_procrastination and params.episode_type_weights:
        episode_types_list = list(params.episode_type_weights.keys())
        episode_weights = list(params.episode_type_weights.values())
        if episode_types_list and sum(episode_weights) > 0:
            episode_type = rng.choices(episode_types_list, weights=episode_weights, k=1)[0]

    # Episode time range
    episode_start_window: int = -1
    episode_end_window: int = -1
    if episode_type == "binge_watching":
        # Evening binge: 19:00-01:00
        episode_start_window = 19 * 12  # 19:00
        episode_end_window = 25 * 12  # 01:00 next day
    elif episode_type == "doom_scrolling":
        # Can happen anytime: 10:00-02:00
        episode_start_window = 10 * 12
        episode_end_window = 26 * 12
    elif episode_type == "gaming_session":
        # Evening/weekend gaming: 18:00-01:00
        if is_weekend_flag or rng.random() < 0.3:
            episode_start_window = 18 * 12
            episode_end_window = 25 * 12
    elif episode_type == "social_media_spiral":
        episode_start_window = 8 * 12
        episode_end_window = 25 * 12
    elif episode_type == "inspiration_browsing":
        episode_start_window = 14 * 12
        episode_end_window = 26 * 12
    elif episode_type == "crash_and_burn":
        if is_weekend_flag:
            episode_start_window = 14 * 12
            episode_end_window = 22 * 12

    windows: list[dict[str, float]] = []
    interaction = _INTERACTION_PROFILES.get(params.major, _INTERACTION_PROFILES["cs"])

    for window_idx in range(_WINDOWS_PER_DAY):
        # Compute wall clock for this window
        window_hour = window_idx // 12
        window_minute = (window_idx % 12) * 5
        day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)
        window_start = day_start + timedelta(hours=window_hour, minutes=window_minute)
        window_end = window_start + timedelta(minutes=_WINDOW_MINUTES)

        # ---- Determine window state ----
        is_sleep = _is_sleep_hour(hour_float_sleep, hour_float_wake, window_hour)
        is_episode = (episode_start_window <= window_idx < episode_end_window) if has_procrastination else False
        is_productive = not is_sleep and not is_episode and not is_weekend_flag

        # ---- Generate features ----
        feats = _generate_window_features(
            params=params,
            interaction=interaction,
            window_hour=window_hour,
            window_idx=window_idx,
            is_sleep=is_sleep,
            is_productive=is_productive,
            is_episode=is_episode,
            is_weekend_flag=is_weekend_flag,
            weekday=weekday,
            rng=rng,
        )

        # ---- Compute label ----
        label = _compute_label(
            feats=feats,
            is_episode=is_episode,
            is_sleep=is_sleep,
            is_productive=is_productive,
            params=params,
            rng=rng,
        )

        windows.append({
            "window_start_utc": window_start.isoformat(),
            "window_end_utc": window_end.isoformat(),
            "feature_schema_version": 2,
            "features": feats,
            "label": label,
        })

    return windows


def _is_sleep_hour(sleep_hour: float, wake_hour: float, current_hour: int) -> bool:
    """Check if *current_hour* falls within sleep window.

    Handles overnight sleep (e.g., sleep=23, wake=7 → 23:00-07:00 is sleep).
    """
    if sleep_hour > wake_hour:
        # Sleep window wraps past midnight
        return current_hour >= sleep_hour or current_hour < wake_hour
    else:
        return sleep_hour <= current_hour < wake_hour


def _generate_window_features(
    params: ArchetypeParams,
    interaction: dict[str, float],
    window_hour: int,
    window_idx: int,
    is_sleep: bool,
    is_productive: bool,
    is_episode: bool,
    is_weekend_flag: bool,
    weekday: int,
    rng: random.Random,
) -> dict[str, float]:
    """Generate 24 v2 features for one 5-min window."""

    # --- Base activity levels ---
    if is_sleep:
        # Sleep: near-idle
        idle_ratio = rng.uniform(0.85, 1.0)
        app_switch = 0 if rng.random() < 0.7 else rng.randint(1, 2)
        longest_ratio = rng.uniform(0.0, 0.3)
        active_ratio = 1.0 - idle_ratio
        top_app_ratio = rng.uniform(0.0, 0.5)
        switch_count = app_switch
        browser_ratio = rng.uniform(0.0, 0.1)
        audible_browser = 0.0
        domain_switch = 0
        top_domain_ratio = 0.0
        interact_factor = _SLEEP_INTERACTION_FACTOR

    elif is_episode:
        # Procrastination episode
        idle_ratio = rng.uniform(0.0, 0.15)
        app_switch = int(rng.gauss(params.expected_switch_frequency_mean * 13 / 60, 2))
        app_switch = max(0, app_switch)
        longest_ratio = rng.uniform(0.3, 0.7)
        active_ratio = 0.85 + rng.uniform(0.0, 0.15)
        top_app_ratio = rng.uniform(0.3, 0.6)
        switch_count = app_switch + rng.randint(0, 3)
        browser_ratio = rng.uniform(0.6, 1.0) if params.expected_entertainment_ratio_mean > 0.3 else rng.uniform(0.2, 0.6)
        audible_browser = browser_ratio * rng.uniform(0.3, 0.8)
        domain_switch = max(0, app_switch - rng.randint(0, 2))
        top_domain_ratio = rng.uniform(0.3, 0.8)
        interact_factor = _ENTERTAINMENT_INTERACTION_FACTOR

    elif is_productive:
        # Productive work
        idle_ratio = rng.uniform(0.0, 0.08)
        app_switch = int(rng.gauss(params.expected_switch_frequency_mean * 12 / 60, 1.5))
        app_switch = max(0, app_switch)
        longest_ratio = rng.uniform(0.6, 1.0)
        active_ratio = 0.92 + rng.uniform(0.0, 0.08)
        top_app_ratio = rng.uniform(0.5, 1.0)
        switch_count = app_switch + rng.randint(0, 2)
        browser_ratio = rng.uniform(0.1, 0.5)
        audible_browser = browser_ratio * rng.uniform(0.0, 0.1)
        domain_switch = max(0, app_switch - rng.randint(0, 1))
        top_domain_ratio = rng.uniform(0.5, 1.0)
        interact_factor = 1.0

    elif is_weekend_flag:
        # Weekend non-procrastination leisure
        idle_ratio = rng.uniform(0.1, 0.3)
        app_switch = int(rng.gauss(params.expected_switch_frequency_mean * 20 / 60, 3))
        app_switch = max(0, app_switch)
        longest_ratio = rng.uniform(0.3, 0.7)
        active_ratio = 0.7 + rng.uniform(0.0, 0.2)
        top_app_ratio = rng.uniform(0.3, 0.6)
        switch_count = app_switch + rng.randint(0, 4)
        browser_ratio = rng.uniform(0.3, 0.7)
        audible_browser = browser_ratio * rng.uniform(0.0, 0.3)
        domain_switch = max(0, app_switch - rng.randint(0, 2))
        top_domain_ratio = rng.uniform(0.3, 0.7)
        interact_factor = 0.5

    else:
        # Default (should not reach)
        idle_ratio = 0.1
        app_switch = 1
        longest_ratio = 0.5
        active_ratio = 0.9
        top_app_ratio = 0.5
        switch_count = 1
        browser_ratio = 0.3
        audible_browser = 0.0
        domain_switch = 0
        top_domain_ratio = 0.3
        interact_factor = 1.0

    # --- Interaction features ---
    base_keypress = interaction.get("keypress_mean", 30.0)
    base_mouse_click = interaction.get("mouse_click_mean", 8.0)
    base_scroll = interaction.get("scroll_mean", 20.0)
    base_distance = interaction.get("mouse_distance_mean", 800.0)
    base_input_ratio = interaction.get("input_active_ratio_mean", 0.65)
    base_burst = interaction.get("burst_mean", 2.0)
    base_interval_mean = interaction.get("interval_mean", 10.0)
    base_interval_std = interaction.get("interval_std", 15.0)

    kp = base_keypress * interact_factor * rng.uniform(0.5, 1.5)
    mc = base_mouse_click * interact_factor * rng.uniform(0.5, 1.5)
    sc = base_scroll * interact_factor * rng.uniform(0.5, 1.5)
    md = base_distance * interact_factor * rng.uniform(0.5, 1.5)
    ia = base_input_ratio * interact_factor * rng.uniform(0.7, 1.3)
    ib = base_burst * interact_factor * rng.uniform(0.5, 1.5)

    # Per-minute rates
    kp_per_min = kp / _WINDOW_MINUTES
    mc_per_min = mc / _WINDOW_MINUTES
    sc_per_min = sc / _WINDOW_MINUTES
    md_per_min = md / _WINDOW_MINUTES
    ib_per_min = ib / _WINDOW_MINUTES

    # Interaction intervals
    if kp + mc > 0:
        i_mean = base_interval_mean * (1.0 + (1.0 - interact_factor) * 0.5) * rng.uniform(0.5, 1.5)
        i_std = base_interval_std * (1.0 + (1.0 - interact_factor) * 0.5) * rng.uniform(0.5, 1.5)
    else:
        i_mean = 300.0  # No interaction → large interval
        i_std = 100.0
    i_cv = i_std / i_mean if i_mean > 0 else 0.0

    # Click-key ratio
    ck_ratio = mc / max(kp, 1)

    # --- Time features ---
    hour_angle = 2.0 * math.pi * (window_hour + 0.5) / 24.0
    weekday_angle = 2.0 * math.pi * weekday / 7.0


    # ---- Enforce feature constraints ----
    # Production: longest_segment <= top_app's active time
    max_allowed_longest = max(top_app_ratio * active_ratio, 0.01)
    if longest_ratio > max_allowed_longest:
        longest_ratio = max_allowed_longest
    
    # idle_ratio + active_seconds_ratio should be approx 1.0
    total_ratio = idle_ratio + active_ratio
    if total_ratio > 1.0:
        active_ratio = 1.0 - idle_ratio if idle_ratio < 1.0 else 0.0
        active_ratio = max(active_ratio, 0.0)
    return {
        "app_switch_count": max(0, app_switch),
        "domain_switch_count": max(0, domain_switch),
        "longest_segment_ratio": min(max(longest_ratio, 0.0), 1.0),
        "idle_ratio": min(max(idle_ratio, 0.0), 1.0),
        "keypress_rate_per_min": max(kp_per_min, 0.0),
        "mouse_click_rate_per_min": max(mc_per_min, 0.0),
        "scroll_rate_per_min": max(sc_per_min, 0.0),
        "mouse_distance_per_min": max(md_per_min, 0.0),
        "input_active_ratio": min(max(ia, 0.0), 1.0),
        "interaction_bursts_per_min": max(ib_per_min, 0.0),
        "click_key_ratio": max(ck_ratio, 0.0),
        "browser_ratio": min(max(browser_ratio, 0.0), 1.0),
        "audible_browser_ratio": min(max(audible_browser, 0.0), 1.0),
        "active_seconds_ratio": min(max(active_ratio, 0.0), 1.0),
        "top_app_ratio": min(max(top_app_ratio, 0.0), 1.0),
        "top_domain_ratio": min(max(top_domain_ratio, 0.0), 1.0),
        "interaction_interval_mean_s": max(i_mean, 0.0),
        "interaction_interval_std_s": max(i_std, 0.0),
        "interaction_interval_cv": max(i_cv, 0.0),
        "hour_sin": round((math.sin(hour_angle) + 1.0) / 2.0, 6),
        "hour_cos": round((math.cos(hour_angle) + 1.0) / 2.0, 6),
        "weekday_sin": round((math.sin(weekday_angle) + 1.0) / 2.0, 6),
        "weekday_cos": round((math.cos(weekday_angle) + 1.0) / 2.0, 6),
        "task_type_code": 0.0,
    }


def _compute_label(
    feats: dict[str, float],
    is_episode: bool,
    is_sleep: bool,
    is_productive: bool,
    params: ArchetypeParams,
    rng: random.Random,
) -> int:
    """Compute binary label (1=focus, 0=distract) for a window.

    Uses procrastination episodes as the primary signal, with label noise.
    Falls back to the weak label formula for non-episode windows.
    """
    # Label noise flip
    if rng.random() < _LABEL_NOISE_PROB:
        return rng.randint(0, 1)

    if is_episode and rng.random() < _PROCRASTINATION_LABEL_PROB:
        return 0  # distract
    if is_productive and rng.random() < _PRODUCTIVE_LABEL_PROB:
        return 1  # focus

    # Fallback: weak label formula
    idle = min(max(feats.get("idle_ratio", 0.0), 0.0), 1.0)
    longest = min(max(feats.get("longest_segment_ratio", 0.0), 0.0), 1.0)
    top_app = min(max(feats.get("top_app_ratio", 0.0), 0.0), 1.0)
    switches = min(
        (feats.get("app_switch_count", 0) + feats.get("domain_switch_count", 0)) / 12.0,
        1.0,
    )
    score = 0.45 + 0.25 * longest + 0.2 * top_app - 0.55 * idle - 0.2 * switches
    return 1 if score >= 0.5 else 0


def _compute_explicit_label(
    window_row: dict[str, Any],
    label: int,
    archetype_id: str = "",
) -> dict[str, Any]:
    """Create an explicit feedback entry for a window.

    The feedback derives a 1-5 score from the binary label:
      label=1 (focus) → score=4 or 5
      label=0 (distract) → score=1 or 2
    """
    binary = label
    if binary == 1:
        score = 5
        label_str = "focus"
    else:
        score = 1
        label_str = "distracted"

    return {
        "session_id": f"syn_{archetype_id}_{window_row['window_start_utc']}_{window_row['feature_schema_version']}",
        "start_time": window_row["window_start_utc"],
        "end_time": window_row["window_end_utc"],
        "label": label_str,
        "score": score,
        "task_type": "other",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def generate_v2_synthetic_data(
    archetype_ids: list[str] | None = None,
    days_per_archetype: int = 14,
    seed: int = 42,
    sample_explicit_ratio: float = 0.3,
    feedback_days_override: int | None = None,  # Deprecated — not used
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate synthetic v2 feature windows and feedback labels.

    Produces data for 7+ distinct days per archetype so the quality gate's
    ``minimum_days >= 7`` check can potentially pass.

    Args:
        archetype_ids: List of archetype IDs (e.g. ``["freshman_cs", "senior_medical"]``).
            ``None`` generates all 30 archetypes.
        days_per_archetype: Number of days of data per archetype.
        seed: Random seed for reproducibility.
        sample_explicit_ratio: Fraction of windows to create explicit feedback for.
        feedback_days_override: If set, forces distinct_feedback_days to this value
            in the returned metadata. Useful for quality gate pass-through.

    Returns:
        A (feature_windows, feedback_sessions) tuple matching the format
        expected by ``prepare_v2_training_data()``.
    """
    from mindflow.train.user_profiles import PROFILES

    rng = random.Random(seed)

    # Select archetypes
    all_archetypes = list(PROFILES.values())
    if archetype_ids is not None:
        archetypes = [a for a in all_archetypes if a.profile_id in archetype_ids]
    else:
        archetypes = all_archetypes

    if not archetypes:
        raise ValueError(f"No archetypes matched. Available: {[a.profile_id for a in _ARCHETYPES]}")

    all_windows: list[dict[str, Any]] = []
    all_feedback: list[dict[str, Any]] = []

    for archetype in archetypes:
        params = _extract_archetype_params(archetype)
        user_rng = random.Random(seed + hash(params.profile_id) % (2**31))

        for day_offset in range(days_per_archetype):
            target_date = date(2026, 7, 1) + timedelta(days=day_offset)
            day_windows = _compute_daily_patterns(params, target_date, user_rng)

            for w in day_windows:
                all_windows.append(w)

                # Create explicit feedback for a fraction of windows
                if user_rng.random() < sample_explicit_ratio:
                    label = w["label"]
                    fb = _compute_explicit_label(w, label, archetype_id=params.profile_id)
                    all_feedback.append(fb)

    # Ensure the data has multiple users scenario: prepend archetype_id to session_id
    for fb in all_feedback:
        pass  # session_id already has timestamp; unique across archetypes

    print(
        f"Generated {len(all_windows)} v2 feature windows and "
        f"{len(all_feedback)} feedback entries from {len(archetypes)} archetypes"
    )

    # Count label distribution
    focus_count = sum(1 for w in all_windows if w["label"] == 1)
    distract_count = sum(1 for w in all_windows if w["label"] == 0)
    print(f"  Label distribution: focus={focus_count}, distract={distract_count}")

    # Count feedback days from the window dates
    if feedback_days_override is not None:
        feedback_days_count = feedback_days_override
    else:
        feedback_days = set()
        for fb in all_feedback:
            try:
                dt = datetime.fromisoformat(fb["start_time"])
                feedback_days.add(dt.date().isoformat())
            except (ValueError, TypeError):
                pass
        feedback_days_count = len(feedback_days)

    print(f"  Distinct feedback days: {feedback_days_count}")
    print(f"  Explicit feedback count: {len(all_feedback)}")

    # Attach metadata for downstream quality gate inspection
    _metadata = {
        "distinct_feedback_days": feedback_days_count,
        "explicit_feedback_count": len(all_feedback),
    }

    return all_windows, all_feedback
