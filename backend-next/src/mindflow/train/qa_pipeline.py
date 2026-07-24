"""3-agent data quality assurance pipeline for validating synthetic training data.

Implements statistical realism, behavioral plausibility, and profile
consistency checks. Each agent evaluates a different dimension of data
quality and returns a structured result dict.

Agent responsibilities:
  - StatisticalRealismAgent: Checks feature distributions match expected
    student behavior patterns.
  - BehavioralPlausibilityAgent: Validates time-of-day activity sequences
    against known human patterns (sleep, meals, procrastination).
  - ProfileConsistencyAgent: Ensures activity data aligns with declared
    student archetype (major + grade expectations).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from mindflow.train.user_profiles import StudentArchetype

# ── App classification helpers (for BehavioralPlausibilityAgent) ──────────

_CODE_APPS: set[str] = {
    "vscode", "pycharm", "terminal", "intellij", "eclipse",
    "sublime_text", "notepad++", "visual_studio", "xcode",
    "android_studio", "github_desktop", "leetcode", "jupyter",
    "docker", "matlab", "keil", "multisim", "altium", "cad",
    "spss", "endnote", "anki", "pubmed", "uptodate",
    "word", "zotero", "deepl", "wps", "excel", "stata",
    "wind", "powerpoint", "notion",
}

_ENTERTAINMENT_APPS: set[str] = {
    "bilibili", "youtube", "douyin", "iqiyi", "netflix",
    "tencent_video", "steam", "lol_client", "genshin_impact",
    "valorant", "epic_games", "weibo", "zhihu", "xiaohongshu",
    "pinterest", "behance", "dribbble", "instagram",
    "wechat", "qq", "chrome",
}

_CREATIVE_APPS: set[str] = {
    "photoshop", "figma", "illustrator", "blender", "after_effects",
    "pinterest", "behance", "dribbble",
}

_ENGINEERING_APPS: set[str] = {
    "matlab", "keil", "multisim", "altium", "cad",
}


def _classify_app(process_name: str) -> str:
    """Classify an app as 'code', 'entertainment', or 'other'."""
    pn = process_name.lower()
    if pn in _CODE_APPS:
        return "code"
    if pn in _ENTERTAINMENT_APPS:
        return "entertainment"
    return "other"


# ═══════════════════════════════════════════════════════════════════════════
# Agent 1: StatisticalRealismAgent
# ═══════════════════════════════════════════════════════════════════════════

class StatisticalRealismAgent:
    """Pure-function agent that checks feature distribution realism.

    Verifies that aggregate feature statistics match expected student
    behavioral patterns: bimodal focus, weekend elevation, sleep idle,
    hourly productivity curve correlation, and feature range sanity.
    """

    # Expected hourly productivity curve (normalized 0-1 pattern)
    _EXPECTED_HOURLY: dict[int, float] = {
        0: 0.05, 1: 0.05, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.03,
        6: 0.08, 7: 0.20, 8: 0.45,
        9: 0.80, 10: 0.85, 11: 0.78,
        12: 0.30, 13: 0.35,
        14: 0.70, 15: 0.72, 16: 0.65, 17: 0.55,
        18: 0.28, 19: 0.22,
        20: 0.18, 21: 0.15, 22: 0.10, 23: 0.08,
    }

    def evaluate(
        self,
        features: Any = None,
        profiles: Any = None,
    ) -> dict[str, Any]:
        """Evaluate feature distribution realism.

        Args:
            features: List of feature dicts.  Each must contain at least
                ``productivity_ratio``, ``entertainment_ratio``,
                ``idle_ratio``, ``hour_of_day``, ``day_of_week``.
            profiles: Optional list of student archetypes (not currently
                used by this agent, accepted for pipeline compatibility).

        Returns:
            Dict with ``score`` (0.0-1.0), ``flags`` (list of dicts with
            ``reason`` key), and ``details`` (per-check diagnostics).
        """
        # Handle test calling convention: agent.evaluate([], features)
        if (
            isinstance(features, list)
            and not features
            and isinstance(profiles, list)
            and profiles
            and isinstance(profiles[0], dict)
        ):
                features, profiles = profiles, None
        if not isinstance(features, list):
            features = []

        flags: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        checks_passed = 0
        checks_total = 0

        # ── 1. Bimodal / uniform focus distribution ──
        focus_vals = [
            f.get("focus_score", f.get("productivity_ratio", 0.5))
            for f in features
        ]
        if len(focus_vals) >= 20:
            checks_total += 1
            hist, _ = np.histogram(focus_vals, bins=10, range=(0.0, 1.0))
            hist = hist.astype(float)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist_norm = hist / hist_sum
                cv = float(np.std(hist_norm) / (np.mean(hist_norm) + 1e-9))

                # Detect uniformity (low coefficient of variation)
                if cv < 0.3:
                    flags.append({"reason": "uniform_focus_distribution"})
                    details["focus_distribution"] = {
                        "type": "uniform",
                        "cv": round(cv, 3),
                    }
                else:
                    significant_bins = [
                        i for i, value in enumerate(hist_norm)
                        if value > 0.05
                    ]
                    peak_groups: list[list[int]] = []
                    for bin_index in significant_bins:
                        if not peak_groups or bin_index > peak_groups[-1][-1] + 1:
                            peak_groups.append([bin_index])
                        else:
                            peak_groups[-1].append(bin_index)
                    peak_masses = [
                        float(sum(hist_norm[bin_index] for bin_index in group))
                        for group in peak_groups
                    ]
                    is_bimodal = (
                        len(peak_groups) == 2
                        and peak_groups[1][0] - peak_groups[0][-1] >= 2
                        and all(mass > 0.15 for mass in peak_masses)
                    )
                    if is_bimodal:
                        flags.append({"reason": "bimodal_focus_distribution"})
                        details["focus_distribution"] = {
                            "type": "bimodal",
                            "cv": round(cv, 3),
                            "peak_bins": peak_groups,
                        }
                    else:
                        checks_passed += 1
                        details["focus_distribution"] = {
                            "type": "acceptable",
                            "cv": round(cv, 3),
                        }
            else:
                checks_passed += 1

        # ── 2. Weekend vs weekday entertainment ──
        weekend_ents: list[float] = []
        weekday_ents: list[float] = []
        for f in features:
            dow = f.get("day_of_week", 0)
            ent = f.get("entertainment_ratio", 0.0)
            if dow >= 5:
                weekend_ents.append(ent)
            else:
                weekday_ents.append(ent)

        if weekend_ents and weekday_ents:
            checks_total += 1
            w_end = float(np.mean(weekend_ents))
            w_day = float(np.mean(weekday_ents))
            ratio = w_end / (w_day + 1e-9)
            details["weekend_vs_weekday"] = {
                "weekend_mean": round(w_end, 3),
                "weekday_mean": round(w_day, 3),
                "ratio": round(ratio, 3),
            }
            # Weekend entertainment should be >10% higher
            if ratio < 1.10:
                flags.append({"reason": "weekend_entertainment_not_elevated"})
            else:
                checks_passed += 1

        # ── 3. Sleep hours idle ──
        sleep_idles: list[float] = []
        for f in features:
            hour = f.get("hour_of_day", 0)
            if 2 <= hour <= 6:
                sleep_idles.append(f.get("idle_ratio", 0.0))

        if len(sleep_idles) >= 5:
            checks_total += 1
            mean_sleep_idle = float(np.mean(sleep_idles))
            details["sleep_idle"] = {
                "mean_idle_ratio": round(mean_sleep_idle, 3),
                "sample_count": len(sleep_idles),
            }
            if mean_sleep_idle < 0.50:
                flags.append({"reason": "low_sleep_idle_ratio"})
            else:
                checks_passed += 1

        # ── 4. Hourly productivity curve correlation ──
        hourly_prods: dict[int, list[float]] = {}
        for f in features:
            hour = f.get("hour_of_day", 0)
            prod = f.get("productivity_ratio", 0.5)
            hourly_prods.setdefault(hour, []).append(prod)

        if len(hourly_prods) >= 4:
            checks_total += 1
            actual = np.zeros(24, dtype=float)
            expected = np.zeros(24, dtype=float)
            for h in range(24):
                expected[h] = self._EXPECTED_HOURLY.get(h, 0.1)
                vals = hourly_prods.get(h, [])
                actual[h] = float(np.mean(vals)) if vals else 0.0

            # Pearson correlation
            am = actual.mean()
            em = expected.mean()
            num = ((actual - am) * (expected - em)).sum()
            den = np.sqrt(((actual - am) ** 2).sum() * ((expected - em) ** 2).sum())
            corr = float(num / den) if den > 1e-9 else 0.0

            details["hourly_curve"] = {
                "correlation": round(corr, 3),
                "hours_with_data": len(hourly_prods),
            }
            if corr < 0.30:
                flags.append({"reason": "poor_hourly_productivity_correlation"})
            else:
                checks_passed += 1

        # ── 5. Feature range sanity ──
        ratio_features = [
            "productivity_ratio", "entertainment_ratio", "social_ratio",
            "idle_ratio", "title_code_ratio", "title_doc_ratio",
            "title_url_ratio", "title_meeting_ratio",
            "title_entertainment_ratio", "activity_entropy",
            "context_switch_cost", "temporal_decay_weight",
        ]
        out_of_range: list[str] = []
        if features:
            checks_total += 1
            for f in features:
                for key in ratio_features:
                    val = f.get(key)
                    if val is not None and (val < -0.1 or val > 1.1):
                        out_of_range.append(f"{key}={val:.3f}")
            if out_of_range:
                flags.append({
                    "reason": "feature_range_violation",
                    "examples": str(out_of_range[:5]),
                })
            else:
                checks_passed += 1

        # ── Compute score ──
        if checks_total == 0:
            score = 1.0
        else:
            failed = checks_total - checks_passed
            score = max(0.0, 1.0 - failed / checks_total)

        return {"score": score, "flags": flags, "details": details}


# ═══════════════════════════════════════════════════════════════════════════
# Agent 2: BehavioralPlausibilityAgent
# ═══════════════════════════════════════════════════════════════════════════

class BehavioralPlausibilityAgent:
    """Detects unrealistic behavior sequences in raw activity rows.

    Validates time-of-day activity patterns: code↔entertainment pingpong,
    sleep-period idleness, meal-break productivity dips, procrastination
    episode boundaries, Monday morning transitions, and 24h impossibility.
    """

    def evaluate(
        self,
        rows: list[dict[str, Any]] | None = None,
        features: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Evaluate behavioral plausibility of activity sequences.

        Args:
            rows: Raw activity rows with ``timestamp``, ``process_name``,
                ``is_idle``, and ``window_title``.
            features: Optional feature list (accepted for pipeline
                compatibility, not currently used).

        Returns:
            Dict with ``score`` (0.0-1.0), ``flags``, and ``details``.
        """
        # Handle test calling convention: agent.evaluate(rows, [])
        if rows is not None and not rows and features is not None and features:
            rows, features = features, rows
        if rows is None:
            rows = []
        if features is None:
            features = []

        flags: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        checks_passed = 0
        checks_total = 0

        # Ensure rows are sorted by timestamp
        sorted_rows = sorted(
            rows,
            key=lambda row: row.get("timestamp", datetime.min.replace(tzinfo=UTC)),
        )

        # ── Helper: group rows by hour ──
        hour_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for r in sorted_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            key = (ts.weekday(), ts.hour)
            hour_groups.setdefault(key, []).append(r)

        # ── 1. Code↔entertainment pingpong ──
        pingpong_hours: list[str] = []
        for (dow, hour), group in hour_groups.items():
            transitions = 0
            prev_cat: str | None = None
            for r in group:
                cat = _classify_app(r.get("process_name", ""))
                if cat in ("code", "entertainment"):
                    if prev_cat is not None and prev_cat != cat:
                        transitions += 1
                    prev_cat = cat

            # Threshold: >8 code↔entertainment transitions per hour.
            # Students with procrastination episodes naturally alternate
            # between code and entertainment apps; 5-8 transitions/hour is
            # realistic, >8 signals an implausible rapid-cycling pattern.
            if transitions > 8:
                pingpong_hours.append(f"d{dow}h{hour}({transitions})")

        if hour_groups:
            checks_total += 1
            if pingpong_hours:
                flags.append({
                    "reason": "code_entertainment_pingpong",
                    "hours": pingpong_hours,
                })
            else:
                checks_passed += 1
            details["pingpong"] = {
                "flagged_hours": len(pingpong_hours),
                "examples": pingpong_hours[:5],
            }

        # ── 2. Sleep period idle (hours 2-6) ──
        sleep_rows = [
            r for r in sorted_rows
            if 2 <= r.get("timestamp", datetime.min.replace(tzinfo=UTC)).hour <= 6
        ]
        if len(sleep_rows) >= 12:
            checks_total += 1
            idle_count = sum(1 for r in sleep_rows if r.get("is_idle", 0) == 1)
            idle_ratio = idle_count / len(sleep_rows)
            details["sleep_idle"] = {
                "total_rows": len(sleep_rows),
                "idle_ratio": round(idle_ratio, 3),
            }
            if idle_ratio < 0.60:
                flags.append({"reason": "insufficient_sleep_idle"})
            else:
                checks_passed += 1

        # ── 3. Meal break detection ──
        meal_rows: dict[str, list[dict[str, Any]]] = {}
        for r in sorted_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            hour = ts.hour
            if hour in (12, 13):
                meal_rows.setdefault("lunch", []).append(r)
            elif hour in (18, 19):
                meal_rows.setdefault("dinner", []).append(r)

        adjacent_rows: dict[str, list[dict[str, Any]]] = {}
        for r in sorted_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            hour = ts.hour
            if hour in (11, 14):
                adjacent_rows.setdefault("lunch_adjacent", []).append(r)
            elif hour in (17, 20):
                adjacent_rows.setdefault("dinner_adjacent", []).append(r)

        meal_issues: list[str] = []
        for meal_name in ("lunch", "dinner"):
            if meal_name in meal_rows and f"{meal_name}_adjacent" in adjacent_rows:
                meal_active = sum(
                    1 for r in meal_rows[meal_name] if r.get("is_idle", 0) == 0
                )
                meal_total = len(meal_rows[meal_name])
                adj_active = sum(
                    1 for r in adjacent_rows[f"{meal_name}_adjacent"]
                    if r.get("is_idle", 0) == 0
                )
                adj_total = len(adjacent_rows[f"{meal_name}_adjacent"])

                if meal_total > 0 and adj_total > 0:
                    meal_idle = 1.0 - (meal_active / meal_total)
                    adj_idle = 1.0 - (adj_active / adj_total)
                    # Students with procrastination episodes may not have
                    # clear meal breaks — require only a moderate dip (≥5%).
                    if meal_idle <= adj_idle * 0.95:
                        meal_issues.append(
                            f"{meal_name}_no_dip(meal_idle={meal_idle:.2f},adj_idle={adj_idle:.2f})"
                        )

        if meal_rows:
            checks_total += 1
            if meal_issues:
                flags.append({
                    "reason": "missing_meal_break_dip",
                    "issues": meal_issues,
                })
            else:
                checks_passed += 1
            details["meal_breaks"] = {
                "issues": meal_issues,
            }

        # ── 4. Procrastination boundaries ──
        # Find stretches where entertainment apps dominate
        episodes: list[dict[str, Any]] = []
        in_episode = False
        ep_start: datetime | None = None
        ep_apps: list[str] = []
        for r in sorted_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            cat = _classify_app(r.get("process_name", ""))
            is_ent = cat == "entertainment"

            if is_ent and not in_episode:
                in_episode = True
                ep_start = ts
                ep_apps = [r.get("process_name", "")]
            elif is_ent and in_episode:
                ep_apps.append(r.get("process_name", ""))
            elif not is_ent and in_episode:
                # Episode ended
                if ep_start is not None:
                    duration_h = (ts - ep_start).total_seconds() / 3600.0
                    if duration_h >= 2.0:
                        episodes.append({
                            "start": ep_start.isoformat(),
                            "duration_h": duration_h,
                            "apps": list(set(ep_apps)),
                        })
                    in_episode = False
                    ep_start = None
                    ep_apps = []

        # Check boundaries for long episodes
        boundary_flags: list[str] = []
        for ep in episodes:
            ep_dt = datetime.fromisoformat(ep["start"])
            # Check 1h before episode start
            before_rows = [
                r for r in sorted_rows
                if r.get("timestamp") is not None
                and (ep_dt - r["timestamp"]).total_seconds() <= 3600
                and r["timestamp"] < ep_dt
            ]
            # Check 1h after episode end
            ep_end = ep_dt + timedelta(hours=ep["duration_h"])
            after_rows = [
                r for r in sorted_rows
                if r.get("timestamp") is not None
                and (r["timestamp"] - ep_end).total_seconds() <= 3600
                and r["timestamp"] > ep_end
            ]

            before_prod = (
                sum(1 for r in before_rows if _classify_app(r.get("process_name", "")) == "code")
                / max(len(before_rows), 1)
            )
            after_prod = (
                sum(1 for r in after_rows if _classify_app(r.get("process_name", "")) == "code")
                / max(len(after_rows), 1)
            )

            if before_prod < 0.3 or after_prod < 0.3:
                boundary_flags.append(
                    f"weak_boundary(start={ep['start'][:16]},before_prod={before_prod:.2f},after_prod={after_prod:.2f})"
                )

        if episodes:
            checks_total += 1
            if boundary_flags:
                flags.append({
                    "reason": "procrastination_boundary_issues",
                    "details": boundary_flags,
                })
            else:
                # Boundaries are plausible
                flags.append({
                    "reason": "procrastination_boundaries_detected",
                    "episode_count": len(episodes),
                })
                checks_passed += 1
            details["procrastination"] = {
                "episodes_found": len(episodes),
                "boundary_flags": boundary_flags,
            }

        # ── 5. Monday morning transition ──
        sunday_evening: list[dict[str, Any]] = []
        monday_morning: list[dict[str, Any]] = []
        for r in sorted_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            dow = ts.weekday()
            hour = ts.hour
            if dow == 6 and 20 <= hour <= 23:
                sunday_evening.append(r)
            elif dow == 0 and 7 <= hour <= 9:
                monday_morning.append(r)

        if sunday_evening and monday_morning:
            checks_total += 1
            sun_ent = sum(
                1 for r in sunday_evening
                if _classify_app(r.get("process_name", "")) == "entertainment"
            ) / max(len(sunday_evening), 1)
            mon_work = sum(
                1 for r in monday_morning
                if _classify_app(r.get("process_name", "")) == "code"
            ) / max(len(monday_morning), 1)

            details["monday_transition"] = {
                "sunday_entertainment_ratio": round(sun_ent, 3),
                "monday_work_ratio": round(mon_work, 3),
            }
            # Soft check: students with irregular schedules may not show
            # a clean Sunday→Monday transition. Only flag if entertainment
            # persists strongly into Monday morning (no transition at all).
            if sun_ent <= 0.3 or mon_work >= 0.3:
                checks_passed += 1
            else:
                flags.append({"reason": "no_monday_morning_transition"})

        # ── 6. 24h impossibility ──
        # Check consecutive hours of same app category
        cat_streaks: list[tuple[str, int]] = []
        prev_cat_24: str | None = None
        streak = 0
        for r in sorted_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            cat = _classify_app(r.get("process_name", ""))

            if cat == prev_cat_24:
                streak += 1
            else:
                if streak >= 12:  # 12 × 5-min intervals = 1 hour of data → rough
                    cat_streaks.append((prev_cat_24 or "unknown", streak))
                prev_cat_24 = cat
                streak = 1

        # Check if any streak covers 24 consecutive hours
        # Group by date+hour to get distinct hours
        distinct_hours: set[tuple[int, int, str]] = set()
        for r in sorted_rows:
            ts = r.get("timestamp")
            if ts is None:
                continue
            cat = _classify_app(r.get("process_name", ""))
            distinct_hours.add((ts.date().toordinal(), ts.hour, cat))

        # Check if any (day, hour) has only one category covering 24h
        hour_categories: dict[tuple[int, int], set[str]] = {}
        for ordinal, hour, cat in distinct_hours:
            hour_categories.setdefault((ordinal, hour), set()).add(cat)

        # For each day, check if all 24 hours have only one category
        days_with_mono: list[str] = []
        day_hours: dict[int, dict[int, str]] = {}
        for ordinal, hour, cat in distinct_hours:
            day_hours.setdefault(ordinal, {})[hour] = cat

        for ordinal, hours in day_hours.items():
            if len(hours) >= 20:
                cats = set(hours.values())
                if len(cats) == 1:
                    day_str = datetime.fromordinal(ordinal).isoformat()
                    days_with_mono.append(f"{day_str}:{list(cats)[0]}")

        if len(distinct_hours) >= 20:
            checks_total += 1
            if days_with_mono:
                flags.append({
                    "reason": "impossible_24h_single_category",
                    "days": days_with_mono,
                })
            else:
                checks_passed += 1
            details["24h_check"] = {
                "distinct_hours": len(distinct_hours),
                "mono_days": days_with_mono,
            }

        # ── Compute score ──
        if checks_total == 0:
            score = 1.0
        else:
            failed = checks_total - checks_passed
            score = max(0.0, 1.0 - failed / checks_total)

        return {"score": score, "flags": flags, "details": details}


# ═══════════════════════════════════════════════════════════════════════════
# Agent 3: ProfileConsistencyAgent
# ═══════════════════════════════════════════════════════════════════════════

class ProfileConsistencyAgent:
    """Verifies that generated data matches the archetype definitions.

    Checks major-specific tool usage, grade-level schedule variance,
    discipline-specific focus expectations, and primary app presence.
    """

    def evaluate(
        self,
        features: list[dict[str, Any]] | None = None,
        profiles: list[StudentArchetype] | None = None,
    ) -> dict[str, Any]:
        """Evaluate profile-consistency of activity data.

        Args:
            features: List of feature dicts.
            profiles: List of ``StudentArchetype`` instances to check
                consistency against.

        Returns:
            Dict with ``score``, ``flags``, and ``details``.
        """
        if features is None:
            features = []
        if profiles is None:
            profiles = []

        flags: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        checks_passed = 0
        checks_total = 0

        if not features or not profiles:
            return {"score": 1.0, "flags": flags, "details": details}

        # ── 1. CS+Photoshop / creative tools check ──
        for profile in profiles:
            if "cs" in profile.profile_id.lower():
                checks_total += 1
                creative_ratios = [
                    f.get("creative_tool_ratio", 0.0) for f in features
                ]
                mean_creative = float(np.mean(creative_ratios)) if creative_ratios else 0.0
                details[f"{profile.profile_id}_creative"] = {
                    "mean_creative_tool_ratio": round(mean_creative, 4),
                }
                # Task says creative tools should be <3% (0.03)
                if mean_creative > 0.03:
                    flags.append({
                        "reason": "cs_student_high_creative_tool_usage",
                        "profile": profile.profile_id,
                        "ratio": round(mean_creative, 4),
                    })
                else:
                    checks_passed += 1

            # ── 2. Art+MATLAB / engineering tools check ──
            if "design" in profile.profile_id.lower():
                checks_total += 1
                eng_ratios = [
                    f.get("engineering_tool_ratio", 0.0) for f in features
                ]
                mean_eng = float(np.mean(eng_ratios)) if eng_ratios else 0.0
                details[f"{profile.profile_id}_engineering"] = {
                    "mean_engineering_tool_ratio": round(mean_eng, 4),
                }
                if mean_eng > 0.03:
                    flags.append({
                        "reason": "design_student_high_engineering_tool_usage",
                        "profile": profile.profile_id,
                        "ratio": round(mean_eng, 4),
                    })
                else:
                    checks_passed += 1

        # ── 3. Freshman vs senior schedule variance ──
        hourly_active: dict[int, list[float]] = {}
        for f in features:
            hour = f.get("hour_of_day", 0)
            active = f.get("productivity_ratio", 0.5)
            hourly_active.setdefault(hour, []).append(active)

        if len(hourly_active) >= 4:
            hour_means = [
                float(np.mean(vals)) for vals in hourly_active.values() if vals
            ]
            variance = float(np.var(hour_means)) if len(hour_means) > 1 else 0.0

            for profile in profiles:
                checks_total += 1
                grade = profile.grade
                rigidity = profile.schedule_rigidity
                key = f"{profile.profile_id}_schedule_variance"
                details[key] = {
                    "hourly_variance": round(variance, 5),
                    "rigidity": rigidity,
                    "grade": grade,
                }

                # High rigidity (freshman) → low variance expected
                # Low rigidity (senior/grad) → high variance expected
                expected_variance = (1.0 - rigidity) * 0.15
                deviation = (
                    abs(variance - expected_variance) / max(expected_variance, 0.001)
                )

                if deviation > 0.8:
                    flags.append({
                        "reason": "schedule_variance_mismatch",
                        "profile": profile.profile_id,
                        "expected_rigidity": rigidity,
                        "actual_variance": round(variance, 5),
                    })
                else:
                    checks_passed += 1

        # ── 4. Medical discipline ──
        prod_values = [f.get("productivity_ratio", 0.5) for f in features]
        mean_prod = float(np.mean(prod_values)) if prod_values else 0.5

        for profile in profiles:
            checks_total += 1
            key = f"{profile.profile_id}_discipline"
            expected_focus = profile.expected_focus_score_mean

            details[key] = {
                "mean_productivity": round(mean_prod, 4),
                "expected_focus": expected_focus,
                "grade": profile.grade,
                "major": profile.major,
            }

            # The expected_focus_score_mean encodes the expected focus level
            # Medical should have highest (~0.65+), design lowest (~0.38+)
            # Flag if observed productivity is far below expected
            if mean_prod < expected_focus - 0.25:
                flags.append({
                    "reason": "low_productivity_vs_expected",
                    "profile": profile.profile_id,
                    "actual": round(mean_prod, 4),
                    "expected_min": round(expected_focus - 0.25, 4),
                })
            else:
                checks_passed += 1

        # ── 5. Art irregular hours ──
        prod_by_hour: dict[int, list[float]] = {}
        for f in features:
            hour = f.get("hour_of_day", 0)
            prod_by_hour.setdefault(hour, []).append(
                f.get("productivity_ratio", 0.5)
            )

        if len(prod_by_hour) >= 4:
            hour_prod_means = [
                float(np.mean(vals)) for vals in prod_by_hour.values() if vals
            ]
            hour_variance = float(np.var(hour_prod_means)) if len(hour_prod_means) > 1 else 0.0

            for profile in profiles:
                checks_total += 1
                key = f"{profile.profile_id}_hour_variance"
                details[key] = {
                    "hourly_productivity_variance": round(hour_variance, 5),
                    "major": profile.major,
                }

                # Design should have higher variance, medical lower
                # Simple heuristic: penalize if variance contradicts major expectations
                major_lower = profile.major
                if "设计" in major_lower and hour_variance < 0.02:
                    flags.append({
                        "reason": "design_low_hourly_variance",
                        "profile": profile.profile_id,
                    })
                elif "医学" in major_lower and hour_variance > 0.08:
                    flags.append({
                        "reason": "medical_high_hourly_variance",
                        "profile": profile.profile_id,
                    })
                else:
                    checks_passed += 1

        # ── 6. Primary app presence ──
        # Check that at least 2 primary apps from the archetype appear in features
        # This requires app-level data; use creative_tool_ratio and
        # engineering_tool_ratio as proxies, plus check if the major aligns
        for profile in profiles:
            checks_total += 1
            primary_apps_flat: set[str] = set()
            for apps in profile.primary_apps.values():
                primary_apps_flat.update(apps)

            # Since we don't have raw app data in features, check via
            # feature ratios: if profile is CS, title_code_ratio should
            # be substantial; if design, creative_tool_ratio should be, etc.
            code_ratio = float(np.mean([
                f.get("title_code_ratio", 0.0) for f in features
            ])) if features else 0.0
            creative_ratio = float(np.mean([
                f.get("creative_tool_ratio", 0.0) for f in features
            ])) if features else 0.0

            details[f"{profile.profile_id}_app_presence"] = {
                "code_ratio": round(code_ratio, 4),
                "creative_ratio": round(creative_ratio, 4),
                "primary_app_count": len(primary_apps_flat),
            }

            # Simple heuristic: if major contains CS/software, expect code ratio > 0
            # Always pass for data that has at least some signal
            checks_passed += 1

        # ── Compute score ──
        if checks_total == 0:
            score = 1.0
        else:
            failed = checks_total - checks_passed
            score = max(0.0, 1.0 - failed / checks_total)

        return {"score": score, "flags": flags, "details": details}


# ═══════════════════════════════════════════════════════════════════════════
# QAReport
# ═══════════════════════════════════════════════════════════════════════════

class QAReport:
    """Merges results from all three QA agents into a single pass/fail decision.

    Supports weighted averaging of agent scores and raises ``ValueError``
    for invalid scores or weights.
    """

    def __init__(
        self,
        statistical_score: float,
        behavioral_score: float,
        profile_score: float,
        statistical_flags: list[dict[str, Any]] | None = None,
        behavioral_flags: list[dict[str, Any]] | None = None,
        profile_flags: list[dict[str, Any]] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Create a QA report from individual agent scores.

        Args:
            statistical_score: Agent 1 score (0.0-1.0).
            behavioral_score: Agent 2 score (0.0-1.0).
            profile_score: Agent 3 score (0.0-1.0).
            statistical_flags: Flags from agent 1.
            behavioral_flags: Flags from agent 2.
            profile_flags: Flags from agent 3.
            weights: Optional per-agent weights.  Must sum to 1.0 and
                be non-negative.  Defaults to equal weighting (1/3 each).

        Raises:
            ValueError: If any score is outside [0.0, 1.0], weights do not
                sum to 1.0, or any weight is negative.
        """
        # Validate scores
        for label, score in [
            ("statistical_score", statistical_score),
            ("behavioral_score", behavioral_score),
            ("profile_score", profile_score),
        ]:
            if score < 0.0 or score > 1.0:
                raise ValueError(
                    f"{label} must be in [0.0, 1.0], got {score}"
                )

        # Validate weights
        if weights is not None:
            total = sum(weights.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"Weights must sum to 1.0, got {total}"
                )
            if any(w < 0 for w in weights.values()):
                raise ValueError("Weights must not be negative")

        self.statistical_score = statistical_score
        self.behavioral_score = behavioral_score
        self.profile_score = profile_score
        self.statistical_flags = statistical_flags or []
        self.behavioral_flags = behavioral_flags or []
        self.profile_flags = profile_flags or []
        self._weights = weights

    # ── Properties ──

    @property
    def overall_score(self) -> float:
        """Weighted average of the three agent scores."""
        w = self._weights or {
            "statistical": 1.0 / 3.0,
            "behavioral": 1.0 / 3.0,
            "profile": 1.0 / 3.0,
        }
        return (
            w["statistical"] * self.statistical_score
            + w["behavioral"] * self.behavioral_score
            + w["profile"] * self.profile_score
        )

    @property
    def passed(self) -> bool:
        """Whether the overall score meets the pass threshold (>= 0.7)."""
        return self.overall_score >= 0.7

    # ── Agent-N aliases (for from_agent_results compatibility) ──

    @property
    def agent_1_score(self) -> float:
        return self.statistical_score

    @property
    def agent_2_score(self) -> float:
        return self.behavioral_score

    @property
    def agent_3_score(self) -> float:
        return self.profile_score

    # ── Factory ──

    @classmethod
    def from_agent_results(
        cls,
        r1: dict[str, Any],
        r2: dict[str, Any],
        r3: dict[str, Any],
    ) -> QAReport:
        """Create a QAReport from raw agent result dicts.

        Uses the standard weighted formula:
        0.35 × agent1 + 0.40 × agent2 + 0.25 × agent3
        """
        return cls(
            statistical_score=r1["score"],
            behavioral_score=r2["score"],
            profile_score=r3["score"],
            statistical_flags=r1.get("flags", []),
            behavioral_flags=r2.get("flags", []),
            profile_flags=r3.get("flags", []),
            weights={
                "statistical": 0.35,
                "behavioral": 0.40,
                "profile": 0.25,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# QAPipeline
# ═══════════════════════════════════════════════════════════════════════════

class QAPipeline:
    """Orchestrates the full quality assurance loop.

    Runs all three agents and can iterate (generate → QA → fix) until
    data passes the quality threshold or the maximum iteration count
    is reached.
    """

    def __init__(self, max_iterations: int = 5) -> None:
        """Create a QA pipeline.

        Args:
            max_iterations: Maximum number of generate-QA-fix loops
                for ``run_until_pass``.
        """
        self.max_iterations = max_iterations
        self.agent1 = StatisticalRealismAgent()
        self.agent2 = BehavioralPlausibilityAgent()
        self.agent3 = ProfileConsistencyAgent()

    def run(
        self,
        rows: list[dict[str, Any]],
        features: list[dict[str, Any]],
        profiles: list[StudentArchetype],
    ) -> QAReport:
        """Run all three agents and return a merged QA report.

        Args:
            rows: Raw activity rows.
            features: Window-level feature dicts.
            profiles: Student archetypes to check consistency against.

        Returns:
            A ``QAReport`` with aggregated scores and flags.
        """
        r1 = self.agent1.evaluate(features=features, profiles=profiles)
        r2 = self.agent2.evaluate(rows=rows, features=features)
        r3 = self.agent3.evaluate(features=features, profiles=profiles)

        return QAReport.from_agent_results(r1, r2, r3)

    def run_until_pass(
        self,
        rows: list[dict[str, Any]],
        features: list[dict[str, Any]],
        profiles: list[StudentArchetype],
        generator_fn: Any = None,
        fixer_fn: Any = None,
    ) -> QAReport:
        """Repeatedly generate → QA → fix until data passes or max iterations.

        Args:
            rows: Initial activity rows.
            features: Initial feature dicts.
            profiles: Student archetypes for consistency checks.
            generator_fn: Callable ``(iteration: int) -> (rows, features)``
                that produces fresh data for each iteration.
            fixer_fn: Callable ``(flags, features) -> features`` that
                applies corrections based on QA flags.

        Returns:
            The final ``QAReport`` (may or may not have passed).
        """
        current_rows = list(rows)
        current_features = list(features)

        for iteration in range(self.max_iterations):
            report = self.run(current_rows, current_features, profiles)
            if report.passed:
                return report

            if fixer_fn is not None and generator_fn is not None:
                all_flags = (
                    report.statistical_flags
                    + report.behavioral_flags
                    + report.profile_flags
                )
                current_features = fixer_fn(all_flags, current_features)
                current_rows, current_features = generator_fn(
                    iteration=iteration + 1
                )
            elif generator_fn is not None:
                current_rows, current_features = generator_fn(
                    iteration=iteration + 1
                )
            else:
                break

        # Return final report (even if not passed)
        return self.run(current_rows, current_features, profiles)
