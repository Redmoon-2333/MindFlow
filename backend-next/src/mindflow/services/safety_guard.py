"""Intervention safety guardrail based on Vulnerability-Amplifying research.

Inspired by "Vulnerability-Amplifying Interaction Loops: a systematic failure
mode in AI chatbot mental-health interactions" (Weilnhammer et al., arXiv 2026).

Safety dimensions:
  1. **Forbidden content gate** — hard block on medical/crisis/self-harm language
  2. **Frequency guard** — max interventions per hour/day to prevent harassment
  3. **Progressive intensity** — intensity must not escalate beyond configurable ceiling
  4. **Context sensitivity** — suppress/intervene differently based on time-of-day and activity type
  5. **Vulnerability-aware throttling** — stronger throttling when behavioral signals
     suggest emotional vulnerability (high entertainment ratio, late-night patterns)

Design: Pure functions + stateless checks. Called by InterventionService.maybe_intervene()
before dispatching any intervention. All checks are deterministic (no LLM calls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mindflow.domain.intervention import InterventionIntensity

# ── Forbidden word categories (hard-block, never relaxed) ────────────────────────

_FORBIDDEN_MEDICAL: frozenset[str] = frozenset({
    "诊断", "治疗", "患者", "处方", "药物", "剂量",
    "复诊", "挂号", "住院", "手术", "服药", "副作用",
})

_FORBIDDEN_CRISIS: frozenset[str] = frozenset({
    "自杀", "自残", "伤害自己", "不想活", "活不下去",
    "结束生命", "一了百了", "轻生",
})

_FORBIDDEN_HARMFUL: frozenset[str] = frozenset({
    "你应该", "你必须", "你一定是", "毫无疑问",
    "绝对", "肯定是你", "就是你的问题",
})

_ALL_FORBIDDEN: frozenset[str] = _FORBIDDEN_MEDICAL | _FORBIDDEN_CRISIS | _FORBIDDEN_HARMFUL


# ── Safety check result types ────────────────────────────────────────────────────

SafetyLevel = Literal["pass", "warn", "block"]


@dataclass(frozen=True)
class SafetyCheck:
    """Result of a single safety check.

    Attributes:
        level: pass (allows), warn (allows with logging), block (hard stop).
        reason: Human-readable Chinese explanation.
        category: Which safety dimension triggered (for metrics).
    """

    level: SafetyLevel
    reason: str
    category: str = ""


@dataclass(frozen=True)
class SafetyVerdict:
    """Aggregated safety check result.

    Attributes:
        allowed: True if the intervention can be dispatched.
        checks: All executed safety checks.
        blocked_by: Which check blocked the intervention (if any).
        warnings: Non-blocking warnings for logging.
    """

    allowed: bool
    checks: tuple[SafetyCheck, ...] = ()
    blocked_by: str = ""
    warnings: tuple[str, ...] = ()


# ── Individual safety checks ─────────────────────────────────────────────────────


def check_forbidden_content(title: str, message: str) -> SafetyCheck:
    """Hard-block on medical/crisis/absolute-language content.

    Called BEFORE message rendering to catch template injection risks.
    """
    combined = title + " " + message
    for word in _FORBIDDEN_CRISIS:
        if word in combined:
            return SafetyCheck(
                level="block",
                reason=f"危机词汇「{word}」出现在干预内容中",
                category="crisis_language",
            )
    for word in _FORBIDDEN_MEDICAL:
        if word in combined:
            return SafetyCheck(
                level="block",
                reason=f"医学词汇「{word}」出现在干预内容中",
                category="medical_language",
            )
    for word in _FORBIDDEN_HARMFUL:
        if word in combined:
            return SafetyCheck(
                level="block",
                reason=f"绝对化断言「{word}」应替换为建议性表达",
                category="absolutist_language",
            )

    return SafetyCheck(level="pass", reason="", category="content")


def check_intervention_frequency(
    recent_count_1h: int,
    recent_count_24h: int,
    max_per_hour: int = 3,
    max_per_day: int = 15,
) -> SafetyCheck:
    """Frequency guard — prevent intervention flooding.

    Args:
        recent_count_1h: Interventions in the past hour.
        recent_count_24h: Interventions in the past 24h.
        max_per_hour: Hard cap per hour.
        max_per_day: Hard cap per day.

    Returns:
        SafetyCheck: block if caps exceeded, warn if approaching.
    """
    if recent_count_1h >= max_per_hour:
        return SafetyCheck(
            level="block",
            reason=f"过去1小时内已有 {recent_count_1h} 次干预（上限 {max_per_hour}）",
            category="frequency_hour",
        )
    if recent_count_24h >= max_per_day:
        return SafetyCheck(
            level="block",
            reason=f"过去24小时内已有 {recent_count_24h} 次干预（上限 {max_per_day}）",
            category="frequency_day",
        )
    if recent_count_1h >= max_per_hour - 1:
        return SafetyCheck(
            level="warn",
            reason=f"1小时内干预次数接近上限（{recent_count_1h}/{max_per_hour}）",
            category="frequency_hour_warn",
        )
    if recent_count_24h >= max_per_day - 3:
        return SafetyCheck(
            level="warn",
            reason=f"24小时内干预次数接近上限（{recent_count_24h}/{max_per_day}）",
            category="frequency_day_warn",
        )

    return SafetyCheck(level="pass", reason="", category="frequency")


def check_intensity_escalation(
    current: InterventionIntensity,
    previous: InterventionIntensity | None,
    max_intensity: InterventionIntensity = InterventionIntensity.STRICT,
) -> SafetyCheck:
    """Prevent overly aggressive intensity escalation.

    Progressive intensity: GENTLE → STANDARD → STRICT only when user has
    previously accepted STANDARD-level interventions.

    Args:
        current: The intensity about to be used.
        previous: The last used intensity (or None if first).
        max_intensity: Hard ceiling on intensity.

    Returns:
        SafetyCheck: block or warn as appropriate.
    """
    _intensity_order = {
        InterventionIntensity.GENTLE: 0,
        InterventionIntensity.STANDARD: 1,
        InterventionIntensity.STRICT: 2,
    }

    current_val = _intensity_order.get(current, 1)
    max_val = _intensity_order.get(max_intensity, 2)

    if current_val > max_val:
        return SafetyCheck(
            level="block",
            reason=f"干预强度 {current!r} 超过上限 {max_intensity!r}",
            category="intensity_ceiling",
        )

    if previous is not None:
        prev_val = _intensity_order.get(previous, 0)
        if current_val > prev_val + 1:
            return SafetyCheck(
                level="warn",
                reason=f"干预强度从 {previous!r} 跃升至 {current!r}，建议渐进增加",
                category="intensity_jump",
            )

    return SafetyCheck(level="pass", reason="", category="intensity")


def check_context_sensitivity(
    hour_of_day: int,
    entertainment_ratio: float,
    social_media_ratio: float,
) -> SafetyCheck:
    """Context-sensitive suppression — be gentler during vulnerable periods.

    Research (Weilnhammer et al., 2026): aggressive interventions during
    emotional vulnerability windows can create amplification loops.

    Vulnerability signals:
        - Late night (22:00-06:00): rest-oriented, minimal intervention
        - High entertainment ratio (>0.5): possible emotional regulation attempt
        - High social media ratio (>0.4): social support seeking

    Args:
        hour_of_day: Current hour (0-23).
        entertainment_ratio: Ratio of time spent in entertainment apps.
        social_media_ratio: Ratio of time spent in social media.

    Returns:
        SafetyCheck: block/warn for vulnerable contexts.
    """
    is_late_night = hour_of_day >= 22 or hour_of_day < 6
    is_high_entertainment = entertainment_ratio > 0.5
    is_high_social = social_media_ratio > 0.4

    if is_late_night and (is_high_entertainment or is_high_social):
        return SafetyCheck(
            level="block",
            reason="深夜+高娱乐/社交活动——疑似情绪调节期，抑制干预",
            category="vulnerability_window",
        )

    if is_late_night:
        return SafetyCheck(
            level="warn",
            reason="深夜时段，适合弱干预或推迟到次日",
            category="late_night_warn",
        )

    if is_high_entertainment and is_high_social:
        return SafetyCheck(
            level="warn",
            reason="高娱乐+高社交活动并存——用户可能处于放松/社交模式",
            category="mixed_activity_warn",
        )

    return SafetyCheck(level="pass", reason="", category="context")


# ── Aggregated safety evaluation ─────────────────────────────────────────────────


def evaluate_safety(
    title: str,
    message: str,
    intensity: InterventionIntensity,
    previous_intensity: InterventionIntensity | None = None,
    recent_count_1h: int = 0,
    recent_count_24h: int = 0,
    hour_of_day: int = 12,
    entertainment_ratio: float = 0.0,
    social_media_ratio: float = 0.0,
) -> SafetyVerdict:
    """Run all safety checks and return aggregated verdict.

    Checks execute in order of severity: content → frequency → intensity → context.
    Even if a check would block, later checks still run to surface all warnings.

    Args:
        title: Intervention title text.
        message: Intervention body text.
        intensity: Planned intervention intensity.
        previous_intensity: Last used intensity (None = first intervention).
        recent_count_1h: Interventions in past hour.
        recent_count_24h: Interventions in past 24h.
        hour_of_day: Current hour (0-23).
        entertainment_ratio: Entertainment activity ratio.
        social_media_ratio: Social media activity ratio.

    Returns:
        ``SafetyVerdict`` with pass/block decision and all check results.
    """
    checks: list[SafetyCheck] = [
        check_forbidden_content(title, message),
        check_intervention_frequency(recent_count_1h, recent_count_24h),
        check_intensity_escalation(intensity, previous_intensity),
        check_context_sensitivity(
            hour_of_day, entertainment_ratio, social_media_ratio
        ),
    ]

    blocks = [c for c in checks if c.level == "block"]
    warns = [c for c in checks if c.level == "warn"]

    return SafetyVerdict(
        allowed=len(blocks) == 0,
        checks=tuple(checks),
        blocked_by=blocks[0].category if blocks else "",
        warnings=tuple(w.reason for w in warns),
    )


# ── Vulnerability signal extraction ──────────────────────────────────────────────


def extract_vulnerability_signals(
    recent_features: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Extract vulnerability signals from recent behavioral features.

    Used to feed context_sensitivity checks with real data from the
    feature extraction pipeline.

    Args:
        recent_features: List of feature dicts from BehaviorFeatureExtractor,
            covering recent time windows (e.g., past 1-2 hours).

    Returns:
        Dict with entertainment_ratio, social_media_ratio, avg_idle_ratio,
        switch_frequency, productivity_ratio — all averaged over windows.
    """
    if not recent_features:
        return {
            "entertainment_ratio": 0.0,
            "social_media_ratio": 0.0,
            "avg_idle_ratio": 0.0,
            "switch_frequency": 0.0,
            "productivity_ratio": 0.0,
        }

    n = len(recent_features)
    return {
        "entertainment_ratio": round(
            sum(f.get("entertainment_ratio", 0.0) for f in recent_features) / n, 4
        ),
        "social_media_ratio": round(
            sum(f.get("social_ratio", 0.0) for f in recent_features) / n, 4
        ),
        "avg_idle_ratio": round(
            sum(f.get("idle_ratio", 0.0) for f in recent_features) / n, 4
        ),
        "switch_frequency": round(
            sum(f.get("switch_frequency", 0.0) for f in recent_features) / n, 4
        ),
        "productivity_ratio": round(
            sum(f.get("productivity_ratio", 0.0) for f in recent_features) / n, 4
        ),
    }
