"""Single shared Chinese label map for the V2/V4 feature vocabulary.

Every user-facing surface that names a feature — the personal-deviation
summary, the focus-prediction explanation, and the attribution/panel payloads
built by the API layer — reads its Chinese label from here, so one feature
cannot be called two different things in two places.

Pure stdlib: a plain dict plus a total-coverage invariant checked at import
time. Adding a feature to ``V2_FEATURE_NAMES`` without a label fails loudly
instead of silently rendering the raw column name.
"""

from __future__ import annotations

from mindflow.domain.feature_schema import V2_FEATURE_NAMES

#: V2/V4 feature name -> short Chinese label (≤ 8 characters where possible,
#: so it fits a compact chart legend / tooltip).
FEATURE_LABELS_ZH: dict[str, str] = {
    # ── switches / segmentation ──────────────────────────────────────
    "app_switch_count": "应用切换次数",
    "domain_switch_count": "域名切换次数",
    "longest_segment_ratio": "最长专注段占比",
    "idle_ratio": "空闲占比",
    # ── input intensity ─────────────────────────────────────────────
    "keypress_rate_per_min": "键盘输入频率",
    "mouse_click_rate_per_min": "鼠标点击频率",
    "scroll_rate_per_min": "滚动频率",
    "mouse_distance_per_min": "鼠标移动距离",
    "input_active_ratio": "输入活跃占比",
    "interaction_bursts_per_min": "交互爆发频率",
    "click_key_ratio": "点击键盘比",
    # ── browser / app concentration ─────────────────────────────────
    "browser_ratio": "浏览器占比",
    "audible_browser_ratio": "有声浏览占比",
    "active_seconds_ratio": "活跃时长占比",
    "top_app_ratio": "主导应用占比",
    "top_domain_ratio": "主导域名占比",
    # ── interaction rhythm ──────────────────────────────────────────
    "interaction_interval_mean_s": "交互间隔均值",
    "interaction_interval_std_s": "交互间隔波动",
    "interaction_interval_cv": "交互间隔变异系数",
    # ── cyclic time encodings (context, not behaviour) ──────────────
    "hour_sin": "小时(正弦)",
    "hour_cos": "小时(余弦)",
    "weekday_sin": "星期(正弦)",
    "weekday_cos": "星期(余弦)",
    # ── task context (v4) ───────────────────────────────────────────
    "task_type_code": "任务类型编码",
    "task_type_entropy": "任务类型熵",
    "task_type_dominant_ratio": "主导任务占比",
    "task_context_transition": "任务上下文切换",
    "task_unknown_ratio": "未知任务占比",
}

_MISSING = [name for name in V2_FEATURE_NAMES if name not in FEATURE_LABELS_ZH]
if _MISSING:  # pragma: no cover - import-time invariant, cannot be reached once green
    msg = f"FEATURE_LABELS_ZH is missing labels for: {', '.join(_MISSING)}"
    raise AssertionError(msg)


def feature_label(name: str) -> str:
    """Chinese label for *name*, falling back to the raw name.

    The fallback keeps an unknown column visible instead of inventing a
    translation; the import-time check above already guarantees every current
    V2 feature has a real label.
    """
    return FEATURE_LABELS_ZH.get(name, name)


__all__ = ["FEATURE_LABELS_ZH", "feature_label"]
