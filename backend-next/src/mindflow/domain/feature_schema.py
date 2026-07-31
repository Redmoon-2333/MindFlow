"""Single authoritative feature-schema constants.

Owns the v2 feature vocabulary so ``domain`` and ``train`` never drift:
``BaselineModel`` persists Welford stats keyed by these names, telemetry
produces windows carrying exactly this vocabulary, and model training reads
columns in this exact order. Pure stdlib — no numpy/sklearn so the domain
layer stays lightweight and import-free of the ML stack.
"""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = 2
FEATURE_SCHEMA_VERSION = 3

V2_FEATURE_NAMES: tuple[str, ...] = (
    "app_switch_count",
    "domain_switch_count",
    "longest_segment_ratio",
    "idle_ratio",
    "keypress_rate_per_min",
    "mouse_click_rate_per_min",
    "scroll_rate_per_min",
    "mouse_distance_per_min",
    "input_active_ratio",
    "interaction_bursts_per_min",
    "click_key_ratio",
    "browser_ratio",
    "audible_browser_ratio",
    "active_seconds_ratio",
    "top_app_ratio",
    "top_domain_ratio",
    "interaction_interval_mean_s",
    "interaction_interval_std_s",
    "interaction_interval_cv",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "task_type_code",
)
