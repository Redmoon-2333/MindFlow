"""Single authoritative feature-schema constants.

Owns the feature vocabulary so ``domain`` and ``train`` never drift:
``BaselineModel`` persists Welford stats keyed by these names, telemetry
produces windows carrying exactly this vocabulary, and model training reads
columns in this exact order. Pure stdlib — no numpy/sklearn so the domain
layer stays lightweight and import-free of the ML stack.

Version history (a bump is a deliberate, breaking change — never a silent one):

* v2 — 24 columns, ``task_type_code`` present but hard-coded ``0.0``.
* v3 — same 24 columns; switch counting hardened (``count_confirmed_switches``).
* v4 — 28 columns: ``task_type_code`` now carries the observed dominant task
  context, and four task-context columns are appended
  (``task_type_entropy``, ``task_type_dominant_ratio``,
  ``task_context_transition``, ``task_unknown_ratio``).  The first 24 columns
  keep their semantics and order, so downstream readers that address columns
  by name are unaffected; artifacts trained on v3 (or older) must not be
  loaded against v4 windows — see ``train/v2.py::_parse_window`` (which drops
  non-current windows) and ``services/prediction_service.py`` (which rejects a
  model whose stored feature names do not match this vocabulary).
"""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = 4

V2_FEATURE_NAMES: tuple[str, ...] = (
    # ── v2/v3 columns — semantics and order unchanged ──────────────────
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
    # ── v4 columns — task context (see domain/task_context.py) ─────────
    "task_type_entropy",
    "task_type_dominant_ratio",
    "task_context_transition",
    "task_unknown_ratio",
)
