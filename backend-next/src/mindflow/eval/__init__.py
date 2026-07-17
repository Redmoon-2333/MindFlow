"""G006 evaluation toolkit — single-expert vs expert-panel comparison.

This module provides a CLI-driven evaluation harness for comparing the
deterministic rule engine (L3) against the LLM expert panel (L1) on 30
synthetic scenarios covering all 5 procrastination types plus mixed cases.

Usage:
    python -m mindflow.eval --mode both          # mock gateway (default)
    python -m mindflow.eval --mode panel --live   # real DeepSeek gateway

Submodules:
    scenarios  — 30 frozen EvalScenario instances with gold-standard labels
    runner     — run_eval() and compare() with metric computation
    adapters   — rule_engine_analyzer() and panel_analyzer() wrappers
"""

from mindflow.eval.adapters import panel_analyzer, rule_engine_analyzer
from mindflow.eval.runner import ComparisonReport, EvalReport, compare, run_eval
from mindflow.eval.scenarios import (
    ALL_SCENARIOS,
    EvalScenario,
    get_scenario,
    validate_all_scenarios,
)

__all__ = [
    "ALL_SCENARIOS",
    "ComparisonReport",
    "EvalReport",
    "EvalScenario",
    "compare",
    "get_scenario",
    "panel_analyzer",
    "rule_engine_analyzer",
    "run_eval",
    "validate_all_scenarios",
]
