"""FocusPrediction — unified ML prediction contract for online inference.

This is the type that flows from the ML sensing layer to all consumers
(Panel EvidenceBundle, Telemetry API, Chat evidence tools), replacing
the previous ad-hoc dict responses that differed between each caller.

Key contract:
  - ``status`` tells consumers what to expect — never assume ``ready``.
  - ML evidence is always statistical, never causal or diagnostic.
  - ``top_factors`` are heuristic explanations (importance × observation),
    not causal attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION

FocusPredictionStatus = Literal[
    "ready",
    "no_model",
    "no_data",
    "stale",
    "schema_mismatch",
    "inference_error",
]


@dataclass(frozen=True)
class FocusPrediction:
    """Unified ML prediction for a user's focus state.

    Attributes:
        status: Operational status of the prediction.
            - ``ready``: Prediction available and current.
            - ``no_model``: No active model loaded.
            - ``no_data``: Model exists but no feature windows found.
            - ``stale``: Newest feature window is older than the threshold.
            - ``schema_mismatch``: Feature schema version incompatible with model.
            - ``inference_error``: Model inference raised an exception.
        focus_probability: Probability of focus in [0, 1], or None.
        uncertainty: Prediction uncertainty in [0, 1], or None.
        window_count: Number of feature windows used for this prediction.
        coverage_ratio: Fraction of expected windows that were present, in [0, 1].
        data_age_s: Seconds since the newest feature window ended, or None.
        model_version: Active model version tag, or None.
        feature_schema_version: Schema version of the features used.
        top_factors: Top 3 most influential features, each with name, value, importance.
        explanation_method: How ``top_factors`` was computed
            (e.g. ``"global_importance_times_observation"``).
        reason: Human-readable Chinese explanation of the status.
    """

    status: FocusPredictionStatus = "no_model"
    focus_probability: float | None = None
    uncertainty: float | None = None
    distracted_window_ratio: float | None = None
    window_count: int = 0
    newest_window_start_utc: str | None = None
    coverage_ratio: float = 0.0
    data_age_s: float | None = None
    model_version: str | None = None
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    top_factors: list[dict[str, float | str]] = ()
    explanation_method: str = ""
    reason: str = ""


# Data staleness threshold: feature windows older than 15 minutes are stale.
STALE_THRESHOLD_S: float = 900.0

# Minimum coverage ratio below which ML evidence is considered insufficient.
MIN_COVERAGE_RATIO: float = 0.3
