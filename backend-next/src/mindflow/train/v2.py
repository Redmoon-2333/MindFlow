"""Training utilities for privacy-preserving feature schema v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

V2_FEATURE_NAMES = [
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
]

_TASK_TYPE_CODES = {
    "": 0.0,
    "unknown": 0.0,
    "coding": 1.0 / 7.0,
    "writing": 2.0 / 7.0,
    "study": 3.0 / 7.0,
    "meeting": 4.0 / 7.0,
    "admin": 5.0 / 7.0,
    "creative": 6.0 / 7.0,
    "other": 1.0,
}


@dataclass(frozen=True)
class V2TrainingData:
    feature_names: list[str]
    features: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int32]
    sample_weights: npt.NDArray[np.float64]
    groups: npt.NDArray[np.str_]
    explicit_mask: npt.NDArray[np.bool_]
    label_sources: list[str]
    explicit_feedback_count: int
    explicit_focus_count: int
    explicit_distract_count: int
    distinct_feedback_days: int
    mixed_window_count: int


def prepare_v2_training_data(
    feature_windows: list[dict[str, Any]],
    feedback_sessions: list[dict[str, Any]],
) -> V2TrainingData:
    parsed_feedback: list[tuple[str, datetime, datetime, str, int | None, str]] = []
    for row in feedback_sessions:
        feedback = _parse_feedback(row)
        if feedback is not None:
            parsed_feedback.append(feedback)
    explicit_sessions = [row for row in parsed_feedback if row[4] is not None]
    explicit_session_ids = {row[0] for row in explicit_sessions}
    focus_sessions = {row[0] for row in explicit_sessions if row[4] == 1}
    distract_sessions = {row[0] for row in explicit_sessions if row[4] == 0}
    feedback_days = {row[1].date().isoformat() for row in explicit_sessions}

    matrix_rows: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    groups: list[str] = []
    explicit_mask: list[bool] = []
    sources: list[str] = []
    mixed_window_count = 0

    for window in sorted(feature_windows, key=lambda row: str(row.get("window_start_utc", ""))):
        parsed_window = _parse_window(window)
        if parsed_window is None:
            continue
        window_start, window_end, features = parsed_window
        overlapping = [
            feedback
            for feedback in parsed_feedback
            if feedback[1] < window_end and feedback[2] > window_start
        ]
        selected = max(
            overlapping,
            key=lambda feedback: _overlap_seconds(
                window_start,
                window_end,
                feedback[1],
                feedback[2],
            ),
            default=None,
        )
        row_features = dict(features)
        if selected is not None:
            selected_label = selected[4]
            if selected_label is None:
                mixed_window_count += 1
                continue
            label = selected_label
            weight = 1.0
            source = "explicit"
            is_explicit = True
            row_features["task_type_code"] = _task_type_code(selected[5])
        else:
            label = _weak_label(row_features)
            weight = 0.25
            source = "weak"
            is_explicit = False
            row_features.setdefault("task_type_code", 0.0)

        vector = [_finite_float(row_features.get(name, 0.0)) for name in V2_FEATURE_NAMES]
        matrix_rows.append(vector)
        labels.append(label)
        weights.append(weight)
        groups.append(window_start.date().isoformat())
        explicit_mask.append(is_explicit)
        sources.append(source)

    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if matrix.size == 0:
        matrix = np.empty((0, len(V2_FEATURE_NAMES)), dtype=np.float64)

    return V2TrainingData(
        feature_names=list(V2_FEATURE_NAMES),
        features=matrix,
        labels=np.asarray(labels, dtype=np.int32),
        sample_weights=np.asarray(weights, dtype=np.float64),
        groups=np.asarray(groups, dtype=np.str_),
        explicit_mask=np.asarray(explicit_mask, dtype=np.bool_),
        label_sources=sources,
        explicit_feedback_count=len(explicit_session_ids),
        explicit_focus_count=len(focus_sessions),
        explicit_distract_count=len(distract_sessions),
        distinct_feedback_days=len(feedback_days),
        mixed_window_count=mixed_window_count,
    )


def evaluate_v2_candidates(
    data: V2TrainingData,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    mask = data.explicit_mask
    features = data.features[mask]
    labels = data.labels[mask]
    groups = data.groups[mask]
    unique_groups = np.unique(groups)
    if len(features) < 10 or len(np.unique(labels)) < 2 or len(unique_groups) < 2:
        return {
            "status": "insufficient_explicit_data",
            "candidate": {},
            "logistic_baseline": {},
            "rule_baseline": {},
            "folds": [],
        }

    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    candidate_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    logistic_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    rule_probabilities = _rule_probabilities(features, data.feature_names)
    folds: list[dict[str, Any]] = []

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(features, labels, groups),
        start=1,
    ):
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        candidate = RandomForestClassifier(
            n_estimators=100,  # Must match FocusClassifier (classifier.py:25)
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state + fold_index,
            n_jobs=1,
        )
        logistic = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state + fold_index,
            ),
        )
        candidate.fit(features[train_indices], y_train)
        logistic.fit(features[train_indices], y_train)
        candidate_fold_probabilities = candidate.predict_proba(features[test_indices])[:, 1]
        logistic_fold_probabilities = logistic.predict_proba(features[test_indices])[:, 1]
        candidate_probabilities[test_indices] = candidate_fold_probabilities
        logistic_probabilities[test_indices] = logistic_fold_probabilities
        candidate_predictions = (candidate_fold_probabilities >= 0.5).astype(np.int32)
        folds.append({
            "fold": fold_index,
            "train_dates": sorted(set(groups[train_indices].tolist())),
            "test_dates": sorted(set(groups[test_indices].tolist())),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(y_test, candidate_predictions)),
                6,
            ),
        })

    evaluated_mask = np.isfinite(candidate_probabilities) & np.isfinite(logistic_probabilities)
    if not evaluated_mask.any() or not folds:
        return {
            "status": "insufficient_grouped_folds",
            "candidate": {},
            "logistic_baseline": {},
            "rule_baseline": {},
            "folds": folds,
        }

    evaluated_labels = labels[evaluated_mask]
    candidate_metrics = _classification_metrics(
        evaluated_labels,
        candidate_probabilities[evaluated_mask],
    )
    logistic_metrics = _classification_metrics(
        evaluated_labels,
        logistic_probabilities[evaluated_mask],
    )
    rule_metrics = _classification_metrics(
        evaluated_labels,
        rule_probabilities[evaluated_mask],
    )
    fold_scores = [float(fold["balanced_accuracy"]) for fold in folds]
    candidate_metrics["fold_balanced_accuracy_range"] = round(
        max(fold_scores) - min(fold_scores),
        6,
    )
    return {
        "status": "evaluated",
        "candidate": candidate_metrics,
        "logistic_baseline": logistic_metrics,
        "rule_baseline": rule_metrics,
        "folds": folds,
    }


def evaluate_v2_quality_gate(
    evaluation: dict[str, Any],
    *,
    explicit_feedback_count: int,
    explicit_focus_count: int,
    explicit_distract_count: int,
    distinct_feedback_days: int,
) -> dict[str, Any]:
    candidate = evaluation.get("candidate", {})
    rule_baseline = evaluation.get("rule_baseline", {})
    checks = {
        "minimum_days": distinct_feedback_days >= 7,
        "minimum_explicit_feedback": explicit_feedback_count >= 30,
        "minimum_class_feedback": explicit_focus_count >= 10 and explicit_distract_count >= 10,
        "balanced_accuracy": float(candidate.get("balanced_accuracy", 0.0)) >= 0.65,
        "minority_f1": float(candidate.get("minority_f1", 0.0)) >= 0.55,
        "calibration_better_than_rule": (
            "brier_score" in candidate
            and "brier_score" in rule_baseline
            and float(candidate["brier_score"]) < float(rule_baseline["brier_score"])
        ),
        "stable_date_folds": (
            len(evaluation.get("folds", [])) >= 3
            and float(candidate.get("fold_balanced_accuracy_range", 1.0)) <= 0.10
        ),
    }
    is_passed = evaluation.get("status") == "evaluated" and all(checks.values())
    return {
        "passed": is_passed,
        "mode": "ready" if is_passed else "shadow",
        "checks": checks,
        "explicit_feedback_count": explicit_feedback_count,
        "explicit_focus_count": explicit_focus_count,
        "explicit_distract_count": explicit_distract_count,
        "distinct_feedback_days": distinct_feedback_days,
    }


def _parse_feedback(
    row: dict[str, Any],
) -> tuple[str, datetime, datetime, str, int | None, str] | None:
    try:
        session_id = str(row["session_id"])
        start = _parse_datetime(row["start_time"])
        end = _parse_datetime(row["end_time"])
        label_name = str(row.get("label", "")).lower()
        score = int(row.get("score", 3))
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    label: int | None
    if label_name == "mixed" or score == 3:
        label = None
    elif score >= 4:
        label = 1
    elif score <= 2:
        label = 0
    else:
        label = None
    return session_id, start, end, label_name, label, str(row.get("task_type") or "")


def _parse_window(
    row: dict[str, Any],
) -> tuple[datetime, datetime, dict[str, Any]] | None:
    try:
        if int(row.get("feature_schema_version", 0)) != 2:
            return None
        start = _parse_datetime(row["window_start_utc"])
        end = _parse_datetime(row["window_end_utc"])
        features = dict(row.get("features") or row.get("features_json") or {})
    except (KeyError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end, features


def _parse_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _overlap_seconds(
    first_start: datetime,
    first_end: datetime,
    second_start: datetime,
    second_end: datetime,
) -> float:
    return max(0.0, (min(first_end, second_end) - max(first_start, second_start)).total_seconds())


def _task_type_code(task_type: str) -> float:
    return _TASK_TYPE_CODES.get(task_type.strip().lower(), _TASK_TYPE_CODES["other"])


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _weak_label(features: dict[str, Any]) -> int:
    idle_ratio = min(max(_finite_float(features.get("idle_ratio", 0.0)), 0.0), 1.0)
    longest_ratio = min(
        max(_finite_float(features.get("longest_segment_ratio", 0.0)), 0.0),
        1.0,
    )
    top_app_ratio = min(max(_finite_float(features.get("top_app_ratio", 0.0)), 0.0), 1.0)
    switch_penalty = min(
        (_finite_float(features.get("app_switch_count", 0.0))
         + _finite_float(features.get("domain_switch_count", 0.0)))
        / 12.0,
        1.0,
    )
    score = 0.45 + 0.25 * longest_ratio + 0.2 * top_app_ratio - 0.55 * idle_ratio
    score -= 0.2 * switch_penalty
    return int(score >= 0.5)


def _rule_probabilities(
    features: npt.NDArray[np.float64],
    feature_names: list[str],
) -> npt.NDArray[np.float64]:
    positions = {name: index for index, name in enumerate(feature_names)}

    def column(name: str) -> npt.NDArray[np.float64]:
        index = positions.get(name)
        if index is None:
            return np.zeros(len(features), dtype=np.float64)
        return features[:, index]

    idle = np.clip(column("idle_ratio"), 0.0, 1.0)
    longest = np.clip(column("longest_segment_ratio"), 0.0, 1.0)
    top_app = np.clip(column("top_app_ratio"), 0.0, 1.0)
    active = np.clip(column("input_active_ratio"), 0.0, 1.0)
    switches = np.clip(
        (column("app_switch_count") + column("domain_switch_count")) / 12.0,
        0.0,
        1.0,
    )
    audible = np.clip(column("audible_browser_ratio"), 0.0, 1.0)
    probabilities = 0.45 + 0.2 * longest + 0.15 * top_app + 0.1 * active
    probabilities -= 0.4 * idle + 0.15 * switches + 0.05 * audible
    return np.clip(probabilities, 0.01, 0.99)


def _classification_metrics(
    labels: npt.NDArray[np.int32],
    probabilities: npt.NDArray[np.float64],
) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int32)
    class_f1 = [
        f1_score(labels, predictions, pos_label=class_label, zero_division=0)
        for class_label in (0, 1)
    ]
    return {
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 6),
        "minority_f1": round(float(min(class_f1)), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
    }
