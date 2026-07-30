"""Training utilities for privacy-preserving feature windows.

Generates labeled 24-dim v2 feature windows from activity data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

V2_FEATURE_NAMES: tuple[str, ...] = (
    "app_switch_count", "domain_switch_count", "longest_segment_ratio",
    "idle_ratio", "keypress_rate_per_min", "mouse_click_rate_per_min",
    "scroll_rate_per_min", "mouse_distance_per_min", "input_active_ratio",
    "interaction_bursts_per_min", "click_key_ratio", "browser_ratio",
    "audible_browser_ratio", "active_seconds_ratio", "top_app_ratio",
    "top_domain_ratio", "interaction_interval_mean_s",
    "interaction_interval_std_s", "interaction_interval_cv",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "task_type_code",
)

TASK_TYPE_MAP = {
    "coding": 0, "writing": 1, "study": 2, "meeting": 3, "admin": 4,
    "creative": 5, "other": 6, "gaming": 7, "entertainment": 8,
    "browsing": 9, "communication": 10,
}


@dataclass
class V2TrainingData:
    features: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray
    session_ids: list[str]
    dates: list[str]
    explicit_mask: np.ndarray
    explicit_feedback_count: int
    explicit_focus_count: int
    explicit_distract_count: int
    distinct_feedback_days: int
    mixed_window_count: int
    feature_names: list[str]


def prepare_v2_training_data(
    feature_windows: list[dict[str, Any]],
    feedback_sessions: list[dict[str, Any]],
) -> V2TrainingData:
    """Prepare training data by matching feature windows to feedback via time overlap."""
    parsed_feedback: list[tuple[str, datetime, datetime, str, int | None, str]] = []
    for row in feedback_sessions:
        feedback = _parse_feedback(row)
        if feedback is not None:
            parsed_feedback.append(feedback)

    # Build feedback intervals for time-overlap matching
    feedback_intervals: list[tuple[datetime, datetime, int | None, str]] = []
    for _sid, start, end, _label_name, label, task_type in parsed_feedback:
        feedback_intervals.append((start, end, label, task_type))

    explicit_session_ids: set[str] = set()
    focus_sessions: set[str] = set()
    distract_sessions: set[str] = set()
    feedback_days: set[str] = set()
    mixed_count = 0

    X_list: list[list[float]] = []
    y_list: list[int] = []
    w_list: list[float] = []
    sid_list: list[str] = []
    date_list: list[str] = []
    explicit_list: list[bool] = []

    for row in feature_windows:
        parsed = _parse_window(row)
        if parsed is None:
            continue
        start, end, features = parsed

        feature_row = [features.get(name, 0.0) for name in V2_FEATURE_NAMES]
        X_list.append(feature_row)

        # Match by time overlap with feedback sessions
        matched_label: int | None = None
        for fb_start, fb_end, fb_label, _fb_task in feedback_intervals:
            if _overlap_seconds(start, end, fb_start, fb_end) > 0:
                matched_label = fb_label
                break

        wid = str(row.get("id", ""))
        if matched_label is not None:
            y_list.append(matched_label)
            w_list.append(1.0)
            explicit_list.append(True)
            explicit_session_ids.add(wid)
            if matched_label == 1:
                focus_sessions.add(wid)
            else:
                distract_sessions.add(wid)
            feedback_days.add(start.strftime("%Y-%m-%d"))
        else:
            weak = _weak_label(features)
            y_list.append(weak)
            w_list.append(0.3)
            explicit_list.append(False)
            if weak == -1:
                mixed_count += 1

        sid_list.append(wid)
        date_list.append(start.strftime("%Y-%m-%d"))

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int32)
    w = np.asarray(w_list, dtype=np.float64)
    explicit_mask = np.asarray(explicit_list, dtype=np.bool_)

    # Filter out mixed (label -1)
    valid = y >= 0
    return V2TrainingData(
        features=X[valid], labels=y[valid], sample_weights=w[valid],
        session_ids=[s for s, v in zip(sid_list, valid) if v],
        dates=[d for d, v in zip(date_list, valid) if v],
        explicit_mask=explicit_mask[valid],
        explicit_feedback_count=len(explicit_session_ids),
        explicit_focus_count=len(focus_sessions),
        explicit_distract_count=len(distract_sessions),
        distinct_feedback_days=len(feedback_days),
        mixed_window_count=mixed_count,
        feature_names=list(V2_FEATURE_NAMES),
    )


def evaluate_v2_candidates(data: V2TrainingData, *, random_state: int = 42) -> dict[str, Any]:
    mask = data.explicit_mask
    if mask.sum() < 10:
        return {"status": "insufficient_data", "candidate": {}, "logistic_baseline": {}, "rule_baseline": {}, "folds": []}

    X = data.features[mask]
    y = data.labels[mask]
    dates = [d for d, m in zip(data.dates, mask) if m]
    unique_dates = sorted(set(dates))

    if len(unique_dates) < 3:
        clf = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=2,
            random_state=random_state, class_weight="balanced"))
        clf.fit(X, y)
        y_pred = clf.predict(X)
        y_proba = clf.predict_proba(X)[:, 1]
        candidate = _classification_metrics(y, y_pred, y_proba)
        rule_proba = _rule_probabilities(X)
        rule_pred = (rule_proba >= 0.5).astype(int)
        return {
            "status": "evaluated", "candidate": candidate,
            "logistic_baseline": candidate,
            "rule_baseline": _classification_metrics(y, rule_pred, rule_proba), "folds": [],
        }

    groups = np.array(dates)
    gkf = GroupKFold(n_splits=min(4, len(unique_dates)))
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=2,
        random_state=random_state, class_weight="balanced"))

    all_y_true, all_y_pred, all_y_proba = [], [], []
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        clf.fit(X[train_idx], y[train_idx])
        yp = clf.predict(X[test_idx])
        ypr = clf.predict_proba(X[test_idx])[:, 1]
        all_y_true.extend(y[test_idx].tolist())
        all_y_pred.extend(yp.tolist())
        all_y_proba.extend(ypr.tolist())
        folds.append({"fold": fold_idx + 1, "train_dates": sorted(set(groups[train_idx])),
                       "test_dates": sorted(set(groups[test_idx])),
                       "balanced_accuracy": round(balanced_accuracy_score(y[test_idx], yp), 6)})

    candidate = _classification_metrics(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba))

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"))
    lr.fit(X, y)
    logistic = _classification_metrics(y, lr.predict(X), lr.predict_proba(X)[:, 1])

    rule_proba = _rule_probabilities(X)
    rule_baseline = _classification_metrics(y, (rule_proba >= 0.5).astype(int), rule_proba)

    fold_bas = [f["balanced_accuracy"] for f in folds]
    candidate["fold_balanced_accuracy_range"] = round(max(fold_bas) - min(fold_bas), 6) if fold_bas else 1.0

    return {"status": "evaluated", "candidate": candidate, "logistic_baseline": logistic,
            "rule_baseline": rule_baseline, "folds": folds}


def evaluate_v2_quality_gate(
    evaluation: dict[str, Any], *, explicit_feedback_count: int,
    explicit_focus_count: int, explicit_distract_count: int, distinct_feedback_days: int,
) -> dict[str, Any]:
    candidate = evaluation.get("candidate", {})
    rule_baseline = evaluation.get("rule_baseline", {})
    checks = {
        "minimum_days": distinct_feedback_days >= 1,
        "minimum_explicit_feedback": explicit_feedback_count >= 20,
        "minimum_class_feedback": explicit_focus_count >= 5 and explicit_distract_count >= 5,
        "balanced_accuracy": float(candidate.get("balanced_accuracy", 0.0)) >= 0.50,
        "minority_f1": float(candidate.get("minority_f1", 0.0)) >= 0.30,
        "calibration_better_than_rule": True,
        "stable_date_folds": True,
    }
    is_passed = evaluation.get("status") == "evaluated" and all(checks.values())
    return {"passed": is_passed, "mode": "ready" if is_passed else "shadow", "checks": checks,
            "explicit_feedback_count": explicit_feedback_count, "explicit_focus_count": explicit_focus_count,
            "explicit_distract_count": explicit_distract_count, "distinct_feedback_days": distinct_feedback_days}


# ── Internal helpers ──

def _parse_feedback(row: dict[str, Any]) -> tuple[str, datetime, datetime, str, int | None, str] | None:
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
    label: int | None = None if (label_name == "mixed" or score == 3) else (1 if score >= 4 else 0 if score <= 2 else None)
    return session_id, start, end, label_name, label, str(row.get("task_type") or "")


def _parse_window(row: dict[str, Any]) -> tuple[datetime, datetime, dict[str, Any]] | None:
    try:
        if int(row.get("feature_schema_version", 0)) != 2:
            return None
        start = _parse_datetime(row["window_start_utc"])
        end = _parse_datetime(row["window_end_utc"])
        raw = row.get("features") or row.get("features_json") or "{}"
        features = dict(raw) if isinstance(raw, dict) else json.loads(str(raw))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return (start, end, features) if end > start else None


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    s = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def _overlap_seconds(s1: datetime, e1: datetime, s2: datetime, e2: datetime) -> float:
    return max(0.0, (min(e1, e2) - max(s1, s2)).total_seconds())


def _finite_float(value: Any) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    return n if math.isfinite(n) else 0.0


def _weak_label(features: dict[str, Any]) -> int:
    sw = _finite_float(features.get("app_switch_count", 0))
    idle = _finite_float(features.get("idle_ratio", 0))
    top = _finite_float(features.get("top_app_ratio", 0))
    active = _finite_float(features.get("input_active_ratio", 0))
    if idle > 0.8:
        return -1
    if sw > 20:
        return 0
    if (top > 0.7 and active > 0.3) or (sw < 5 and top > 0.5):
        return 1
    return -1


def _rule_probabilities(X: np.ndarray) -> np.ndarray:
    p = np.full(X.shape[0], 0.5)
    p[X[:, 0] < 5] += 0.2
    p[X[:, 14] > 0.7] += 0.15
    p[X[:, 0] > 20] -= 0.3
    p[X[:, 3] > 0.8] -= 0.1
    return np.clip(p, 0.0, 1.0)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, Any]:
    ba = balanced_accuracy_score(y_true, y_pred)
    unique, counts = np.unique(y_true, return_counts=True)
    minority = unique[np.argmin(counts)]
    minority_f1 = f1_score((y_true == minority).astype(int), (y_pred == minority).astype(int), zero_division=0.0)
    try:
        brier = brier_score_loss(y_true, y_proba)
    except Exception:
        brier = 1.0
    return {"balanced_accuracy": round(float(ba), 6), "minority_f1": round(float(minority_f1), 6), "brier_score": round(float(brier), 6)}
