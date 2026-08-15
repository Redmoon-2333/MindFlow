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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Single authoritative vocabulary lives in the domain layer so BaselineModel
# and training can never drift. Re-export keeps ``mindflow.train.v2`` importers
# (telemetry_service, prediction_service) working unchanged.
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES  # noqa: F401
from mindflow.train.config import TRAIN_CONFIG

TASK_TYPE_MAP = {
    "coding": 0, "writing": 1, "study": 2, "meeting": 3, "admin": 4,
    "creative": 5, "other": 6, "gaming": 7, "entertainment": 8,
    "browsing": 9, "communication": 10,
}

# Same class used by ModelManager in production so evaluation and deployment
# share hyperparameters, scaling, and soft-voting behaviour.
def make_v2_classifier() -> Any:
    from mindflow.train.models.ensemble import EnsembleClassifier

    return EnsembleClassifier()


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
    matched_window_count: int
    label_sources: list[str]
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

    # Keep the feedback session id so quality counts are unique sessions,
    # not the number of overlapping feature windows.
    feedback_intervals: list[tuple[str, datetime, datetime, int | None, str]] = []
    for sid, start, end, _label_name, label, task_type in parsed_feedback:
        feedback_intervals.append((sid, start, end, label, task_type))

    explicit_session_ids: set[str] = set()
    focus_sessions: set[str] = set()
    distract_sessions: set[str] = set()
    feedback_days: set[str] = set()
    mixed_count = 0
    matched_window_count = 0

    X_list: list[list[float]] = []
    y_list: list[int] = []
    w_list: list[float] = []
    sid_list: list[str] = []
    date_list: list[str] = []
    explicit_list: list[bool] = []
    source_list: list[str] = []

    for row in feature_windows:
        parsed = _parse_window(row)
        if parsed is None:
            continue
        start, end, features = parsed

        feature_row = [features.get(name, 0.0) for name in V2_FEATURE_NAMES]
        X_list.append(feature_row)

        # Match by time overlap with feedback sessions
        matched_label: int | None = None
        matched_sid: str | None = None
        matched_start: datetime | None = None
        for sid, fb_start, fb_end, fb_label, _fb_task in feedback_intervals:
            if _overlap_seconds(start, end, fb_start, fb_end) > 0:
                matched_label = fb_label
                matched_sid = sid
                matched_start = fb_start
                break

        wid = str(row.get("id", ""))
        if matched_label is not None:
            y_list.append(matched_label)
            w_list.append(1.0)
            explicit_list.append(True)
            source_list.append("explicit")
            matched_window_count += 1
            if matched_sid is not None:
                explicit_session_ids.add(matched_sid)
                if matched_label == 1:
                    focus_sessions.add(matched_sid)
                else:
                    distract_sessions.add(matched_sid)
                if matched_start is not None:
                    feedback_days.add(matched_start.strftime("%Y-%m-%d"))
        else:
            weak = _weak_label(features)
            y_list.append(weak)
            w_list.append(0.3)
            explicit_list.append(False)
            source_list.append("weak")
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
        session_ids=[s for s, v in zip(sid_list, valid, strict=True) if v],
        dates=[d for d, v in zip(date_list, valid, strict=True) if v],
        explicit_mask=explicit_mask[valid],
        explicit_feedback_count=len(explicit_session_ids),
        explicit_focus_count=len(focus_sessions),
        explicit_distract_count=len(distract_sessions),
        distinct_feedback_days=len(feedback_days),
        mixed_window_count=mixed_count,
        matched_window_count=matched_window_count,
        label_sources=[s for s, v in zip(source_list, valid, strict=True) if v],
        feature_names=list(V2_FEATURE_NAMES),
    )


def evaluate_v2_candidates(data: V2TrainingData, *, random_state: int = 42) -> dict[str, Any]:
    """Date-grouped cross-validation with the same classifier used in production.

    Rule and logistic baselines are computed inside the same held-out folds;
    in-sample comparisons are intentionally not reported as evidence.
    """
    mask = data.explicit_mask
    if mask.sum() < 10:
        return {
            "status": "insufficient_data",
            "candidate": {},
            "logistic_baseline": {},
            "rule_baseline": {},
            "folds": [],
            "fold_stability": {},
        }

    X = data.features[mask]
    y = data.labels[mask]
    dates = [d for d, m in zip(data.dates, mask, strict=True) if m]
    unique_dates = sorted(set(dates))

    if len(unique_dates) < 3:
        return {
            "status": "insufficient_data",
            "candidate": {},
            "logistic_baseline": {},
            "rule_baseline": {},
            "folds": [],
            "fold_stability": {"reason": "need >=3 feedback dates"},
        }

    groups = np.array(dates)
    gkf = GroupKFold(n_splits=min(4, len(unique_dates)))
    weights = data.sample_weights[mask]
    gkf = GroupKFold(n_splits=min(TRAIN_CONFIG.group_folds, len(unique_dates)))

    all_y_true: list[int] = []
    all_candidate_pred: list[int] = []
    all_candidate_proba: list[float] = []
    all_logistic_pred: list[int] = []
    all_logistic_proba: list[float] = []
    all_rule_pred: list[int] = []
    all_rule_proba: list[float] = []
    folds: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        y_true = y[test_idx]
        if len(np.unique(y_true)) < 2 or len(y_true) < 5:
            folds.append({
                "fold": fold_idx + 1,
                "train_dates": sorted(set(groups[train_idx])),
                "test_dates": sorted(set(groups[test_idx])),
                "balanced_accuracy": 0.0,
                "reason": "test fold too small for stable metrics",
            })
            continue

        clf = make_v2_classifier()
        clf.fit(
            X[train_idx],
            y[train_idx],
            list(V2_FEATURE_NAMES),
            sample_weight=weights[train_idx],
        )
        yp = clf.predict(X[test_idx])
        ypr = clf.predict_proba(X[test_idx])[:, 1]

        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, random_state=random_state, class_weight="balanced"
            ),
        )
        lr.fit(
            X[train_idx],
            y[train_idx],
            logisticregression__sample_weight=weights[train_idx],
        )
        lp = lr.predict(X[test_idx])
        lpr = lr.predict_proba(X[test_idx])[:, 1]

        rule_proba = _rule_probabilities(X[test_idx])
        rp = (rule_proba >= 0.5).astype(int)

        all_y_true.extend(y_true.tolist())
        all_candidate_pred.extend(yp.tolist())
        all_candidate_proba.extend(ypr.tolist())
        all_logistic_pred.extend(lp.tolist())
        all_logistic_proba.extend(lpr.tolist())
        all_rule_pred.extend(rp.tolist())
        all_rule_proba.extend(rule_proba.tolist())

        folds.append({
            "fold": fold_idx + 1,
            "train_dates": sorted(set(groups[train_idx])),
            "test_dates": sorted(set(groups[test_idx])),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, yp), 6),
            "test_size": int(len(y_true)),
        })

    if not all_y_true:
        return {
            "status": "insufficient_data",
            "candidate": {},
            "logistic_baseline": {},
            "rule_baseline": {},
            "folds": folds,
            "fold_stability": {"reason": "no valid held-out folds"},
        }

    y_true_arr = np.array(all_y_true)
    candidate = _classification_metrics(
        y_true_arr, np.array(all_candidate_pred), np.array(all_candidate_proba)
    )
    logistic = _classification_metrics(
        y_true_arr, np.array(all_logistic_pred), np.array(all_logistic_proba)
    )
    rule_baseline = _classification_metrics(
        y_true_arr, np.array(all_rule_pred), np.array(all_rule_proba)
    )

    fold_bas = [f["balanced_accuracy"] for f in folds if "balanced_accuracy" in f]
    fold_bas = [float(v) for v in fold_bas]
    min_test_size = min((int(f.get("test_size", 0)) for f in folds), default=0)
    if fold_bas:
        candidate["fold_balanced_accuracy_range"] = round(max(fold_bas) - min(fold_bas), 6)
        candidate["fold_min_balanced_accuracy"] = round(min(fold_bas), 6)
        fold_stability = {
            "passed": bool(
                min(fold_bas) >= 0.50
                and (max(fold_bas) - min(fold_bas)) <= 0.35
                and min_test_size >= 5
            ),
            "min_balanced_accuracy": round(min(fold_bas), 6),
            "range": round(max(fold_bas) - min(fold_bas), 6),
            "min_test_size": min_test_size,
        }
    else:
        fold_stability = {"passed": False, "reason": "no stable folds"}

    return {
        "status": "evaluated",
        "candidate": candidate,
        "logistic_baseline": logistic,
        "rule_baseline": rule_baseline,
        "folds": folds,
        "fold_stability": fold_stability,
    }


def evaluate_v2_quality_gate(
    evaluation: dict[str, Any], *, explicit_feedback_count: int,
    explicit_focus_count: int, explicit_distract_count: int, distinct_feedback_days: int,
) -> dict[str, Any]:
    """Honest quality gate based on unique feedback sessions and held-out folds."""
    candidate = evaluation.get("candidate", {})
    rule_baseline = evaluation.get("rule_baseline", {})
    fold_stability = evaluation.get("fold_stability", {}) or {}
    candidate_brier = float(candidate.get("brier_score", 1.0))
    rule_brier = float(rule_baseline.get("brier_score", 1.0))
    checks = {
        "minimum_days": distinct_feedback_days >= 7,
        "minimum_explicit_feedback": explicit_feedback_count >= 20,
        "minimum_class_feedback": explicit_focus_count >= 5 and explicit_distract_count >= 5,
        "balanced_accuracy": float(candidate.get("balanced_accuracy", 0.0)) >= 0.55,
        "minority_f1": float(candidate.get("minority_f1", 0.0)) >= 0.40,
        "calibration_better_than_rule": candidate_brier <= rule_brier + 0.01,
        "stable_date_folds": bool(fold_stability.get("passed", False)),
    }
    is_passed = evaluation.get("status") == "evaluated" and all(checks.values())
    # Progressive deployment tier (architecture plan E/2.1): instead of a
    # binary ready/shadow, a partially-qualified model can still serve at
    # low confidence so users get ML value before the full 7-day gate.
    evaluated = evaluation.get("status") == "evaluated"
    low_conf = (
        evaluated
        and distinct_feedback_days >= 3
        and float(candidate.get("balanced_accuracy", 0.0)) >= 0.55
    )
    tier = "full_ready" if is_passed else ("low_confidence" if low_conf else "shadow")
    return {
        "passed": is_passed,
        "mode": "ready" if is_passed else "shadow",
        "deployment_tier": tier,
        "checks": checks,
        "explicit_feedback_count": explicit_feedback_count,
        "explicit_focus_count": explicit_focus_count,
        "explicit_distract_count": explicit_distract_count,
        "distinct_feedback_days": distinct_feedback_days,
    }


# ── Internal helpers ──

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


def _parse_window(row: dict[str, Any]) -> tuple[datetime, datetime, dict[str, Any]] | None:
    try:
        if int(row.get("feature_schema_version", 0)) != FEATURE_SCHEMA_VERSION:
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
    """Heuristic weak label for un-labelled windows (architecture plan E/2.1).

    Explicit user feedback still wins; this rule only fills windows with no
    overlapping feedback session. The thresholds encode high-confidence
    behavioural signals only (a single app held for a long time, or an
    extreme switch storm) so the weak labels stay conservative:
      - top_app_ratio > 0.9 and idle_ratio < 0.1  -> focus (deep work)
      - app_switch_count > 8 and input_active_ratio < 0.2 -> distract
    Everything else is treated as mixed (excluded from training).
    """
    sw = _finite_float(features.get("app_switch_count", 0))
    idle = _finite_float(features.get("idle_ratio", 0))
    top = _finite_float(features.get("top_app_ratio", 0))
    active = _finite_float(features.get("input_active_ratio", 0))
    if idle > 0.8:
        return -1
    # Deep-focus: mostly one app, low idle, meaningful input.
    if top > 0.9 and idle < 0.1 and active > 0.15:
        return 1
    # Distraction: heavy switching with little focused input.
    if sw > 8 and active < 0.2:
        return 0
    # Keep a couple of gentler legacy signals for early cold-start days.
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


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    ba = balanced_accuracy_score(y_true, y_pred)
    unique, counts = np.unique(y_true, return_counts=True)
    minority = unique[np.argmin(counts)]
    minority_f1 = f1_score(
        (y_true == minority).astype(int),
        (y_pred == minority).astype(int),
        zero_division=0.0,
    )
    try:
        brier = brier_score_loss(y_true, y_proba)
    except Exception:
        brier = 1.0
    result: dict[str, Any] = {
        "balanced_accuracy": round(float(ba), 6),
        "minority_f1": round(float(minority_f1), 6),
        "brier_score": round(float(brier), 6),
    }
    if len(unique) == 2:
        try:
            result["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 6)
            result["average_precision"] = round(float(average_precision_score(y_true, y_proba)), 6)
        except ValueError:
            pass
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    result["calibration"] = _calibration_bins(y_true, y_proba)
    return result


def _calibration_bins(y_true: np.ndarray, y_proba: np.ndarray) -> list[dict[str, float]]:
    """Return binned predicted-vs-observed calibration for held-out data."""
    if len(y_proba) == 0:
        return []
    edges = np.linspace(0.0, 1.0, TRAIN_CONFIG.calibration_bins + 1)
    bins: list[dict[str, float]] = []
    for i in range(TRAIN_CONFIG.calibration_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask_bin = (y_proba >= lo) & (y_proba <= hi)
        count = int(mask_bin.sum())
        if count == 0:
            continue
        bins.append({
            "bin_low": round(lo, 4),
            "bin_high": round(hi, 4),
            "count": count,
            "mean_prediction": round(float(np.mean(y_proba[mask_bin])), 4),
            "fraction_positive": round(float(np.mean(y_true[mask_bin])), 4),
        })
    return bins
