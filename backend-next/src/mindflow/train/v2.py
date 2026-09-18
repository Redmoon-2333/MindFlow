"""Training utilities for privacy-preserving feature windows.

Generates labeled 24-dim v2 feature windows from activity data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
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
from mindflow.train.grouping import SESSION_WEIGHT_TOTAL, build_grouping_plan
from mindflow.train.models.ensemble import CalibrationUnavailableError

TASK_TYPE_MAP = {
    "coding": 0, "writing": 1, "study": 2, "meeting": 3, "admin": 4,
    "creative": 5, "other": 6, "gaming": 7, "entertainment": 8,
    "browsing": 9, "communication": 10,
}

# Same class used by ModelManager in production so evaluation and deployment
# share hyperparameters, scaling, and soft-voting behaviour.
def make_v2_classifier() -> Any:
    from mindflow.train.models.ensemble import EnsembleClassifier

    # Public default stays raw (calibration=None): post-hoc calibration only
    # helps once the dataset is large/clean enough (measured on the
    # window-label-augmented real data 2026-08-20: Brier 0.228->0.223, BA
    # 0.641->0.648).  Production training enables it explicitly via
    # ``run_training(..., calibration="sigmoid")``; small/toy datasets should
    # pass ``calibration=None``.
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
    window_label_count: int = 0
    # Auxiliary window labels widen supervision but must never enter the
    # explicit-only evaluation path. ``train_mask`` is the supervision set
    # (explicit ∪ window_label); ``explicit_mask`` stays feedback-only so
    # metrics keep meaning "accuracy on real user feedback".
    train_mask: np.ndarray | None = None
    window_label_mask: np.ndarray | None = None
    conflict_window_count: int = 0
    ambiguous_window_count: int = 0
    #: Real feedback session id per sample ("" when the row is not explicit).
    #: Distinct from ``session_ids``, which holds the feature-window id used to
    #: align rows with their source windows.
    sample_feedback_ids: list[str] = field(default_factory=list)
    #: date -> the feedback sessions that matched windows on that date.
    session_dates: dict[str, list[str]] = field(default_factory=dict)
    #: Date-block group id per sample (cross-midnight sessions merged).
    group_ids: list[str] = field(default_factory=list)
    #: group id -> the calendar dates it covers.
    date_groups: dict[str, list[str]] = field(default_factory=dict)
    #: How many sessions had their dates merged into one group.
    merged_session_count: int = 0
    explicit_weight_total: float = 0.0
    auxiliary_weight_total: float = 0.0
    legacy_weight_total: float = 0.0


# Confidence weight for user-calibrated window labels (option B).  These are
# auto-annotated from the user's per-day activity calibration, so they sit
# between explicit per-session feedback (1.0) and the weak heuristic (0.3).
WINDOW_LABEL_WEIGHT = 0.8

# A feedback session only labels a window when it covers at least this share
# of the window's duration. A one-second overlap from a session that mostly
# happened elsewhere is not evidence about this window.
MIN_WINDOW_COVERAGE = 0.5


def prepare_v2_training_data(
    feature_windows: list[dict[str, Any]],
    feedback_sessions: list[dict[str, Any]],
    window_labels: dict[str, int] | None = None,
) -> V2TrainingData:
    """Prepare training data by matching feature windows to feedback via time overlap.

    ``window_labels`` is an optional opt-in source (id -> 1/0/-1) of
    user-calibrated window labels.  Explicit feedback still wins for any
    window it overlaps; the remaining windows with a window label become
    strong annotated samples (weight ``WINDOW_LABEL_WEIGHT``) instead of weak
    heuristic ones.  Quality-gate counts stay feedback-only — this only
    increases the supervision available to the classifier/evaluator.
    """
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
    window_label_count = 0

    X_list: list[list[float]] = []
    y_list: list[int] = []
    w_list: list[float] = []
    sid_list: list[str] = []
    date_list: list[str] = []
    explicit_list: list[bool] = []
    source_list: list[str] = []
    window_label_list: list[bool] = []
    # Per-sample provenance needed downstream: which real feedback session a
    # row belongs to ("" when not explicit), and every date each session's
    # windows landed on (so a cross-midnight session still groups together).
    sample_session_list: list[str] = []
    session_dates: dict[str, set[str]] = {}
    conflict_count = 0
    ambiguous_count = 0

    for row in feature_windows:
        parsed = _parse_window(row)
        if parsed is None:
            continue
        start, end, features = parsed

        feature_row = [features.get(name, 0.0) for name in V2_FEATURE_NAMES]
        X_list.append(feature_row)

        # Match by sufficient time overlap with feedback sessions. Every
        # qualifying session is considered (not just the first), so a window
        # covered by two sessions with opposite labels is detected as a
        # conflict instead of silently adopting whichever came first.
        window_seconds = (end - start).total_seconds()
        matched_label: int | None = None
        matched_sid: str | None = None
        matched_start: datetime | None = None
        seen_labels: set[int] = set()
        matching_sessions: set[str] = set()
        best_cover = 0.0
        for sid, fb_start, fb_end, fb_label, _fb_task in feedback_intervals:
            overlap = _overlap_seconds(start, end, fb_start, fb_end)
            if overlap <= 0:
                continue
            coverage = overlap / window_seconds if window_seconds > 0 else 0.0
            if coverage < MIN_WINDOW_COVERAGE:
                continue
            if fb_label is None:
                # A `mixed` session overlapping this window makes the window's
                # feedback ambiguous: it must not silently fall through to a
                # window label or the weak heuristic.
                ambiguous_count += 1
                matched_label = None
                matched_sid = None
                seen_labels = {-1}
                break
            seen_labels.add(fb_label)
            matching_sessions.add(sid)
            if coverage > best_cover or (
                coverage == best_cover and (matched_sid is None or sid < matched_sid)
            ):
                best_cover = coverage
                matched_label = fb_label
                matched_sid = sid
                matched_start = fb_start

        if len(seen_labels) > 1:
            # Same window, contradictory feedback -> exclude from statistics
            # rather than pick a winner. Counted and reported, never dropped
            # silently, and never allowed to fall through to a weaker source.
            conflict_count += 1
            matched_label = None
            matched_sid = None
            seen_labels = {-2}

        wid = str(row.get("id", ""))
        sample_date = start.strftime("%Y-%m-%d")
        matched_session = ""
        if matched_label is not None and seen_labels not in ({-1}, {-2}):
            y_list.append(matched_label)
            w_list.append(1.0)
            explicit_list.append(True)
            window_label_list.append(False)
            source_list.append("explicit")
            matched_window_count += 1
            # Grouping uses every compatible session, independently of the
            # single attribution chosen for session-balanced weights.
            for sid in matching_sessions:
                session_dates.setdefault(sid, set()).add(sample_date)
            # The session id is the real feedback session, not the window id.
            if matched_sid is not None:
                matched_session = matched_sid
                explicit_session_ids.add(matched_sid)
                if matched_label == 1:
                    focus_sessions.add(matched_sid)
                else:
                    distract_sessions.add(matched_sid)
                if matched_start is not None:
                    feedback_days.add(matched_start.strftime("%Y-%m-%d"))
        elif seen_labels == {-1}:
            # Ambiguous (mixed feedback) — no label, no fallback.
            y_list.append(-1)
            w_list.append(0.0)
            explicit_list.append(False)
            window_label_list.append(False)
            source_list.append("feedback_mixed")
            mixed_count += 1
        elif seen_labels == {-2}:
            # Contradictory feedback — no label, no fallback.
            y_list.append(-1)
            w_list.append(0.0)
            explicit_list.append(False)
            window_label_list.append(False)
            source_list.append("feedback_conflict")
            mixed_count += 1
        elif window_labels is not None and (window_label := window_labels.get(wid)) is not None:
            if window_label >= 0:
                y_list.append(window_label)
                w_list.append(WINDOW_LABEL_WEIGHT)
                # Auxiliary supervision only: NOT part of the explicit
                # evaluation path (that path gates deployment).
                explicit_list.append(False)
                window_label_list.append(True)
                source_list.append("window_label")
                window_label_count += 1
            else:
                # Explicitly excluded (mixed) window label -> drop like a
                # weak-mixed sample would be.
                y_list.append(-1)
                w_list.append(0.0)
                explicit_list.append(False)
                window_label_list.append(False)
                source_list.append("window_label_excluded")
                mixed_count += 1
        else:
            weak = _weak_label(features)
            y_list.append(weak)
            w_list.append(0.3)
            explicit_list.append(False)
            window_label_list.append(False)
            source_list.append("weak")
            if weak == -1:
                mixed_count += 1

        sid_list.append(wid)
        sample_session_list.append(matched_session)
        date_list.append(sample_date)

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int32)
    w = np.asarray(w_list, dtype=np.float64)
    explicit_mask = np.asarray(explicit_list, dtype=np.bool_)
    window_label_mask = np.asarray(window_label_list, dtype=np.bool_)

    # Filter out mixed (label -1)
    valid = y >= 0

    kept_dates = [d for d, v in zip(date_list, valid, strict=True) if v]
    kept_sources = [s for s, v in zip(source_list, valid, strict=True) if v]
    kept_sessions = [s for s, v in zip(sample_session_list, valid, strict=True) if v]
    kept_labels = y[valid]

    # Re-weight here so every consumer — evaluation *and* the deployed
    # classifier — trains on the same supervision. Computing session-balanced
    # weights only inside the evaluator left deployment using the old
    # per-window weights, so the metrics would not describe the shipped model.
    plan = build_grouping_plan(
        dates=kept_dates,
        labels=kept_labels.tolist(),
        sources=kept_sources,
        window_dates_by_session={sid: sorted(ds) for sid, ds in session_dates.items()},
        session_id_by_sample=kept_sessions,
    )

    return V2TrainingData(
        features=X[valid], labels=kept_labels,
        sample_weights=np.asarray(plan.weights, dtype=np.float64),
        session_ids=[s for s, v in zip(sid_list, valid, strict=True) if v],
        dates=kept_dates,
        explicit_mask=explicit_mask[valid],
        explicit_feedback_count=len(explicit_session_ids),
        explicit_focus_count=len(focus_sessions),
        explicit_distract_count=len(distract_sessions),
        distinct_feedback_days=len(feedback_days),
        mixed_window_count=mixed_count,
        matched_window_count=matched_window_count,
        label_sources=kept_sources,
        feature_names=list(V2_FEATURE_NAMES),
        window_label_count=window_label_count,
        train_mask=explicit_mask[valid] | window_label_mask[valid],
        window_label_mask=window_label_mask[valid],
        conflict_window_count=conflict_count,
        ambiguous_window_count=ambiguous_count,
        sample_feedback_ids=kept_sessions,
        session_dates={sid: sorted(ds) for sid, ds in session_dates.items()},
        group_ids=plan.group_ids,
        date_groups={g: list(ds) for g, ds in plan.group_dates.items()},
        merged_session_count=len(plan.merged_sessions),
        explicit_weight_total=plan.explicit_total_weight,
        auxiliary_weight_total=plan.auxiliary_total_weight,
        legacy_weight_total=round(float(w[valid].sum()), 6),
    )


def training_sample_weights(data: V2TrainingData, mask: np.ndarray) -> np.ndarray:
    """Recompute supervision budgets from these training rows alone."""
    sessions = data.sample_feedback_ids or ["" for _ in data.dates]
    plan = build_grouping_plan(
        dates=[d for d, keep in zip(data.dates, mask, strict=True) if keep],
        labels=data.labels[mask].tolist(),
        sources=[s for s, keep in zip(data.label_sources, mask, strict=True) if keep],
        window_dates_by_session={},
        session_id_by_sample=[s for s, keep in zip(sessions, mask, strict=True) if keep],
    )
    return np.asarray(plan.weights, dtype=np.float64)


def evaluate_v2_candidates(
    data: V2TrainingData, *, random_state: int = 42, calibration: str | None = None,
) -> dict[str, Any]:
    """Grouped cross-validation on explicit feedback only.

    The held-out folds contain **only** windows labelled by real user
    feedback. Auxiliary window labels and heuristic samples train the folds
    but are never scored: this evaluation decides whether the model may be
    promoted, so it must be measured against the user's own judgements.
    Auxiliary-label behaviour is reported separately by
    :func:`evaluate_auxiliary_signal`.

    Grouping is by *date block*, not by raw calendar date: dates joined by a
    session that runs across midnight are merged into one group, so no session
    can contribute rows to both sides of a fold boundary. Sample weights are
    session-balanced — each feedback session contributes a fixed total split
    across its windows — so a long session cannot outvote a short one.

    Rule and logistic baselines are computed inside the same held-out folds;
    in-sample comparisons are intentionally not reported as evidence.

    ``calibration`` mirrors ``run_training`` — production passes
    ``"sigmoid"`` so evaluation matches the deployed classifier exactly.
    """
    calibration_result: dict[str, Any] = {
        "method": calibration,
        "status": "unavailable" if calibration is not None else "not_requested",
    }
    mask = data.explicit_mask
    if mask.sum() < 10:
        return {
            "status": "insufficient_data",
            "calibration": calibration_result,
            "candidate": {},
            "logistic_baseline": {},
            "rule_baseline": {},
            "folds": [],
            "fold_stability": {},
            "explicit_sample_count": int(mask.sum()),
        }

    X = data.features[mask]
    y = data.labels[mask]
    dates = [d for d, m in zip(data.dates, mask, strict=True) if m]
    sample_sources = [s for s, m in zip(data.label_sources, mask, strict=True) if m]
    sample_sessions = (
        [s for s, m in zip(data.sample_feedback_ids, mask, strict=True) if m]
        if data.sample_feedback_ids
        else ["" for _ in dates]
    )

    # Date blocks: merge dates spanned by one session so cross-midnight
    # sessions cannot straddle a fold boundary.
    if data.group_ids and len(data.group_ids) == len(data.dates):
        group_ids = [g for g, m in zip(data.group_ids, mask, strict=True) if m]
        plan = None
    else:
        plan = build_grouping_plan(
            dates=dates,
            labels=y.tolist(),
            sources=sample_sources,
            window_dates_by_session=dict(data.session_dates),
            session_id_by_sample=sample_sessions,
        )
        group_ids = plan.group_ids

    if len(set(group_ids)) < 3:
        return {
            "status": "insufficient_data",
            "calibration": calibration_result,
            "candidate": {},
            "logistic_baseline": {},
            "rule_baseline": {},
            "folds": [],
            "fold_stability": {"reason": "need >=3 independent date groups"},
            "explicit_sample_count": int(mask.sum()),
            "date_group_count": len(set(group_ids)),
        }

    if calibration is not None:
        from mindflow.train.models.ensemble import EnsembleClassifier

        def _make_clf() -> Any:
            return EnsembleClassifier(calibration=calibration)
    else:

        def _make_clf() -> Any:
            return make_v2_classifier()

    _group_dates_map: dict[str, list[str]] = dict(data.date_groups)
    if not _group_dates_map and plan is not None:
        _group_dates_map = {g: list(ds) for g, ds in plan.group_dates.items()}
    groups = np.asarray(group_ids, dtype=object)
    gkf = GroupKFold(n_splits=min(TRAIN_CONFIG.group_folds, len(set(group_ids))))

    # Supervision set available for training inside each fold: every labelled
    # sample (explicit feedback, auxiliary window labels, weak heuristics).
    # Rows belonging to a held-out *date block* are removed, so a
    # cross-midnight session cannot leak its other dates into the trainer.
    sup_mask = data.train_mask if data.train_mask is not None else mask
    sup_sessions = (
        list(data.sample_feedback_ids)
        if data.sample_feedback_ids
        else ["" for _ in data.dates]
    )
    # Reuse merged groups, but compute supervision budgets after splitting.
    if data.group_ids and len(data.group_ids) == len(data.dates):
        sup_groups = np.asarray(data.group_ids, dtype=object)
    else:
        sup_plan = build_grouping_plan(
            dates=list(data.dates),
            labels=data.labels.tolist(),
            sources=list(data.label_sources),
            window_dates_by_session=dict(data.session_dates),
            session_id_by_sample=sup_sessions,
        )
        sup_groups = np.asarray(sup_plan.group_ids, dtype=object)

    all_y_true: list[int] = []
    all_candidate_pred: list[int] = []
    all_candidate_proba: list[float] = []
    all_logistic_pred: list[int] = []
    all_logistic_proba: list[float] = []
    all_rule_pred: list[int] = []
    all_rule_proba: list[float] = []
    folds: list[dict[str, Any]] = []
    rule_baseline_unavailable = False
    calibration_failures: list[str] = []

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

        held_out_groups = set(groups[test_idx])
        fold_train = sup_mask & ~np.isin(sup_groups, list(held_out_groups))
        Xtr = data.features[fold_train]
        ytr = data.labels[fold_train]
        # Session-balanced weights for the fold's training rows.
        wtr = training_sample_weights(data, fold_train)
        aux_in_train = int((data.window_label_mask[fold_train].sum())
                           if data.window_label_mask is not None else 0)

        if len(Xtr) < 10 or len(np.unique(ytr)) < 2:
            folds.append({
                "fold": fold_idx + 1,
                "train_dates": sorted(set(groups[train_idx])),
                "test_dates": sorted(set(groups[test_idx])),
                "balanced_accuracy": 0.0,
                "reason": "train fold has too few labelled samples",
            })
            continue

        clf = _make_clf()
        try:
            clf.fit(
                Xtr,
                ytr,
                list(data.feature_names),
                sample_weight=wtr,
                groups=sup_groups[fold_train],
                label_sources=np.asarray(data.label_sources)[fold_train],
                session_ids=np.asarray(sup_sessions)[fold_train],
            )
            if calibration is not None and clf.calibrator is None:
                raise CalibrationUnavailableError("Requested calibration was not fitted")
        except CalibrationUnavailableError as exc:
            calibration_failures.append(str(exc))
            folds.append({
                "fold": fold_idx + 1,
                "train_groups": sorted(set(sup_groups[fold_train])),
                "test_groups": sorted(held_out_groups),
                "balanced_accuracy": 0.0,
                "reason": "calibration_unavailable",
                "calibration": {
                    "method": calibration, "status": "unavailable", "reason": str(exc),
                },
            })
            continue
        yp = clf.predict(X[test_idx])
        ypr = clf.predict_proba(X[test_idx])[:, 1]

        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, random_state=random_state, class_weight="balanced"
            ),
        )
        lr.fit(
            Xtr,
            ytr,
            logisticregression__sample_weight=wtr,
        )
        lp = lr.predict(X[test_idx])
        lpr = lr.predict_proba(X[test_idx])[:, 1]

        try:
            rule_proba = _rule_probabilities(X[test_idx], data.feature_names)
        except ValueError:
            # This feature set has no behavioural columns to anchor the rule
            # baseline on (an ablation arm). Report it as unavailable rather
            # than fabricating a baseline from unrelated columns.
            rule_proba = None
        rp = (
            (rule_proba >= 0.5).astype(int)
            if rule_proba is not None
            else np.zeros(len(y_true), dtype=int)
        )

        all_y_true.extend(y_true.tolist())
        all_candidate_pred.extend(yp.tolist())
        all_candidate_proba.extend(ypr.tolist())
        all_logistic_pred.extend(lp.tolist())
        all_logistic_proba.extend(lpr.tolist())
        all_rule_pred.extend(rp.tolist())
        if rule_proba is not None:
            all_rule_proba.extend(rule_proba.tolist())
        else:
            rule_baseline_unavailable = True

        folds.append({
            "fold": fold_idx + 1,
            "train_groups": sorted(set(sup_groups[fold_train])),
            "test_groups": sorted(set(groups[test_idx])),
            "train_dates": sorted({d for g in set(sup_groups[fold_train])
                                   for d in _group_dates_map.get(g, [])}),
            "test_dates": sorted({d for g in set(groups[test_idx])
                                  for d in _group_dates_map.get(g, [])}),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, yp), 6),
            "test_size": int(len(y_true)),
            "train_size": int(len(Xtr)),
            "auxiliary_train_samples": aux_in_train,
            "calibration": {
                "method": calibration,
                "status": "fitted" if calibration is not None else "not_requested",
            },
        })

    if calibration_failures:
        calibration_result["reason"] = "; ".join(dict.fromkeys(calibration_failures))
    elif all_y_true and calibration is not None:
        calibration_result["status"] = "fitted"

    if not all_y_true:
        return {
            "status": "calibration_unavailable" if calibration_failures else "insufficient_data",
            "calibration": calibration_result,
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
    rule_baseline: dict[str, Any] = {}
    if not rule_baseline_unavailable and all_rule_proba:
        rule_baseline = _classification_metrics(
            y_true_arr, np.array(all_rule_pred), np.array(all_rule_proba)
        )
    else:
        rule_baseline = {
            "status": "unavailable",
            "reason": (
                "feature set lacks the behavioural columns the rule baseline "
                "is defined on"
            ),
        }

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
        "status": "calibration_unavailable" if calibration_failures else "evaluated",
        "calibration": calibration_result,
        "candidate": candidate,
        "logistic_baseline": logistic,
        "rule_baseline": rule_baseline,
        "folds": folds,
        "fold_stability": fold_stability,
        "explicit_sample_count": int(mask.sum()),
        "auxiliary_training_samples": int(
            (data.train_mask.sum() - mask.sum()) if data.train_mask is not None else 0
        ),
        "date_group_count": len(set(group_ids)),
        "merged_cross_midnight_sessions": (
            plan.merged_sessions if plan is not None
            else data.merged_session_count
        ),
        "session_weight_total": SESSION_WEIGHT_TOTAL,
    }


def evaluate_auxiliary_signal(
    data: V2TrainingData, *, random_state: int = 42,
) -> dict[str, Any]:
    """Measure how well auxiliary window labels agree with real feedback.

    Auxiliary labels are a *training* signal; they must not influence the
    deployment gate. This reports their agreement with the explicit-feedback
    distribution on the same date folds, so the value of widening
    supervision is visible without contaminating the gate.

    Returns ``{"status": "not_available"}`` when there are no auxiliary
    samples or too few explicit samples to compare against.
    """
    if data.window_label_mask is None or int(data.window_label_mask.sum()) == 0:
        return {"status": "not_available", "reason": "no auxiliary window labels"}
    if int(data.explicit_mask.sum()) < 10:
        return {"status": "not_available", "reason": "fewer than 10 explicit samples"}

    aux_y = data.labels[data.window_label_mask]
    aux_focus = int((aux_y == 1).sum())
    aux_distract = int((aux_y == 0).sum())
    explicit_y = data.labels[data.explicit_mask]
    explicit_focus_rate = (
        float((explicit_y == 1).mean()) if len(explicit_y) else None
    )
    return {
        "status": "evaluated",
        "auxiliary_samples": int(len(aux_y)),
        "auxiliary_focus": aux_focus,
        "auxiliary_distract": aux_distract,
        "auxiliary_focus_rate": round(aux_focus / len(aux_y), 4) if len(aux_y) else None,
        "explicit_focus_rate": round(explicit_focus_rate, 4) if explicit_focus_rate else None,
        "note": (
            "Auxiliary window labels widen supervision only; the quality gate "
            "and all deployable metrics above use explicit feedback exclusively."
        ),
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
    calibration = evaluation.get("calibration") or {}
    calibration_available = (
        calibration.get("status") == "fitted"
        and calibration.get("method") in ("sigmoid", "isotonic")
    ) or (
        calibration.get("status") == "not_requested"
        and "method" in calibration and calibration["method"] is None
    )
    checks = {
        "minimum_days": distinct_feedback_days >= 7,
        "minimum_explicit_feedback": explicit_feedback_count >= 20,
        "minimum_class_feedback": explicit_focus_count >= 5 and explicit_distract_count >= 5,
        "balanced_accuracy": float(candidate.get("balanced_accuracy", 0.0)) >= 0.55,
        "minority_f1": float(candidate.get("minority_f1", 0.0)) >= 0.40,
        "calibration_better_than_rule": candidate_brier <= rule_brier + 0.01,
        "calibration_available": calibration_available,
        "stable_date_folds": bool(fold_stability.get("passed", False)),
    }
    is_passed = evaluation.get("status") == "evaluated" and all(checks.values())
    # Progressive deployment tier (architecture plan E/2.1): instead of a
    # binary ready/shadow, a partially-qualified model can still serve at
    # low confidence so users get ML value before the full 7-day gate.
    evaluated = evaluation.get("status") == "evaluated"
    low_conf = (
        evaluated
        and calibration_available
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


def _rule_probabilities(
    X: np.ndarray, feature_names: list[str] | None = None
) -> np.ndarray:
    """Heuristic rule baseline used as the quality gate's comparison point.

    Resolves the four features it uses **by name**. The previous positional
    version (columns 0, 3, 14) silently read whatever happened to sit at those
    offsets, so any change to the feature set produced a meaningless baseline
    instead of an error — and an ablation that removed columns crashed.

    Raises:
        ValueError: when a required feature is absent. A baseline computed from
            the wrong columns is worse than a loud failure; callers that drop
            features on purpose (ablations) must handle this explicitly.
    """
    names = list(feature_names) if feature_names is not None else list(V2_FEATURE_NAMES)
    required = ("app_switch_count", "top_app_ratio", "idle_ratio")
    missing = [name for name in required if name not in names]
    if missing:
        msg = f"rule baseline needs behavioural features {missing!r}; present: {names!r}"
        raise ValueError(msg)

    def column(feature: str) -> np.ndarray:
        return X[:, names.index(feature)]

    p = np.full(X.shape[0], 0.5)
    switch_count = column("app_switch_count")
    p[switch_count < 5] += 0.2
    p[column("top_app_ratio") > 0.7] += 0.15
    p[switch_count > 20] -= 0.3
    p[column("idle_ratio") > 0.8] -= 0.1
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
