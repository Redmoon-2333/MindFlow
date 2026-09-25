"""Training utilities for privacy-preserving feature windows.

Generates labeled 24-dim v2 feature windows from activity data.

Phase 3.1 (honest evaluation protocol) and 3.4/3.5 (labeling functions and
candidate model set) live in the sibling modules ``evaluation``, ``labeling``
and ``candidates``; this module keeps the data preparation and the
``evaluate_v2_candidates`` / ``evaluate_v2_quality_gate`` entry points that the
training pipeline and the readiness service call, and embeds their output in
the training report.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

# Single authoritative vocabulary lives in the domain layer so BaselineModel
# and training can never drift. Re-export keeps ``mindflow.train.v2`` importers
# (telemetry_service, prediction_service) working unchanged.
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES  # noqa: F401
from mindflow.train import labeling as labeling_functions
from mindflow.train.candidates import (
    CANDIDATE_NAMES,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    RF_XGB_SOFT_VOTING,
    RULE_ENGINE,
    XGBOOST,
    CandidateFit,
    fit_candidate,
    rule_probabilities,
    select_candidate,
)
from mindflow.train.config import TRAIN_CONFIG
from mindflow.train.evaluation import (
    ABSTENTION_BAND_HALF_WIDTH,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EvaluationFold,
    PredictionPool,
    build_forward_chaining_folds,
    classification_metrics,
    fold_stability,
    pooled_report,
    reliability_table,
    temporal_drift_report,
)
from mindflow.train.grouping import SESSION_WEIGHT_TOTAL, build_grouping_plan
from mindflow.train.models.ensemble import CalibrationUnavailableError

TASK_TYPE_MAP = {
    "coding": 0, "writing": 1, "study": 2, "meeting": 3, "admin": 4,
    "creative": 5, "other": 6, "gaming": 7, "entertainment": 8,
    "browsing": 9, "communication": 10,
}

#: Fit order inside one fold.  The soft-voting candidate goes first: it is the
#: only one that can fail (out-of-sample calibration), and a fold it cannot
#: calibrate is dropped whole — consistently for every candidate — so the
#: comparison never mixes a calibrated pool with an uncalibrated one.
_CANDIDATE_FIT_ORDER: tuple[str, ...] = (
    RF_XGB_SOFT_VOTING, RULE_ENGINE, LOGISTIC_REGRESSION, RANDOM_FOREST, XGBOOST,
)


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
    #: Auditable per-labeling-function report (phase 3.4).  Measured over the
    #: kept training rows; exposed in the training report through
    #: :func:`evaluate_v2_candidates`.
    labeling_function_report: dict[str, Any] = field(default_factory=dict)


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
    feedback_context: dict[str, dict[str, Any]] = {}
    for row in feedback_sessions:
        feedback = _parse_feedback(row)
        if feedback is not None:
            parsed_feedback.append(feedback)
            feedback_context[feedback[0]] = {
                labeling_functions.POST_INTERVENTION_STATE_KEY: row.get(
                    labeling_functions.POST_INTERVENTION_STATE_KEY
                ),
                labeling_functions.POST_INTERVENTION_HELPFULNESS_KEY: row.get(
                    labeling_functions.POST_INTERVENTION_HELPFULNESS_KEY
                ),
            }

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
    # Phase 3.4 provenance: each row's raw labeling-function outputs plus the
    # explicit label it carries (if any).  ``build_labeling_function_report``
    # turns these into coverage/conflict/agreement per function.
    lf_outputs_list: list[dict[str, int]] = []
    lf_explicit_list: list[int | None] = []
    lf_contexts_list: list[dict[str, Any]] = []
    lf_features_list: list[dict[str, Any]] = []

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
        explicit_label_for_row: int | None = None
        if matched_label is not None and seen_labels not in ({-1}, {-2}):
            y_list.append(matched_label)
            w_list.append(1.0)
            explicit_list.append(True)
            window_label_list.append(False)
            source_list.append("explicit")
            matched_window_count += 1
            explicit_label_for_row = matched_label
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
            # Weak supervision is now resolved from the explicit labeling
            # functions (phase 3.4) instead of one opaque helper; the
            # composition is unchanged, so the labels are too.
            weak = labeling_functions.weak_label(features)
            y_list.append(weak)
            w_list.append(0.3)
            explicit_list.append(False)
            window_label_list.append(False)
            source_list.append("weak")
            if weak == -1:
                mixed_count += 1

        # Per-row labeling-function provenance (phase 3.4).  Recorded for every
        # parsed window, including the ones about to be dropped, so the report
        # can say what each function did on the data that actually arrived.
        row_context = dict(feedback_context.get(matched_session or "", {}))
        row_context["explicit_label"] = explicit_label_for_row
        lf_outputs_list.append(
            labeling_functions.compute_labeling_function_outputs(features, row_context)
        )
        lf_explicit_list.append(explicit_label_for_row)
        lf_contexts_list.append(row_context)
        lf_features_list.append(dict(features))

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

    # Labeling-function provenance over exactly the rows that survive, so the
    # agreement numbers describe the same frame the classifier trains on.
    kept_features = [f for f, v in zip(lf_features_list, valid, strict=True) if v]
    kept_lf_labels = [label for label, v in zip(lf_explicit_list, valid, strict=True) if v]
    kept_lf_contexts = [c for c, v in zip(lf_contexts_list, valid, strict=True) if v]
    labeling_report = labeling_functions.build_labeling_function_report(
        kept_features, kept_lf_labels, contexts=kept_lf_contexts,
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
        labeling_function_report=labeling_report,
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


def _unavailable_reason(name: str) -> str:
    """Why a candidate has no held-out metrics in this environment."""
    if name == RULE_ENGINE:
        return (
            "feature set lacks the behavioural columns the rule baseline "
            "is defined on"
        )
    if name == XGBOOST:
        return "xgboost is not installed in this environment"
    return "candidate produced no held-out predictions"


def _per_fold_accuracy(reports: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    """Per-candidate balanced accuracy keyed by fold index, from fold reports."""
    per_fold: dict[str, dict[int, float]] = {}
    for report in reports:
        candidates = report.get("candidates") or {}
        for name, entry in candidates.items():
            if isinstance(entry, dict) and "balanced_accuracy" in entry:
                per_fold.setdefault(name, {})[int(report["fold"])] = float(
                    entry["balanced_accuracy"]
                )
    return per_fold


def _scheme_stability(reports: list[dict[str, Any]], scheme: str) -> dict[str, Any]:
    """Fold-stability summary for one scheme, with skipped folds failing it."""
    values = [
        float(report["balanced_accuracy"])
        for report in reports if "balanced_accuracy" in report
    ]
    sizes = [int(report.get("test_size", 0)) for report in reports]
    stability = fold_stability(values, sizes, scheme=scheme)
    skipped = [report for report in reports if "reason" in report]
    if skipped:
        stability["passed"] = False
        stability["skipped_folds"] = len(skipped)
        stability["skip_reasons"] = sorted({str(report["reason"]) for report in skipped})
    return stability


def _not_evaluated_extras(
    labeling_report: dict[str, Any],
    *,
    reason: str,
    calibration_failures: list[str] | None = None,
) -> dict[str, Any]:
    """Every phase-3.1 output key in its "not evaluated" form.

    Present (rather than missing) so the quality gate fails closed for the
    right, visible reason instead of tripping over an absent key.
    """
    return {
        "candidate_name": None,
        "primary_scheme": None,
        "candidates": {},
        "candidate_selection": {
            "rule": "keep_simplest_unless_stably_beaten",
            "status": "not_available",
            "selected": None,
            "publication_model": RF_XGB_SOFT_VOTING,
            "publication_consistent": False,
            "reason": reason,
            "decisions": [],
        },
        "forward_chaining": {
            "status": "not_evaluated",
            "scheme": "forward_chaining",
            "policy": "leave_future_days_out",
            "fold_count": 0,
            "folds": [],
            "candidates": {},
            "reason": reason,
            "calibration_failures": list(calibration_failures or []),
        },
        "date_fold_stability": {
            "scheme": "date_folds", "passed": False, "reason": reason,
        },
        "future_fold_stability": {
            "scheme": "forward_chaining", "passed": False, "reason": reason,
        },
        "session_metrics": {"status": "not_available", "reason": reason},
        "date_metrics": {"status": "not_available", "reason": reason},
        "probability_metrics": {"status": "not_available", "reason": reason},
        "bootstrap": {"status": "not_available", "reason": reason},
        "abstention": {"status": "not_available", "reason": reason},
        "shadow_drift": {"status": "not_available", "reason": reason},
        "labeling_functions": labeling_report,
    }


def evaluate_v2_candidates(
    data: V2TrainingData, *, random_state: int = 42, calibration: str | None = None,
) -> dict[str, Any]:
    """Honest out-of-fold evaluation of the candidate model set.

    The held-out folds contain **only** windows labelled by real user
    feedback. Auxiliary window labels and heuristic samples train the folds
    but are never scored: this evaluation decides whether the model may be
    promoted, so it must be measured against the user's own judgements.
    Auxiliary-label behaviour is reported separately by
    :func:`evaluate_auxiliary_signal`.

    Two fold schemes run, both leak-free:

    * ``forward_chaining`` (primary since phase 3.1) — expanding window over
      chronologically ordered date blocks: every fold trains only on blocks
      strictly earlier than the blocks it is scored on, so no future date and
      no future session can appear in training;
    * ``date_folds`` — the legacy ``GroupKFold`` over date blocks (dates joined
      by a cross-midnight session are one block).

    Grouping is by *date block*, not by raw calendar date, so no session can
    contribute rows to both sides of a fold boundary. Sample weights are
    session-balanced — each feedback session contributes a fixed total split
    across its windows — so a long session cannot outvote a short one.

    Five candidates are compared in every run (rule engine, balanced logistic
    regression, random forest, XGBoost, RF+XGBoost soft voting); the winner is
    chosen by the stability rule in :mod:`mindflow.train.candidates` and the
    decision trail is recorded under ``candidate_selection``. Metrics are
    reported at window, session and date-block level, with PR-AUC/Brier/ECE, a
    reliability table, session-bootstrap confidence intervals, an
    abstention/coverage curve and a shadow-drift check.

    ``calibration`` mirrors ``run_training`` — production passes
    ``"sigmoid"`` so evaluation matches the deployed classifier exactly.
    """
    calibration_result: dict[str, Any] = {
        "method": calibration,
        "status": "unavailable" if calibration is not None else "not_requested",
    }
    labeling_report = dict(data.labeling_function_report or {})
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
            **_not_evaluated_extras(
                labeling_report, reason="fewer than 10 explicit feedback samples"
            ),
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
            **_not_evaluated_extras(
                labeling_report, reason="need >=3 independent date groups"
            ),
        }

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

    def _evaluate_scheme(
        scheme: str, fold_defs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[Any]]], list[str]]:
        """Fit every candidate on every fold of one scheme.

        Returns the per-fold reports, the pooled per-candidate predictions
        (mask-space positions plus labels/scores), and any calibration failure
        messages.
        """
        reports: list[dict[str, Any]] = []
        scheme_pools: dict[str, dict[str, list[Any]]] = {
            name: {"positions": [], "y_true": [], "y_pred": [], "y_proba": []}
            for name in CANDIDATE_NAMES
        }
        failures: list[str] = []
        for fold_def in fold_defs:
            test_idx = np.asarray(fold_def["test_idx"], dtype=np.int_)
            y_true = y[test_idx]
            fold_report: dict[str, Any] = {
                "fold": int(fold_def["index"]),
                "scheme": scheme,
                "train_groups": list(fold_def["train_groups"]),
                "test_groups": list(fold_def["test_groups"]),
                "train_dates": list(fold_def["train_dates"]),
                "test_dates": list(fold_def["test_dates"]),
                "test_size": int(len(test_idx)),
            }
            if scheme == "forward_chaining":
                fold_report["cutoff_date"] = fold_def["cutoff_date"]
            if len(np.unique(y_true)) < 2 or len(y_true) < 5:
                fold_report["balanced_accuracy"] = 0.0
                fold_report["reason"] = "test fold too small for stable metrics"
                reports.append(fold_report)
                continue

            # Train on EXACTLY the fold's declared training groups. Using
            # "every support row that is not in the test groups" instead would
            # leak future date blocks into a forward-chaining fold (and would
            # make the fold's reported ``train_groups`` disagree with the rows
            # the candidate was actually fitted on).
            declared_train_groups = list(fold_def["train_groups"])
            fold_train = sup_mask & np.isin(sup_groups, declared_train_groups)
            Xtr = data.features[fold_train]
            ytr = data.labels[fold_train]
            # Session-balanced weights for the fold's training rows.
            wtr = training_sample_weights(data, fold_train)
            aux_in_train = int((data.window_label_mask[fold_train].sum())
                               if data.window_label_mask is not None else 0)

            if len(Xtr) < 10 or len(np.unique(ytr)) < 2:
                fold_report["balanced_accuracy"] = 0.0
                fold_report["reason"] = "train fold has too few labelled samples"
                reports.append(fold_report)
                continue

            fold_fits: dict[str, CandidateFit] = {}
            calibration_failed = False
            for name in _CANDIDATE_FIT_ORDER:
                try:
                    fold_fits[name] = fit_candidate(
                        name,
                        Xtr,
                        ytr,
                        wtr,
                        X[test_idx],
                        feature_names=list(data.feature_names),
                        random_state=random_state,
                        groups=sup_groups[fold_train],
                        label_sources=np.asarray(data.label_sources)[fold_train],
                        session_ids=np.asarray(sup_sessions)[fold_train],
                        calibration=calibration,
                    )
                except CalibrationUnavailableError as exc:
                    # The fold is dropped whole: mixing a calibrated pool with
                    # an uncalibrated one would make the comparison meaningless.
                    failures.append(str(exc))
                    fold_report["balanced_accuracy"] = 0.0
                    fold_report["reason"] = "calibration_unavailable"
                    fold_report["calibration"] = {
                        "method": calibration, "status": "unavailable",
                        "reason": str(exc),
                    }
                    reports.append(fold_report)
                    calibration_failed = True
                    break
            if calibration_failed:
                continue

            fold_candidates: dict[str, Any] = {}
            for name, fit in fold_fits.items():
                if fit.predictions is None or fit.probabilities is None:
                    fold_candidates[name] = {
                        "status": fit.status, "reason": fit.reason,
                    }
                    continue
                metrics = classification_metrics(
                    y_true, fit.predictions, fit.probabilities
                )
                fold_candidates[name] = {
                    key: metrics[key]
                    for key in (
                        "balanced_accuracy", "minority_f1", "brier_score",
                        "pr_auc", "expected_calibration_error",
                    )
                }
                scheme_pools[name]["positions"].extend(int(i) for i in test_idx)
                scheme_pools[name]["y_true"].extend(int(v) for v in y_true)
                scheme_pools[name]["y_pred"].extend(int(v) for v in fit.predictions)
                scheme_pools[name]["y_proba"].extend(float(v) for v in fit.probabilities)

            fold_report["balanced_accuracy"] = float(
                fold_candidates.get(RF_XGB_SOFT_VOTING, {}).get("balanced_accuracy", 0.0)
            )
            fold_report["train_size"] = int(len(Xtr))
            fold_report["auxiliary_train_samples"] = aux_in_train
            fold_report["calibration"] = {
                "method": calibration,
                "status": "fitted" if calibration is not None else "not_requested",
            }
            fold_report["candidates"] = fold_candidates
            reports.append(fold_report)
        return reports, scheme_pools, failures

    def _pool(scheme_pools: dict[str, dict[str, list[Any]]], name: str) -> PredictionPool | None:
        rows = scheme_pools.get(name, {}).get("y_true") or []
        if not rows:
            return None
        positions = np.asarray(scheme_pools[name]["positions"], dtype=np.int_)
        return PredictionPool(
            y_true=np.asarray(scheme_pools[name]["y_true"], dtype=np.int_),
            y_pred=np.asarray(scheme_pools[name]["y_pred"], dtype=np.int_),
            y_proba=np.asarray(scheme_pools[name]["y_proba"], dtype=np.float64),
            session_ids=[sample_sessions[i] for i in positions],
            group_ids=[group_ids[i] for i in positions],
            dates=[dates[i] for i in positions],
            positions=positions,
        )

    def _aggregate(scheme_pools: dict[str, dict[str, list[Any]]]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name in CANDIDATE_NAMES:
            pool = _pool(scheme_pools, name)
            if pool is None:
                continue
            metrics = classification_metrics(pool.y_true, pool.y_pred, pool.y_proba)
            scores[name] = float(metrics["balanced_accuracy"])
        return scores

    # ── Scheme 1: legacy date-grouped folds ──────────────────────────────
    gkf = GroupKFold(n_splits=min(TRAIN_CONFIG.group_folds, len(set(group_ids))))
    date_fold_defs: list[dict[str, Any]] = []
    for fold_index, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        test_groups = sorted({str(g) for g in groups[test_idx]})
        train_groups = sorted({str(g) for g in groups[train_idx]})
        date_fold_defs.append({
            "index": fold_index,
            "test_idx": test_idx,
            "test_groups": test_groups,
            "train_groups": train_groups,
            "train_dates": sorted({d for g in train_groups
                                   for d in _group_dates_map.get(g, [])}),
            "test_dates": sorted({d for g in test_groups
                                  for d in _group_dates_map.get(g, [])}),
            "cutoff_date": None,
        })
    folds, date_pools, calibration_failures = _evaluate_scheme("date_folds", date_fold_defs)

    if not any(date_pools[name]["y_true"] for name in CANDIDATE_NAMES):
        if calibration_failures:
            calibration_result["reason"] = "; ".join(dict.fromkeys(calibration_failures))
        return {
            "status": "calibration_unavailable" if calibration_failures else "insufficient_data",
            "calibration": calibration_result,
            "candidate": {},
            "logistic_baseline": {},
            "rule_baseline": {},
            "folds": folds,
            "fold_stability": {"reason": "no valid held-out folds"},
            **_not_evaluated_extras(
                labeling_report,
                reason="no valid held-out folds",
                calibration_failures=calibration_failures,
            ),
        }
    if calibration_failures:
        calibration_result["reason"] = "; ".join(dict.fromkeys(calibration_failures))
    elif calibration is not None:
        calibration_result["status"] = "fitted"

    # ── Scheme 2 (primary): leave-future-days-out ────────────────────────
    forward_folds: list[EvaluationFold] = build_forward_chaining_folds(
        group_ids, group_dates=_group_dates_map,
    )
    forward_fold_defs: list[dict[str, Any]] = [
        {
            "index": fold.index,
            "test_idx": np.asarray(
                [i for i, g in enumerate(group_ids) if g in set(fold.test_groups)],
                dtype=np.int_,
            ),
            "test_groups": list(fold.test_groups),
            "train_groups": list(fold.train_groups),
            "train_dates": list(fold.train_dates),
            "test_dates": list(fold.test_dates),
            "cutoff_date": fold.cutoff_date,
        }
        for fold in forward_folds
    ]
    forward_reports, forward_pools, forward_failures = _evaluate_scheme(
        "forward_chaining", forward_fold_defs
    )

    # ── Candidate selection on the primary scheme ────────────────────────
    forward_available = any(
        forward_pools[name]["y_true"] for name in CANDIDATE_NAMES
    )
    primary_scheme = "forward_chaining" if forward_available else "date_folds"
    primary_pools = forward_pools if forward_available else date_pools
    secondary_pools = date_pools if forward_available else forward_pools

    primary_aggregate = _aggregate(primary_pools)
    secondary_aggregate = _aggregate(secondary_pools) or None
    primary_per_fold = _per_fold_accuracy(
        forward_reports if forward_available else folds
    )
    selection = select_candidate(
        primary_aggregate, primary_per_fold, secondary_aggregate=secondary_aggregate,
    )
    selected = selection.get("selected")
    if not selected:
        selected = next(
            (name for name in CANDIDATE_NAMES if _pool(primary_pools, name) is not None),
            LOGISTIC_REGRESSION,
        )
        selection["selected"] = selected
        selection["reason"] = (
            "no candidate produced comparable metrics; fell back to the first "
            "available pool"
        )

    # Each fold's headline accuracy now describes the selected candidate — the
    # model the gate is about — instead of the always-complex ensemble.
    for report in [*folds, *forward_reports]:
        entry = (report.get("candidates") or {}).get(selected) or {}
        if "balanced_accuracy" in entry:
            report["balanced_accuracy"] = float(entry["balanced_accuracy"])

    selected_pool = _pool(primary_pools, selected)
    if selected_pool is None:  # pragma: no cover - guarded by the checks above
        raise RuntimeError("selected candidate has no held-out predictions")
    primary_report = pooled_report(
        selected_pool,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        abstention_half_width=ABSTENTION_BAND_HALF_WIDTH,
    )

    primary_fold_reports = forward_reports if forward_available else folds
    primary_fold_values = [
        float(report["balanced_accuracy"])
        for report in primary_fold_reports
        if "balanced_accuracy" in report
    ]
    candidate = dict(primary_report["metrics"])
    if primary_fold_values:
        candidate["fold_balanced_accuracy_range"] = round(
            max(primary_fold_values) - min(primary_fold_values), 6
        )
        candidate["fold_min_balanced_accuracy"] = round(min(primary_fold_values), 6)

    date_stability = _scheme_stability(folds, "date_folds")
    future_stability = _scheme_stability(forward_reports, "forward_chaining")
    # Legacy key keeps its meaning: stability of the date-grouped folds.
    fold_stability_result = dict(date_stability)

    candidate_metrics: dict[str, Any] = {}
    for name in CANDIDATE_NAMES:
        pool = _pool(primary_pools, name)
        if pool is None:
            candidate_metrics[name] = {
                "status": "unavailable",
                "reason": _unavailable_reason(name),
            }
            continue
        metrics = classification_metrics(pool.y_true, pool.y_pred, pool.y_proba)
        candidate_metrics[name] = {"status": "evaluated", **metrics}

    rule_pool = _pool(primary_pools, RULE_ENGINE)
    if rule_pool is not None:
        rule_baseline: dict[str, Any] = classification_metrics(
            rule_pool.y_true, rule_pool.y_pred, rule_pool.y_proba
        )
    else:
        rule_baseline = {
            "status": "unavailable",
            "reason": (
                "feature set lacks the behavioural columns the rule baseline "
                "is defined on"
            ),
        }
    logistic_pool = _pool(primary_pools, LOGISTIC_REGRESSION)
    logistic_baseline: dict[str, Any] = (
        classification_metrics(
            logistic_pool.y_true, logistic_pool.y_pred, logistic_pool.y_proba
        )
        if logistic_pool is not None
        else {"status": "unavailable"}
    )

    shadow_drift = temporal_drift_report(
        X[selected_pool.positions],
        selected_pool.y_proba,
        list(data.feature_names),
        list(selected_pool.dates),
    )

    forward_candidate_metrics: dict[str, Any] = {}
    for name in CANDIDATE_NAMES:
        pool = _pool(forward_pools, name)
        forward_candidate_metrics[name] = (
            {
                "status": "evaluated",
                **classification_metrics(pool.y_true, pool.y_pred, pool.y_proba),
            }
            if pool is not None
            else {"status": "unavailable", "reason": _unavailable_reason(name)}
        )

    return {
        "status": "calibration_unavailable" if calibration_failures else "evaluated",
        "calibration": calibration_result,
        "candidate": candidate,
        "candidate_name": selected,
        "candidates": candidate_metrics,
        "candidate_selection": selection,
        "primary_scheme": primary_scheme,
        "logistic_baseline": logistic_baseline,
        "rule_baseline": rule_baseline,
        "folds": folds,
        "fold_stability": fold_stability_result,
        "date_fold_stability": date_stability,
        "future_fold_stability": future_stability,
        "forward_chaining": {
            "status": "evaluated" if forward_available else "not_available",
            "scheme": "forward_chaining",
            "policy": "leave_future_days_out",
            "fold_count": len(forward_reports),
            "folds": forward_reports,
            "candidates": forward_candidate_metrics,
            "calibration_failures": list(dict.fromkeys(forward_failures)),
        },
        "session_metrics": primary_report["session_metrics"],
        "date_metrics": primary_report["date_metrics"],
        "probability_metrics": {
            "status": "evaluated",
            "pr_auc": candidate.get("pr_auc"),
            "roc_auc": candidate.get("roc_auc"),
            "brier_score": candidate.get("brier_score"),
            "expected_calibration_error": candidate.get("expected_calibration_error"),
            "reliability_table": candidate.get("reliability_table", []),
            "bin_count": len(candidate.get("reliability_table", []) or []),
            "sample_count": int(len(selected_pool)),
        },
        "bootstrap": primary_report["bootstrap"],
        "abstention": primary_report["abstention"],
        "shadow_drift": shadow_drift,
        "labeling_functions": labeling_report,
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
    drift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Honest quality gate: every condition must hold, or the model stays shadow.

    Conditions (phase 3.6 — thresholds are never relaxed to let a candidate
    through, and no evaluation definition is changed to make one pass):

    ============================  ==========================================
    ``minimum_days``              >= 7 distinct feedback days
    ``minimum_explicit_feedback`` >= 20 unique feedback sessions
    ``minimum_class_feedback``    >= 5 focus and >= 5 distracted sessions
    ``balanced_accuracy``         >= 0.55 on the primary held-out scheme
    ``minority_f1``               >= 0.40
    ``calibration_better_than_rule``  Brier no worse than the rule baseline
                                  by more than 0.01
    ``calibration_available``     an out-of-sample calibrator was fitted (or
                                  explicitly not requested)
    ``stable_date_folds``         the date-grouped folds are stable
    ``stable_future_folds``       the leave-future-days-out folds are stable
    ``no_anomalous_drift``        the shadow drift check stayed below its PSI
                                  threshold
    ============================  ==========================================

    ``stable_future_folds`` and ``no_anomalous_drift`` fail **closed**: when
    the evaluation carries no forward-chaining section or no drift report, the
    gate cannot confirm them and the candidate stays in shadow.  Passing an
    explicit ``drift`` overrides the drift section of ``evaluation``.
    """
    candidate = evaluation.get("candidate", {})
    rule_baseline = evaluation.get("rule_baseline", {})
    fold_stability = evaluation.get("fold_stability", {}) or {}
    future_stability = evaluation.get("future_fold_stability") or {}
    drift_report = drift if drift is not None else (evaluation.get("shadow_drift") or {})
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
        "stable_future_folds": bool(future_stability.get("passed", False)),
        "no_anomalous_drift": bool(drift_report) and not bool(
            drift_report.get("anomalous", True)
        ),
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
        "details": {
            "minimum_days": {
                "observed": distinct_feedback_days, "threshold": 7, "comparison": ">=",
            },
            "minimum_explicit_feedback": {
                "observed": explicit_feedback_count, "threshold": 20, "comparison": ">=",
            },
            "minimum_class_feedback": {
                "observed": {
                    "focus": explicit_focus_count, "distracted": explicit_distract_count,
                },
                "threshold": 5,
                "comparison": ">=",
            },
            "balanced_accuracy": {
                "observed": float(candidate.get("balanced_accuracy", 0.0)),
                "threshold": 0.55,
                "comparison": ">=",
            },
            "minority_f1": {
                "observed": float(candidate.get("minority_f1", 0.0)),
                "threshold": 0.40,
                "comparison": ">=",
            },
            "calibration_better_than_rule": {
                "observed": {
                    "candidate_brier": candidate_brier, "rule_brier": rule_brier,
                },
                "threshold": 0.01,
                "comparison": "candidate_brier <= rule_brier + threshold",
            },
            "calibration_available": {
                "observed": calibration.get("status"),
                "method": calibration.get("method"),
            },
            "stable_date_folds": {
                "observed": fold_stability.get("passed"),
                "min_balanced_accuracy": fold_stability.get("min_balanced_accuracy"),
                "range": fold_stability.get("range"),
                "min_test_size": fold_stability.get("min_test_size"),
            },
            "stable_future_folds": {
                "observed": future_stability.get("passed"),
                "min_balanced_accuracy": future_stability.get("min_balanced_accuracy"),
                "range": future_stability.get("range"),
                "min_test_size": future_stability.get("min_test_size"),
                "folds": future_stability.get("fold_count"),
            },
            "no_anomalous_drift": {
                "observed": drift_report.get("statistic"),
                "threshold": drift_report.get("threshold"),
                "anomalous": drift_report.get("anomalous"),
                "unit": drift_report.get("unit"),
            },
        },
        "candidate_name": evaluation.get("candidate_name"),
        "primary_scheme": evaluation.get("primary_scheme"),
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
    """Composed weak label for an un-labelled window.

    Kept as the public entry point of the pre-3.4 helper, but the decision now
    comes from the explicit labeling functions in
    :mod:`mindflow.train.labeling` — same guard, same order, same thresholds,
    so nothing about the weak labels changed; they are just attributable now.
    """
    return labeling_functions.weak_label(features)


def _rule_probabilities(
    X: np.ndarray, feature_names: list[str] | None = None
) -> np.ndarray:
    """Heuristic rule baseline used as the quality gate's comparison point.

    Resolves the features it uses **by name** (see
    :func:`mindflow.train.candidates.rule_probabilities`); a positional read
    would silently produce a meaningless baseline when the feature set changes.

    Raises:
        ValueError: when a required feature is absent.
    """
    names = list(feature_names) if feature_names is not None else list(V2_FEATURE_NAMES)
    return np.asarray(rule_probabilities(np.asarray(X, dtype=np.float64), names))


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    """Window-level metrics; delegates to the evaluation protocol module."""
    return classification_metrics(
        np.asarray(y_true, dtype=np.int_),
        np.asarray(y_pred, dtype=np.int_),
        np.asarray(y_proba, dtype=np.float64),
    )


def _calibration_bins(y_true: np.ndarray, y_proba: np.ndarray) -> list[dict[str, Any]]:
    """Binned predicted-vs-observed calibration table for held-out data."""
    return reliability_table(
        np.asarray(y_true, dtype=np.int_), np.asarray(y_proba, dtype=np.float64)
    )
