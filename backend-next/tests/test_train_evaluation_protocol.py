"""Regression tests for the phase 3.1 / 3.4 / 3.5 / 3.6 ML contracts.

The honest-evaluation workstream (``train/evaluation.py``, ``train/labeling.py``,
``train/candidates.py`` and the ``evaluate_v2_candidates`` /
``evaluate_v2_quality_gate`` entry points in ``train/v2.py``) shipped without
dedicated regression tests.  This module pins the contracts that make the
protocol honest, in the order the phase documents state them:

1. only explicit user feedback is ever scored;
2. forward-chaining folds never see the future (no date *and* no session);
3. folds exist only with >= 3 date blocks and honour ``MIN_FOLD_TEST_SAMPLES``;
4. window metrics are also aggregated to the feedback session and the date block;
5. probabilities are measured as probabilities (PR-AUC / Brier / ECE / table);
6. bootstrap uncertainty is session-resampled and reproducible under a seed;
7. abstention coverage is measured per confidence band;
8. labeling functions report coverage/conflict/agreement and never override a
   user's own verdict;
9. five candidates are always compared, and a complex model is promoted only
   when it is *stably* better;
10. the research HMM stays outside the publication chain;
11. every quality-gate condition can independently force shadow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.train import labeling, v2
from mindflow.train.candidates import (
    CANDIDATE_NAMES,
    DEFAULT_PUBLICATION_MODEL,
    DEPLOYABLE_ORDER,
    LOGISTIC_REGRESSION,
    PROMOTION_MARGIN,
    RANDOM_FOREST,
    RF_XGB_SOFT_VOTING,
    RULE_ENGINE,
    XGBOOST,
    available_candidate_names,
    fit_candidate,
    select_candidate,
)
from mindflow.train.evaluation import (
    ABSTENTION_BAND_HALF_WIDTH,
    ABSTENTION_BAND_WIDTHS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    BOOTSTRAP_UNIT,
    DRIFT_PSI_THRESHOLD,
    MIN_FOLD_TEST_SAMPLES,
    PredictionPool,
    abstention_metrics,
    bootstrap_confidence_intervals,
    build_forward_chaining_folds,
    date_level_metrics,
    expected_calibration_error,
    fold_stability,
    probability_metrics,
    reliability_table,
    session_bootstrap_draws,
    session_level_metrics,
)
from mindflow.train.models import ensemble
from mindflow.train.models.ensemble import EnsembleClassifier
from mindflow.train.models.hmm import HMM_IN_PRODUCTION_CHAIN, HMM_READMISSION_CRITERIA
from mindflow.train.models.manager import ModelManager
from mindflow.train.v2 import (
    evaluate_v2_candidates,
    evaluate_v2_quality_gate,
    prepare_v2_training_data,
)

# ── Fixtures and row builders ────────────────────────────────────────────

#: ``mouse_distance_per_min`` doubles as a provenance marker so a test can tell
#: which rows were scored: every scored row carries the explicit marker.
_EXPLICIT_MARKER = 0.0
_AUXILIARY_MARKER = 500.0
_WEAK_MARKER = 900.0

_FOCUS_FEATURES: dict[str, float] = {
    "top_app_ratio": 0.97,
    "idle_ratio": 0.01,
    "input_active_ratio": 0.8,
    "longest_segment_ratio": 0.95,
    "app_switch_count": 0.0,
    "domain_switch_count": 0.0,
}
_DISTRACT_FEATURES: dict[str, float] = {
    "top_app_ratio": 0.1,
    "idle_ratio": 0.7,
    "input_active_ratio": 0.05,
    "longest_segment_ratio": 0.05,
    "app_switch_count": 14.0,
    "domain_switch_count": 9.0,
}


def _features(*, focus: bool, marker: float) -> dict[str, float]:
    features = dict.fromkeys(V2_FEATURE_NAMES, 0.0)
    features.update(_FOCUS_FEATURES if focus else _DISTRACT_FEATURES)
    features["mouse_distance_per_min"] = marker
    return features


def _window(
    start: datetime, *, focus: bool, marker: float, window_id: str | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "window_start_utc": start.isoformat(),
        "window_end_utc": (start + timedelta(minutes=5)).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": _features(focus=focus, marker=marker),
    }
    if window_id is not None:
        row["id"] = window_id
    return row


def _feedback(session_id: str, start: datetime, end: datetime, *, focus: bool) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "label": "focus" if focus else "distracted",
        "score": 5 if focus else 1,
        "task_type": "coding",
    }


def _dataset(
    *,
    days: int = 8,
    auxiliary: bool = True,
    weak: bool = True,
    cross_midnight: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Explicit-feedback windows, plus (optionally) auxiliary and weak rows.

    Each day has two feedback sessions with opposite labels (3 windows each) so
    every date block carries both classes.  Auxiliary windows carry a
    user-calibrated ``window_label`` and weak windows are only reachable through
    the heuristic labeling functions: both must train the folds but never be
    scored.
    """
    base = datetime(2026, 9, 1, 9, tzinfo=UTC)
    windows: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    window_labels: dict[str, int] = {}
    for day in range(days):
        day_start = base + timedelta(days=day)
        for session in range(2):
            focus = (day + session) % 2 == 0
            start = day_start + timedelta(hours=3 * session)
            feedback.append(
                _feedback(f"s{day}-{session}", start, start + timedelta(minutes=15), focus=focus)
            )
            for offset in range(3):
                windows.append(
                    _window(
                        start + timedelta(minutes=5 * offset),
                        focus=focus,
                        marker=_EXPLICIT_MARKER,
                    )
                )
        if auxiliary:
            start = day_start + timedelta(hours=7)
            for offset, label in enumerate((1, 0, 1)):
                window_id = f"aux-{day}-{offset}"
                windows.append(
                    _window(
                        start + timedelta(minutes=5 * offset),
                        focus=bool(label),
                        marker=_AUXILIARY_MARKER,
                        window_id=window_id,
                    )
                )
                window_labels[window_id] = label
        if weak:
            start = day_start + timedelta(hours=8)
            for offset in range(2):
                windows.append(
                    _window(
                        start + timedelta(minutes=5 * offset),
                        focus=True,
                        marker=_WEAK_MARKER,
                        window_id=f"weak-{day}-{offset}",
                    )
                )
    if cross_midnight:
        # A session running over midnight labels windows on two dates: the
        # grouping contract must keep both dates in one fold.
        night_start = base + timedelta(hours=14, minutes=50)
        feedback.append(
            _feedback("night", night_start, night_start + timedelta(minutes=40), focus=True)
        )
        windows.append(_window(night_start, focus=True, marker=_EXPLICIT_MARKER))
        windows.append(
            _window(night_start + timedelta(minutes=15), focus=True, marker=_EXPLICIT_MARKER)
        )
    return windows, feedback, window_labels


@dataclass
class _Bundle:
    """One real evaluation run, with the pools and fits it produced recorded."""

    data: Any
    evaluation: dict[str, Any]
    pools: list[PredictionPool] = field(default_factory=list)
    fits: list[tuple[str, np.ndarray, np.ndarray]] = field(default_factory=list)


@pytest.fixture(scope="module")
def evaluation_bundle() -> _Bundle:
    """Run ``evaluate_v2_candidates`` once on a fixture with auxiliary labels.

    The candidate fits and the pooled report are intercepted so a test can see
    exactly which rows were trained on and which rows were scored.  Forests are
    shrunk for speed only — every code path (groups, weights, calibration,
    pooling) stays production's.
    """
    windows, feedback, window_labels = _dataset(cross_midnight=True)
    data = prepare_v2_training_data(windows, feedback, window_labels=window_labels)
    pools: list[PredictionPool] = []
    fits: list[tuple[str, np.ndarray, np.ndarray]] = []
    original_pooled_report = v2.pooled_report
    original_fit_candidate = v2.fit_candidate
    original_rf = EnsembleClassifier._RF_PARAMS
    original_xgb = EnsembleClassifier._XGB_PARAMS

    def capture_pool(pool: PredictionPool, **kwargs: Any) -> dict[str, Any]:
        pools.append(pool)
        return original_pooled_report(pool, **kwargs)

    def capture_fit(
        name: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray,
        x_test: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        fits.append((name, np.asarray(x_train).copy(), np.asarray(x_test).copy()))
        return original_fit_candidate(name, x_train, y_train, sample_weight, x_test, **kwargs)

    EnsembleClassifier._RF_PARAMS = {
        "n_estimators": 5, "max_depth": 3, "random_state": 42, "n_jobs": 1,
    }
    EnsembleClassifier._XGB_PARAMS = {
        "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
        "objective": "binary:logistic", "random_state": 42, "verbosity": 0,
    }
    v2.pooled_report = capture_pool
    v2.fit_candidate = capture_fit
    try:
        evaluation = evaluate_v2_candidates(data)
    finally:
        v2.pooled_report = original_pooled_report
        v2.fit_candidate = original_fit_candidate
        EnsembleClassifier._RF_PARAMS = original_rf
        EnsembleClassifier._XGB_PARAMS = original_xgb
    return _Bundle(data=data, evaluation=evaluation, pools=pools, fits=fits)


# ── 1. Explicit feedback is the only scored truth ────────────────────────


def test_evaluation_never_scores_auxiliary_or_weak_labels(evaluation_bundle: _Bundle) -> None:
    data = evaluation_bundle.data
    evaluation = evaluation_bundle.evaluation

    # Fixture sanity: the frame really does contain non-explicit supervision.
    assert data.window_label_count > 0
    assert "weak" in data.label_sources
    assert data.train_mask is not None
    assert int(data.train_mask.sum()) > int(data.explicit_mask.sum())

    assert evaluation["status"] == "evaluated"
    assert len(evaluation_bundle.pools) == 1, "the gate is decided on one pooled holdout"
    pool = evaluation_bundle.pools[0]
    assert len(pool) == len(pool.positions)

    # The scored rows are exactly the explicit rows of the scored date blocks:
    # the evaluation mask *is* the explicit-feedback mask.
    explicit_positions = np.flatnonzero(data.explicit_mask)
    scored_sources = {data.label_sources[index] for index in explicit_positions[pool.positions]}
    assert scored_sources == {"explicit"}
    scored_date_groups = {
        group
        for fold in evaluation["forward_chaining"]["folds"]
        for group in fold["test_groups"]
    }
    expected_scored = {
        int(index) for index in explicit_positions if data.group_ids[index] in scored_date_groups
    }
    assert set(explicit_positions[pool.positions].tolist()) == expected_scored
    assert evaluation["probability_metrics"]["sample_count"] == len(expected_scored)
    assert evaluation["explicit_sample_count"] == int(data.explicit_mask.sum())

    # Every fold's declared test size must equal the explicit rows of its test
    # blocks: no auxiliary row may ever be scored.
    explicit_per_group: dict[str, int] = {}
    for index in explicit_positions:
        group = data.group_ids[index]
        explicit_per_group[group] = explicit_per_group.get(group, 0) + 1
    for fold in evaluation["folds"] + evaluation["forward_chaining"]["folds"]:
        assert fold["test_size"] == sum(
            explicit_per_group.get(group, 0) for group in fold["test_groups"]
        )


def test_auxiliary_labels_train_the_folds_but_are_never_scored(
    evaluation_bundle: _Bundle,
) -> None:
    data = evaluation_bundle.data
    marker = data.feature_names.index("mouse_distance_per_min")
    assert evaluation_bundle.fits, "the fixture must actually fit candidates"

    aux_seen_in_training = False
    for name, x_train, x_test in evaluation_bundle.fits:
        assert set(np.unique(x_test[:, marker]).tolist()) == {_EXPLICIT_MARKER}
        if name == RULE_ENGINE:
            continue  # the heuristic baseline ignores its training rows
        if _AUXILIARY_MARKER in set(np.unique(x_train[:, marker]).tolist()):
            aux_seen_in_training = True
    assert aux_seen_in_training, "auxiliary rows must widen supervision"


# ── 2./3. Forward chaining: no future leakage, documented arity rules ────


def _fold_groups(
    sizes: list[int], *, start_day: int = 1
) -> tuple[list[str], dict[str, list[str]]]:
    group_ids = [f"g{index}" for index, size in enumerate(sizes) for _ in range(size)]
    dates = {f"g{index}": [f"2026-09-{start_day + index:02d}"] for index in range(len(sizes))}
    return group_ids, dates


def test_forward_chaining_needs_three_date_blocks() -> None:
    for sizes in ([], [10], [10, 10]):
        group_ids, dates = _fold_groups(sizes)
        assert build_forward_chaining_folds(group_ids, group_dates=dates) == []

    group_ids, dates = _fold_groups([10, 10, 10])
    folds = build_forward_chaining_folds(group_ids, group_dates=dates)
    assert [fold.index for fold in folds] == [1, 2]
    # Chunk 0 is training-only history: the first scored block never trains on
    # itself, and the last fold trains on every earlier block.
    assert folds[0].train_groups == ["g0"]
    assert folds[0].test_groups == ["g1"]
    assert folds[1].train_groups == ["g0", "g1"]
    assert folds[1].test_groups == ["g2"]


def test_forward_chaining_never_trains_on_a_future_block() -> None:
    group_ids, dates = _fold_groups([12, 6, 9, 15, 7])
    folds = build_forward_chaining_folds(group_ids, group_dates=dates)
    order = sorted(dates, key=lambda group: dates[group][0])

    assert len(folds) >= 2
    for fold in folds:
        assert fold.scheme == "forward_chaining"
        assert set(fold.train_groups).isdisjoint(fold.test_groups)
        assert set(fold.train_idx).isdisjoint(fold.test_idx)
        assert fold.train_dates and fold.test_dates
        assert fold.cutoff_date == max(fold.train_dates)
        assert fold.cutoff_date is not None
        # Every scored block is strictly later than everything trained on.
        assert fold.cutoff_date < min(fold.test_dates)
        # Training is a chronological prefix; the scored blocks are exactly the
        # blocks that follow it.
        assert fold.train_groups == order[: len(fold.train_groups)]
        following = order[
            len(fold.train_groups) : len(fold.train_groups) + len(fold.test_groups)
        ]
        assert fold.test_groups == following
        assert len(fold.test_idx) == sum(group_ids.count(group) for group in fold.test_groups)
    # Expanding window: each fold inherits every earlier fold's training set.
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert set(earlier.train_groups) < set(later.train_groups)


def test_forward_chaining_respects_min_test_samples() -> None:
    group_ids, dates = _fold_groups([6, 6, 6, 2])

    default_folds = build_forward_chaining_folds(group_ids, group_dates=dates)
    assert len(default_folds) == 1, "the split with a 2-row scored block is rejected"
    assert len(default_folds[0].test_idx) == 8
    assert len(default_folds[0].test_idx) >= MIN_FOLD_TEST_SAMPLES

    permissive = build_forward_chaining_folds(group_ids, group_dates=dates, min_test_samples=1)
    assert [len(fold.test_idx) for fold in permissive] == [6, 6, 2]
    for fold in permissive:
        assert len(fold.test_idx) >= 1
    for fold in default_folds:
        assert len(fold.test_idx) >= MIN_FOLD_TEST_SAMPLES


def test_forward_chaining_reports_the_widest_split_when_none_is_large_enough() -> None:
    """No split satisfies the minimum -> the instability is visible, not hidden."""
    group_ids, dates = _fold_groups([10, 10, 1, 1])
    folds = build_forward_chaining_folds(group_ids, group_dates=dates)

    assert [len(fold.test_idx) for fold in folds] == [10, 1, 1]
    stability = fold_stability(
        [0.9] * len(folds), [len(fold.test_idx) for fold in folds], scheme="forward_chaining"
    )
    assert stability["passed"] is False
    assert stability["min_test_size"] == 1
    assert stability["thresholds"]["min_test_size"] == MIN_FOLD_TEST_SAMPLES
    assert stability["scheme"] == "forward_chaining"


def test_forward_chaining_folds_keep_sessions_and_future_dates_out_of_training(
    evaluation_bundle: _Bundle,
) -> None:
    data = evaluation_bundle.data
    evaluation = evaluation_bundle.evaluation
    folds = evaluation["forward_chaining"]["folds"]
    assert evaluation["primary_scheme"] == "forward_chaining"
    assert len(folds) >= 3

    sessions_by_group: dict[str, set[str]] = {}
    groups_by_session: dict[str, set[str]] = {}
    for group, session in zip(data.group_ids, data.sample_feedback_ids, strict=True):
        if not session:
            continue
        sessions_by_group.setdefault(group, set()).add(session)
        groups_by_session.setdefault(session, set()).add(group)
    # The cross-midnight session really does cover two dates in one block.
    assert len(groups_by_session["night"]) == 1

    for fold in folds:
        assert set(fold["train_groups"]).isdisjoint(fold["test_groups"])
        assert fold["cutoff_date"] < min(fold["test_dates"])
        assert max(fold["train_dates"]) == fold["cutoff_date"]
        train_sessions = set().union(
            *(sessions_by_group.get(group, set()) for group in fold["train_groups"])
        )
        test_sessions = set().union(
            *(sessions_by_group.get(group, set()) for group in fold["test_groups"])
        )
        assert train_sessions.isdisjoint(test_sessions), (
            "a feedback session must never sit on both sides of a fold"
        )
        assert test_sessions, "every scored block carries explicit feedback"
    # The same guarantee holds for the legacy date-grouped scheme.
    for fold in evaluation["folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["test_groups"])
        assert set(fold["train_dates"]).isdisjoint(fold["test_dates"])


# ── 4. Session- and date-level metrics ───────────────────────────────────


def _hand_built_pool() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    y_true = np.asarray([1, 1, 1, 0, 0, 1, 0, 0, 0], dtype=np.int_)
    y_pred = np.asarray([1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int_)
    y_proba = np.asarray([0.9, 0.8, 0.2, 0.1, 0.6, 0.4, 0.2, 0.3, 0.1], dtype=np.float64)
    sessions = ["s1", "s1", "s1", "s2", "s2", "s3", "s4", "s4", "s4"]
    groups = ["d1", "d1", "d1", "d2", "d2", "d3", "d3", "d3", "d3"]
    return y_true, y_pred, y_proba, sessions, groups


def test_session_metrics_aggregate_window_predictions() -> None:
    y_true, y_pred, y_proba, sessions, _groups = _hand_built_pool()
    metrics = session_level_metrics(y_true, y_pred, y_proba, sessions)

    assert metrics["status"] == "evaluated"
    assert metrics["unit"] == "feedback_session"
    assert metrics["aggregation"] == "majority_vote_per_unit"
    assert metrics["unit_count"] == len(set(sessions)) == 4
    assert metrics["window_count"] == len(y_true) == 9
    assert sum(unit["windows"] for unit in metrics["per_unit"]) == metrics["window_count"]

    by_unit = {unit["unit"]: unit for unit in metrics["per_unit"]}
    # s1: votes [1, 1, 0] -> focus; label 1 -> correct.
    assert by_unit["s1"]["prediction"] == 1
    assert by_unit["s1"]["label"] == 1
    assert by_unit["s1"]["mean_probability"] == pytest.approx((0.9 + 0.8 + 0.2) / 3)
    assert by_unit["s1"]["correct"] is True
    # s2: one vote each way -> the mean probability breaks the tie (0.35 -> 0).
    assert by_unit["s2"]["prediction"] == 0
    assert by_unit["s2"]["correct"] is True
    # s3: its single window is predicted wrong.
    assert by_unit["s3"]["prediction"] == 0
    assert by_unit["s3"]["label"] == 1
    assert by_unit["s3"]["correct"] is False
    # Unit-level truth/prediction pairs: (1,1), (0,0), (1,0), (0,0).
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)


def test_date_metrics_aggregate_the_same_rows_into_date_blocks() -> None:
    y_true, y_pred, y_proba, _sessions, groups = _hand_built_pool()
    metrics = date_level_metrics(y_true, y_pred, y_proba, groups)

    assert metrics["status"] == "evaluated"
    assert metrics["unit"] == "date_block"
    assert metrics["unit_count"] == 3
    assert metrics["window_count"] == len(y_true)
    assert sum(unit["windows"] for unit in metrics["per_unit"]) == len(y_true)
    by_unit = {unit["unit"]: unit for unit in metrics["per_unit"]}
    assert by_unit["d1"]["label"] == 1 and by_unit["d1"]["prediction"] == 1
    assert by_unit["d1"]["mean_probability"] == pytest.approx((0.9 + 0.8 + 0.2) / 3)
    # d3 holds [1, 0, 0, 0] -> majority label 0, every window predicted 0.
    assert by_unit["d3"]["windows"] == 4
    assert by_unit["d3"]["label"] == 0 and by_unit["d3"]["prediction"] == 0
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_session_and_date_metrics_are_not_available_without_ids() -> None:
    y_true, y_pred, y_proba, sessions, _groups = _hand_built_pool()

    assert session_level_metrics(y_true, y_pred, y_proba, [])["status"] == "not_available"
    without_ids = list(sessions)
    without_ids[1] = ""
    unavailable = session_level_metrics(y_true, y_pred, y_proba, without_ids)
    assert unavailable["status"] == "not_available"
    assert "session" in unavailable["reason"]
    assert date_level_metrics(y_true, y_pred, y_proba, [])["status"] == "not_available"


# ── 5. PR-AUC / Brier / ECE / reliability table ──────────────────────────


def test_perfect_predictions_are_perfectly_scored() -> None:
    y_true = np.asarray([0, 0, 1, 1], dtype=np.int_)
    y_proba = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    metrics = probability_metrics(y_true, y_proba)

    assert metrics["brier_score"] == 0.0
    assert metrics["expected_calibration_error"] == 0.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["sample_count"] == 4
    assert sum(row["count"] for row in metrics["reliability_table"]) == 4
    assert all(row["gap"] == 0.0 for row in metrics["reliability_table"])


def test_inverted_predictor_scores_worse_than_the_base_rate() -> None:
    y_true = np.asarray([0, 0, 1, 1] * 5, dtype=np.int_)
    base_rate = float(y_true.mean())
    inverted = np.where(y_true == 1, 0.0, 1.0).astype(np.float64)

    correct = probability_metrics(y_true, y_true.astype(np.float64))
    inverted_metrics = probability_metrics(y_true, inverted)
    base_metrics = probability_metrics(y_true, np.full(len(y_true), base_rate))

    assert inverted_metrics["brier_score"] == pytest.approx(1.0)
    assert base_metrics["brier_score"] == pytest.approx(0.25)
    assert inverted_metrics["brier_score"] > base_metrics["brier_score"]
    assert base_metrics["brier_score"] > correct["brier_score"] == 0.0
    assert correct["pr_auc"] == 1.0
    assert inverted_metrics["pr_auc"] <= base_metrics["pr_auc"] == pytest.approx(base_rate)
    assert inverted_metrics["expected_calibration_error"] > 0.5


def test_expected_calibration_error_measures_the_calibration_gap() -> None:
    # Every sample is negative but the model claims 0.9: ECE is the whole gap.
    confident = expected_calibration_error(
        np.zeros(10, dtype=np.int_), np.full(10, 0.9, dtype=np.float64)
    )
    assert confident == pytest.approx(0.9)

    # A calibrated constant matches the observed rate exactly.
    half = np.asarray([0, 1] * 5, dtype=np.int_)
    assert expected_calibration_error(
        half, np.full(len(half), 0.5, dtype=np.float64)
    ) == pytest.approx(0.0)
    assert expected_calibration_error(np.asarray([], dtype=np.int_), np.asarray([])) == 0.0


def test_reliability_table_assigns_every_sample_to_exactly_one_bin() -> None:
    """Edge values belong to one bin only (the legacy table double-counted them)."""
    y_true = np.asarray([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int_)
    y_proba = np.linspace(0.0, 1.0, 11)
    table = reliability_table(y_true, y_proba)

    assert sum(row["count"] for row in table) == len(y_proba) == 11
    assert [row["bin_low"] for row in table] == sorted(row["bin_low"] for row in table)
    for row in table:
        assert row["bin_low"] < row["bin_high"]
        assert row["weight"] == pytest.approx(row["count"] / len(y_proba), abs=1e-6)
        assert row["gap"] == pytest.approx(
            row["fraction_positive"] - row["mean_prediction"], abs=1e-6
        )
        assert row["ece_contribution"] == pytest.approx(
            row["weight"] * abs(row["gap"]), abs=1e-6
        )
        assert 0.0 <= row["mean_prediction"] <= 1.0
        assert 0.0 <= row["fraction_positive"] <= 1.0
    assert sum(row["weight"] for row in table) == pytest.approx(1.0, abs=1e-6)
    assert sum(row["ece_contribution"] for row in table) == pytest.approx(
        expected_calibration_error(y_true, y_proba), abs=1e-6
    )


def test_probability_metrics_without_two_classes_report_no_ranking_metric() -> None:
    y_true = np.asarray([1, 1, 1, 1], dtype=np.int_)
    metrics = probability_metrics(y_true, np.asarray([0.6, 0.7, 0.8, 0.9]))

    assert metrics["pr_auc"] is None
    assert metrics["roc_auc"] is None
    assert metrics["brier_score"] is not None
    assert metrics["expected_calibration_error"] is not None


# ── 6. Session bootstrap ─────────────────────────────────────────────────


def _bootstrap_pool() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """20 sessions x 3 windows; the last six sessions are predicted wrong."""
    sessions = [f"s{index}" for index in range(20) for _ in range(3)]
    labels = [index % 2 for index in range(20) for _ in range(3)]
    predictions = [
        (index % 2) if index < 14 else (1 - index % 2) for index in range(20) for _ in range(3)
    ]
    probabilities = [0.9 if prediction else 0.1 for prediction in predictions]
    return (
        np.asarray(labels, dtype=np.int_),
        np.asarray(predictions, dtype=np.int_),
        np.asarray(probabilities, dtype=np.float64),
        sessions,
    )


def test_session_bootstrap_draws_whole_sessions() -> None:
    sessions = [f"s{index}" for index in range(6) for _ in range(4)]
    draws = session_bootstrap_draws(sessions, resamples=25, seed=BOOTSTRAP_SEED)

    assert len(draws) == 25
    positions: dict[str, list[int]] = {}
    for position, session in enumerate(sessions):
        positions.setdefault(session, []).append(position)
    for drawn, rows in draws:
        assert len(drawn) == len(set(sessions)) == 6
        expected_rows = [row for session in drawn for row in positions[session]]
        assert sorted(rows.tolist()) == sorted(expected_rows)
        # A session is drawn whole: all of its windows move together.
        assert set(rows.tolist()) == {row for session in drawn for row in positions[session]}
    assert session_bootstrap_draws([], resamples=5) == []
    assert session_bootstrap_draws(sessions, resamples=0) == []


def test_bootstrap_intervals_are_deterministic_under_a_fixed_seed() -> None:
    y_true, y_pred, y_proba, sessions = _bootstrap_pool()
    first = bootstrap_confidence_intervals(
        y_true, y_pred, y_proba, sessions, resamples=200, seed=BOOTSTRAP_SEED
    )
    again = bootstrap_confidence_intervals(
        y_true, y_pred, y_proba, sessions, resamples=200, seed=BOOTSTRAP_SEED
    )
    other_seed = bootstrap_confidence_intervals(
        y_true, y_pred, y_proba, sessions, resamples=200, seed=BOOTSTRAP_SEED + 1
    )

    assert first == again
    assert first["unit"] == BOOTSTRAP_UNIT == "explicit_feedback_session"
    assert first["seed"] == BOOTSTRAP_SEED
    assert first["resamples"] == 200
    assert first["session_count"] == 20
    assert first["row_count"] == len(y_true)
    assert first["metrics"]["balanced_accuracy"]["valid_resamples"] + (
        first["one_class_resamples_skipped"]
    ) == 200
    assert first["metrics"] != other_seed["metrics"]

    draws_a = session_bootstrap_draws(sessions, resamples=5, seed=BOOTSTRAP_SEED)
    draws_b = session_bootstrap_draws(sessions, resamples=5, seed=BOOTSTRAP_SEED + 1)
    assert [drawn for drawn, _rows in draws_a] != [drawn for drawn, _rows in draws_b]


def test_bootstrap_intervals_bracket_the_point_estimates() -> None:
    y_true, y_pred, y_proba, sessions = _bootstrap_pool()
    report = bootstrap_confidence_intervals(
        y_true, y_pred, y_proba, sessions, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
    )

    assert report["status"] == "evaluated"
    for name, metric in report["metrics"].items():
        assert metric["point"] is not None, name
        assert metric["lower"] is not None and metric["upper"] is not None, name
        assert metric["lower"] <= metric["point"] <= metric["upper"], name
        assert metric["lower"] <= metric["upper"]
        assert metric["std"] is not None and metric["std"] >= 0.0
        assert metric["valid_resamples"] > 0
    # 14 of 20 sessions are predicted correctly, as a block.
    assert report["metrics"]["accuracy"]["point"] == pytest.approx(0.7)
    assert report["metrics"]["balanced_accuracy"]["point"] == pytest.approx(0.7)


def test_bootstrap_is_not_available_without_sessions() -> None:
    y_true, y_pred, y_proba, _sessions = _bootstrap_pool()
    missing = bootstrap_confidence_intervals(y_true, y_pred, y_proba, [])
    assert missing["status"] == "not_available"
    assert missing["unit"] == BOOTSTRAP_UNIT

    empty = bootstrap_confidence_intervals(
        y_true, y_pred, y_proba, [f"s{i}" for i in range(len(y_true))], resamples=0
    )
    assert empty["status"] == "not_available"


# ── 7. Abstention / coverage ─────────────────────────────────────────────


def test_abstention_coverage_shrinks_as_the_band_widens() -> None:
    y_proba = np.linspace(0.0, 1.0, 21)
    y_true = np.asarray([1 if value >= 0.5 else 0 for value in y_proba], dtype=np.int_)
    report = abstention_metrics(y_true, y_true.copy(), y_proba)

    assert report["status"] == "evaluated"
    assert report["policy"] == "abstain_inside_confidence_band"
    assert report["band"]["half_width"] == pytest.approx(ABSTENTION_BAND_HALF_WIDTH)
    assert report["band"]["low"] == pytest.approx(0.35)
    assert report["band"]["high"] == pytest.approx(0.65)
    assert report["sample_count"] == len(y_true)

    bands = report["by_band_width"]
    assert [row["half_width"] for row in bands] == sorted(ABSTENTION_BAND_WIDTHS)
    coverages = [row["coverage"] for row in bands]
    assert coverages == sorted(coverages, reverse=True)
    assert len(set(coverages)) == len(coverages), "each widening must abstain on more samples"
    for row in bands:
        assert row["covered_count"] + row["abstained_count"] == len(y_true)
        assert row["coverage"] == pytest.approx(row["covered_count"] / len(y_true))
        assert row["abstention_rate"] == pytest.approx(1.0 - row["coverage"])
        assert row["band_low"] == pytest.approx(0.5 - row["half_width"], abs=1e-6)
        assert row["band_high"] == pytest.approx(0.5 + row["half_width"], abs=1e-6)

    # The headline numbers describe the default band, scored on the exact
    # covered subset: a high coverage number can never hide a bad subset.
    selected = next(row for row in bands if row["half_width"] == ABSTENTION_BAND_HALF_WIDTH)
    assert report["coverage"] == selected["coverage"]
    assert report["covered_count"] == selected["covered_count"]
    assert report["abstained_count"] == selected["abstained_count"]
    covered = np.abs(y_proba - 0.5) > ABSTENTION_BAND_HALF_WIDTH
    assert report["covered_accuracy"] == pytest.approx(
        float((y_true[covered] == y_true[covered]).mean())
    )
    assert report["covered_accuracy"] == selected["covered_accuracy"]
    assert report["abstained_accuracy"] == selected["abstained_accuracy"]
    assert report["covered_balanced_accuracy"] == selected["covered_balanced_accuracy"]
    assert report["covered_minority_f1"] == selected["covered_minority_f1"]


def test_abstention_scores_the_covered_subset_separately_from_the_headline() -> None:
    """Coverage is high while the covered subset is still not perfect."""
    y_true = np.asarray([1, 1, 1, 0, 0, 0, 1, 0, 1, 0], dtype=np.int_)
    y_pred = np.asarray([1, 1, 0, 0, 0, 1, 1, 0, 1, 0], dtype=np.int_)
    y_proba = np.asarray([0.9, 0.8, 0.85, 0.1, 0.05, 0.15, 0.5, 0.45, 0.55, 0.5])
    report = abstention_metrics(y_true, y_pred, y_proba)

    covered = np.abs(y_proba - 0.5) > ABSTENTION_BAND_HALF_WIDTH
    assert report["covered_count"] == int(covered.sum())
    assert report["coverage"] == pytest.approx(float(covered.mean()))
    assert report["covered_accuracy"] == pytest.approx(
        float((y_pred[covered] == y_true[covered]).mean())
    )
    assert report["abstained_accuracy"] == pytest.approx(
        float((y_pred[~covered] == y_true[~covered]).mean())
    )
    # Confident-but-wrong windows make the covered subset less than perfect,
    # which is exactly what the separate covered metrics exist to expose.
    assert report["covered_accuracy"] < 1.0
    assert report["covered_balanced_accuracy"] <= report["covered_accuracy"] + 0.5
    assert report["abstained_accuracy"] == 1.0

    # A custom band is reported next to the configured ones.
    custom = abstention_metrics(y_true, y_pred, y_proba, band_half_width=0.2)
    widths = [row["half_width"] for row in custom["by_band_width"]]
    assert custom["band"]["half_width"] == pytest.approx(0.2)
    assert len(widths) == len(ABSTENTION_BAND_WIDTHS) + 1
    assert widths == sorted(widths)
    empty = abstention_metrics(np.asarray([], dtype=np.int_), np.asarray([], dtype=np.int_),
                               np.asarray([]))
    assert empty["status"] == "not_available"


# ── 8. Labeling functions ────────────────────────────────────────────────

_FOCUS_ROW: dict[str, float] = {
    "top_app_ratio": 0.95,
    "idle_ratio": 0.02,
    "input_active_ratio": 0.5,
    "app_switch_count": 1.0,
}
_DISTRACT_ROW: dict[str, float] = {
    "top_app_ratio": 0.2,
    "idle_ratio": 0.3,
    "input_active_ratio": 0.05,
    "app_switch_count": 12.0,
}
_CONFLICT_ROW: dict[str, float] = {
    "top_app_ratio": 0.95,       # long single-app dwell says FOCUS
    "idle_ratio": 0.02,
    "input_active_ratio": 0.18,  # ...while the switch storm says DISTRACT
    "app_switch_count": 12.0,
}
_IDLE_ROW: dict[str, float] = {
    "top_app_ratio": 0.95,
    "idle_ratio": 0.95,
    "input_active_ratio": 0.5,
    "app_switch_count": 1.0,
}
_ENTERTAINMENT_ROW: dict[str, float] = {
    "top_app_ratio": 0.2,
    "idle_ratio": 0.05,
    "input_active_ratio": 0.6,
    "audible_browser_ratio": 0.5,
    "app_switch_count": 0.0,
}


def _entry(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["functions"] if item["name"] == name)


def test_labeling_function_report_reports_coverage_conflict_and_agreement() -> None:
    rows = [_FOCUS_ROW, _FOCUS_ROW, _DISTRACT_ROW, _DISTRACT_ROW, _IDLE_ROW, _ENTERTAINMENT_ROW]
    explicit: list[int | None] = [1, 1, 0, 0, None, 0]
    report = labeling.build_labeling_function_report(rows, explicit, universe="unit_test")

    assert report["status"] == "reported"
    assert report["rows_considered"] == 6
    assert report["ground_truth_rows"] == 5
    assert report["universe"] == "unit_test"
    assert [item["name"] for item in report["functions"]] == [
        spec.name for spec in labeling.LABELING_FUNCTIONS
    ]
    for item in report["functions"]:
        assert item["fires"] + item["abstains"] == 6
        assert item["coverage"] == pytest.approx(item["fires"] / 6, abs=1e-6)
        assert item["conflict_rate"] == pytest.approx(
            item["conflict_rows"] / item["fires"] if item["fires"] else 0.0, abs=1e-6
        )
        if item["comparable_rows"]:
            assert item["agreement_rate"] == pytest.approx(
                item["agreement_rows"] / item["comparable_rows"], abs=1e-6
            )
        else:
            assert item["agreement_rate"] is None

    # The ground truth agrees with itself wherever it exists, by construction.
    truth = _entry(report, labeling.GROUND_TRUTH_NAME)
    assert truth["kind"] == "ground_truth"
    assert truth["fires"] == 5
    assert truth["agreement_rate"] == 1.0
    assert truth["in_weak_composition"] is False

    # The deep-work function fires on exactly the two focus rows, and agrees.
    dwell = _entry(report, "long_single_app_dwell")
    assert dwell["fires"] == 2
    assert dwell["votes"] == {"focus": 2, "distracted": 0}
    assert dwell["coverage"] == pytest.approx(2 / 6)
    assert dwell["agreement_rate"] == 1.0
    assert dwell["conflict_rows"] == 0
    assert dwell["in_weak_composition"] is True

    # ...as does the switch-storm function on the two distracted rows.
    switch = _entry(report, "high_switch_frequency")
    assert switch["fires"] == 2
    assert switch["votes"] == {"focus": 0, "distracted": 2}
    assert switch["agreement_rate"] == 1.0

    # The idle guard is a veto, not a label: it never votes and is never scored.
    guard = _entry(report, labeling.GUARD_NAME)
    assert guard["kind"] == "guard"
    assert guard["fires"] == 1
    assert guard["votes"] == {"focus": 0, "distracted": 0}
    assert guard["agreement_rate"] is None

    # The entertainment function is measured but advisory.
    entertainment = _entry(report, "input_active_entertainment_share")
    assert entertainment["advisory"] is True
    assert entertainment["in_weak_composition"] is False
    assert entertainment["fires"] == 1
    assert entertainment["votes"] == {"focus": 0, "distracted": 1}
    assert entertainment["agreement_rate"] == 1.0


def test_labeling_functions_report_conflicts_between_each_other() -> None:
    report = labeling.build_labeling_function_report([_CONFLICT_ROW, _FOCUS_ROW], [1, 1])

    assert report["conflicting_rows"] >= 1
    dwell = _entry(report, "long_single_app_dwell")
    switch = _entry(report, "high_switch_frequency")
    assert dwell["conflict_rows"] == 1
    assert dwell["conflicts_with"]["high_switch_frequency"] == 1
    assert switch["conflict_rows"] == 1
    assert switch["conflicts_with"]["long_single_app_dwell"] == 1
    assert report["conflict_matrix"]["long_single_app_dwell"]["high_switch_frequency"] == 1

    # The composition resolves the conflict in the documented order.
    outputs = labeling.compute_labeling_function_outputs(_CONFLICT_ROW)
    assert outputs["long_single_app_dwell"] == labeling.FOCUS
    assert outputs["high_switch_frequency"] == labeling.DISTRACT
    assert labeling.weak_label_from_functions(outputs) == labeling.FOCUS


def test_labeling_functions_never_override_explicit_feedback() -> None:
    """A user's verdict wins even when the heuristics disagree with it."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    # The user says "focus" for a window whose features look distracted.
    window = _window(start, focus=False, marker=_EXPLICIT_MARKER, window_id="w1")
    feedback = [_feedback("s1", start, start + timedelta(minutes=15), focus=True)]

    outputs = labeling.compute_labeling_function_outputs(
        window["features"], {"explicit_label": 1}
    )
    assert outputs["high_switch_frequency"] == labeling.DISTRACT
    assert outputs[labeling.GROUND_TRUTH_NAME] == labeling.FOCUS
    # The heuristic really does disagree with the user...
    assert labeling.weak_label(window["features"]) == labeling.DISTRACT

    # ...and the training frame still keeps the user's label for that window,
    # both against the weak heuristic and against a contradicting window label.
    data = prepare_v2_training_data([window], feedback, window_labels={"w1": 0})
    assert data.labels.tolist() == [1]
    assert data.label_sources == ["explicit"]
    assert data.explicit_mask.tolist() == [True]
    assert data.window_label_count == 0

    # The report shows the heuristic losing, rather than the label changing.
    switch = _entry(data.labeling_function_report, "high_switch_frequency")
    assert switch["fires"] == 1
    assert switch["comparable_rows"] == 1
    assert switch["agreement_rows"] == 0
    assert switch["agreement_rate"] == 0.0


def test_advisory_labeling_functions_stay_out_of_the_composition() -> None:
    report = labeling.build_labeling_function_report([_ENTERTAINMENT_ROW], [0])
    composition = report["composition"]

    assert composition["order"] == list(labeling.COMPOSITION_ORDER)
    assert composition["guard"] == labeling.GUARD_NAME
    assert composition["ground_truth"] == labeling.GROUND_TRUTH_NAME
    assert "input_active_entertainment_share" in composition["advisory"]
    assert "input_active_entertainment_share" not in composition["composed"]

    outputs = labeling.compute_labeling_function_outputs(_ENTERTAINMENT_ROW)
    assert outputs["input_active_entertainment_share"] == labeling.DISTRACT
    # Nothing in the composition votes, so the row stays unlabelled.
    assert labeling.weak_label_from_functions(outputs) == labeling.ABSTAIN
    assert labeling.weak_label(_ENTERTAINMENT_ROW) == labeling.ABSTAIN

    # Only a *state* answer after an intervention is a label; helpfulness is not.
    only_helpfulness = {labeling.POST_INTERVENTION_HELPFULNESS_KEY: "helpful"}
    assert labeling.lf_post_intervention_response({}, only_helpfulness) == labeling.ABSTAIN
    assert labeling.lf_post_intervention_response(
        {}, {labeling.POST_INTERVENTION_STATE_KEY: "focus"}
    ) == labeling.FOCUS
    assert labeling.lf_post_intervention_response(
        {}, {labeling.POST_INTERVENTION_STATE_KEY: "mixed"}
    ) == labeling.ABSTAIN


def test_idle_guard_vetoes_a_window_other_functions_would_label() -> None:
    outputs = labeling.compute_labeling_function_outputs(_IDLE_ROW)

    assert outputs[labeling.GUARD_NAME] == 1
    assert outputs["sustained_attention"] == labeling.FOCUS
    assert labeling.weak_label_from_functions(outputs) == labeling.ABSTAIN
    assert labeling.weak_label(_IDLE_ROW) == labeling.ABSTAIN


# ── 9. Candidate set and the conservative selection rule ─────────────────


def test_candidate_ladder_covers_the_five_named_candidates() -> None:
    assert CANDIDATE_NAMES == (
        RULE_ENGINE,
        LOGISTIC_REGRESSION,
        RANDOM_FOREST,
        XGBOOST,
        RF_XGB_SOFT_VOTING,
    )
    assert len(CANDIDATE_NAMES) == 5
    assert set(DEPLOYABLE_ORDER) < set(CANDIDATE_NAMES)
    assert RULE_ENGINE not in DEPLOYABLE_ORDER, "the heuristic fallback is never published"
    assert DEFAULT_PUBLICATION_MODEL == RF_XGB_SOFT_VOTING


def test_selection_keeps_the_simplest_model_within_the_promotion_margin() -> None:
    """The ensemble ties-to-slightly-wins: the simple model is kept."""
    aggregate = {
        LOGISTIC_REGRESSION: 0.70,
        RANDOM_FOREST: 0.705,
        XGBOOST: 0.706,
        RF_XGB_SOFT_VOTING: 0.70 + PROMOTION_MARGIN - 0.001,
    }
    per_fold = {
        LOGISTIC_REGRESSION: {1: 0.70, 2: 0.70, 3: 0.70, 4: 0.70},
        RANDOM_FOREST: {1: 0.71, 2: 0.70, 3: 0.70, 4: 0.71},
        XGBOOST: {1: 0.71, 2: 0.71, 3: 0.70, 4: 0.70},
        RF_XGB_SOFT_VOTING: {1: 0.72, 2: 0.72, 3: 0.72, 4: 0.72},
    }
    selection = select_candidate(aggregate, per_fold)

    assert selection["status"] == "selected"
    assert selection["selected"] == LOGISTIC_REGRESSION
    assert selection["rule"] == "keep_simplest_unless_stably_beaten"
    assert selection["ladder"] == list(DEPLOYABLE_ORDER)
    assert selection["candidate_count"] == len(CANDIDATE_NAMES) == 5
    assert set(selection["complexity"]) == set(CANDIDATE_NAMES)
    assert selection["publication_model"] == RF_XGB_SOFT_VOTING
    assert selection["publication_consistent"] is False
    ensemble_decision = next(
        decision
        for decision in selection["decisions"]
        if decision["candidate"] == RF_XGB_SOFT_VOTING
    )
    assert ensemble_decision["promoted"] is False
    assert ensemble_decision["aggregate_gain"] < PROMOTION_MARGIN
    assert ensemble_decision["win_ratio"] == 1.0, "the margin alone keeps the simple model"
    assert "promotion margin" in ensemble_decision["reason"]
    assert selection["reason"].startswith(LOGISTIC_REGRESSION)


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("margin", "promotion margin"),
        ("fold_win_ratio", "shared folds"),
        ("max_fold_loss", "loses one fold"),
        ("secondary_scheme", "secondary fold scheme"),
    ],
)
def test_a_complex_candidate_must_be_stably_better_to_be_promoted(
    case: str, expected_reason: str
) -> None:
    aggregate = {
        LOGISTIC_REGRESSION: 0.70,
        RANDOM_FOREST: 0.705,
        XGBOOST: 0.706,
        RF_XGB_SOFT_VOTING: 0.80,
    }
    per_fold = {
        LOGISTIC_REGRESSION: {1: 0.70, 2: 0.70, 3: 0.70, 4: 0.70},
        RANDOM_FOREST: {1: 0.71, 2: 0.70, 3: 0.70, 4: 0.71},
        XGBOOST: {1: 0.71, 2: 0.71, 3: 0.70, 4: 0.70},
        RF_XGB_SOFT_VOTING: {1: 0.82, 2: 0.82, 3: 0.82, 4: 0.82},
    }
    secondary: dict[str, float] | None = None
    if case == "margin":
        aggregate[RF_XGB_SOFT_VOTING] = 0.70 + PROMOTION_MARGIN / 2
    elif case == "fold_win_ratio":
        per_fold[RF_XGB_SOFT_VOTING] = {1: 0.82, 2: 0.60, 3: 0.60, 4: 0.60}
    elif case == "max_fold_loss":
        per_fold[RF_XGB_SOFT_VOTING] = {1: 0.82, 2: 0.82, 3: 0.82, 4: 0.50}
    else:
        secondary = {
            LOGISTIC_REGRESSION: 0.70,
            RANDOM_FOREST: 0.70,
            XGBOOST: 0.70,
            RF_XGB_SOFT_VOTING: 0.60,
        }

    selection = select_candidate(aggregate, per_fold, secondary_aggregate=secondary)

    assert selection["selected"] == LOGISTIC_REGRESSION
    ensemble_decision = next(
        decision
        for decision in selection["decisions"]
        if decision["candidate"] == RF_XGB_SOFT_VOTING
    )
    assert ensemble_decision["promoted"] is False
    assert expected_reason in ensemble_decision["reason"]


def test_a_stably_better_ensemble_is_promoted() -> None:
    aggregate = {
        LOGISTIC_REGRESSION: 0.70,
        RANDOM_FOREST: 0.71,
        XGBOOST: 0.705,
        RF_XGB_SOFT_VOTING: 0.80,
    }
    per_fold = {
        LOGISTIC_REGRESSION: {1: 0.70, 2: 0.70, 3: 0.70, 4: 0.70},
        RANDOM_FOREST: {1: 0.71, 2: 0.71, 3: 0.70, 4: 0.70},
        XGBOOST: {1: 0.70, 2: 0.71, 3: 0.70, 4: 0.71},
        RF_XGB_SOFT_VOTING: {1: 0.85, 2: 0.85, 3: 0.80, 4: 0.70},
    }
    selection = select_candidate(aggregate, per_fold)

    assert selection["selected"] == RF_XGB_SOFT_VOTING
    assert selection["publication_consistent"] is True
    assert "won the ladder" in selection["reason"]


def test_selection_reports_not_available_without_any_metrics() -> None:
    selection = select_candidate({}, {})

    assert selection["status"] == "not_available"
    assert selection["selected"] is None
    assert selection["publication_consistent"] is False
    assert selection["decisions"] == []


def test_xgboost_candidate_degrades_explicitly_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ensemble, "_xgb_available", False)
    features = np.zeros((12, len(V2_FEATURE_NAMES)), dtype=np.float64)
    labels = np.asarray([0, 1] * 6, dtype=np.int_)
    weights = np.ones(12, dtype=np.float64)

    assert XGBOOST not in available_candidate_names()
    assert set(available_candidate_names()) == set(CANDIDATE_NAMES) - {XGBOOST}
    fit = fit_candidate(
        XGBOOST, features, labels, weights, features,
        feature_names=list(V2_FEATURE_NAMES), random_state=42,
    )
    assert fit.status == "unavailable"
    assert fit.predictions is None
    assert "xgboost" in fit.reason


def test_real_evaluation_compares_all_five_candidates(evaluation_bundle: _Bundle) -> None:
    evaluation = evaluation_bundle.evaluation

    assert set(evaluation["candidates"]) == set(CANDIDATE_NAMES)
    assert len(evaluation["candidates"]) == 5
    assert set(evaluation["forward_chaining"]["candidates"]) == set(CANDIDATE_NAMES)
    assert evaluation["candidate_name"] in DEPLOYABLE_ORDER
    assert evaluation["candidate_selection"]["selected"] == evaluation["candidate_name"]
    assert evaluation["candidate_selection"]["candidate_count"] == 5
    assert evaluation["rule_baseline"]["sample_count"] > 0
    assert evaluation["logistic_baseline"]["sample_count"] > 0
    # Every metric level the protocol promises is present and evaluated.
    assert evaluation["session_metrics"]["status"] == "evaluated"
    assert evaluation["date_metrics"]["status"] == "evaluated"
    assert evaluation["probability_metrics"]["status"] == "evaluated"
    assert evaluation["bootstrap"]["status"] == "evaluated"
    assert evaluation["bootstrap"]["unit"] == BOOTSTRAP_UNIT
    assert evaluation["abstention"]["status"] == "evaluated"
    assert evaluation["shadow_drift"]["status"] == "evaluated"
    assert evaluation["labeling_functions"]["status"] == "reported"


# ── 10. The HMM stays outside the publication chain ──────────────────────


def test_hmm_is_not_a_candidate_nor_a_publication_input() -> None:
    assert HMM_IN_PRODUCTION_CHAIN is False
    assert HMM_READMISSION_CRITERIA
    assert any("explicit feedback" in criterion for criterion in HMM_READMISSION_CRITERIA)
    assert not any("hmm" in name.lower() for name in CANDIDATE_NAMES)
    assert not any("hmm" in name.lower() for name in DEPLOYABLE_ORDER)


def test_default_training_does_not_fit_the_hmm(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(60, 6))
    labels = np.asarray([index % 2 for index in range(60)], dtype=np.int_)
    names = [f"f{index}" for index in range(features.shape[1])]
    manager = ModelManager(models_dir=tmp_path)

    summary = manager.train_all(features, names, labels, np.ones(60, dtype=np.float64))

    assert summary.hmm["status"] == "research_artifact_only"
    assert summary.hmm["trained"] is False
    assert summary.hmm["readmission_criteria"] == list(HMM_READMISSION_CRITERIA)
    assert manager.hmm._is_fitted is False
    # An unfitted research HMM must not make the deployed models look unusable.
    status = manager.readiness_status()
    assert status["ready"] is True
    assert "hmm_not_fitted" in status["reasons"]
    assert "classifier_not_fitted" not in status["reasons"]
    # It is still written with every version, so the on-disk layout is unchanged.
    saved = manager.save_all(activate=False)
    assert "hmm" in saved
    assert Path(tmp_path / saved["hmm"]).exists()


def test_quality_gate_has_no_hmm_condition() -> None:
    gate = evaluate_v2_quality_gate(_passing_evaluation(), **_PASSING_GATE_COUNTS)

    assert gate["passed"] is True
    assert not any("hmm" in check for check in gate["checks"])


# ── 11. Every quality-gate condition can independently force shadow ──────

_PASSING_GATE_COUNTS: dict[str, int] = {
    "explicit_feedback_count": 30,
    "explicit_focus_count": 15,
    "explicit_distract_count": 15,
    "distinct_feedback_days": 8,
}

#: The checks the gate must expose, in one place, so a new or removed
#: condition cannot slip through unnoticed.
_GATE_CHECKS: tuple[str, ...] = (
    "minimum_days",
    "minimum_explicit_feedback",
    "minimum_class_feedback",
    "balanced_accuracy",
    "minority_f1",
    "calibration_better_than_rule",
    "calibration_available",
    "stable_date_folds",
    "stable_future_folds",
    "no_anomalous_drift",
)


def _passing_evaluation() -> dict[str, Any]:
    return {
        "status": "evaluated",
        "candidate_name": LOGISTIC_REGRESSION,
        "primary_scheme": "forward_chaining",
        "calibration": {"method": "sigmoid", "status": "fitted"},
        "candidate": {
            "balanced_accuracy": 0.72,
            "minority_f1": 0.66,
            "brier_score": 0.14,
        },
        "rule_baseline": {"brier_score": 0.22},
        "fold_stability": {"scheme": "date_folds", "passed": True},
        "future_fold_stability": {"scheme": "forward_chaining", "passed": True},
        "shadow_drift": {
            "status": "evaluated",
            "anomalous": False,
            "statistic": 0.05,
            "threshold": DRIFT_PSI_THRESHOLD,
        },
    }


def _broken_gate_inputs(check: str) -> tuple[dict[str, Any], dict[str, int]]:
    """Return gate inputs where exactly *check* fails."""
    evaluation = _passing_evaluation()
    counts = dict(_PASSING_GATE_COUNTS)
    if check == "minimum_days":
        counts["distinct_feedback_days"] = 6
    elif check == "minimum_explicit_feedback":
        counts["explicit_feedback_count"] = 19
    elif check == "minimum_class_feedback":
        counts["explicit_focus_count"] = 4
    elif check == "balanced_accuracy":
        evaluation["candidate"]["balanced_accuracy"] = 0.54
    elif check == "minority_f1":
        evaluation["candidate"]["minority_f1"] = 0.39
    elif check == "calibration_better_than_rule":
        evaluation["candidate"]["brier_score"] = 0.22 + 0.011
    elif check == "calibration_available":
        evaluation["calibration"] = {"method": "sigmoid", "status": "unavailable"}
    elif check == "stable_date_folds":
        evaluation["fold_stability"] = {"scheme": "date_folds", "passed": False}
    elif check == "stable_future_folds":
        evaluation["future_fold_stability"] = {"scheme": "forward_chaining", "passed": False}
    elif check == "no_anomalous_drift":
        evaluation["shadow_drift"]["anomalous"] = True
    else:  # pragma: no cover - guards the parametrisation itself
        raise AssertionError(f"unknown gate check {check!r}")
    return evaluation, counts


def test_quality_gate_exposes_every_condition_and_passes_a_complete_fixture() -> None:
    gate = evaluate_v2_quality_gate(_passing_evaluation(), **_PASSING_GATE_COUNTS)

    assert tuple(gate["checks"]) == _GATE_CHECKS
    assert all(gate["checks"].values())
    assert gate["passed"] is True
    assert gate["mode"] == "ready"
    assert gate["deployment_tier"] == "full_ready"
    assert set(gate["details"]) == set(_GATE_CHECKS)


@pytest.mark.parametrize("check", _GATE_CHECKS)
def test_each_quality_gate_condition_can_force_shadow(check: str) -> None:
    evaluation, counts = _broken_gate_inputs(check)
    gate = evaluate_v2_quality_gate(evaluation, **counts)

    assert gate["checks"][check] is False, check
    others = {name: value for name, value in gate["checks"].items() if name != check}
    assert all(others.values()), f"only {check} should fail: {others}"
    assert gate["passed"] is False
    assert gate["mode"] == "shadow"
    assert gate["deployment_tier"] != "full_ready"


def test_quality_gate_thresholds_are_inclusive_at_the_boundary() -> None:
    at_minimum = dict(_PASSING_GATE_COUNTS, explicit_feedback_count=20)
    assert evaluate_v2_quality_gate(
        _passing_evaluation(), **at_minimum
    )["checks"]["minimum_explicit_feedback"] is True

    just_below = dict(_PASSING_GATE_COUNTS, distinct_feedback_days=6)
    assert evaluate_v2_quality_gate(
        _passing_evaluation(), **just_below
    )["checks"]["minimum_days"] is False
    at_seven = dict(_PASSING_GATE_COUNTS, distinct_feedback_days=7)
    assert evaluate_v2_quality_gate(
        _passing_evaluation(), **at_seven
    )["checks"]["minimum_days"] is True
    one_class_short = dict(_PASSING_GATE_COUNTS, explicit_distract_count=4)
    assert evaluate_v2_quality_gate(
        _passing_evaluation(), **one_class_short
    )["checks"]["minimum_class_feedback"] is False

    # Brier "no worse than the rule baseline by more than 0.01" is inclusive.
    evaluation = _passing_evaluation()
    evaluation["candidate"]["brier_score"] = evaluation["rule_baseline"]["brier_score"] + 0.01
    assert evaluate_v2_quality_gate(
        evaluation, **_PASSING_GATE_COUNTS
    )["checks"]["calibration_better_than_rule"] is True
    evaluation["candidate"]["brier_score"] += 1e-6
    assert evaluate_v2_quality_gate(
        evaluation, **_PASSING_GATE_COUNTS
    )["checks"]["calibration_better_than_rule"] is False


@pytest.mark.parametrize("section", ["future_fold_stability", "shadow_drift"])
def test_quality_gate_fails_closed_without_future_folds_or_drift(section: str) -> None:
    evaluation = _passing_evaluation()
    del evaluation[section]
    gate = evaluate_v2_quality_gate(evaluation, **_PASSING_GATE_COUNTS)

    expected = {
        "future_fold_stability": "stable_future_folds",
        "shadow_drift": "no_anomalous_drift",
    }[section]
    assert gate["checks"][expected] is False
    assert gate["passed"] is False
    assert gate["mode"] == "shadow"


def test_explicit_drift_report_overrides_the_evaluation_section() -> None:
    evaluation = _passing_evaluation()
    evaluation["shadow_drift"]["anomalous"] = True
    overridden = evaluate_v2_quality_gate(
        evaluation,
        **_PASSING_GATE_COUNTS,
        drift={"status": "evaluated", "anomalous": False, "statistic": 0.01},
    )
    assert overridden["checks"]["no_anomalous_drift"] is True
    assert overridden["passed"] is True

    injected = evaluate_v2_quality_gate(
        _passing_evaluation(),
        **_PASSING_GATE_COUNTS,
        drift={"status": "evaluated", "anomalous": True, "statistic": 0.9},
    )
    assert injected["checks"]["no_anomalous_drift"] is False
    assert injected["passed"] is False
    assert injected["details"]["no_anomalous_drift"]["observed"] == 0.9
