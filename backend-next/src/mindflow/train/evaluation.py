"""Honest evaluation protocol for the V2 focus classifier (phase 3.1).

What this module guarantees, and why each guarantee exists:

1. **Explicit feedback is the only scored truth, and scoring happens out of
   fold.**  Callers pass only rows that carry a real user label; the fold
   builders never look at auxiliary window labels or heuristic samples.
2. **Two fold schemes, both leak-free.**
   * ``date_folds`` — the legacy grouped split: ``GroupKFold`` over *date
     blocks* (dates joined by a cross-midnight session are one block).
   * ``forward_chaining`` — the primary scheme since phase 3.1: an expanding
     window that trains only on blocks strictly *earlier* than the test
     blocks ("leave future days out").  Its first chunk is pure history, so
     no test block ever had a date (or a session) inside a training fold.
3. **Three metric levels.**  Window-level metrics can be dominated by a single
   long session, so the same held-out predictions are also aggregated up to the
   feedback session (one user judgement = one unit) and to the date block.
4. **Probabilities are measured as probabilities.**  PR-AUC (average
   precision), Brier, ECE and a reliability-diagram-ready binned table.
5. **Uncertainty is bootstrap-based and resampled by session.**  Windows from
   one session are one judgement; resampling rows would fake precision.  The
   unit, the number of resamples and the seed are recorded in the report so a
   second run reproduces the same interval bit for bit.
6. **Abstention is measured, not assumed.**  For a confidence band around 0.5,
   the report says what share of samples the model still answers (coverage)
   and how accurate it is on the covered subset.

Nothing in this module touches deployment policy; it produces numbers that
``mindflow.train.v2`` embeds in the training report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from mindflow.train.config import TRAIN_CONFIG

# ── Fold policy ──────────────────────────────────────────────────────────

#: A held-out fold below this many scored samples cannot support a stable
#: metric, so it never counts as evidence of stability.
MIN_FOLD_TEST_SAMPLES = 5

#: Fold-stability thresholds (unchanged from the pre-3.1 grouped split; the
#: forward-chaining scheme is judged by the same numbers so a model cannot
#: pass one scheme and quietly fail the other).
FOLD_MIN_BALANCED_ACCURACY = 0.50
FOLD_MAX_BALANCED_ACCURACY_RANGE = 0.35

#: Schemes that must both be stable before a candidate may be published.
FOLD_SCHEMES: tuple[str, ...] = ("date_folds", "forward_chaining")

# ── Bootstrap policy ─────────────────────────────────────────────────────

#: Documented bootstrap contract: resample unit, draw count, seed, level.
BOOTSTRAP_UNIT = "explicit_feedback_session"
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 42
BOOTSTRAP_CONFIDENCE = 0.95

# ── Abstention policy ────────────────────────────────────────────────────

#: Default confidence band.  ``|p - 0.5| <= half_width`` means "not sure
#: enough to answer", i.e. the model abstains on everything in [0.35, 0.65].
ABSTENTION_BAND_HALF_WIDTH = 0.15
#: Alternative band widths reported next to the default one, so the operating
#: point can be chosen from data instead of guessed.
ABSTENTION_BAND_WIDTHS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.40)

# ── Shadow-drift policy ──────────────────────────────────────────────────

#: Population Stability Index above which a shift is treated as anomalous.
#: 0.25 is the conventional "major shift" cut-off (0.10 = minor).
DRIFT_PSI_THRESHOLD = 0.25
#: Probability bins used for the prediction-drift PSI.
DRIFT_PROBABILITY_BINS = 10
#: Feature quantile bins used for per-feature PSI.
DRIFT_FEATURE_BINS = 4


# ── Folds ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationFold:
    """One held-out fold: which rows train, which rows are scored."""

    index: int
    scheme: str
    train_idx: npt.NDArray[np.int_]
    test_idx: npt.NDArray[np.int_]
    train_groups: list[str] = field(default_factory=list)
    test_groups: list[str] = field(default_factory=list)
    train_dates: list[str] = field(default_factory=list)
    test_dates: list[str] = field(default_factory=list)
    #: Latest training date; forward-chaining folds guarantee every test date
    #: is strictly later than this.
    cutoff_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "scheme": self.scheme,
            "train_groups": list(self.train_groups),
            "test_groups": list(self.test_groups),
            "train_dates": list(self.train_dates),
            "test_dates": list(self.test_dates),
            "train_size": int(len(self.train_idx)),
            "test_size": int(len(self.test_idx)),
            "cutoff_date": self.cutoff_date,
        }


def group_order(
    group_ids: list[str], group_dates: dict[str, list[str]] | None = None
) -> list[str]:
    """Order date blocks chronologically (earliest covered date, then id).

    The tie-break on the group id keeps the order deterministic for two blocks
    that cover exactly the same dates.
    """
    dates = group_dates or {}

    def key(group: str) -> tuple[str, str]:
        known = sorted(date for date in dates.get(group, []) if date)
        return (known[0] if known else group, group)

    return sorted(set(group_ids), key=key)


def _chunk_groups(
    order: list[str], sizes: dict[str, int], n_chunks: int
) -> list[list[str]]:
    """Split *order* into contiguous chunks balanced by sample count."""
    n_chunks = max(1, min(n_chunks, len(order)))
    total = sum(sizes.get(group, 0) for group in order)
    chunks: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    remaining_total = total
    remaining_chunks = n_chunks
    for position, group in enumerate(order):
        current.append(group)
        current_size += sizes.get(group, 0)
        remaining_total -= sizes.get(group, 0)
        chunks_left_after = remaining_chunks - 1
        groups_left = len(order) - position - 1
        if chunks_left_after <= 0 or groups_left < chunks_left_after:
            continue
        fair_share = (current_size + remaining_total) / remaining_chunks
        if current_size >= fair_share - 1e-9:
            chunks.append(current)
            current, current_size = [], 0
            remaining_chunks -= 1
    if current:
        chunks.append(current)
    return chunks


def build_forward_chaining_folds(
    group_ids: list[str],
    *,
    group_dates: dict[str, list[str]] | None = None,
    max_folds: int = TRAIN_CONFIG.group_folds,
    min_test_samples: int = MIN_FOLD_TEST_SAMPLES,
) -> list[EvaluationFold]:
    """Expanding-window folds that never train on a future date block.

    Fold *k* trains on every block before chunk *k* and is scored on chunk *k*.
    Chunk 0 is training-only history.  The number of chunks is the largest one
    (up to ``max_folds``) whose scored chunks each hold at least
    ``min_test_samples`` rows; a fold too small to be stable would otherwise
    make the stability check meaningless.

    Returns an empty list when there are fewer than three date blocks: with two
    blocks the first scored fold would have no training history at all.
    """
    order = group_order(group_ids, group_dates)
    if len(order) < 3:
        return []

    sizes: dict[str, int] = {}
    for group in group_ids:
        sizes[group] = sizes.get(group, 0) + 1

    selected: list[list[str]] | None = None
    fold_count = 0
    for candidate_folds in range(min(max_folds, len(order) - 1), 0, -1):
        chunks = _chunk_groups(order, sizes, candidate_folds + 1)
        scored = chunks[1:]
        if len(scored) == candidate_folds and all(
            sum(sizes.get(group, 0) for group in chunk) >= min_test_samples
            for chunk in scored
        ):
            selected = chunks
            fold_count = candidate_folds
            break
    if selected is None:
        # No split satisfies the minimum scored size; report the widest one
        # anyway so the instability is visible instead of silently skipped.
        fold_count = min(max_folds, len(order) - 1)
        selected = _chunk_groups(order, sizes, fold_count + 1)

    group_index: dict[str, list[int]] = {}
    for position, group in enumerate(group_ids):
        group_index.setdefault(group, []).append(position)

    dates = group_dates or {}
    folds: list[EvaluationFold] = []
    for fold_number in range(1, min(fold_count, len(selected) - 1) + 1):
        train_groups = [g for chunk in selected[:fold_number] for g in chunk]
        test_groups = list(selected[fold_number])
        train_idx = np.asarray(
            [i for group in train_groups for i in group_index.get(group, [])],
            dtype=np.int_,
        )
        test_idx = np.asarray(
            [i for group in test_groups for i in group_index.get(group, [])],
            dtype=np.int_,
        )
        train_dates = sorted({d for group in train_groups for d in dates.get(group, [group])})
        test_dates = sorted({d for group in test_groups for d in dates.get(group, [group])})
        folds.append(
            EvaluationFold(
                index=fold_number,
                scheme="forward_chaining",
                train_idx=train_idx,
                test_idx=test_idx,
                train_groups=sorted(train_groups),
                test_groups=sorted(test_groups),
                train_dates=train_dates,
                test_dates=test_dates,
                cutoff_date=train_dates[-1] if train_dates else None,
            )
        )
    return folds


def fold_stability(
    per_fold_balanced_accuracy: list[float],
    per_fold_test_size: list[int],
    *,
    scheme: str,
) -> dict[str, Any]:
    """Summarise whether a scheme's folds are stably good.

    Same criteria for both schemes (see the module constants): every fold must
    clear the balanced-accuracy floor, the spread must stay bounded, and no
    fold may be too small to be evidence.
    """
    if not per_fold_balanced_accuracy:
        return {"scheme": scheme, "passed": False, "reason": "no valid held-out folds"}
    values = [float(value) for value in per_fold_balanced_accuracy]
    min_size = min(per_fold_test_size) if per_fold_test_size else 0
    minimum = min(values)
    spread = max(values) - minimum
    return {
        "scheme": scheme,
        "passed": bool(
            minimum >= FOLD_MIN_BALANCED_ACCURACY
            and spread <= FOLD_MAX_BALANCED_ACCURACY_RANGE
            and min_size >= MIN_FOLD_TEST_SAMPLES
        ),
        "fold_count": len(values),
        "min_balanced_accuracy": round(minimum, 6),
        "max_balanced_accuracy": round(max(values), 6),
        "range": round(spread, 6),
        "min_test_size": int(min_size),
        "per_fold_balanced_accuracy": [round(value, 6) for value in values],
        "thresholds": {
            "min_balanced_accuracy": FOLD_MIN_BALANCED_ACCURACY,
            "max_range": FOLD_MAX_BALANCED_ACCURACY_RANGE,
            "min_test_size": MIN_FOLD_TEST_SAMPLES,
        },
    }


# ── Probability metrics ──────────────────────────────────────────────────


def expected_calibration_error(
    y_true: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    *,
    n_bins: int = TRAIN_CONFIG.calibration_bins,
) -> float:
    """Sample-weighted mean |observed - predicted| over probability bins."""
    rows = reliability_table(y_true, y_proba, n_bins=n_bins)
    return float(sum(float(row["ece_contribution"]) for row in rows))


def reliability_table(
    y_true: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    *,
    n_bins: int = TRAIN_CONFIG.calibration_bins,
) -> list[dict[str, Any]]:
    """Binned predicted-vs-observed table, ready to plot as a reliability diagram.

    Bins are half-open ``[low, high)`` with the last bin closed, so every
    sample lands in exactly one bin (the legacy table used closed edges on both
    sides, which double-counted values that sat exactly on an edge).
    """
    if len(y_proba) == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_proba)
    rows: list[dict[str, Any]] = []
    for index in range(n_bins):
        low, high = float(edges[index]), float(edges[index + 1])
        last = index == n_bins - 1
        in_bin = (y_proba >= low) & ((y_proba <= high) if last else (y_proba < high))
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_prediction = float(np.mean(y_proba[in_bin]))
        fraction_positive = float(np.mean(y_true[in_bin]))
        rows.append({
            "bin_low": round(low, 4),
            "bin_high": round(high, 4),
            "count": count,
            "weight": round(count / total, 6),
            "mean_prediction": round(mean_prediction, 6),
            "fraction_positive": round(fraction_positive, 6),
            "gap": round(fraction_positive - mean_prediction, 6),
            "ece_contribution": round(
                (count / total) * abs(fraction_positive - mean_prediction), 6
            ),
        })
    return rows


def probability_metrics(
    y_true: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    *,
    n_bins: int = TRAIN_CONFIG.calibration_bins,
) -> dict[str, Any]:
    """PR-AUC, ROC-AUC, Brier, ECE and the reliability table for one pool."""
    result: dict[str, Any] = {"sample_count": int(len(y_true))}
    try:
        result["brier_score"] = round(float(brier_score_loss(y_true, y_proba)), 6)
    except (ValueError, TypeError):
        result["brier_score"] = None
    if len(np.unique(y_true)) == 2:
        try:
            result["pr_auc"] = round(float(average_precision_score(y_true, y_proba)), 6)
            result["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 6)
        except ValueError:
            result["pr_auc"] = None
            result["roc_auc"] = None
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    result["expected_calibration_error"] = round(
        expected_calibration_error(y_true, y_proba, n_bins=n_bins), 6
    )
    result["reliability_table"] = reliability_table(y_true, y_proba, n_bins=n_bins)
    result["bin_count"] = len(result["reliability_table"])
    return result


def classification_metrics(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    *,
    with_confusion_matrix: bool = True,
) -> dict[str, Any]:
    """Window-level metrics: the historical key set plus the new probability ones."""
    unique, counts = np.unique(y_true, return_counts=True)
    if len(unique) == 0:
        return {"balanced_accuracy": 0.0, "minority_f1": 0.0, "brier_score": 1.0}
    minority = int(unique[int(np.argmin(counts))])
    balanced = float(balanced_accuracy_score(y_true, y_pred))
    minority_f1 = float(
        f1_score(
            (y_true == minority).astype(int),
            (y_pred == minority).astype(int),
            zero_division=0.0,
        )
    )
    result: dict[str, Any] = {
        "balanced_accuracy": round(balanced, 6),
        "minority_f1": round(minority_f1, 6),
        "minority_class": minority,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "sample_count": int(len(y_true)),
    }
    probabilities = probability_metrics(y_true, y_proba)
    result["brier_score"] = probabilities["brier_score"]
    if probabilities["brier_score"] is None:
        result["brier_score"] = 1.0
    result.update({
        "pr_auc": probabilities["pr_auc"],
        # Legacy name for the same PR-AUC number (``average_precision_score``):
        # older reports and downstream readers use this key.
        "average_precision": probabilities["pr_auc"],
        "roc_auc": probabilities["roc_auc"],
        "expected_calibration_error": probabilities["expected_calibration_error"],
        "reliability_table": probabilities["reliability_table"],
    })
    if with_confusion_matrix and len(unique) == 2:
        result["confusion_matrix"] = _confusion_matrix(y_true, y_pred)
    # Legacy key: the pre-3.1 binned table, now edge-correct.
    result["calibration"] = probabilities["reliability_table"]
    return result


def _confusion_matrix(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_]
) -> list[list[int]]:
    matrix = np.zeros((2, 2), dtype=int)
    for truth, prediction in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        if truth in (0, 1) and prediction in (0, 1):
            matrix[int(truth), int(prediction)] += 1
    return matrix.tolist()


# ── Session / date aggregation ───────────────────────────────────────────


def _aggregate_units(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    unit_ids: list[str],
    *,
    unit_name: str,
) -> dict[str, Any]:
    """Aggregate window predictions to one row per unit (session or date block).

    Within a unit the *label* is the majority window label (rows of one
    feedback session share one label by construction) and the *prediction* is
    the majority vote of the window predictions, with the mean probability as
    the tie-break.  That is exactly how the product consumes the model: one
    answer per session, not one per five-minute window.
    """
    order: list[str] = []
    buckets: dict[str, list[int]] = {}
    for position, unit in enumerate(unit_ids):
        if unit not in buckets:
            buckets[unit] = []
            order.append(unit)
        buckets[unit].append(position)
    if not order:
        return {"status": "not_available", "unit": unit_name, "unit_count": 0}

    units_true: list[int] = []
    units_pred: list[int] = []
    units_proba: list[float] = []
    per_unit: list[dict[str, Any]] = []
    for unit in order:
        positions = np.asarray(buckets[unit], dtype=np.int_)
        labels = y_true[positions]
        votes = y_pred[positions]
        mean_proba = float(np.mean(y_proba[positions]))
        label = int(np.bincount(labels, minlength=2).argmax())
        focus_votes = int((votes == 1).sum())
        distract_votes = int((votes == 0).sum())
        if focus_votes == distract_votes:
            prediction = int(mean_proba >= 0.5)
        else:
            prediction = int(focus_votes > distract_votes)
        units_true.append(label)
        units_pred.append(prediction)
        units_proba.append(mean_proba)
        per_unit.append({
            "unit": unit,
            "windows": int(len(positions)),
            "label": label,
            "prediction": prediction,
            "mean_probability": round(mean_proba, 6),
            "correct": bool(label == prediction),
        })

    true_arr = np.asarray(units_true, dtype=np.int_)
    pred_arr = np.asarray(units_pred, dtype=np.int_)
    proba_arr = np.asarray(units_proba, dtype=np.float64)
    metrics = classification_metrics(true_arr, pred_arr, proba_arr)
    return {
        "status": "evaluated",
        "unit": unit_name,
        "unit_count": len(order),
        "window_count": int(len(y_true)),
        "aggregation": "majority_vote_per_unit",
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "minority_f1": metrics["minority_f1"],
        "brier_score": metrics["brier_score"],
        "pr_auc": metrics["pr_auc"],
        "per_unit": per_unit,
    }


def session_level_metrics(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    session_ids: list[str],
) -> dict[str, Any]:
    """Metrics with the feedback session as the unit of analysis."""
    if not session_ids or any(not session for session in session_ids):
        return {
            "status": "not_available",
            "unit": "feedback_session",
            "reason": "holdout rows carry no feedback session id",
        }
    return _aggregate_units(
        y_true, y_pred, y_proba, session_ids, unit_name="feedback_session"
    )


def date_level_metrics(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    group_ids: list[str],
) -> dict[str, Any]:
    """Metrics with the (cross-midnight-merged) date block as the unit."""
    if not group_ids:
        return {
            "status": "not_available",
            "unit": "date_block",
            "reason": "holdout rows carry no date block id",
        }
    return _aggregate_units(y_true, y_pred, y_proba, group_ids, unit_name="date_block")


# ── Session bootstrap ────────────────────────────────────────────────────


def session_bootstrap_draws(
    session_ids: list[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[tuple[list[str], npt.NDArray[np.int_]]]:
    """Draw ``resamples`` bootstrap samples of *sessions*, expanded to rows.

    The unit is the session, not the row: a drawn session contributes **all**
    of its windows, so a long session cannot be half-counted.  ``seed`` makes
    the whole sequence reproducible; the same inputs always produce the same
    draws.

    Returns one ``(drawn_session_ids, row_indices)`` pair per resample.
    """
    positions: dict[str, list[int]] = {}
    order: list[str] = []
    for position, session in enumerate(session_ids):
        if session not in positions:
            positions[session] = []
            order.append(session)
        positions[session].append(position)
    if not order:
        return []
    rng = np.random.default_rng(seed)
    draws: list[tuple[list[str], npt.NDArray[np.int_]]] = []
    for _ in range(max(0, resamples)):
        picked = rng.integers(0, len(order), size=len(order))
        drawn = [order[int(index)] for index in picked]
        rows = [row for session in drawn for row in positions[session]]
        draws.append((drawn, np.asarray(rows, dtype=np.int_)))
    return draws


def bootstrap_confidence_intervals(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    session_ids: list[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Percentile bootstrap CIs for the headline metrics, resampled by session."""
    point = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 0
        else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_proba)) if len(y_true) else 1.0,
    }
    minority = _minority_class(y_true)
    point["minority_f1"] = float(
        f1_score(
            (y_true == minority).astype(int),
            (y_pred == minority).astype(int),
            zero_division=0.0,
        )
        if minority is not None else 0.0
    )
    point["pr_auc"] = (
        float(average_precision_score(y_true, y_proba))
        if len(np.unique(y_true)) == 2 else float("nan")
    )

    draws = session_bootstrap_draws(session_ids, resamples=resamples, seed=seed)
    if not draws:
        return {
            "status": "not_available",
            "unit": BOOTSTRAP_UNIT,
            "reason": "no feedback sessions to resample",
        }

    collected: dict[str, list[float]] = {name: [] for name in point}
    skipped = 0
    for _drawn, rows in draws:
        sample_true = y_true[rows]
        sample_pred = y_pred[rows]
        sample_proba = y_proba[rows]
        if len(sample_true) == 0 or len(np.unique(sample_true)) < 2:
            # A resample that drew only one class carries no balanced-accuracy
            # or PR-AUC information; counting it would fabricate a 1.0.
            skipped += 1
            continue
        collected["balanced_accuracy"].append(
            float(balanced_accuracy_score(sample_true, sample_pred))
        )
        collected["accuracy"].append(float(accuracy_score(sample_true, sample_pred)))
        collected["brier_score"].append(float(brier_score_loss(sample_true, sample_proba)))
        sample_minority = _minority_class(sample_true)
        collected["minority_f1"].append(
            float(
                f1_score(
                    (sample_true == sample_minority).astype(int),
                    (sample_pred == sample_minority).astype(int),
                    zero_division=0.0,
                )
                if sample_minority is not None else 0.0
            )
        )
        collected["pr_auc"].append(
            float(average_precision_score(sample_true, sample_proba))
        )

    alpha = (1.0 - confidence) / 2.0
    metrics: dict[str, Any] = {}
    for name, values in collected.items():
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        metrics[name] = {
            "point": round(float(point[name]), 6) if math.isfinite(point[name]) else None,
            "lower": round(float(np.quantile(finite, alpha)), 6) if finite.size else None,
            "upper": round(float(np.quantile(finite, 1.0 - alpha)), 6) if finite.size else None,
            "std": round(float(np.std(finite)), 6) if finite.size else None,
            "valid_resamples": int(finite.size),
        }
    return {
        "status": "evaluated",
        "unit": BOOTSTRAP_UNIT,
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence_level": confidence,
        "session_count": len(set(session_ids)),
        "row_count": int(len(y_true)),
        "one_class_resamples_skipped": skipped,
        "metrics": metrics,
        "note": (
            "Percentile bootstrap. Each resample draws whole feedback sessions "
            "with replacement, so a session's windows always move together."
        ),
    }


def _minority_class(y_true: npt.NDArray[np.int_]) -> int | None:
    unique, counts = np.unique(y_true, return_counts=True)
    if len(unique) == 0:
        return None
    return int(unique[int(np.argmin(counts))])


# ── Abstention / coverage ────────────────────────────────────────────────


def abstention_metrics(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    *,
    band_half_width: float = ABSTENTION_BAND_HALF_WIDTH,
    widths: tuple[float, ...] = ABSTENTION_BAND_WIDTHS,
) -> dict[str, Any]:
    """Coverage and covered-subset accuracy for a confidence band.

    The model abstains whenever ``|p - 0.5| <= band_half_width``.  Coverage is
    the share of samples it still answers; covered accuracy is the accuracy on
    exactly that subset, so a high coverage number can never hide a bad one.
    """
    if len(y_true) == 0:
        return {"status": "not_available", "reason": "no held-out samples"}
    options = [float(width) for width in widths]
    if band_half_width not in options:
        options.append(float(band_half_width))
    options = sorted(set(options))
    by_width = [
        _coverage_at_width(y_true, y_pred, y_proba, width) for width in options
    ]
    selected = next(
        row for row in by_width if row["half_width"] == round(float(band_half_width), 6)
    )
    return {
        "status": "evaluated",
        "policy": "abstain_inside_confidence_band",
        "band": {
            "low": round(0.5 - band_half_width, 6),
            "high": round(0.5 + band_half_width, 6),
            "half_width": round(float(band_half_width), 6),
        },
        "sample_count": int(len(y_true)),
        "covered_count": selected["covered_count"],
        "abstained_count": selected["abstained_count"],
        "coverage": selected["coverage"],
        "abstention_rate": selected["abstention_rate"],
        "covered_accuracy": selected["covered_accuracy"],
        "covered_balanced_accuracy": selected["covered_balanced_accuracy"],
        "covered_minority_f1": selected["covered_minority_f1"],
        "abstained_accuracy": selected["abstained_accuracy"],
        "by_band_width": by_width,
        "note": (
            "Coverage falls monotonically as the band widens; the covered "
            "subset is what the user actually sees."
        ),
    }


def _coverage_at_width(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    y_proba: npt.NDArray[np.float64],
    half_width: float,
) -> dict[str, Any]:
    covered = np.abs(y_proba - 0.5) > half_width
    total = int(len(y_true))
    covered_count = int(covered.sum())
    row: dict[str, Any] = {
        "half_width": round(float(half_width), 6),
        "band_low": round(0.5 - half_width, 6),
        "band_high": round(0.5 + half_width, 6),
        "covered_count": covered_count,
        "abstained_count": total - covered_count,
        "coverage": round(covered_count / total, 6) if total else 0.0,
        "abstention_rate": round((total - covered_count) / total, 6) if total else 0.0,
        "covered_accuracy": None,
        "covered_balanced_accuracy": None,
        "covered_minority_f1": None,
        "abstained_accuracy": None,
    }
    if covered_count:
        row["covered_accuracy"] = round(
            float(accuracy_score(y_true[covered], y_pred[covered])), 6
        )
        if len(np.unique(y_true[covered])) > 1:
            row["covered_balanced_accuracy"] = round(
                float(balanced_accuracy_score(y_true[covered], y_pred[covered])), 6
            )
        covered_minority = _minority_class(y_true[covered])
        if covered_minority is not None:
            row["covered_minority_f1"] = round(
                float(
                    f1_score(
                        (y_true[covered] == covered_minority).astype(int),
                        (y_pred[covered] == covered_minority).astype(int),
                        zero_division=0.0,
                    )
                ),
                6,
            )
    abstained = ~covered
    if abstained.any():
        row["abstained_accuracy"] = round(
            float(accuracy_score(y_true[abstained], y_pred[abstained])), 6
        )
    return row


# ── Shadow drift ─────────────────────────────────────────────────────────


def population_stability_index(
    reference: npt.NDArray[np.float64],
    comparison: npt.NDArray[np.float64],
    *,
    edges: npt.NDArray[np.float64] | None = None,
    n_bins: int = 10,
) -> float:
    """PSI between two 1-D distributions (0 = identical, >0.25 = major shift)."""
    if reference.size == 0 or comparison.size == 0:
        return 0.0
    if edges is None:
        edges = _quantile_edges(reference, n_bins)
    if edges.size < 2:
        return 0.0
    reference_counts, _ = np.histogram(reference, bins=edges)
    comparison_counts, _ = np.histogram(comparison, bins=edges)
    reference_share = reference_counts / max(1, reference_counts.sum())
    comparison_share = comparison_counts / max(1, comparison_counts.sum())
    epsilon = 1e-6
    reference_share = np.clip(reference_share, epsilon, None)
    comparison_share = np.clip(comparison_share, epsilon, None)
    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def _quantile_edges(
    values: npt.NDArray[np.float64], n_bins: int
) -> npt.NDArray[np.float64]:
    """Bin edges from the reference quantiles, widened so every value falls inside."""
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, max(2, n_bins) + 1))
    edges = np.unique(quantiles)
    if edges.size < 2:
        return np.asarray([], dtype=np.float64)
    edges = edges.astype(np.float64)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def temporal_drift_report(
    features: npt.NDArray[np.float64],
    y_proba: npt.NDArray[np.float64],
    feature_names: list[str],
    order_keys: list[str],
    *,
    threshold: float = DRIFT_PSI_THRESHOLD,
    top_features: int = 3,
) -> dict[str, Any]:
    """Compare the earliest half of the scored window against the latest half.

    This is the *shadow drift* signal the quality gate consumes: a candidate
    whose inputs or whose own predicted probabilities shift materially between
    the start and the end of the same evaluation window is not stable enough to
    publish, even when its pooled metrics look fine.

    Returns ``status="not_available"`` (which the gate fails closed on) when
    there are too few rows or only one distinct time key to split.
    """
    if len(features) == 0 or len(features) != len(y_proba):
        return {"status": "not_available", "reason": "no scored rows"}
    positions = sorted(range(len(order_keys)), key=lambda index: (order_keys[index], index))
    if len({order_keys[index] for index in positions}) < 2:
        return {"status": "not_available", "reason": "single date block in the holdout"}
    midpoint = len(positions) // 2
    if midpoint < 1 or len(positions) - midpoint < 1:
        return {"status": "not_available", "reason": "holdout too small to split"}
    early = np.asarray(positions[:midpoint], dtype=np.int_)
    late = np.asarray(positions[midpoint:], dtype=np.int_)

    probability_edges = np.linspace(0.0, 1.0, DRIFT_PROBABILITY_BINS + 1)
    prediction_psi = population_stability_index(
        y_proba[early], y_proba[late], edges=probability_edges
    )
    feature_psis: list[dict[str, Any]] = []
    for column, name in enumerate(feature_names):
        if column >= features.shape[1]:
            break
        edges = _quantile_edges(features[early, column], DRIFT_FEATURE_BINS)
        feature_psis.append({
            "feature": name,
            "psi": round(
                population_stability_index(
                    features[early, column], features[late, column], edges=edges
                ),
                6,
            ),
        })
    feature_psis.sort(key=lambda row: float(row["psi"]), reverse=True)
    max_feature_psi = float(feature_psis[0]["psi"]) if feature_psis else 0.0
    statistic = max(prediction_psi, max_feature_psi)
    return {
        "status": "evaluated",
        "unit": "earliest_half_vs_latest_half",
        "split_key": order_keys[positions[midpoint]],
        "reference_count": int(len(early)),
        "comparison_count": int(len(late)),
        "threshold": threshold,
        "prediction_psi": round(prediction_psi, 6),
        "max_feature_psi": round(max_feature_psi, 6),
        "statistic": round(statistic, 6),
        "top_features": feature_psis[:top_features],
        "anomalous": bool(statistic > threshold),
    }


# ── Pooled report ────────────────────────────────────────────────────────


@dataclass
class PredictionPool:
    """Held-out predictions from one scheme, with the provenance to explain them."""

    y_true: npt.NDArray[np.int_]
    y_pred: npt.NDArray[np.int_]
    y_proba: npt.NDArray[np.float64]
    session_ids: list[str] = field(default_factory=list)
    group_ids: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    #: Positions of the scored rows inside the caller's evaluation mask, so a
    #: caller can look the original feature rows back up (drift checks).
    positions: npt.NDArray[np.int_] = field(
        default_factory=lambda: np.asarray([], dtype=np.int_)
    )

    def __len__(self) -> int:
        return int(len(self.y_true))


def pooled_report(
    pool: PredictionPool,
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    abstention_half_width: float = ABSTENTION_BAND_HALF_WIDTH,
) -> dict[str, Any]:
    """Every metric level for one scheme's held-out pool."""
    y_true, y_pred, y_proba = pool.y_true, pool.y_pred, pool.y_proba
    return {
        "metrics": classification_metrics(y_true, y_pred, y_proba),
        "session_metrics": session_level_metrics(
            y_true, y_pred, y_proba, pool.session_ids
        ),
        "date_metrics": date_level_metrics(y_true, y_pred, y_proba, pool.group_ids),
        "bootstrap": bootstrap_confidence_intervals(
            y_true, y_pred, y_proba, pool.session_ids,
            resamples=bootstrap_resamples, seed=bootstrap_seed,
        ),
        "abstention": abstention_metrics(
            y_true, y_pred, y_proba, band_half_width=abstention_half_width
        ),
    }


__all__ = [
    "ABSTENTION_BAND_HALF_WIDTH",
    "ABSTENTION_BAND_WIDTHS",
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "BOOTSTRAP_UNIT",
    "DRIFT_PSI_THRESHOLD",
    "FOLD_MAX_BALANCED_ACCURACY_RANGE",
    "FOLD_MIN_BALANCED_ACCURACY",
    "FOLD_SCHEMES",
    "MIN_FOLD_TEST_SAMPLES",
    "EvaluationFold",
    "PredictionPool",
    "abstention_metrics",
    "bootstrap_confidence_intervals",
    "build_forward_chaining_folds",
    "classification_metrics",
    "date_level_metrics",
    "expected_calibration_error",
    "fold_stability",
    "group_order",
    "pooled_report",
    "population_stability_index",
    "probability_metrics",
    "reliability_table",
    "session_bootstrap_draws",
    "session_level_metrics",
    "temporal_drift_report",
]
