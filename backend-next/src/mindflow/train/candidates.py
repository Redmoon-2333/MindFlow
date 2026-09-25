"""Candidate model set and the stability-based selection rule (phase 3.5).

Every training run compares exactly five candidates:

1. ``rule_engine``           — the always-available heuristic baseline;
2. ``logistic_regression``   — class-balanced linear model;
3. ``random_forest``         — balanced Random Forest;
4. ``xgboost``               — gradient boosting (when installed);
5. ``rf_xgb_soft_voting``    — Random Forest + XGBoost soft voting, the model
   the production pipeline has been publishing.

Selection rule (deliberately conservative, and recorded in the report): start
from the *simplest* candidate and only promote a more complex one when it is
**stably better** across held-out folds — an aggregate balanced-accuracy gain
of at least :data:`PROMOTION_MARGIN`, a win on at least
:data:`PROMOTION_FOLD_WIN_RATIO` of the shared folds, no single fold lost by
more than :data:`PROMOTION_MAX_FOLD_LOSS`, and no contradiction on the
secondary scheme.  A complex ensemble that merely ties, or that wins on
average while losing on individual folds, does not win: with small samples that
is noise, and the simpler model is kept.

The rule engine is compared but never selected: it is the production fallback,
not a trained model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mindflow.train.grouping import session_balanced_weights
from mindflow.train.models.ensemble import CalibrationUnavailableError, EnsembleClassifier

RULE_ENGINE = "rule_engine"
LOGISTIC_REGRESSION = "logistic_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"
RF_XGB_SOFT_VOTING = "rf_xgb_soft_voting"

#: The five candidates, in the order they are reported.
CANDIDATE_NAMES: tuple[str, ...] = (
    RULE_ENGINE,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    XGBOOST,
    RF_XGB_SOFT_VOTING,
)

#: Model complexity rank.  Random Forest and XGBoost share a rank: neither is
#: a strict generalisation of the other, so the comparison order between them
#: is the tuple order above, not a complexity claim.
CANDIDATE_COMPLEXITY: dict[str, int] = {
    RULE_ENGINE: 0,
    LOGISTIC_REGRESSION: 1,
    RANDOM_FOREST: 2,
    XGBOOST: 2,
    RF_XGB_SOFT_VOTING: 3,
}

#: Trained candidates, simplest first — the ladder the selection rule walks.
DEPLOYABLE_ORDER: tuple[str, ...] = (
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    XGBOOST,
    RF_XGB_SOFT_VOTING,
)

#: The artifact the current production pipeline trains and publishes.  Kept
#: explicit so a report can say whether the selected candidate and the
#: published artifact agree.
DEFAULT_PUBLICATION_MODEL = RF_XGB_SOFT_VOTING

# ── Promotion thresholds ─────────────────────────────────────────────────

#: Required aggregate balanced-accuracy gain over the current incumbent.
PROMOTION_MARGIN = 0.02
#: Share of shared folds the challenger must win outright.
PROMOTION_FOLD_WIN_RATIO = 0.6
#: Largest balanced-accuracy loss tolerated on any single shared fold.
PROMOTION_MAX_FOLD_LOSS = 0.10

#: Feature columns the rule baseline is defined on.  Resolved by name: a
#: positional read silently produces a meaningless baseline when the feature
#: set changes.
RULE_REQUIRED_FEATURES: tuple[str, ...] = (
    "app_switch_count",
    "top_app_ratio",
    "idle_ratio",
)


def rule_probabilities(
    features: npt.NDArray[np.float64], feature_names: list[str] | None = None
) -> npt.NDArray[np.float64]:
    """Heuristic rule-engine probability of focus, for the rule baseline.

    Raises:
        ValueError: when a required behavioural feature is absent.  A baseline
            computed from the wrong columns is worse than a loud failure.
    """
    names = list(feature_names) if feature_names is not None else []
    missing = [name for name in RULE_REQUIRED_FEATURES if name not in names]
    if missing:
        msg = f"rule baseline needs behavioural features {missing!r}; present: {names!r}"
        raise ValueError(msg)

    def column(feature: str) -> npt.NDArray[np.float64]:
        return np.asarray(features[:, names.index(feature)], dtype=np.float64)

    p = np.full(features.shape[0], 0.5)
    switch_count = column("app_switch_count")
    p[switch_count < 5] += 0.2
    p[column("top_app_ratio") > 0.7] += 0.15
    p[switch_count > 20] -= 0.3
    p[column("idle_ratio") > 0.8] -= 0.1
    return np.clip(p, 0.0, 1.0).astype(np.float64)


@dataclass
class CandidateFit:
    """One candidate's out-of-fold predictions, or why it has none."""

    name: str
    status: str = "fitted"
    predictions: npt.NDArray[np.int_] | None = None
    probabilities: npt.NDArray[np.float64] | None = None
    reason: str = ""


@dataclass
class FoldResult:
    """Every candidate's predictions for one held-out fold."""

    fits: dict[str, CandidateFit] = field(default_factory=dict)
    #: Candidate that calibration/provenance made unavailable in this fold.
    failures: list[str] = field(default_factory=list)


def available_candidate_names() -> tuple[str, ...]:
    """Candidate names this environment can actually fit (XGBoost is optional)."""
    from mindflow.train.models import ensemble

    names = [RULE_ENGINE, LOGISTIC_REGRESSION, RANDOM_FOREST]
    if bool(getattr(ensemble, "_xgb_available", False)):
        names.append(XGBOOST)
    names.append(RF_XGB_SOFT_VOTING)
    return tuple(names)


def make_ensemble(calibration: str | None) -> EnsembleClassifier:
    """The soft-voting candidate, built exactly as production builds it."""
    return EnsembleClassifier(calibration=calibration)


def fit_candidate(
    name: str,
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.int_],
    sample_weight: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
    *,
    feature_names: list[str],
    random_state: int,
    groups: npt.NDArray[Any] | None = None,
    label_sources: npt.NDArray[Any] | None = None,
    session_ids: npt.NDArray[Any] | None = None,
    calibration: str | None = None,
) -> CandidateFit:
    """Fit one candidate and predict the held-out fold.

    ``rf_xgb_soft_voting`` is fitted through :class:`EnsembleClassifier` with
    the same provenance arguments production passes, so its fold behaviour —
    including the grouped calibration holdout — is identical to deployment.

    Raises:
        CalibrationUnavailableError: only for the soft-voting candidate, when
            the requested calibration cannot be fitted out of sample.
    """
    from mindflow.train.models import ensemble

    if name == RULE_ENGINE:
        try:
            proba = rule_probabilities(x_test, feature_names)
        except ValueError as exc:
            return CandidateFit(name=name, status="unavailable", reason=str(exc))
        return CandidateFit(
            name=name,
            predictions=np.asarray(proba >= 0.5, dtype=np.int_),
            probabilities=np.asarray(proba, dtype=np.float64),
        )

    if name == LOGISTIC_REGRESSION:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, random_state=random_state, class_weight="balanced"
            ),
        )
        model.fit(x_train, y_train, logisticregression__sample_weight=sample_weight)
        return CandidateFit(
            name=name,
            predictions=np.asarray(model.predict(x_test), dtype=np.int_),
            probabilities=np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64),
        )

    if name == RANDOM_FOREST:
        # Same hyperparameters as the ensemble's forest, plus balanced class
        # weights so it competes with the balanced logistic baseline.
        forest = RandomForestClassifier(
            **{**EnsembleClassifier._RF_PARAMS, "class_weight": "balanced"}
        )
        forest.fit(x_train, y_train, sample_weight=sample_weight)
        return CandidateFit(
            name=name,
            predictions=np.asarray(forest.predict(x_test), dtype=np.int_),
            probabilities=np.asarray(forest.predict_proba(x_test)[:, 1], dtype=np.float64),
        )

    if name == XGBOOST:
        if not bool(getattr(ensemble, "_xgb_available", False)):
            return CandidateFit(
                name=name, status="unavailable", reason="xgboost is not installed"
            )
        from xgboost import XGBClassifier

        booster = XGBClassifier(**EnsembleClassifier._XGB_PARAMS)
        booster.fit(x_train, y_train, sample_weight=sample_weight)
        return CandidateFit(
            name=name,
            predictions=np.asarray(booster.predict(x_test).astype(int), dtype=np.int_),
            probabilities=np.asarray(booster.predict_proba(x_test)[:, 1], dtype=np.float64),
        )

    if name == RF_XGB_SOFT_VOTING:
        model = make_ensemble(calibration)
        model.fit(
            x_train, y_train, feature_names,
            sample_weight=sample_weight,
            groups=groups,
            label_sources=label_sources,
            session_ids=session_ids,
        )
        if calibration is not None and model.calibrator is None:
            raise CalibrationUnavailableError("Requested calibration was not fitted")
        return CandidateFit(
            name=name,
            predictions=np.asarray(model.predict(x_test), dtype=np.int_),
            probabilities=np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64),
        )

    raise ValueError(f"unknown candidate {name!r}")


# ── Selection rule ───────────────────────────────────────────────────────


def _fold_comparison(
    challenger: str,
    incumbent: str,
    per_fold: dict[str, dict[int, float]],
) -> dict[str, Any]:
    """Fold-by-fold comparison on the folds where both candidates are defined."""
    shared = sorted(set(per_fold.get(challenger, {})) & set(per_fold.get(incumbent, {})))
    if not shared:
        return {
            "shared_folds": 0,
            "fold_wins": 0,
            "fold_losses": 0,
            "win_ratio": None,
            "max_fold_loss": None,
        }
    wins = 0
    losses = 0
    max_loss = 0.0
    for fold in shared:
        challenger_value = per_fold[challenger][fold]
        incumbent_value = per_fold[incumbent][fold]
        if challenger_value > incumbent_value:
            wins += 1
        elif challenger_value < incumbent_value:
            losses += 1
        max_loss = max(max_loss, incumbent_value - challenger_value)
    return {
        "shared_folds": len(shared),
        "fold_wins": wins,
        "fold_losses": losses,
        "win_ratio": round(wins / len(shared), 6),
        "max_fold_loss": round(max_loss, 6),
    }


def promotion_decision(
    challenger: str,
    incumbent: str,
    aggregate: dict[str, float],
    per_fold: dict[str, dict[int, float]],
    *,
    secondary_aggregate: dict[str, float] | None = None,
    margin: float = PROMOTION_MARGIN,
    fold_win_ratio: float = PROMOTION_FOLD_WIN_RATIO,
    max_fold_loss: float = PROMOTION_MAX_FOLD_LOSS,
) -> dict[str, Any]:
    """Whether *challenger* may replace *incumbent*, with the reasons either way."""
    decision: dict[str, Any] = {
        "candidate": challenger,
        "incumbent": incumbent,
        "thresholds": {
            "margin": margin,
            "fold_win_ratio": fold_win_ratio,
            "max_fold_loss": max_fold_loss,
        },
        "promoted": False,
        "reason": "",
    }
    if challenger not in aggregate or incumbent not in aggregate:
        decision["reason"] = (
            f"{challenger} has no comparable metrics on the primary scheme"
        )
        return decision

    gain = aggregate[challenger] - aggregate[incumbent]
    decision["aggregate_gain"] = round(gain, 6)
    comparison = _fold_comparison(challenger, incumbent, per_fold)
    decision.update(comparison)

    if secondary_aggregate is not None and (
        challenger in secondary_aggregate and incumbent in secondary_aggregate
    ):
        secondary_gap = secondary_aggregate[challenger] - secondary_aggregate[incumbent]
        decision["secondary_gain"] = round(secondary_gap, 6)
        if secondary_gap < -margin:
            decision["reason"] = (
                f"{challenger} is {abs(secondary_gap):.4f} worse than {incumbent} "
                "on the secondary fold scheme"
            )
            return decision

    if gain < margin:
        decision["reason"] = (
            f"aggregate balanced-accuracy gain {gain:+.4f} is below the "
            f"{margin:.2f} promotion margin"
        )
        return decision
    win_ratio = decision.get("win_ratio")
    if win_ratio is None or win_ratio < fold_win_ratio:
        decision["reason"] = (
            f"{challenger} wins only {decision.get('fold_wins', 0)}/"
            f"{decision.get('shared_folds', 0)} shared folds "
            f"(needs {fold_win_ratio:.0%})"
        )
        return decision
    if float(decision.get("max_fold_loss", 0.0)) > max_fold_loss:
        decision["reason"] = (
            f"{challenger} loses one fold by {decision['max_fold_loss']:.4f} "
            f"(limit {max_fold_loss:.2f})"
        )
        return decision
    decision["promoted"] = True
    decision["reason"] = (
        f"{challenger} is stably better: {gain:+.4f} aggregate balanced accuracy "
        f"and {decision['fold_wins']}/{decision['shared_folds']} fold wins"
    )
    return decision


def select_candidate(
    aggregate: dict[str, float],
    per_fold: dict[str, dict[int, float]],
    *,
    secondary_aggregate: dict[str, float] | None = None,
    ladder: tuple[str, ...] = DEPLOYABLE_ORDER,
    publication_model: str = DEFAULT_PUBLICATION_MODEL,
    margin: float = PROMOTION_MARGIN,
    fold_win_ratio: float = PROMOTION_FOLD_WIN_RATIO,
    max_fold_loss: float = PROMOTION_MAX_FOLD_LOSS,
) -> dict[str, Any]:
    """Walk the complexity ladder and keep the simplest model that is not stably beaten."""
    available = [name for name in ladder if name in aggregate]
    if not available:
        return {
            "rule": "keep_simplest_unless_stably_beaten",
            "status": "not_available",
            "selected": None,
            "publication_model": publication_model,
            "publication_consistent": False,
            "reason": "no candidate produced metrics on the primary fold scheme",
            "decisions": [],
        }

    selected = available[0]
    decisions: list[dict[str, Any]] = []
    for challenger in available[1:]:
        decision = promotion_decision(
            challenger, selected, aggregate, per_fold,
            secondary_aggregate=secondary_aggregate,
            margin=margin, fold_win_ratio=fold_win_ratio, max_fold_loss=max_fold_loss,
        )
        decisions.append(decision)
        if decision["promoted"]:
            selected = challenger

    promoted = [decision for decision in decisions if decision["promoted"]]
    if promoted:
        reason = (
            f"{selected} won the ladder: "
            + "; ".join(str(decision["reason"]) for decision in promoted)
        )
    else:
        reason = (
            f"{selected} kept: no more complex candidate was stably better "
            f"(see decisions for the measured gaps)"
        )
    return {
        "rule": "keep_simplest_unless_stably_beaten",
        "status": "selected",
        "selected": selected,
        "candidate_count": len(CANDIDATE_NAMES),
        "ladder": list(ladder),
        "complexity": dict(CANDIDATE_COMPLEXITY),
        "thresholds": {
            "margin": margin,
            "fold_win_ratio": fold_win_ratio,
            "max_fold_loss": max_fold_loss,
        },
        "decisions": decisions,
        "reason": reason,
        "publication_model": publication_model,
        "publication_consistent": selected == publication_model,
        "note": (
            "The winner is the simplest candidate that no more complex "
            "candidate beat stably across folds. The published artifact is "
            "chosen by the activation-policy path "
            "(mindflow.train.pipeline, owned separately); when it disagrees "
            "with `selected`, the report says so instead of hiding it."
        ),
    }


__all__ = [
    "CANDIDATE_COMPLEXITY",
    "CANDIDATE_NAMES",
    "DEFAULT_PUBLICATION_MODEL",
    "DEPLOYABLE_ORDER",
    "LOGISTIC_REGRESSION",
    "PROMOTION_FOLD_WIN_RATIO",
    "PROMOTION_MARGIN",
    "PROMOTION_MAX_FOLD_LOSS",
    "RANDOM_FOREST",
    "RF_XGB_SOFT_VOTING",
    "RULE_ENGINE",
    "XGBOOST",
    "CandidateFit",
    "FoldResult",
    "available_candidate_names",
    "fit_candidate",
    "make_ensemble",
    "promotion_decision",
    "rule_probabilities",
    "select_candidate",
    "session_balanced_weights",
]
