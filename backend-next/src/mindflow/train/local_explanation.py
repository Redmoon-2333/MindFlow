"""Local (per-prediction) attribution for the focus classifier.

Answers "why did the model say *this* window is 73% focus?" — as opposed to
:mod:`mindflow.train.explain`, which ranks features across a whole training
set. The output is the additive decomposition behind one reported
``focus_probability``, and it is deliberately honest about its scale: the
contribution values live in whatever space the method can prove additivity in
(calibrated probability for TreeSHAP, log-odds for the linear fallback), and
``output_space`` says which.

Theory
------
Shapley values (Lundberg & Lee 2017) attribute a model output to its inputs
such that ``base + Σ φ_i`` reproduces that output exactly. TreeSHAP
(Lundberg et al. 2020) computes them in polynomial time for tree ensembles,
with a path-dependent background.

Two transformations break naive additivity, and both are handled by a
**proportional rescale** — an approximation that preserves additivity by
distributing the total change in proportion to each SHAP value:

1. **XGBoost margin → probability.** ``TreeExplainer`` on an
   ``XGBClassifier`` returns SHAP values in log-odds space, while the soft
   vote averages *probabilities*. Each window's values are scaled by
   ``(p_xgb(x) − p_xgb(base)) / (m(x) − m(base))``, the local slope of the
   sigmoid, so they add up to the probability difference instead of the
   margin difference.
2. **Platt / isotonic calibration.** The reported probability is the
   calibrated one. The combined contributions are scaled by
   ``(p_cal(x) − p_cal(base)) / (f(x) − f(base))`` so the sum lands on the
   calibrated output rather than the raw soft vote.

Neither rescale is exact for a non-linear map — that is why the note tells the
user this is an approximation — but both keep the invariant
``Σ contribution ≈ reported_probability − base_value``, which is what makes
the explanation checkable instead of decorative.

The whole module is import-safe without ``shap`` installed: the dependency is
imported lazily and an absent install degrades to ``method="unavailable"``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ExplanationMethod = Literal["tree_shap", "linear", "unavailable"]
OutputSpace = Literal["probability", "logit"]

#: Contract §1 caps the contribution list at 8 entries.
MAX_CONTRIBUTIONS: int = 8

#: Below this difference a rescale denominator is treated as degenerate and
#: the transformation is skipped rather than amplified into noise.
RESCALE_EPSILON: float = 1e-6

#: The one phrase the UI relies on to tell users the decomposition is not
#: exact. Kept as a constant so the contract text and the note cannot drift.
PROPORTIONAL_RESCALE_NOTE: str = "比例重标定（保持可加性的近似）"


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's signed push on the predicted output.

    Attributes:
        feature: V2 feature name.
        value: Mean raw (unscaled) value of the feature over the windows used.
        baseline_value: Personal-baseline expectation for the same local
            (hour, weekday) bucket, or None when no baseline covers it.
        contribution: Signed additive contribution, in ``output_space``.
            Positive pushes the prediction toward "focus".
    """

    feature: str
    value: float
    baseline_value: float | None
    contribution: float


@dataclass(frozen=True)
class LocalExplanation:
    """Additive local explanation of one aggregated focus prediction.

    ``base_value + Σ contributions`` reproduces the model output for the
    aggregated window mean (up to the rescale approximation documented in the
    module docstring). ``contributions`` is sorted by ``|contribution|``
    descending and capped at :data:`MAX_CONTRIBUTIONS`.
    """

    method: ExplanationMethod
    base_value: float | None
    output_space: OutputSpace
    contributions: list[FeatureContribution] = field(default_factory=list)
    note: str = ""


class LocalExplainer:
    """Computes :class:`LocalExplanation` for a fitted focus classifier.

    Supports the two model families the production pipeline loads:

    * Random Forest / RF+XGBoost soft-voting ensemble → ``tree_shap`` in
      probability space;
    * Logistic Regression → ``linear`` in log-odds space.

    Anything else (or a missing ``shap`` install) yields
    ``method="unavailable"`` with no contributions, rather than a plausible
    but unfounded ranking.

    Args:
        classifier: A fitted classifier exposing ``predict_proba`` and
            ``scaler``, e.g. ``EnsembleClassifier`` or ``FocusClassifier``.
    """

    def __init__(self, classifier: Any) -> None:
        self._classifier = classifier
        #: TreeExplainers are built once per LocalExplainer. The prediction
        #: service caches one LocalExplainer per model version, so this cache
        #: is implicitly keyed by model version and cannot serve a stale model.
        self._tree_explainers: dict[str, Any] = {}
        self._shap: Any = None
        try:
            import shap

            self._shap = shap
        except ImportError:
            logger.warning("shap is not installed — local explanations unavailable")

    # ── Public API ────────────────────────────────────────────────────────

    def explain(
        self,
        matrix: npt.NDArray[Any],
        feature_names: list[str],
        *,
        focus_probas: npt.NDArray[Any],
        baseline_values: dict[str, float] | None = None,
    ) -> LocalExplanation:
        """Explain the mean focus probability of *matrix*.

        Args:
            matrix: Raw (unscaled) feature matrix, shape ``(n_windows, n_features)``.
            feature_names: Column names, aligned with *matrix*.
            focus_probas: The calibrated focus probability of each row — the
                same numbers the service averages into ``focus_probability``.
            baseline_values: Optional per-feature personal-baseline expectation
                for the current bucket; missing features report None.

        Returns:
            A :class:`LocalExplanation`. Never raises: an unsupported model or
            a numerical failure degrades to ``method="unavailable"``.
        """
        if self._shap is None:
            return _unavailable("未安装 shap，无法生成局部解释")

        try:
            rf, xgb = _tree_models(self._classifier)
            if rf is not None:
                return self._explain_trees(
                    rf, xgb, matrix, feature_names, focus_probas, baseline_values
                )
            linear = _linear_model(self._classifier)
            if linear is not None:
                return self._explain_linear(linear, matrix, feature_names, baseline_values)
        except Exception as exc:
            # A failed explanation must not take the prediction down with it,
            # and it must not be reported as a successful one.
            logger.opt(exception=True).warning("Local explanation failed: {}", exc)
            return _unavailable(f"局部解释计算失败（{type(exc).__name__}）")

        return _unavailable("当前模型类型不支持局部解释")

    # ── TreeSHAP ──────────────────────────────────────────────────────────

    def _explain_trees(
        self,
        rf: Any,
        xgb: Any,
        matrix: npt.NDArray[Any],
        feature_names: list[str],
        focus_probas: npt.NDArray[Any],
        baseline_values: dict[str, float] | None,
    ) -> LocalExplanation:
        scaled = _scaled(self._classifier, matrix)

        # (1) Random Forest: path-dependent TreeSHAP is already in probability
        # space, so only the class-1 (focus) column is selected. The model's
        # own output for each row is recovered from the decomposition itself
        # (``base + Σφ``) instead of a second ``predict_proba`` pass — the
        # additive invariant makes them identical, and the forest's parallel
        # predict costs more than the SHAP values do.
        rf_explainer = self._tree_explainer("rf", rf)
        rf_phi, rf_base = _shap_output(rf_explainer, scaled)
        phi = np.asarray(rf_phi, dtype=float)
        base_total = rf_base
        n_models = 1

        if xgb is not None:
            # (2) XGBoost: SHAP is in margin (log-odds) space. Rescale each
            # window into probability space so the two models can be averaged.
            # ``margin = margin_base + Σφ_xgb`` by the same invariant.
            xgb_explainer = self._tree_explainer("xgb", xgb)
            xgb_phi, margin_base = _shap_output(xgb_explainer, scaled)
            xgb_phi = np.asarray(xgb_phi, dtype=float)
            margin = margin_base + xgb_phi.sum(axis=1)
            denominator = margin - margin_base
            degenerate = np.abs(denominator) < RESCALE_EPSILON
            slope = np.where(
                degenerate,
                1.0,
                (_sigmoid(margin) - float(_sigmoid(margin_base)))
                / np.where(degenerate, 1.0, denominator),
            )
            phi = phi + xgb_phi * slope[:, None]
            base_total += float(_sigmoid(margin_base))
            n_models = 2

        # (3) Soft vote: the mean over the models that were available, exactly
        # as ``EnsembleClassifier._soft_vote_proba`` combines them.
        phi = phi / n_models
        base_value = base_total / n_models
        output_space: OutputSpace = "probability"
        # The uncalibrated soft-vote probability of each window.
        f_x = base_value + phi.sum(axis=1)

        # (4) Calibration: move the total onto the reported probability scale.
        calibrator = getattr(self._classifier, "calibrator", None)
        if calibrator is not None:
            calibrated = np.asarray(focus_probas, dtype=float)
            p_base = float(self._calibrated(np.asarray([base_value], dtype=float))[0])
            denominator = f_x - base_value
            slope = np.where(
                np.abs(denominator) < RESCALE_EPSILON,
                1.0,
                (calibrated - p_base)
                / np.where(np.abs(denominator) < RESCALE_EPSILON, 1.0, denominator),
            )
            phi = phi * slope[:, None]
            base_value = p_base

        note = (
            "基于 TreeSHAP 的局部贡献（概率尺度，正贡献推向“专注”）。"
            "XGBoost 的对数几率贡献与 Platt 概率校准均按 "
            f"{PROPORTIONAL_RESCALE_NOTE} 处理，因此各项之和与"
            "报告概率之间存在极小的近似误差。"
        )
        if xgb is None:
            note = (
                "基于 TreeSHAP 的局部贡献（概率尺度，正贡献推向“专注”）。"
                "当前模型仅含随机森林，无 XGBoost 与概率校准环节。"
            )
        elif calibrator is None:
            note = (
                "基于 TreeSHAP 的局部贡献（概率尺度，正贡献推向“专注”）。"
                "XGBoost 的对数几率贡献按 "
                f"{PROPORTIONAL_RESCALE_NOTE} 处理；该模型未做概率校准。"
            )

        return LocalExplanation(
            method="tree_shap",
            base_value=round(base_value, 6),
            output_space=output_space,
            contributions=_rank(phi, matrix, feature_names, baseline_values),
            note=note,
        )

    # ── Linear ────────────────────────────────────────────────────────────

    def _explain_linear(
        self,
        model: LogisticRegression,
        matrix: npt.NDArray[Any],
        feature_names: list[str],
        baseline_values: dict[str, float] | None,
    ) -> LocalExplanation:
        scaled = _scaled(self._classifier, matrix)
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        intercept = float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0])
        # A logistic regression is exactly additive in log-odds: intercept plus
        # coefficient × standardised value reproduces the logit, so no rescale
        # is needed and none is applied.
        phi = scaled * coefficients
        return LocalExplanation(
            method="linear",
            base_value=round(intercept, 6),
            output_space="logit",
            contributions=_rank(phi, matrix, feature_names, baseline_values),
            note=(
                "基于线性模型的局部贡献（对数几率尺度，正贡献推向“专注”）。"
                "贡献 = 系数 × 标准化特征值，基准值为模型截距，二者严格可加。"
            ),
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _tree_explainer(self, key: str, model: Any) -> Any:
        """Memoised ``TreeExplainer`` for *model* (construction is ~10–20 ms)."""
        explainer = self._tree_explainers.get(key)
        if explainer is None:
            explainer = self._shap.TreeExplainer(model)
            self._tree_explainers[key] = explainer
        return explainer

    def _calibrated(self, raw_probability: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply the classifier's own calibrator to raw soft-vote probabilities.

        Delegates to ``_apply_calibration`` so the explanation cannot drift
        from ``predict_proba`` (sigmoid vs isotonic, and its clipping).
        """
        raw = np.asarray(raw_probability, dtype=float)
        stacked = np.column_stack([1.0 - raw, raw])
        return np.asarray(self._classifier._apply_calibration(stacked), dtype=float)[:, 1]


# ── Module helpers ────────────────────────────────────────────────────────


def _unavailable(reason: str) -> LocalExplanation:
    """An honest "no explanation" result."""
    return LocalExplanation(
        method="unavailable",
        base_value=None,
        output_space="probability",
        contributions=[],
        note=reason,
    )


def _tree_models(classifier: Any) -> tuple[Any, Any]:
    """Return ``(rf_model, xgb_model_or_None)`` for a supported classifier.

    Both are validated by type, so a mock or an unexpected estimator yields
    ``(None, None)`` and the caller falls through to the linear path instead
    of handing a MagicMock to SHAP.
    """
    rf = getattr(classifier, "rf_model", None)
    if rf is None:
        model = getattr(classifier, "model", None)
        if isinstance(model, RandomForestClassifier):
            rf = model
    if not isinstance(rf, RandomForestClassifier):
        return None, None

    xgb = getattr(classifier, "xgb_model", None)
    if xgb is not None and not _is_xgb_classifier(xgb):
        xgb = None
    return rf, xgb


def _is_xgb_classifier(model: Any) -> bool:
    """True when *model* is an xgboost classifier (False if xgboost is absent)."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return False
    return isinstance(model, XGBClassifier)


def _linear_model(classifier: Any) -> LogisticRegression | None:
    """The classifier's logistic-regression estimator, if that is what it is."""
    model = getattr(classifier, "model", classifier)
    return model if isinstance(model, LogisticRegression) else None


def _scaled(classifier: Any, matrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Apply the classifier's scaler, matching ``predict_proba``'s input path."""
    scaler = getattr(classifier, "scaler", None)
    if scaler is None:
        return np.asarray(matrix, dtype=float)
    return np.asarray(scaler.transform(matrix), dtype=float)


def _shap_output(explainer: Any, scaled: npt.NDArray[Any]) -> tuple[Any, float]:
    """SHAP values and expected value for the positive (focus) class.

    Normalises the API shapes ``shap`` has used across versions: a list of
    per-class arrays, ``(n, f, 2)`` for binary tree models, or plain
    ``(n, f)`` for a single-output model such as an XGBClassifier booster.
    """
    raw = np.asarray(explainer.shap_values(scaled))
    expected = np.asarray(explainer.expected_value, dtype=float).reshape(-1)
    if raw.ndim == 3:
        # (n, f, n_classes) — column 1 is "focus".
        values = raw[:, :, 1]
        base = float(expected[1]) if expected.size > 1 else float(expected[0])
    else:
        values = raw
        base = float(expected[0])
    return values, base


def _sigmoid(x: npt.NDArray[Any] | float) -> npt.NDArray[Any]:
    """Numerically stable logistic function."""
    arr = np.asarray(x, dtype=float)
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(arr, -500.0, 500.0))), dtype=float)


def _rank(
    phi: npt.NDArray[Any],
    matrix: npt.NDArray[Any],
    feature_names: list[str],
    baseline_values: dict[str, float] | None,
) -> list[FeatureContribution]:
    """Average per-window SHAP values and rank them for the contract.

    The reported ``value`` is the mean *raw* feature value, so the number in
    the payload matches what the user sees elsewhere; the contribution stays
    on the model's output scale.
    """
    mean_phi = np.asarray(phi, dtype=float).mean(axis=0)
    mean_values = np.asarray(matrix, dtype=float).mean(axis=0)
    ranked = sorted(
        range(len(feature_names)),
        key=lambda index: abs(float(mean_phi[index])),
        reverse=True,
    )[:MAX_CONTRIBUTIONS]
    return [
        FeatureContribution(
            feature=feature_names[index],
            value=round(float(mean_values[index]), 6),
            baseline_value=(
                round(float(baseline_values[feature_names[index]]), 6)
                if baseline_values is not None
                and feature_names[index] in baseline_values
                else None
            ),
            contribution=round(float(mean_phi[index]), 6),
        )
        for index in ranked
        if math.isfinite(float(mean_phi[index]))
    ]
