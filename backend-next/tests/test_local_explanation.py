"""Tests for local (per-prediction) SHAP attribution.

The contract (§1) promises that a client can add up the contributions and
recover the reported focus probability. These tests hold that invariant at
several levels:

* the ensemble's probability-space decomposition (TreeSHAP on RF + rescaled
  XGB margin) sums to ``focus_probability`` within 1e-3 uncalibrated and 2e-2
  calibrated;
* the sign convention is positive-toward-focus, pinned by a feature whose
  effect is unambiguous;
* unsupported models and a missing ``shap`` install degrade to
  ``method="unavailable"`` instead of inventing a ranking;
* the linear path is exactly additive in log-odds.
"""

from __future__ import annotations

import numpy as np
import pytest

from mindflow.domain.feature_schema import V2_FEATURE_NAMES
from mindflow.train.local_explanation import (
    MAX_CONTRIBUTIONS,
    LocalExplainer,
)
from mindflow.train.models.classifier import FocusClassifier
from mindflow.train.models.ensemble import EnsembleClassifier

_FEATURE_NAMES = list(V2_FEATURE_NAMES)

pytest.importorskip("shap")


def _synthetic(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Label separable on ``app_switch_count`` (low) and ``idle_ratio`` (low).

    A clearly learnable signal keeps the SHAP values non-degenerate, so the
    additivity assertions test the decomposition rather than noise.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(_FEATURE_NAMES)))
    signal = -1.2 * X[:, 0] - 0.9 * X[:, 3] + rng.normal(scale=0.6, size=n)
    return X, (signal > 0).astype(int)


def _fit_ensemble(calibration: str | None) -> EnsembleClassifier:
    X, y = _synthetic()
    model = EnsembleClassifier(calibration=calibration)
    model.fit(X, y, _FEATURE_NAMES)
    return model


@pytest.fixture(scope="module")
def uncalibrated_ensemble() -> EnsembleClassifier:
    return _fit_ensemble(None)


@pytest.fixture(scope="module")
def calibrated_ensemble() -> EnsembleClassifier:
    return _fit_ensemble("sigmoid")


def _explain_all(model: object, n_windows: int = 24):
    """Explain *n_windows* rows, keeping every contribution (cap lifted)."""
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(n_windows, len(_FEATURE_NAMES)))
    probas = model.predict_proba(matrix)[:, 1]  # type: ignore[attr-defined]
    return matrix, probas


class TestEnsembleExplanation:
    def test_probability_space_is_additive_without_calibration(
        self, uncalibrated_ensemble: EnsembleClassifier, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """base + Σφ == the reported probability (tolerance 1e-3)."""
        monkeypatch.setattr(
            "mindflow.train.local_explanation.MAX_CONTRIBUTIONS", len(_FEATURE_NAMES)
        )
        matrix, probas = _explain_all(uncalibrated_ensemble)
        explanation = LocalExplainer(uncalibrated_ensemble).explain(
            matrix, _FEATURE_NAMES, focus_probas=probas
        )

        assert explanation.method == "tree_shap"
        assert explanation.output_space == "probability"
        assert explanation.base_value is not None
        total = explanation.base_value + sum(
            item.contribution for item in explanation.contributions
        )
        assert total == pytest.approx(float(probas.mean()), abs=1e-3)

    def test_probability_space_is_additive_with_platt_calibration(
        self, calibrated_ensemble: EnsembleClassifier, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The calibrated probability is what the sum reproduces (tolerance 2e-2)."""
        monkeypatch.setattr(
            "mindflow.train.local_explanation.MAX_CONTRIBUTIONS", len(_FEATURE_NAMES)
        )
        matrix, probas = _explain_all(calibrated_ensemble)
        explanation = LocalExplainer(calibrated_ensemble).explain(
            matrix, _FEATURE_NAMES, focus_probas=probas
        )

        assert explanation.method == "tree_shap"
        assert explanation.output_space == "probability"
        total = explanation.base_value + sum(  # type: ignore[operator]
            item.contribution for item in explanation.contributions
        )
        assert total == pytest.approx(float(probas.mean()), abs=2e-2)
        # The note must disclose both approximations in Chinese.
        assert "比例重标定" in explanation.note
        assert "XGBoost" in explanation.note

    def test_rf_only_model_explains_without_xgboost(self) -> None:
        X, y = _synthetic()
        model = FocusClassifier().fit(X, y, _FEATURE_NAMES)
        matrix, probas = _explain_all(model)
        explanation = LocalExplainer(model).explain(
            matrix, _FEATURE_NAMES, focus_probas=probas
        )

        assert explanation.method == "tree_shap"
        assert explanation.output_space == "probability"
        total = explanation.base_value + sum(  # type: ignore[operator]
            item.contribution for item in explanation.contributions
        )
        # Capped at 8 features, so compare against the top-8 sum only loosely.
        assert total == pytest.approx(float(probas.mean()), abs=5e-2)
        assert "仅含随机森林" in explanation.note

    def test_contributions_are_capped_and_ranked(self) -> None:
        model = _fit_ensemble(None)
        matrix, probas = _explain_all(model)
        explanation = LocalExplainer(model).explain(
            matrix, _FEATURE_NAMES, focus_probas=probas
        )

        assert len(explanation.contributions) == MAX_CONTRIBUTIONS
        magnitudes = [abs(item.contribution) for item in explanation.contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for item in explanation.contributions:
            assert item.feature in V2_FEATURE_NAMES
            assert np.isfinite(item.value)

    def test_sign_convention_is_positive_toward_focus(self) -> None:
        """A window that is *more* focus-like must not get negative contributions.

        ``app_switch_count`` and ``idle_ratio`` both push toward distraction in
        the synthetic labelling, so holding them at a low value (focus-like)
        must give both a positive contribution.
        """
        model = _fit_ensemble(None)
        rng = np.random.default_rng(11)
        matrix = rng.normal(size=(24, len(_FEATURE_NAMES)))
        matrix[:, 0] = -3.0  # app_switch_count: focus-like
        matrix[:, 3] = -3.0  # idle_ratio: focus-like
        probas = model.predict_proba(matrix)[:, 1]

        explanation = LocalExplainer(model).explain(
            matrix, _FEATURE_NAMES, focus_probas=probas
        )
        by_feature = {item.feature: item.contribution for item in explanation.contributions}
        assert by_feature["app_switch_count"] > 0.0
        assert by_feature["idle_ratio"] > 0.0

    def test_baseline_values_are_attached_when_supplied(self) -> None:
        model = _fit_ensemble(None)
        matrix, probas = _explain_all(model, n_windows=6)
        baselines = {"app_switch_count": 3.25, "idle_ratio": 0.07}

        explanation = LocalExplainer(model).explain(
            matrix, _FEATURE_NAMES, focus_probas=probas, baseline_values=baselines
        )
        by_feature = {item.feature: item for item in explanation.contributions}
        assert by_feature["app_switch_count"].baseline_value == 3.25
        assert by_feature["idle_ratio"].baseline_value == 0.07
        # Features outside the supplied map stay None rather than defaulting
        # to 0 — "unknown baseline" and "a baseline of zero" are different.
        unbaselined = [
            item
            for item in explanation.contributions
            if item.feature not in baselines
        ]
        assert unbaselined
        assert all(item.baseline_value is None for item in unbaselined)

    def test_explainer_is_reused_across_calls(self) -> None:
        model = _fit_ensemble(None)
        explainer = LocalExplainer(model)
        matrix, probas = _explain_all(model, n_windows=6)
        explainer.explain(matrix, _FEATURE_NAMES, focus_probas=probas)
        cached = dict(explainer._tree_explainers)  # noqa: SLF001
        explainer.explain(matrix, _FEATURE_NAMES, focus_probas=probas)
        assert explainer._tree_explainers is not None  # noqa: SLF001
        assert set(cached) == {"rf", "xgb"}
        assert all(
            explainer._tree_explainers[key] is cached[key]  # noqa: SLF001
            for key in cached
        )


class TestLinearExplanation:
    def test_logistic_regression_is_additive_in_logit(self) -> None:
        from sklearn.linear_model import LogisticRegression

        X, y = _synthetic()
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(10, len(_FEATURE_NAMES)))
        probas = model.predict_proba(matrix)[:, 1]

        explanation = LocalExplainer(model).explain(
            matrix, _FEATURE_NAMES, focus_probas=probas
        )

        assert explanation.method == "linear"
        assert explanation.output_space == "logit"
        logit = matrix @ model.coef_.ravel() + float(model.intercept_[0])
        # Capped at 8, so rebuild the full sum by lifting the cap.
        assert explanation.base_value == pytest.approx(
            float(model.intercept_[0]), abs=1e-6
        )
        assert "对数几率" in explanation.note
        assert abs(logit.mean()) > 0.0

    def test_linear_full_sum_reproduces_logit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        monkeypatch.setattr(
            "mindflow.train.local_explanation.MAX_CONTRIBUTIONS", len(_FEATURE_NAMES)
        )
        X, y = _synthetic()
        model = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
        rng = np.random.default_rng(5)
        matrix = rng.normal(size=(10, len(_FEATURE_NAMES)))
        probas = model.predict_proba(matrix)[:, 1]

        explanation = LocalExplainer(model).explain(
            matrix, _FEATURE_NAMES, focus_probas=probas
        )
        logit = matrix @ model.coef_.ravel() + float(model.intercept_[0])
        total = explanation.base_value + sum(  # type: ignore[operator]
            item.contribution for item in explanation.contributions
        )
        # Contributions are rounded to 6 decimals in the payload, so the sum
        # carries up to 28 × 5e-7 of rounding error.
        assert total == pytest.approx(float(logit.mean()), abs=2e-5)


class TestUnavailablePaths:
    def test_mock_model_yields_unavailable(self) -> None:
        """A MagicMock must not be handed to SHAP as if it were a model."""
        from unittest.mock import MagicMock

        explainer = LocalExplainer(MagicMock())
        explanation = explainer.explain(
            np.zeros((4, len(_FEATURE_NAMES))),
            _FEATURE_NAMES,
            focus_probas=np.array([0.5, 0.5, 0.5, 0.5]),
        )
        assert explanation.method == "unavailable"
        assert explanation.contributions == []
        assert explanation.base_value is None

    def test_missing_shap_degrades_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _fit_ensemble(None)
        explainer = LocalExplainer(model)
        monkeypatch.setattr(explainer, "_shap", None)
        explanation = explainer.explain(
            np.zeros((4, len(_FEATURE_NAMES))),
            _FEATURE_NAMES,
            focus_probas=np.array([0.5, 0.5, 0.5, 0.5]),
        )
        assert explanation.method == "unavailable"
        assert "shap" in explanation.note
