"""SHAP-based model explainability for behavior pattern interpretation.

Provides ``ModelExplainer``, which wraps a fitted FocusClassifier and produces
SHAP-based explanations of which behavioral features most influence
focus/distraction predictions.  Matches the XAI methodology described in
"From Prediction to Causation: XAI with ML for Procrastination" (2025).

Uses TreeExplainer for Random Forest models (the default), with KernelExplainer
as a fallback.  All shap imports are lazy — the module is importable even when
shap is not installed.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt


class ModelExplainer:
    """SHAP-based model explainability for behavior pattern interpretation.

    Takes a fitted classifier (e.g. ``FocusClassifier``) and a list of
    feature names, then computes SHAP values to explain which behavioral
    features drive focus/distraction classifications.

    If shap is not installed, all methods return a graceful degradation
    dict with ``"note": "SHAP not installed."`` instead of raising.

    Args:
        classifier: A fitted classifier instance.  Must expose ``.model``
            (the underlying sklearn estimator) and optionally ``.scaler``
            (a fitted ``StandardScaler`` for input preprocessing).
        feature_names: Ordered list of feature column names.
    """

    def __init__(self, classifier: Any, feature_names: list[str]) -> None:
        self._classifier = classifier
        self.feature_names = feature_names
        self._available = False
        self._explainer: Any = None
        self._shap: Any = None

        try:
            import shap  # noqa: F811

            self._shap = shap
            self._available = True
        except ImportError:
            self._available = False

    # ── Public API ────────────────────────────────────────────────────────

    def explain(
        self, X: npt.NDArray[Any], max_samples: int = 100
    ) -> dict[str, Any]:
        """Compute SHAP values for a feature matrix and return structured explanations.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            max_samples: Maximum number of samples to pass to SHAP (larger
                matrices are randomly subsampled for performance).

        Returns:
            A dict with:
            - ``feature_importance``: list of ``{name, importance}`` sorted desc.
            - ``summary``: human-readable summary string.
        """
        if not self._available:
            return {
                "feature_importance": [],
                "summary": "SHAP not installed. Install with: pip install shap",
            }

        # Subsample if needed
        if len(X) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        X_proc = self._scale(X_sample)
        sv = self._compute_shap_values(X_proc)
        if sv is None:
            return {
                "feature_importance": [],
                "summary": "SHAP computation failed (model not fitted?).",
            }

        # Per-feature mean absolute SHAP → importance ranking
        importance = np.abs(sv).mean(axis=0)
        ranked = sorted(
            (
                {
                    "name": self.feature_names[i],
                    "importance": round(float(importance[i]), 6),
                }
                for i in range(len(self.feature_names))
            ),
            key=lambda x: cast(float, x["importance"]),
            reverse=True,
        )

        top3 = ranked[:3]
        summary = (
            "Top features driving behavior classification: "
            + ", ".join(f"{f['name']} ({f['importance']:.4f})" for f in top3)
        )

        return {
            "feature_importance": ranked,
            "summary": summary,
        }

    def explain_sample(self, features: npt.NDArray[Any]) -> dict[str, Any]:
        """Explain a single sample's prediction.

        Args:
            features: 1-D feature vector of shape ``(n_features,)`` or
                2-D array of shape ``(1, n_features)``.

        Returns:
            A dict with:
            - ``prediction``: predicted class (1=focus, 0=distraction).
            - ``confidence``: model confidence for the predicted class.
            - ``top_factors``: list of ``{feature, impact, magnitude}`` for
              the top-3 most influential features.
        """
        if not self._available:
            return {
                "prediction": -1,
                "confidence": 0.0,
                "top_factors": [],
                "note": "SHAP not installed. Install with: pip install shap",
            }

        features_2d = (
            features.reshape(1, -1) if features.ndim == 1 else features
        )

        X_proc = self._scale(features_2d)
        model = self._get_model()

        try:
            pred = int(model.predict(X_proc)[0])
            proba = model.predict_proba(X_proc)[0]
            confidence = round(float(max(proba)), 4)
        except Exception:
            return {
                "prediction": -1,
                "confidence": 0.0,
                "top_factors": [],
                "note": "Prediction failed (model not fitted?).",
            }

        sv = self._compute_shap_values(X_proc)
        if sv is None or sv.shape[0] == 0:
            return {
                "prediction": pred,
                "confidence": confidence,
                "top_factors": [],
                "note": "SHAP computation failed.",
            }

        sample_sv = sv[0]
        abs_sv = np.abs(sample_sv)
        top_indices = np.argsort(abs_sv)[-3:][::-1]

        top_factors: list[dict[str, Any]] = []
        for idx in top_indices:
            top_factors.append({
                "feature": self.feature_names[idx],
                "impact": "positive" if sample_sv[idx] > 0 else "negative",
                "magnitude": round(float(abs_sv[idx]), 6),
            })

        return {
            "prediction": pred,
            "confidence": confidence,
            "top_factors": top_factors,
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _scale(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply the classifier's scaler if available, else return X unchanged."""
        scaler = getattr(self._classifier, "scaler", None)
        if scaler is not None:
            return np.asarray(scaler.transform(X))
        return X

    def _get_model(self) -> Any:
        """Return the underlying sklearn estimator."""
        return getattr(self._classifier, "model", self._classifier)

    def _compute_shap_values(
        self, X_proc: npt.NDArray[Any]
    ) -> npt.NDArray[Any] | None:
        """Compute SHAP values for preprocessed input.

        Uses TreeExplainer for tree-based models; falls back to a coarser
        KernelExplainer when the model is not tree-based.
        """
        if self._shap is None:
            return None

        model = self._get_model()

        try:
            # TreeExplainer works on any sklearn tree ensemble
            explainer = self._shap.TreeExplainer(model)
            raw = explainer.shap_values(X_proc)
        except Exception:
            # KernelExplainer fallback (slower, approximate)
            try:
                background = X_proc[: min(50, len(X_proc))]
                if len(background) == 0:
                    return None

                def _predict_fn(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
                    return cast(npt.NDArray[Any], np.asarray(model.predict_proba(x)[:, 1]))

                explainer = self._shap.KernelExplainer(
                    _predict_fn,
                    background,
                )
                raw = explainer.shap_values(X_proc)
            except Exception:
                return None

        # Normalise across SHAP API versions:
        # - Old API (list of arrays): raw = [neg_class, pos_class]
        # - New API binary:       (n_samples, n_features)
        # - New API multiclass:   (n_samples, n_features, n_classes)
        if isinstance(raw, list):
            return np.asarray(raw[1])
        if raw.ndim == 3:
            return cast(npt.NDArray[Any], raw[:, :, 1])  # class 1 = focus
        return cast(npt.NDArray[Any], np.asarray(raw))
