"""Ensemble classifier combining Random Forest and XGBoost with soft voting.

Matches the ensemble approach validated in IEEE 2026 procrastination paper:
RF (n_estimators=100, max_depth=10) + XGBoost (n_estimators=100, max_depth=6,
learning_rate=0.1) with soft voting = averaging predicted probabilities.

XGBoost is optional — falls back to RF-only when not installed.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_XGB_CLASS_MARKER = "EnsembleClassifier"

try:
    from xgboost import XGBClassifier  # noqa: W0611 is handled by the flag below

    _xgb_available = True
except ImportError:
    _xgb_available = False


class EnsembleClassifier:
    """Random Forest + XGBoost ensemble with soft voting.

    Same public API as ``FocusClassifier`` so ``ModelManager`` can use them
    interchangeably. When xgboost is unavailable, degrades gracefully to
    RF-only mode; ``predict`` and ``predict_proba`` still work correctly.
    """

    _RF_PARAMS: dict[str, Any] = {
        "n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1
    }
    _XGB_PARAMS: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "binary:logistic",
        "random_state": 42,
        "verbosity": 0,
    }

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(**self._RF_PARAMS)
        self.xgb_model: Any = None
        self._xgb_available = _xgb_available

        if self._xgb_available:
            self.xgb_model = XGBClassifier(**self._XGB_PARAMS)
        else:
            logger.info("xgboost not installed — ensemble will use RF only")

        self.feature_names_: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        feature_names: list[str],
        sample_weight: npt.NDArray[Any] | None = None,
    ) -> EnsembleClassifier:
        """Train both RF and XGBoost on scaled data.

        Args:
            X: feature matrix of shape (n_samples, n_features).
            y: binary labels (1=focus, 0=distraction).
            feature_names: names for each feature column.
            sample_weight: per-sample confidence weights.

        Returns:
            self
        """
        self.feature_names_ = feature_names
        X_scaled = self.scaler.fit_transform(X)

        rf_sw = sample_weight if sample_weight is not None else None
        self.rf_model.fit(X_scaled, y, sample_weight=rf_sw)

        if self._xgb_available and self.xgb_model is not None:
            xgb_sw = sample_weight if sample_weight is not None else None
            self.xgb_model.fit(X_scaled, y, sample_weight=xgb_sw)

        self._is_fitted = True
        return self

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Soft-vote class labels (1=focus, 0=distraction).

        Averages RF and XGBoost predicted probabilities, then argmax.
        Falls back to RF-only when xgboost is unavailable.
        """
        proba = self.predict_proba(X)
        return np.asarray(proba.argmax(axis=1))

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Soft-vote class probabilities.

        Returns the element-wise mean of RF and XGBoost probability arrays
        when xgboost is available, otherwise RF-only probabilities.
        """
        X_scaled = self.scaler.transform(X)
        rf_proba = np.asarray(self.rf_model.predict_proba(X_scaled))

        if self._xgb_available and self.xgb_model is not None:
            xgb_proba = np.asarray(self.xgb_model.predict_proba(X_scaled))
            return cast(npt.NDArray[Any], (rf_proba + xgb_proba) / 2.0)

        return rf_proba

    def get_feature_importance(self) -> dict[str, Any]:
        """Return feature importance scores.

        Returns:
            Dict mapping feature names to their RF importance (float),
            plus an ``"xgboost"`` sub-dict with XGBoost feature importance
            when available.
        """
        if not self._is_fitted:
            return {}

        importance: dict[str, Any] = {
            name: round(float(imp), 6)
            for name, imp in zip(
                self.feature_names_, self.rf_model.feature_importances_, strict=True
            )
        }

        if self._xgb_available and self.xgb_model is not None:
            importance["xgboost"] = {
                name: round(float(imp), 6)
                for name, imp in zip(
                    self.feature_names_,
                    self.xgb_model.feature_importances_,
                    strict=True,
                )
            }

        return importance

    def evaluate(self, X_test: npt.NDArray[Any], y_test: npt.NDArray[Any]) -> dict[str, Any]:
        """Evaluate ensemble performance.

        Returns dict with: accuracy, precision, recall, f1, cv_mean, cv_std.
        Cross-validation uses the RF model (stable scorer, matches
        FocusClassifier behaviour).
        """
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.predict(X_test)

        class_counts = np.bincount(np.asarray(y_test, dtype=np.int32), minlength=2)
        nonzero_counts = class_counts[class_counts > 0]
        cv_splits = min(5, len(y_test), int(nonzero_counts.min(initial=0)))
        if cv_splits >= 2:
            cv_scores = cross_val_score(self.rf_model, X_scaled, y_test, cv=cv_splits)
            cv_mean = round(float(cv_scores.mean()), 4)
            cv_std = round(float(cv_scores.std()), 4)
        else:
            cv_mean = 0.0
            cv_std = 0.0

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize both models and metadata.

        The ``__class__`` marker lets ``ModelManager._load_versions()``
        dispatch to the correct ``from_dict`` class method.
        """
        data: dict[str, Any] = {
            "__class__": _XGB_CLASS_MARKER,
            "rf_model": self.rf_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names_,
            "is_fitted": self._is_fitted,
            "xgb_available": self._xgb_available,
        }

        if self._xgb_available and self.xgb_model is not None:
            data["xgb_model"] = self.xgb_model
        else:
            data["xgb_model"] = None

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnsembleClassifier:
        """Deserialize an ensemble classifier.

        Works for both RF+XGBoost and RF-only serialized states.
        """
        instance = cls()
        instance.rf_model = data["rf_model"]
        instance.scaler = data["scaler"]
        instance.feature_names_ = list(data.get("feature_names", []))
        instance._is_fitted = bool(data.get("is_fitted", False))
        instance._xgb_available = bool(data.get("xgb_available", _xgb_available))
        instance.xgb_model = data.get("xgb_model")

        return instance
