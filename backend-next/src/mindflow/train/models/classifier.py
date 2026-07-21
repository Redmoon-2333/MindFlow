"""Random Forest focus/distraction classifier."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class FocusClassifier:
    """Random Forest classifier for focus vs distraction.

    Retained for pipeline completeness; in production the weak-supervision
    ``ConsensusLabeler`` is the primary labeling mechanism.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.feature_names_: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        feature_names: list[str],
        sample_weight: npt.NDArray[Any] | None = None,
    ) -> FocusClassifier:
        """Train the classifier.

        Args:
            X: feature matrix of shape (n_samples, n_features)
            y: binary labels (1=focus, 0=distraction)
            feature_names: names for each feature column
            sample_weight: per-sample confidence weights

        Returns:
            self
        """
        self.feature_names_ = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict class labels (1=focus, 0=distraction)."""
        X_scaled = self.scaler.transform(X)
        return np.asarray(self.model.predict(X_scaled))

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return np.asarray(self.model.predict_proba(X_scaled))

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        if not self._is_fitted:
            return {}
        return {
            name: round(float(imp), 6)
            for name, imp in zip(
                self.feature_names_, self.model.feature_importances_, strict=True
            )
        }

    def evaluate(self, X_test: npt.NDArray[Any], y_test: npt.NDArray[Any]) -> dict[str, Any]:
        """Evaluate model performance.

        Returns dict with: accuracy, precision, recall, f1, cv_mean, cv_std.
        """
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)

        cv_scores = cross_val_score(self.model, X_scaled, y_test, cv=5)

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names_,
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FocusClassifier:
        instance = cls()
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names_ = list(data.get("feature_names", []))
        instance._is_fitted = bool(data.get("is_fitted", False))
        return instance
