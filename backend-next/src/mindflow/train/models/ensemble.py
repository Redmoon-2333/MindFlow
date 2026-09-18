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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from mindflow.train.grouping import session_balanced_weights

logger = logging.getLogger(__name__)

_XGB_CLASS_MARKER = "EnsembleClassifier"

try:
    from xgboost import XGBClassifier  # noqa: W0611 is handled by the flag below

    _xgb_available = True
except ImportError:
    _xgb_available = False


class CalibrationUnavailableError(ValueError):
    """Requested out-of-sample calibration could not be fitted safely."""


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

    def __init__(self, calibration: str | None = None) -> None:
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(**self._RF_PARAMS)
        self.xgb_model: Any = None
        self._xgb_available = _xgb_available

        if self._xgb_available:
            self.xgb_model = XGBClassifier(**self._XGB_PARAMS)
        else:
            logger.info("xgboost not installed — ensemble will use RF only")

        # Post-hoc probability calibration.
        #   None     (default) — legacy raw soft-vote probabilities.
        #   "sigmoid"          — Platt scaling (1-D logistic).
        #   "isotonic"         — isotonic regression.
        # Production observations showed tree ensembles are overconfident at
        # the extremes (0.9-1.0 confidence bin -> ~3% actual positives), but
        # measurements on BOTH a real-data replay (2026-08-20) and the clean
        # synthetic eval show calibration degrades Brier/Balanced-Accuracy at
        # the current small data sizes, so it stays OFF by default and is an
        # explicit opt-in for once training data is large/clean enough.
        self.calibration: str | None = calibration if calibration else None
        self.calibrator: Any = None
        self.feature_names_: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        feature_names: list[str],
        sample_weight: npt.NDArray[Any] | None = None,
        groups: npt.NDArray[Any] | None = None,
        *,
        label_sources: npt.NDArray[Any] | None = None,
        session_ids: npt.NDArray[Any] | None = None,
    ) -> EnsembleClassifier:
        """Train both RF and XGBoost on scaled data, then calibrate.

        A grouped (or, without groups, stratified) 25% holdout is kept out of
        the base models and used to fit the probability calibrator out-of-sample.

        The scaler is fitted on the base-model training split only; fitting it
        on all rows would leak the holdout's mean/variance into the model that
        is later evaluated on that holdout.

        Args:
            X: feature matrix of shape (n_samples, n_features).
            y: binary labels (1=focus, 0=distraction).
            feature_names: names for each feature column.
            sample_weight: per-sample confidence weights.
            groups: optional group id per row (e.g. the capture date). When
                given, the calibration holdout is split by group so rows from
                one day cannot sit in both the base-model and calibrator sets.
            label_sources: optional aligned provenance. Together with session_ids,
                recomputes each internal split's weights instead of slicing the
                caller's full-data budget. Without provenance, sample_weight
                keeps its generic per-row meaning.
            session_ids: feedback session id per row; empty for auxiliary rows.

        Returns:
            self

        Raises:
            CalibrationUnavailableError: requested calibration cannot be fitted.
        """
        self._is_fitted = False
        self.calibrator = None
        self.feature_names_ = feature_names
        y_arr = np.asarray(y)
        if (label_sources is None) != (session_ids is None):
            raise ValueError("label_sources and session_ids must be supplied together")
        if label_sources is not None and session_ids is not None:
            label_sources, session_ids = np.asarray(label_sources), np.asarray(session_ids)
            if (
                label_sources.ndim != 1 or session_ids.ndim != 1
                or len(label_sources) != len(y_arr) or len(session_ids) != len(y_arr)
            ):
                raise ValueError("Training provenance must align with rows")
            if np.any((label_sources == "explicit") & (session_ids == "")):
                raise ValueError("Explicit training rows need feedback session ids")
            if self.calibration is not None and groups is None:
                raise CalibrationUnavailableError("Calibration with provenance requires groups")

        # Honest holdout for calibration (only when we have enough of both
        # classes to both train and calibrate).
        calib_ix: npt.NDArray[Any] | None = None
        train_ix: npt.NDArray[Any] | None = None
        if self.calibration is not None:
            if self.calibration not in ("isotonic", "sigmoid"):
                raise CalibrationUnavailableError(f"Unsupported calibration: {self.calibration}")
            if len(y_arr) < 20 or len(np.unique(y_arr)) != 2:
                raise CalibrationUnavailableError(
                    "Calibration needs at least 20 rows and both classes"
                )
            try:
                train_ix, calib_ix = self._calibration_split(X, y_arr, groups)
            except Exception as exc:
                raise CalibrationUnavailableError(f"Calibration split unavailable: {exc}") from exc

        def subset_weights(indices: npt.NDArray[Any]) -> npt.NDArray[Any] | None:
            if label_sources is None or session_ids is None:
                return sample_weight[indices] if sample_weight is not None else None
            weights = np.asarray(session_balanced_weights(
                sources=label_sources[indices].tolist(),
                session_ids=session_ids[indices].tolist(),
                allow_auxiliary_only=calib_ix is None,
            ), dtype=np.float64)
            if calib_ix is not None and any(
                weights[y_arr[indices] == label].sum() <= 0 for label in (0, 1)
            ):
                raise CalibrationUnavailableError(
                    "Calibration split needs positive local supervision weight for both classes"
                )
            return weights

        base_ix = train_ix if train_ix is not None else np.arange(len(y_arr))
        swt = subset_weights(base_ix)
        swc = subset_weights(calib_ix) if calib_ix is not None else None

        # Fit the scaler on the base-model training rows only.
        if train_ix is not None:
            self.scaler.fit(X[train_ix])
        else:
            self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        if train_ix is not None:
            Xt = X_scaled[train_ix]
            yt = y_arr[train_ix]
        else:
            Xt, yt = X_scaled, y_arr

        self.rf_model.fit(Xt, yt, sample_weight=swt)

        if self._xgb_available and self.xgb_model is not None:
            self.xgb_model.fit(Xt, yt, sample_weight=swt)

        # Fit calibrator on the holdout (out-of-sample probabilities).
        if calib_ix is not None:
            try:
                raw_p = self._soft_vote_proba(X_scaled[calib_ix])[:, 1]
                self.calibrator = self._fit_calibrator(
                    raw_p, y_arr[calib_ix],
                    sample_weight=swc,
                )
            except Exception as exc:
                raise CalibrationUnavailableError(f"Calibration fit unavailable: {exc}") from exc

        self._is_fitted = True
        return self

    @staticmethod
    def _calibration_split(
        X: npt.NDArray[Any],
        y_arr: npt.NDArray[Any],
        groups: npt.NDArray[Any] | None,
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Split rows into (base-train, calibrate), keeping supplied groups whole.

        Try a bounded, deterministic set of group holdouts. If none supports
        both classes on both sides, calibration is unavailable, never row-wise.
        """
        n = len(y_arr)
        if groups is not None:
            groups_arr = np.asarray(groups)
            if groups_arr.ndim != 1 or len(groups_arr) != n:
                raise CalibrationUnavailableError("Calibration groups must align with rows")
            if len(np.unique(groups_arr)) >= 2:
                splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=42)
                for train_ix, calib_ix in splitter.split(X, y_arr, groups_arr):
                    if (
                        len(calib_ix) >= 5
                        and len(train_ix) >= 5
                        and len(np.unique(y_arr[calib_ix])) == 2
                        and len(np.unique(y_arr[train_ix])) == 2
                    ):
                        return train_ix, calib_ix
            raise CalibrationUnavailableError(
                "No valid grouped calibration split found with both classes on each side"
            )

        tr, ca = train_test_split(
            np.arange(n),
            test_size=0.25,
            stratify=y_arr,
            random_state=42,
        )
        return tr, ca

    # ── Calibration helpers ─────────────────────────────────────────────

    def _fit_calibrator(
        self,
        raw_p: npt.NDArray[Any],
        y_true: npt.NDArray[Any],
        sample_weight: npt.NDArray[Any] | None = None,
    ) -> Any:
        """Fit a probability→probability calibrator on out-of-sample scores."""
        if self.calibration == "sigmoid":
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(
                np.asarray(raw_p, dtype=float).reshape(-1, 1), np.asarray(y_true),
                sample_weight=sample_weight,
            )
            return lr
        # Isotonic regression restores monotonicity; clips outside the fitted
        # range instead of extrapolating (safer on small data).
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(np.asarray(raw_p, dtype=float), np.asarray(y_true), sample_weight=sample_weight)
        return iso

    def _apply_calibration(self, proba: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Map a raw (n,2) probability array through the fitted calibrator."""
        if self.calibrator is None:
            return proba
        raw_p = np.asarray(proba[:, 1], dtype=float)
        if self.calibration == "sigmoid":
            cal_p = np.asarray(
                self.calibrator.predict_proba(raw_p.reshape(-1, 1))[:, 1],
                dtype=float,
            )
        else:
            cal_p = np.asarray(self.calibrator.predict(raw_p), dtype=float)
        cal_p = np.clip(cal_p, 0.0, 1.0)
        return np.column_stack([1.0 - cal_p, cal_p])

    def _soft_vote_proba(self, X_scaled: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Raw soft-vote class probabilities (RF + XGB mean, or RF only)."""
        rf_proba = np.asarray(self.rf_model.predict_proba(X_scaled))
        if self._xgb_available and self.xgb_model is not None:
            xgb_proba = np.asarray(self.xgb_model.predict_proba(X_scaled))
            return cast(npt.NDArray[Any], (rf_proba + xgb_proba) / 2.0)
        return rf_proba

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Soft-vote class labels (1=focus, 0=distraction).

        Averages RF and XGBoost predicted probabilities (then applies
        probability calibration), and argmax.  Falls back to RF-only when
        xgboost is unavailable.
        """
        proba = self.predict_proba(X)
        return np.asarray(proba.argmax(axis=1))

    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Soft-vote class probabilities, calibrated.

        Returns the element-wise mean of RF and XGBoost probability arrays
        (RF-only when xgboost is unavailable) mapped through the post-hoc
        probability calibrator when one was fitted.
        """
        X_scaled = self.scaler.transform(X)
        return self._apply_calibration(self._soft_vote_proba(X_scaled))

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
            "calibration": self.calibration,
            "calibrator": self.calibrator,
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
        instance.calibration = data.get("calibration")
        instance.calibrator = data.get("calibrator")

        return instance
