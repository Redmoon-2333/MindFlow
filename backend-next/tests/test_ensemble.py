"""Tests for EnsembleClassifier (RF + XGBoost soft voting).

Covers:
  - fit / predict / predict_proba (both with and without xgboost)
  - feature importance + xgboost sub-key
  - evaluate metrics
  - to_dict / from_dict round-trip
  - ModelManager integration with use_ensemble=True/False
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mindflow.train.models import EnsembleClassifier, FocusClassifier, ModelManager


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample feature matrix matching typical extractor output."""
    rng = np.random.default_rng(42)
    n = 50
    features = np.zeros((n, 14), dtype=np.float64)
    features[:, 0] = rng.poisson(3, n).astype(np.float64)  # unique_app_count
    features[:, 1] = rng.exponential(15, n).astype(np.float64)  # switch_frequency
    features[:, 2] = rng.beta(5, 2, n).astype(np.float64)  # productivity_ratio
    features[:, 3] = rng.beta(1, 5, n).astype(np.float64)  # entertainment_ratio
    features[:, 4] = rng.beta(1, 6, n).astype(np.float64)  # social_ratio
    features[:, 5] = rng.exponential(500, n).astype(np.float64)  # max_app_duration
    features[:, 6] = rng.beta(1, 10, n).astype(np.float64)  # idle_ratio
    features[:, 7] = rng.integers(0, 24, n).astype(np.float64)  # hour_of_day
    features[:, 8] = rng.integers(0, 7, n).astype(np.float64)  # day_of_week
    features[:, 9] = rng.beta(2, 5, n).astype(np.float64)  # title_code_ratio
    features[:, 10] = rng.beta(2, 5, n).astype(np.float64)  # title_doc_ratio
    features[:, 11] = rng.beta(2, 5, n).astype(np.float64)  # title_url_ratio
    features[:, 12] = rng.beta(1, 8, n).astype(np.float64)  # title_meeting_ratio
    features[:, 13] = rng.beta(1, 8, n).astype(np.float64)  # title_entertainment_ratio
    return features


@pytest.fixture
def binary_labels() -> np.ndarray:
    """Balanced binary labels."""
    rng = np.random.default_rng(99)
    return rng.integers(0, 2, 50).astype(np.int32)


@pytest.fixture
def feature_names() -> list[str]:
    return [f"f{i}" for i in range(14)]


@pytest.fixture
def model_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestEnsembleClassifier:
    """Core EnsembleClassifier unit tests.

    These tests exercise the ensemble classifier directly.  Whether xgboost
    is installed or not, the API is expected to work identically.
    """

    def test_fit_and_predict(
        self,
        sample_features: np.ndarray,
        binary_labels: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Fit should succeed and predict should return binary labels."""
        clf = EnsembleClassifier()
        clf.fit(sample_features, binary_labels, feature_names)
        assert clf._is_fitted

        preds = clf.predict(sample_features[:5])
        assert len(preds) == 5
        assert all(p in (0, 1) for p in preds)

    def test_predict_proba_shape(
        self,
        sample_features: np.ndarray,
        binary_labels: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """predict_proba should return (n_samples, 2) array."""
        clf = EnsembleClassifier()
        clf.fit(sample_features, binary_labels, feature_names)
        proba = clf.predict_proba(sample_features[:3])
        assert proba.shape == (3, 2)

    def test_feature_importance(
        self,
        sample_features: np.ndarray,
        binary_labels: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """get_feature_importance should return dict with all feature names."""
        clf = EnsembleClassifier()
        clf.fit(sample_features, binary_labels, feature_names)
        importance = clf.get_feature_importance()
        assert len(importance) >= len(feature_names)

        for fn in feature_names:
            assert fn in importance

    def test_not_fitted_importance(self) -> None:
        """get_feature_importance before fit should return empty dict."""
        clf = EnsembleClassifier()
        assert clf.get_feature_importance() == {}

    def test_evaluate(
        self,
        sample_features: np.ndarray,
        binary_labels: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """evaluate should return expected metrics dict."""
        clf = EnsembleClassifier()
        n = len(sample_features)
        split = int(n * 0.7)
        x_train, x_test = sample_features[:split], sample_features[split:]
        y_train, y_test = binary_labels[:split], binary_labels[split:]

        clf.fit(x_train, y_train, feature_names)
        metrics = clf.evaluate(x_test, y_test)

        for key in ("accuracy", "precision", "recall", "f1", "cv_mean", "cv_std"):
            assert key in metrics, f"Missing metric: {key}"
        for val in metrics.values():
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

    def test_calibration_off_by_default(
        self,
        sample_features: np.ndarray,
        binary_labels: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """calibration is OFF by default (legacy raw probabilities)."""
        clf = EnsembleClassifier()
        clf.fit(sample_features, binary_labels, feature_names)
        assert clf.calibration is None
        assert clf.calibrator is None

        proba = clf.predict_proba(sample_features[:3])
        assert proba.shape == (3, 2)
        assert float(np.abs(proba.sum(axis=1) - 1.0).max()) < 1e-9

    def test_calibration_none_returns_raw_probabilities(
        self,
        sample_features: np.ndarray,
        binary_labels: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """calibration=None disables post-hoc calibration (legacy behaviour)."""
        clf = EnsembleClassifier(calibration=None)
        clf.fit(sample_features, binary_labels, feature_names)
        assert clf.calibrator is None

        proba = clf.predict_proba(sample_features[:3])
        assert proba.shape == (3, 2)
        assert float(np.abs(proba.sum(axis=1) - 1.0).max()) < 1e-9

    def test_sigmoid_calibration_opt_in(self) -> None:
        """calibration='sigmoid' fits a Platt calibrator when opted in."""
        rng = np.random.default_rng(7)
        features = rng.normal(size=(80, 4))
        labels = (features[:, 0] > 0).astype(np.int32)
        clf = EnsembleClassifier(calibration="sigmoid")
        clf.fit(features, labels, [f"f{i}" for i in range(4)])
        assert clf.calibrator is not None
        proba = clf.predict_proba(features[:5])
        assert proba.shape == (5, 2)
        assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0

    def test_isotonic_calibration_opt_in(self) -> None:
        """calibration='isotonic' is available as an opt-in variant."""
        rng = np.random.default_rng(3)
        features = rng.normal(size=(80, 4))
        labels = (features[:, 0] > 0).astype(np.int32)
        clf = EnsembleClassifier(calibration="isotonic")
        clf.fit(features, labels, [f"f{i}" for i in range(4)])
        assert clf.calibrator is not None
        proba = clf.predict_proba(features[:5])
        assert proba.shape == (5, 2)

    def test_sigmoid_calibration_roundtrip(self) -> None:
        """calibration='sigmoid' serializes its calibrator after round-trip."""
        rng = np.random.default_rng(9)
        features = rng.normal(size=(60, 4))
        labels = (features[:, 0] + features[:, 1] * 0.5 > 0).astype(np.int32)
        clf = EnsembleClassifier(calibration="sigmoid")
        clf.fit(features, labels, [f"f{i}" for i in range(4)])
        assert clf.calibrator is not None

        data = clf.to_dict()
        clf2 = EnsembleClassifier.from_dict(data)
        assert clf2.calibration == "sigmoid"
        assert clf2.calibrator is not None
        assert np.allclose(clf.predict_proba(features[:5]), clf2.predict_proba(features[:5]))

    def test_to_dict_from_dict_roundtrip(
        self,
        sample_features: np.ndarray,
        binary_labels: np.ndarray,
        feature_names: list[str],
    ) -> None:
        """Serialization round-trip should preserve fitted state."""
        clf = EnsembleClassifier()
        clf.fit(sample_features, binary_labels, feature_names)
        proba_before = clf.predict_proba(sample_features[:3])

        data = clf.to_dict()
        assert data.get("__class__") == "EnsembleClassifier"
        assert data["is_fitted"] is True

        clf2 = EnsembleClassifier.from_dict(data)
        assert clf2._is_fitted
        proba_after = clf2.predict_proba(sample_features[:3])
        assert np.allclose(proba_before, proba_after)


class TestModelManagerWithEnsemble:
    """ModelManager integration tests with the ensemble classifier."""

    def test_use_ensemble_true_creates_ensemble(
        self, model_dir: Path,
    ) -> None:
        """use_ensemble=True (default) should create EnsembleClassifier."""
        manager = ModelManager(models_dir=model_dir, use_ensemble=True)
        # Even if xgboost isn't installed, it should still be a valid classifier
        assert isinstance(manager.classifier, (EnsembleClassifier, FocusClassifier))

    def test_use_ensemble_false_creates_focus_classifier(
        self, model_dir: Path,
    ) -> None:
        """use_ensemble=False should create FocusClassifier."""
        manager = ModelManager(models_dir=model_dir, use_ensemble=False)
        assert isinstance(manager.classifier, FocusClassifier)

    def test_train_all_with_ensemble(
        self,
        sample_features: np.ndarray,
        feature_names: list[str],
        binary_labels: np.ndarray,
        model_dir: Path,
    ) -> None:
        """train_all should work with ensemble enabled."""
        manager = ModelManager(models_dir=model_dir, use_ensemble=True)
        w = np.ones(50, dtype=np.float64)

        summary = manager.train_all(sample_features, feature_names, binary_labels, w)
        assert summary.clustering is not None
        assert summary.classifier is not None
        assert summary.hmm is not None

    def test_save_and_load_with_ensemble(
        self,
        sample_features: np.ndarray,
        feature_names: list[str],
        binary_labels: np.ndarray,
        model_dir: Path,
    ) -> None:
        """Save/load round-trip should preserve classifier type."""
        manager = ModelManager(models_dir=model_dir, use_ensemble=True)
        w = np.ones(50, dtype=np.float64)

        manager.train_all(sample_features, feature_names, binary_labels, w)
        saved = manager.save_all()

        for _name, filename in saved.items():
            assert (model_dir / filename).exists()

        manager2 = ModelManager(models_dir=model_dir, use_ensemble=True)
        loaded = manager2.load_latest()
        assert loaded
        assert isinstance(manager2.classifier, (EnsembleClassifier, FocusClassifier))

    def test_train_all_without_ensemble(
        self,
        sample_features: np.ndarray,
        feature_names: list[str],
        binary_labels: np.ndarray,
        model_dir: Path,
    ) -> None:
        """train_all should work with ensemble disabled."""
        manager = ModelManager(models_dir=model_dir, use_ensemble=False)
        w = np.ones(50, dtype=np.float64)

        summary = manager.train_all(sample_features, feature_names, binary_labels, w)
        assert summary.classifier is not None
