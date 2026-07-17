"""Tests for ML models and ModelManager.

Focuses on:
  - BehaviorClustering: fit produces clusters, predict works
  - BehaviorHMM: hmmlearn availability handling fallback
  - ModelManager: versioned save/load/rollback
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mindflow.train.models import (
    BehaviorClustering,
    BehaviorHMM,
    FocusClassifier,
    ModelManager,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample feature matrix (similar to what extractor produces)."""
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
def model_dir() -> Path:
    """Temporary directory for model save/load tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


# ── BehaviorClustering ────────────────────────────────────────────────────────


class TestBehaviorClustering:
    def test_dbscan_fit(self, sample_features: np.ndarray) -> None:
        """DBSCAN should produce clusters."""
        clustering = BehaviorClustering(method="dbscan")
        clusters = clustering.fit(sample_features)
        assert len(clusters) > 0
        assert clustering.labels_ is not None
        assert len(clustering.labels_) == len(sample_features)

    def test_kmeans_fit(self, sample_features: np.ndarray) -> None:
        """KMeans should produce clusters."""
        clustering = BehaviorClustering(method="kmeans")
        clusters = clustering.fit(sample_features)
        assert len(clusters) > 0
        assert clustering.labels_ is not None

    def test_cluster_has_labels(self, sample_features: np.ndarray) -> None:
        """Fitted clusters should have human-readable labels."""
        clustering = BehaviorClustering(method="dbscan")
        clusters = clustering.fit(sample_features)
        for c in clusters:
            assert c.label in (
                "noise", "deep_focus", "shallow_work", "browsing",
                "procrastination", "idle",
            ) or c.label.startswith("cluster_")

    def test_predict(self, sample_features: np.ndarray) -> None:
        """predict() should return same-length array."""
        clustering = BehaviorClustering(method="kmeans")
        clustering.fit(sample_features)
        preds = clustering.predict(sample_features[:5])
        assert len(preds) == 5

    def test_focus_score_range(self, sample_features: np.ndarray) -> None:
        """Focus scores should be in [0, 1]."""
        clustering = BehaviorClustering(method="kmeans")
        clusters = clustering.fit(sample_features)
        for c in clusters:
            assert 0.0 <= c.avg_focus_score <= 1.0, (
                f"Focus score {c.avg_focus_score} out of range"
            )

    def test_predict_before_fit(self) -> None:
        """predict() before fit() should return all -1."""
        clustering = BehaviorClustering()
        features = np.zeros((5, 14), dtype=np.float64)
        preds = clustering.predict(features)
        assert np.all(preds == -1)


# ── BehaviorHMM ───────────────────────────────────────────────────────────────


class TestBehaviorHMM:
    def test_markov_chain_fallback(self) -> None:
        """HMM should fit even without hmmlearn (Markov chain fallback)."""
        hmm = BehaviorHMM(n_states=3)
        sequences = [
            np.array([0, 1, 2, 1, 0, 0, 1, 2, 2, 1], dtype=np.int32),
            np.array([0, 0, 1, 1, 2, 0, 1], dtype=np.int32),
        ]
        hmm.fit(sequences)
        assert hmm._is_fitted
        assert hmm.transition_matrix is not None
        assert hmm.transition_matrix.shape == (3, 3)

    def test_transition_matrix_rows_sum_to_one(self) -> None:
        """Each row of the transition matrix should sum to 1."""
        hmm = BehaviorHMM(n_states=4)
        sequences = [
            np.array([0, 1, 2, 3, 2, 1, 0], dtype=np.int32),
            np.array([1, 2, 3, 3, 2], dtype=np.int32),
        ]
        hmm.fit(sequences)
        for i in range(4):
            assert abs(hmm.transition_matrix[i].sum() - 1.0) < 1e-6

    def test_predict_next_state(self) -> None:
        """predict_next_state should return distribution over states."""
        hmm = BehaviorHMM(n_states=3)
        sequences = [
            np.array([0, 1, 2, 1, 0, 0, 1], dtype=np.int32),
        ]
        hmm.fit(sequences)
        result = hmm.predict_next_state(0)
        assert "next_state" in result
        assert "probabilities" in result
        assert "next_state_name" in result
        assert len(result["probabilities"]) == 3

    def test_predict_before_fit(self) -> None:
        """predict_next_state before fit should return uniform distribution."""
        hmm = BehaviorHMM(n_states=5)
        result = hmm.predict_next_state(0)
        assert result["next_state"] == 0
        assert all(p == 0.2 for p in result["probabilities"])

    def test_steady_state(self) -> None:
        """get_steady_state should return a valid probability distribution."""
        hmm = BehaviorHMM(n_states=2)
        sequences = [
            np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=np.int32),
        ]
        hmm.fit(sequences)
        steady = hmm.get_steady_state()
        assert len(steady) == 2
        assert abs(steady.sum() - 1.0) < 1e-6
        assert all(p >= 0 for p in steady)

    def test_empty_sequences(self) -> None:
        """Empty sequences should not crash."""
        hmm = BehaviorHMM(n_states=3)
        hmm.fit([])
        assert hmm._is_fitted


# ── FocusClassifier ───────────────────────────────────────────────────────────


class TestFocusClassifier:
    def test_fit_and_predict(self, sample_features: np.ndarray) -> None:
        """FocusClassifier should fit and predict binary labels."""
        classifier = FocusClassifier()
        n = len(sample_features)
        y = np.array([1 if i < n // 2 else 0 for i in range(n)], dtype=np.int32)
        feature_names = [f"f{i}" for i in range(14)]

        classifier.fit(sample_features, y, feature_names)
        preds = classifier.predict(sample_features[:5])
        assert len(preds) == 5
        assert all(p in (0, 1) for p in preds)

    def test_predict_proba(self, sample_features: np.ndarray) -> None:
        """predict_proba should return probability scores."""
        classifier = FocusClassifier()
        n = len(sample_features)
        y = np.array([1 if i < n // 2 else 0 for i in range(n)], dtype=np.int32)
        feature_names = [f"f{i}" for i in range(14)]

        classifier.fit(sample_features, y, feature_names)
        proba = classifier.predict_proba(sample_features[:3])
        assert proba.shape == (3, 2)

    def test_feature_importance(self, sample_features: np.ndarray) -> None:
        """get_feature_importance should return feature names."""
        classifier = FocusClassifier()
        n = len(sample_features)
        y = np.array([1 if i < n // 2 else 0 for i in range(n)], dtype=np.int32)
        feature_names = [f"f{i}" for i in range(14)]

        classifier.fit(sample_features, y, feature_names)
        importance = classifier.get_feature_importance()
        assert len(importance) == 14
        for name in feature_names:
            assert name in importance

    def test_not_fitted_importance(self) -> None:
        """get_feature_importance before fit should return empty dict."""
        classifier = FocusClassifier()
        assert classifier.get_feature_importance() == {}


# ── ModelManager ──────────────────────────────────────────────────────────────


class TestModelManager:
    def test_train_all_returns_summary(self, sample_features: np.ndarray, model_dir: Path) -> None:
        """train_all should return TrainingSummary with all sections."""
        manager = ModelManager(models_dir=model_dir)
        feature_names = [f"f{i}" for i in range(14)]
        y = np.array([1 if i < 25 else 0 for i in range(50)], dtype=np.int32)
        w = np.ones(50, dtype=np.float64)

        summary = manager.train_all(sample_features, feature_names, y, w)
        assert summary.clustering is not None
        assert summary.classifier is not None
        assert summary.hmm is not None

    def test_save_and_load_latest(self, sample_features: np.ndarray, model_dir: Path) -> None:
        """save_all then load_latest should restore models."""
        manager = ModelManager(models_dir=model_dir)
        feature_names = [f"f{i}" for i in range(14)]
        y = np.array([1 if i < 25 else 0 for i in range(50)], dtype=np.int32)
        w = np.ones(50, dtype=np.float64)

        manager.train_all(sample_features, feature_names, y, w)
        saved = manager.save_all()

        # Verify saved filenames exist
        for _name, filename in saved.items():
            assert (model_dir / filename).exists()

        # Verify latest.json exist
        assert (model_dir / "latest.json").exists()

        # Create a new manager and load
        manager2 = ModelManager(models_dir=model_dir)
        loaded = manager2.load_latest()
        assert loaded

    def test_list_versions(self, sample_features: np.ndarray, model_dir: Path) -> None:
        """list_versions should return saved version tags."""
        manager = ModelManager(models_dir=model_dir)
        feature_names = [f"f{i}" for i in range(14)]
        y = np.array([1 if i < 25 else 0 for i in range(50)], dtype=np.int32)
        w = np.ones(50, dtype=np.float64)

        manager.train_all(sample_features, feature_names, y, w)
        manager.save_all()

        versions = manager.list_versions()
        assert len(versions) >= 1
        assert all(len(v) == 8 and v.isdigit() for v in versions)

    def test_load_version(self, sample_features: np.ndarray, model_dir: Path) -> None:
        """load_version should load a specific tag."""
        manager = ModelManager(models_dir=model_dir)
        feature_names = [f"f{i}" for i in range(14)]
        y = np.array([1 if i < 25 else 0 for i in range(50)], dtype=np.int32)
        w = np.ones(50, dtype=np.float64)

        manager.train_all(sample_features, feature_names, y, w)
        saved = manager.save_all()
        tag = list(saved.values())[0].split("-")[1].split(".")[0]

        manager2 = ModelManager(models_dir=model_dir)
        loaded = manager2.load_version(tag)
        assert loaded

    def test_rollback(self, sample_features: np.ndarray, model_dir: Path) -> None:
        """rollback should update latest.json."""
        manager = ModelManager(models_dir=model_dir)
        feature_names = [f"f{i}" for i in range(14)]
        y = np.array([1 if i < 25 else 0 for i in range(50)], dtype=np.int32)
        w = np.ones(50, dtype=np.float64)

        manager.train_all(sample_features, feature_names, y, w)
        manager.save_all()

        tag = manager.current_version_tag
        assert tag is not None

        # Rollback to same tag (should succeed)
        assert manager.rollback(tag)

    def test_load_latest_on_empty_dir(self, model_dir: Path) -> None:
        """load_latest on empty dir should return False."""
        manager = ModelManager(models_dir=model_dir)
        assert not manager.load_latest()

    def test_current_version_tag(self, sample_features: np.ndarray, model_dir: Path) -> None:
        """current_version_tag should return the tag from latest.json."""
        manager = ModelManager(models_dir=model_dir)
        feature_names = [f"f{i}" for i in range(14)]
        y = np.array([1 if i < 25 else 0 for i in range(50)], dtype=np.int32)
        w = np.ones(50, dtype=np.float64)

        manager.train_all(sample_features, feature_names, y, w)
        manager.save_all()

        tag = manager.current_version_tag
        assert tag is not None
        assert len(tag) == 8

    def test_latest_json_structure(self, sample_features: np.ndarray, model_dir: Path) -> None:
        """latest.json should have clustering, classifier, hmm keys."""
        manager = ModelManager(models_dir=model_dir)
        feature_names = [f"f{i}" for i in range(14)]
        y = np.array([1 if i < 25 else 0 for i in range(50)], dtype=np.int32)
        w = np.ones(50, dtype=np.float64)

        manager.train_all(sample_features, feature_names, y, w)
        manager.save_all()

        pointer = json.loads((model_dir / "latest.json").read_text(encoding="utf-8"))
        assert "clustering" in pointer
        assert "classifier" in pointer
        assert "hmm" in pointer
