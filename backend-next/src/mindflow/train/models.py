"""ML models for MindFlow behavior analysis.

Ported from ``backend/mindflow/analyzer/ml_models.py``.
Key improvements vs. the original:
  - **Versioned model persistence**: ``ModelManager`` saves models as
    ``models/{name}-{YYYYMMDD}.pkl`` and maintains a ``latest.json`` pointer
    file, solving the original's "fixed filename prevents rollback" P1 issue.
  - **FocusClassifier** is kept but simplified — it is less relevant for the
    new architecture where ``ConsensusLabeler`` handles weak supervision;
    retained for pipeline completeness.
  - **BehaviorClustering** unchanged: DBSCAN with auto-eps / KMeans fallback.
  - **BehaviorHMM** keeps the hmmlearn → Markov chain fallback path.

All models accept and return plain numpy arrays (no pandas dependency).
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class BehaviorCluster:
    """Result of clustering a behavior pattern."""

    cluster_id: int
    label: str
    centroid_features: npt.NDArray[Any]
    sample_count: int
    avg_focus_score: float


@dataclass
class TrainingSummary:
    """Lightweight summary of one train_all run (non-serializable fields excluded)."""

    clustering: dict[str, Any]
    classifier: dict[str, Any]
    hmm: dict[str, Any]


# ── BehaviorClustering ────────────────────────────────────────────────────────


class BehaviorClustering:
    """Unsupervised clustering of behavior patterns with DBSCAN/KMeans.

    Args:
        method: ``"dbscan"`` (default) or ``"kmeans"``.
    """

    CLUSTER_LABEL_MAP: dict[int, str] = {
        -1: "noise",
        0: "deep_focus",
        1: "shallow_work",
        2: "browsing",
        3: "procrastination",
        4: "idle",
    }

    def __init__(self, method: str = "dbscan") -> None:
        self.method = method.lower()
        self.scaler = StandardScaler()
        self.model: DBSCAN | KMeans | None = None
        self.labels_: npt.NDArray[Any] | None = None
        self._cluster_info: list[BehaviorCluster] = []

    def fit(self, features: npt.NDArray[Any]) -> list[BehaviorCluster]:
        """Fit clustering model and assign human-readable labels.

        Args:
            features: numpy array of shape ``(n_samples, n_features)``.

        Returns:
            List of ``BehaviorCluster`` with human-readable labels.
        """
        X_scaled = self.scaler.fit_transform(features)

        if self.method == "dbscan":
            n_features = X_scaled.shape[1]
            eps = max(0.5, np.sqrt(n_features) * 0.5)
            min_samples = max(3, int(len(X_scaled) * 0.02))
            self.model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        else:
            n_clusters = min(5, max(2, int(np.sqrt(len(X_scaled)))))
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        self.labels_ = self.model.fit_predict(X_scaled)
        self._cluster_info = self._assign_cluster_labels(X_scaled, features)
        return self._cluster_info

    def _assign_cluster_labels(
        self,
        X_scaled: npt.NDArray[Any],
        X_raw: npt.NDArray[Any],
    ) -> list[BehaviorCluster]:
        """Analyze cluster centroids and assign human-readable labels."""
        if self.labels_ is None:
            return []

        unique_labels = sorted(set(self.labels_))
        result: list[BehaviorCluster] = []

        for label in unique_labels:
            mask = self.labels_ == label
            cluster_features = X_raw[mask]
            sample_count = int(mask.sum())
            centroid = np.mean(cluster_features, axis=0)
            focus_score = self._compute_focus_score(centroid)

            cluster_label = self.CLUSTER_LABEL_MAP.get(
                label, f"cluster_{label}"
            )

            result.append(
                BehaviorCluster(
                    cluster_id=int(label),
                    label=cluster_label,
                    centroid_features=centroid,
                    sample_count=sample_count,
                    avg_focus_score=round(float(focus_score), 4),
                )
            )

        if not any(c.cluster_id != -1 for c in result):
            return result

        non_noise = [c for c in result if c.cluster_id != -1]
        non_noise.sort(key=lambda c: c.avg_focus_score, reverse=True)

        human_labels = [
            "deep_focus", "shallow_work", "browsing", "procrastination", "idle",
        ]
        for i, cluster in enumerate(non_noise):
            if i < len(human_labels):
                cluster.label = human_labels[i]

        return result

    @staticmethod
    def _compute_focus_score(centroid: npt.NDArray[Any]) -> float:
        """Heuristic focus score from feature centroid.

        Higher productivity_ratio = higher focus.
        Higher entertainment_ratio / social_ratio / idle_ratio = lower focus.
        """
        if len(centroid) < 5:
            return 0.5

        productivity_ratio = float(centroid[2]) if len(centroid) > 2 else 0.0
        entertainment_ratio = float(centroid[3]) if len(centroid) > 3 else 0.0
        social_ratio = float(centroid[4]) if len(centroid) > 4 else 0.0
        idle_ratio = float(centroid[6]) if len(centroid) > 6 else 0.0
        switch_freq = float(centroid[1]) if len(centroid) > 1 else 0.0

        switch_penalty = min(switch_freq / 60.0, 1.0)

        score = (
            productivity_ratio * 0.50
            + (1.0 - entertainment_ratio) * 0.15
            + (1.0 - social_ratio) * 0.10
            + (1.0 - idle_ratio) * 0.15
            + (1.0 - switch_penalty) * 0.10
        )
        return float(np.clip(score, 0.0, 1.0))

    def predict(self, features: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Predict cluster labels for new data (nearest-centroid fallback for DBSCAN)."""
        if self.model is None:
            return np.full(len(features), -1, dtype=int)

        X_scaled = self.scaler.transform(features)

        if hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X_scaled))

        centroids = self._get_cluster_centroids()
        if not centroids:
            return np.full(len(features), -1, dtype=int)

        centroid_ids, centroid_vecs = zip(*centroids, strict=False)
        centroid_array = np.stack(centroid_vecs, axis=0)
        distances = np.zeros((len(X_scaled), len(centroid_array)))
        for j, c_vec in enumerate(centroid_array):
            distances[:, j] = np.linalg.norm(X_scaled - c_vec, axis=1)
        nearest = np.argmin(distances, axis=1)
        return np.array([centroid_ids[i] for i in nearest], dtype=int)

    def _get_cluster_centroids(self) -> list[tuple[int, npt.NDArray[Any]]]:
        """Return list of (cluster_id, centroid) for non-noise clusters."""
        centroids: list[tuple[int, npt.NDArray[Any]]] = []
        for c in self._cluster_info:
            if c.cluster_id >= 0 and c.centroid_features is not None:
                centroids.append((c.cluster_id, c.centroid_features))
        return centroids

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scaler": self.scaler,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehaviorClustering:
        instance = cls(method=str(data.get("method", "dbscan")))
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        return instance


# ── FocusClassifier ───────────────────────────────────────────────────────────


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


# ── BehaviorHMM ───────────────────────────────────────────────────────────────


class BehaviorHMM:
    """Hidden Markov Model for behavior state transitions.

    Falls back to a simple Markov chain (transition matrix) if hmmlearn is
    unavailable at import time. This matches the old backend's strategy.
    """

    STATE_NAMES = ["deep_focus", "shallow_work", "browsing", "procrastination", "idle"]

    def __init__(self, n_states: int = 5) -> None:
        self.n_states = n_states
        self.model: Any = None
        self.state_names = self.STATE_NAMES[:n_states]
        self.transition_matrix: npt.NDArray[Any] | None = None
        self._is_fitted: bool = False

    def fit(self, sequences: list[npt.NDArray[Any]]) -> BehaviorHMM:
        """Fit HMM to sequences of state observations.

        Args:
            sequences: list of 1D arrays, each containing a sequence of
                state IDs (0-indexed).

        Returns:
            self
        """
        self.transition_matrix = self._compute_transition_matrix(sequences)

        try:
            import hmmlearn.hmm as hmm

            X, lengths = self._prepare_hmm_data(sequences)
            if X is not None and len(X) >= 10:
                self.model = hmm.CategoricalHMM(
                    n_components=self.n_states,
                    random_state=42,
                    n_iter=100,
                    tol=1e-4,
                )
                self.model.fit(X, lengths)
        except ImportError:
            self.model = None

        self._is_fitted = True
        return self

    def _compute_transition_matrix(self, sequences: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
        """Compute Markov transition matrix from state sequences."""
        matrix = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        counts = np.zeros((self.n_states, self.n_states), dtype=np.int32)

        for seq in sequences:
            for i in range(len(seq) - 1):
                s_from = int(seq[i])
                s_to = int(seq[i + 1])
                if 0 <= s_from < self.n_states and 0 <= s_to < self.n_states:
                    counts[s_from, s_to] += 1

        for i in range(self.n_states):
            row_sum = counts[i].sum()
            if row_sum > 0:
                matrix[i] = counts[i] / row_sum
            else:
                matrix[i] = np.ones(self.n_states) / self.n_states

        return matrix

    @staticmethod
    def _prepare_hmm_data(
        sequences: list[npt.NDArray[Any]],
    ) -> tuple[npt.NDArray[Any] | None, npt.NDArray[Any] | None]:
        """Prepare data for hmmlearn format."""
        all_observations: list[int] = []
        lengths: list[int] = []
        for seq in sequences:
            n_states = 5  # default; caller should ensure state IDs are valid
            filtered = [int(s) for s in seq if 0 <= int(s) < n_states]
            if filtered:
                all_observations.extend(filtered)
                lengths.append(len(filtered))
        if not all_observations:
            return None, None
        lengths_array = np.array(lengths, dtype=np.int32)
        X = np.array(all_observations).reshape(-1, 1)
        return X, lengths_array

    def predict_next_state(self, current_state: int) -> dict[str, Any]:
        """Predict most likely next state and probability distribution.

        Args:
            current_state: current state ID (0-indexed).

        Returns:
            dict with keys: ``next_state`` (int), ``probabilities`` (list[float]),
            ``next_state_name`` (str).
        """
        if not self._is_fitted or self.transition_matrix is None:
            uniform = 1.0 / self.n_states
            return {
                "next_state": 0,
                "probabilities": [uniform] * self.n_states,
                "next_state_name": self.state_names[0],
            }

        probs = self._get_transition_probs(current_state)
        next_state = int(np.argmax(probs))

        return {
            "next_state": next_state,
            "probabilities": [round(float(p), 4) for p in probs],
            "next_state_name": self.state_names[next_state],
        }

    def _get_transition_probs(self, state: int) -> npt.NDArray[Any]:
        """Get transition probabilities from hmmlearn or matrix."""
        if self.model is not None:
            try:
                transmat = self.model.transmat_
                if 0 <= state < transmat.shape[0]:
                    return np.asarray(transmat[state])
            except (AttributeError, IndexError):
                pass

        if self.transition_matrix is not None and 0 <= state < self.n_states:
            return np.asarray(self.transition_matrix[state])

        return np.ones(self.n_states) / self.n_states

    def get_transition_matrix(self) -> npt.NDArray[Any]:
        """Return the transition matrix as a 2D numpy array."""
        if self.transition_matrix is None:
            return np.ones((self.n_states, self.n_states)) / self.n_states
        return self.transition_matrix

    def get_steady_state(self) -> npt.NDArray[Any]:
        """Compute steady-state distribution via eigenvector of transition matrix.

        Returns probability distribution over states.
        """
        mat = self.get_transition_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(mat.T)
        idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
        steady = np.asarray(np.real(eigenvectors[:, idx]))
        steady = steady / steady.sum()
        return np.asarray(np.clip(steady, 0.0, 1.0))


# ── ModelManager (versioned) ──────────────────────────────────────────────────


class ModelManager:
    """Central model management with versioned persistence.

    Solves the old backend's P1 technical debt: models are saved with a
    date-stamped filename (``{name}-{YYYYMMDD}.pkl``) and ``latest.json``
    tracks the current active version, enabling rollback by simply updating
    the pointer file.

    Directory layout::

        models/
        +- latest.json           # {"clustering": "clustering-20260717.pkl", ...}
        +- clustering-20260717.pkl
        +- classifier-20260717.pkl
        +- hmm-20260717.pkl
    """

    def __init__(self, models_dir: str | Path = Path("data/models")) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.models_dir / "latest.json"
        self.clustering = BehaviorClustering()
        self.classifier = FocusClassifier()
        self.hmm = BehaviorHMM()

    @property
    def _today_tag(self) -> str:
        return date.today().strftime("%Y%m%d")

    # ── Training ──────────────────────────────────────────────────────────

    def train_all(
        self,
        features: npt.NDArray[Any],
        feature_names: list[str],
        labels: npt.NDArray[Any],
        sample_weight: npt.NDArray[Any] | None = None,
        min_confidence: float = 0.0,
    ) -> TrainingSummary:
        """Train all models and return summary.

        Args:
            features: Feature matrix ``(n_samples, n_features)``.
            feature_names: Names for each feature column.
            labels: Binary labels (1=focus, 0=distraction).
            sample_weight: Per-sample confidence weights.
            min_confidence: Filter samples below this confidence.

        Returns:
            ``TrainingSummary`` with clustering, classifier, hmm subsections.
        """
        summary_clustering: dict[str, Any] = {}
        summary_classifier: dict[str, Any] = {}
        summary_hmm: dict[str, Any] = {}

        # ── Clustering ──
        cluster_info = self.clustering.fit(features)
        summary_clustering = {
            "n_clusters": len([c for c in cluster_info if c.cluster_id != -1]),
            "noise_points": sum(1 for c in cluster_info if c.cluster_id == -1),
            "clusters": [
                {
                    "id": c.cluster_id,
                    "label": c.label,
                    "count": c.sample_count,
                    "avg_focus_score": c.avg_focus_score,
                }
                for c in cluster_info
            ],
        }

        # ── Classifier ──
        high_conf_mask = (
            np.ones(len(features), dtype=bool)
            if sample_weight is None
            else sample_weight >= min_confidence
        )
        X_high = features[high_conf_mask]
        y_high = labels[high_conf_mask]
        sw_high = (
            None
            if sample_weight is None
            else sample_weight[high_conf_mask]
        )
        low_conf_count = int((~high_conf_mask).sum())

        if len(np.unique(y_high)) >= 2 and len(X_high) >= 10:
            svw = (
                sw_high
                if sw_high is not None
                else np.ones(len(X_high))
            )
            X_train, X_test, y_train, y_test, sw_train, _ = train_test_split(
                X_high,
                y_high,
                svw,
                test_size=0.2,
                random_state=42,
                stratify=y_high,
            )
            self.classifier.fit(
                X_train,
                y_train,
                feature_names,
                sample_weight=sw_train if sample_weight is not None else None,
            )
            eval_metrics = self.classifier.evaluate(X_test, y_test)
            summary_classifier = {
                "accuracy": eval_metrics.get("accuracy", 0.0),
                "precision": eval_metrics.get("precision", 0.0),
                "recall": eval_metrics.get("recall", 0.0),
                "f1": eval_metrics.get("f1", 0.0),
                "cv_mean": eval_metrics.get("cv_mean", 0.0),
                "cv_std": eval_metrics.get("cv_std", 0.0),
                "feature_importance": self.classifier.get_feature_importance(),
                "high_confidence_samples": len(X_high),
                "filtered_low_confidence": low_conf_count,
            }
        else:
            summary_classifier = {
                "error": "Not enough data for supervised training",
                "n_samples": len(X_high),
                "n_classes": int(len(np.unique(y_high))),
                "filtered_low_confidence": low_conf_count,
            }

        # ── HMM ──
        sequences = self._build_state_sequences()
        if sequences:
            self.hmm.fit(sequences)
            tm = self.hmm.get_transition_matrix()
            steady = self.hmm.get_steady_state()
            summary_hmm = {
                "transition_matrix": [
                    [round(float(v), 4) for v in row] for row in tm
                ],
                "steady_state": [round(float(v), 4) for v in steady],
                "state_names": list(self.hmm.state_names),
            }
        else:
            summary_hmm = {"error": "No valid state sequences for HMM training"}

        return TrainingSummary(
            clustering=summary_clustering,
            classifier=summary_classifier,
            hmm=summary_hmm,
        )

    def _build_state_sequences(self) -> list[npt.NDArray[Any]]:
        """Build state ID sequences from clustering labels (single sequence)."""
        if self.clustering.labels_ is None or len(self.clustering.labels_) < 2:
            return []
        return [self.clustering.labels_.astype(int)]

    # ── Versioned persistence ─────────────────────────────────────────────

    def save_all(self) -> dict[str, str]:
        """Save all models with date-stamped filenames and update latest.json.

        Returns:
            Dict mapping model names to their saved filenames.
        """
        tag = self._today_tag

        names: dict[str, str] = {
            "clustering": f"clustering-{tag}.pkl",
            "classifier": f"classifier-{tag}.pkl",
            "hmm": f"hmm-{tag}.pkl",
        }

        joblib.dump(self.clustering.to_dict(), str(self.models_dir / names["clustering"]))
        joblib.dump(self.classifier.to_dict(), str(self.models_dir / names["classifier"]))

        hmm_data: dict[str, Any] = {
            "transition_matrix": (
                self.hmm.transition_matrix.tolist()
                if self.hmm.transition_matrix is not None
                else None
            ),
            "state_names": list(self.hmm.state_names),
            "n_states": self.hmm.n_states,
            "is_fitted": self.hmm._is_fitted,
        }
        joblib.dump(hmm_data, str(self.models_dir / names["hmm"]))

        # Write / update latest pointer
        self._write_latest(names)

        return names

    def _write_latest(self, names: dict[str, str]) -> None:
        """Write latest.json pointer file."""
        existing: dict[str, str] = {}
        if self.latest_path.exists():
            with suppress(json.JSONDecodeError, OSError):
                existing = json.loads(self.latest_path.read_text(encoding="utf-8"))
        existing.update(names)
        self.latest_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_latest(self) -> bool:
        """Load the latest version of all models. Returns True if successful."""
        if not self.latest_path.exists():
            return False

        try:
            pointer: dict[str, str] = json.loads(
                self.latest_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return False

        required = ["clustering", "classifier", "hmm"]
        if not all(k in pointer for k in required):
            return False

        return self._load_versions(pointer)

    def load_version(self, tag: str) -> bool:
        """Load a specific dated version of all models.

        Args:
            tag: Date string in ``YYYYMMDD`` format (e.g. ``"20260717"``).

        Returns:
            True if all three model files for that tag exist and load.
        """
        names = {
            "clustering": f"clustering-{tag}.pkl",
            "classifier": f"classifier-{tag}.pkl",
            "hmm": f"hmm-{tag}.pkl",
        }
        return self._load_versions(names)

    def list_versions(self) -> list[str]:
        """List all available model version tags from filenames.

        Returns:
            Sorted list of ``YYYYMMDD`` tags for which all three model files
            exist, newest first.
        """
        tags: set[str] = set()
        for f in self.models_dir.glob("*.pkl"):
            parts = f.stem.split("-")
            if len(parts) >= 2 and len(parts[-1]) == 8 and parts[-1].isdigit():
                tags.add(parts[-1])

        valid: list[str] = []
        for tag in sorted(tags, reverse=True):
            required_exists = all(
                (self.models_dir / f"{name}-{tag}.pkl").exists()
                for name in ["clustering", "classifier", "hmm"]
            )
            if required_exists:
                valid.append(tag)

        return valid

    def rollback(self, tag: str) -> bool:
        """Rollback to a specific version and update latest.json.

        Args:
            tag: Date string in ``YYYYMMDD`` format.

        Returns:
            True if rollback succeeded.
        """
        if not self.load_version(tag):
            return False

        names = {
            "clustering": f"clustering-{tag}.pkl",
            "classifier": f"classifier-{tag}.pkl",
            "hmm": f"hmm-{tag}.pkl",
        }
        self._write_latest(names)
        return True

    def _load_versions(self, name_map: dict[str, str]) -> bool:
        """Load models from explicit filenames."""
        try:
            clustering_path = self.models_dir / name_map["clustering"]
            classifier_path = self.models_dir / name_map["classifier"]
            hmm_path = self.models_dir / name_map["hmm"]

            if not all(p.exists() for p in [clustering_path, classifier_path, hmm_path]):
                return False

            self.clustering = BehaviorClustering.from_dict(
                joblib.load(str(clustering_path))
            )
            self.classifier = FocusClassifier.from_dict(
                joblib.load(str(classifier_path))
            )

            hmm_data: dict[str, Any] = joblib.load(str(hmm_path))
            self.hmm = BehaviorHMM(n_states=int(hmm_data.get("n_states", 5)))
            tm = hmm_data.get("transition_matrix")
            self.hmm.transition_matrix = (
                np.array(tm) if tm is not None else None
            )
            self.hmm._is_fitted = bool(hmm_data.get("is_fitted", False))
            return True

        except (FileNotFoundError, EOFError, KeyError, ValueError):
            return False

    @property
    def current_version_tag(self) -> str | None:
        """Return the current version tag from latest.json, or None."""
        if not self.latest_path.exists():
            return None
        try:
            pointer: dict[str, str] = json.loads(
                self.latest_path.read_text(encoding="utf-8")
            )
            tag = pointer.get("clustering", "")
            if tag.startswith("clustering-") and tag.endswith(".pkl"):
                return tag[len("clustering-"): -len(".pkl")]
        except (json.JSONDecodeError, OSError):
            pass
        return None
