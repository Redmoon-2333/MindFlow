"""ML models for MindFlow behavior analysis: clustering, classification, HMM."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class BehaviorCluster:
    """Result of clustering a behavior pattern."""

    cluster_id: int
    label: str
    centroid_features: np.ndarray
    sample_count: int
    avg_focus_score: float


class BehaviorClustering:
    """Unsupervised clustering of behavior patterns with DBSCAN/KMeans."""

    CLUSTER_LABEL_MAP: dict[int, str] = {
        -1: "noise",
        0: "deep_focus",
        1: "shallow_work",
        2: "browsing",
        3: "procrastination",
        4: "idle",
    }

    def __init__(self, method: str = "dbscan"):
        self.method = method.lower()
        self.scaler = StandardScaler()
        self.model = None
        self.labels_ = None
        self._cluster_info: list[BehaviorCluster] = []

    def fit(self, features: np.ndarray) -> list[BehaviorCluster]:
        """Fit clustering model and assign human-readable labels based on feature analysis.

        Args:
            features: numpy array of shape (n_samples, n_features)

        Returns:
            List of BehaviorCluster with human-readable labels.
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
        self, X_scaled: np.ndarray, X_raw: np.ndarray
    ) -> list[BehaviorCluster]:
        """Analyze cluster centroids and assign human-readable labels."""
        unique_labels = sorted(set(self.labels_))
        result: list[BehaviorCluster] = []

        for label in unique_labels:
            mask = self.labels_ == label
            cluster_features = X_raw[mask]
            sample_count = int(mask.sum())

            centroid = np.mean(cluster_features, axis=0)
            focus_score = self._compute_focus_score(centroid)

            if label == -1:
                cluster_label = "noise"
            else:
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

        human_labels = ["deep_focus", "shallow_work", "browsing", "procrastination", "idle"]
        for i, cluster in enumerate(non_noise):
            if i < len(human_labels):
                cluster.label = human_labels[i]

        return result

    def _compute_focus_score(self, centroid: np.ndarray) -> float:
        """Heuristic focus score from feature centroid.

        Assumes features: unique_app_count, switch_frequency, productivity_ratio,
        entertainment_ratio, social_ratio, max_app_duration, idle_ratio,
        hour_of_day, day_of_week

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
        max_switches = 60.0
        switch_penalty = min(switch_freq / max(1, max_switches), 1.0)

        score = (
            productivity_ratio * 0.50
            + (1.0 - entertainment_ratio) * 0.15
            + (1.0 - social_ratio) * 0.10
            + (1.0 - idle_ratio) * 0.15
            + (1.0 - switch_penalty) * 0.10
        )
        return float(np.clip(score, 0.0, 1.0))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.

        KMeans: uses model.predict directly.
        DBSCAN or no model: assigns each point to nearest non-noise cluster centroid.
        """
        if self.model is None:
            return np.full(len(features), -1, dtype=int)

        X_scaled = self.scaler.transform(features)

        if hasattr(self.model, "predict"):
            return self.model.predict(X_scaled)

        centroids = self._get_cluster_centroids()
        if not centroids:
            return np.full(len(features), -1, dtype=int)

        centroid_ids, centroid_vecs = zip(*centroids)
        centroid_array = np.stack(centroid_vecs, axis=0)
        distances = np.zeros((len(X_scaled), len(centroid_array)))
        for j, c in enumerate(centroid_array):
            distances[:, j] = np.linalg.norm(X_scaled - c, axis=1)
        nearest = np.argmin(distances, axis=1)
        return np.array([centroid_ids[i] for i in nearest], dtype=int)

    def _get_cluster_centroids(self) -> list[tuple[int, np.ndarray]]:
        """Return list of (cluster_id, centroid) for non-noise clusters."""
        centroids: list[tuple[int, np.ndarray]] = []
        for c in self._cluster_info:
            if c.cluster_id >= 0 and c.centroid_features is not None:
                centroids.append((c.cluster_id, c.centroid_features))
        return centroids

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {
            "model": self.model,
            "scaler": self.scaler,
            "method": self.method,
            "labels_": self.labels_,
            "_cluster_info": self._cluster_info,
        }
        joblib.dump(data, str(path))

    @classmethod
    def load(cls, path: Path) -> "BehaviorClustering":
        """Load model from disk."""
        data: dict[str, object] = joblib.load(str(path))
        instance = cls(method=str(data.get("method", "dbscan")))
        instance.scaler = data["scaler"]
        instance.model = data["model"]
        instance.labels_ = data.get("labels_")
        instance._cluster_info = data.get("_cluster_info", [])
        return instance


class FocusClassifier:
    """Random Forest classifier for focus vs distraction."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.feature_names_: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        sample_weight: np.ndarray | None = None,
    ) -> "FocusClassifier":
        """Train the classifier.

        Args:
            X: feature matrix of shape (n_samples, n_features)
            y: binary labels (1=focus, 0=distraction)
            feature_names: names for each feature column
            sample_weight: per-sample confidence weights, same shape as y.

        Returns:
            self
        """
        self.feature_names_ = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        if not self._is_fitted:
            return {}
        importances = self.model.feature_importances_
        return {
            name: round(float(imp), 6)
            for name, imp in zip(self.feature_names_, importances)
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance.

        Returns dict with keys: accuracy, precision, recall, f1, cv_mean, cv_std.
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
            "classification_report": classification_report(
                y_test, y_pred, target_names=["distraction", "focus"], output_dict=True
            ),
        }

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names_,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(data, str(path))

    @classmethod
    def load(cls, path: Path) -> "FocusClassifier":
        """Load model from disk."""
        data: dict[str, object] = joblib.load(str(path))
        instance = cls()
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_names_ = data.get("feature_names", [])
        instance._is_fitted = bool(data.get("is_fitted", False))
        return instance


class BehaviorHMM:
    """Hidden Markov Model for behavior state transitions.

    Falls back to simple Markov chain if hmmlearn is unavailable.
    """

    STATE_NAMES = ["deep_focus", "shallow_work", "browsing", "procrastination", "idle"]

    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.model = None
        self.state_names = self.STATE_NAMES[:n_states]
        self.transition_matrix: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def fit(self, sequences: list[np.ndarray]) -> "BehaviorHMM":
        """Fit HMM to sequences of state observations.

        Args:
            sequences: list of 1D arrays, each containing a sequence of state IDs.

        Returns:
            self
        """
        self.transition_matrix = self._compute_transition_matrix(sequences)

        try:
            from hmmlearn import hmm
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

    def _compute_transition_matrix(self, sequences: list[np.ndarray]) -> np.ndarray:
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

    def _prepare_hmm_data(
        self, sequences: list[np.ndarray]
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare data for hmmlearn format."""
        all_observations: list[int] = []
        lengths: list[int] = []
        for seq in sequences:
            filtered = [int(s) for s in seq if 0 <= int(s) < self.n_states]
            if filtered:
                all_observations.extend(filtered)
                lengths.append(len(filtered))
        if not all_observations:
            return None, None
        lengths_array = np.array(lengths, dtype=np.int32)
        X = np.array(all_observations).reshape(-1, 1)
        return X, lengths_array

    def predict_next_state(self, current_state: int) -> dict[str, float]:
        """Predict most likely next state and probability distribution.

        Args:
            current_state: current state ID (0-indexed)

        Returns:
            dict with keys: next_state (int), probabilities (list[float]),
            next_state_name (str)
        """
        if not self._is_fitted or self.transition_matrix is None:
            return {
                "next_state": 0,
                "probabilities": [1.0 / self.n_states] * self.n_states,
                "next_state_name": self.state_names[0],
            }

        probs = self._get_transition_probs(current_state)
        next_state = int(np.argmax(probs))

        return {
            "next_state": next_state,
            "probabilities": [round(float(p), 4) for p in probs],
            "next_state_name": self.state_names[next_state],
        }

    def _get_transition_probs(self, state: int) -> np.ndarray:
        """Get transition probabilities from hmmlearn or matrix."""
        if self.model is not None:
            try:
                transmat = self.model.transmat_
                if 0 <= state < transmat.shape[0]:
                    return transmat[state]
            except (AttributeError, IndexError):
                pass

        if self.transition_matrix is not None and 0 <= state < self.n_states:
            return self.transition_matrix[state]

        return np.ones(self.n_states) / self.n_states

    def get_transition_matrix(self) -> np.ndarray:
        """Return the transition matrix as a 2D numpy array."""
        if self.transition_matrix is None:
            return np.ones((self.n_states, self.n_states)) / self.n_states
        return self.transition_matrix

    def get_steady_state(self) -> np.ndarray:
        """Compute steady-state distribution via eigenvector of transition matrix.

        Returns probability distribution over states.
        """
        mat = self.get_transition_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(mat.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        steady = np.real(eigenvectors[:, idx])
        steady = steady / steady.sum()
        return np.clip(steady, 0.0, 1.0)


class ModelManager:
    """Central model management for MindFlow behavior analysis."""

    def __init__(self, models_dir: Path = Path("data/models")):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.clustering = BehaviorClustering()
        self.classifier = FocusClassifier()
        self.hmm = BehaviorHMM()

    def train_all(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
        min_confidence: float = 0.0,
    ) -> dict:
        """Train all models and return summary dict with metrics.

        Args:
            features_df: DataFrame with feature columns
            labels: binary labels (1=focus, 0=distraction)
            sample_weight: per-sample confidence weights
            min_confidence: filter samples below this confidence before training

        Returns:
            dict with keys: clustering, classifier, hmm_transition, hmm_steady
        """
        summary: dict = {"clustering": {}, "classifier": {}, "hmm": {}}

        feature_cols = [
            c for c in features_df.columns
            if c not in ("window_start", "date")
        ]
        X = features_df[feature_cols].to_numpy(dtype=np.float64)

        cluster_info = self.clustering.fit(X)
        summary["clustering"] = {
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

        high_conf_mask = (
            np.ones(len(X), dtype=bool)
            if sample_weight is None
            else sample_weight >= min_confidence
        )
        X_high = X[high_conf_mask]
        y_high = labels[high_conf_mask]
        sw_high = (
            None if sample_weight is None
            else sample_weight[high_conf_mask]
        )
        low_conf_count = int((~high_conf_mask).sum())

        if len(np.unique(y_high)) >= 2 and len(X_high) >= 10:
            X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
                X_high, y_high,
                sw_high if sw_high is not None else np.ones(len(X_high)),
                test_size=0.2,
                random_state=42,
                stratify=y_high,
            )
            self.classifier.fit(
                X_train, y_train, feature_cols,
                sample_weight=sw_train if sample_weight is not None else None,
            )
            eval_metrics = self.classifier.evaluate(X_test, y_test)
            summary["classifier"] = {
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
            summary["classifier"] = {
                "error": "Not enough data for supervised training",
                "n_samples": len(X_high),
                "n_classes": len(np.unique(y_high)),
                "filtered_low_confidence": low_conf_count,
            }

        sequences = self._build_state_sequences(features_df, labels)
        if sequences:
            self.hmm.fit(sequences)
            tm = self.hmm.get_transition_matrix()
            steady = self.hmm.get_steady_state()
            summary["hmm"] = {
                "transition_matrix": [
                    [round(float(v), 4) for v in row] for row in tm
                ],
                "steady_state": [round(float(v), 4) for v in steady],
                "state_names": self.hmm.state_names,
            }
        else:
            summary["hmm"] = {"error": "No valid state sequences for HMM training"}

        return summary

    def _build_state_sequences(
        self, features_df: pd.DataFrame, labels: np.ndarray
    ) -> list[np.ndarray]:
        """Build state ID sequences from feature data.

        Uses clustering results to assign each time window a state,
        then groups consecutive windows belonging to the same day into
        sequences that capture real temporal transitions.

        Falls back to rule-based state assignment if clustering failed.
        """
        if "window_start" not in features_df.columns:
            return []

        df = features_df.copy()
        states = self._assign_states(df, labels)

        if states is None or len(states) == 0:
            return []

        df["_state"] = states
        df["_date"] = pd.to_datetime(df["window_start"]).dt.date

        sequences: list[np.ndarray] = []
        for _, day_df in df.groupby("_date", sort=True):
            day_states = day_df.sort_values("window_start")["_state"].values
            if len(day_states) >= 2:
                sequences.append(day_states.astype(int))

        return sequences

    def _assign_states(
        self, features_df: pd.DataFrame, labels: np.ndarray
    ) -> np.ndarray | None:
        """Assign each window a state ID using clustering or rule-based fallback."""
        if self.clustering.labels_ is not None and len(self.clustering.labels_) == len(features_df):
            return self.clustering.labels_

        feature_cols = [
            c for c in features_df.columns
            if c not in ("window_start", "date")
        ]
        if "productivity_ratio" not in features_df.columns:
            return None

        result = np.zeros(len(features_df), dtype=int)
        for i, (_, row) in enumerate(features_df.iterrows()):
            pr = float(row.get("productivity_ratio", 0.5))
            er = float(row.get("entertainment_ratio", 0.0))
            sr = float(row.get("social_ratio", 0.0))
            ir = float(row.get("idle_ratio", 0.0))

            if ir > 0.7:
                result[i] = 4
            elif pr > 0.7:
                result[i] = 0
            elif pr > 0.4:
                result[i] = 1
            elif er > 0.4 or sr > 0.4:
                result[i] = 3
            else:
                result[i] = 2

        return result

    def save_all(self) -> None:
        """Save all trained models to disk."""
        self.clustering.save(self.models_dir / "clustering.joblib")
        self.classifier.save(self.models_dir / "classifier.joblib")

        hmm_data: dict[str, object] = {
            "transition_matrix": self.hmm.transition_matrix,
            "state_names": self.hmm.state_names,
            "n_states": self.hmm.n_states,
        }
        joblib.dump(hmm_data, str(self.models_dir / "hmm.joblib"))

    def load_all(self) -> bool:
        """Load all models from disk. Returns True if successful."""
        clustering_path = self.models_dir / "clustering.joblib"
        classifier_path = self.models_dir / "classifier.joblib"
        hmm_path = self.models_dir / "hmm.joblib"

        if not all(p.exists() for p in [clustering_path, classifier_path, hmm_path]):
            return False

        try:
            self.clustering = BehaviorClustering.load(clustering_path)
            self.classifier = FocusClassifier.load(classifier_path)

            hmm_data: dict[str, object] = joblib.load(str(hmm_path))
            self.hmm = BehaviorHMM(n_states=int(hmm_data.get("n_states", 5)))
            self.hmm.transition_matrix = np.array(hmm_data.get("transition_matrix"))
            self.hmm._is_fitted = True
            return True
        except (FileNotFoundError, EOFError, KeyError, ValueError):
            return False
