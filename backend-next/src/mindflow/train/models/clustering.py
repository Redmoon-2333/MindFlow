"""Unsupervised behavior-pattern clustering (DBSCAN with KMeans fallback)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from mindflow.train.models.types import BehaviorCluster


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
