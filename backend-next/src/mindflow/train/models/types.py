"""Shared data types for the train.models package.

``BehaviorCluster`` is produced by ``BehaviorClustering`` and consumed by
``ModelManager``; ``TrainingSummary`` is returned by ``ModelManager.train_all``.
They live here (rather than in any single model module) so both the clustering
model and the manager can import them without a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy.typing as npt


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
