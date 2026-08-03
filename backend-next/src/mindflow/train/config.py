"""Central ML hyperparameter and evaluation policy.

Keeping these in one module means experiments and production share the same
values; a change here is a deliberate, reviewable model change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierConfig:
    random_state: int = 42
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1


@dataclass(frozen=True)
class ClusteringConfig:
    method: str = "dbscan"
    min_samples_fraction: float = 0.02
    kmeans_max_clusters: int = 5


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 5
    n_iter: int = 100
    tol: float = 1e-4


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    group_folds: int = 4
    min_explicit_samples: int = 10
    min_feedback_dates: int = 3
    calibration_bins: int = 10


CLASSIFIER_CONFIG = ClassifierConfig()
CLUSTERING_CONFIG = ClusteringConfig()
HMM_CONFIG = HMMConfig()
TRAIN_CONFIG = TrainConfig()
