"""ML training pipeline for MindFlow behavior models.

Wave 8a migration from the old ``mindflow.analyzer`` package.
Provides synthetic data generation, feature extraction, clustering,
HMM training, versioned model persistence, and a CLI runner.
"""

from mindflow.train.features import BehaviorFeatureExtractor
from mindflow.train.models import BehaviorClustering, BehaviorHMM, ModelManager
from mindflow.train.pipeline import TrainingReport, run_training
from mindflow.train.synthetic_data import generate_synthetic_data

__all__ = [
    "generate_synthetic_data",
    "BehaviorFeatureExtractor",
    "BehaviorClustering",
    "BehaviorHMM",
    "ModelManager",
    "TrainingReport",
    "run_training",
]
