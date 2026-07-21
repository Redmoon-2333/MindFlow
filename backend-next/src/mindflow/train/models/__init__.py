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

This was a single 800-line ``models.py`` module; it is now a package split by
model (``clustering``, ``classifier``, ``hmm``, ``manager``) with shared
dataclasses in ``types``. This ``__init__`` re-exports every public name so
``from mindflow.train.models import X`` keeps working unchanged.
"""

from __future__ import annotations

from mindflow.train.models.classifier import FocusClassifier
from mindflow.train.models.clustering import BehaviorClustering
from mindflow.train.models.hmm import BehaviorHMM
from mindflow.train.models.manager import ModelManager
from mindflow.train.models.types import BehaviorCluster, TrainingSummary

__all__ = [
    "BehaviorCluster",
    "BehaviorClustering",
    "BehaviorHMM",
    "FocusClassifier",
    "ModelManager",
    "TrainingSummary",
]
