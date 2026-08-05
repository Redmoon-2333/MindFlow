"""ML training pipeline for MindFlow behavior models.

Wave 8a migration from the old ``mindflow.analyzer`` package.
Provides V2 feature extraction (24-dim), clustering, HMM training,
versioned model persistence, and a CLI runner.

V1 pipeline (synthetic_data.py, raw event-based training) was
removed in the V2 consolidation — all training now uses the 24-dim
feature schema (train/v2.py / synthetic_v2.py).
"""

from mindflow.train.models import BehaviorClustering, BehaviorHMM, ModelManager
from mindflow.train.pipeline import TrainingReport, run_training
from mindflow.train.user_profiles import (
    EPISODES,
    PROFILES,
    ProcrastinationEpisode,
    StudentArchetype,
    get_archetype,
    get_episode,
    list_archetype_ids,
)

__all__ = [
    "BehaviorClustering",
    "BehaviorHMM",
    "ModelManager",
    "TrainingReport",
    "run_training",
    # User profiles
    "PROFILES",
    "EPISODES",
    "StudentArchetype",
    "ProcrastinationEpisode",
    "get_archetype",
    "get_episode",
    "list_archetype_ids",
]
