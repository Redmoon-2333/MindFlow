"""Training pipeline orchestrator for MindFlow.

Combines synthetic/real data, feature extraction, weak-supervision labeling,
baseline update, ML model training, and report generation into a single
``run_training()`` entry point with a typed ``TrainingReport`` result.

Each step is independently testable.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

# ── Domain imports (read-only — we do NOT modify domain/) ─────────────────────
from mindflow.domain.events import ActivityEvent
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.train.models import ModelManager
from mindflow.train.v2 import (
    evaluate_v2_candidates,
    evaluate_v2_quality_gate,
    prepare_v2_training_data,
)


def _tag_from_filename(filename: str) -> str:
    """Extract the version tag from e.g. ``classifier-20260731_120000_ab12.pkl``."""
    if "-" not in filename:
        return ""
    return filename.removesuffix(".pkl").split("-", 1)[1]


@dataclass
class TrainingReport:
    """Full report from one training pipeline run.

    Serialized as JSON report file alongside model artifacts.
    """

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: str = "synthetic_v2"  # "synthetic_v2" | "db"
    total_records: int = 0
    windows_extracted: int = 0
    n_focus: int = 0
    n_distract: int = 0
    avg_confidence: float = 0.0
    baseline_updated: int = 0
    filtered_windows: int = 0
    quality_gate: dict[str, Any] = field(default_factory=dict)
    activated: bool = False
    clustering: dict[str, Any] = field(default_factory=dict)
    classifier: dict[str, Any] = field(default_factory=dict)
    hmm: dict[str, Any] = field(default_factory=dict)
    saved_models: dict[str, str] = field(default_factory=dict)
    version_tag: str | None = None
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    model_mode: str = "rule_engine_only"
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_training(
    source: Literal["synthetic_v2", "db"] = "synthetic_v2",
    data_dir: str | Path = Path("data"),
    models_dir: str | Path = Path("data/models"),
    user_id: int = 1,
    events: Sequence[ActivityEvent] | None = None,
    days: int = 14,
    samples_per_hour: int = 12,
    seed: int = 42,
    num_users: int = 1,
    include_procrastination: bool = False,
    user_profiles: list[str] | None = None,
    min_confidence: float = 0.2,
    min_baseline_samples: int = 30,
    feature_windows: list[dict[str, Any]] | None = None,
    feedback_sessions: list[dict[str, Any]] | None = None,
    use_window_labels: bool = False,
    calibration: str | None = "sigmoid",
) -> TrainingReport:
    """Run the full training pipeline.

    Both sources use pre-computed V2 feature windows. Synthetic training
    generates schema-v3 windows from student archetypes; database training
    consumes stored windows plus explicit focus feedback.

    Args:
        source: ``"synthetic_v2"`` generates windows; ``"db"`` uses supplied windows.
        data_dir: Reserved compatibility path for callers of the former V1 pipeline.
        models_dir: Directory for model artifacts.
        user_id: Reserved compatibility identifier for callers of the former V1 pipeline.
        events: Reserved compatibility input; V2 database training uses ``feature_windows``.
        days: Number of synthetic days per archetype.
        samples_per_hour: Deprecated; V2 windows have a fixed five-minute resolution.
        seed: Random seed for synthetic data.
        num_users: Number of archetypes to generate when ``user_profiles`` is omitted.
        include_procrastination: Deprecated; V2 archetypes already model procrastination.
        user_profiles: Explicit archetype IDs; takes precedence over ``num_users``.
        min_confidence: Reserved compatibility input for callers of the former V1 pipeline.
        min_baseline_samples: Reserved compatibility input for the former V1 pipeline.

    Returns:
        ``TrainingReport`` with all metrics and artifact paths.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(models_dir)

    if samples_per_hour != 12:
        warnings.warn(
            "samples_per_hour is ignored because V2 uses fixed five-minute windows",
            DeprecationWarning,
            stacklevel=2,
        )
    if include_procrastination:
        warnings.warn(
            "include_procrastination is ignored because V2 archetypes already model it",
            DeprecationWarning,
            stacklevel=2,
        )

    report = TrainingReport(source=source)
    if source == "db":
        if feature_windows:
            return _run_v2_training(
                feature_windows=feature_windows,
                feedback_sessions=feedback_sessions or [],
                models_path=models_path,
                source=source,
                use_window_labels=use_window_labels,
                calibration=calibration,
            )
        raise ValueError(
            "No V2 feature windows found in database. "
            "Run --source synthetic_v2 first to generate synthetic windows, "
            "or collect more activity data."
        )

    # ── V2 synthetic data path ──────────────────────────────────────────
    if source == "synthetic_v2":
        print("[synth-v2] Generating synthetic v2 feature windows from archetypes...")
        from mindflow.train.synthetic_v2 import generate_v2_synthetic_data
        from mindflow.train.user_profiles import list_archetype_ids

        if user_profiles is not None:
            profile_ids = user_profiles
        else:
            available_profiles = list_archetype_ids()
            if not 1 <= num_users <= len(available_profiles):
                raise ValueError(
                    f"num_users must be between 1 and {len(available_profiles)}"
                )
            profile_ids = available_profiles[:num_users]
        syn_windows, syn_feedback = generate_v2_synthetic_data(
            archetype_ids=profile_ids,
            days_per_archetype=days,
            seed=seed,
            sample_explicit_ratio=0.3,
        )
        report.total_records = len(syn_windows)
        return _run_v2_training(
            feature_windows=syn_windows,
            feedback_sessions=syn_feedback,
            models_path=models_path,
            source=source,
            use_window_labels=False,  # synthetic data has no window-label source
            calibration=calibration,
        )

    # V1 pipeline (raw-event-based feature extraction) has been removed.
    # All training now goes through the V2 24-dim feature schema
    # (synthetic_v2 or db with pre-computed feature windows).
    raise ValueError(
        f"Unsupported training source: {source!r}. "
        "Use --source synthetic_v2 for synthetic data or --source db for "
        "real V2 feature windows from the database."
    )


def _extract_window_labels(
    feature_windows: list[dict[str, Any]],
) -> dict[str, int]:
    """Map feature-window id -> int label (1=focus, 0=distracted, -1=mixed).

    Reads the ``label`` column that user-calibrated window annotations store
    (``focus``/``distracted`` strings, or raw ints).  Returns an empty dict
    when no usable labels exist; unknown/None labels are skipped.
    """
    labels: dict[str, int] = {}
    for row in feature_windows:
        wid = str(row.get("id", ""))
        raw = row.get("label")
        if not wid or raw is None:
            continue
        if isinstance(raw, str):
            label = {"focus": 1, "distracted": 0, "mixed": -1}.get(raw.strip().lower())
        else:
            try:
                label = int(raw)
            except (TypeError, ValueError):
                continue
        if label is not None:
            labels[wid] = label
    return labels


def _run_v2_training(
    *,
    feature_windows: list[dict[str, Any]],
    feedback_sessions: list[dict[str, Any]],
    models_path: Path,
    source: str,
    use_window_labels: bool = False,
    calibration: str | None = "sigmoid",
) -> TrainingReport:
    report = TrainingReport(
        source=source,
        total_records=len(feature_windows),
        windows_extracted=len(feature_windows),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    window_labels = _extract_window_labels(feature_windows) if use_window_labels else None
    training_data = prepare_v2_training_data(
        feature_windows, feedback_sessions, window_labels=window_labels,
    )
    # Evaluation and deployment use the same explicit-only, weighted samples.
    explicit_mask = training_data.explicit_mask
    train_X = training_data.features[explicit_mask]
    train_y = training_data.labels[explicit_mask]
    train_w = training_data.sample_weights[explicit_mask]
    report.filtered_windows = len(feature_windows) - len(training_data.features)
    report.n_focus = int(np.sum(train_y == 1))
    report.n_distract = int(np.sum(train_y == 0))
    report.avg_confidence = round(float(np.mean(train_w)), 4) if len(train_w) else 0.0
    evaluation = evaluate_v2_candidates(training_data, calibration=calibration)
    report.evaluation = evaluation
    report.quality_gate = evaluate_v2_quality_gate(
        evaluation,
        explicit_feedback_count=training_data.explicit_feedback_count,
        explicit_focus_count=training_data.explicit_focus_count,
        explicit_distract_count=training_data.explicit_distract_count,
        distinct_feedback_days=training_data.distinct_feedback_days,
    )
    report.model_mode = "rule_engine_only"

    v2_models_path = models_path / "v2"
    v2_models_path.mkdir(parents=True, exist_ok=True)
    if len(train_X) >= 10 and len(np.unique(train_y)) >= 2:
        manager = ModelManager(
            models_dir=v2_models_path,
            use_ensemble=True,
            calibration=calibration,  # matches evaluate_v2_candidates()
        )
        summary = manager.train_all(
            train_X,
            training_data.feature_names,
            train_y,
            sample_weight=train_w,
            min_confidence=0.0,
        )
        report.clustering = summary.clustering
        report.classifier = {**summary.classifier, "grouped_evaluation": evaluation}
        if evaluation.get("candidate", {}).get("balanced_accuracy") is not None:
            report.classifier["balanced_accuracy"] = evaluation["candidate"]["balanced_accuracy"]
        report.hmm = summary.hmm
        should_activate = bool(report.quality_gate["passed"])
        saved = manager.save_all(
            activate=should_activate,
            manifest={
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": list(training_data.feature_names),
                "explicit_feedback_count": training_data.explicit_feedback_count,
                "distinct_feedback_days": training_data.distinct_feedback_days,
                "quality_gate": report.quality_gate,
                "evaluation": evaluation,
                "source": source,
            },
        )
        report.saved_models = saved
        report.activated = should_activate
        report.model_mode = "ready" if should_activate else "shadow"
        report.version_tag = _tag_from_filename(saved.get("classifier", "")) or None

    report_path = v2_models_path / "training_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
