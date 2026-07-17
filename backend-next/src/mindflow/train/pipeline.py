"""Training pipeline orchestrator for MindFlow.

Combines synthetic/real data, feature extraction, weak-supervision labeling,
baseline update, ML model training, and report generation into a single
``run_training()`` entry point with a typed ``TrainingReport`` result.

Each step is independently testable.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

# ── Domain imports (read-only — we do NOT modify domain/) ─────────────────────
from mindflow.domain.baseline import BaselineModel
from mindflow.domain.events import ActivityEvent, make_event
from mindflow.domain.labeling import ConsensusLabeler
from mindflow.train.features import BehaviorFeatureExtractor
from mindflow.train.models import ModelManager
from mindflow.train.synthetic_data import generate_synthetic_data


@dataclass
class TrainingReport:
    """Full report from one training pipeline run.

    Serialized as JSON report file alongside model artifacts.
    """

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: str = "synthetic"  # "synthetic" | "db"
    total_records: int = 0
    windows_extracted: int = 0
    n_focus: int = 0
    n_distract: int = 0
    avg_confidence: float = 0.0
    baseline_updated: int = 0
    clustering: dict[str, Any] = field(default_factory=dict)
    classifier: dict[str, Any] = field(default_factory=dict)
    hmm: dict[str, Any] = field(default_factory=dict)
    saved_models: dict[str, str] = field(default_factory=dict)
    version_tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_training(
    source: Literal["synthetic", "db"] = "synthetic",
    data_dir: str | Path = Path("data"),
    models_dir: str | Path = Path("data/models"),
    user_id: int = 1,
    events: Sequence[ActivityEvent] | None = None,
    days: int = 14,
    samples_per_hour: int = 12,
    seed: int = 42,
    min_confidence: float = 0.5,
    min_baseline_samples: int = 30,
) -> TrainingReport:
    """Run the full training pipeline.

    Steps:
      1. Load or generate raw activity data.
      2. Extract behavioral features (30-min windows).
      3. Apply weak-supervision labeling (ConsensusLabeler).
      4. Update personal baseline (BaselineModel).
      5. Train ML models (clustering + classifier + HMM).
      6. Save models (versioned) + training report JSON.

    Args:
        source: ``"synthetic"`` generates data; ``"db"`` requires ``events``.
        data_dir: Directory for data artifacts (baseline, report).
        models_dir: Directory for model artifacts.
        user_id: User identifier for baseline.
        events: Activity events (required when source="db").
        days: Number of synthetic days (source="synthetic" only).
        samples_per_hour: Synthetic data resolution.
        seed: Random seed for synthetic data.
        min_confidence: Minimum confidence for supervised training.
        min_baseline_samples: Minimum samples for baseline sufficiency check.

    Returns:
        ``TrainingReport`` with all metrics and artifact paths.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    models_path = Path(models_dir)

    report = TrainingReport(source=source)

    # ── Step 1: Load / generate data ──────────────────────────────────────
    raw_rows: list[dict[str, Any]] = []
    if source == "synthetic":
        print("[1/6] Generating synthetic activity data...")
        raw_rows = generate_synthetic_data(
            days=days,
            samples_per_hour=samples_per_hour,
            seed=seed,
        )
        report.total_records = len(raw_rows)
        print(f"       Generated {len(raw_rows):,} activity records")

        _rows_for_features: Sequence[ActivityEvent] = [
            make_event(
                user_id=user_id,
                timestamp_utc=r["timestamp"],
                duration_s=r["duration_seconds"],
                app_name=r["process_name"],
                window_title=r["window_title"],
                process_name=r["process_name"],
                is_idle=bool(r["is_idle"]),
            )
            for r in raw_rows
        ]
    elif source == "db":
        if events is None or len(events) == 0:
            print("[1/6] ERROR: source='db' requires non-empty events list.", file=sys.stderr)
            return report
        print(f"[1/6] Using {len(events)} real activity events from database...")
        report.total_records = len(events)
        _rows_for_features = list(events)
    else:
        raise ValueError(f"Unknown source: {source!r}")

    # ── Step 2: Extract features ──────────────────────────────────────────
    print("[2/6] Extracting behavioral features...")
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    feature_rows = extractor.extract_session_features(_rows_for_features)
    report.windows_extracted = len(feature_rows)
    print(f"       Extracted {len(feature_rows)} session windows")
    if not feature_rows:
        print("       ERROR: No features extracted. Exiting.")
        return report

    # ── Step 3: Weak supervision labeling ─────────────────────────────────
    print("[3/6] Applying weak-supervision labeling...")
    labeler = ConsensusLabeler()
    labels, confidences = labeler.label_dataframe(feature_rows)

    report.n_focus = int(sum(labels))
    report.n_distract = len(labels) - report.n_focus
    report.avg_confidence = round(
        sum(confidences) / len(confidences), 4
    ) if confidences else 0.0
    print(f"       Labels: {report.n_focus} focus, {report.n_distract} distraction")
    print(f"       Avg confidence: {report.avg_confidence:.3f}")

    # ── Step 4: Build baseline ────────────────────────────────────────────
    print("[4/6] Building personal behavior baseline...")
    # Ensure feature rows have the keys BaselineModel expects
    _rows_with_process = list(feature_rows)
    if source == "synthetic" and raw_rows:
        # Add process_name from raw data to first matching window
        _enrich_with_process(_rows_with_process, raw_rows)

    baseline = BaselineModel(user_id=user_id)
    # feature rows are dict[str, Any]; update() accepts Mapping — cast the
    # list invariance away instead of silencing the checker (slop-scan fix).
    baseline_rows = cast("list[Mapping[str, Any]]", _rows_with_process)
    report.baseline_updated = baseline.update(baseline_rows)
    has_data = baseline.has_sufficient_data(min_baseline_samples)
    print(f"       Baseline updated with {report.baseline_updated} windows")
    print(f"       Sufficient data: {has_data}")

    # ── Step 5: Train ML models ───────────────────────────────────────────
    print("[5/6] Training ML models (clustering + classifier + HMM)...")
    feature_names = extractor.get_feature_names()

    # Build feature matrix (same columns as old backend)
    X = _build_feature_matrix(feature_rows, feature_names)
    y = np.array(labels, dtype=np.int32)
    w = np.array(confidences, dtype=np.float64)

    manager = ModelManager(models_dir=models_path)
    summary = manager.train_all(
        X, feature_names, y,
        sample_weight=w,
        min_confidence=min_confidence,
    )

    report.clustering = summary.clustering
    report.classifier = summary.classifier
    report.hmm = summary.hmm

    # Print summary (matching old train.py style)
    _print_clustering_summary(summary.clustering)
    _print_classifier_summary(summary.classifier)
    _print_hmm_summary(summary.hmm)

    # ── Step 6: Save artifacts ────────────────────────────────────────────
    print("[6/6] Saving artifacts...")

    # Baseline
    baseline_path = data_path / f"baseline_user{user_id}.json"
    baseline.save(baseline_path)

    # Models (versioned)
    saved = manager.save_all()
    report.saved_models = saved
    report.version_tag = manager.current_version_tag

    # Training report JSON
    report_path = models_path / "training_report.json"
    report_data = report.to_dict()
    # Remove non-serializable fields
    report_data["saved_models"] = saved
    report_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("Artifacts saved:")
    for f in sorted(models_path.glob("*")):
        if f.is_file():
            print(f"  - {f.name}")
    if baseline_path.exists():
        print(f"  - {baseline_path.name}")
    print("  - training_report.json")
    print("\nNext: restart the API server to pick up new models.")

    return report


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_feature_matrix(
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> npt.NDArray[Any]:
    """Build numpy feature matrix from feature dicts."""
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    for i, row_dict in enumerate(rows):
        for j, col in enumerate(feature_names):
            X[i, j] = float(row_dict.get(col, 0.0))
    return X


def _enrich_with_process(
    feature_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> None:
    """Add process_name from raw rows to feature rows by time window proximity.

    This is approximate: we pick the most common app in the time range.
    Only needed for synthetic data path (real ActivityEvent already has it).
    """
    from collections import Counter

    # Sort raw rows by timestamp for sequential scanning
    sorted_raw = sorted(raw_rows, key=lambda r: r["timestamp"])

    window_minutes = 30

    for feat in feature_rows:
        ws_str = feat.get("window_start", "")
        if not ws_str:
            continue
        # Parse window_start as ISO
        try:
            ws = datetime.fromisoformat(ws_str)
        except (ValueError, TypeError):
            continue

        we = ws + timedelta(minutes=window_minutes)  # approx window end

        # Collect process names within window
        apps_in_window: list[str] = []
        for r in sorted_raw:
            ts = r["timestamp"]
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    continue
            if ws <= ts < we:
                apps_in_window.append(str(r.get("process_name", "unknown")))
            elif ts >= we:
                break

        if apps_in_window:
            feat["process_name"] = Counter(apps_in_window).most_common(1)[0][0]


def _print_clustering_summary(clustering: dict[str, Any]) -> None:
    """Print clustering summary to stdout (matching old train.py style)."""
    print("\n  Clustering:")
    print(f"    Clusters: {clustering.get('n_clusters', 0)}")
    for cl in clustering.get("clusters", []):
        print(
            f"    {cl['label']}: {cl['count']} samples, "
            f"focus_score={cl['avg_focus_score']:.3f}"
        )


def _print_classifier_summary(classifier: dict[str, Any]) -> None:
    """Print classifier summary to stdout."""
    print("\n  Classifier:")
    if "error" in classifier:
        print(f"    WARNING: {classifier['error']}")
    else:
        print(f"    Accuracy: {classifier.get('accuracy', 'N/A')}")
        print(f"    Precision: {classifier.get('precision', 'N/A')}")
        print(f"    Recall: {classifier.get('recall', 'N/A')}")
        print(f"    F1: {classifier.get('f1', 'N/A')}")
        print(f"    CV (5-fold): {classifier.get('cv_mean', 'N/A')} "
              f"+/- {classifier.get('cv_std', 'N/A')}")
        print(f"    Training samples: {classifier.get('high_confidence_samples', 'N/A')}")
        print(f"    Filtered (low conf): {classifier.get('filtered_low_confidence', 0)}")


def _print_hmm_summary(hmm: dict[str, Any]) -> None:
    """Print HMM summary to stdout."""
    print("\n  HMM:")
    if "error" in hmm:
        print(f"    WARNING: {hmm['error']}")
        return

    tm = hmm.get("transition_matrix", [])
    names = hmm.get("state_names", [])
    if tm and names:
        print("    Transition Matrix:")
        print(
            "    " + " " * 15 + " ".join(f"{n:>10s}" for n in names[:5])
        )
        for i, row in enumerate(tm[:5]):
            line = f"    {names[i]:15s} " + " ".join(f"{v:10.3f}" for v in row[:5])
            print(line)

    steady = hmm.get("steady_state", [])
    if steady:
        print("\n    Steady-state distribution:")
        for name, p in zip(names, steady, strict=True):
            print(f"    {name:15s}: {p:.3f}")
