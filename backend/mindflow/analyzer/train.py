"""Training script for MindFlow behavior models.

Usage:
    cd backend && python -m mindflow.analyzer.train

Generates synthetic data, extracts features, creates pseudo-labels,
trains clustering, classifier, and HMM models, then saves them to disk.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np

from mindflow.analyzer.data_pipeline import (
    generate_synthetic_data,
    BehaviorFeatureExtractor,
    AppClassifier,
)
from mindflow.analyzer.ml_models import ModelManager


def main() -> None:
    """Run the full training pipeline and print summary."""
    print("=" * 60)
    print("MindFlow Behavior Model Training Pipeline")
    print("=" * 60)

    print("\n[1/5] Generating synthetic activity data (14 days, 12 samples/hour)...")
    raw_data = generate_synthetic_data(days=14, samples_per_hour=12)
    print(f"  Generated {len(raw_data)} activity records")
    print(f"  Date range: {raw_data['timestamp'].min()} → {raw_data['timestamp'].max()}")
    print(f"  Idle ratio: {raw_data['is_idle'].mean():.2%}")

    print("\n[2/5] Extracting behavioral features...")
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features_df = extractor.extract_session_features(raw_data)
    print(f"  Extracted {len(features_df)} session windows")
    print(f"  Features: {extractor.get_feature_names()}")

    if features_df.empty:
        print("  ERROR: No feature windows extracted. Exiting.")
        sys.exit(1)

    print("\n[3/5] Creating pseudo-labels for supervised training...")
    feature_cols = [
        c for c in features_df.columns
        if c not in ("window_start",) and features_df[c].dtype in ("float64", "float32", "int64")
    ]

    labels = np.where(features_df["productivity_ratio"] > 0.6, 1, 0)
    n_focus = int(labels.sum())
    n_distraction = int(len(labels) - n_focus)
    print(f"  Focus sessions: {n_focus} ({n_focus/len(labels):.1%})")
    print(f"  Distraction sessions: {n_distraction} ({n_distraction/len(labels):.1%})")

    print("\n[4/5] Training all models...")
    manager = ModelManager(models_dir=Path("data/models"))
    summary = manager.train_all(features_df, labels)

    print("  Clustering results:")
    clustering = summary.get("clustering", {})
    for cluster in clustering.get("clusters", []):
        print(
            f"    Cluster {cluster['id']} ({cluster['label']}): "
            f"{cluster['count']} samples, focus_score={cluster['avg_focus_score']:.3f}"
        )
    if clustering.get("noise_points", 0) > 0:
        print(f"    Noise points: {clustering['noise_points']}")

    print("\n  Classifier results:")
    classifier_summary = summary.get("classifier", {})
    if "error" in classifier_summary:
        print(f"    {classifier_summary['error']}")
    else:
        print(f"    Accuracy:  {classifier_summary.get('accuracy', 'N/A')}")
        print(f"    Precision: {classifier_summary.get('precision', 'N/A')}")
        print(f"    Recall:    {classifier_summary.get('recall', 'N/A')}")
        print(f"    F1 Score:  {classifier_summary.get('f1', 'N/A')}")
        print(f"    CV Mean:   {classifier_summary.get('cv_mean', 'N/A')} ± {classifier_summary.get('cv_std', 'N/A')}")

        print("\n  Feature importance:")
        importances = classifier_summary.get("feature_importance", {})
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:5]:
            print(f"    {name:<30s}: {imp:.4f}")

    print("\n  HMM Transition Matrix:")
    hmm_summary = summary.get("hmm", {})
    if "error" in hmm_summary:
        print(f"    {hmm_summary['error']}")
    else:
        state_names = hmm_summary.get("state_names", [])
        tm = hmm_summary.get("transition_matrix", [])
        header = " " * 12 + "".join(f"{s:>12s}" for s in state_names)
        print(f"    {header}")
        for i, row in enumerate(tm):
            row_str = "".join(f"{v:12.4f}" for v in row)
            print(f"    {state_names[i]:12s}{row_str}")

        steady = hmm_summary.get("steady_state", [])
        print("\n  Steady-state distribution:")
        for name, prob in zip(state_names, steady):
            print(f"    {name:15s}: {prob:.4f}")

    print("\n[5/5] Saving models to disk...")
    manager.save_all()
    models_dir = manager.models_dir
    saved_files = list(models_dir.glob("*.joblib"))
    for f in sorted(saved_files):
        size_kb = f.stat().st_size / 1024
        print(f"  Saved: {f} ({size_kb:.1f} KB)")

    print("\n" + "=" * 60)
    print("Training complete. Models ready for inference.")
    print("=" * 60)

    summary_path = models_dir / "training_summary.json"
    serializable_summary = _make_serializable(summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved to: {summary_path}")


def _make_serializable(obj: object) -> object:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
