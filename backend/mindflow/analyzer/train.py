"""Training pipeline for MindFlow behavior models.

Builds personal baseline from collected data, runs clustering and HMM,
and validates deviation detection.

Usage:
    cd backend && python -m mindflow.analyzer.train              # synthetic data
    cd backend && python -m mindflow.analyzer.train --from-db    # real data
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from mindflow.analyzer.data_pipeline import (
    generate_synthetic_data,
    BehaviorFeatureExtractor,
)
from mindflow.analyzer.ml_models import ModelManager
from mindflow.analyzer.baseline import BaselineModel
from mindflow.analyzer.deviation import DeviationDetector
from mindflow.analyzer.labeling import ConsensusLabeler
from mindflow.analyzer.context_packer import LLMContextPacker


def load_real_data() -> pd.DataFrame:
    """Load real activity data from the local SQLite database."""
    from mindflow.models.database import SessionLocal
    from mindflow.models.schemas import ActivityLog, User

    db = SessionLocal()
    try:
        user = db.query(User).first()
        if user is None:
            print("No user found in database. Start the collector first:")
            print("  curl -X POST http://localhost:8765/api/v1/collector/start")
            sys.exit(1)

        activities = (
            db.query(ActivityLog)
            .filter(ActivityLog.user_id == user.id)
            .order_by(ActivityLog.timestamp.asc())
            .all()
        )

        if len(activities) < 100:
            print(
                f"Only {len(activities)} activity records found. "
                f"Need at least 100 for meaningful training. "
                f"Let the collector run longer."
            )
            sys.exit(1)

        rows = [
            {
                "timestamp": a.timestamp,
                "process_name": a.process_name,
                "window_title": a.window_title or "",
                "duration_seconds": a.duration_seconds,
                "is_idle": a.is_idle,
            }
            for a in activities
        ]

        print(f"Loaded {len(rows)} real activity records from database")
        return pd.DataFrame(rows)
    finally:
        db.close()


def main(from_db: bool = False) -> None:
    print("=" * 60)
    title = "MindFlow Model Training — Real Data" if from_db else "MindFlow Model Training — Synthetic Data"
    print(title)
    print("=" * 60)

    # ── Step 1: Load data ──
    if from_db:
        print("\n[1/5] Loading real activity data from database...")
        raw_data = load_real_data()
    else:
        print("\n[1/5] Generating synthetic activity data (14 days, 12 samples/hour)...")
        raw_data = generate_synthetic_data(days=14, samples_per_hour=12)
        print(f"  Generated {len(raw_data):,} activity records")

    # ── Step 2: Extract features ──
    print("\n[2/5] Extracting behavioral features...")
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features_df = extractor.extract_session_features(raw_data)
    print(f"  Extracted {len(features_df)} session windows")
    feature_names = extractor.get_feature_names()
    print(f"  Features ({len(feature_names)}): {feature_names[:6]}...")

    if features_df.empty:
        print("  ERROR: No feature windows extracted. Exiting.")
        return

    # ── Step 3: Weak supervision labeling ──
    labeler = ConsensusLabeler()
    labels, confidences = labeler.label_dataframe(features_df)

    n_focus = int((labels == 1).sum())
    n_distract = int((labels == 0).sum())
    avg_conf = float(confidences.mean()) if len(confidences) > 0 else 0.0
    print(f"\n  Labels: {n_focus} focus, {n_distract} distraction")
    print(f"  Avg confidence: {avg_conf:.3f}")

    high_conf_mask = confidences >= 0.5
    print(f"  High-confidence samples: {high_conf_mask.sum()}/{len(confidences)}")

    # ── Step 4: Build baseline ──
    print("\n[3/5] Building personal behavior baseline...")
    baseline = BaselineModel(user_id=1)
    n_updated = baseline.update(features_df)
    print(f"  Baseline updated with {n_updated} windows")
    print(f"  Sufficient data: {baseline.has_sufficient_data(30)}")

    # Show sample stats
    rep_stats = baseline.get_stats(10, 1)
    any_built = any(
        s.get("n", 0) >= 2 for s in rep_stats.values()
    )
    if any_built:
        print(f"  Sample baseline (Tue 10am):")
        for feat in ["switch_frequency", "unique_app_count", "max_app_duration"]:
            if feat in rep_stats and rep_stats[feat]["n"] >= 2:
                s = rep_stats[feat]
                print(f"    {feat}: mean={s['mean']:.1f}, std={s['std']:.1f}, n={int(s['n'])}")
        print(f"  Top apps (Tue 10am): {baseline.get_top_apps(10, 1, 5)}")
    else:
        print("  Not enough data per time bucket yet. Need more days of collection.")

    # ── Step 5: Train ML models ──
    print("\n[4/5] Training ML models (clustering + classifier + HMM)...")
    manager = ModelManager(models_dir=Path("data/models"))
    summary = manager.train_all(
        features_df,
        labels,
        sample_weight=confidences,
        min_confidence=0.5,
    )

    print(f"\n  Clustering:")
    c = summary.get("clustering", {})
    print(f"    Clusters: {c.get('n_clusters', 0)}")
    for cl in c.get("clusters", []):
        print(f"    {cl['label']}: {cl['count']} samples, focus_score={cl['avg_focus_score']:.3f}")

    print(f"\n  Classifier:")
    cf = summary.get("classifier", {})
    if "error" in cf:
        print(f"    WARNING: {cf['error']}")
        print(f"    (need more data — at least 2 classes with 10+ samples each)")
    else:
        print(f"    Accuracy: {cf.get('accuracy', 'N/A')}")
        print(f"    Precision: {cf.get('precision', 'N/A')}")
        print(f"    Recall: {cf.get('recall', 'N/A')}")
        print(f"    F1: {cf.get('f1', 'N/A')}")
        print(f"    CV (5-fold): {cf.get('cv_mean', 'N/A')} ± {cf.get('cv_std', 'N/A')}")
        print(f"    Training samples: {cf.get('high_confidence_samples', 'N/A')}")
        print(f"    Filtered (low conf): {cf.get('filtered_low_confidence', 0)}")

    print(f"\n  HMM:")
    h = summary.get("hmm", {})
    if "error" in h:
        print(f"    WARNING: {h['error']}")
    else:
        tm = h.get("transition_matrix", [])
        names = h.get("state_names", [])
        if tm and names:
            print(f"    Transition Matrix:")
            header = "    " + " " * 15 + " ".join(f"{n:>10s}" for n in names[:5])
            print(header)
            for i, row in enumerate(tm[:5]):
                line = f"    {names[i]:15s} " + " ".join(f"{v:10.3f}" for v in row[:5])
                print(line)

            steady = h.get("steady_state", [])
            if steady:
                print(f"\n    Steady-state distribution:")
                for name, p in zip(names, steady):
                    print(f"    {name:15s}: {p:.3f}")

    # ── Step 6: Deviation detection on same data ──
    print("\n[5/5] Deviation detection check...")
    detector = DeviationDetector(baseline)
    daily = detector.daily_summary(features_df)
    anomalies = detector.analyze_dataframe(features_df)

    print(f"  Total windows: {daily['total_windows']}")
    print(f"  Anomalies: {daily['anomaly_count']} ({daily['anomaly_ratio']:.1%})")
    print(f"  Severity: {daily['severity_counts']}")
    print(f"  Avg deviation: {daily['average_deviation']:.3f}")

    if anomalies:
        print(f"\n  Top 3 anomalies:")
        for i, a in enumerate(anomalies[:3]):
            print(f"    [{a['severity'].upper()}] {a['window_start']} "
                  f"deviation={a['overall_deviation']:.2f}")
            for d in a.get("top_deviations", []):
                print(f"      {d['feature']}: {d['z_score']:+.2f} ({d['direction']})")

    # ── Save all ──
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    baseline.save(models_dir / "baseline_user1.json")
    manager.save_all()

    # Save training summary
    summary_data = {
        "data_source": "real" if from_db else "synthetic",
        "total_samples": len(features_df),
        "n_focus": n_focus,
        "n_distract": n_distract,
        "avg_confidence": avg_conf,
        "classifier": {
            k: v for k, v in cf.items()
            if k not in ("classification_report", "feature_importance")
        },
        "clustering": {
            "n_clusters": c.get("n_clusters", 0),
            "noise_points": c.get("noise_points", 0),
        },
    }
    (models_dir / "training_summary.json").write_text(
        json.dumps(summary_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("Artifacts saved to data/models/")
    for f in sorted(models_dir.glob("*")):
        if f.is_file():
            print(f"  - {f.name}")
    print(f"\nNext: restart the API server to pick up new models.")
    if not from_db:
        print("After collecting real data, re-run with: python -m mindflow.analyzer.train --from-db")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MindFlow behavior models")
    parser.add_argument(
        "--from-db",
        action="store_true",
        dest="from_db",
        help="Use real activity data from the local database instead of synthetic data",
    )
    args = parser.parse_args()
    main(from_db=args.from_db)
