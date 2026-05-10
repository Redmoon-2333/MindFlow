"""Training pipeline for MindFlow behavior models.

Builds personal baseline from collected data, runs clustering and HMM,
and validates deviation detection.

Usage:
    cd backend && python -m mindflow.analyzer.train
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

from mindflow.analyzer.data_pipeline import (
    generate_synthetic_data,
    BehaviorFeatureExtractor,
)
from mindflow.analyzer.ml_models import ModelManager, BehaviorClustering, BehaviorHMM
from mindflow.analyzer.baseline import BaselineModel
from mindflow.analyzer.deviation import DeviationDetector
from mindflow.analyzer.context_packer import LLMContextPacker


def main() -> None:
    print("=" * 60)
    print("MindFlow Baseline + Deviation Detection Pipeline")
    print("=" * 60)

    print("\n[1/5] Generating synthetic activity data (14 days, 12 samples/hour)...")
    raw_data = generate_synthetic_data(days=14, samples_per_hour=12)
    print(f"  Generated {len(raw_data):,} activity records")

    print("\n[2/5] Extracting behavioral features...")
    extractor = BehaviorFeatureExtractor(window_minutes=30)
    features_df = extractor.extract_session_features(raw_data)
    print(f"  Extracted {len(features_df)} session windows")
    feature_names = extractor.get_feature_names()
    print(f"  Features ({len(feature_names)}): {feature_names[:6]}...")

    if features_df.empty:
        print("  ERROR: No feature windows extracted. Exiting.")
        return

    # Split: first 7 days → baseline, last 7 days → test
    if "window_start" in features_df.columns:
        features_df["_date"] = pd.to_datetime(features_df["window_start"]).dt.date
        all_dates = sorted(features_df["_date"].unique())
        split_date = all_dates[len(all_dates) // 2]
        baseline_df = features_df[features_df["_date"] < split_date].copy()
        test_df = features_df[features_df["_date"] >= split_date].copy()
        print(f"\n  Baseline period: {baseline_df['_date'].min()} → {baseline_df['_date'].max()} ({len(baseline_df)} windows)")
        print(f"  Test period:     {test_df['_date'].min()} → {test_df['_date'].max()} ({len(test_df)} windows)")
    else:
        baseline_df = features_df.iloc[:len(features_df)//2].copy()
        test_df = features_df.iloc[len(features_df)//2:].copy()

    print("\n[3/5] Building personal behavior baseline...")
    baseline = BaselineModel(user_id=1)
    n_updated = baseline.update(baseline_df)
    print(f"  Baseline updated with {n_updated} windows")
    print(f"  Sufficient data: {baseline.has_sufficient_data(30)}")

    # Show sample baseline stats
    rep_stats = baseline.get_stats(10, 1)
    print(f"  Sample baseline (Tue 10am):")
    for feat in ["switch_frequency", "unique_app_count", "max_app_duration"]:
        if feat in rep_stats and rep_stats[feat]["n"] >= 2:
            s = rep_stats[feat]
            print(f"    {feat}: mean={s['mean']:.1f}, std={s['std']:.1f}, n={s['n']}")

    print(f"  Top apps (Tue 10am): {baseline.get_top_apps(10, 1, 5)}")

    print("\n[4/5] Detecting deviations on test data...")
    detector = DeviationDetector(baseline)
    anomalies = detector.analyze_dataframe(test_df)
    daily = detector.daily_summary(test_df)

    print(f"  Total test windows: {daily['total_windows']}")
    print(f"  Anomalies detected: {daily['anomaly_count']} ({daily['anomaly_ratio']:.1%})")
    print(f"  Severity breakdown: {daily['severity_counts']}")
    print(f"  Avg deviation: {daily['average_deviation']:.3f}")
    print(f"  Most anomalous hour: {daily['most_anomalous_hour']}")

    if anomalies:
        print(f"\n  Top 3 anomalies:")
        for i, a in enumerate(anomalies[:3]):
            print(f"    [{a['severity'].upper()}] {a['window_start']} "
                  f"deviation={a['overall_deviation']:.2f}")
            for d in a.get("top_deviations", []):
                print(f"      {d['feature']}: {d['z_score']:+.2f} ({d['direction']})")

    print("\n[5/5] Clustering + HMM on full data...")
    feature_cols = [c for c in features_df.columns
                    if c not in ("window_start", "_date", "date")
                    and features_df[c].dtype in ("float64", "float32", "int64")]
    X = features_df[feature_cols].to_numpy(dtype=np.float64)

    clustering = BehaviorClustering(method="dbscan")
    cluster_info = clustering.fit(X)
    print(f"  Clusters: {len([c for c in cluster_info if c.cluster_id != -1])}")
    for c in sorted(cluster_info, key=lambda x: x.avg_focus_score, reverse=True):
        print(f"    {c.label}: {c.sample_count} samples, focus_score={c.avg_focus_score:.3f}")

    # Build state sequences from clustering labels for HMM
    if clustering.labels_ is not None and "window_start" in features_df.columns:
        features_df["_state"] = clustering.labels_
        hmm = BehaviorHMM()
        sequences = []
        for _, day_df in features_df.groupby("_date" if "_date" in features_df.columns else "window_start"):
            day_states = day_df.sort_values("window_start")["_state"].values
            if len(day_states) >= 2:
                sequences.append(day_states.astype(int))
        if sequences:
            hmm.fit(sequences)
            tm = hmm.get_transition_matrix()
            print(f"\n  HMM Transition Matrix (first 3 states):")
            names = hmm.state_names[:3]
            for i, row in enumerate(tm[:3]):
                print(f"    {names[i]:15s}: " + " ".join(f"{v:.3f}" for v in row[:3]))

    # Generate LLM context
    print("\n--- LLM Context Sample ---")
    packer = LLMContextPacker()
    ctx = packer.pack_daily_report(
        baseline=baseline,
        anomalies=anomalies[:5],
        daily_summary=daily,
        focus_score=round(float(features_df["productivity_ratio"].mean()) * 100, 1),
        top_apps=[{"app": "vscode", "minutes": 180}, {"app": "chrome", "minutes": 120}],
    )
    # Print just the first 800 chars
    print(ctx[:800] + "..." if len(ctx) > 800 else ctx)

    # Save artifacts
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    baseline.save(models_dir / "baseline_user1.json")
    clustering.save(models_dir / "clustering.joblib")
    print(f"\nArtifacts saved to {models_dir}/")
    print("  - baseline_user1.json")
    print("  - clustering.joblib")


if __name__ == "__main__":
    main()
