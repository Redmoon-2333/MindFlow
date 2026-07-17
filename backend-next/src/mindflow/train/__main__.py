"""CLI entry point for the MindFlow training pipeline.

Usage:

    # Train with synthetic data (default 14 days)
    python -m mindflow.train

    # Train with synthetic data, explicit args
    python -m mindflow.train --source synthetic --days 7 --samples-per-hour 6

    # Train with real data from database
    python -m mindflow.train --source db

    # List available model versions
    python -m mindflow.train --list-versions

    # Rollback to a specific version
    python -m mindflow.train --rollback 20260717
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mindflow.train.models import ModelManager
from mindflow.train.pipeline import run_training


def _resolve_project_root() -> Path:
    """Walk up from cwd to find the backend-next project root (contains pyproject.toml)."""
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MindFlow behavior models",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "db"],
        default="synthetic",
        help="Data source: synthetic (default) or real db events",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Number of synthetic days (default: 14)",
    )
    parser.add_argument(
        "--samples-per-hour",
        type=int,
        default=12,
        dest="samples_per_hour",
        help="Synthetic data samples per hour (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data (default: 42)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        dest="data_dir",
        help="Data directory (default: <project-root>/data)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="",
        dest="models_dir",
        help="Models directory (default: <project-root>/data/models)",
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        dest="list_versions",
        help="List available model versions and exit",
    )
    parser.add_argument(
        "--rollback",
        type=str,
        default="",
        help="Rollback models to given YYYYMMDD version and exit",
    )

    args = parser.parse_args()

    project_root = _resolve_project_root()
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data"
    models_dir = (
        Path(args.models_dir) if args.models_dir else project_root / "data" / "models"
    )

    manager = ModelManager(models_dir=models_dir)

    # ── List versions ─────────────────────────────────────────────────────
    if args.list_versions:
        versions = manager.list_versions()
        if not versions:
            print("No model versions found in", models_dir)
            sys.exit(0)
        print(f"Available model versions ({len(versions)}):")
        for v in versions:
            current = " (current)" if v == manager.current_version_tag else ""
            print(f"  - {v}{current}")
        sys.exit(0)

    # ── Rollback ──────────────────────────────────────────────────────────
    if args.rollback:
        tag = args.rollback.strip()
        print(f"Rolling back to version {tag}...")
        if manager.rollback(tag):
            print(f"  Rollback successful. Models from {tag} are now active.")
            print("  latest.json updated.")
        else:
            print(f"  Rollback FAILED. Version {tag} not found or incomplete.")
            print("  Use --list-versions to see available versions.")
            sys.exit(1)
        sys.exit(0)

    # ── Run training ──────────────────────────────────────────────────────
    print("=" * 60)
    title = (
        "MindFlow Model Training — Real Data"
        if args.source == "db"
        else "MindFlow Model Training — Synthetic Data"
    )
    print(title)
    print("=" * 60)

    report = run_training(
        source=args.source,
        data_dir=data_dir,
        models_dir=models_dir,
        days=args.days,
        samples_per_hour=args.samples_per_hour,
        seed=args.seed,
    )

    if report.total_records == 0:
        print("\nTraining pipeline did not complete (no data).")
        sys.exit(1)

    print(f"\nTraining report saved to {models_dir / 'training_report.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
