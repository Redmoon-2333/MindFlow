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
import json
import sqlite3
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import platformdirs

from mindflow.domain.events import ActivityEvent
from mindflow.train.models import ModelManager
from mindflow.train.pipeline import run_training


def _resolve_project_root() -> Path:
    """Walk up from cwd to find the backend-next project root (contains pyproject.toml)."""
    cwd = Path.cwd().resolve()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return cwd


def load_database_events(
    database_path: Path,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    now_utc: datetime | None = None,
) -> list[ActivityEvent]:
    if not database_path.exists():
        return []

    cutoff = now_utc or datetime.now(UTC)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=UTC)
    connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id, user_id, timestamp, duration_s, event_type, data_json "
            "FROM activity_events ORDER BY timestamp"
        ).fetchall()
    finally:
        connection.close()

    events: list[ActivityEvent] = []
    for row in rows:
        try:
            timestamp = datetime.fromisoformat(str(row["timestamp"]))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            timestamp = timestamp.astimezone(UTC)
            duration_s = float(row["duration_s"])
            if timestamp > cutoff or duration_s <= 0:
                continue
            if start_date is not None and timestamp.date() < start_date:
                continue
            if end_date is not None and timestamp.date() > end_date:
                continue
            payload = json.loads(str(row["data_json"]))
            payload["timestamp_utc"] = timestamp.isoformat()
            events.append(ActivityEvent.from_dict({
                "id": row["id"],
                "user_id": row["user_id"],
                "timestamp_utc": timestamp,
                "duration_s": duration_s,
                "event_type": row["event_type"],
                "data": payload,
            }))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return events


def load_database_v2_data(
    database_path: Path,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    user_id: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not database_path.exists():
        return [], []
    connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        table_names = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        required = {
            "behavior_feature_windows",
            "focus_session_feedback",
            "focus_sessions",
        }
        if not required.issubset(table_names):
            return [], []
        window_rows = connection.execute(
            "SELECT window_start_utc, window_end_utc, feature_schema_version, "
            "features_json, label FROM behavior_feature_windows "
            "WHERE user_id = ? AND feature_schema_version = 2 "
            "ORDER BY window_start_utc",
            (user_id,),
        ).fetchall()
        feedback_rows = connection.execute(
            "SELECT f.session_id, f.label, f.score, f.task_type, f.created_at, "
            "s.start_time, s.end_time FROM focus_session_feedback AS f "
            "JOIN focus_sessions AS s ON s.id = f.session_id AND s.user_id = f.user_id "
            "WHERE f.user_id = ? ORDER BY f.created_at",
            (user_id,),
        ).fetchall()
    finally:
        connection.close()

    windows: list[dict[str, object]] = []
    for row in window_rows:
        try:
            timestamp = datetime.fromisoformat(str(row["window_start_utc"]).replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            timestamp = timestamp.astimezone(UTC)
            if start_date is not None and timestamp.date() < start_date:
                continue
            if end_date is not None and timestamp.date() > end_date:
                continue
            features = json.loads(str(row["features_json"]))
            if not isinstance(features, dict):
                continue
            windows.append({
                "window_start_utc": row["window_start_utc"],
                "window_end_utc": row["window_end_utc"],
                "feature_schema_version": row["feature_schema_version"],
                "features": features,
                "label": row["label"],
            })
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

    feedback = [dict(row) for row in feedback_rows]
    return windows, feedback


def _default_database_path() -> Path:
    return Path(platformdirs.user_data_dir("mindflow", ensure_exists=True)) / "mindflow.db"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MindFlow behavior models",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic_v2", "db"],
        default="synthetic_v2",
        help="Data source: synthetic_v2 (archetype-based 24-dim windows) or db (real V2 feature windows)",
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
        "--num-users",
        type=int,
        default=1,
        dest="num_users",
        help="Number of virtual users to generate (default: 1)",
    )
    parser.add_argument(
        "--include-procrastination",
        action="store_true",
        dest="include_procrastination",
        help="Include realistic procrastination patterns in synthetic data",
    )
    parser.add_argument(
        "--user-profiles",
        type=str,
        default="",
        dest="user_profiles",
        help="Comma-separated profile IDs (e.g. junior_cs,senior_business,grad_medical). "
             "Use 'all' for all 30 archetypes.",
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
        "--database-path",
        type=str,
        default="",
        dest="database_path",
        help="SQLite database path for --source db",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="",
        dest="start_date",
        help="Inclusive UTC start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="",
        dest="end_date",
        help="Inclusive UTC end date in YYYY-MM-DD format",
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

    # V2 models live under models_dir/v2/
    v2_models_dir = models_dir / "v2"
    manager = ModelManager(models_dir=v2_models_dir, use_ensemble=False)

    # ── List versions ─────────────────────────────────────────────────────
    if args.list_versions:
        versions = manager.list_versions()
        if not versions:
            print("No model versions found in", v2_models_dir)
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

    # Parse profiles
    profiles_arg: list[str] | None = None
    if args.user_profiles:
        if args.user_profiles.lower() == "all":
            from mindflow.train.user_profiles import list_archetype_ids
            profiles_arg = list_archetype_ids()
        else:
            profiles_arg = [p.strip() for p in args.user_profiles.split(",") if p.strip()]

    events: list[ActivityEvent] | None = None
    feature_windows: list[dict[str, object]] | None = None
    feedback_sessions: list[dict[str, object]] | None = None
    if args.source == "db":
        database_path = Path(args.database_path) if args.database_path else _default_database_path()
        start_date = date.fromisoformat(args.start_date) if args.start_date else None
        end_date = date.fromisoformat(args.end_date) if args.end_date else None
        events = load_database_events(
            database_path,
            start_date=start_date,
            end_date=end_date,
        )
        feature_windows, feedback_sessions = load_database_v2_data(
            database_path,
            start_date=start_date,
            end_date=end_date,
        )
        print(f"Loaded {len(events)} valid events from {database_path}")
        print(
            f"Loaded {len(feature_windows)} v2 feature windows and "
            f"{len(feedback_sessions)} feedback sessions"
        )

    report = run_training(
        source=args.source,
        data_dir=data_dir,
        models_dir=models_dir,
        days=args.days,
        samples_per_hour=args.samples_per_hour,
        seed=args.seed,
        num_users=args.num_users if not profiles_arg else len(profiles_arg),
        include_procrastination=args.include_procrastination,
        user_profiles=profiles_arg,
        events=events,
        feature_windows=feature_windows,
        feedback_sessions=feedback_sessions,
    )

    if report.total_records == 0:
        print("\nTraining pipeline did not complete (no data).")
        sys.exit(1)

    print(f"\nTraining report saved to {models_dir / 'training_report.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
