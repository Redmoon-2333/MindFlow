"""Run MindFlow ML and LangGraph experiment rounds end to end.

Usage (from backend-next/):

    python scripts/run_experiments.py --runs ml_r1,ml_r2,ml_r3,lg_r1,lg_r2,lg_r3

The script copies the production SQLite database into
``data/experiments/<timestamp>/input.db``, rebuilds v3 feature windows on the
copy, trains ML rounds, runs live panel evaluation rounds through the existing
eval CLI, and writes a human-readable comparison report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mindflow.domain.events import ActivityEvent
from mindflow.services.telemetry_features import build_v2_feature_window
from mindflow.train.__main__ import load_database_v2_data
from mindflow.train.pipeline import run_training

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_NEXT = PROJECT_ROOT
EXPERIMENTS_ROOT = PROJECT_ROOT / "data" / "experiments"
PROD_DB = Path(
    r"C:\Users\lenovo\AppData\Local\mindflow\mindflow\mindflow.db"
)
PROD_MODELS = Path(
    r"C:\Users\lenovo\AppData\Local\mindflow\mindflow\models\v2"
)
DATES = ("2026-07-24", "2026-07-29", "2026-07-31")
SCENARIOS = ("IMP-001", "DEC-001", "PER-001", "EMO-001", "TAV-001")


def _now_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def snapshot_db(run_dir: Path) -> Path:
    source = sqlite3.connect(f"file:{PROD_DB}?mode=ro", uri=True)
    target_path = run_dir / "input.db"
    target = sqlite3.connect(str(target_path))
    with target:
        source.backup(target)
    target.close()
    source.close()
    return target_path


def rebuild_v3_windows(db_path: Path) -> int:
    """Rebuild 5-minute feature windows on a copied DB using v3 semantics."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    events: list[ActivityEvent] = []
    for row in con.execute(
        "SELECT id, user_id, timestamp, duration_s, event_type, data_json "
        "FROM activity_events ORDER BY timestamp"
    ):
        try:
            ts = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            payload = json.loads(str(row["data_json"]))
            payload["timestamp_utc"] = ts.isoformat()
            events.append(ActivityEvent.from_dict({
                "id": row["id"],
                "user_id": row["user_id"],
                "timestamp_utc": ts,
                "duration_s": float(row["duration_s"]),
                "event_type": row["event_type"],
                "data": payload,
            }))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    buckets = [dict(row) for row in con.execute(
        "SELECT * FROM interaction_buckets ORDER BY window_start_utc"
    )]
    if not events:
        con.close()
        return 0

    first = min(e.timestamp_utc for e in events)
    last = max(e.timestamp_utc for e in events)
    start = first.replace(minute=first.minute // 5 * 5, second=0, microsecond=0)
    window_start = start
    inserted = 0
    with con:
        while window_start < last:
            window_end = window_start + timedelta(minutes=5)
            window_events = [
                e for e in events
                if window_start <= e.timestamp_utc < window_end
            ]
            window_buckets = [
                b for b in buckets
                if window_start <= _parse_dt(b.get("window_start_utc")) < window_end
            ]
            if window_events or window_buckets:
                features = build_v2_feature_window(
                    window_events, window_buckets, [], window_start, window_end
                )
                features_json = json.dumps(features, ensure_ascii=False)
                con.execute(
                    "INSERT OR REPLACE INTO behavior_feature_windows "
                    "(id, user_id, window_start_utc, window_end_utc, "
                    "feature_schema_version, features_json, label, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"exp-{window_start.isoformat()}",
                        1,
                        window_start.isoformat(),
                        window_end.isoformat(),
                        int(features["feature_schema_version"]),
                        features_json,
                        None,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                inserted += 1
            window_start = window_end
    con.close()
    return inserted


def _parse_dt(value: object) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


def load_v2_windows_as_current(db_path: Path) -> list[dict]:
    """Load legacy v2 windows but mark them as the current schema for training."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT window_start_utc, window_end_utc, feature_schema_version, features_json, label "
        "FROM behavior_feature_windows WHERE user_id = 1 AND feature_schema_version = 2"
    ).fetchall()
    con.close()
    windows = []
    for row in rows:
        try:
            features = json.loads(str(row["features_json"]))
            windows.append({
                "window_start_utc": row["window_start_utc"],
                "window_end_utc": row["window_end_utc"],
                "feature_schema_version": 3,
                "features": features,
                "label": row["label"],
            })
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return windows


def run_ml_round(name: str, run_dir: Path, db_path: Path, use_v3: bool) -> dict:
    if use_v3:
        count = rebuild_v3_windows(db_path)
        print(f"[{name}] rebuilt {count} v3 windows")
    windows, feedback = load_database_v2_data(db_path, user_id=1)
    windows = load_v2_windows_as_current(db_path) if not use_v3 else windows
    models_dir = run_dir / "models" / name
    report = run_training(
        source="db",
        models_dir=models_dir,
        feature_windows=windows,
        feedback_sessions=feedback,
    )
    data = {
        "round": name,
        "window_count": len(windows),
        "feedback_count": len(feedback),
        "report": report.to_dict(),
    }
    _write_json(run_dir / f"ml_{name}.json", data)
    return data


def run_langgraph_round(name: str, run_dir: Path) -> dict:
    output_dir = run_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mindflow.eval",
        "--mode",
        "both",
        "--live",
        "--yes",
        "--scenario-ids",
        ",".join(SCENARIOS),
        "--round-name",
        name,
        "--output-dir",
        str(output_dir),
    ]
    log_path = run_dir / f"lg_{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=BACKEND_NEXT, stdout=log, stderr=subprocess.STDOUT)
    reports = sorted(output_dir.glob(f"*{name}.json"))
    payload = {
        "round": name,
        "scenario_ids": list(SCENARIOS),
        "exit_code": result.returncode,
        "reports": [r.name for r in reports],
        "log": log_path.name,
    }
    _write_json(run_dir / f"lg_{name}.json", payload)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manifest(run_dir: Path, rounds: list[str]) -> None:
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "production_db": str(PROD_DB),
        "production_models": str(PROD_MODELS),
        "rounds": rounds,
        "dates": list(DATES),
        "scenarios": list(SCENARIOS),
    }
    _write_json(run_dir / "manifest.json", manifest)


def write_comparison(run_dir: Path) -> None:
    lines = [
        "# MindFlow 实验对比报告",
        "",
        f"- 实验目录: `{run_dir.name}`",
        f"- 生成时间: {datetime.now(UTC).isoformat()}",
        "",
        "## ML 轮次",
        "",
    ]
    for name in ("ml_ml_r1.json", "ml_ml_r2.json", "ml_ml_r3.json"):
        path = run_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        report = data.get("report", {})
        gate = report.get("quality_gate", {})
        lines.extend([
            f"### {data.get('round', name)}",
            f"- 窗口数: {data.get('window_count')}，反馈数: {data.get('feedback_count')}",
            f"- 质量门: {'通过' if gate.get('passed') else '未通过'} ({gate.get('mode')})",
            f"- 评估: {json.dumps(report.get('evaluation', {}).get('candidate', {}), ensure_ascii=False)}",
            "",
        ])
    lines.extend(["## LangGraph 轮次", ""])
    for name in ("lg_lg_r1.json", "lg_lg_r2.json", "lg_lg_r3.json"):
        path = run_dir / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.extend([
            f"### {data.get('round', name)}",
            f"- 场景: {', '.join(data.get('scenario_ids', []))}",
            f"- 退出码: {data.get('exit_code')}",
            f"- 报告: {', '.join(data.get('reports', []))}",
            "",
        ])
    (run_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MindFlow experiment rounds")
    parser.add_argument(
        "--runs",
        default="ml_r1,ml_r2,ml_r3,lg_r1,lg_r2,lg_r3",
        help="Comma-separated round names",
    )
    args = parser.parse_args()
    rounds = [r.strip() for r in args.runs.split(",") if r.strip()]

    run_dir = EXPERIMENTS_ROOT / _now_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {run_dir}")

    db_path = snapshot_db(run_dir)
    write_manifest(run_dir, rounds)

    for name in rounds:
        if name == "ml_r1":
            shutil.copy2(PROD_MODELS / "training_report.json", run_dir / "ml_ml_r1.json")
        elif name in ("ml_r2", "ml_r3"):
            run_ml_round(name, run_dir, db_path, use_v3=(name == "ml_r3"))
        elif name.startswith("lg_"):
            run_langgraph_round(name, run_dir)
        else:
            print(f"Unknown round {name}, skipping")

    write_comparison(run_dir)
    print(f"Done. See {run_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
