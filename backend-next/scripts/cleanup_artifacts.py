"""Clean up development artifacts that accumulate in the workspace.

Run:  uv run python scripts/cleanup_artifacts.py [--dry-run]

Targets (audit report — historical artifact accumulation):
  - .mypy_cache/          (observed 449 MB)
  - .test_runs/           (stale per-test databases, observed 482 files)
  - .pytest_cache/        (pytest node-id cache)
  - .hypothesis/          (hypothesis example cache — harmless but large)
  - data/experiments/*/input.db  (8 MB snapshots per experiment run)
  - runtime_logs/         (old backend logs, kept 30 days by default)

Keeps:  source, tests, docs, data/models (managed by ModelManager retention),
data/eval_reports, and the most recent experiment directory.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1_048_576
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1_048_576


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="print what would be removed without deleting"
    )
    args = parser.parse_args()

    targets: list[Path] = [
        ROOT / ".mypy_cache",
        ROOT / ".test_runs",
        ROOT / ".pytest_cache",
        ROOT / ".hypothesis",
        ROOT / ".ruff_cache",
    ]

    # Experiment input.db snapshots: keep the newest experiment dir.
    experiments = ROOT / "data" / "experiments"
    if experiments.is_dir():
        runs = sorted(experiments.iterdir(), key=lambda p: p.name, reverse=True)
        for run in runs[1:]:
            for db in run.glob("input.db"):
                targets.append(db)

    # Old runtime logs (keep the last 10).
    logs = ROOT / "runtime_logs"
    if logs.is_dir():
        log_files = sorted(logs.glob("*.log"), key=lambda p: p.name, reverse=True)
        for old in log_files[10:]:
            targets.append(old)

    total_mb = 0.0
    for target in targets:
        if not target.exists():
            continue
        mb = _size_mb(target)
        total_mb += mb
        if args.dry_run:
            print(f"[dry-run] would remove {target.relative_to(ROOT)} ({mb:.1f} MB)")
        elif target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            print(f"removed {target.relative_to(ROOT)} ({mb:.1f} MB)")
        else:
            target.unlink(missing_ok=True)
            print(f"removed {target.relative_to(ROOT)} ({mb:.1f} MB)")

    print(f"---\ntotal freed: {total_mb:.1f} MB ({'dry run' if args.dry_run else 'done'})")


if __name__ == "__main__":
    main()