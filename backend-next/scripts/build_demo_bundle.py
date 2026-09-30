"""Build the distributable synthetic model without reading a personal database."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.train.pipeline import run_training

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "demo-models" / "v4"


def main() -> None:
    experiment = ROOT / "data" / "experiments" / "20260930_demo_bundle"
    if experiment.exists() or BUNDLE.exists():
        raise SystemExit("Refusing to overwrite an existing experiment or demo bundle")
    report = run_training(
        source="synthetic_v2",
        num_users=6,
        days=14,
        seed=42,
        models_dir=experiment / "models",
        allow_activation=False,
    )
    if not report.saved_models:
        raise SystemExit("Training produced no loadable artifacts")
    BUNDLE.mkdir(parents=True)
    files: dict[str, dict[str, str | int]] = {}
    for kind, filename in report.saved_models.items():
        target = BUNDLE / f"{kind}-demo_v4.pkl"
        shutil.copyfile(experiment / "models" / "v2" / filename, target)
        files[kind] = {
            "name": target.name,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "bytes": target.stat().st_size,
        }
    manifest = {
        "version": "demo_v4",
        "source": "synthetic_v2",
        "demo_only": True,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(V2_FEATURE_NAMES),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("scikit-learn", "numpy", "scipy", "joblib", "xgboost")
        },
        "recipe": {"users": 6, "days": 14, "seed": 42},
        "files": files,
    }
    target = BUNDLE / "bundle.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Bundle manifest SHA256:", hashlib.sha256(target.read_bytes()).hexdigest())
    print("Total artifact bytes:", sum(int(item["bytes"]) for item in files.values()))
    print("No database, training rows, signing keys, or API credentials exported.")


if __name__ == "__main__":
    main()
