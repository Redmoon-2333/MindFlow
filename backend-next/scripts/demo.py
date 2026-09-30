"""Run the bundled synthetic ML demo in a separate, offline data directory."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "demo-models" / "v4"
MANIFEST_SHA256 = "ae66ae8b19669e98b342cb35de8e1519d8382bc65450db8531e371a9de904199"
MARKER = ".mindflow-demo.json"
KINDS = ("clustering", "classifier", "hmm")


def verified_bundle(bundle: Path = BUNDLE) -> dict[str, Any]:
    payload = (bundle / "bundle.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Demo manifest differs from the checksum pinned in this release")
    manifest: dict[str, Any] = json.loads(payload)
    for name, expected in manifest["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Demo requires {name}=={expected}; run uv sync --locked --extra ml")
    for kind in KINDS:
        info = manifest["files"][kind]
        if info["name"] != f"{kind}-demo_v4.pkl":
            raise ValueError("Unexpected demo artifact name")
        artifact = (bundle / info["name"]).read_bytes()
        if len(artifact) != info["bytes"] or hashlib.sha256(artifact).hexdigest() != info["sha256"]:
            raise ValueError(f"Demo artifact checksum mismatch: {kind}")
    return manifest


def install_model(data_dir: Path, bundle: Path = BUNDLE) -> None:
    from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
    from mindflow.train.models.manager import ModelManager
    from mindflow.train.serialization import _load_or_create_signing_key, sign_model_file

    data_dir = data_dir.resolve()
    manifest = verified_bundle(bundle)
    if manifest["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("Demo feature schema is incompatible with this application")
    if manifest["feature_names"] != list(V2_FEATURE_NAMES):
        raise ValueError("Demo feature order is incompatible with this application")
    marker = data_dir / MARKER
    if data_dir.exists() and any(data_dir.iterdir()) and not marker.is_file():
        raise ValueError("Refusing to use an existing non-demo data directory")
    if (
        marker.is_file()
        and json.loads(marker.read_text(encoding="utf-8")).get("demo_only") is not True
    ):
        raise ValueError("Invalid demo directory marker")
    data_dir.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text('{"demo_only": true, "seeded": false}\n', encoding="utf-8")
    models = data_dir / "models" / "v2"
    models.mkdir(parents=True, exist_ok=True)
    # Never replace a model the user has trained inside this demo directory.
    if (models / "latest.json").exists():
        key_path = models / "model_signing.key"
        if not key_path.is_file() or key_path.stat().st_size != 32:
            raise ValueError("Existing demo model has no valid local signing key")
        if not ModelManager(models, use_ensemble=False).load_latest():
            raise ValueError("Existing demo model cannot be loaded; it was not overwritten")
        return
    names: dict[str, str] = {}
    key = _load_or_create_signing_key(models)
    for kind in KINDS:
        name = manifest["files"][kind]["name"]
        payload = (bundle / name).read_bytes()
        # Recheck the exact bytes written, not an earlier read of a mutable file.
        if hashlib.sha256(payload).hexdigest() != manifest["files"][kind]["sha256"]:
            raise ValueError(f"Demo artifact changed during installation: {kind}")
        (models / name).write_bytes(payload)
        sign_model_file(models / name, key)
        names[kind] = name
    manager = ModelManager(models_dir=models, use_ensemble=False)
    if not manager._verify_loadable(names):
        raise ValueError("Signed demo artifacts failed their trial load")
    report = {
        "source": "synthetic_v2",
        "demo_only": True,
        "version_tag": "demo_v4",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_mode": "shadow",
        "activated": False,
        "quality_gate": {
            "passed": False, "mode": "shadow", "deployment_tier": "shadow", "checks": {},
        },
    }
    ModelManager._atomic_write_text(
        models / "training_report.json", json.dumps(report, indent=2) + "\n",
    )
    ModelManager._atomic_write_text(
        models / "manifest-demo_v4.json", json.dumps(manifest, indent=2) + "\n",
    )
    ModelManager._atomic_write_text(models / "latest.json", json.dumps(names, indent=2) + "\n")


async def seed_demo(data_dir: Path, app: Any) -> None:
    from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
    from scripts.seed_visual_baseline import seed

    marker = data_dir / MARKER
    state = json.loads(marker.read_text(encoding="utf-8"))
    if not state["seeded"]:
        await seed(data_dir)
        from sqlalchemy import select, update

        from mindflow.infrastructure.repositories.focus import focus_sessions
        from mindflow.infrastructure.schema import activity_events

        # The screenshot fixture represents full days. Make today's demo rows
        # historical rather than placing its later slots in the future.
        now = datetime.now(UTC)
        today = now.astimezone().date()
        engine = app.state.session_factory.kw["bind"]
        async with engine.begin() as conn:
            sessions = (await conn.execute(select(focus_sessions).where(
                focus_sessions.c.date == today.isoformat(),
            ))).mappings().all()
            durations = [
                datetime.fromisoformat(row["end_time"])
                - datetime.fromisoformat(row["start_time"])
                for row in sessions
            ]
            cursor = now - sum(durations, timedelta()) - timedelta(minutes=45)
            for row, duration in zip(sessions, durations, strict=True):
                await conn.execute(update(focus_sessions).where(
                    focus_sessions.c.id == row["id"],
                ).values(start_time=cursor.isoformat(), end_time=(cursor + duration).isoformat()))
                cursor += duration + timedelta(minutes=15)
            events = (await conn.execute(select(activity_events).where(
                activity_events.c.timestamp >= today.isoformat(),
            ).order_by(activity_events.c.timestamp))).mappings().all()
            for index, row in enumerate(events):
                timestamp = now - timedelta(minutes=(len(events) - index) * 15)
                payload = json.loads(row["data_json"])
                payload["timestamp_utc"] = timestamp.isoformat()
                await conn.execute(update(activity_events).where(
                    activity_events.c.id == row["id"],
                ).values(timestamp=timestamp.isoformat(), data_json=json.dumps(payload)))
        state["seeded"] = True
        marker.write_text(json.dumps(state) + "\n", encoding="utf-8")
    # Refresh synthetic inference windows, keeping any user's demo-page edits.
    from mindflow.train.synthetic_v2 import generate_v2_synthetic_data
    from mindflow.train.user_profiles import list_archetype_ids

    windows, _ = generate_v2_synthetic_data(
        archetype_ids=list_archetype_ids()[:1], days_per_archetype=1, seed=42,
    )
    now = datetime.now(UTC).replace(microsecond=0)
    await app.state.telemetry_repository.upsert_feature_windows([
        {
            "user_id": 1,
            "window_start_utc": now - timedelta(minutes=(24 - index) * 5),
            "window_end_utc": now - timedelta(minutes=(23 - index) * 5),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "features_json": json.dumps(row["features"]),
            "label": None,
        }
        for index, row in enumerate(windows[108:132])
    ])


def create_demo_app(data_dir: Path, port: int) -> Any:
    # Do not inherit the host's personal configuration or online credentials.
    for name in tuple(os.environ):
        if name.startswith("MINDFLOW_") or name in {
            "API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
            "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
            "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING",
        }:
            os.environ.pop(name)
    from mindflow import config
    from mindflow.app import create_app

    config._cached_data_dir = data_dir
    settings = config.Settings(
        data_dir=data_dir, models_dir=data_dir / "models", port=port,
        run_collectors=False, run_scheduler=False,
        llm=config.LLMSettings(api_key=None, deepseek_api_key=None, ollama_enabled=False),
    )
    config.SETTINGS = settings
    app = create_app(settings, configure_logging=False)
    app.state.demo_mode = True
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: Any) -> AsyncIterator[None]:
        async with original_lifespan(application):
            await seed_demo(data_dir, application)
            yield

    app.router.lifespan_context = lifespan
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "demo")
    parser.add_argument("--port", type=int, default=8870)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--login", action="store_true", help="Print a one-time local login URL")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    install_model(data_dir)
    app = create_demo_app(data_dir, args.port)
    if args.login:
        from mindflow.bootstrap import request_bootstrap_url
        from mindflow.config import get_settings
        print(asyncio.run(request_bootstrap_url(get_settings())))
        return
    if args.prepare_only:
        async def prepare() -> None:
            async with app.router.lifespan_context(app):
                print("Synthetic demo prepared; no personal database or model was modified.")
        asyncio.run(prepare())
        return
    if not (ROOT.parent / "frontend" / "dist" / "index.html").is_file():
        parser.error("Build the frontend first: cd ../frontend && npm ci && npm run build")
    print(f"OFFLINE DEMO: synthetic data and ML only, http://127.0.0.1:{args.port}")
    print(f'Login: uv run python scripts/demo.py --data-dir "{data_dir}"'
          f" --port {args.port} --login")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
