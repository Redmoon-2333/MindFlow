"""The portable demo is explicit, synthetic, signed locally, and non-destructive."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mindflow import config
from mindflow.domain.feature_schema import V2_FEATURE_NAMES
from mindflow.infrastructure.notification import LogOnlyNotifier
from mindflow.train.models.manager import ModelManager
from scripts import demo


def test_ml_dependencies_pin_the_published_serialization_environment() -> None:
    with (demo.ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    dependencies = set(project["project"]["optional-dependencies"]["ml"])
    manifest = json.loads((demo.BUNDLE / "bundle.json").read_text(encoding="utf-8"))
    assert {
        f"{name}=={version}" for name, version in manifest["packages"].items()
    } <= dependencies


def test_bundled_model_loads_and_predicts_with_current_schema(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    demo.install_model(target)
    manager = ModelManager(target / "models" / "v2", use_ensemble=False)
    assert manager.load_latest()
    assert manager.current_version_tag == "demo_v4"
    assert manager.classifier.feature_names_ == list(V2_FEATURE_NAMES)
    probabilities = manager.classifier.predict_proba(np.zeros((2, len(V2_FEATURE_NAMES))))
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_installation_generates_distinct_machine_local_keys(tmp_path: Path) -> None:
    targets = [tmp_path / "first", tmp_path / "second"]
    for target in targets:
        demo.install_model(target)
    keys = [(target / "models" / "v2" / "model_signing.key").read_bytes() for target in targets]
    assert all(len(key) == 32 for key in keys)
    assert keys[0] != keys[1]
    assert not list(demo.BUNDLE.glob("*.key"))
    assert not list(demo.BUNDLE.glob("*.hmac"))


def test_existing_personal_directory_is_never_modified(tmp_path: Path) -> None:
    sentinel = tmp_path / "mindflow.db"
    sentinel.write_bytes(b"private existing database")
    with pytest.raises(ValueError, match="existing non-demo"):
        demo.install_model(tmp_path)
    assert sentinel.read_bytes() == b"private existing database"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["mindflow.db"]


def test_repeat_install_preserves_existing_model_and_key(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    demo.install_model(target)
    before = {
        path.relative_to(target): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target.rglob("*") if path.is_file()
    }
    demo.install_model(target)
    after = {
        path.relative_to(target): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in target.rglob("*") if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("filename", ["bundle.json", "classifier-demo_v4.pkl"])
def test_modified_release_is_rejected_before_creating_target(
    tmp_path: Path, filename: str,
) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(demo.BUNDLE, bundle)
    with (bundle / filename).open("ab") as stream:
        stream.write(b"tampered")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="checksum"):
        demo.install_model(target, bundle)
    assert not target.exists()


def test_incompatible_package_version_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(demo.importlib.metadata, "version", lambda _name: "0.0.0")
    with pytest.raises(ValueError, match="uv sync --locked"):
        demo.verified_bundle()


def test_demo_backend_auth_status_prediction_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "demo"
    demo.install_model(target)
    monkeypatch.setattr(config, "_cached_data_dir", target)
    monkeypatch.setattr(config, "SETTINGS", None)
    for name, value in tuple(os.environ.items()):
        if name.startswith("MINDFLOW_"):
            monkeypatch.setenv(name, value)
    app = demo.create_demo_app(target, 8870)
    with (
        patch("mindflow.app.create_notifier", return_value=LogOnlyNotifier()),
        TestClient(app, base_url="http://127.0.0.1:8870") as client,
    ):
        token = (target / "token").read_text(encoding="utf-8").strip()
        ticket = client.post(
            "/api/v1/auth/bootstrap/ticket", headers={"Authorization": f"Bearer {token}"},
        )
        assert ticket.status_code == 200
        assert client.post(
            "/api/v1/auth/bootstrap", json={"ticket": ticket.json()["ticket"]},
        ).status_code == 204
        status = client.get("/api/v1/analytics/model-status").json()
        assert status["loaded"] is True
        assert status["ready"] is False
        assert status["demo_only"] is True
        assert status["mode"] == "shadow"
        assert status["version"] == "demo_v4"
        prediction = client.get("/api/v1/telemetry/focus-prediction").json()
        assert prediction["status"] == "ready"
        assert 0 <= prediction["focus_probability"] <= 1
        current = client.portal.call(app.state.prediction_service.predict_latest)
        assert current.model_version == "demo_v4"
        # The oldest bucket crosses the rolling two-hour query boundary by
        # the time the request arrives; the other 23 must remain available.
        assert 23 <= current.window_count <= 24
        assert current.focus_probability == pytest.approx(prediction["focus_probability"])
        activities = client.get("/api/v1/activities").json()
        assert activities["total"] > 0
        assert app.state.settings.run_collectors is False
        assert app.state.settings.run_scheduler is False
        assert app.state.settings.llm.deepseek_api_key is None
    report = json.loads((target / "models" / "v2" / "training_report.json").read_text())
    assert report["quality_gate"]["passed"] is False
    assert report["activated"] is False
