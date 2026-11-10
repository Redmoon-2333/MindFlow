"""Release integrity and real cold-start inference, separate from personal training."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mindflow.app import create_app
from mindflow.config import LLMSettings, Settings
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.infrastructure.notification import LogOnlyNotifier
from mindflow.train import base_bundle
from mindflow.train.base_training import score_predictions
from mindflow.train.models.manager import ModelManager


def test_base_release_is_schema_current_and_does_not_touch_personal_models(tmp_path: Path) -> None:
    personal = tmp_path / "v2"
    personal.mkdir()
    sentinel = personal / "latest.json"
    sentinel.write_text('{"private": "existing-personal-model"}', encoding="utf-8")
    before = sentinel.read_bytes()
    manager = base_bundle.load_bundled_base_model(tmp_path)
    assert sentinel.read_bytes() == before
    assert manager.model_origin == "bundled_base"
    assert manager.current_version_tag == base_bundle.BASE_VERSION
    assert manager.classifier.feature_names_ == list(V2_FEATURE_NAMES)
    probabilities = manager.classifier.predict_proba(np.zeros((2, len(V2_FEATURE_NAMES))))
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)
    assert manager.readiness_status()["ready"] is True
    assert not list(base_bundle.BUNDLE_DIR.glob("*.key"))
    assert not list(base_bundle.BUNDLE_DIR.glob("*.hmac"))


def test_repeat_install_uses_same_local_key_and_separate_machines_get_new_keys(
    tmp_path: Path,
) -> None:
    manager = base_bundle.load_bundled_base_model(tmp_path / "first")
    key = (manager.models_dir / "model_signing.key").read_bytes()
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manager.models_dir.iterdir()
    }
    again = base_bundle.load_bundled_base_model(tmp_path / "first")
    assert files == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in again.models_dir.iterdir()
    }
    other = base_bundle.load_bundled_base_model(tmp_path / "second")
    assert key != (other.models_dir / "model_signing.key").read_bytes()


@pytest.mark.parametrize("filename", ["bundle.json", f"classifier-{base_bundle.BASE_VERSION}.pkl"])
def test_changed_release_is_rejected_before_installation(tmp_path: Path, filename: str) -> None:
    bundle = tmp_path / "release"
    shutil.copytree(base_bundle.BUNDLE_DIR, bundle)
    with (bundle / filename).open("ab") as stream:
        stream.write(b"tampered")
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="checksum"):
        base_bundle.load_bundled_base_model(target, bundle=bundle)
    assert not target.exists()


def test_changed_cache_is_rejected_before_unpickling(tmp_path: Path) -> None:
    manager = base_bundle.load_bundled_base_model(tmp_path)
    artifact = manager.models_dir / f"classifier-{base_bundle.BASE_VERSION}.pkl"
    artifact.write_bytes(artifact.read_bytes() + b"modified")
    with (
        patch("mindflow.train.models.manager.joblib.load") as unsafe_load,
        pytest.raises(ValueError, match="Cached base artifact checksum"),
    ):
        base_bundle.load_bundled_base_model(tmp_path)
    unsafe_load.assert_not_called()


def test_dependency_mismatch_fails_before_creating_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base_bundle.importlib.metadata, "version", lambda _: "0.0.0")
    with pytest.raises(ValueError, match="uv sync --locked"):
        base_bundle.load_bundled_base_model(tmp_path / "models")
    assert not (tmp_path / "models").exists()


def test_class_specific_metrics_use_distinct_denominators() -> None:
    rows = [{"label": value} for value in [1, 1, 1, 0]]
    metrics = score_predictions(rows, np.array([0.9, 0.2, 0.8, 0.1]))
    assert metrics["normal_work_false_alert_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["distraction_recall"] == 1.0
    assert metrics["focus_count"] == 3
    assert metrics["distracted_count"] == 1


def _login(client: TestClient, app: object) -> None:
    token = app.state.system_token  # type: ignore[attr-defined]
    headers = {"host": "localhost"}
    ticket = client.post(
        "/api/v1/auth/bootstrap/ticket",
        headers={**headers, "Authorization": f"Bearer {token}"},
    )
    assert ticket.status_code == 200
    assert (
        client.post(
            "/api/v1/auth/bootstrap",
            json={"ticket": ticket.json()["ticket"]},
            headers=headers,
        ).status_code
        == 204
    )


def test_default_cold_start_uses_base_for_real_prediction_not_personal_ready(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        run_scheduler=False,
        run_collectors=False,
        llm=LLMSettings(api_key=None, deepseek_api_key=None, ollama_enabled=False),
    )
    app = create_app(settings, configure_logging=False)
    with (
        patch("mindflow.app.create_notifier", return_value=LogOnlyNotifier()),
        TestClient(app) as client,
    ):
        _login(client, app)
        status = client.get("/api/v1/analytics/model-status", headers={"host": "localhost"}).json()
        assert status["mode"] == "base"
        assert status["loaded"] is True
        assert status["ready"] is False
        assert status["inference_available"] is True
        assert status["synthetic_only"] is True
        assert status["model_origin"] == "bundled_base"
        assert status["personalized"] is False
        assert not (settings.models_dir / "v2" / "latest.json").exists()
        now = datetime.now(UTC)
        features = dict.fromkeys(V2_FEATURE_NAMES, 0.0)
        features.update(active_seconds_ratio=1.0, top_app_ratio=1.0, task_type_code=3.0)
        client.portal.call(
            app.state.telemetry_repository.upsert_feature_windows,
            [
                {
                    "user_id": 1,
                    "window_start_utc": now - timedelta(minutes=5 * (24 - index)),
                    "window_end_utc": now - timedelta(minutes=5 * (23 - index)),
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "features_json": json.dumps(features),
                    "label": None,
                }
                for index in range(24)
            ],
        )
        prediction = client.get(
            "/api/v1/telemetry/focus-prediction",
            headers={"host": "localhost"},
        ).json()
        assert prediction["status"] == "ready"
        current = client.portal.call(app.state.prediction_service.predict_latest)
        assert current.model_version == base_bundle.BASE_VERSION
        assert 0 <= prediction["focus_probability"] <= 1


def test_default_base_can_be_disabled_for_personal_only_runtime(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        run_scheduler=False,
        run_collectors=False,
        base_model_enabled=False,
        llm=LLMSettings(api_key=None, deepseek_api_key=None, ollama_enabled=False),
    )
    app = create_app(settings, configure_logging=False)
    with (
        patch("mindflow.app.create_notifier", return_value=LogOnlyNotifier()),
        TestClient(app) as client,
    ):
        assert app.state.v2_model_manager is None
        assert app.state.v2_training_mode == "rule_engine_only"
        assert not (settings.models_dir / "bundled").exists()
        _login(client, app)
        assert (
            client.get(
                "/api/v1/analytics/model-status",
                headers={"host": "localhost"},
            ).json()["loaded"]
            is False
        )


def test_personal_manager_origin_is_preserved(tmp_path: Path) -> None:
    assert ModelManager(tmp_path, use_ensemble=False).model_origin == "personal"


def test_release_probabilities_match_fixed_heldout_predictions(tmp_path: Path) -> None:
    from mindflow.train.models.ensemble import EnsembleClassifier

    data = json.loads((base_bundle.BUNDLE_DIR / "prediction_examples.json").read_text("utf-8"))
    examples = data["examples"]
    manager = base_bundle.load_bundled_base_model(tmp_path)
    assert isinstance(manager.classifier, EnsembleClassifier)
    assert manager.classifier.xgb_model is not None
    assert manager.classifier.calibrator is not None
    matrix = np.array([[row["features"][name] for name in V2_FEATURE_NAMES] for row in examples])
    expected = [row["focus_probability"] for row in examples]
    np.testing.assert_allclose(manager.classifier.predict_proba(matrix)[:, 1], expected, atol=1e-7)


def _personal_artifacts(models: Path, *, old_schema: bool = False, fitted: bool = True) -> str:
    base = base_bundle.load_bundled_base_model(models)
    manager = ModelManager(models / "v2", use_ensemble=False)
    manager.classifier = base.classifier
    manager.clustering = base.clustering
    if old_schema:
        manager.classifier.feature_names_ = list(V2_FEATURE_NAMES[:24])
    manager.classifier._is_fitted = fitted
    names = manager.save_all(activate=True)
    return names["classifier"].removeprefix("classifier-").removesuffix(".pkl")


@pytest.mark.parametrize("old_schema,fitted", [(True, True), (False, False)])
def test_unusable_signed_personal_model_falls_back_without_overwriting(
    tmp_path: Path, old_schema: bool, fitted: bool,
) -> None:
    settings = Settings(
        data_dir=tmp_path, run_scheduler=False, run_collectors=False,
        llm=LLMSettings(api_key=None, deepseek_api_key=None, ollama_enabled=False),
    )
    _personal_artifacts(settings.models_dir, old_schema=old_schema, fitted=fitted)
    pointer = settings.models_dir / "v2" / "latest.json"
    original = pointer.read_bytes()
    app = create_app(settings, configure_logging=False)
    with (
        patch("mindflow.app.create_notifier", return_value=LogOnlyNotifier()),
        TestClient(app),
    ):
        assert app.state.v2_model_manager.model_origin == "bundled_base"
        assert app.state.v2_training_mode == "base"
        assert pointer.read_bytes() == original


def test_ready_personal_model_has_priority_on_startup(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path, run_scheduler=False, run_collectors=False,
        llm=LLMSettings(api_key=None, deepseek_api_key=None, ollama_enabled=False),
    )
    version = _personal_artifacts(settings.models_dir)
    app = create_app(settings, configure_logging=False)
    with (
        patch("mindflow.app.create_notifier", return_value=LogOnlyNotifier()),
        patch("mindflow.train.base_bundle.load_bundled_base_model") as fallback,
        TestClient(app),
    ):
        fallback.assert_not_called()
        assert app.state.v2_model_manager.model_origin == "personal"
        assert app.state.v2_model_manager.current_version_tag == version


def test_personal_ready_refresh_replaces_base_for_all_consumers(tmp_path: Path) -> None:
    from mindflow.train.pipeline import TrainingReport

    settings = Settings(
        data_dir=tmp_path, run_scheduler=False, run_collectors=False,
        llm=LLMSettings(api_key=None, deepseek_api_key=None, ollama_enabled=False),
    )
    app = create_app(settings, configure_logging=False)
    with (
        patch("mindflow.app.create_notifier", return_value=LogOnlyNotifier()),
        TestClient(app) as client,
    ):
        previous = app.state.v2_model_manager
        assert previous.model_origin == "bundled_base"
        shadow_report = TrainingReport(model_mode="shadow", activated=False)
        app.state.training_job_service._update_shadow_mode(app.state, shadow_report)
        assert app.state.v2_model_manager is previous
        version = _personal_artifacts(settings.models_dir)
        report = TrainingReport(model_mode="ready", activated=True, version_tag=version)
        client.portal.call(app.state.training_job_service._refresh_ready_manager, app.state, report)
        assert app.state.v2_model_manager is not previous
        assert app.state.v2_model_manager.model_origin == "personal"
        assert app.state.v2_training_mode == "ready"
        assert app.state.prediction_service._model_manager is app.state.v2_model_manager
        assert app.state.telemetry_service._model_manager is app.state.v2_model_manager


@pytest.mark.parametrize("damage", ["hmac", "key", "pointer"])
def test_cache_signature_and_pointer_handling(tmp_path: Path, damage: str) -> None:
    manager = base_bundle.load_bundled_base_model(tmp_path)
    if damage == "pointer":
        manager.latest_path.write_text('{"classifier":"elsewhere.pkl"}', encoding="utf-8")
        assert (
            base_bundle.load_bundled_base_model(tmp_path).current_version_tag
            == base_bundle.BASE_VERSION
        )
        return
    if damage == "key":
        (manager.models_dir / "model_signing.key").unlink()
    else:
        (manager.models_dir / f"classifier-{base_bundle.BASE_VERSION}.pkl.hmac").write_text("bad")
    with patch("mindflow.train.models.manager.joblib.load") as unsafe_load:
        with pytest.raises(Exception, match="signing key|Signature verification"):
            base_bundle.load_bundled_base_model(tmp_path)
        unsafe_load.assert_not_called()
