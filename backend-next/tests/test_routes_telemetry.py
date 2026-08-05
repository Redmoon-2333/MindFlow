from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_prediction_service import _make_feature_window, _make_mock_model_manager

from mindflow.api.routes.telemetry import router
from mindflow.infrastructure.repositories.preferences import PreferencesRepository, user_preferences
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import metadata
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.services.telemetry_service import TelemetryService


@pytest.fixture
async def telemetry_app(engine, session_factory, tmp_path) -> FastAPI:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.run_sync(user_preferences.metadata.create_all)
    repository = TelemetryRepository(session_factory)
    preferences = PreferencesRepository(session_factory)
    service = TelemetryService(repository, preferences, data_dir=tmp_path)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.telemetry_service = service
    return app


def test_preferences_default_disabled(telemetry_app: FastAPI) -> None:
    response = TestClient(telemetry_app).get("/api/v1/telemetry/status")

    assert response.status_code == 200
    assert response.json()["preferences"]["input_telemetry_enabled"] is False
    assert response.json()["preferences"]["browser_tracking_enabled"] is False


def test_preferences_patch(telemetry_app: FastAPI) -> None:
    response = TestClient(telemetry_app).patch(
        "/api/v1/telemetry/preferences",
        json={"input_telemetry_enabled": True, "interaction_retention_days": 7},
    )

    assert response.status_code == 200
    assert response.json()["input_telemetry_enabled"] is True


def test_browser_pair_and_domain_heartbeat(telemetry_app: FastAPI) -> None:
    client = TestClient(telemetry_app)
    pairing = client.post("/api/v1/telemetry/browser/pairing-code").json()
    paired = client.post(
        "/api/v1/telemetry/browser/pair",
        json={"code": pairing["code"]},
    )
    token = paired.json()["token"]
    status = client.get("/api/v1/telemetry/status").json()
    assert status["browser_paired"] is True

    heartbeat = client.post(
        "/api/v1/telemetry/browser/heartbeat",
        headers={"X-Browser-Token": token},
        json={
            "timestamp_utc": datetime(2026, 7, 24, 8, tzinfo=UTC).isoformat(),
            "duration_s": 5,
            "browser_name": "edge",
            "domain": "Docs.Python.org/path?secret=1",
            "audible": False,
            "incognito": False,
        },
    )

    assert heartbeat.status_code == 200
    assert heartbeat.json()["domain"] == "docs.python.org"


def test_incognito_heartbeat_is_ignored(telemetry_app: FastAPI) -> None:
    client = TestClient(telemetry_app)
    code = client.post("/api/v1/telemetry/browser/pairing-code").json()["code"]
    token = client.post(
        "/api/v1/telemetry/browser/pair", json={"code": code}
    ).json()["token"]

    response = client.post(
        "/api/v1/telemetry/browser/heartbeat",
        headers={"X-Browser-Token": token},
        json={
            "timestamp_utc": datetime(2026, 7, 24, 8, tzinfo=UTC).isoformat(),
            "duration_s": 5,
            "browser_name": "edge",
            "domain": "example.com",
            "audible": False,
            "incognito": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["ignored"] is True


def test_focus_prediction_degrades_to_rule_engine(telemetry_app: FastAPI) -> None:
    response = TestClient(telemetry_app).get("/api/v1/telemetry/focus-prediction")

    assert response.status_code == 200
    assert response.json()["mode"] == "rule_engine_only"


class _StubFocusPredictionService:
    """Canned ``predict_latest_focus`` payload for driving route responses."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    async def predict_latest_focus(self, user_id: int = 1) -> dict[str, object]:
        return self._payload


def _app_with_focus_prediction(payload: dict[str, object]) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.telemetry_service = _StubFocusPredictionService(payload)
    return app


class TestFocusPredictionResponseContract:
    """GET /telemetry/focus-prediction returns the canonical typed response.

    The route binds ``FocusPredictionResponse`` as its response_model: only
    ``focus_probability``, ``status``, ``mode``, and ``reason`` may appear,
    probabilities are numeric only for ``ready`` and null otherwise, and an
    out-of-contract status fails loudly rather than being coerced.
    """

    def test_focus_prediction_ready_returns_numeric_probability(self) -> None:
        app = _app_with_focus_prediction(
            {
                "mode": "ready",
                "focus_probability": 0.73,
                "status": "ready",
                "reason": "",
            }
        )
        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        assert response.json() == {
            "focus_probability": 0.73,
            "status": "ready",
            "mode": "ready",
            "reason": "",
        }

    @pytest.mark.parametrize(
        ("status", "mode"),
        [
            ("no_model", "rule_engine_only"),
            ("no_data", "ready"),
            ("stale", "ready"),
            ("schema_mismatch", "rule_engine_only"),
            ("inference_error", "rule_engine_only"),
        ],
    )
    def test_focus_prediction_non_ready_returns_null_probability(
        self, status: str, mode: str
    ) -> None:
        app = _app_with_focus_prediction(
            {
                "mode": mode,
                "focus_probability": None,
                "status": status,
                "reason": f"reason for {status}",
            }
        )
        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == status
        assert body["focus_probability"] is None
        assert body["mode"] == mode
        assert body["reason"] == f"reason for {status}"

    def test_focus_prediction_response_contains_only_contract_fields(self) -> None:
        # The service may return extra diagnostic fields; the contract filters them.
        app = _app_with_focus_prediction(
            {
                "mode": "rule_engine_only",
                "focus_probability": None,
                "status": "no_model",
                "reason": "未加载 ML 模型",
                "uncertainty": 1.0,
                "top_factors": [],
                "feature_schema_version": 3,
                "window_count": 0,
                "window_start_utc": None,
                "coverage_ratio": 0.0,
                "data_age_s": None,
                "explanation_method": "",
                "model_version": None,
            }
        )
        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        assert set(response.json().keys()) == {
            "focus_probability",
            "status",
            "mode",
            "reason",
        }

    def test_focus_prediction_rejects_invalid_status_from_service(self) -> None:
        app = _app_with_focus_prediction(
            {
                "mode": "ready",
                "focus_probability": None,
                "status": "bogus",
                "reason": "x",
            }
        )
        response = TestClient(app, raise_server_exceptions=False).get(
            "/api/v1/telemetry/focus-prediction"
        )

        assert response.status_code == 500


# ── Real-service route tests (not stub-only) ─────────────────────────────────

_MISSING = MagicMock()  # sentinel: replaced with a real mock model manager


def _app_with_real_focus_prediction(
    windows: list[dict],
    data_dir: Path,
    *,
    focus_proba: float = 0.75,
    model_manager: MagicMock | None = _MISSING,
    classifier_overrides: dict | None = None,
) -> FastAPI:
    """Build an app whose route runs a real FocusPredictionService/TelemetryService.

    Only the telemetry repository is a mock; the services, route, and
    ``FocusPredictionResponse`` response model are the production ones.
    """
    repository = AsyncMock()
    repository.list_feature_windows_in_range.return_value = windows
    if model_manager is _MISSING:
        model_manager = _make_mock_model_manager(focus_proba=focus_proba)
    if model_manager is not None and classifier_overrides:
        for key, value in classifier_overrides.items():
            setattr(model_manager.classifier, key, value)
    prediction_service = FocusPredictionService(
        telemetry_repository=repository,
        model_manager=model_manager,
    )
    service = TelemetryService(
        repository=repository,
        preferences_repository=AsyncMock(),
        data_dir=data_dir,
        prediction_service=prediction_service,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.telemetry_service = service
    return app


class TestFocusPredictionRealServiceThroughRoute:
    """GET /telemetry/focus-prediction through the real service stack.

    The normalization lives in ``TelemetryService.predict_latest_focus``:
    the route response must carry a numeric probability only for ``ready``
    and present-and-null for every non-ready status, with the reason intact.
    """

    def test_ready_returns_numeric_probability(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]
        app = _app_with_real_focus_prediction(windows, tmp_path)

        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert body["focus_probability"] == pytest.approx(0.75)
        assert body["mode"] == "ready"
        assert body["reason"] == ""

    def test_old_window_stale_returns_null_probability(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        base = now - timedelta(minutes=150)
        windows = [
            _make_feature_window(base + timedelta(minutes=5 * i)) for i in range(24)
        ]
        app = _app_with_real_focus_prediction(windows, tmp_path)

        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "stale"
        assert body["focus_probability"] is None
        assert body["mode"] == "ready"
        assert body["reason"], "stale reason must be preserved"

    def test_low_coverage_stale_returns_null_probability(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        windows = [_make_feature_window(now - timedelta(minutes=5))]
        app = _app_with_real_focus_prediction(windows, tmp_path)

        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "stale"
        assert body["focus_probability"] is None
        assert body["reason"], "stale reason must be preserved"

    def test_schema_mismatch_returns_null_probability(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]
        app = _app_with_real_focus_prediction(
            windows, tmp_path, classifier_overrides={"feature_names_": ["old_feature"]}
        )

        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "schema_mismatch"
        assert body["focus_probability"] is None
        assert body["mode"] == "rule_engine_only"
        assert body["reason"], "schema_mismatch reason must be preserved"

    def test_inference_error_returns_null_probability(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]
        model_manager = _make_mock_model_manager()
        model_manager.classifier.predict_proba.side_effect = RuntimeError("model crash")
        app = _app_with_real_focus_prediction(windows, tmp_path, model_manager=model_manager)

        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "inference_error"
        assert body["focus_probability"] is None
        assert body["mode"] == "rule_engine_only"
        assert body["reason"], "inference_error reason must be preserved"

    def test_no_data_returns_null_probability(self, tmp_path: Path) -> None:
        app = _app_with_real_focus_prediction([], tmp_path)

        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "no_data"
        assert body["focus_probability"] is None
        assert body["mode"] == "ready"
        assert body["reason"], "no_data reason must be preserved"

    def test_no_model_returns_null_probability(self, tmp_path: Path) -> None:
        app = _app_with_real_focus_prediction([], tmp_path, model_manager=None)

        response = TestClient(app).get("/api/v1/telemetry/focus-prediction")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "no_model"
        assert body["focus_probability"] is None
        assert body["mode"] == "rule_engine_only"
        assert body["reason"], "no_model reason must be preserved"
