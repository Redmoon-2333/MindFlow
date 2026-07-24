from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.routes.telemetry import router
from mindflow.infrastructure.repositories.preferences import PreferencesRepository, user_preferences
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository, metadata
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
