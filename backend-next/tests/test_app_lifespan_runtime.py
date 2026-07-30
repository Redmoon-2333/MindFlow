"""Full application lifespan smoke tests for runtime wiring."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import mindflow.app as app_module
from mindflow.app import create_app
from mindflow.config import Settings
from mindflow.runtime import RuntimeServices
from mindflow.services.scheduler import AsyncioScheduler


def test_full_lifespan_starts_and_stops_with_background_roles_disabled(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path,
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'mindflow.db'}",
        run_scheduler=False,
        run_collectors=False,
        timezone="Asia/Shanghai",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        ready = client.get("/api/v1/health/ready", headers={"host": "localhost"})
        assert ready.status_code == 200
        assert app.state.runtime.settings.run_scheduler is False
        assert app.state.runtime.settings.run_collectors is False
        assert app.state.scheduler._running is False
        assert app.state.chat_service is not None
        assert app.state.panel_service is not None
        assert app.state.workflow_port is None, (
            "workflow_port should be None with default new_analysis_graph=False"
        )
        assert app.state.panel_service._timezone == "Asia/Shanghai"
        assert app.state.llm_service._timezone == "Asia/Shanghai"
        assert (
            app.state.collector_service is None
            or app.state.collector_service.status != "running"
        )


def test_full_app_bootstrap_cookie_static_api_and_websocket_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frontend_dist = tmp_path / "frontend"
    frontend_dist.mkdir()
    (frontend_dist / "index.html").write_text(
        "<main>MindFlow integration UI</main>", encoding="utf-8"
    )
    monkeypatch.setattr(app_module, "_frontend_dist_dir", lambda: frontend_dist)

    settings = Settings(
        data_dir=tmp_path / "data",
        db_url=f"sqlite+aiosqlite:///{tmp_path / 'mindflow.db'}",
        run_scheduler=False,
        run_collectors=False,
    )
    app = create_app(settings)

    with TestClient(app) as client:
        host = {"host": "localhost"}
        root = client.get("/", headers=host)
        assert root.status_code == 200
        assert "MindFlow integration UI" in root.text
        assert client.get("/panel", headers=host).status_code == 200
        assert client.get("/api/v1/preferences", headers=host).status_code == 401

        issued = client.post(
            "/api/v1/auth/bootstrap/ticket",
            headers={**host, "Authorization": f"Bearer {app.state.system_token}"},
        )
        assert issued.status_code == 200
        exchanged = client.post(
            "/api/v1/auth/bootstrap",
            headers=host,
            json={"ticket": issued.json()["ticket"]},
        )
        assert exchanged.status_code == 204

        preferences = client.get("/api/v1/preferences", headers=host)
        assert preferences.status_code == 200
        assert preferences.json() == {}
        assert client.get("/api/v1/missing", headers=host).status_code == 404

        with client.websocket_connect("/api/v1/ws", headers=host) as websocket:
            websocket.send_json({"type": "ping"})
            assert websocket.receive_json()["type"] == "pong"


async def test_lifespan_fails_fast_and_disposes_engine_when_migrations_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = SimpleNamespace(dispose=AsyncMock())
    integrity = AsyncMock(side_effect=AssertionError("integrity check must not run"))
    close_websockets = AsyncMock(return_value=0)
    monkeypatch.setattr(app_module, "create_engine", lambda _url: engine)
    monkeypatch.setattr(app_module, "create_session_factory", lambda _engine: object())
    monkeypatch.setattr(app_module, "run_migrations", AsyncMock(return_value=False))
    monkeypatch.setattr(app_module, "integrity_check", integrity)
    monkeypatch.setattr(app_module, "close_all_connections", close_websockets)

    app = create_app(
        Settings(
            data_dir=tmp_path,
            db_url=f"sqlite+aiosqlite:///{tmp_path / 'mindflow.db'}",
            run_scheduler=False,
            run_collectors=False,
        )
    )

    with pytest.raises(RuntimeError, match="Database migration failed"):
        async with app_module._lifespan(app):
            pytest.fail("lifespan yielded after migration failure")

    integrity.assert_not_awaited()
    engine.dispose.assert_awaited_once()
    close_websockets.assert_awaited_once()


async def test_lifespan_cleans_up_when_application_runtime_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutdown_runtime = AsyncMock(wraps=app_module._shutdown_runtime_services)
    close_websockets = AsyncMock(return_value=0)
    monkeypatch.setattr(app_module, "_shutdown_runtime_services", shutdown_runtime)
    monkeypatch.setattr(app_module, "close_all_connections", close_websockets)

    app = create_app(
        Settings(
            data_dir=tmp_path,
            db_url=f"sqlite+aiosqlite:///{tmp_path / 'mindflow.db'}",
            run_scheduler=False,
            run_collectors=False,
        )
    )

    with pytest.raises(RuntimeError, match="application failed"):
        async with app_module._lifespan(app):
            raise RuntimeError("application failed")

    shutdown_runtime.assert_awaited_once_with(app.state.runtime)
    close_websockets.assert_awaited_once()


async def test_lifespan_rolls_back_services_after_partial_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collector_start = AsyncMock()
    collector_stop = AsyncMock()
    telemetry_start = AsyncMock(side_effect=RuntimeError("telemetry start failed"))
    telemetry_stop = AsyncMock()
    close_websockets = AsyncMock(return_value=0)
    monkeypatch.setattr(app_module.CollectorService, "start", collector_start)
    monkeypatch.setattr(app_module.CollectorService, "stop", collector_stop)
    monkeypatch.setattr(app_module.InputTelemetryService, "start", telemetry_start)
    monkeypatch.setattr(app_module.InputTelemetryService, "stop", telemetry_stop)
    monkeypatch.setattr(
        app_module.TelemetryService,
        "get_preferences",
        AsyncMock(return_value={"input_telemetry_enabled": True}),
    )
    monkeypatch.setattr(app_module, "close_all_connections", close_websockets)

    # On Windows without optional collector deps (pywin32, psutil),
    # create_collector() raises CollectorUnavailableError before
    # CollectorService is instantiated, so collector_start is never
    # awaited.  Provide a minimal fake collector so CollectorService
    # construction succeeds and the start/stop rollback is exercised.
    fake_collector = SimpleNamespace(
        snapshot=AsyncMock(),
        idle_seconds=AsyncMock(),
    )
    monkeypatch.setattr(app_module, "create_collector", lambda: fake_collector)

    app = create_app(
        Settings(
            data_dir=tmp_path,
            db_url=f"sqlite+aiosqlite:///{tmp_path / 'mindflow.db'}",
            run_scheduler=False,
            run_collectors=True,
        )
    )

    with pytest.raises(RuntimeError, match="telemetry start failed"):
        async with app_module._lifespan(app):
            pytest.fail("lifespan yielded after service startup failure")

    collector_start.assert_awaited_once()
    telemetry_start.assert_awaited_once()
    telemetry_stop.assert_awaited_once()
    collector_stop.assert_awaited_once()
    close_websockets.assert_awaited_once()


async def test_runtime_start_does_not_wait_for_startup_recovery(tmp_path: Path) -> None:
    scheduler = AsyncioScheduler(timezone="UTC")
    recovery_started = asyncio.Event()
    recovery_release = asyncio.Event()

    async def _recover(_: datetime) -> None:
        recovery_started.set()
        await recovery_release.wait()

    scheduler._startup_recovery = _recover
    runtime = RuntimeServices(
        settings=Settings(
            data_dir=tmp_path,
            run_scheduler=True,
            run_collectors=False,
        ),
        engine=object(),
        session_factory=object(),
        scheduled_job_runs_repository=AsyncMock(),
        scheduler=scheduler,
    )

    await asyncio.wait_for(
        app_module._start_runtime_services(
            runtime,
            input_telemetry_enabled=False,
        ),
        timeout=0.1,
    )

    assert scheduler._running is True
    await asyncio.wait_for(recovery_started.wait(), timeout=1.0)
    assert any(not task.done() for task in scheduler._tasks)
    recovery_release.set()
    await scheduler.shutdown()
