"""Runtime service aggregation and lifecycle tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI

from mindflow.app import _publish_runtime_state, _shutdown_runtime_services, _start_runtime_services
from mindflow.config import Settings
from mindflow.runtime import RuntimeServices


def _runtime(tmp_path, *, run_scheduler: bool, run_collectors: bool) -> RuntimeServices:
    settings = Settings(
        data_dir=tmp_path, run_scheduler=run_scheduler, run_collectors=run_collectors
    )
    return RuntimeServices(
        settings=settings,
        engine=MagicMock(),
        session_factory=MagicMock(),
        scheduled_job_runs_repository=MagicMock(),
        scheduler=MagicMock(),
        collector_service=MagicMock(),
        input_telemetry_service=MagicMock(),
        panel_service=MagicMock(),
        chat_service=MagicMock(),
        llm_service=MagicMock(),
    )


async def test_runtime_role_switches_control_background_start(tmp_path) -> None:
    runtime = _runtime(tmp_path, run_scheduler=False, run_collectors=False)
    runtime.collector_service.start = AsyncMock()
    runtime.input_telemetry_service.start = AsyncMock()
    await _start_runtime_services(runtime, input_telemetry_enabled=True)
    runtime.scheduler.start.assert_not_called()
    runtime.collector_service.start.assert_not_awaited()
    runtime.input_telemetry_service.start.assert_not_awaited()


async def test_shutdown_waits_for_scheduler_before_releasing_dependencies(tmp_path) -> None:
    runtime = _runtime(tmp_path, run_scheduler=True, run_collectors=True)
    order: list[str] = []

    def recorder(name: str):
        async def record() -> None:
            order.append(name)

        return record

    runtime.scheduler.shutdown = AsyncMock(side_effect=recorder("scheduler"))
    runtime.panel_service.aclose = AsyncMock(side_effect=recorder("panel"))
    runtime.chat_service.aclose = AsyncMock(side_effect=recorder("chat"))
    runtime.llm_service.aclose = AsyncMock(side_effect=recorder("llm"))
    runtime.input_telemetry_service.stop = AsyncMock(side_effect=recorder("input"))
    runtime.collector_service.stop = AsyncMock(side_effect=recorder("collector"))
    runtime.engine.dispose = AsyncMock(side_effect=recorder("engine"))
    await _shutdown_runtime_services(runtime)
    assert order[0] == "scheduler"
    assert order[-1] == "engine"


def test_runtime_is_published_with_legacy_state_aliases(tmp_path) -> None:
    app = FastAPI()
    runtime = _runtime(tmp_path, run_scheduler=True, run_collectors=True)
    _publish_runtime_state(app, runtime)
    assert app.state.runtime is runtime
    assert app.state.scheduler is runtime.scheduler
    assert app.state.engine is runtime.engine
    assert app.state.collector_service is runtime.collector_service
