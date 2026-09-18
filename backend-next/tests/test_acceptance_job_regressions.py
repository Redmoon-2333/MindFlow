"""A07/A08 and publication regressions; no real training or external services."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.analytics import router
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.baseline import BaselineRepository, baseline_models
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
    focus_sessions,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.repositories.training_jobs import TrainingJobRepository
from mindflow.services.training_job_service import TrainingJobService, _JobState
from mindflow.train.models.manager import ModelManager, ModelPublicationError
from mindflow.train.pipeline import TrainingReport


@pytest.fixture
async def jobs_repo(engine, session_factory, create_tables):
    async with engine.begin() as conn:
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(baseline_models.metadata.create_all)
    return TrainingJobRepository(session_factory)


def service(repo=None, user_id=1):
    return TrainingJobService(
        telemetry_repo=SimpleNamespace(
            list_feature_windows=AsyncMock(return_value=[]),
            list_focus_feedback=AsyncMock(return_value=[]),
        ),
        focus_repo=SimpleNamespace(list_all=AsyncMock(return_value=[])),
        jobs_repo=repo,
        user_id=user_id,
    )


def app_for(jobs, session_factory=None):
    app = FastAPI()
    register_exception_handlers(app)
    app.state.training_job_service = jobs
    if session_factory is not None:
        app.state.telemetry_repository = TelemetryRepository(session_factory)
        app.state.focus_repository = SQLAlchemyFocusSessionRepository(session_factory)
        app.state.activity_repository = SQLAlchemyActivityRepository(session_factory)
        app.state.baseline_repository = BaselineRepository(session_factory)
        app.state.settings = None
    app.include_router(router, prefix="/api/v1")
    return app


async def test_restart_http_recovers_latest_and_historical_jobs(jobs_repo, session_factory):
    await jobs_repo.create(job_id="old", started_at="2026-09-18T00:00:00+00:00")
    await jobs_repo.update("old", status="succeeded", activated=True, version_tag="old-model")
    await jobs_repo.create(job_id="running", started_at="2026-09-19T00:00:00+00:00")
    await jobs_repo.update("running", status="training")
    jobs = service(jobs_repo)
    assert await jobs.recover_after_restart() == 1
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(jobs, session_factory)), base_url="http://test"
    ) as client:
        recovered = await client.get("/api/v1/analytics/training-jobs/running")
        history = await client.get("/api/v1/analytics/training-jobs/old")
        readiness = await client.get("/api/v1/analytics/training-readiness")
        cancelled_history = await client.post("/api/v1/analytics/training-jobs/old/cancel")
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "interrupted"
    assert recovered.json()["completed_at"]
    assert history.status_code == 200
    assert history.json()["activated"] is True
    assert cancelled_history.json() == history.json()
    assert readiness.status_code == 200
    assert readiness.json()["current_training_job"]["status"] == "interrupted"
    assert jobs.current_job.job_id == "running"
    assert jobs.current_job.status == "interrupted"
    assert (await jobs.await_completion()).status == "interrupted"
    assert jobs.get_job("running").status == "interrupted"


async def test_repository_and_http_are_user_scoped(jobs_repo):
    await jobs_repo.create(job_id="private", user_id=2)
    await jobs_repo.update("private", user_id=1, status="failed")
    assert await jobs_repo.get("private", user_id=1) is None
    assert (await jobs_repo.get("private", user_id=2))["status"] == "pending"
    jobs = service(jobs_repo, user_id=1)
    assert await jobs.recover_after_restart() == 0
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(jobs)), base_url="http://test"
    ) as client:
        assert (await client.get("/api/v1/analytics/training-jobs/private")).status_code == 404
        assert (
            await client.post("/api/v1/analytics/training-jobs/private/cancel")
        ).status_code == 404
    assert jobs.current_job is None
    owner = service(jobs_repo, user_id=2)
    assert await owner.recover_after_restart() == 1
    assert (await owner.get_job_detail("private")).status == "interrupted"


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "cancelled", "interrupted"])
async def test_repository_terminal_snapshot_never_regresses(jobs_repo, terminal):
    await jobs_repo.create(job_id="job")
    await jobs_repo.update("job", status=terminal, activated=False, error="final")
    await jobs_repo.update("job", status="training", activated=True, error="stale")
    await jobs_repo.update("job", status="succeeded", activated=True, error="later")
    row = await jobs_repo.get("job")
    assert row["status"] == terminal
    assert row["activated"] is False
    assert row["error"] == "final"


async def test_repository_nonterminal_status_never_moves_backwards(jobs_repo):
    await jobs_repo.create(job_id="job")
    await jobs_repo.update("job", status="training")
    await jobs_repo.update("job", status="preparing_data", error="stale")
    assert (await jobs_repo.get("job"))["status"] == "training"
    assert (await jobs_repo.get("job"))["error"] is None


def test_get_job_preserves_synchronous_current_job_contract():
    jobs = service()
    assert jobs.get_job("missing") is None
    jobs._current = _JobState(job_id="current")
    assert jobs.get_job("current").status == "pending"
    assert jobs.get_job("different") is None


async def test_create_failure_does_not_admit_an_unpersisted_job(monkeypatch):
    training = AsyncMock()
    monkeypatch.setattr("mindflow.services.training_job_service.run_training", training)
    repo = SimpleNamespace(create=AsyncMock(side_effect=OSError("disk full")))
    jobs = service(repo)
    with pytest.raises(OSError, match="disk full"):
        await jobs.start_job()
    assert jobs.current_job is None
    training.assert_not_called()


async def test_flush_failure_is_reported_but_later_terminal_write_is_attempted():
    writes = []

    async def update(job_id, **values):
        writes.append(values["status"])
        if values["status"] == "training":
            raise OSError("database unavailable")

    jobs = service(SimpleNamespace(update=update))
    job = _JobState(job_id="job")
    jobs._set_status(job, "training")
    jobs._set_terminal(job, "failed")
    with pytest.raises(RuntimeError, match="persistence failed"):
        await jobs.shutdown()
    assert writes == ["training", "failed"]


async def test_terminal_persistence_failure_is_visible_and_completion_raises(
    tmp_path, monkeypatch, jobs_repo
):
    update = jobs_repo.update

    async def fail_terminal(job_id, **values):
        if values["status"] == "succeeded":
            raise OSError("terminal commit failed")
        await update(job_id, **values)

    monkeypatch.setattr(jobs_repo, "update", fail_terminal)
    monkeypatch.setattr(
        "mindflow.services.training_job_service.run_training",
        lambda **kwargs: TrainingReport(source="db", model_mode="shadow"),
    )
    jobs = service(jobs_repo)
    started = await jobs.start_job(app_state=SimpleNamespace(
        settings=SimpleNamespace(models_dir=tmp_path, data_dir=tmp_path),
        v2_model_manager=None, v2_training_mode="rule_engine_only",
        prediction_service=None, telemetry_service=None,
    ))
    with pytest.raises(RuntimeError, match="persistence failed"):
        await jobs.await_completion()
    assert (await jobs_repo.get(started.job_id))["status"] == "training"
    assert "PersistenceError" in jobs.get_job(started.job_id).error
    with pytest.raises(RuntimeError, match="persistence failed"):
        await jobs.shutdown()


async def test_shutdown_flushes_serialized_writes():
    entered = asyncio.Event()
    release = asyncio.Event()
    writes = []

    async def update(job_id, **values):
        if values["status"] == "training":
            entered.set()
            await release.wait()
        writes.append(values["status"])

    jobs = service(SimpleNamespace(update=update))
    job = _JobState(job_id="job")
    jobs._current = job
    jobs._set_status(job, "training")
    await entered.wait()
    jobs._set_terminal(job, "succeeded")
    closing = asyncio.create_task(jobs.shutdown())
    await asyncio.sleep(0)
    try:
        assert not closing.done(), "shutdown returned before outstanding writes"
    finally:
        release.set()
        await closing
        await asyncio.sleep(0.01)
    assert writes == ["training", "succeeded"]
    jobs._set_status(job, "training")
    assert job.status == "succeeded"


class ModelConsumer:
    def __init__(self, manager, fail=False):
        self._model_manager = manager
        self.fail = fail

    def attach_model_manager(self, manager):
        self._model_manager = manager
        if self.fail:
            raise RuntimeError("injected attach failure after assignment")


@pytest.mark.parametrize("previous", [True, False])
@pytest.mark.parametrize("failure", ["attach_first", "attach", "pipeline", "load", "mismatch"])
async def test_failed_publication_restores_disk_and_all_references(
    tmp_path, monkeypatch, previous, failure, jobs_repo
):
    models = tmp_path / "models"
    original = ModelManager(models / "v2", use_ensemble=False)
    if previous:
        original.save_all()
        assert original.load_latest()
    before = original.latest_path.read_bytes() if previous else None
    old_tag = original.current_version_tag
    prediction_old = object()
    telemetry_old = object()
    state = SimpleNamespace(
        settings=SimpleNamespace(models_dir=models, data_dir=tmp_path),
        v2_model_manager=original if previous else None,
        v2_training_mode="ready" if previous else "rule_engine_only",
        prediction_service=ModelConsumer(prediction_old, fail=failure == "attach_first"),
        telemetry_service=ModelConsumer(telemetry_old, fail=failure == "attach"),
    )
    previous_mode = state.v2_training_mode

    def pipeline(**kwargs):
        candidate = ModelManager(models / "v2", use_ensemble=False)
        candidate.save_all()
        if failure == "pipeline":
            raise RuntimeError("injected after disk activation, before report")
        return TrainingReport(
            source="db", model_mode="ready", activated=True,
            version_tag="wrong-version" if failure == "mismatch" else candidate.current_version_tag,
        )

    if failure == "load":
        monkeypatch.setattr(ModelManager, "load_latest", lambda self: False)
    monkeypatch.setattr("mindflow.services.training_job_service.run_training", pipeline)
    jobs = service(jobs_repo)
    started = await jobs.start_job(app_state=state)
    final = await jobs.await_completion()
    assert final.status == "failed"
    assert final.activated is False
    assert state.v2_model_manager is (original if previous else None)
    assert state.prediction_service._model_manager is prediction_old
    assert state.telemetry_service._model_manager is telemetry_old
    assert state.v2_training_mode == previous_mode
    assert (await jobs_repo.get(started.job_id))["activated"] is False
    if previous:
        assert original.latest_path.read_bytes() == before
        assert original.current_version_tag == old_tag
    else:
        assert not original.latest_path.exists()


async def test_successful_publication_flushes_activated_and_keeps_history(
    tmp_path, monkeypatch, jobs_repo
):
    state = SimpleNamespace(
        settings=SimpleNamespace(models_dir=tmp_path, data_dir=tmp_path),
        v2_model_manager=None, v2_training_mode="rule_engine_only",
        prediction_service=ModelConsumer(None), telemetry_service=ModelConsumer(None),
    )

    def pipeline(**kwargs):
        candidate = ModelManager(tmp_path / "v2", use_ensemble=False)
        candidate.save_all()
        return TrainingReport(
            source="db", activated=True, model_mode="ready",
            version_tag=candidate.current_version_tag,
        )

    monkeypatch.setattr("mindflow.services.training_job_service.run_training", pipeline)
    jobs = service(jobs_repo)
    first = await jobs.start_job(app_state=state)
    final = await jobs.await_completion()
    assert final.status == "succeeded"
    assert final.activated is True
    assert (await jobs_repo.get(first.job_id))["activated"] is True
    assert state.v2_model_manager is state.prediction_service._model_manager
    assert state.v2_model_manager is state.telemetry_service._model_manager
    assert state.v2_model_manager.current_version_tag == final.version_tag
    await jobs.start_job(app_state=state)
    await jobs.await_completion()
    assert (await jobs.get_job_detail(first.job_id)).model_dump() == final.model_dump()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(jobs)), base_url="http://test"
    ) as client:
        assert (await client.get(
            f"/api/v1/analytics/training-jobs/{first.job_id}"
        )).json() == final.model_dump()


async def test_shutdown_pending_job_flushes_cancelled_without_running_pipeline(
    tmp_path, monkeypatch, jobs_repo
):
    training = AsyncMock()
    monkeypatch.setattr("mindflow.services.training_job_service.run_training", training)
    jobs = service(jobs_repo)
    started = await jobs.start_job(app_state=SimpleNamespace(
        settings=SimpleNamespace(models_dir=tmp_path, data_dir=tmp_path),
    ))
    await jobs.shutdown()
    assert (await jobs.await_completion()).status == "cancelled"
    assert (await jobs_repo.get(started.job_id))["status"] == "cancelled"
    training.assert_not_called()


def test_save_retains_previous_active_artifacts_and_signatures(tmp_path, monkeypatch):
    manager = ModelManager(tmp_path, use_ensemble=False)
    tags = iter(["20000101", "20000102", "20000103"])
    monkeypatch.setattr(ModelManager, "_new_version_tag", property(lambda self: next(tags)))
    monkeypatch.setattr(ModelManager, "_MAX_KEPT_VERSIONS", 1)
    previous = manager.save_all()
    manager.save_all(activate=False)
    candidate = ModelManager(tmp_path, use_ensemble=False)
    candidate.save_all()
    for filename in previous.values():
        assert (tmp_path / filename).exists()
        assert (tmp_path / (filename + ".hmac")).exists()
    assert candidate.rollback("20000101")


def test_loaded_manager_version_does_not_follow_candidate_disk_pointer(tmp_path):
    original = ModelManager(tmp_path, use_ensemble=False)
    original.save_all()
    original.load_latest()
    previous_tag = original.current_version_tag
    candidate = ModelManager(tmp_path, use_ensemble=False)
    candidate.save_all()
    assert candidate.current_version_tag != previous_tag
    assert original.current_version_tag == previous_tag


@pytest.mark.parametrize("broken", ["{broken json", "[]", '{"classifier": 42}'])
def test_corrupt_previous_pointer_fails_closed(tmp_path, broken):
    manager = ModelManager(tmp_path, use_ensemble=False)
    manager.latest_path.write_text(broken, encoding="utf-8")
    with pytest.raises(ModelPublicationError, match="previous active pointer"):
        manager.save_all()
    assert manager.latest_path.read_text(encoding="utf-8") == broken


@pytest.mark.parametrize("cancel_closer", [False, True])
async def test_shutdown_waits_for_training_thread_before_terminal_and_flush(
    tmp_path, monkeypatch, jobs_repo, cancel_closer
):
    entered = threading.Event()
    release = threading.Event()

    def pipeline(**kwargs):
        entered.set()
        assert release.wait(5)
        return TrainingReport(source="db", model_mode="shadow")

    monkeypatch.setattr("mindflow.services.training_job_service.run_training", pipeline)
    jobs = service(jobs_repo)
    await jobs.start_job(app_state=SimpleNamespace(
        settings=SimpleNamespace(models_dir=tmp_path, data_dir=tmp_path),
        v2_model_manager=None, v2_training_mode="rule_engine_only",
        prediction_service=None, telemetry_service=None,
    ))
    assert await asyncio.to_thread(entered.wait, 5)
    closing = asyncio.create_task(jobs.shutdown())
    await asyncio.sleep(0.02)
    if cancel_closer:
        closing.cancel()
        await asyncio.sleep(0.02)
    try:
        assert not closing.done(), "training thread still owns potential disk writes"
    finally:
        release.set()
        await closing
    final = await jobs.await_completion()
    assert final.status == "succeeded"
    assert (await jobs_repo.get(final.job_id))["status"] == "succeeded"


@pytest.mark.parametrize("target", ["run", "shutdown", "interleaved"])
@pytest.mark.parametrize("phase", ["thread", "publication", "flush"])
@pytest.mark.parametrize("fail_publication", [False, True])
async def test_repeated_cancellation_joins_all_owned_writes(
    tmp_path, monkeypatch, jobs_repo, target, phase, fail_publication
):
    thread_entered = threading.Event()
    thread_release = threading.Event()
    thread_finished = threading.Event()
    publication_entered = asyncio.Event()
    publication_release = asyncio.Event()
    flush_entered = asyncio.Event()
    flush_release = asyncio.Event()
    writes = []
    original_update = jobs_repo.update
    jobs = service(jobs_repo)
    original_refresh = jobs._refresh_ready_manager
    state = SimpleNamespace(
        settings=SimpleNamespace(models_dir=tmp_path, data_dir=tmp_path),
        v2_model_manager=None, v2_training_mode="rule_engine_only",
        prediction_service=ModelConsumer(None),
        telemetry_service=ModelConsumer(None, fail=fail_publication),
    )

    def pipeline(**kwargs):
        thread_entered.set()
        assert thread_release.wait(10)
        candidate = ModelManager(tmp_path / "v2", use_ensemble=False)
        candidate.save_all()
        writes.append("candidate")
        thread_finished.set()
        return TrainingReport(
            source="db", activated=True, model_mode="ready",
            version_tag=candidate.current_version_tag,
        )

    async def refresh(app_state, report):
        publication_entered.set()
        await publication_release.wait()
        await original_refresh(app_state, report)

    async def update(job_id, **values):
        if values["status"] in ("succeeded", "failed", "cancelled"):
            flush_entered.set()
            await flush_release.wait()
        await original_update(job_id, **values)
        writes.append(values["status"])

    monkeypatch.setattr("mindflow.services.training_job_service.run_training", pipeline)
    monkeypatch.setattr(jobs, "_refresh_ready_manager", refresh)
    monkeypatch.setattr(jobs_repo, "update", update)
    started = await jobs.start_job(app_state=state)
    run = jobs._current._task
    closing = None
    try:
        assert await asyncio.to_thread(thread_entered.wait, 5)
        if phase != "thread":
            thread_release.set()
            await asyncio.wait_for(publication_entered.wait(), 5)
        if phase == "flush":
            publication_release.set()
            await asyncio.wait_for(flush_entered.wait(), 5)
        closing = asyncio.create_task(jobs.shutdown())
        await asyncio.sleep(0)
        cancellations = (
            [run, closing, run, closing] if target == "interleaved"
            else [run if target == "run" else closing] * 2
        )
        for cancelled in cancellations:
            cancelled.cancel()
            await asyncio.sleep(0.02)
            if phase == "thread":
                assert not thread_finished.is_set()
            assert not run.done(), "run escaped before its owned work completed"
            assert not closing.done(), "shutdown escaped before its owned work completed"
            assert not jobs._current._done.is_set()
            if phase != "flush":
                assert jobs.get_job(started.job_id).status == "training"
            else:
                assert not {"succeeded", "failed", "cancelled"}.intersection(writes)
    finally:
        thread_release.set()
        publication_release.set()
        flush_release.set()
        await asyncio.wait_for(asyncio.gather(
            *[task for task in (run, closing) if task is not None],
            return_exceptions=True,
        ), 10)
        assert await asyncio.to_thread(thread_finished.wait, 5)
    final = await jobs.await_completion()
    assert final.status == ("failed" if fail_publication else "succeeded")
    assert final.activated is not fail_publication
    assert (await jobs_repo.get(started.job_id))["status"] == final.status
    latest = tmp_path / "v2" / "latest.json"
    assert latest.exists() is not fail_publication
    if fail_publication:
        assert state.v2_model_manager is None
        assert state.prediction_service._model_manager is None
        assert state.telemetry_service._model_manager is None
    else:
        assert state.v2_model_manager is state.prediction_service._model_manager
        assert state.v2_model_manager is state.telemetry_service._model_manager
    disk_after_close = {p.name: p.read_bytes() for p in (tmp_path / "v2").iterdir()}
    writes_after_close = list(writes)
    await asyncio.sleep(0.03)
    assert writes == writes_after_close
    assert {p.name: p.read_bytes() for p in (tmp_path / "v2").iterdir()} == disk_after_close
