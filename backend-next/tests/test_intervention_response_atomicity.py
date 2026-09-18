"""Regression tests for intervention response atomicity and job persistence.

Covers two audit findings:

* an intervention response used to overwrite unconditionally, so a duplicate
  submission (or the desktop popup's timeout default) could replace the
  answer a user actually gave;
* training job state lived only in memory, so a restart erased the record of
  an in-flight run instead of reporting it as interrupted.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.infrastructure.repositories.training_jobs import TrainingJobRepository
from mindflow.infrastructure.schema import metadata


@pytest.fixture
async def session_factory(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'audit.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.fixture
async def intervention_repo(session_factory) -> InterventionLogRepository:
    return InterventionLogRepository(session_factory=session_factory)


async def _create(repo: InterventionLogRepository, intervention_id: str) -> None:
    await repo.log_triggered(
        user_id=1,
        intervention_type="gentle_reminder",
        cbt_technique="stimulus_control",
        context={},
        intervention_id=intervention_id,
        triggered_at=datetime(2026, 9, 19, 9, tzinfo=UTC),
    )


# ── Response atomicity ──────────────────────────────────────────────────


async def test_first_human_response_is_recorded(intervention_repo) -> None:
    await _create(intervention_repo, "iv-1")
    row = await intervention_repo.update_response("iv-1", "accepted", 1.5)
    assert row is not None
    assert row["user_response"] == "accepted"
    assert row["response_source"] == "human"


async def test_duplicate_human_response_does_not_overwrite(intervention_repo) -> None:
    """A second click (double-submit, stale tab) must not rewrite the answer."""
    await _create(intervention_repo, "iv-2")
    await intervention_repo.update_response("iv-2", "accepted", 1.0)

    row = await intervention_repo.update_response("iv-2", "ignored", 9.0)

    assert row is not None
    assert row["user_response"] == "accepted", "the first human answer stands"
    assert row["response_latency_s"] == 1.0


async def test_auto_timeout_does_not_override_human_response(intervention_repo) -> None:
    """A popup timeout firing late must not erase a real click."""
    await _create(intervention_repo, "iv-3")
    await intervention_repo.update_response("iv-3", "accepted", 2.0, source="human")

    row = await intervention_repo.update_response("iv-3", "ignored", 30.0, source="auto")

    assert row is not None
    assert row["user_response"] == "accepted"
    assert row["response_source"] == "human"


async def test_human_response_overrides_earlier_auto_default(intervention_repo) -> None:
    """If the popup timed out first, a later real click still wins."""
    await _create(intervention_repo, "iv-4")
    await intervention_repo.update_response("iv-4", "ignored", 30.0, source="auto")

    row = await intervention_repo.update_response("iv-4", "accepted", 31.0, source="human")

    assert row is not None
    assert row["user_response"] == "accepted"
    assert row["response_source"] == "human"


async def test_auto_response_fills_empty_slot(intervention_repo) -> None:
    await _create(intervention_repo, "iv-5")
    row = await intervention_repo.update_response("iv-5", "ignored", 30.0, source="auto")
    assert row is not None
    assert row["user_response"] == "ignored"
    assert row["response_source"] == "auto"


async def test_second_auto_response_is_idempotent(intervention_repo) -> None:
    await _create(intervention_repo, "iv-6")
    await intervention_repo.update_response("iv-6", "ignored", 30.0, source="auto")
    row = await intervention_repo.update_response("iv-6", "accepted", 31.0, source="auto")
    assert row is not None
    assert row["user_response"] == "ignored"


async def test_missing_intervention_returns_none(intervention_repo) -> None:
    assert await intervention_repo.update_response("nope", "accepted", 0.0) is None


# ── Training job persistence ────────────────────────────────────────────


async def test_training_job_lifecycle_round_trips(session_factory) -> None:
    repo = TrainingJobRepository(session_factory=session_factory)
    await repo.create(job_id="job-1", user_id=1, started_at="2026-09-19T09:00:00+00:00")
    await repo.update(
        "job-1",
        status="succeeded",
        completed_at="2026-09-19T09:05:00+00:00",
        activated=True,
        version_tag="20260919_090500_abc123",
        feature_schema_version=3,
        quality_gate={"passed": True},
        evaluation={"status": "evaluated"},
    )

    row = await repo.get("job-1")

    assert row is not None
    assert row["status"] == "succeeded"
    assert row["activated"] is True
    assert row["version_tag"] == "20260919_090500_abc123"
    assert row["quality_gate"] == {"passed": True}
    assert row["evaluation"] == {"status": "evaluated"}


async def test_restart_marks_running_jobs_interrupted(session_factory) -> None:
    """A job that was mid-flight when the process died must not look active."""
    repo = TrainingJobRepository(session_factory=session_factory)
    await repo.create(job_id="job-live", user_id=1)
    await repo.update("job-live", status="training")
    await repo.create(job_id="job-done", user_id=1)
    await repo.update("job-done", status="succeeded", completed_at="2026-09-19T09:00:00+00:00")

    changed = await repo.mark_interrupted(user_id=1)

    assert changed == 1, "only the non-terminal job is touched"
    live = await repo.get("job-live")
    done = await repo.get("job-done")
    assert live is not None and live["status"] == "interrupted"
    assert live["error"]
    assert done is not None and done["status"] == "succeeded", "terminal rows are preserved"


async def test_restart_recovery_is_idempotent(session_factory) -> None:
    repo = TrainingJobRepository(session_factory=session_factory)
    await repo.create(job_id="job-a", user_id=1)
    await repo.update("job-a", status="preparing_data")

    assert await repo.mark_interrupted(user_id=1) == 1
    assert await repo.mark_interrupted(user_id=1) == 0, "already interrupted"


async def test_latest_job_returns_most_recent(session_factory) -> None:
    repo = TrainingJobRepository(session_factory=session_factory)
    await repo.create(job_id="old", user_id=1, started_at="2026-09-18T09:00:00+00:00")
    await repo.create(job_id="new", user_id=1, started_at="2026-09-19T09:00:00+00:00")

    latest = await repo.latest_for_user(user_id=1)

    assert latest is not None
    assert latest["job_id"] == "new"


async def test_get_unknown_job_returns_none(session_factory) -> None:
    repo = TrainingJobRepository(session_factory=session_factory)
    assert await repo.get("does-not-exist") is None
    assert await repo.latest_for_user(user_id=99) is None
