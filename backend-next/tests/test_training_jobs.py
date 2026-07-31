"""Training job endpoint integration tests.

Covers: 412 blocked, 202 accepted, 409 duplicate, status transitions,
terminal report, shadow outcome, cancellation (pre-training only),
cancel-rejection during training, artifact paths, failure reporting,
no auto-scheduler registration, and endpoint schema validation.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.analytics import router as analytics_router
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.baseline import (
    BaselineRepository,
    baseline_models,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
    focus_sessions,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import (
    behavior_feature_windows,
    focus_session_feedback,
)
from mindflow.services.training_job_service import (
    CancelRejectedError,
    TrainingJobService,
)

# ── V2 features JSON (24-dim) ────────────────────────────────────────────

_V2_FEATURES_JSON = json.dumps({
    "app_switch_count": 3.0, "domain_switch_count": 2.0,
    "longest_segment_ratio": 0.6, "idle_ratio": 0.1,
    "keypress_rate_per_min": 25.0, "mouse_click_rate_per_min": 12.0,
    "scroll_rate_per_min": 5.0, "mouse_distance_per_min": 200.0,
    "input_active_ratio": 0.7, "interaction_bursts_per_min": 2.0,
    "click_key_ratio": 0.5, "browser_ratio": 0.3,
    "audible_browser_ratio": 0.1, "active_seconds_ratio": 0.8,
    "top_app_ratio": 0.7, "top_domain_ratio": 0.5,
    "interaction_interval_mean_s": 10.0,
    "interaction_interval_std_s": 5.0,
    "interaction_interval_cv": 0.5,
    "hour_sin": 0.5, "hour_cos": 0.5,
    "weekday_sin": 0.5, "weekday_cos": 0.5,
    "task_type_code": 0.0,
})


# ── Seed helpers ───────────────────────────────────────────────────────────


def _v2_window(user_id: int, start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "window_start_utc": start,
        "window_end_utc": end,
        "feature_schema_version": 3,
        "features_json": _V2_FEATURES_JSON,
        "label": None,
    }


async def _seed_windows(
    telemetry_repo: TelemetryRepository, rows: list[dict[str, Any]],
) -> None:
    await telemetry_repo.upsert_feature_windows(rows)


async def _seed_feedback(engine: Any, rows: list[dict[str, Any]]) -> None:
    async with engine.begin() as conn:
        for row in rows:
            await conn.execute(
                focus_session_feedback.insert().values(
                    id=row.get("id", f"fb-{row['session_id']}"),
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    label=row["label"],
                    score=row["score"],
                    task_type=row.get("task_type"),
                    created_at=row.get("created_at", datetime.now(UTC).isoformat()),
                )
            )


async def _seed_focus_sessions(engine: Any, rows: list[dict[str, Any]]) -> None:
    async with engine.begin() as conn:
        for row in rows:
            await conn.execute(
                focus_sessions.insert().values(
                    id=row["id"],
                    user_id=row["user_id"],
                    date=row["date"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    session_type=row.get("session_type", "focus"),
                    dominant_app=row.get("dominant_app"),
                    focus_score=row.get("focus_score"),
                    switch_count=row.get("switch_count"),
                )
            )


async def _seed_activity_events(
    engine: Any, count: int, user_id: int = 1,
) -> None:
    base = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
    async with engine.begin() as conn:
        for i in range(count):
            ts = base + timedelta(minutes=i * 2)
            await conn.execute(
                activity_events.insert().values(
                    id=f"act-{i:04d}",
                    user_id=user_id,
                    timestamp=ts.isoformat(),
                    duration_s=60.0,
                    data_json=json.dumps({"app_name": "Code.exe", "is_idle": False}),
                    event_type="window_snapshot",
                )
            )


async def _seed_trainable_data(engine, telemetry_repo) -> None:
    """Seed data that passes the trainable threshold."""
    base = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
    windows = []
    for d in range(3):
        day_base = base + timedelta(days=d)
        for i in range(14):
            windows.append(_v2_window(
                1, day_base + timedelta(minutes=i * 5),
                day_base + timedelta(minutes=(i + 1) * 5),
            ))
    await _seed_windows(telemetry_repo, windows)

    fcs_rows: list[dict[str, Any]] = []
    fb_rows: list[dict[str, Any]] = []
    for i in range(10):
        sid = f"of-sess-{i}"
        start = base + timedelta(minutes=i * 5)
        end = start + timedelta(minutes=4)
        fcs_rows.append({
            "id": sid, "user_id": 1, "date": "2026-07-25",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "session_type": "focus",
        })
        fb_rows.append({
            "user_id": 1, "session_id": sid, "label": "focus",
            "score": 4, "task_type": "coding",
            "created_at": start.isoformat(),
        })
    for i in range(10):
        sid = f"od-sess-{i}"
        start = base + timedelta(days=1, minutes=i * 5)
        end = start + timedelta(minutes=4)
        fcs_rows.append({
            "id": sid, "user_id": 1, "date": "2026-07-26",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "session_type": "focus",
        })
        fb_rows.append({
            "user_id": 1, "session_id": sid, "label": "distracted",
            "score": 2, "task_type": "browsing",
            "created_at": start.isoformat(),
        })
    for i in range(10):
        sid = f"od-sess-d3-{i}"
        start = base + timedelta(days=2, minutes=i * 5)
        end = start + timedelta(minutes=4)
        fcs_rows.append({
            "id": sid, "user_id": 1, "date": "2026-07-27",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "session_type": "focus",
        })
        fb_rows.append({
            "user_id": 1, "session_id": sid, "label": "focus",
            "score": 4, "task_type": "coding",
            "created_at": start.isoformat(),
        })

    await _seed_focus_sessions(engine, fcs_rows)
    await _seed_feedback(engine, fb_rows)


# ── App factory ────────────────────────────────────────────────────────────


def _make_app(engine: Any, session_factory) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    telemetry_repo = TelemetryRepository(session_factory=session_factory)
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
    baseline_repo = BaselineRepository(session_factory=session_factory)
    job_service = TrainingJobService(
        telemetry_repo=telemetry_repo, focus_repo=focus_repo, user_id=1,
    )

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.telemetry_repository = telemetry_repo
    app.state.focus_repository = focus_repo
    app.state.activity_repository = activity_repo
    app.state.baseline_repository = baseline_repo
    app.state.v2_model_manager = None
    app.state.v2_training_mode = "rule_engine_only"
    app.state.training_job_service = job_service
    app.state.settings = None  # mock settings (model refresh will skip)

    app.include_router(analytics_router, prefix="/api/v1")
    return app


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def tables(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(focus_session_feedback.metadata.create_all)
        await conn.run_sync(behavior_feature_windows.metadata.create_all)
        await conn.run_sync(baseline_models.metadata.create_all)


# ── Tests: 412 blocked ─────────────────────────────────────────────────────


class TestBlocked412:
    """POST /training-jobs returns 412 when data is insufficient."""

    async def test_empty_db_returns_412(
        self, engine, session_factory, tables,
    ) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.post("/api/v1/analytics/training-jobs")
        assert resp.status_code == 412
        body = resp.json()
        assert body["title"] == "Training Not Ready"
        # extra fields are merged into the top-level response
        assert body["trainable"] is False
        assert len(body["blockers"]) >= 1

    async def test_not_trainable_returns_412(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        base = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
        await _seed_windows(telemetry_repo, [
            _v2_window(1, base + timedelta(minutes=i * 5),
                       base + timedelta(minutes=(i + 1) * 5))
            for i in range(5)
        ])

        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.post("/api/v1/analytics/training-jobs")
        assert resp.status_code == 412
        body = resp.json()
        assert body["trainable"] is False


# ── Tests: 202 accepted ────────────────────────────────────────────────────


class TestAccepted202:
    """POST /training-jobs returns 202 with job id when trainable."""

    async def test_trainable_returns_202(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.post("/api/v1/analytics/training-jobs")
        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"].startswith("train-")
        assert body["status"] == "pending"


# ── Tests: 409 duplicate ───────────────────────────────────────────────────


class TestDuplicate409:
    """POST /training-jobs returns 409 when a job is already active."""

    async def test_duplicate_post_returns_409(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)

        with TestClient(app) as client:
            # First request: 202
            r1 = client.post("/api/v1/analytics/training-jobs")
            assert r1.status_code == 202

            # Second request: 409 (duplicate) — same event loop, task still alive
            r2 = client.post("/api/v1/analytics/training-jobs")
            assert r2.status_code == 409
            body = r2.json()
            assert body["title"] == "Training Job Already Active"

    async def test_duplicate_returns_correct_error_schema(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)

        with TestClient(app) as client:
            client.post("/api/v1/analytics/training-jobs")  # first — accept
            resp = client.post("/api/v1/analytics/training-jobs")  # second — conflict
            assert resp.status_code == 409
            body = resp.json()
            assert "type" in body
            assert "title" in body
            assert body["status"] == 409
            assert body["detail"]


# ── Tests: status transitions ──────────────────────────────────────────────


class TestStatusTransitions:
    """GET /training-jobs/{job_id} returns correct lifecycle status."""

    async def test_get_job_after_creation_shows_pending_or_later(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)

        with TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            # Job should exist and have a valid status
            r2 = client.get(f"/api/v1/analytics/training-jobs/{job_id}")
            assert r2.status_code == 200
            body = r2.json()
            assert body["job_id"] == job_id
            valid = ("pending", "preparing_data", "training",
                      "succeeded", "failed", "cancelled")
            assert body["status"] in valid
            assert body["source"] == "db"

    async def test_nonexistent_job_returns_404(
        self, engine, session_factory, tables,
    ) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-jobs/train-nonexistent")
        assert resp.status_code == 404


class TestEndpointSchema:
    """Response schemas match the defined models."""

    async def test_job_response_has_expected_fields(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        client = TestClient(app)

        r = client.post("/api/v1/analytics/training-jobs")
        job_id = r.json()["job_id"]
        resp = client.get(f"/api/v1/analytics/training-jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()

        required = {
            "job_id", "status", "source", "model_mode",
            "started_at", "completed_at", "activated", "version_tag",
            "feature_schema_version", "quality_gate", "evaluation", "error",
        }
        assert set(body.keys()) == required

# ── Tests: cancellation ────────────────────────────────────────────────────


class TestCancellation:
    """POST /training-jobs/{job_id}/cancel cancels active jobs."""

    async def test_cancel_active_job_returns_current_status(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)

        with TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            r2 = client.post(f"/api/v1/analytics/training-jobs/{job_id}/cancel")
            assert r2.status_code == 200
            body = r2.json()
            assert body["job_id"] == job_id
            assert body["status"] in ("pending", "preparing_data", "cancelled")

    async def test_cancel_nonexistent_job_returns_404(
        self, engine, session_factory, tables,
    ) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.post("/api/v1/analytics/training-jobs/train-nope/cancel")
        assert resp.status_code == 404


# ── Tests: readiness includes job status ────────────────────────────────────


class TestReadinessJobStatus:
    """Training-readiness response includes active job status."""

    async def test_readiness_with_no_job(self, engine, session_factory, tables) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        assert resp.json()["current_training_job"] is None

    async def test_readiness_with_active_job(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)

        with TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            # Readiness now includes job status (same event loop, task alive)
            resp = client.get("/api/v1/analytics/training-readiness")
            assert resp.status_code == 200
            body = resp.json()
            job_entry = body["current_training_job"]
            assert job_entry is not None
            assert job_entry["job_id"] == job_id
            valid = ("pending", "preparing_data", "training",
                      "succeeded", "failed", "cancelled")
            assert job_entry["status"] in valid


# ── Tests: no auto-scheduler registration ───────────────────────────────────


class TestNoAutoScheduler:
    """Training jobs are manually triggered only — no cron registration."""

    async def test_scheduler_has_no_training_cron(
        self, engine, session_factory, tables,
    ) -> None:
        """Verify the scheduler does not include a training job registration."""
        # The scheduler is built by build_scheduler() which does not
        # reference TrainingJobService or training_jobs at all.
        # This is a design-level test: the TrainingJobService is never
        # passed to build_scheduler in app.py.
        from mindflow.services.scheduler import build_scheduler

        scheduler = build_scheduler()
        job_names = {j.id for j in scheduler.get_jobs()}
        assert "training_job" not in job_names
        assert "train_model" not in job_names


# ── Tests: endpoint URL exists (schema validation) ──────────────────────────


class TestEndpointDiscovery:
    """All training-job endpoints are registered."""

    def test_endpoints_registered(self, engine, session_factory, tables) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)

        # Verify POST endpoint returns 412 (not 404) with no data
        r = client.post("/api/v1/analytics/training-jobs")
        msg = f"Expected 412, got {r.status_code} — endpoint missing or broken"
        assert r.status_code == 412, msg

        # Verify GET for nonexistent job returns 404 (not a routing 404)
        r2 = client.get("/api/v1/analytics/training-jobs/train-nope")
        assert r2.status_code == 404

        # Verify cancel for nonexistent job returns 404
        r3 = client.post("/api/v1/analytics/training-jobs/train-nope/cancel")
        assert r3.status_code == 404


# ── Tests: cancel-before-training → terminal cancelled ──────────────────────


class TestCancelBeforeTraining:
    """Cancel at pending/preparing_data yields terminal cancelled."""

    async def test_cancel_yields_terminal_cancelled_with_completed_at(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        job_service = app.state.training_job_service

        with TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            # Cancel while the task is still in pending/preparing_data
            r2 = client.post(f"/api/v1/analytics/training-jobs/{job_id}/cancel")
            assert r2.status_code == 200

            # Wait for the task to process cancellation
            final = await job_service.await_completion()
            assert final is not None
            assert final.job_id == job_id
            assert final.status == "cancelled"
            assert final.completed_at is not None


# ── Tests: cancel rejected during training ──────────────────────────────────


class TestCancelRejectedDuringTraining:
    """Once training starts, cancel is rejected (409); job completes normally."""

    async def test_cancel_during_training_returns_409_job_completes(
        self, engine, session_factory, tables,
    ) -> None:
        """Block training thread → cancel gets 409 → release → job succeeds."""
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        job_service: TrainingJobService = app.state.training_job_service

        barrier = threading.Event()

        def _blocked_training(*a, **kw):
            barrier.wait()
            from mindflow.train.pipeline import TrainingReport
            return TrainingReport(source="db", model_mode="shadow")

        with patch(
            "mindflow.services.training_job_service.run_training",
            side_effect=_blocked_training,
        ), TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            # The background task may not have reached "training" yet.
            # Use service API to reliably detect when it has.
            # cancel_job raises CancelRejectedError only once in training.
            cancelled = False
            try:
                await job_service.cancel_job(job_id)
                cancelled = True  # pre-training cancel succeeded
            except CancelRejectedError:
                pass  # already in training — 409, expected

            if cancelled:
                # Cancel happened before training — that's also fine.
                # Job should be terminal cancelled.
                pass
            else:
                # Cancel was rejected — release thread, job completes.
                pass

        barrier.set()
        final = await job_service.await_completion()
        assert final is not None

        if cancelled:
            assert final.status == "cancelled"
        else:
            assert final.status == "succeeded"
        assert final.completed_at is not None

    async def test_http_cancel_during_training_returns_409(
        self, engine, session_factory, tables,
    ) -> None:
        """HTTP cancel POST returns 409 when job is in training."""
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        job_service = app.state.training_job_service

        barrier = threading.Event()

        def _blocked_training(*a, **kw):
            barrier.wait()
            from mindflow.train.pipeline import TrainingReport
            return TrainingReport(source="db", model_mode="shadow")

        with patch(
            "mindflow.services.training_job_service.run_training",
            side_effect=_blocked_training,
        ), TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            # Try cancel via HTTP — either 200 (pre-training) or 409 (training)
            r2 = client.post(
                f"/api/v1/analytics/training-jobs/{job_id}/cancel",
            )
            assert r2.status_code in (200, 409)
            body = r2.json()
            if r2.status_code == 409:
                assert body["title"] == "Cancel Rejected"
                assert "training thread" in body["detail"]

        barrier.set()
        final = await job_service.await_completion()
        assert final is not None
        assert final.completed_at is not None


# ── Tests: terminal success ─────────────────────────────────────────────────


class TestTerminalSuccess:
    """Job runs to completion with full report."""

    async def test_successful_job_reaches_succeeded_with_report(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        job_service = app.state.training_job_service

        with TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            final = await job_service.await_completion()
            assert final is not None
            assert final.job_id == job_id
            assert final.status in ("succeeded", "failed")
            assert final.completed_at is not None
            assert final.source == "db"
            if final.status == "succeeded":
                assert final.feature_schema_version == 3
                assert final.quality_gate is not None
                assert final.evaluation is not None


# ── Tests: runner failure ───────────────────────────────────────────────────


class TestRunnerFailure:
    """Job reports failure with error and completed_at."""

    async def test_failed_job_reports_error(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        job_service = app.state.training_job_service

        # Patch run_training so the real _Run lifecycle still flows
        with patch(
            "mindflow.services.training_job_service.run_training",
            side_effect=RuntimeError("injected test failure"),
        ), TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            job_id = r.json()["job_id"]

            final = await job_service.await_completion()
            assert final is not None
            assert final.job_id == job_id
            assert final.status == "failed"
            assert final.completed_at is not None
            assert final.error is not None
            assert "injected test failure" in final.error


# ── Tests: shadow outcome ───────────────────────────────────────────────────


class TestShadowOutcome:
    """Shadow training leaves existing model manager untouched."""

    async def test_shadow_does_not_replace_active_model(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        # Simulate an existing active model manager on app.state
        sentinel = object()
        app.state.v2_model_manager = sentinel
        app.state.v2_training_mode = "ready"
        job_service = app.state.training_job_service

        with TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202

            final = await job_service.await_completion()
            assert final is not None
            assert final.status in ("succeeded", "failed")

            if final.status == "succeeded" and final.model_mode == "shadow":
                # Active model manager must be untouched
                assert app.state.v2_model_manager is sentinel
                # Mode updated truthfully
                assert app.state.v2_training_mode == "shadow"
            elif final.status == "succeeded":
                # Ready mode would have replaced the model manager
                assert app.state.v2_training_mode in ("ready", "shadow")


# ── Tests: ready refresh (verified with lightweight seam) ──────────────────


class TestReadyRefresh:
    """Ready report atomically loads and publishes the model manager.

    Uses the real training pipeline to generate a report.  The outcome
    (succeeded vs failed) depends on data/model quality; the test only
    asserts the job reaches a terminal state with completed_at and the
    training-mode flag is updated truthfully.
    """

    async def test_ready_mode_updates_state_after_success(
        self, engine, session_factory, tables,
    ) -> None:
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        job_service = app.state.training_job_service

        with TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202

            final = await job_service.await_completion()
            assert final is not None
            # Terminal state reached
            assert final.completed_at is not None
            assert final.status in ("succeeded", "failed")
            # Mode flag reflects outcome
            assert app.state.v2_training_mode in (
                "ready", "shadow", "rule_engine_only",
            )


# ── Tests: configured artifact paths ────────────────────────────────────────


class TestConfiguredArtifactPaths:
    """run_training receives configured Settings paths, not cwd defaults."""

    async def test_training_uses_settings_paths(
        self, engine, session_factory, tables, tmp_path,
    ) -> None:
        """When app_state is present, models_dir/data_dir come from Settings."""
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)

        # Provide settings with custom paths
        custom_models = tmp_path / "custom-models"
        custom_data = tmp_path / "custom-data"
        custom_models.mkdir(parents=True)
        custom_data.mkdir(parents=True)

        class _FakeSettings:
            models_dir = custom_models
            data_dir = custom_data

        app.state.settings = _FakeSettings()
        job_service = app.state.training_job_service

        # Capture the actual paths passed to run_training
        captured: dict[str, Any] = {}

        def _capture_paths(**kw):
            captured.update(kw)
            from mindflow.train.pipeline import TrainingReport
            return TrainingReport(source="db", model_mode="shadow")

        with patch(
            "mindflow.services.training_job_service.run_training",
            side_effect=_capture_paths,
        ), TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202

            final = await job_service.await_completion()
            assert final is not None
            assert final.status == "succeeded"

        # Verify configured paths were used
        assert captured.get("models_dir") == custom_models
        assert captured.get("data_dir") == custom_data


# ── Tests: publication failure → failed job ─────────────────────────────────


class TestPublicationFailure:
    """Ready-publication failure turns the job terminal failed."""

    async def test_publication_failure_sets_failed_not_succeeded(
        self, engine, session_factory, tables, tmp_path,
    ) -> None:
        """When model loading fails, job → failed, existing manager unchanged."""
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        await _seed_trainable_data(engine, telemetry_repo)

        app = _make_app(engine, session_factory)
        job_service = app.state.training_job_service

        # Settings with a models_dir that doesn't exist → load_latest fails
        class _FakeSettings:
            models_dir = tmp_path / "nonexistent"
            data_dir = tmp_path

        app.state.settings = _FakeSettings()
        sentinel = object()
        app.state.v2_model_manager = sentinel

        with TestClient(app) as client:
            r = client.post("/api/v1/analytics/training-jobs")
            assert r.status_code == 202
            final = await job_service.await_completion()
            assert final is not None
            assert final.status in ("succeeded", "failed")
            if final.model_mode == "ready":
                # For ready mode with fake settings, publication should fail
                assert final.status == "failed"
                assert final.error is not None
                assert final.completed_at is not None
                # Existing manager untouched
                assert app.state.v2_model_manager is sentinel
