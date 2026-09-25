"""Automatic-training activation policy (Phase 1.2).

One policy, enforced in one place: a run may only move the active-model
pointer (``latest.json``) when the quality gate passed **and** the caller
permitted activation.

Covers:
  - ``run_training(allow_activation=False)`` never activates, even on a
    passing gate; the candidate is still written as a shadow version and the
    suppression is recorded on the report.
  - ``run_training(allow_activation=True)`` still activates on a passing gate
    (no behaviour regression).
  - ``TrainingJobService.auto_train_if_due()`` starts a job with
    ``allow_activation=False`` and threads that through to ``run_training``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.analytics import router as analytics_router
from mindflow.api.schemas import TrainingJobResponse
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.services.training_job_service import TrainingJobService
from mindflow.train.__main__ import main
from mindflow.train.pipeline import TrainingReport, run_training


def _gate_passing_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """V2 windows + explicit feedback that clear every quality gate.

    Same shape as ``test_train_pipeline.test_v2_training_activates_only_after_all_gates_pass``:
    eight distinct feedback days, four sessions per day, both classes present.
    """
    start = datetime(2026, 7, 1, 8, tzinfo=UTC)
    feature_windows: list[dict[str, Any]] = []
    feedback_sessions: list[dict[str, Any]] = []
    session_index = 0
    for day_index in range(8):
        # 12 sessions per day (6 focus / 6 distracted) so that even the first
        # forward-chaining chunk — which is training-only history — holds
        # enough labelled, two-class rows for the fold to be evaluable. With
        # the stricter phase-3.1 gates a tiny fixture correctly stays in
        # shadow, which would make this activation-policy test vacuous.
        for class_index in range(12):
            is_focus = class_index % 2 == 0
            session_start = start + timedelta(days=day_index, hours=class_index)
            feature_windows.append({
                "window_start_utc": session_start.isoformat(),
                "window_end_utc": (session_start + timedelta(minutes=5)).isoformat(),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "features": {
                    "idle_ratio": 0.01 if is_focus else 0.7,
                    "longest_segment_ratio": 0.98 if is_focus else 0.05,
                    "top_app_ratio": 0.98 if is_focus else 0.1,
                    "input_active_ratio": 0.7 if is_focus else 0.05,
                    "app_switch_count": 0 if is_focus else 12,
                    "domain_switch_count": 0 if is_focus else 8,
                },
            })
            feedback_sessions.append({
                "session_id": f"session-{session_index}",
                "start_time": session_start.isoformat(),
                "end_time": (session_start + timedelta(minutes=30)).isoformat(),
                "label": "focus" if is_focus else "distracted",
                "score": 5 if is_focus else 1,
                "task_type": "coding",
            })
            session_index += 1
    return feature_windows, feedback_sessions


def _run(work_dir: Path, **kwargs: Any) -> TrainingReport:
    feature_windows, feedback_sessions = _gate_passing_data()
    return run_training(
        source="db",
        data_dir=work_dir / "data",
        models_dir=work_dir / "models",
        feature_windows=feature_windows,
        feedback_sessions=feedback_sessions,
        calibration=None,  # toy data: skip post-hoc calibration
        **kwargs,
    )


def _v2_dir(work_dir: Path) -> Path:
    return work_dir / "models" / "v2"


def _versioned_report(work_dir: Path, tag: str) -> dict[str, Any]:
    path = _v2_dir(work_dir) / f"training_report-{tag}.json"
    assert path.exists(), f"per-version report missing for tag {tag}"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


# ── Publication guard: the other precondition for activation ────────────────
#
# These tests are about the *activation policy* (may the caller move the active
# pointer?), not about the evaluation-artifact consistency check.  On the toy
# fixture all five candidates are perfect, so the conservative selection rule
# keeps `logistic_regression` while the pipeline trains the ensemble — which the
# publication guard correctly blocks (that case is pinned end-to-end in
# `tests/test_publication_guard.py`).  To keep the two concerns separate, the
# activation tests stub the guard as "confirmed consistent" explicitly.


@pytest.fixture
def consistent_publication(monkeypatch: pytest.MonkeyPatch):
    """Force the publication guard to confirm consistency."""
    from mindflow.train import pipeline as pipeline_module
    from mindflow.train.publication import ENSEMBLE_CANDIDATE, PublicationDecision

    def _allowed(*_args: object, **_kwargs: object) -> PublicationDecision:
        return PublicationDecision(
            True, ENSEMBLE_CANDIDATE, ENSEMBLE_CANDIDATE, "test: consistent",
        )

    monkeypatch.setattr(pipeline_module, "evaluate_publication", _allowed)
    return _allowed


# ── Pipeline: suppressed activation ─────────────────────────────────────────


def test_allow_activation_false_never_activates_on_passing_gate(tmp_path: Path) -> None:
    """A passing gate plus ``allow_activation=False`` stays shadow-only."""
    report = _run(tmp_path, allow_activation=False)

    # The gate really passed — this is suppression, not a gate failure.
    assert report.quality_gate["passed"] is True
    assert report.activated is False
    assert report.model_mode == "shadow"
    assert report.activation_allowed is False
    assert report.activation_suppressed_reason is not None
    assert report.activation_error is None

    # The active pointer was never created.
    assert not (_v2_dir(tmp_path) / "latest.json").exists()

    # The shadow candidate is still on disk with a version tag.
    assert report.version_tag
    assert (_v2_dir(tmp_path) / f"classifier-{report.version_tag}.pkl").exists()
    assert (_v2_dir(tmp_path) / f"manifest-{report.version_tag}.json").exists()

    # Suppression is recorded in the serialized report too.
    persisted = _versioned_report(tmp_path, str(report.version_tag))
    assert persisted["activation_allowed"] is False
    assert persisted["activation_suppressed_reason"] == report.activation_suppressed_reason
    assert persisted["model_mode"] == "shadow"
    assert persisted["activated"] is False


def test_suppressed_run_leaves_existing_active_pointer_byte_identical(
    tmp_path: Path,
    consistent_publication: object,
) -> None:
    """An already-active version survives a suppressed run untouched."""
    first = _run(tmp_path, allow_activation=True)
    assert first.activated is True
    latest = _v2_dir(tmp_path) / "latest.json"
    before = latest.read_bytes()

    second = _run(tmp_path, allow_activation=False)

    assert second.quality_gate["passed"] is True
    assert second.activated is False
    assert second.model_mode == "shadow"
    assert latest.read_bytes() == before
    assert json.loads(before.decode("utf-8")) == json.loads(
        latest.read_text(encoding="utf-8")
    )


def test_allow_activation_true_still_activates_on_passing_gate(
    tmp_path: Path,
    consistent_publication: object,
) -> None:
    """The guard must not regress the explicit, user-requested path."""
    report = _run(tmp_path, allow_activation=True)

    assert report.quality_gate["passed"] is True
    assert report.activated is True
    assert report.model_mode == "ready"
    assert report.activation_allowed is True
    assert report.activation_suppressed_reason is None
    assert report.activation_blocked_reason is None
    assert (_v2_dir(tmp_path) / "latest.json").exists()


def test_default_activation_policy_is_unchanged(
    tmp_path: Path,
    consistent_publication: object,
) -> None:
    """Omitting the parameter keeps the historical (activating) behaviour."""
    report = _run(tmp_path)

    assert report.activation_allowed is True
    assert report.activated is True
    assert report.model_mode == "ready"


# ── CLI: manual runs activate unless the operator opts out ──────────────────


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["mindflow-train", "--source", "synthetic_v2"], True),
        (["mindflow-train", "--source", "synthetic_v2", "--no-activate"], False),
    ],
)
def test_cli_passes_allow_activation_through(
    argv: list[str],
    expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--no-activate`` is the only way a manual CLI run suppresses activation."""
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> TrainingReport:
        captured.update(kwargs)
        return TrainingReport(source="db", total_records=1)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr("mindflow.train.__main__.run_training", _capture)

    main()

    assert captured["allow_activation"] is expected


# ── Job service: automatic runs are shadow-only ─────────────────────────────


def _auto_service(*, feedback_since_last_run: int = 10) -> TrainingJobService:
    return TrainingJobService(
        telemetry_repo=SimpleNamespace(
            list_feature_windows=AsyncMock(return_value=[]),
            list_focus_feedback=AsyncMock(return_value=[]),
            count_focus_feedback_since=AsyncMock(return_value=feedback_since_last_run),
        ),
        focus_repo=SimpleNamespace(list_all=AsyncMock(return_value=[])),
        user_id=1,
    )


async def test_auto_train_if_due_starts_shadow_only_job() -> None:
    """``auto_train_if_due`` marks the job as not allowed to activate."""
    jobs = _auto_service()
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> TrainingReport:
        captured.update(kwargs)
        return TrainingReport(source="db", model_mode="shadow")

    with patch(
        "mindflow.services.training_job_service.run_training",
        side_effect=_capture,
    ):
        started = await jobs.auto_train_if_due()
        assert started is True
        response = jobs.get_job(jobs.current_job.job_id)
        assert response is not None
        assert response.allow_activation is False
        final = await jobs.await_completion()

    assert final is not None
    assert final.status == "succeeded"
    assert final.allow_activation is False
    # The flag reached the training call itself, not just the response.
    assert captured["allow_activation"] is False


async def test_manual_start_job_is_allowed_to_activate_by_default() -> None:
    """The user-confirmed path keeps activation allowed (no API regression)."""
    jobs = _auto_service()
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> TrainingReport:
        captured.update(kwargs)
        return TrainingReport(source="db", model_mode="shadow")

    with patch(
        "mindflow.services.training_job_service.run_training",
        side_effect=_capture,
    ):
        response = await jobs.start_job()
        assert response.allow_activation is True
        await jobs.await_completion()

    assert captured["allow_activation"] is True


async def test_shadow_only_job_does_not_publish_a_ready_model(tmp_path: Path) -> None:
    """End-to-end: a suppressed job never loads/attaches a ready manager."""
    jobs = _auto_service()
    attached: list[Any] = []

    class _Consumer:
        def attach_model_manager(self, manager: Any) -> None:
            attached.append(manager)

    app_state = SimpleNamespace(
        settings=SimpleNamespace(models_dir=tmp_path / "models", data_dir=tmp_path),
        v2_model_manager=None,
        v2_training_mode="rule_engine_only",
        prediction_service=_Consumer(),
        telemetry_service=_Consumer(),
    )

    feature_windows, feedback_sessions = _gate_passing_data()
    with patch(
        "mindflow.services.training_job_service.run_training",
        side_effect=lambda **kwargs: run_training(
            **{
                **kwargs,
                "feature_windows": feature_windows,
                "feedback_sessions": feedback_sessions,
                "calibration": None,  # toy data: skip post-hoc calibration
            },
        ),
    ):
        started = await jobs.auto_train_if_due(app_state=app_state)
        assert started is True
        final = await jobs.await_completion()

    assert final is not None
    assert final.status == "succeeded"
    assert final.allow_activation is False
    assert final.activated is False
    assert final.model_mode == "shadow"
    assert app_state.v2_model_manager is None
    assert app_state.v2_training_mode == "shadow"
    assert attached == []
    assert not ((tmp_path / "models" / "v2") / "latest.json").exists()


async def test_auto_train_if_due_declines_without_new_feedback() -> None:
    """Cooldown/fresh-feedback guard still declines before any job starts."""
    jobs = _auto_service(feedback_since_last_run=0)
    assert await jobs.auto_train_if_due() is False
    assert jobs.current_job is None


# ── API surface: the policy flag is visible and backward compatible ─────────


def _job_app(jobs: TrainingJobService) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.state.training_job_service = jobs
    app.state.v2_model_manager = None
    app.state.v2_training_mode = "rule_engine_only"
    app.state.settings = None
    app.include_router(analytics_router, prefix="/api/v1")
    return app


async def test_auto_started_job_reports_shadow_only_over_http() -> None:
    """An automatic job is observable as shadow-only through the API."""
    jobs = _auto_service()
    with patch(
        "mindflow.services.training_job_service.run_training",
        side_effect=lambda **kwargs: TrainingReport(source="db", model_mode="shadow"),
    ):
        assert await jobs.auto_train_if_due() is True
        job_id = jobs.current_job.job_id
        client = TestClient(_job_app(jobs))
        started = client.get(f"/api/v1/analytics/training-jobs/{job_id}")
        assert started.status_code == 200
        assert started.json()["allow_activation"] is False
        await jobs.await_completion()

    final = client.get(f"/api/v1/analytics/training-jobs/{job_id}")
    assert final.status_code == 200
    assert final.json()["allow_activation"] is False
    assert final.json()["model_mode"] == "shadow"
    assert final.json()["activated"] is False


def test_schema_default_keeps_older_payloads_valid() -> None:
    """Payloads without the new field still validate (backward compatible)."""
    response = TrainingJobResponse.model_validate({
        "job_id": "train-old",
        "status": "succeeded",
        "source": "db",
        "model_mode": "shadow",
        "activated": False,
    })
    assert response.allow_activation is True
