"""Manually-triggered V2 training jobs with in-memory lifecycle management.

One active job per process. The synchronous ``run_training`` pipeline is
dispatched to a worker thread via ``asyncio.to_thread``.

Cancellation contract:
- ``pending`` / ``preparing_data`` → terminal ``cancelled`` synchronously.
- Once ``training`` starts (``asyncio.to_thread`` entered), cancellation
  is rejected with 409.  The training thread may already write artifacts
  including ``save_all(activate=True)``, so the service cannot guarantee
  that activation was prevented even if it signalled cancellation earlier.
  The safe contract: let the job run to terminal succeeded/failed.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from mindflow.api.schemas import JobStatus, TrainingJobResponse, TrainingJobSummary
from mindflow.infrastructure.repositories.focus import SQLAlchemyFocusSessionRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.train.models.manager import ModelManager
from mindflow.train.pipeline import TrainingReport, run_training

# ── Protocol for app.state (avoids Any in public signatures) ────────────────


class _AppStateLike(Protocol):
    """Minimal protocol for the app.state attributes the service needs."""

    settings: Any
    v2_model_manager: Any
    v2_training_mode: str
    prediction_service: Any | None
    telemetry_service: Any | None


# ── Errors ──────────────────────────────────────────────────────────────────


class ConcurrencyError(Exception):
    """Raised when a second training job is requested while one is active."""


class CancelRejectedError(Exception):
    """Raised when cancellation is requested after training has started."""


class PublicationError(Exception):
    """Raised when ready-model publication fails — job becomes failed."""


# ── Internal job state ──────────────────────────────────────────────────────


@dataclass
class _JobState:
    """Internal mutable state for one training job."""

    job_id: str
    status: JobStatus = "pending"
    source: str = "db"
    model_mode: str = "rule_engine_only"
    started_at: str | None = None
    completed_at: str | None = None
    activated: bool = False
    version_tag: str | None = None
    feature_schema_version: int | None = None
    quality_gate: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    error: str | None = None
    _cancelled: threading.Event = field(default_factory=threading.Event)
    _task: asyncio.Task[None] | None = None
    _done: threading.Event = field(default_factory=threading.Event)

    def to_response(self) -> TrainingJobResponse:
        return TrainingJobResponse(
            job_id=self.job_id,
            status=self.status,
            source=self.source,
            model_mode=self.model_mode,
            started_at=self.started_at,
            completed_at=self.completed_at,
            activated=self.activated,
            version_tag=self.version_tag,
            feature_schema_version=self.feature_schema_version,
            quality_gate=self.quality_gate,
            evaluation=self.evaluation,
            error=self.error,
        )

    def to_summary(self) -> TrainingJobSummary:
        return TrainingJobSummary(
            job_id=self.job_id,
            status=self.status,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


# ── Service ─────────────────────────────────────────────────────────────────


class TrainingJobService:
    """Manages the lifecycle of a single in-process training job.

    Guarantees at most one active job via an ``asyncio.Lock`` guard.
    CPU-bound ``run_training`` is dispatched to a thread-pool executor
    so the event loop is never blocked.

    Cancellation is only accepted in ``pending``/``preparing_data``;
    once ``training`` begins, cancellation is rejected (409) because
    the thread may have already called ``save_all(activate=True)``.
    """

    def __init__(
        self,
        telemetry_repo: TelemetryRepository,
        focus_repo: SQLAlchemyFocusSessionRepository,
        user_id: int = 1,
    ) -> None:
        self._telemetry_repo = telemetry_repo
        self._focus_repo = focus_repo
        self._user_id = user_id
        self._lock = asyncio.Lock()
        self._current: _JobState | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def current_job(self) -> TrainingJobSummary | None:
        """Return a snapshot of the active/latest job, or None."""
        if self._current is None:
            return None
        return self._current.to_summary()

    def get_job(self, job_id: str) -> TrainingJobResponse | None:
        """Return full job detail by id, or None if not found."""
        if self._current is None or self._current.job_id != job_id:
            return None
        return self._current.to_response()

    async def start_job(
        self, *, app_state: _AppStateLike | None = None,
    ) -> TrainingJobResponse:
        """Create and dispatch a training job.

        The caller must ensure ``trainable`` is True BEFORE calling
        this method.  The lock prevents TOCTOU races between the
        readiness check and job creation.

        Args:
            app_state: Optional ``app.state`` for post-training
                       model-manager refresh and artifact paths.
                       If None, refresh is skipped and default paths used.

        Returns:
            A 202-style response with job id and ``pending`` status.

        Raises:
            ConcurrencyError: Another job is already active (409).
        """
        async with self._lock:
            if self._current is not None and _is_terminal(self._current.status):
                self._current = None

            if self._current is not None:
                raise ConcurrencyError(
                    f"Training job {self._current.job_id} is already active "
                    f"(status={self._current.status})"
                )

            job = _JobState(
                job_id=f"train-{uuid.uuid4().hex[:12]}",
                status="pending",
                started_at=datetime.now(UTC).isoformat(),
            )
            self._current = job

            # Resolve artifact paths from settings when available.
            models_dir: str | Path = Path("data/models")
            data_dir: str | Path = Path("data")
            if app_state is not None:
                settings = getattr(app_state, "settings", None)
                if settings is not None:
                    models_dir = settings.models_dir
                    data_dir = settings.data_dir

            job._task = asyncio.create_task(
                self._run(
                    job,
                    app_state=app_state,
                    models_dir=models_dir,
                    data_dir=data_dir,
                ),
                name=f"training-job-{job.job_id}",
            )
            return job.to_response()

    async def cancel_job(self, job_id: str) -> TrainingJobResponse | None:
        """Cancel a job in ``pending`` or ``preparing_data``.

        Returns None if no job with the given id exists.
        Rejects cancellation (raises ``CancelRejectedError``) once
        the job has entered ``training`` — the thread may already
        have written activated artifacts.
        """
        async with self._lock:
            if self._current is None or self._current.job_id != job_id:
                return None
            job = self._current
            if _is_terminal(job.status):
                return job.to_response()
            if job.status == "training":
                raise CancelRejectedError(
                    f"Cannot cancel training job {job_id}: "
                    "training thread is already running"
                )
            job._cancelled.set()
            return job.to_response()

    async def await_completion(self) -> TrainingJobResponse | None:
        """Wait for the current job to reach a terminal state.

        Test/convenience API — blocks until the owned task completes.
        Returns the terminal job response, or None if no job exists.
        """
        job = self._current
        if job is None:
            return None
        await asyncio.to_thread(job._done.wait)
        return job.to_response()

    async def shutdown(self) -> None:
        """Cancel any active pre-training job and wait for its task."""
        async with self._lock:
            if self._current is not None and not _is_terminal(self._current.status):
                self._current._cancelled.set()

        task: asyncio.Task[None] | None = None
        if self._current is not None:
            task = self._current._task
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=30.0)
            except asyncio.CancelledError:
                pass
            except TimeoutError:
                logger.warning(
                    "Training job {} did not shut down within 30s",
                    self._current.job_id if self._current else "?",
                )

    # ── Internal ────────────────────────────────────────────────────────────

    async def _run(
        self,
        job: _JobState,
        *,
        app_state: _AppStateLike | None = None,
        models_dir: str | Path = Path("data/models"),
        data_dir: str | Path = Path("data"),
    ) -> None:
        """Background coroutine that manages the training lifecycle."""
        report: TrainingReport | None = None
        try:
            # ── Phase: preparing_data ──────────────────────────────────
            if job._cancelled.is_set():
                self._set_terminal(job, "cancelled")
                return
            self._set_status(job, "preparing_data")

            uid = self._user_id
            windows = await self._telemetry_repo.list_feature_windows(
                uid, feature_schema_version=2,
            )
            sessions = await self._focus_repo.list_all(uid)
            session_map: dict[str, dict[str, Any]] = {s["id"]: s for s in sessions}
            feedback_raw = await self._telemetry_repo.list_focus_feedback(uid)

            feedback_with_times: list[dict[str, Any]] = []
            for fb in feedback_raw:
                sid = fb["session_id"]
                fcs = session_map.get(sid)
                if fcs is None:
                    continue
                feedback_with_times.append({
                    "session_id": sid,
                    "start_time": fcs["start_time"],
                    "end_time": fcs["end_time"],
                    "label": fb["label"],
                    "score": fb["score"],
                    "task_type": fb.get("task_type"),
                })

            # ── Phase: training (offloaded to thread) ──────────────────
            if job._cancelled.is_set():
                self._set_terminal(job, "cancelled")
                return
            self._set_status(job, "training")

            # Once we enter asyncio.to_thread, cancellation is no longer
            # accepted — the thread may call save_all(activate=True).
            report = await asyncio.to_thread(
                run_training,
                source="db",
                data_dir=data_dir,
                models_dir=models_dir,
                feature_windows=windows,
                feedback_sessions=feedback_with_times,
            )

            # ── After training thread returns ──────────────────────────
            job.activated = report.activated
            job.model_mode = report.model_mode
            job.feature_schema_version = report.feature_schema_version
            job.version_tag = report.version_tag
            job.quality_gate = report.quality_gate
            job.evaluation = report.evaluation

            # ── Publication ────────────────────────────────────────────
            if report.model_mode == "ready" and app_state is not None:
                try:
                    await self._refresh_ready_manager(app_state, report)
                except Exception as exc:
                    raise PublicationError(
                        f"Ready-model publication failed: {exc}"
                    ) from exc
            elif report.model_mode == "shadow" and app_state is not None:
                self._update_shadow_mode(app_state, report)

            self._set_terminal(job, "succeeded")

        except asyncio.CancelledError:
            self._set_terminal(job, "cancelled")
        except PublicationError as exc:
            logger.error("Training job {} publication failed: {}", job.job_id, exc)
            job.error = _safe_str(exc)
            self._set_terminal(job, "failed")
        except Exception as exc:
            logger.opt(exception=True).error(
                "Training job {} failed: {}", job.job_id, exc,
            )
            job.error = _safe_str(exc)
            self._set_terminal(job, "failed")
        finally:
            job._done.set()

    # ── Status helpers ──────────────────────────────────────────────────────

    def _set_status(self, job: _JobState, status: JobStatus) -> None:
        job.status = status
        logger.info("Training job {} → {}", job.job_id, status)

    def _set_terminal(self, job: _JobState, status: JobStatus) -> None:
        now = datetime.now(UTC).isoformat()
        if job.completed_at is None:
            job.completed_at = now
        self._set_status(job, status)

    # ── Model-manager refresh ───────────────────────────────────────────────

    async def _refresh_ready_manager(
        self, app_state: _AppStateLike, report: TrainingReport,
    ) -> None:
        """Atomically load and publish the newly-activated model manager.

        Only called when ``report.model_mode == "ready"`` AND quality
        gate passed.  The existing active model is replaced.
        Raises ``PublicationError`` (via the caller's handling) on failure.
        """
        settings = getattr(app_state, "settings", None)
        if settings is None:
            raise PublicationError("app.state.settings not available")

        model_base_dir = settings.models_dir
        new_manager = ModelManager(
            models_dir=model_base_dir / "v2", use_ensemble=False,
        )
        if not new_manager.load_latest():
            raise PublicationError("load_latest() failed for ready models")

        prediction_service = getattr(app_state, "prediction_service", None)
        telemetry_service = getattr(app_state, "telemetry_service", None)
        if prediction_service is not None:
            prediction_service.attach_model_manager(new_manager)
        if telemetry_service is not None:
            telemetry_service.attach_model_manager(new_manager)

        app_state.v2_model_manager = new_manager
        app_state.v2_training_mode = "ready"
        logger.info(
            "Ready model manager activated (version: {})",
            new_manager.current_version_tag,
        )

    def _update_shadow_mode(
        self, app_state: _AppStateLike, report: TrainingReport,
    ) -> None:
        """Record shadow outcome without touching the active model.

        The existing ``v2_model_manager`` and attached prediction/
        telemetry services are left unchanged; only the training
        mode flag is updated truthfully.
        """
        app_state.v2_training_mode = "shadow"
        logger.info(
            "Shadow training completed; active model unchanged "
            "(mode={}, activated={})", report.model_mode, report.activated,
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_terminal(status: JobStatus) -> bool:
    return status in ("succeeded", "failed", "cancelled")


def _safe_str(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
