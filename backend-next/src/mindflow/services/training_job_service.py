"""V2 training jobs with ordered persistence and guarded runtime publication.

One active job per process. The synchronous ``run_training`` pipeline is
dispatched to a worker thread via ``asyncio.to_thread``.

Activation policy: ``start_job(allow_activation=...)`` decides whether the run
may move the active-model pointer. The default (``True``) is the
user-confirmed/manual path; automatic scheduler-driven runs pass ``False`` and
therefore only ever produce a shadow candidate.

Cancellation contract:
- ``pending`` / ``preparing_data`` → terminal ``cancelled`` synchronously.
- Once ``training`` starts (``asyncio.to_thread`` entered), cancellation
  is rejected with 409.  A job started with ``allow_activation=True`` may
  already write artifacts including ``save_all(activate=True)``, so the service
  cannot guarantee that activation was prevented even if it signalled
  cancellation earlier.  The safe contract: let the job run to terminal
  succeeded/failed.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

from loguru import logger

from mindflow.api.schemas import JobStatus, TrainingJobResponse, TrainingJobSummary
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.infrastructure.repositories.focus import SQLAlchemyFocusSessionRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.repositories.training_jobs import TrainingJobRepository
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
    # Publication guard evidence (in-memory only, like ``allow_activation``):
    # which candidate the evaluation selected, which classifier was actually
    # trained, and why activation was blocked when it was.
    publication: dict[str, Any] | None = None
    evaluation_candidate: str | None = None
    deployed_classifier: str | None = None
    activation_blocked_reason: str | None = None
    # In-memory only: whether this job may move the active-model pointer.
    # Deliberately not persisted (no DB column/migration) — a job recovered
    # from a historical row reports the backward-compatible default ``True``.
    allow_activation: bool = True
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
            allow_activation=self.allow_activation,
            publication=self.publication,
            evaluation_candidate=self.evaluation_candidate,
            deployed_classifier=self.deployed_classifier,
            activation_blocked_reason=self.activation_blocked_reason,
        )

    def to_summary(self) -> TrainingJobSummary:
        return TrainingJobSummary(
            job_id=self.job_id,
            status=self.status,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )


@dataclass
class _PublicationSnapshot:
    latest_path: Path
    latest_text: str | None
    manager: Any
    mode: str
    consumers: list[tuple[Any, Any]]


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
        jobs_repo: TrainingJobRepository | None = None,
    ) -> None:
        self._telemetry_repo = telemetry_repo
        self._focus_repo = focus_repo
        self._user_id = user_id
        self._lock = asyncio.Lock()
        self._current: _JobState | None = None
        # Optional persistence: when wired, job lifecycle survives restarts and
        # in-flight runs are recorded as ``interrupted`` rather than vanishing.
        self._jobs_repo = jobs_repo
        self._persist_tail: asyncio.Task[None] | None = None
        self._persist_errors: list[Exception] = []
        self._closing = False

    async def recover_after_restart(self) -> int:
        """Mark any non-terminal persisted job ``interrupted``.

        Called once during startup. Returns how many rows were changed so the
        caller can log it. No-op when persistence is not wired.
        """
        if self._jobs_repo is None:
            return 0
        count = await self._jobs_repo.mark_interrupted(self._user_id)
        row = await self._jobs_repo.latest_for_user(self._user_id)
        if row is not None:
            response = TrainingJobResponse.model_validate(row)
            self._current = _JobState(**response.model_dump())
            self._current._done.set()
        return count

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def current_job(self) -> TrainingJobSummary | None:
        """Return a snapshot of the active/latest job, or None."""
        if self._current is None:
            return None
        return self._current.to_summary()

    def get_job(self, job_id: str) -> TrainingJobResponse | None:
        """Return the current in-memory job (legacy synchronous contract)."""
        if self._current is not None and self._current.job_id == job_id:
            return self._current.to_response()
        return None

    async def get_job_detail(self, job_id: str) -> TrainingJobResponse | None:
        """Resolve current or historical detail within this service's user scope."""
        current = self.get_job(job_id)
        if current is not None:
            return current
        if self._jobs_repo is not None:
            row = await self._jobs_repo.get(job_id, user_id=self._user_id)
            if row is not None:
                return TrainingJobResponse.model_validate(row)
        return None

    async def start_job(
        self,
        *,
        app_state: _AppStateLike | None = None,
        allow_activation: bool = True,
    ) -> TrainingJobResponse:
        """Create and dispatch a training job.

        The caller must ensure ``trainable`` is True BEFORE calling
        this method.  The lock prevents TOCTOU races between the
        readiness check and job creation.

        Args:
            app_state: Optional ``app.state`` for post-training
                       model-manager refresh and artifact paths.
                       If None, refresh is skipped and default paths used.
            allow_activation: Whether the job may move the active-model
                       pointer. Defaults to True for the user-confirmed
                       path; ``auto_train_if_due`` passes False so automatic
                       runs can only produce a shadow candidate.

        Returns:
            A 202-style response with job id and ``pending`` status.

        Raises:
            ConcurrencyError: Another job is already active (409).
        """
        async with self._lock:
            if self._closing:
                raise ConcurrencyError("Training job service is shutting down")
            if self._current is not None and not _is_terminal(self._current.status):
                raise ConcurrencyError(
                    f"Training job {self._current.job_id} is already active "
                    f"(status={self._current.status})"
                )

            await self.flush()
            job = _JobState(
                job_id=f"train-{uuid.uuid4().hex[:12]}",
                status="pending",
                started_at=datetime.now(UTC).isoformat(),
                allow_activation=allow_activation,
            )
            if self._jobs_repo is not None:
                await self._jobs_repo.create(
                    job_id=job.job_id,
                    user_id=self._user_id,
                    started_at=job.started_at,
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

    # ── Auto incremental training (architecture plan F/2.2) ─────────────

    _AUTO_MIN_NEW_FEEDBACK: int = 5
    _AUTO_MIN_INTERVAL_HOURS: int = 24

    async def auto_train_if_due(
        self,
        *,
        app_state: _AppStateLike | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Trigger a shadow training run when new feedback has accumulated.

        Architecture plan F/2.2: the model should improve on its own as the
        user keeps giving feedback, without a manual click. This method is
        called from the scheduler (hourly). It starts a job only when:
          - at least ``_AUTO_MIN_NEW_FEEDBACK`` explicit feedback rows were
            created since the last run, AND
          - the previous job finished more than ``_AUTO_MIN_INTERVAL_HOURS``
            ago (or never ran).

        The job always runs in shadow mode (never auto-activates), keeping
        the manual activation path authoritative.  That is enforced, not just
        documented: the job is started with ``allow_activation=False`` so the
        pipeline writes the candidate as a shadow version and leaves the
        active-model pointer (``latest.json``) untouched.  Only a
        user-confirmed manual run (``start_job(allow_activation=True)``) or an
        independent publication task may move the active pointer.

        Returns:
            True when a training job was started, False otherwise.
        """
        # Guard: an active job already running.
        if self._current is not None and not _is_terminal(self._current.status):
            return False

        # Cooldown since the last completed run.
        now = now or datetime.now(UTC)
        if self._current is not None and self._current.completed_at is not None:
            try:
                last_done = datetime.fromisoformat(self._current.completed_at)
                if (
                    (now - last_done).total_seconds()
                    < self._AUTO_MIN_INTERVAL_HOURS * 3600
                ):
                    return False
            except (ValueError, TypeError):
                pass

        # Count feedback rows created since the last run (or all rows).
        since = None
        if self._current is not None and self._current.completed_at is not None:
            since = self._current.completed_at
        try:
            count = await self._telemetry_repo.count_focus_feedback_since(
                self._user_id, since=since
            )
        except Exception:
            return False
        if count < self._AUTO_MIN_NEW_FEEDBACK:
            return False

        await self.start_job(app_state=app_state, allow_activation=False)
        return True

    async def cancel_job(self, job_id: str) -> TrainingJobResponse | None:
        """Cancel a job in ``pending`` or ``preparing_data``.

        Returns None if no job with the given id exists.
        Rejects cancellation (raises ``CancelRejectedError``) once
        the job has entered ``training`` — the thread may already
        have written activated artifacts.
        """
        async with self._lock:
            if self._current is None or self._current.job_id != job_id:
                historical = await self.get_job_detail(job_id)
                return historical if historical and _is_terminal(historical.status) else None
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
        await self.flush()
        return job.to_response()

    async def shutdown(self) -> None:
        """Stop admission, join the owned worker, then flush before DB teardown."""
        async with self._lock:
            self._closing = True
            if self._current is not None and not _is_terminal(self._current.status):
                self._current._cancelled.set()
        await _join_owned_task(asyncio.create_task(self._shutdown()))

    async def _shutdown(self) -> None:
        task: asyncio.Task[None] | None = None
        if self._current is not None:
            task = self._current._task
        if task is not None:
            # Cancelling to_thread only cancels the awaiter, not disk writes.
            # Once training starts, join it through publication before teardown.
            if self._current is not None and self._current.status in (
                "pending", "preparing_data",
            ):
                task.cancel()
            try:
                await _join_owned_task(task)
            except asyncio.CancelledError:
                if self._current is not None and not self._current._done.is_set():
                    self._set_terminal(self._current, "cancelled")
        try:
            await self.flush()
        finally:
            if task is not None and task.done() and self._current is not None:
                self._current._done.set()

    async def flush(self) -> None:
        """Wait for ordered lifecycle writes; never silently acknowledge a failed flush."""
        while self._persist_tail is not None:
            tail = self._persist_tail
            await _join_owned_task(tail)
            if tail is self._persist_tail:
                break
        if self._persist_errors:
            raise RuntimeError("Training job persistence failed") from self._persist_errors[0]

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
        snapshot: _PublicationSnapshot | None = None
        try:
            # ── Phase: preparing_data ──────────────────────────────────
            if job._cancelled.is_set():
                self._set_terminal(job, "cancelled")
                return
            self._set_status(job, "preparing_data")

            uid = self._user_id
            windows = await self._telemetry_repo.list_feature_windows(
                uid, feature_schema_version=FEATURE_SCHEMA_VERSION,
            )
            sessions = await self._focus_repo.list_all(uid)
            session_map: dict[str, dict[str, Any]] = {s["id"]: s for s in sessions}
            feedback_raw = await self._telemetry_repo.list_focus_feedback(uid)

            feedback_with_times: list[dict[str, Any]] = []
            for fb in feedback_raw:
                sid = fb["session_id"]
                fcs = session_map.get(sid)
                start_time = fb.get("session_start_utc") or (fcs or {}).get("start_time")
                end_time = fb.get("session_end_utc") or (fcs or {}).get("end_time")
                if not start_time or not end_time:
                    continue
                feedback_with_times.append({
                    "session_id": sid,
                    "start_time": start_time,
                    "end_time": end_time,
                    "label": fb["label"],
                    "score": fb["score"],
                    "task_type": fb.get("task_type"),
                })

            # ── Phase: training (offloaded to thread) ──────────────────
            if job._cancelled.is_set():
                self._set_terminal(job, "cancelled")
                return
            snapshot = self._snapshot_publication(app_state, Path(models_dir))
            self._set_status(job, "training")

            # Once we enter asyncio.to_thread, cancellation is no longer
            # accepted — the thread may call save_all(activate=True) when the
            # job was allowed to activate.
            use_window_labels = bool(
                getattr(
                    getattr(app_state, "settings", None),
                    "training_use_window_labels",
                    False,
                )
            )
            worker = asyncio.create_task(asyncio.to_thread(
                run_training,
                source="db",
                data_dir=data_dir,
                models_dir=models_dir,
                feature_windows=windows,
                feedback_sessions=feedback_with_times,
                use_window_labels=use_window_labels,
                allow_activation=job.allow_activation,
            ))
            # Repeated cancellation cannot detach the thread from its owner.
            report = await _join_owned_task(worker)

            # ── After training thread returns ──────────────────────────
            job.activated = report.activated
            job.model_mode = report.model_mode
            job.feature_schema_version = report.feature_schema_version
            job.version_tag = report.version_tag
            job.quality_gate = report.quality_gate
            job.evaluation = report.evaluation
            # Publication-guard evidence travels with the job so a shadow outcome
            # is explainable without reading the on-disk report.
            job.publication = report.publication or None
            job.evaluation_candidate = report.evaluation_candidate
            job.deployed_classifier = report.deployed_classifier
            job.activation_blocked_reason = report.activation_blocked_reason

            # ── Publication ────────────────────────────────────────────
            if report.model_mode == "ready" and app_state is not None:
                try:
                    await _join_owned_task(asyncio.create_task(
                        self._refresh_ready_manager(app_state, report),
                    ))
                    job.activated = True
                except Exception as exc:
                    raise PublicationError(
                        f"Ready-model publication failed: {exc}"
                    ) from exc
            elif report.model_mode == "shadow" and app_state is not None:
                self._update_shadow_mode(app_state, report)

            self._set_terminal(job, "succeeded")

        except asyncio.CancelledError:
            job.error = job.error or "cancelled during training"
            self._set_terminal(job, "cancelled")
        except PublicationError as exc:
            logger.error("Training job {} publication failed: {}", job.job_id, exc)
            job.error = _safe_str(exc)
            # If publication failed after the training thread already activated
            # a disk version, memory and disk can disagree. Roll the pointer
            # back so the running process and the model directory agree again.
            job.activated = False
            self._rollback_activation(app_state, snapshot, job)
            self._set_terminal(job, "failed")
        except Exception as exc:
            logger.opt(exception=True).error(
                "Training job {} failed: {}", job.job_id, exc,
            )
            job.error = _safe_str(exc)
            job.activated = False
            self._rollback_activation(app_state, snapshot, job)
            self._set_terminal(job, "failed")
        finally:
            try:
                await self.flush()
            except RuntimeError:
                detail = "PersistenceError: lifecycle state could not be durably confirmed"
                job.error = f"{job.error}; {detail}" if job.error else detail
                logger.exception("Training job {} could not flush lifecycle state", job.job_id)
            finally:
                job._done.set()

    @staticmethod
    def _snapshot_publication(
        app_state: _AppStateLike | None, models_dir: Path,
    ) -> _PublicationSnapshot:
        latest = models_dir / "v2" / "latest.json"
        consumers = []
        for name in ("prediction_service", "telemetry_service"):
            consumer = getattr(app_state, name, None)
            if consumer is not None:
                consumers.append((consumer, getattr(consumer, "_model_manager", None)))
        return _PublicationSnapshot(
            latest_path=latest,
            latest_text=latest.read_text(encoding="utf-8") if latest.exists() else None,
            manager=getattr(app_state, "v2_model_manager", None),
            mode=getattr(app_state, "v2_training_mode", "rule_engine_only"),
            consumers=consumers,
        )

    def _rollback_activation(
        self,
        app_state: _AppStateLike | None,
        snapshot: _PublicationSnapshot | None,
        job: _JobState,
    ) -> None:
        """Restore the pre-training pointer and every pre-publication reference."""
        if snapshot is None:
            return
        if app_state is not None:
            app_state.v2_model_manager = snapshot.manager
            app_state.v2_training_mode = snapshot.mode
        # Both consumers own this field; do not re-enter an attach hook that
        # may itself have raised after assigning the candidate.
        for consumer, manager in snapshot.consumers:
            consumer._model_manager = manager
        try:
            if snapshot.latest_text is None:
                snapshot.latest_path.unlink(missing_ok=True)
            else:
                ModelManager._atomic_write_text(snapshot.latest_path, snapshot.latest_text)
        except OSError as exc:
            job.error = f"{job.error}; disk rollback failed: {_safe_str(exc)}"
            logger.error("Model pointer rollback failed: {}", exc)

    # ── Status helpers ──────────────────────────────────────────────────────

    def _set_status(self, job: _JobState, status: JobStatus) -> None:
        if _is_terminal(job.status):
            return
        job.status = status
        logger.info("Training job {} → {}", job.job_id, status)
        # Serialize snapshots; completion/shutdown explicitly verify the flush.
        self._schedule_persist(job, status=status)

    def _set_terminal(self, job: _JobState, status: JobStatus) -> None:
        if _is_terminal(job.status):
            return
        now = datetime.now(UTC).isoformat()
        if job.completed_at is None:
            job.completed_at = now
        job.status = status
        logger.info("Training job {} → {}", job.job_id, status)
        self._schedule_persist(job, status=status, terminal=True)

    def _schedule_persist(
        self,
        job: _JobState,
        *,
        status: str | None = None,
        terminal: bool = False,
    ) -> None:
        """Queue an immutable snapshot behind the previous lifecycle write."""
        if self._jobs_repo is None:
            return
        repo = self._jobs_repo
        previous = self._persist_tail
        values = job.to_response().model_dump()
        values.pop("job_id")
        values.pop("source")
        values.pop("started_at")
        # In-memory-only policy flag: the ``training_jobs`` row schema is frozen
        # (no column, no migration), so it must never reach ``repo.update``.
        values.pop("allow_activation")
        # Same for the publication-guard evidence (plan item 1): report/manifest
        # carry it, the ``training_jobs`` row does not.
        for in_memory_only in (
            "publication",
            "evaluation_candidate",
            "deployed_classifier",
            "activation_blocked_reason",
        ):
            values.pop(in_memory_only, None)
        values["status"] = status
        values["completed_at"] = job.completed_at if terminal else None
        values["activated"] = job.activated if terminal else None

        async def persist() -> None:
            if previous is not None:
                await previous
            try:
                await repo.update(job.job_id, user_id=self._user_id, **values)
            except Exception as exc:
                self._persist_errors.append(exc)
                logger.warning("Training job persistence write failed: {}", exc)

        self._persist_tail = asyncio.create_task(persist())

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
        if not report.activated or new_manager.current_version_tag != report.version_tag:
            raise PublicationError("active disk version does not match the ready training report")

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


_T = TypeVar("_T")


async def _join_owned_task(task: asyncio.Task[_T]) -> _T:
    """Defer caller cancellation until owned side effects have really finished."""
    while not task.done():
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Shield every retry: a second cancel must not reach the child.
            continue
    return task.result()


def _is_terminal(status: JobStatus) -> bool:
    return status in ("succeeded", "failed", "cancelled", "interrupted")


def _safe_str(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
