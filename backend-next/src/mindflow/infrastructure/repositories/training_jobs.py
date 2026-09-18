"""Persistence for training job lifecycle state.

``TrainingJobService`` used to hold ``_current`` purely in memory, so a restart
erased the record of any in-flight run. The table written here keeps the
lifecycle observable across restarts, and — importantly — makes an
interrupted run visible as ``interrupted`` rather than leaving the UI to
assume either "still running" or "never existed".

Every method opens its own session (project convention: no cross-request
session sharing).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.infrastructure.schema import training_jobs

_TERMINAL = ("succeeded", "failed", "cancelled", "interrupted")
_PREDECESSORS = {
    "pending": ("pending",),
    "preparing_data": ("pending", "preparing_data"),
    "training": ("pending", "preparing_data", "training"),
    **{status: ("pending", "preparing_data", "training") for status in _TERMINAL},
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dumps(value: dict[str, Any] | None) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _loads(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


class TrainingJobRepository:
    """SQLite-backed store for training job rows."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(
        self,
        *,
        job_id: str,
        user_id: int = 1,
        source: str = "db",
        started_at: str | None = None,
    ) -> None:
        """Insert a new job row in ``pending`` state."""
        async with self._session_factory() as session, session.begin():
            await session.execute(
                training_jobs.insert().values(
                    job_id=job_id,
                    user_id=user_id,
                    status="pending",
                    source=source,
                    model_mode="rule_engine_only",
                    started_at=started_at or _now(),
                    created_at=_now(),
                    updated_at=_now(),
                )
            )

    async def update(
        self,
        job_id: str,
        *,
        user_id: int = 1,
        status: str | None = None,
        model_mode: str | None = None,
        completed_at: str | None = None,
        activated: bool | None = None,
        version_tag: str | None = None,
        feature_schema_version: int | None = None,
        quality_gate: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Patch an owned job without moving backwards or rewriting a terminal snapshot."""
        if status is not None and status not in _PREDECESSORS:
            raise ValueError(f"Unknown training job status: {status}")
        values: dict[str, Any] = {"updated_at": _now()}
        if status is not None:
            values["status"] = status
        if model_mode is not None:
            values["model_mode"] = model_mode
        if completed_at is not None:
            values["completed_at"] = completed_at
        if activated is not None:
            values["activated"] = activated
        if version_tag is not None:
            values["version_tag"] = version_tag
        if feature_schema_version is not None:
            values["feature_schema_version"] = feature_schema_version
        if quality_gate is not None:
            values["quality_gate_json"] = _dumps(quality_gate)
        if evaluation is not None:
            values["evaluation_json"] = _dumps(evaluation)
        if error is not None:
            values["error"] = error

        async with self._session_factory() as session, session.begin():
            await session.execute(
                training_jobs.update()
                .where(
                    training_jobs.c.job_id == job_id,
                    training_jobs.c.user_id == user_id,
                    training_jobs.c.status.in_(
                        _PREDECESSORS[status]
                        if status else ("pending", "preparing_data", "training")
                    ),
                )
                .values(**values)
            )

    async def get(self, job_id: str, *, user_id: int = 1) -> dict[str, Any] | None:
        """Return one owned job row as a dict, or None."""
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    training_jobs.select().where(
                        training_jobs.c.job_id == job_id,
                        training_jobs.c.user_id == user_id,
                    )
                )
            ).mappings().first()
        return self._to_dict(row) if row else None

    async def latest_for_user(self, user_id: int = 1) -> dict[str, Any] | None:
        """Return the most recently started job for *user_id*, or None."""
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    training_jobs.select()
                    .where(training_jobs.c.user_id == user_id)
                    .order_by(training_jobs.c.started_at.desc(), training_jobs.c.created_at.desc())
                    .limit(1)
                )
            ).mappings().first()
        return self._to_dict(row) if row else None

    async def mark_interrupted(self, user_id: int = 1) -> int:
        """Mark non-terminal rows as ``interrupted``; return how many changed.

        Called once during startup. A job that was ``training`` when the
        process died cannot be resumed: the thread holding it is gone, and the
        model directory may hold a partially written candidate. Recording
        ``interrupted`` is the honest state — it neither claims success nor
        hides that something was in flight.
        """
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                training_jobs.update()
                .where(
                    training_jobs.c.user_id == user_id,
                    training_jobs.c.status.not_in(_TERMINAL),
                )
                .values(
                    status="interrupted",
                    completed_at=_now(),
                    updated_at=_now(),
                    error="process restarted while the job was running",
                )
            )
            count = int(getattr(result, "rowcount", 0) or 0)
        if count:
            logger.warning(
                "Marked {} training job(s) interrupted after restart", count
            )
        return count

    @staticmethod
    def _to_dict(row: Any) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "user_id": row["user_id"],
            "status": row["status"],
            "source": row["source"],
            "model_mode": row["model_mode"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "activated": bool(row["activated"]),
            "version_tag": row["version_tag"],
            "feature_schema_version": row["feature_schema_version"],
            "quality_gate": _loads(row["quality_gate_json"]),
            "evaluation": _loads(row["evaluation_json"]),
            "error": row["error"],
        }


__all__ = ["TrainingJobRepository"]
