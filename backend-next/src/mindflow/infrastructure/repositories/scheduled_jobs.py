"""Atomic persistence for scheduled job run claims."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

SCHEDULED_JOB_RUN_LEASE = timedelta(minutes=30)

metadata = sa.MetaData()
scheduled_job_runs = sa.Table(
    "scheduled_job_runs",
    metadata,
    sa.Column("job_name", sa.Text(), nullable=False),
    sa.Column("local_date", sa.Text(), nullable=False),
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("attempt_count", sa.Integer(), nullable=False),
    sa.Column("started_at", sa.Text(), nullable=False),
    sa.Column("heartbeat_at", sa.Text(), nullable=False),
    sa.Column("finished_at", sa.Text(), nullable=True),
    sa.Column("last_error", sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint("job_name", "local_date"),
)


class ScheduledJobRunsRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def claim(
        self, job_name: str, local_date: date, *, retry_failed: bool = False
    ) -> int | None:
        now = datetime.now(UTC)
        values = {
            "job_name": job_name,
            "local_date": local_date.isoformat(),
            "status": "running",
            "attempt_count": 1,
            "started_at": now.isoformat(),
            "heartbeat_at": now.isoformat(),
            "finished_at": None,
            "last_error": None,
        }
        async with self._session_factory() as session, session.begin():
            statement = (
                sqlite_insert(scheduled_job_runs)
                .values(**values)
                .on_conflict_do_nothing(index_elements=["job_name", "local_date"])
                .returning(scheduled_job_runs.c.attempt_count)
            )
            result = await session.execute(statement)
            claimed_attempt = result.scalar_one_or_none()
            if claimed_attempt is not None:
                return int(claimed_attempt)

            claimable = sa.and_(
                scheduled_job_runs.c.status == "running",
                sa.func.coalesce(
                    scheduled_job_runs.c.heartbeat_at,
                    scheduled_job_runs.c.started_at,
                )
                <= (now - SCHEDULED_JOB_RUN_LEASE).isoformat(),
            )
            if retry_failed:
                claimable = sa.or_(
                    claimable,
                    scheduled_job_runs.c.status.in_(("failed", "cancelled")),
                )
            retry = (
                sa.update(scheduled_job_runs)
                .where(
                    scheduled_job_runs.c.job_name == job_name,
                    scheduled_job_runs.c.local_date == local_date.isoformat(),
                    claimable,
                )
                .values(
                    status="running",
                    attempt_count=scheduled_job_runs.c.attempt_count + 1,
                    started_at=now.isoformat(),
                    heartbeat_at=now.isoformat(),
                    finished_at=None,
                    last_error=None,
                )
                .returning(scheduled_job_runs.c.attempt_count)
            )
            retry_result = await session.execute(retry)
            retry_attempt = retry_result.scalar_one_or_none()
            return int(retry_attempt) if retry_attempt is not None else None

    async def heartbeat(
        self, job_name: str, local_date: date, *, attempt_count: int
    ) -> bool:
        statement = (
            sa.update(scheduled_job_runs)
            .where(
                scheduled_job_runs.c.job_name == job_name,
                scheduled_job_runs.c.local_date == local_date.isoformat(),
                scheduled_job_runs.c.status == "running",
                scheduled_job_runs.c.attempt_count == attempt_count,
            )
            .values(heartbeat_at=datetime.now(UTC).isoformat())
            .returning(scheduled_job_runs.c.attempt_count)
        )
        async with self._session_factory() as session, session.begin():
            result = await session.execute(statement)
            return result.scalar_one_or_none() is not None

    async def has_succeeded(self, job_name: str, local_date: date) -> bool:
        statement = sa.select(
            sa.exists().where(
                scheduled_job_runs.c.job_name == job_name,
                scheduled_job_runs.c.local_date == local_date.isoformat(),
                scheduled_job_runs.c.status == "succeeded",
            )
        )
        async with self._session_factory() as session:
            return bool(await session.scalar(statement))

    async def mark_succeeded(
        self, job_name: str, local_date: date, *, attempt_count: int
    ) -> bool:
        return await self._finish(
            job_name,
            local_date,
            attempt_count=attempt_count,
            status="succeeded",
            error=None,
        )

    async def mark_failed(
        self,
        job_name: str,
        local_date: date,
        *,
        attempt_count: int,
        error: str,
    ) -> bool:
        return await self._finish(
            job_name,
            local_date,
            attempt_count=attempt_count,
            status="failed",
            error=error,
        )

    async def mark_cancelled(
        self, job_name: str, local_date: date, *, attempt_count: int
    ) -> bool:
        return await self._finish(
            job_name,
            local_date,
            attempt_count=attempt_count,
            status="cancelled",
            error="cancelled",
        )

    async def _finish(
        self,
        job_name: str,
        local_date: date,
        *,
        attempt_count: int,
        status: str,
        error: str | None,
    ) -> bool:
        statement = (
            sa.update(scheduled_job_runs)
            .where(
                scheduled_job_runs.c.job_name == job_name,
                scheduled_job_runs.c.local_date == local_date.isoformat(),
                scheduled_job_runs.c.status == "running",
                scheduled_job_runs.c.attempt_count == attempt_count,
            )
            .values(status=status, finished_at=datetime.now(UTC).isoformat(), last_error=error)
            .returning(scheduled_job_runs.c.attempt_count)
        )
        async with self._session_factory() as session, session.begin():
            result = await session.execute(statement)
            return result.scalar_one_or_none() is not None
