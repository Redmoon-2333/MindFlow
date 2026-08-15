"""SQLAlchemy-backed task repository (smart-prioritization data source).

Stores user tasks for the ``smart_prioritization`` intervention.  All
timestamps are ISO8601 UTC text, matching the shared schema conventions.

Table schema matches Alembic migration 0023:

  tasks:
    id                 TEXT PK (UUIDv7)
    user_id            INTEGER NOT NULL
    title              TEXT NOT NULL
    description        TEXT NOT NULL
    priority           INTEGER NOT NULL (1..5)
    status             TEXT NOT NULL ('pending' | 'in_progress' | 'done')
    deadline_utc       TEXT (nullable, ISO8601 UTC)
    estimated_minutes  INTEGER (nullable)
    created_at         TEXT NOT NULL (ISO8601 UTC)
    updated_at         TEXT NOT NULL (ISO8601 UTC)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.ids import new_id
from mindflow.domain.tasks import MAX_PRIORITY, MIN_PRIORITY, Task, TaskStatus
from mindflow.infrastructure.schema import tasks as tasks_table


def _parse_deadline(raw: str | None) -> datetime | None:
    """Parse an ISO deadline string into an aware UTC datetime, or None."""
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _row_to_task(row: Any) -> Task:
    """Map a SQLAlchemy row to the frozen Task domain object."""
    return Task(
        id=row.id,
        user_id=row.user_id,
        title=row.title,
        description=row.description or "",
        priority=row.priority,
        status=cast(TaskStatus, row.status),
        deadline_utc=_parse_deadline(row.deadline_utc),
        estimated_minutes=row.estimated_minutes,
        created_at=datetime.fromisoformat(row.created_at),
        updated_at=datetime.fromisoformat(row.updated_at),
    )


def _clamp_priority(value: int | None) -> int:
    """Clamp a caller-supplied priority into the 1..5 scale."""
    if value is None:
        return 3
    return max(MIN_PRIORITY, min(MAX_PRIORITY, value))


class SQLAlchemyTaskRepository:
    """Task CRUD backed by SQLAlchemy Core + async SQLite.

    Args:
        session_factory: Async session maker bound to the application engine.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create(
        self,
        user_id: int,
        title: str,
        *,
        description: str = "",
        priority: int | None = None,
        status: TaskStatus = "pending",
        deadline_utc: str | None = None,
        estimated_minutes: int | None = None,
    ) -> Task:
        """Create a task and return the persisted row."""
        now = datetime.now(UTC)
        task_id = new_id()
        values: dict[str, Any] = {
            "id": task_id,
            "user_id": user_id,
            "title": title,
            "description": description,
            "priority": _clamp_priority(priority),
            "status": status,
            "deadline_utc": deadline_utc,
            "estimated_minutes": estimated_minutes,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        async with self._session_factory() as session, session.begin():
            await session.execute(sa.insert(tasks_table).values(**values))
        created = await self.get(task_id)
        if created is None:
            raise RuntimeError(f"Task {task_id} vanished after insert")
        return created

    async def get(self, task_id: str) -> Task | None:
        """Return one task by id, or None."""
        stmt = sa.select(tasks_table).where(tasks_table.c.id == task_id)
        async with self._session_factory() as session:
            row = (await session.execute(stmt)).first()
        return _row_to_task(row) if row is not None else None

    async def list_tasks(
        self,
        user_id: int,
        *,
        status: TaskStatus | None = None,
    ) -> list[Task]:
        """Return the user's tasks, optionally filtered by status.

        Ordering: done tasks sink to the end; within a status, deadline
        proximity (nulls last) then priority then creation time.
        """
        stmt = sa.select(tasks_table).where(tasks_table.c.user_id == user_id)
        if status is not None:
            stmt = stmt.where(tasks_table.c.status == status)
        stmt = stmt.order_by(
            tasks_table.c.status == "done",
            sa.case((tasks_table.c.deadline_utc.is_(None), 1), else_=0),
            tasks_table.c.deadline_utc,
            tasks_table.c.priority.desc(),
            tasks_table.c.created_at,
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        return [_row_to_task(row) for row in rows]

    async def pending_by_priority(self, user_id: int, limit: int = 3) -> list[Task]:
        """Return open (pending/in_progress) tasks ranked for intervention.

        Deadline-bearing tasks rank first (soonest deadline first), then
        higher priority, then earliest creation.  This is the ranking the
        ``smart_prioritization`` intervention feeds to the user.
        """
        stmt = (
            sa.select(tasks_table)
            .where(
                tasks_table.c.user_id == user_id,
                tasks_table.c.status.in_(["pending", "in_progress"]),
            )
            .order_by(
                sa.case((tasks_table.c.deadline_utc.is_(None), 1), else_=0),
                tasks_table.c.deadline_utc,
                tasks_table.c.priority.desc(),
                tasks_table.c.created_at,
            )
            .limit(max(1, limit))
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
        return [_row_to_task(row) for row in rows]

    async def update(self, task_id: str, **fields: Any) -> Task | None:
        """Update allowed task fields; return the refreshed task or None."""
        allowed = {
            "title",
            "description",
            "priority",
            "status",
            "deadline_utc",
            "estimated_minutes",
        }
        changes = {k: v for k, v in fields.items() if k in allowed}
        if not changes:
            return await self.get(task_id)
        if "priority" in changes:
            changes["priority"] = _clamp_priority(changes["priority"])
        changes["updated_at"] = datetime.now(UTC).isoformat()
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.update(tasks_table)
                .where(tasks_table.c.id == task_id)
                .values(**changes)
            )
            if cast(Any, result).rowcount == 0:
                return None
        return await self.get(task_id)

    async def delete(self, task_id: str) -> bool:
        """Delete a task; return True when a row was removed."""
        async with self._session_factory() as session, session.begin():
            result = await session.execute(
                sa.delete(tasks_table).where(tasks_table.c.id == task_id)
            )
            return bool(cast(Any, result).rowcount)
