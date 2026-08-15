"""Task API routes (smart-prioritization data source).

Endpoints:
  - GET    /api/v1/tasks        — list tasks (optional ``status`` filter)
  - POST   /api/v1/tasks        — create a task
  - PATCH  /api/v1/tasks/{id}   — update task fields
  - DELETE /api/v1/tasks/{id}   — delete a task
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query  # noqa: B008

from mindflow.api.deps import get_task_repo
from mindflow.api.errors import _not_found
from mindflow.api.schemas import (
    TaskCommandResponse,
    TaskCreateRequest,
    TaskListResponse,
    TaskResponse,
    TaskStatusValue,
    TaskUpdateRequest,
)
from mindflow.domain.tasks import Task
from mindflow.infrastructure.repositories.tasks import SQLAlchemyTaskRepository

router = APIRouter(tags=["tasks"])


def _task_to_response(task: Task) -> TaskResponse:
    """Map the frozen domain Task to its API response shape."""
    return TaskResponse(
        id=task.id,
        title=task.title,
        description=task.description,
        priority=task.priority,
        status=task.status,
        deadline_utc=task.deadline_utc.isoformat() if task.deadline_utc else None,
        estimated_minutes=task.estimated_minutes,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
    )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    status: TaskStatusValue | None = Query(  # noqa: B008
        default=None, description="Filter by status"
    ),
    repo: SQLAlchemyTaskRepository = Depends(get_task_repo),  # noqa: B008
) -> TaskListResponse:
    """List the user's tasks, optionally filtered by status."""
    tasks = await repo.list_tasks(user_id=1, status=status)
    return TaskListResponse(
        items=[_task_to_response(t) for t in tasks],
        count=len(tasks),
    )


@router.post("/tasks", response_model=TaskResponse, status_code=201)
async def create_task(
    body: TaskCreateRequest,
    repo: SQLAlchemyTaskRepository = Depends(get_task_repo),  # noqa: B008
) -> TaskResponse:
    """Create a task."""
    task = await repo.create(
        user_id=1,
        title=body.title,
        description=body.description,
        priority=body.priority,
        status=body.status,
        deadline_utc=body.deadline_utc,
        estimated_minutes=body.estimated_minutes,
    )
    return _task_to_response(task)


@router.patch("/tasks/{task_id}", response_model=TaskResponse)
async def update_task(
    body: TaskUpdateRequest,
    task_id: str = Path(..., description="Task UUID"),  # noqa: B008
    repo: SQLAlchemyTaskRepository = Depends(get_task_repo),  # noqa: B008
) -> TaskResponse:
    """Update one task; omitted fields keep their stored values."""
    fields = body.model_dump(exclude_none=True)
    task = await repo.update(task_id, **fields)
    if task is None:
        raise _not_found(f"任务 {task_id}")
    return _task_to_response(task)


@router.delete("/tasks/{task_id}", response_model=TaskCommandResponse)
async def delete_task(
    task_id: str = Path(..., description="Task UUID"),  # noqa: B008
    repo: SQLAlchemyTaskRepository = Depends(get_task_repo),  # noqa: B008
) -> TaskCommandResponse:
    """Delete one task."""
    deleted = await repo.delete(task_id)
    if not deleted:
        raise _not_found(f"任务 {task_id}")
    return TaskCommandResponse(task_id=task_id)
