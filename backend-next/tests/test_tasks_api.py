"""Tests for the task API and repository (smart-prioritization data source)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.tasks import router as tasks_router
from mindflow.infrastructure.repositories.tasks import SQLAlchemyTaskRepository
from mindflow.infrastructure.schema import metadata as schema_metadata


@pytest.fixture
async def app(engine, session_factory) -> FastAPI:
    """FastAPI app with the tasks router and a real task repository."""
    async with engine.begin() as conn:
        await conn.run_sync(schema_metadata.create_all)

    repo = SQLAlchemyTaskRepository(session_factory=session_factory)
    fastapi_app = FastAPI()
    register_exception_handlers(fastapi_app)
    fastapi_app.include_router(tasks_router, prefix="/api/v1")
    fastapi_app.state.task_repository = repo
    return fastapi_app


class TestTaskApi:
    """Task CRUD route tests."""

    async def test_create_and_get(self, app) -> None:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/tasks",
            json={
                "title": "写完实验报告",
                "description": "双创项目实验部分",
                "priority": 4,
                "deadline_utc": "2026-08-20T12:00:00+00:00",
                "estimated_minutes": 120,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "写完实验报告"
        assert data["priority"] == 4
        assert data["status"] == "pending"
        assert data["deadline_utc"] is not None

        listed = client.get("/api/v1/tasks").json()
        assert listed["count"] == 1
        assert listed["items"][0]["id"] == data["id"]

    async def test_list_status_filter(self, app) -> None:
        client = TestClient(app)
        client.post("/api/v1/tasks", json={"title": "任务A"})
        done_id = client.post("/api/v1/tasks", json={"title": "任务B"}).json()["id"]
        client.patch(f"/api/v1/tasks/{done_id}", json={"status": "done"})

        pending = client.get("/api/v1/tasks", params={"status": "pending"}).json()
        done = client.get("/api/v1/tasks", params={"status": "done"}).json()
        assert pending["count"] == 1
        assert pending["items"][0]["title"] == "任务A"
        assert done["count"] == 1
        assert done["items"][0]["title"] == "任务B"

    async def test_update_and_delete(self, app) -> None:
        client = TestClient(app)
        created = client.post("/api/v1/tasks", json={"title": "旧标题"}).json()

        updated = client.patch(
            f"/api/v1/tasks/{created['id']}", json={"title": "新标题", "priority": 5}
        )
        assert updated.status_code == 200
        assert updated.json()["title"] == "新标题"
        assert updated.json()["priority"] == 5

        deleted = client.delete(f"/api/v1/tasks/{created['id']}")
        assert deleted.status_code == 200

        missing = client.delete(f"/api/v1/tasks/{created['id']}")
        assert missing.status_code == 404

    async def test_update_missing_returns_404(self, app) -> None:
        client = TestClient(app)
        resp = client.patch("/api/v1/tasks/nonexistent", json={"title": "x"})
        assert resp.status_code == 404

    async def test_create_rejects_out_of_range_priority(self, app) -> None:
        """The Pydantic boundary rejects out-of-range priority (422)."""
        client = TestClient(app)
        resp = client.post("/api/v1/tasks", json={"title": "高优先级", "priority": 99})
        assert resp.status_code == 422


class TestTaskRepository:
    """Repository ranking behaviour (the intervention's data contract)."""

    @pytest.fixture
    async def repo(self, engine, session_factory) -> SQLAlchemyTaskRepository:
        async with engine.begin() as conn:
            await conn.run_sync(schema_metadata.create_all)
        return SQLAlchemyTaskRepository(session_factory=session_factory)

    async def test_pending_ranking_deadline_then_priority(self, repo) -> None:
        """Deadline-bearing tasks first (soonest first), then priority."""
        await repo.create(user_id=1, title="无截止日期", priority=5)
        await repo.create(
            user_id=1,
            title="后截止",
            priority=5,
            deadline_utc="2026-08-30T00:00:00+00:00",
        )
        await repo.create(
            user_id=1,
            title="先截止",
            priority=1,
            deadline_utc="2026-08-20T00:00:00+00:00",
        )
        await repo.create(user_id=1, title="已完成", status="done")

        ranked = await repo.pending_by_priority(user_id=1, limit=3)
        titles = [task.title for task in ranked]
        assert titles == ["先截止", "后截止", "无截止日期"]

    async def test_update_unknown_returns_none(self, repo) -> None:
        assert await repo.update("missing", title="x") is None

    async def test_create_clamps_priority(self, repo) -> None:
        """Repository-level defence: out-of-range priorities clamp to 1..5."""
        high = await repo.create(user_id=1, title="高", priority=99)
        low = await repo.create(user_id=1, title="低", priority=-3)
        assert high.priority == 5
        assert low.priority == 1

    async def test_delete_reports_missing(self, repo) -> None:
        assert await repo.delete("missing") is False

    async def test_created_at_iso_format(self, repo) -> None:
        task = await repo.create(user_id=1, title="时间格式")
        datetime.fromisoformat(task.created_at.isoformat())
        assert task.created_at.tzinfo is not None


class TestRowMapping:
    """Deadline parsing edge cases."""

    def test_deadline_parse_utc_suffix(self) -> None:
        from mindflow.infrastructure.repositories.tasks import _parse_deadline

        parsed = _parse_deadline("2026-08-20T12:00:00+00:00")
        assert parsed == datetime(2026, 8, 20, 12, 0, tzinfo=UTC)

    def test_deadline_parse_naive_gets_utc(self) -> None:
        from mindflow.infrastructure.repositories.tasks import _parse_deadline

        parsed = _parse_deadline("2026-08-20T12:00:00")
        assert parsed is not None and parsed.tzinfo == UTC

    def test_deadline_parse_empty_is_none(self) -> None:
        from mindflow.infrastructure.repositories.tasks import _parse_deadline

        assert _parse_deadline("") is None
        assert _parse_deadline(None) is None
