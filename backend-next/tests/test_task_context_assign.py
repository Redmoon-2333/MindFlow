"""Appendix A: unrecognised apps/sites can be assigned a permanent context.

The period review can only report "未识别" for time the classifier cannot
place. These endpoints close the loop: list what is unknown (local-only, never
sent to an LLM), then remember the user's answer as a rule the rollup reads on
every future build.

The tests use the real classifier and the real repositories so "unknown" means
exactly what the rollup means by it; only the recompute target is a stub, since
its job here is to prove it was started, not to rebuild anything.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.app_classification import router as app_classification_router
from mindflow.api.routes.task_context_assign import router as task_context_router
from mindflow.domain.events import make_event
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.app_classification import (
    AppClassificationRulesRepository,
)
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.task_context import TaskContextRulesRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.services.telemetry_service import TelemetryService

_BASE = datetime(2026, 11, 10, 9, 0, tzinfo=UTC)


class _StubTelemetryService:
    """Records rebuild calls instead of rebuilding (the point is the schedule)."""

    def __init__(self) -> None:
        self.calls: list[tuple[datetime, datetime, int]] = []

    async def rebuild_feature_windows(
        self,
        start: datetime,
        end: datetime,
        user_id: int = 1,
        *,
        apply: bool = False,
    ) -> Any:
        self.calls.append((start, end, user_id))
        return None


class _Repos:
    """The four stores the listing and the assignment both touch."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory
        self.activity = SQLAlchemyActivityRepository(session_factory=session_factory)
        self.telemetry = TelemetryRepository(session_factory=session_factory)
        self.classification = AppClassificationRulesRepository(session_factory=session_factory)
        self.task_context = TaskContextRulesRepository(session_factory=session_factory)


@pytest.fixture
async def repos(engine, session_factory, create_tables) -> AsyncIterator[_Repos]:
    yield _Repos(session_factory)


def _service(repos: _Repos, tmp_path: Any) -> TelemetryService:
    return TelemetryService(
        repository=repos.telemetry,
        preferences_repository=PreferencesRepository(
            session_factory=repos.session_factory
        ),
        data_dir=tmp_path,
        activity_repository=repos.activity,
        classification_rules_repository=repos.classification,
        task_context_rules_repository=repos.task_context,
    )


async def _event(
    repos: _Repos,
    process: str,
    *,
    offset_s: float,
    duration_s: float,
    idle: bool = False,
) -> None:
    await repos.activity.append_event(
        make_event(
            user_id=1,
            timestamp_utc=_BASE + timedelta(seconds=offset_s),
            duration_s=duration_s,
            process_name=process,
            is_idle=idle,
        )
    )


async def _segment(
    repos: _Repos, *, domain: str, browser_name: str, duration_s: float
) -> None:
    await repos.telemetry.save_browser_heartbeat(
        user_id=1,
        timestamp_utc=_BASE,
        duration_s=duration_s,
        browser_name=browser_name,
        domain=domain,
        audible=False,
        context_key="",
    )


class TestListUnclassified:
    async def test_unknown_apps_are_listed_by_minutes(
        self, repos: _Repos, tmp_path: Any
    ) -> None:
        # A long unknown app, a short one, and an idle stretch that must not
        # count: idle time is not "using the app".
        await _event(repos, "MysteryTool.exe", offset_s=0, duration_s=600)
        await _event(repos, "OtherTool.exe", offset_s=600, duration_s=180)
        await _event(repos, "MysteryTool.exe", offset_s=780, duration_s=1200, idle=True)
        service = _service(repos, tmp_path)

        items = await service.list_unclassified(
            1, _BASE, _BASE + timedelta(hours=1), limit=10
        )

        assert [(item.kind, item.display, item.minutes) for item in items] == [
            ("app", "MysteryTool", 10),
            ("app", "OtherTool", 3),
        ]
        assert items[0].key == "MysteryTool.exe"

    async def test_placeholder_unknown_process_is_not_listed(
        self, repos: _Repos, tmp_path: Any
    ) -> None:
        # Collectors write "unknown" when no process name was readable.
        await _event(repos, "unknown", offset_s=0, duration_s=600)
        await _event(repos, "MysteryTool.exe", offset_s=600, duration_s=120)
        service = _service(repos, tmp_path)

        items = await service.list_unclassified(
            1, _BASE, _BASE + timedelta(hours=1), limit=10
        )

        assert [item.display for item in items] == ["MysteryTool"]

    async def test_browser_processes_are_excluded_but_sites_are_listed(
        self, repos: _Repos, tmp_path: Any
    ) -> None:
        """A browser is classified by its site, so the process itself is no question.

        The browser here is one the built-in classifier does not know, which is
        the case where its sites are genuinely unrecognised: a *recognised*
        browser (chrome/edge/firefox/safari) gives every domain it shows the
        browser's own context, so nothing about the site is unknown — that is
        the rollup's resolution chain, and the listing reports the same view.
        """
        await _event(repos, "mysterybrowser.exe", offset_s=0, duration_s=300)
        await _segment(
            repos,
            domain="unknown-place.example",
            browser_name="mysterybrowser",
            duration_s=240,
        )
        service = _service(repos, tmp_path)

        items = await service.list_unclassified(
            1, _BASE, _BASE + timedelta(hours=1), limit=10
        )

        assert [(item.kind, item.display) for item in items] == [
            ("site", "unknown-place.example")
        ]
        assert items[0].key == "unknown-place.example"
        assert items[0].minutes == 4

    async def test_classified_apps_and_sites_are_not_listed(
        self, repos: _Repos, tmp_path: Any
    ) -> None:
        await _event(repos, "code.exe", offset_s=0, duration_s=300)
        await _segment(
            repos, domain="docs.python.org", browser_name="chrome", duration_s=300
        )
        service = _service(repos, tmp_path)

        assert await service.list_unclassified(
            1, _BASE, _BASE + timedelta(hours=1), limit=10
        ) == []

    async def test_an_existing_rule_removes_the_item(
        self, repos: _Repos, tmp_path: Any
    ) -> None:
        await _event(repos, "MysteryTool.exe", offset_s=0, duration_s=600)
        await repos.classification.add(
            user_id=1,
            rule={
                "process_name": "MysteryTool.exe",
                "window_title_pattern": None,
                "category": "code",
                "priority": 50,
            },
        )
        service = _service(repos, tmp_path)

        assert await service.list_unclassified(
            1, _BASE, _BASE + timedelta(hours=1), limit=10
        ) == []

    async def test_limit_and_sub_minute_items(
        self, repos: _Repos, tmp_path: Any
    ) -> None:
        await _event(repos, "A.exe", offset_s=0, duration_s=600)
        await _event(repos, "B.exe", offset_s=600, duration_s=300)
        await _event(repos, "C.exe", offset_s=900, duration_s=20)  # rounds to 0
        service = _service(repos, tmp_path)

        items = await service.list_unclassified(
            1, _BASE, _BASE + timedelta(hours=1), limit=1
        )

        assert [(item.display, item.minutes) for item in items] == [("A", 10)]


@pytest.fixture
def app(repos: _Repos, tmp_path: Any) -> FastAPI:
    application = FastAPI()
    register_exception_handlers(application)
    application.include_router(task_context_router, prefix="/api/v1")
    application.include_router(app_classification_router, prefix="/api/v1")
    application.state.settings = type("S", (), {"timezone": "UTC"})()
    application.state.telemetry_service = _service(repos, tmp_path)
    application.state.classification_rules_repository = repos.classification
    application.state.task_context_rules_repository = repos.task_context
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client
    # An assignment schedules a background rebuild; a test that does not await
    # it would leave the task touching the database after its event loop is
    # gone ("Event loop is closed" in the aiosqlite worker). Drain it here.
    for task in list(getattr(app.state, "task_context_recompute_tasks", ())):
        await asyncio.gather(task, return_exceptions=True)


class TestRoutes:
    async def test_unclassified_route_returns_items(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/task-context/unclassified?days=7")

        assert response.status_code == 200
        assert response.json() == {"items": []}

    async def test_unclassified_route_rejects_unsupported_span(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/api/v1/task-context/unclassified?days=5")

        assert response.status_code == 422

    async def test_app_assignment_writes_a_process_rule_and_replaces_the_old_one(
        self, client: AsyncClient
    ) -> None:
        first = await client.post(
            "/api/v1/task-context/assign",
            json={"kind": "app", "key": "MysteryTool.exe", "context": "coding"},
        )
        assert first.status_code == 201
        assert first.json() == {"saved": True, "recompute": "started"}

        # The same app, different case, a different answer: the old rule goes.
        second = await client.post(
            "/api/v1/task-context/assign",
            json={"kind": "app", "key": "mysterytool.EXE", "context": "entertainment"},
        )
        assert second.status_code == 201

        rules = (await client.get("/api/v1/app-classifications")).json()
        assert len(rules) == 1
        assert rules[0]["process_name"] == "mysterytool.EXE"
        assert rules[0]["category"] == "entertainment"
        assert rules[0]["window_title_pattern"] is None
        assert rules[0]["priority"] == 50

    async def test_app_assignment_leaves_title_patterned_rules_alone(
        self, client: AsyncClient, repos: _Repos
    ) -> None:
        await repos.classification.add(
            user_id=1,
            rule={
                "process_name": "MysteryTool.exe",
                "window_title_pattern": "%lecture%",
                "category": "browser_work",
                "priority": 80,
            },
        )

        response = await client.post(
            "/api/v1/task-context/assign",
            json={"kind": "app", "key": "MysteryTool.exe", "context": "coding"},
        )

        assert response.status_code == 201
        rules = (await client.get("/api/v1/app-classifications")).json()
        # Both survive: the narrow title rule and the new app-wide one.
        assert {(r["window_title_pattern"], r["category"]) for r in rules} == {
            ("%lecture%", "browser_work"),
            (None, "code"),
        }

    async def test_site_assignment_normalises_the_host_and_replaces_the_old_rule(
        self, client: AsyncClient, repos: _Repos
    ) -> None:
        first = await client.post(
            "/api/v1/task-context/assign",
            json={
                "kind": "site",
                "key": "https://News.Example.com/some/private/path?q=1",
                "context": "reading",
            },
        )
        assert first.status_code == 201

        second = await client.post(
            "/api/v1/task-context/assign",
            json={"kind": "site", "key": "news.example.com", "context": "entertainment"},
        )
        assert second.status_code == 201

        rules = await repos.task_context.get_all(1)
        assert len(rules) == 1
        # Only the host is stored — a pasted URL cannot smuggle a path in.
        assert rules[0]["match_value"] == "news.example.com"
        assert rules[0]["task_context"] == "entertainment"
        assert rules[0]["priority"] == 50

    async def test_site_assignment_rejects_an_invalid_domain(
        self, client: AsyncClient
    ) -> None:
        response = await client.post(
            "/api/v1/task-context/assign",
            json={"kind": "site", "key": "https://", "context": "reading"},
        )

        assert response.status_code == 422

    async def test_unknown_context_is_rejected(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/task-context/assign",
            json={"kind": "app", "key": "MysteryTool.exe", "context": "unknown"},
        )

        assert response.status_code == 422

    async def test_assignment_starts_a_background_recompute(
        self, app: FastAPI, client: AsyncClient
    ) -> None:
        stub = _StubTelemetryService()
        app.state.telemetry_service = stub

        response = await client.post(
            "/api/v1/task-context/assign",
            json={"kind": "app", "key": "MysteryTool.exe", "context": "coding"},
        )

        assert response.status_code == 201
        tasks = app.state.task_context_recompute_tasks
        assert len(tasks) == 1
        # Let the scheduled rebuild run before the loop closes.
        await next(iter(tasks))
        # One bounded call per *local business day*, fourteen days back.
        assert len(stub.calls) == 14
        assert all(user_id == 1 for _, _, user_id in stub.calls)
        assert stub.calls[0][0] == stub.calls[-1][0] - timedelta(days=13)
        # Each block is one business day: midnight to just before midnight.
        assert all(start.hour == 0 and start.minute == 0 for start, _, _ in stub.calls)
        assert all(
            end.date() == start.date() and end.hour == 23 for start, end, _ in stub.calls
        )
        assert not app.state.task_context_recompute_tasks
