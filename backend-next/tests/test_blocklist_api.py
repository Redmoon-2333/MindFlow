"""Tests for the blocklist repository, API, and browser polling endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.blocklist import router as blocklist_router
from mindflow.api.routes.telemetry import router as telemetry_router
from mindflow.infrastructure.repositories.blocklist import SQLAlchemyBlocklistRepository
from mindflow.infrastructure.schema import metadata as schema_metadata


@pytest.fixture
async def repo(engine, session_factory) -> SQLAlchemyBlocklistRepository:
    async with engine.begin() as conn:
        await conn.run_sync(schema_metadata.create_all)
    return SQLAlchemyBlocklistRepository(session_factory=session_factory)


class TestBlocklistRepository:
    """Repository behaviour for the environment_optimization execution path."""

    async def test_ensure_blocked_inserts_then_reenables(self, repo) -> None:
        await repo.ensure_blocked(user_id=1, domain="youtube.com", reason="干预")
        assert await repo.list_enabled(1) == ["youtube.com"]

        # Disable, then ensure_blocked again — the same row re-enables.
        assert await repo.set_enabled(1, "youtube.com", False) is True
        assert await repo.list_enabled(1) == []

        await repo.ensure_blocked(user_id=1, domain="youtube.com")
        assert await repo.list_enabled(1) == ["youtube.com"]

    async def test_list_enabled_filters_disabled(self, repo) -> None:
        await repo.ensure_blocked(user_id=1, domain="a.com")
        await repo.ensure_blocked(user_id=1, domain="b.com")
        await repo.set_enabled(1, "a.com", False)
        assert await repo.list_enabled(1) == ["b.com"]

    async def test_list_all_includes_disabled(self, repo) -> None:
        await repo.ensure_blocked(user_id=1, domain="a.com")
        await repo.set_enabled(1, "a.com", False)
        rows = await repo.list_all(1)
        assert len(rows) == 1
        assert rows[0]["enabled"] is False
        assert rows[0]["domain"] == "a.com"

    async def test_remove(self, repo) -> None:
        await repo.ensure_blocked(user_id=1, domain="x.com")
        assert await repo.remove(1, "x.com") is True
        assert await repo.remove(1, "x.com") is False

    async def test_normalizes_pasted_urls(self, repo) -> None:
        await repo.ensure_blocked(
            user_id=1, domain="https://Example.COM:443/some/path"
        )
        assert await repo.list_enabled(1) == ["example.com"]

    async def test_ignores_empty_domain(self, repo) -> None:
        assert await repo.ensure_blocked(user_id=1, domain="   ") is False
        assert await repo.list_enabled(1) == []


@pytest.fixture
async def app(repo) -> FastAPI:
    fastapi_app = FastAPI()
    register_exception_handlers(fastapi_app)
    fastapi_app.include_router(blocklist_router, prefix="/api/v1")
    fastapi_app.state.blocklist_repository = repo
    return fastapi_app


class TestBlocklistApi:
    """User-facing management endpoints."""

    async def test_add_list_toggle_remove(self, app) -> None:
        client = TestClient(app)
        added = client.post(
            "/api/v1/interventions/blocklist",
            json={"domain": "reddit.com", "reason": "手动添加"},
        )
        assert added.status_code == 200

        listed = client.get("/api/v1/interventions/blocklist").json()
        assert listed["count"] == 1
        assert listed["items"][0]["domain"] == "reddit.com"
        assert listed["items"][0]["reason"] == "手动添加"

        toggled = client.patch(
            "/api/v1/interventions/blocklist/reddit.com", json={"enabled": False}
        )
        assert toggled.status_code == 200
        assert client.get("/api/v1/interventions/blocklist").json()["items"][0][
            "enabled"
        ] is False

        removed = client.delete("/api/v1/interventions/blocklist/reddit.com")
        assert removed.status_code == 200
        assert client.get("/api/v1/interventions/blocklist").json()["count"] == 0

    async def test_toggle_missing_returns_404(self, app) -> None:
        client = TestClient(app)
        resp = client.patch(
            "/api/v1/interventions/blocklist/nope.com", json={"enabled": True}
        )
        assert resp.status_code == 404


class TestBrowserBlocklistEndpoint:
    """Extension polling endpoint: browser-token authenticated."""

    @pytest.fixture
    async def browser_app(self, repo) -> FastAPI:
        fastapi_app = FastAPI()
        register_exception_handlers(fastapi_app)
        fastapi_app.include_router(telemetry_router, prefix="/api/v1")
        fastapi_app.state.blocklist_repository = repo

        telemetry_mock = AsyncMock()
        telemetry_mock.verify_browser_token = AsyncMock(return_value=True)
        fastapi_app.state.telemetry_service = telemetry_mock
        return fastapi_app

    async def test_returns_enabled_domains_with_valid_token(
        self, browser_app, repo
    ) -> None:
        await repo.ensure_blocked(user_id=1, domain="tiktok.com")
        await repo.ensure_blocked(user_id=1, domain="weibo.com")
        await repo.set_enabled(1, "tiktok.com", False)

        client = TestClient(browser_app)
        resp = client.get(
            "/api/v1/telemetry/browser/blocklist",
            headers={"X-Browser-Token": "valid"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"domains": ["weibo.com"], "version": 1}

    async def test_rejects_invalid_token(self, browser_app) -> None:
        browser_app.state.telemetry_service.verify_browser_token = AsyncMock(
            return_value=False
        )
        client = TestClient(browser_app)
        resp = client.get(
            "/api/v1/telemetry/browser/blocklist",
            headers={"X-Browser-Token": "wrong"},
        )
        assert resp.status_code == 401
