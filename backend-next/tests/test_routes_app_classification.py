"""Tests for /api/v1/app-classifications endpoints.

Covers:
  - GET  /app-classifications: list rules (empty, with data)
  - POST /app-classifications: add rule (happy path, invalid category)
  - PUT  /app-classifications: replace all rules
  - DELETE /app-classifications/{id}: remove rule (existing, non-existent)
  - GET  /app-classifications/unknown-apps: list unclassified apps
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.app_classification import router as app_class_router
from mindflow.infrastructure.repositories.activity import activity_events
from mindflow.infrastructure.repositories.app_classification import (
    AppClassificationRulesRepository,
    app_classification_rules,
)


@pytest.fixture
async def app(engine, session_factory) -> FastAPI:
    """Create a test FastAPI app with required tables created and repo injected."""
    async with engine.begin() as conn:
        await conn.run_sync(app_classification_rules.metadata.create_all)
        await conn.run_sync(activity_events.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(app_class_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.engine = engine
    app.state.migration_applied = True

    repo = AppClassificationRulesRepository(session_factory=session_factory)
    app.state.classification_rules_repository = repo
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestAppClassificationRoutes:
    """Verify app-classification endpoints."""

    def test_get_empty_list(self, client):
        """GET /app-classifications returns empty list when no rules exist."""
        resp = client.get("/api/v1/app-classifications")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_add_rule_returns_201(self, client):
        """POST /app-classifications with valid body returns 201 with the new rule."""
        payload = {
            "process_name": "bilibili.exe",
            "category": "browser_work",
            "priority": 10,
        }
        resp = client.post("/api/v1/app-classifications", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["process_name"] == "bilibili.exe"
        assert data["category"] == "browser_work"
        assert data["priority"] == 10
        assert data["id"] is not None

    def test_add_rule_invalid_category_422(self, client):
        """POST with invalid category returns 422 (Pydantic validation error)."""
        payload = {
            "process_name": "bilibili.exe",
            "category": "invalid_category",
        }
        resp = client.post("/api/v1/app-classifications", json=payload)
        assert resp.status_code == 422

    def test_delete_rule_returns_204(self, client):
        """DELETE an existing rule returns 204."""
        add_resp = client.post(
            "/api/v1/app-classifications",
            json={"process_name": "foo.exe", "category": "other"},
        )
        rule_id = add_resp.json()["id"]

        del_resp = client.delete(f"/api/v1/app-classifications/{rule_id}")
        assert del_resp.status_code == 204

        # Verify it's gone
        get_resp = client.get("/api/v1/app-classifications")
        assert get_resp.json() == []

    def test_delete_nonexistent_returns_204(self, client):
        """DELETE with non-existent ID returns 204 (idempotent no-op)."""
        resp = client.delete("/api/v1/app-classifications/nonexistent-id")
        assert resp.status_code == 204

    def test_put_replaces_all(self, client):
        """PUT replaces all existing rules with the provided list."""
        # Add one rule first
        client.post(
            "/api/v1/app-classifications",
            json={"process_name": "old.exe", "category": "other"},
        )

        # Replace with new set
        payload = [
            {"process_name": "bilibili.exe", "category": "entertainment", "priority": 5},
            {"process_name": "code.exe", "category": "code", "priority": 1},
        ]
        put_resp = client.put("/api/v1/app-classifications", json=payload)
        assert put_resp.status_code == 200
        data = put_resp.json()
        assert len(data) == 2
        names = {r["process_name"] for r in data}
        assert names == {"bilibili.exe", "code.exe"}

        # Verify old rule is gone
        get_resp = client.get("/api/v1/app-classifications")
        assert len(get_resp.json()) == 2

    def test_get_after_add(self, client):
        """GET returns all added rules."""
        client.post(
            "/api/v1/app-classifications",
            json={"process_name": "a.exe", "category": "code", "priority": 1},
        )
        client.post(
            "/api/v1/app-classifications",
            json={"process_name": "b.exe", "category": "entertainment", "priority": 5},
        )

        resp = client.get("/api/v1/app-classifications")
        assert resp.status_code == 200
        rules = resp.json()
        assert len(rules) == 2
        # Higher priority first
        assert rules[0]["priority"] == 5
        assert rules[1]["priority"] == 1

    def test_unknown_apps_endpoint_returns_list(self, client):
        """GET /app-classifications/unknown-apps returns list (empty when no events)."""
        resp = client.get("/api/v1/app-classifications/unknown-apps")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_unknown_apps_endpoint_accepts(self, client):
        """unknown-apps endpoint is accessible and returns valid JSON."""
        resp = client.get("/api/v1/app-classifications/unknown-apps")
        assert resp.status_code == 200

    def test_add_rule_with_window_title_pattern(self, client):
        """POST with window_title_pattern stores it correctly."""
        payload = {
            "process_name": "notion.exe",
            "category": "document",
            "window_title_pattern": "%Meeting%",
            "priority": 3,
        }
        resp = client.post("/api/v1/app-classifications", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["window_title_pattern"] == "%Meeting%"
