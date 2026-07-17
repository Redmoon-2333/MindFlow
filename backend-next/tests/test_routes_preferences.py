"""Tests for /api/v1/preferences endpoints.

Covers:
  - GET /preferences: read (default empty, stored values)
  - PUT /preferences: full replace
  - PATCH /preferences: partial merge
  - Edge cases: empty body, complex nested JSON
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.preferences import router as preferences_router
from mindflow.infrastructure.repositories.preferences import (
    PreferencesRepository,
    user_preferences,
)


@pytest.fixture
async def app(engine, session_factory) -> FastAPI:
    """Create a test app with preferences table created."""
    async with engine.begin() as conn:
        await conn.run_sync(user_preferences.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(preferences_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.engine = engine
    app.state.migration_applied = True

    repo = PreferencesRepository(session_factory=session_factory)
    app.state.preferences_repository = repo
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestPreferencesRoutes:
    """Verify preferences endpoints."""

    def test_get_default_empty(self, client):
        """GET /preferences should return empty dict by default."""
        resp = client.get("/api/v1/preferences")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {}

    def test_put_and_get(self, client):
        """PUT then GET should return the same data."""
        payload = {"theme": "dark", "language": "zh-CN"}
        put_resp = client.put("/api/v1/preferences", json=payload)
        assert put_resp.status_code == 200
        assert put_resp.json() == payload

        get_resp = client.get("/api/v1/preferences")
        assert get_resp.json() == payload

    def test_put_replace(self, client):
        """PUT should replace all existing preferences."""
        client.put("/api/v1/preferences", json={"old_key": "old_value"})
        client.put("/api/v1/preferences", json={"new_key": "new_value"})

        resp = client.get("/api/v1/preferences")
        assert resp.json() == {"new_key": "new_value"}

    def test_patch_merge(self, client):
        """PATCH should merge into existing preferences."""
        client.put("/api/v1/preferences", json={"key1": "v1", "key2": "v2"})
        patch_resp = client.patch("/api/v1/preferences", json={"key2": "updated"})
        assert patch_resp.status_code == 200

        get_resp = client.get("/api/v1/preferences")
        assert get_resp.json() == {"key1": "v1", "key2": "updated"}

    def test_patch_remove_key(self, client):
        """PATCH with null value should remove the key."""
        client.put("/api/v1/preferences", json={"key1": "v1", "key2": "v2"})
        patch_resp = client.patch("/api/v1/preferences", json={"key1": None})
        assert patch_resp.status_code == 200

        get_resp = client.get("/api/v1/preferences")
        assert "key1" not in get_resp.json()
        assert get_resp.json() == {"key2": "v2"}

    def test_complex_nested_preferences(self, client):
        """Preferences should support complex nested structures."""
        payload = {
            "display": {
                "theme": "dark",
                "font_size": 14,
            },
            "notifications": {
                "enabled": True,
                "quiet_hours": ["22:00", "07:00"],
            },
        }
        client.put("/api/v1/preferences", json=payload)
        resp = client.get("/api/v1/preferences")
        assert resp.json() == payload
