"""Tests for /api/v1/activities endpoints.

Covers:
  - GET /activities: paginated list, date filtering
  - GET /activities/current: latest activity
  - Success paths, error paths, edge cases (empty result)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.activities import router as activities_router
from mindflow.domain.events import make_event
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)


@pytest.fixture
async def app(engine, session_factory) -> FastAPI:
    """Create a test app with DB-backed activity repository and sample data."""
    # Create tables first
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(activities_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.engine = engine
    app.state.migration_applied = True

    repo = SQLAlchemyActivityRepository(session_factory=session_factory)

    # Insert test events
    now = datetime.now(UTC)
    for i in range(5):
        event = make_event(
            user_id=1,
            timestamp_utc=now - timedelta(minutes=i * 10),
            duration_s=5.0,
            app_name=f"TestApp{i}",
            window_title=f"Window {i}",
        )
        await repo.append_event(event)

    app.state.activity_repository = repo
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class TestActivitiesRoutes:
    """Verify activities endpoints."""

    def test_list_activities_default(self, client):
        """GET /activities should return paginated results."""
        resp = client.get("/api/v1/activities")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert len(data["items"]) > 0
        assert data["page"] == 1
        assert data["page_size"] == 50

    def test_list_activities_pagination(self, client):
        """Page and page_size parameters should work."""
        resp = client.get("/api/v1/activities?page=1&page_size=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) <= 2
        assert data["page"] == 1
        assert data["page_size"] == 2

    def test_list_activities_date_filter(self, client):
        """Date filtering should return matching events."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        resp = client.get(
            f"/api/v1/activities?start_date={today}&end_date={today}"
        )
        assert resp.status_code == 200

    def test_list_activities_invalid_date(self, client):
        """Invalid date format should return 422."""
        resp = client.get("/api/v1/activities?start_date=not-a-date")
        assert resp.status_code == 422
        data = resp.json()
        assert data["type"] == "https://mindflow.app/errors/validation-error"

    def test_list_activities_start_after_end(self, client):
        """start_date after end_date should return 422."""
        resp = client.get(
            "/api/v1/activities?start_date=2025-01-01&end_date=2024-01-01"
        )
        assert resp.status_code == 422

    def test_get_current_activity(self, client):
        """GET /activities/current should return the latest event."""
        resp = client.get("/api/v1/activities/current")
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "data" in data
        assert "app_name" in data["data"]

    def test_get_current_activity_format(self, client):
        """Current activity response should have correct structure."""
        resp = client.get("/api/v1/activities/current")
        data = resp.json()
        assert "timestamp" in data
        assert "event_type" in data
        assert "duration_s" in data
        assert "data" in data


class TestActivitiesEmptyRepo:
    """Verify edge cases with empty repository."""

    @pytest.fixture
    async def empty_app(self, engine, session_factory) -> FastAPI:
        """Create an app with empty activity repository."""
        async with engine.begin() as conn:
            await conn.run_sync(activity_events.metadata.create_all)

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(activities_router, prefix="/api/v1")
        app.state.collector_service = None
        app.state.engine = engine
        app.state.migration_applied = True

        repo = SQLAlchemyActivityRepository(session_factory=session_factory)
        app.state.activity_repository = repo
        return app

    def test_empty_list(self, empty_app):
        """GET /activities should return empty list when no events."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/activities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_current_not_found(self, empty_app):
        """GET /activities/current should return 404 when no events."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/activities/current")
        assert resp.status_code == 404
        data = resp.json()
        assert data["type"] == "https://mindflow.app/errors/not-found"
