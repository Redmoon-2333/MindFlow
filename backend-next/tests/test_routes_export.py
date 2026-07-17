"""Tests for /api/v1/export endpoint.

Covers (3 main paths):
  1. Happy path: export CSV with data
  2. Range > 90 days → 422
  3. Invalid format → 422
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.export import router as export_router


def _make_mock_deps(app: FastAPI) -> None:
    """Set up mock dependencies on app.state for the export route.

    The route uses Depends(get_activity_repo), Depends(get_focus_repo),
    and Depends(get_report_repo).  We override them at test time.
    """
    mock_activity_repo = AsyncMock()
    mock_activity_repo.query_range = AsyncMock(return_value=[])

    mock_focus_repo = AsyncMock()
    mock_focus_repo.query_range = AsyncMock(return_value=[])

    mock_report_repo = AsyncMock()
    mock_report_repo.query_range = AsyncMock(return_value=[])

    app.dependency_overrides = {}

    # We need to override the Depends() calls.  Since the route imports
    # deps functions directly, we patch over them via the app's
    # dependency_overrides mechanism.
    from mindflow.api import deps

    app.dependency_overrides[deps.get_activity_repo] = lambda: mock_activity_repo
    app.dependency_overrides[deps.get_focus_repo] = lambda: mock_focus_repo
    app.dependency_overrides[deps.get_report_repo] = lambda: mock_report_repo


class TestExportEndpoint:
    """GET /api/v1/export."""

    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(export_router, prefix="/api/v1")
        _make_mock_deps(app)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        return TestClient(app)

    # ── Happy path ────────────────────────────────────────────────

    def test_export_csv_returns_200(self, client: TestClient) -> None:
        """CSV export with default range should return 200."""
        resp = client.get("/api/v1/export?fmt=csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "attachment" in resp.headers.get("content-disposition", "")
        assert ".csv" in resp.headers.get("content-disposition", "")

    def test_export_json_returns_200(self, client: TestClient) -> None:
        """JSON export with default range should return 200."""
        resp = client.get("/api/v1/export?fmt=json")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert ".json" in resp.headers.get("content-disposition", "")

    def test_export_with_explicit_dates(self, client: TestClient) -> None:
        """Export with explicit start/end dates should return 200."""
        resp = client.get(
            "/api/v1/export?fmt=csv&start=2026-07-01T00:00:00&end=2026-07-17T23:59:59"
        )
        assert resp.status_code == 200

    # ── Range validation ─────────────────────────────────────────

    def test_export_range_over_90_days_returns_422(self, client: TestClient) -> None:
        """Date range > 90 days should return 422."""
        resp = client.get(
            "/api/v1/export?fmt=csv&start=2026-01-01T00:00:00&end=2026-07-17T23:59:59"
        )
        assert resp.status_code == 422

    def test_export_exactly_90_days_returns_200(self, client: TestClient) -> None:
        """Date range exactly 90 days should be allowed."""
        resp = client.get(
            "/api/v1/export?fmt=csv&start=2026-04-18T00:00:00&end=2026-07-17T23:59:59"
        )
        # With date objects, 90 days in the actual delta
        assert resp.status_code == 200

    def test_export_end_before_start_returns_422(self, client: TestClient) -> None:
        """End date before start date should return 422."""
        resp = client.get(
            "/api/v1/export?fmt=csv&start=2026-07-17T00:00:00&end=2026-07-01T00:00:00"
        )
        assert resp.status_code == 422

    # ── Format validation ─────────────────────────────────────────

    def test_export_invalid_format_returns_422(self, client: TestClient) -> None:
        """Invalid format string should return 422."""
        resp = client.get("/api/v1/export?fmt=xml")
        assert resp.status_code == 422

    def test_export_missing_format_defaults_to_csv(self, client: TestClient) -> None:
        """Missing format parameter should default to csv."""
        resp = client.get("/api/v1/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")

    # ── Invalid date format ───────────────────────────────────────

    def test_export_invalid_date_returns_422(self, client: TestClient) -> None:
        """Invalid date string should return 422."""
        resp = client.get("/api/v1/export?start=not-a-date")
        assert resp.status_code == 422
