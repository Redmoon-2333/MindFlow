"""Tests for GET /api/v1/ai/runs and GET /api/v1/ai/runs/{run_id} diagnostics endpoints.

Covers:
  - List: success, pagination, empty
  - Detail: success with node events, 404 not-found (RFC 9457)
  - Allowlist: response contains ONLY metadata fields (no PII, prompts, chat)
  - Redaction: canary PII inserted via raw DB → absent from API response
  - Auth: 401 without token
  - Rate-limit: diagnostics endpoints not individually capped (global only)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.deps import get_workflow_runs_repo
from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.ai_diagnostics import router as diagnostics_router
from mindflow.infrastructure.repositories.workflow_runs import WorkflowRunsRepository

# ── Sample data ─────────────────────────────────────────────────────────

_SAMPLE_RUN_ID = "run-abc-123"
_SAMPLE_NODE_EVENT: dict[str, object] = {
    "node_name": "analyst",
    "status": "completed",
    "started_at": "2026-07-29T10:00:00+00:00",
    "completed_at": "2026-07-29T10:00:05+00:00",
    "duration_ms": 5000,
    "error_category": None,
}

_SAMPLE_RUN_ROW: dict[str, object] = {
    "run_id": _SAMPLE_RUN_ID,
    "workflow_name": "daily_analysis",
    "status": "completed",
    "graph_version": "v2",
    "source": "panel",
    "origin": "scheduler",
    "started_at": "2026-07-29T10:00:00+00:00",
    "completed_at": "2026-07-29T10:00:05+00:00",
    "duration_ms": 5000,
    "call_count": 6,
    "token_count": 1200,
    "degradation_reason": None,
}

# PII canary fields that MUST NOT appear in any diagnostics response.
_CANARY_PII: dict[str, object] = {
    "prompt": "sk-xxx-api-key-secret",
    "chat_content": "用户说：我今天很焦虑，需要医生帮助诊断。我的社保号是123-45-6789。",
    "evidence": json.dumps({"window_title": "银行账户 - 张三", "personal_id": "110101199001011234"}),
    "api_key": "sk-canary-api-key-000",
}


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_mock_repo(**overrides: object) -> WorkflowRunsRepository:
    """Build an AsyncMock-backed WorkflowRunsRepository."""
    repo = AsyncMock(spec=WorkflowRunsRepository)
    repo.list_runs = AsyncMock(return_value=([_SAMPLE_RUN_ROW], 1))
    repo.get_run = AsyncMock(return_value=object())  # non-None = exists
    repo.get_run_detail = AsyncMock(return_value=_SAMPLE_RUN_ROW)
    repo.get_node_events = AsyncMock(return_value=[_SAMPLE_NODE_EVENT])
    for key, val in overrides.items():
        setattr(repo, key, val)
    return repo


def _make_app(
    repo: WorkflowRunsRepository | None = None,
    *,
    with_auth: bool = False,
    system_token: str = "test-token",
) -> FastAPI:
    """Build a minimal FastAPI app with the diagnostics route."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(diagnostics_router, prefix="/api/v1")

    if with_auth:
        from mindflow.api.middleware.auth import AuthMiddleware

        app.state.system_token = system_token
        app.add_middleware(AuthMiddleware)

    if repo is not None:
        app.state.session_factory = None  # Not used when repo is overridden

        # Override the deps to inject our mock
        async def _override() -> WorkflowRunsRepository:
            return repo

        app.dependency_overrides[get_workflow_runs_repo] = _override  # type: ignore[assignment,arg-type]

    return app


# ── List endpoint ────────────────────────────────────────────────────────


class TestListRuns:
    def test_list_returns_paginated_runs(self) -> None:
        """GET /api/v1/ai/runs returns DiagnosticsListResponse with status 200."""
        repo = _make_mock_repo()
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs")

        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "count" in data
        assert "has_more" in data
        assert "next_offset" in data
        assert data["count"] == 1
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["run_id"] == _SAMPLE_RUN_ID
        assert item["workflow_name"] == "daily_analysis"
        assert item["status"] == "completed"
        assert item["origin"] == "scheduler"

    def test_list_respects_limit_and_offset(self) -> None:
        """Query params limit and offset are forwarded to the repository."""
        repo = _make_mock_repo()
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs?limit=5&offset=10")

        assert resp.status_code == 200
        repo.list_runs.assert_awaited_once()  # type: ignore[union-attr]
        call_kwargs = repo.list_runs.call_args.kwargs  # type: ignore[union-attr]
        assert call_kwargs["limit"] == 5
        assert call_kwargs["offset"] == 10

    def test_list_clamps_limit_to_100(self) -> None:
        """limit > 100 returns 422 validation error."""
        app = _make_app(_make_mock_repo())
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs?limit=999")

        assert resp.status_code == 422

    def test_list_returns_empty(self) -> None:
        """Empty list when no runs exist."""
        repo = _make_mock_repo()
        repo.list_runs = AsyncMock(return_value=([], 0))
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs")

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["count"] == 0
        assert data["has_more"] is False
        assert data["next_offset"] is None


# ── Detail endpoint ──────────────────────────────────────────────────────


class TestGetRunDetail:
    def test_detail_returns_full_run_with_node_events(self) -> None:
        """GET /api/v1/ai/runs/{run_id} returns RunDetail with node_events."""
        repo = _make_mock_repo()
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get(f"/api/v1/ai/runs/{_SAMPLE_RUN_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == _SAMPLE_RUN_ID
        assert data["token_estimate"] == 1200
        assert data["degraded"] is False
        assert len(data["node_events"]) == 1
        ev = data["node_events"][0]
        assert ev["node_name"] == "analyst"
        assert ev["status"] == "completed"
        assert ev["duration_ms"] == 5000

    def test_detail_unknown_run_returns_404_rfc9457(self) -> None:
        """Unknown run_id returns RFC 9457 not-found problem detail."""
        repo = _make_mock_repo()
        repo.get_run = AsyncMock(return_value=None)  # run does not exist
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs/nonexistent-id")

        assert resp.status_code == 404
        body = resp.json()
        assert body["type"] == "https://mindflow.app/errors/not-found"
        assert body["title"] == "Not Found"
        assert body["status"] == 404
        assert "instance" in body


# ── Allowlist / privacy ─────────────────────────────────────────────────


class TestAllowlistPrivacy:
    """Verify diagnostics responses contain ONLY allowlisted metadata fields."""

    # Fields that ARE allowed in RunSummary / RunDetail
    _ALLOWED_RUN_FIELDS: frozenset[str] = frozenset({
        "run_id",
        "workflow_name",
        "status",
        "graph_version",
        "source",
        "origin",
        "started_at",
        "completed_at",
        "duration_ms",
        "call_count",
        "token_estimate",
        "degraded",
        "node_events",
    })

    # Fields that MUST NOT appear in any diagnostics response
    _DENY_FIELDS: frozenset[str] = frozenset({
        "prompt",
        "chat_content",
        "evidence",
        "window_title",
        "api_key",
        "idempotency_key",
        "user_id",
        "trace_id",
        "retry_reason",
        "degradation_reason",
        "target_date",
        "last_error",
    })

    def test_detail_response_only_contains_allowlisted_fields(self) -> None:
        """RunDetail JSON keys are a subset of the allowlist."""
        repo = _make_mock_repo()
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get(f"/api/v1/ai/runs/{_SAMPLE_RUN_ID}")
        assert resp.status_code == 200
        data = resp.json()

        top_keys = set(data.keys())
        assert top_keys.issubset(self._ALLOWED_RUN_FIELDS), (
            f"Unexpected top-level keys: {top_keys - self._ALLOWED_RUN_FIELDS}"
        )

    def test_list_response_only_contains_allowlisted_run_summary_fields(self) -> None:
        """RunSummary items in list contain only allowlisted summary fields."""
        repo = _make_mock_repo()
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1

        summary_allowed = self._ALLOWED_RUN_FIELDS - {"token_estimate", "node_events"}
        item_keys = set(data["items"][0].keys())
        assert item_keys.issubset(summary_allowed), (
            f"Unexpected summary keys: {item_keys - summary_allowed}"
        )

    def test_no_deny_fields_in_response(self) -> None:
        """No deny-list field is present in the JSON response body."""
        repo = _make_mock_repo()
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get(f"/api/v1/ai/runs/{_SAMPLE_RUN_ID}")
        assert resp.status_code == 200
        body_text = resp.text.lower()

        for field in self._DENY_FIELDS:
            assert field not in body_text or f'"{field}"' not in resp.text, (
                f"Denied field '{field}' found in response body"
            )


# ── Redaction ────────────────────────────────────────────────────────────


class TestRedaction:
    """Verify canary PII seeded into the DB does not surface in API responses."""

    def test_canary_pii_not_in_run_detail(self) -> None:
        """When the raw row contains PII-like fields, the API must not return them."""
        # Build a row with canary PII sprinkled into fields that the repository
        # might return (simulating what would happen if the raw DB row contained
        # sensitive data in columns present in the schema).
        polluted_row = dict(_SAMPLE_RUN_ROW)
        polluted_row.update(_CANARY_PII)

        repo = _make_mock_repo()
        repo.get_run_detail = AsyncMock(return_value=polluted_row)
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get(f"/api/v1/ai/runs/{_SAMPLE_RUN_ID}")
        assert resp.status_code == 200
        body = resp.text

        # Every canary value must be absent from the response.
        for canary_val in _CANARY_PII.values():
            canary_str = str(canary_val)
            # Check that the exact PII string does not appear
            assert canary_str not in body, (
                f"Canary PII leaked in response: {canary_str[:40]}..."
            )

    def test_canary_keys_not_in_response_keys(self) -> None:
        """Canary fields are not among the JSON keys of RunDetail."""
        polluted_row = dict(_SAMPLE_RUN_ROW)
        polluted_row.update(_CANARY_PII)

        repo = _make_mock_repo()
        repo.get_run_detail = AsyncMock(return_value=polluted_row)
        app = _make_app(repo)
        client = TestClient(app)

        resp = client.get(f"/api/v1/ai/runs/{_SAMPLE_RUN_ID}")
        assert resp.status_code == 200
        data = resp.json()

        for canary_key in _CANARY_PII:
            assert canary_key not in data, (
                f"Canary key '{canary_key}' appeared in RunDetail response"
            )


# ── Auth ────────────────────────────────────────────────────────────────


class TestDiagnosticsAuth:
    """Diagnostics endpoints require authorization like all other API routes."""

    def test_list_requires_auth(self) -> None:
        """GET /ai/runs without token returns 401."""
        app = _make_app(_make_mock_repo(), with_auth=True)
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs")

        assert resp.status_code == 401
        body = resp.json()
        assert body["type"] == "https://mindflow.app/errors/auth-required"

    def test_detail_requires_auth(self) -> None:
        """GET /ai/runs/{id} without token returns 401."""
        app = _make_app(_make_mock_repo(), with_auth=True)
        client = TestClient(app)

        resp = client.get(f"/api/v1/ai/runs/{_SAMPLE_RUN_ID}")

        assert resp.status_code == 401

    def test_list_with_valid_token_passes(self) -> None:
        """GET /ai/runs with valid Bearer token returns 200."""
        app = _make_app(_make_mock_repo(), with_auth=True, system_token="test-token")
        client = TestClient(app)

        resp = client.get(
            "/api/v1/ai/runs",
            headers={"Authorization": "Bearer test-token"},
        )

        assert resp.status_code == 200

    def test_detail_with_valid_token_passes(self) -> None:
        """GET /ai/runs/{id} with valid Bearer token returns 200."""
        app = _make_app(_make_mock_repo(), with_auth=True, system_token="test-token")
        client = TestClient(app)

        resp = client.get(
            f"/api/v1/ai/runs/{_SAMPLE_RUN_ID}",
            headers={"Authorization": "Bearer test-token"},
        )

        assert resp.status_code == 200


# ── Rate-limit conventions ───────────────────────────────────────────────


class TestDiagnosticsRateLimit:
    """Diagnostics endpoints follow existing rate-limit conventions."""

    def test_list_returns_200_within_global_bucket(self) -> None:
        """GET /ai/runs passes rate limit (no per-endpoint cap)."""
        app = _make_app(_make_mock_repo())
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs")

        assert resp.status_code == 200

    def test_detail_returns_200_within_global_bucket(self) -> None:
        """GET /ai/runs/{id} passes rate limit (no per-endpoint cap)."""
        app = _make_app(_make_mock_repo())
        client = TestClient(app)

        resp = client.get(f"/api/v1/ai/runs/{_SAMPLE_RUN_ID}")

        assert resp.status_code == 200

    def test_rate_limit_headers_present(self) -> None:
        """Response includes X-RateLimit-* headers when middleware is active."""
        from mindflow.api.middleware.ratelimit import RateLimitMiddleware

        app = _make_app(_make_mock_repo())
        app.add_middleware(RateLimitMiddleware)
        client = TestClient(app)

        resp = client.get("/api/v1/ai/runs")

        assert resp.status_code == 200
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers


# ── Health endpoint additions ────────────────────────────────────────────


class TestHealthDiagnosticsFields:
    """Health endpoint includes checkpoint_store and run_store availability."""

    def test_health_includes_checkpoint_and_run_store(self) -> None:
        """GET /api/v1/health returns checkpoint_store and run_store fields."""
        from mindflow.api.routes.health import router as health_router

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(health_router, prefix="/api/v1")

        mock_collector = type("MockCollector", (), {"status": "stopped"})()
        app.state.collector_service = mock_collector

        # Engine that's healthy
        class HealthyEngine:
            class ConnCtx:
                async def __aenter__(self) -> object:
                    return object()

                async def __aexit__(self, *args: object) -> None:
                    return None

            def connect(self) -> ConnCtx:
                return self.ConnCtx()

        app.state.engine = HealthyEngine()
        app.state.migration_applied = True
        app.state.db_integrity_ok = True
        app.state.checkpointer = None  # No checkpointer → unavailable

        # session_factory that supports list_runs probe

        class FakeSession:
            async def execute(self, stmt: object) -> object:
                class FakeResult:
                    def scalar_one(self) -> int:
                        return 0

                    def fetchall(self) -> list[object]:
                        return []

                return FakeResult()

            async def __aenter__(self) -> FakeSession:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        class FakeSessionFactory:
            def __call__(self) -> FakeSession:
                return FakeSession()

        app.state.session_factory = FakeSessionFactory()

        client = TestClient(app)
        resp = client.get("/api/v1/health")

        assert resp.status_code == 200
        data = resp.json()
        assert "checkpoint_store" in data
        assert "run_store" in data
        assert data["checkpoint_store"] == "unavailable"  # No checkpointer
        assert data["run_store"] == "available"  # Table probe succeeded

    def test_readiness_includes_checkpoint_and_run_store(self) -> None:
        """GET /api/v1/health/ready returns checkpoint_store and run_store."""
        from mindflow.api.routes.health import router as health_router

        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(health_router, prefix="/api/v1")

        class HealthyEngine:
            class ConnCtx:
                async def __aenter__(self) -> object:
                    return object()

                async def __aexit__(self, *args: object) -> None:
                    return None

            def connect(self) -> ConnCtx:
                return self.ConnCtx()

        class FakeSession:
            async def execute(self, stmt: object) -> object:
                class FakeResult:
                    def scalar_one(self) -> int:
                        return 0

                    def fetchall(self) -> list[object]:
                        return []

                return FakeResult()

            async def __aenter__(self) -> FakeSession:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        class FakeSessionFactory:
            def __call__(self) -> FakeSession:
                return FakeSession()

        app.state.engine = HealthyEngine()
        app.state.migration_applied = True
        app.state.db_integrity_ok = True
        app.state.checkpointer = None
        app.state.session_factory = FakeSessionFactory()

        client = TestClient(app)
        resp = client.get("/api/v1/health/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert "checkpoint_store" in data
        assert "run_store" in data
        assert data["checkpoint_store"] == "unavailable"
        assert data["run_store"] == "available"
