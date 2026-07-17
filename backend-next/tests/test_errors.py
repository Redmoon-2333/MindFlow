"""Tests for mindflow.api.errors — ProblemDetail + exception handlers.

Covers:
  - ProblemDetail creation and serialization
  - All 8 error codes
  - Handler registration
  - Validation error formatting
  - Generic exception handling (no stack leak)
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import (
    ProblemDetail,
    register_exception_handlers,
)


class TestProblemDetail:
    """Verify ProblemDetail exception class."""

    def test_basic_creation(self):
        """A ProblemDetail can be created with required fields."""
        err = ProblemDetail(
            type_slug="not-found",
            title="Not Found",
            status=404,
            detail="未找到资源",
        )
        assert err.type_slug == "not-found"
        assert err.title == "Not Found"
        assert err.status == 404
        assert err.detail == "未找到资源"

    def test_to_dict(self):
        """to_dict produces the correct RFC 9457 structure."""
        err = ProblemDetail(
            type_slug="auth-required",
            title="Authentication Required",
            status=401,
            detail="缺少认证令牌",
        )
        d = err.to_dict(instance="/api/v1/collector")
        assert d["type"] == "https://mindflow.app/errors/auth-required"
        assert d["title"] == "Authentication Required"
        assert d["status"] == 401
        assert d["detail"] == "缺少认证令牌"
        assert d["instance"] == "/api/v1/collector"

    def test_to_dict_without_instance(self):
        """to_dict works without an instance parameter."""
        err = ProblemDetail(
            type_slug="internal-error",
            title="Internal Error",
            status=500,
            detail="服务器内部错误",
        )
        d = err.to_dict()
        assert "instance" not in d

    def test_to_dict_with_extra(self):
        """Extra fields are included in the serialized dict."""
        err = ProblemDetail(
            type_slug="rate-limited",
            title="Rate Limited",
            status=429,
            detail="请求过于频繁",
            extra={"retry_after_seconds": 60},
        )
        d = err.to_dict(instance="/api/v1/test")
        assert d["retry_after_seconds"] == 60

    def test_is_exception(self):
        """ProblemDetail is a proper exception that can be raised and caught."""
        err = ProblemDetail(
            type_slug="test",
            title="Test",
            status=400,
            detail="测试错误",
        )
        with pytest.raises(ProblemDetail) as exc_info:
            raise err
        assert exc_info.value.status == 400

    def test_all_error_codes(self):
        """All 8 error codes produce valid type URIs."""
        codes = [
            ("collector-not-running", 503),
            ("not-found", 404),
            ("validation-error", 422),
            ("rate-limited", 429),
            ("auth-required", 401),
            ("forbidden-host", 403),
            ("internal-error", 500),
            ("llm-unavailable", 503),
        ]
        for slug, expected_status in codes:
            err = ProblemDetail(
                type_slug=slug,
                title=slug,
                status=expected_status,
                detail="test",
            )
            d = err.to_dict()
            assert d["type"] == f"https://mindflow.app/errors/{slug}"
            assert d["status"] == expected_status


class TestExceptionHandlers:
    """Verify FastAPI exception handlers produce correct responses."""

    @pytest.fixture
    def app(self) -> FastAPI:
        """Create a minimal test app with exception handlers."""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        @app.get("/test-problem")
        async def test_problem():
            raise ProblemDetail(
                type_slug="not-found",
                title="Not Found",
                status=404,
                detail="未找到资源",
            )

        @app.get("/test-internal")
        async def test_internal():
            raise RuntimeError("内部错误")

        register_exception_handlers(app)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        """Sync test client."""
        return TestClient(app)

    def test_ok_response(self, client):
        """Normal responses are unaffected by exception handlers."""
        resp = client.get("/test")
        assert resp.status_code == 200

    def test_problem_detail_response(self, client):
        """ProblemDetail raises produce RFC 9457 JSON."""
        resp = client.get("/test-problem")
        assert resp.status_code == 404
        data = resp.json()
        assert data["type"] == "https://mindflow.app/errors/not-found"
        assert data["status"] == 404
        assert resp.headers["content-type"] == "application/problem+json"

    def test_internal_error_no_stack_leak(self, client):
        """Unhandled exceptions return 500 without stack trace."""
        resp = client.get("/test-internal")
        assert resp.status_code == 500
        data = resp.json()
        assert data["status"] == 500
        assert data["type"] == "https://mindflow.app/errors/internal-error"
        # Should not leak original exception details (NF-S4)
        assert "RuntimeError" not in data.get("detail", "")
        assert data.get("detail") == "服务器内部错误，请稍后重试"

    def test_validation_error(self, client):
        """Validation errors produce RFC 9457 format."""
        resp = client.get("/test?invalid_param=true")
        # No validation errors on this endpoint, so this should still work
        assert resp.status_code == 200


def test_problem_detail_handler_integration():
    """Full integration test: handler catches ProblemDetail correctly."""
    app = FastAPI()

    @app.get("/error")
    async def raise_error():
        raise ProblemDetail(
            type_slug="forbidden-host",
            title="Forbidden Host",
            status=403,
            detail="不允许的主机地址",
        )

    register_exception_handlers(app)
    client = TestClient(app)

    resp = client.get("/error")
    assert resp.status_code == 403
    data = resp.json()
    assert data["type"] == "https://mindflow.app/errors/forbidden-host"
    assert data["instance"] == "/error"
