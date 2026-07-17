"""Tests for HostValidationMiddleware.

Covers:
  - Valid hosts: localhost, 127.0.0.1, [::1] (with any port)
  - Invalid hosts → 403 with RFC 9457
  - Host header parsing (IPv6 with port)
  - Edge cases: missing Host header, multiple ports
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.middleware.host import HostValidationMiddleware, _parse_host


class TestParseHost:
    """Verify Host header parsing."""

    def test_localhost(self):
        hostname, port = _parse_host("localhost")
        assert hostname == "localhost"
        assert port is None

    def test_localhost_with_port(self):
        hostname, port = _parse_host("localhost:8765")
        assert hostname == "localhost"
        assert port == 8765

    def test_ipv4(self):
        hostname, port = _parse_host("127.0.0.1")
        assert hostname == "127.0.0.1"
        assert port is None

    def test_ipv4_with_port(self):
        hostname, port = _parse_host("127.0.0.1:8080")
        assert hostname == "127.0.0.1"
        assert port == 8080

    def test_ipv6(self):
        hostname, port = _parse_host("[::1]")
        assert hostname == "::1"
        assert port is None

    def test_ipv6_with_port(self):
        hostname, port = _parse_host("[::1]:8765")
        assert hostname == "::1"
        assert port == 8765

    def test_external_host(self):
        hostname, port = _parse_host("example.com")
        assert hostname == "example.com"
        assert port is None

    def test_external_host_with_port(self):
        hostname, port = _parse_host("evil.com:9999")
        assert hostname == "evil.com"
        assert port == 9999

    def test_ipv6_bracket_suffix_smuggling_not_trusted(self):
        """[::1].evil.com must NOT parse to a trusted '::1' (review P1-1)."""
        hostname, _port = _parse_host("[::1].evil.com")
        assert hostname == "[::1].evil.com"  # full untrusted string

    def test_ipv6_malformed_port_not_trusted(self):
        """[::1]:notaport must not fall back to a trusted hostname (review P1-1)."""
        hostname, _port = _parse_host("[::1]:notaport")
        assert hostname == "[::1]:notaport"


class TestHostValidationMiddleware:
    """Verify Host header validation."""

    @pytest.fixture
    def client(self) -> TestClient:
        app = FastAPI()

        @app.get("/api/v1/test")
        async def test_endpoint():
            return {"ok": True}

        register_exception_handlers(app)
        app.add_middleware(HostValidationMiddleware)
        return TestClient(app)

    def test_localhost_allowed(self, client):
        """Requests from localhost should pass."""
        resp = client.get("/api/v1/test", headers={"host": "localhost"})
        assert resp.status_code == 200

    def test_localhost_with_port(self, client):
        """localhost with any port should pass."""
        resp = client.get("/api/v1/test", headers={"host": "localhost:8765"})
        assert resp.status_code == 200

    def test_ipv4_allowed(self, client):
        """127.0.0.1 should pass."""
        resp = client.get("/api/v1/test", headers={"host": "127.0.0.1"})
        assert resp.status_code == 200

    def test_ipv4_with_port(self, client):
        """127.0.0.1 with any port should pass."""
        resp = client.get("/api/v1/test", headers={"host": "127.0.0.1:8080"})
        assert resp.status_code == 200

    def test_ipv6_allowed(self, client):
        """[::1] should pass."""
        resp = client.get("/api/v1/test", headers={"host": "[::1]"})
        assert resp.status_code == 200

    def test_ipv6_with_port(self, client):
        """[::1] with any port should pass."""
        resp = client.get("/api/v1/test", headers={"host": "[::1]:8765"})
        assert resp.status_code == 200

    def test_external_host_blocked(self, client):
        """External hosts should return 403."""
        resp = client.get("/api/v1/test", headers={"host": "evil.com"})
        assert resp.status_code == 403
        data = resp.json()
        assert data["type"] == "https://mindflow.app/errors/forbidden-host"
        assert data["status"] == 403

    def test_ip_external_blocked(self, client):
        """External IP addresses should return 403."""
        resp = client.get("/api/v1/test", headers={"host": "192.168.1.1"})
        assert resp.status_code == 403

    def test_empty_host_allowed(self, client):
        """Missing Host header should pass through."""
        resp = client.get("/api/v1/test", headers={"host": ""})
        # Empty host is allowed through (default behavior)
        assert resp.status_code == 200

    def test_case_insensitive(self, client):
        """Hostname matching should be case-insensitive."""
        resp = client.get("/api/v1/test", headers={"host": "LOCALHOST"})
        assert resp.status_code == 200


class TestMalformedHostRequestLevel:
    """E2E-discovered regression: request.url parsing crashes on bracketed
    IPv6 Host with a suffix. Middleware must return 403, never 500."""

    def test_bracket_suffix_host_yields_403_not_500(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from mindflow.api.middleware.host import HostValidationMiddleware

        app = FastAPI()
        app.add_middleware(HostValidationMiddleware)

        @app.get("/api/v1/ping")
        async def ping():  # pragma: no cover - trivial
            return {"ok": True}

        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/v1/ping", headers={"host": "[::1].evil.com"})
        assert resp.status_code == 403
        assert resp.headers["content-type"].startswith("application/problem+json")
