"""Tests for RateLimitMiddleware (token bucket rate limiter).

Covers:
  - TokenBucket refill and consume behaviour
  - Global rate limit (100 req/min)
  - Per-endpoint rate limits (attribution 1/30s, 20/day)
  - 429 response with RFC 9457 format
  - X-RateLimit-Remaining and X-RateLimit-Reset headers
  - Edge cases: daily hard limit, bucket exhaustion
  - F2: Auth-before-RateLimit ordering (unauth'd + over-limit request must
    get 401, not 429 — see app.py's create_app middleware ordering note)
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.middleware.auth import AuthMiddleware
from mindflow.api.middleware.ratelimit import RateLimitMiddleware, TokenBucket


class TestTokenBucket:
    """Verify TokenBucket algorithm."""

    async def test_initial_capacity(self):
        """A fresh bucket has full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        allowed, remaining, _reset = await bucket.consume()
        assert allowed is True
        assert remaining == pytest.approx(9.0, abs=0.1)

    async def test_consume_until_empty(self):
        """Consuming all tokens should exhaust the bucket."""
        bucket = TokenBucket(capacity=3, refill_rate=1.0)
        for _ in range(3):
            allowed, _remaining, _reset = await bucket.consume()
            assert allowed is True
        allowed, remaining, _reset = await bucket.consume()
        assert allowed is False
        assert remaining == pytest.approx(0.0, abs=0.1)

    async def test_refill_over_time(self):
        """Tokens should refill over time based on refill_rate."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        # Exhaust the bucket
        for _ in range(10):
            await bucket.consume()
        allowed, _remaining, _reset = await bucket.consume()
        assert allowed is False

        # Wait for refill
        time.sleep(0.15)  # Should have ~1.5 tokens
        allowed, remaining, _reset = await bucket.consume()
        assert allowed is True
        assert remaining == pytest.approx(0.5, abs=0.3)

    async def test_daily_hard_limit(self):
        """Daily hard limit should cap total usage."""
        bucket = TokenBucket(capacity=100, refill_rate=100.0, daily_hard_limit=5)
        for _ in range(5):
            allowed, _remaining, _reset = await bucket.consume()
            assert allowed is True
        # Should hit daily limit
        allowed, _remaining, _reset = await bucket.consume()
        assert allowed is False

    async def test_daily_limit_resets(self):
        """Daily limit should reset after 24h."""
        bucket = TokenBucket(capacity=100, refill_rate=100.0, daily_hard_limit=2)
        for _ in range(2):
            await bucket.consume()
        allowed, _remaining, _reset = await bucket.consume()
        assert allowed is False
        # Simulate day passing by manipulating internal state
        bucket._last_day_check = 0
        bucket._day_usage = 0
        allowed, _remaining, _reset = await bucket.consume()
        assert allowed is True

    async def test_never_exceeds_capacity(self):
        """Refill should never exceed the bucket capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=10.0)
        time.sleep(0.5)  # Would refill ~5 tokens
        bucket._refill()
        assert bucket._tokens == pytest.approx(5.0, abs=0.1)

    async def test_reset_timestamp(self):
        """Reset timestamp should be a future time."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        await bucket.consume(5)  # Exhaust
        _allowed, _remaining, reset_ts = await bucket.consume()
        assert reset_ts > time.time()


class TestRateLimitMiddleware:
    """Verify RateLimitMiddleware."""

    @pytest.fixture
    def app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/api/v1/test")
        async def test_endpoint():
            return {"ok": True}

        @app.post("/api/v1/analytics/attribution")
        async def attribution():
            return {"ok": True}

        app.add_middleware(RateLimitMiddleware, global_capacity=100, global_refill_rate=1000.0)
        return app

    @pytest.fixture
    def client(self, app: FastAPI) -> TestClient:
        return TestClient(app)

    def test_normal_request(self, client):
        """Normal requests should pass through."""
        resp = client.get("/api/v1/test")
        assert resp.status_code == 200
        assert "X-RateLimit-Remaining" in resp.headers
        assert "X-RateLimit-Reset" in resp.headers

    def test_rate_limit_headers_present(self, client):
        """Rate limit headers should be present on all responses."""
        resp = client.get("/api/v1/test")
        remaining = int(resp.headers["X-RateLimit-Remaining"])
        assert remaining > 0
        reset = int(resp.headers["X-RateLimit-Reset"])
        assert reset > 0

    def test_endpoint_specific_limit(self, client):
        """Endpoints with specific limits should have their own bucket."""
        # Attribution endpoint should work
        resp = client.post("/api/v1/analytics/attribution", json={})
        assert resp.status_code == 200


def test_rate_limit_global_exhaustion():
    """When the global bucket is exhausted, requests should get 429."""
    app = FastAPI()

    @app.get("/api/v1/test")
    async def test():
        return {"ok": True}

    # Very small bucket so we can exhaust it immediately
    app.add_middleware(RateLimitMiddleware, global_capacity=1, global_refill_rate=0.001)
    client = TestClient(app)

    # First request should pass
    resp = client.get("/api/v1/test")
    assert resp.status_code == 200

    # Second should be rate-limited
    resp = client.get("/api/v1/test")
    assert resp.status_code == 429
    data = resp.json()
    assert data["type"] == "https://mindflow.app/errors/rate-limited"
    assert data["status"] == 429


class TestAuthBeforeRateLimit:
    """F2: reproduces app.py's create_app middleware stack ordering.

    add_middleware is LIFO, so to make Auth run BEFORE RateLimit, Auth must
    be added AFTER RateLimit — exactly mirroring create_app's corrected
    order. An unauthenticated request that also happens to be over the rate
    limit must be rejected with 401 (auth), not 429 (rate limit): metering
    an unauthenticated request at all is the bug this fix closes.
    """

    def _make_app(self, expected_token: str = "secret-token") -> FastAPI:
        app = FastAPI()

        @app.get("/api/v1/test")
        async def test_endpoint():
            return {"ok": True}

        app.state.system_token = expected_token
        register_exception_handlers(app)

        # Same addition order as create_app: RateLimit first (inner),
        # Auth second (outer, runs first).
        app.add_middleware(
            RateLimitMiddleware, global_capacity=1, global_refill_rate=0.001
        )
        app.add_middleware(AuthMiddleware)
        return app

    def test_unauthenticated_and_over_limit_gets_401_not_429(self) -> None:
        """Auth must reject before RateLimit ever sees the request."""
        app = self._make_app()
        client = TestClient(app)

        # Exhaust the 1-token global bucket with an authenticated request
        # first, so the *next* request is both unauthenticated AND over
        # the rate limit.
        resp = client.get(
            "/api/v1/test", headers={"Authorization": "Bearer secret-token"}
        )
        assert resp.status_code == 200

        resp = client.get("/api/v1/test")  # no Authorization header
        assert resp.status_code == 401, (
            "unauthenticated request must be rejected by Auth before "
            "RateLimit gets a chance to return 429"
        )
        assert resp.json()["type"] == "https://mindflow.app/errors/auth-required"

    def test_authenticated_and_over_limit_still_gets_429(self) -> None:
        """A properly authenticated request is still subject to rate limiting."""
        app = self._make_app()
        client = TestClient(app)
        headers = {"Authorization": "Bearer secret-token"}

        resp = client.get("/api/v1/test", headers=headers)
        assert resp.status_code == 200

        resp = client.get("/api/v1/test", headers=headers)
        assert resp.status_code == 429
        assert resp.json()["type"] == "https://mindflow.app/errors/rate-limited"

