"""In-memory token bucket rate limiter middleware.

Uses a simple in-memory token bucket algorithm (no Redis dependency) since
this is a local-only desktop application §4.4.

Rate limits:
  - Global: 100 requests/minute, single shared bucket (local single-user app;
    per-IP separation is meaningless on localhost). Single-process only —
    buckets are in-memory and NOT shared across workers.
  - Per-endpoint: configurable via ``endpoint_limits`` dict keyed by path

The middleware adds the following response headers:
  - ``X-RateLimit-Remaining``: Number of tokens remaining in the current window
  - ``X-RateLimit-Reset``: Unix timestamp when the bucket resets

When the limit is exceeded, returns a 429 response in RFC 9457 format.

Token bucket parameters:
  - ``capacity``: Maximum number of tokens the bucket can hold
  - ``refill_rate``: Tokens added per second
  - ``daily_hard_limit``: Optional hard daily cap (e.g. 20/day for attribution)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_PROBLEM_BASE_URI: str = "https://mindflow.app/errors"


class TokenBucket:
    """In-memory token bucket with configurable capacity, refill, and daily cap.

    Args:
        capacity: Maximum number of tokens.
        refill_rate: Tokens added per second.
        daily_hard_limit: Optional hard limit per 24-hour window.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        daily_hard_limit: int | None = None,
    ) -> None:
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._daily_hard_limit = daily_hard_limit
        self._tokens = capacity
        self._last_refill = time.time()
        self._day_usage = 0
        self._last_day_check = time.time()
        # Serializes read-check-deduct so concurrent coroutines cannot both
        # pass the `tokens >= 1` check and drive the bucket negative (review P2-1).
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

        # Reset daily counter if day changed
        if now - self._last_day_check > 86400:
            self._day_usage = 0
            self._last_day_check = now

    async def consume(self, tokens: float = 1.0) -> tuple[bool, float, float]:
        """Try to consume *tokens* from the bucket (coroutine-safe).

        Returns:
            Tuple of (allowed, remaining, reset_timestamp).
        """
        async with self._lock:
            self._refill()

            if self._daily_hard_limit is not None and self._day_usage >= self._daily_hard_limit:
                return False, 0.0, self._last_refill + 86400

            if self._tokens >= tokens:
                self._tokens -= tokens
                if self._daily_hard_limit is not None:
                    self._day_usage += 1
                return True, self._tokens, self._last_refill + (
                    self._capacity - self._tokens
                ) / max(self._refill_rate, 0.001)

            next_token_time = (tokens - self._tokens) / max(self._refill_rate, 0.001)
            return False, 0.0, self._last_refill + next_token_time


# Default per-endpoint rate limit configurations
_DEFAULT_ENDPOINT_LIMITS: dict[str, TokenBucket] = {
    "/api/v1/analytics/attribution": TokenBucket(
        capacity=5,
        refill_rate=1.0 / 30.0,
        daily_hard_limit=20,
    ),
    "/api/v1/analytics/train": TokenBucket(
        capacity=3,
        refill_rate=1.0 / 60.0,
        daily_hard_limit=3,
    ),
}


def _build_429_response(
    path: str,
    reset_ts: float,
) -> Response:
    """Build a 429 Too Many Requests response in RFC 9457 format."""
    retry_after = max(1, int(reset_ts - time.time()))
    body = {
        "type": f"{_PROBLEM_BASE_URI}/rate-limited",
        "title": "Rate Limited",
        "status": 429,
        "detail": "请求过于频繁，请稍后再试",
        "instance": path,
        "retry_after_seconds": retry_after,
    }
    return Response(
        status_code=429,
        content=json.dumps(body, ensure_ascii=False),
        media_type="application/problem+json",
        headers={
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(reset_ts)),
            "Retry-After": str(retry_after),
        },
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces token bucket rate limits.

    Applies a global rate limit (100 req/min) and per-endpoint limits where
    configured. Missing tokens result in a 429 RFC 9457 error.

    Args:
        app: The ASGI application.
        global_capacity: Global bucket capacity (default 100).
        global_refill_rate: Global refill rate in tokens/second.
        endpoint_limits: Optional dict of path -> TokenBucket for per-endpoint limits.
    """

    def __init__(
        self,
        app: Any,
        global_capacity: float = 100.0,
        global_refill_rate: float = 100.0 / 60.0,
        endpoint_limits: dict[str, TokenBucket] | None = None,
    ) -> None:
        super().__init__(app)
        self._global_bucket = TokenBucket(
            capacity=global_capacity,
            refill_rate=global_refill_rate,
        )
        self._endpoint_limits = endpoint_limits or _DEFAULT_ENDPOINT_LIMITS.copy()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.scope["path"]

        # Check per-endpoint limit first (more specific)
        endpoint_bucket = self._endpoint_limits.get(path)
        if endpoint_bucket is not None:
            allowed, _remaining, reset_ts = await endpoint_bucket.consume()
            if not allowed:
                return _build_429_response(path, reset_ts)

        # Check global bucket
        allowed, remaining, reset_ts = await self._global_bucket.consume()
        if not allowed:
            return _build_429_response(path, reset_ts)

        # Add rate limit headers to the response
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(int(remaining))
        response.headers["X-RateLimit-Reset"] = str(int(reset_ts))
        return response
