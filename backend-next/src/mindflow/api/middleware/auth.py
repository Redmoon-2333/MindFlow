"""Bearer token authentication middleware.

All endpoints (except health check and OpenAPI docs) require a valid Bearer
token in the ``Authorization`` header. The token is loaded from a local file
on startup and stored in ``app.state.system_token`` for access at request time.

Exempt paths (/api/v1/health, /docs, /openapi.json) are defined in
``EXEMPT_PATHS``.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from mindflow.infrastructure.security.token_manager import verify_token

_PROBLEM_BASE_URI: str = "https://mindflow.app/errors"

# Paths that don't require authentication
_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})


def _auth_required_response(path: str) -> Response:
    """Build a 401 response in RFC 9457 format."""
    body = {
        "type": f"{_PROBLEM_BASE_URI}/auth-required",
        "title": "Authentication Required",
        "status": 401,
        "detail": "缺少认证令牌或令牌无效",
        "instance": path,
    }
    return Response(
        status_code=401,
        content=json.dumps(body, ensure_ascii=False),
        media_type="application/problem+json",
    )


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that validates Bearer tokens on protected endpoints.

    The expected token is read from ``request.app.state.system_token`` at
    request time, allowing it to be set during the lifespan lifecycle.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path

        # Exempt health, docs, and OpenAPI schema endpoints
        if path in _EXEMPT_PATHS or path.startswith("/docs") or path.startswith(
            "/openapi.json"
        ) or path.startswith("/redoc"):
            return await call_next(request)

        expected_token: str = getattr(request.app.state, "system_token", "")

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _auth_required_response(path)

        token = auth_header.removeprefix("Bearer ").strip()
        if not verify_token(token, expected_token):
            return _auth_required_response(path)

        return await call_next(request)
