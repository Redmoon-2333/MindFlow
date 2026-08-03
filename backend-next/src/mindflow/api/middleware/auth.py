"""Bearer or HttpOnly-cookie authentication middleware for the local API."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from mindflow.infrastructure.security.token_manager import (
    SessionTokenStore,
    verify_token,
)

_PROBLEM_BASE_URI = "https://mindflow.app/errors"
_COOKIE_NAME = "mindflow_session"
_EXEMPT_PATHS = frozenset({
    "/api/v1/health",
    "/api/v1/health/live",
    "/api/v1/health/ready",
    "/api/v1/auth/bootstrap",
    "/api/v1/telemetry/browser/pair",
    "/api/v1/telemetry/browser/heartbeat",
    "/docs",
    "/openapi.json",
    "/redoc",
})
_EXEMPT_PREFIXES = ("/docs/", "/redoc/")


def _auth_required_response(path: str) -> Response:
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
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.scope["path"]
        if (
            not path.startswith("/api/")
            or path in _EXEMPT_PATHS
            or any(path.startswith(p) for p in _EXEMPT_PREFIXES)
        ):
            return await call_next(request)

        expected = getattr(request.app.state, "system_token", "")
        auth_header = request.headers.get("Authorization", "")
        bearer = (
            auth_header.removeprefix("Bearer ").strip()
            if auth_header.startswith("Bearer ")
            else ""
        )
        cookie = request.cookies.get(_COOKIE_NAME, "")
        session_store = getattr(request.app.state, "browser_sessions", None)
        cookie_valid = (
            isinstance(session_store, SessionTokenStore)
            and session_store.verify(cookie)
        )
        if not (verify_token(bearer, expected) or cookie_valid):
            return _auth_required_response(path)
        return await call_next(request)
