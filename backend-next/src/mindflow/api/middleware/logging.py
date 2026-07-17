"""Request logging middleware — request_id generation + structured logging.

Adds a unique ``X-Request-ID`` header to each request and logs structured
information about the request lifecycle: method, path, status code, and
duration in milliseconds.

Uses loguru's ``bind(request_id=...)`` for contextual logging.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds request_id and logs structured request data.

    Every incoming request gets a unique ``X-Request-ID`` header.
    After the response is generated, a structured log line is emitted with:
      - method, path, status_code
      - duration in milliseconds
      - request_id

    The ``request_id`` is bound to loguru's context so downstream log calls
    within the same request automatically carry the id.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = uuid4().hex[:16]
        request.state.request_id = request_id

        start = time.perf_counter()

        with logger.contextualize(request_id=request_id):
            response: Response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000

            response.headers["X-Request-ID"] = request_id

            logger.info(
                "{} {} {} ({:.0f}ms)",
                request.method,
                request.scope["path"],
                response.status_code,
                elapsed_ms,
            )

        return response
