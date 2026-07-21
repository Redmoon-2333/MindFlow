"""RFC 9457 ProblemDetail exception class and FastAPI exception handlers.

All error responses follow the standard Problem Details format:

  ```json
  {
    "type": "https://mindflow.app/errors/<error-code>",
    "title": "<English title>",
    "status": <HTTP status>,
    "detail": "<Chinese description>",
    "instance": "/api/v1/<path>"
  }
  ```

The ``type`` URI uses English identifiers for machine readability while
``detail`` uses Chinese for the user-facing message (per requirements §4.2).

Error codes implemented (8 total):
  - collector-not-running  503
  - not-found              404
  - validation-error       422
  - rate-limited           429
  - auth-required          401
  - forbidden-host         403
  - internal-error         500
  - llm-unavailable        503
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from mindflow.errors import NoActivityDataError

_PROBLEM_BASE_URI: str = "https://mindflow.app/errors"
"""Base URI for error types. Per §4.2, not a resolvable URL — just an identifier."""


# ── Exception class ──────────────────────────────────────────────────────────


class ProblemDetail(Exception):  # noqa: N818
    """An RFC 9457 Problem Details exception that renders as ``problem+json``.

    Raise this inside route handlers or middleware to produce a structured
    error response. The exception handler registered in ``register_handlers``
    catches it and converts it to the standard JSON format.

    Args:
        type_slug: Path segment after ``errors/`` (e.g. ``"not-found"``).
        title: Short English title for the error category.
        status: HTTP status code.
        detail: Human-readable Chinese description.
        instance: The request path that triggered the error (optional, set
            automatically by the exception handler).
        extra: Additional fields to include in the response body.
    """

    def __init__(
        self,
        type_slug: str,
        title: str,
        status: int,
        detail: str,
        instance: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.type_slug = type_slug
        self.title = title
        self.status = status
        self.detail = detail
        self.instance = instance
        self.extra = extra or {}
        super().__init__(self.detail)

    def to_dict(self, instance: str | None = None) -> dict[str, Any]:
        """Serialize to a problem+json dict."""
        body: dict[str, Any] = {
            "type": f"{_PROBLEM_BASE_URI}/{self.type_slug}",
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }
        resolved_instance = self.instance or instance
        if resolved_instance:
            body["instance"] = resolved_instance
        body.update(self.extra)
        return body


# ── Error factories ──────────────────────────────────────────────────────────


def _rate_limited(retry_after: int = 60) -> ProblemDetail:
    """Return a 429 error with retry-after hint."""
    return ProblemDetail(
        type_slug="rate-limited",
        title="Rate Limited",
        status=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="请求过于频繁，请稍后再试",
        extra={"retry_after_seconds": retry_after},
    )


def _auth_required() -> ProblemDetail:
    """Return a 401 error for missing/invalid authentication."""
    return ProblemDetail(
        type_slug="auth-required",
        title="Authentication Required",
        status=status.HTTP_401_UNAUTHORIZED,
        detail="缺少认证令牌或令牌无效",
    )


def _forbidden_host() -> ProblemDetail:
    """Return a 403 error for untrusted Host header."""
    return ProblemDetail(
        type_slug="forbidden-host",
        title="Forbidden Host",
        status=status.HTTP_403_FORBIDDEN,
        detail="不允许的主机地址，仅支持本地访问",
    )


def _collector_not_running(instance: str | None = None) -> ProblemDetail:
    """Return a 503 error when the collector is not active."""
    return ProblemDetail(
        type_slug="collector-not-running",
        title="Collector Not Running",
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="数据采集器未运行，请先启动采集器",
        instance=instance,
    )


def _not_found(resource: str = "资源") -> ProblemDetail:
    """Return a 404 error."""
    return ProblemDetail(
        type_slug="not-found",
        title="Not Found",
        status=status.HTTP_404_NOT_FOUND,
        detail=f"未找到{resource}",
    )


def _internal_error() -> ProblemDetail:
    """Return a 500 error without exposing stack details."""
    return ProblemDetail(
        type_slug="internal-error",
        title="Internal Error",
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="服务器内部错误，请稍后重试",
    )


def _llm_unavailable() -> ProblemDetail:
    """Return a 503 when the LLM tier degraded to the rule engine (review P2-3).

    Note: per the requirements error table this signals degradation — the
    request itself still completes via the rule-engine fallback where possible.
    """
    return ProblemDetail(
        type_slug="llm-unavailable",
        title="LLM Unavailable",
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="LLM 服务暂不可用，已降级为规则引擎分析",
    )


# ── Exception handlers ───────────────────────────────────────────────────────


def _problem_handler(request: Request, exc: ProblemDetail) -> JSONResponse:
    """Convert ProblemDetail exception to RFC 9457 JSON response."""
    return JSONResponse(
        status_code=exc.status,
        content=exc.to_dict(instance=str(request.scope["path"])),
        headers={"Content-Type": "application/problem+json"},
    )


def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert Pydantic/FastAPI validation errors to RFC 9457 format."""
    errors = exc.errors()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "type": f"{_PROBLEM_BASE_URI}/validation-error",
            "title": "Validation Error",
            "status": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "detail": "请求参数验证失败",
            "instance": str(request.scope["path"]),
            "validation_errors": [
                {
                    "loc": err.get("loc", []),
                    "msg": err.get("msg", ""),
                    "type": err.get("type", ""),
                }
                for err in errors
            ],
        },
        headers={"Content-Type": "application/problem+json"},
    )


def _no_activity_handler(request: Request, exc: NoActivityDataError) -> JSONResponse:
    """Map the service-layer ``NoActivityDataError`` to an RFC 9457 404.

    Lets services signal "nothing to analyse" without importing the HTTP
    layer — the dependency points api → services, not the reverse (E4).
    """
    err = _not_found(exc.resource)
    return JSONResponse(
        status_code=err.status,
        content=err.to_dict(instance=str(request.scope["path"])),
        headers={"Content-Type": "application/problem+json"},
    )


def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions.

    Never leaks stack traces to the client — logs at ERROR level and returns
    a generic 500 response (per NF-S4: output sanitisation).
    """
    logger.error("Unhandled exception processing {}: {}", request.scope["path"], exc)
    err = _internal_error()
    return JSONResponse(
        status_code=err.status,
        content=err.to_dict(instance=str(request.scope["path"])),
        headers={"Content-Type": "application/problem+json"},
    )


# ── Registration ─────────────────────────────────────────────────────────────


def register_exception_handlers(app: FastAPI) -> None:
    """Register all RFC 9457 exception handlers on the FastAPI app.

    Must be called after ``app = FastAPI()``, before any routes are registered.

    For the generic handler, we register it for ``RuntimeError`` and
    ``Exception`` at the FastAPI app level AND on the underlying
    ExceptionMiddleware to ensure isinstance-based matching works correctly.

    Args:
        app: The FastAPI application instance.
    """
    handler_t = Callable[[Request, Exception], Response]

    app.add_exception_handler(ProblemDetail, cast(handler_t, _problem_handler))
    app.add_exception_handler(RequestValidationError, cast(handler_t, _validation_handler))

    # Service-layer domain error → 404. Registered before the Exception
    # catch-all so Starlette's MRO lookup prefers this more-specific handler
    # (E4: services raise this instead of importing api.errors).
    app.add_exception_handler(NoActivityDataError, cast(handler_t, _no_activity_handler))

    # Register for both Exception and RuntimeError to handle exact type matching
    # in Starlette's wrap_app_handling_exceptions (which uses exact type lookup)
    app.add_exception_handler(RuntimeError, cast(handler_t, _generic_handler))
    app.add_exception_handler(Exception, cast(handler_t, _generic_handler))
