"""Host header validation middleware.

Only allows requests where the Host header is one of:
  - localhost (with any port)
  - 127.0.0.1 (with any port)
  - [::1] (with any port)

All other Host values receive a 403 Forbidden response.
This prevents DNS rebinding attacks (per threat model §4.1, NF-S2).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

_PROBLEM_BASE_URI: str = "https://mindflow.app/errors"


def _parse_host(host_header: str) -> tuple[str, int | None]:
    """Parse a Host header into (hostname, port).

    Handles IPv6 [::1]:port syntax.
    """
    host_header = host_header.strip()
    if host_header.startswith("["):
        bracket_end = host_header.find("]")
        if bracket_end == -1:
            return host_header, None
        hostname = host_header[1:bracket_end]
        rest = host_header[bracket_end + 1 :]
        if not rest:
            return hostname, None
        if rest.startswith(":"):
            try:
                return hostname, int(rest[1:])
            except ValueError:
                # Malformed port ("[::1]:evil") — treat the whole header as the
                # hostname so it fails the trust check (review P1-1).
                return host_header, None
        # Trailing garbage after the bracket ("[::1].evil.com") would otherwise
        # be silently discarded, letting attackers smuggle a trusted-looking
        # IPv6 literal in front of their real domain (review P1-1).
        return host_header, None
    if ":" in host_header:
        parts = host_header.rsplit(":", 1)
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return host_header, None
    return host_header, None


_TRUSTED_HOST_LOWERCASE: set[str] = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _forbidden_host_response(path: str) -> Response:
    """Build a 403 response in RFC 9457 format."""
    body = {
        "type": f"{_PROBLEM_BASE_URI}/forbidden-host",
        "title": "Forbidden Host",
        "status": 403,
        "detail": "不允许的主机地址，仅支持本地访问",
        "instance": path,
    }
    return Response(
        status_code=403,
        content=json.dumps(body, ensure_ascii=False),
        media_type="application/problem+json",
    )


class HostValidationMiddleware(BaseHTTPMiddleware):
    """Middleware that rejects requests with untrusted Host headers.

    Only permits localhost, 127.0.0.1, and [::1] (with any port).
    All other hosts trigger a 403 Forbidden response in RFC 9457 format.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        host_header = request.headers.get("host", "")
        if host_header:
            hostname, _port = _parse_host(host_header)
            if hostname.lower() not in _TRUSTED_HOST_LOWERCASE:
                return _forbidden_host_response(str(request.url.path))

        return await call_next(request)
