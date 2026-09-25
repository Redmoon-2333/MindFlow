"""HTTP-boundary policy shared by SDK models and raw completion consumers.

Three pieces live here:

* :class:`GatedHTTPClient` — an ``httpx.AsyncClient`` that acquires the shared
  LLM concurrency gate once per HTTP attempt, so every raw consumer (Ollama,
  intervention copy, structured attribution) respects the same in-flight cap.
* :class:`ProviderHTTPClient` — the ECNU-aware variant used by the SDK and raw
  ``post`` consumers; it also injects the campus gateway's thinking fields.
* :class:`SharedHTTPClientPool` — the *single owner* of the fallback tier's
  HTTP client. The Ollama degradation path used to build (and close) a fresh
  ``httpx.AsyncClient`` per call, which opened a new connection pool for every
  request; the pool is now created once, reused, and closed once by whoever
  owns it (``ProviderRegistry``).

Retry/backoff arithmetic is shared with :mod:`mindflow.infrastructure.llm.client`
so both paths back off identically.
"""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

import httpx

from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.ecnu import ecnu_request_fields

#: Shared backoff ceiling (matches ``LLMSettings.timeout_s`` scale).
_BACKOFF_CAP_S: float = 60.0

#: Connection-pool bounds for the shared fallback client. The concurrency gate,
#: not the pool, is what bounds in-flight generations; keep-alive connections
#: are what make reuse worthwhile.
_DEFAULT_LIMITS = httpx.Limits(max_connections=8, max_keepalive_connections=4)


# ── Retry/backoff arithmetic (single implementation) ──────────────────────────


def compute_backoff(attempt: int, retry_after: int | None = None) -> float:
    """Delay before retry number *attempt* (zero-based).

    ``Retry-After`` (integer seconds) wins when present; otherwise exponential
    backoff with jitter: ``min(2 ** attempt + random.uniform(0, 1), cap)``.
    """
    if retry_after is not None and retry_after > 0:
        return min(float(retry_after), _BACKOFF_CAP_S)
    return min(float(2**attempt) + random.uniform(0.0, 1.0), _BACKOFF_CAP_S)


def parse_retry_after(response: httpx.Response) -> int | None:
    """Parse ``Retry-After`` as an integer number of seconds (None when absent)."""
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return int(header)
    except (ValueError, TypeError):
        return None


# ── Per-attempt metrics (queue vs HTTP split), for observability ───────────────


@dataclass(frozen=True, slots=True)
class HTTPAttemptMetrics:
    """One HTTP attempt's timing and outcome — no body, headers, or URL."""

    queue_latency_ms: float = 0.0
    http_latency_ms: float = 0.0
    status_class: str = ""
    error_category: str = ""


_attempt_sink: ContextVar[list[HTTPAttemptMetrics] | None] = ContextVar(
    "mindflow_llm_http_attempts", default=None,
)


class HTTPAttemptCollector:
    """Collect metrics for every HTTP attempt made inside this context.

    A class rather than a generator context manager so callers can still read
    :attr:`attempts` after the block has exited — including on the failure path,
    where the partial attempt log is exactly what needs recording.
    """

    def __init__(self) -> None:
        self.attempts: list[HTTPAttemptMetrics] = []
        self._token: Token[list[HTTPAttemptMetrics] | None] | None = None

    def __enter__(self) -> HTTPAttemptCollector:
        self._token = _attempt_sink.set(self.attempts)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _attempt_sink.reset(self._token)
            self._token = None


@contextmanager
def collect_http_attempts() -> Iterator[list[HTTPAttemptMetrics]]:
    """Collect metrics for every HTTP attempt made inside this context.

    The collector is a ``ContextVar`` so concurrent requests never share a sink.
    """
    collector = HTTPAttemptCollector()
    with collector:
        yield collector.attempts


def record_http_attempt(metrics: HTTPAttemptMetrics) -> None:
    """Append *metrics* to the active collector (no-op when none is active)."""
    sink = _attempt_sink.get()
    if sink is not None:
        sink.append(metrics)


def status_class(status_code: int) -> str:
    """Return ``"2xx"``/``"4xx"``/``"5xx"`` for *status_code*, else ``""``."""
    if 100 <= status_code < 600:
        return f"{status_code // 100}xx"
    return ""


def transport_error_category(exc: BaseException) -> str:
    """Sanitized category for a failed attempt — never the exception message."""
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return "network"
    if isinstance(exc, httpx.HTTPStatusError):
        return "http_error"
    return "error"


# ── Gated clients ─────────────────────────────────────────────────────────────


class _GatedStream(httpx.AsyncByteStream):
    """Keep a streaming generation's slot until the response is closed."""

    def __init__(self, stream: httpx.AsyncByteStream, gate: LLMConcurrencyGate) -> None:
        self._stream = stream
        self._gate = gate
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._stream.aclose()
        finally:
            await self._gate.__aexit__()


class GatedHTTPClient(httpx.AsyncClient):
    """Acquire once per HTTP attempt, never around a workflow or retry loop.

    Provider-neutral: this is the client the fallback (Ollama) tier uses so it
    shares the same concurrency budget as the SDK paths, without inheriting
    ECNU's payload rewriting.
    """

    def __init__(self, gate: LLMConcurrencyGate, **kwargs: Any) -> None:
        self._gate = gate
        super().__init__(**kwargs)

    async def send(
        self, request: httpx.Request, *, stream: bool = False, **kwargs: Any,
    ) -> httpx.Response:
        queue_started = time.perf_counter()
        await self._gate.__aenter__()
        queue_ms = (time.perf_counter() - queue_started) * 1000.0
        http_started = time.perf_counter()
        handed_off = False
        try:
            response = await super().send(request, stream=stream, **kwargs)
            if stream and not response.is_closed:
                assert isinstance(response.stream, httpx.AsyncByteStream)
                response.stream = _GatedStream(response.stream, self._gate)
                handed_off = True
            record_http_attempt(HTTPAttemptMetrics(
                queue_latency_ms=queue_ms,
                http_latency_ms=(time.perf_counter() - http_started) * 1000.0,
                status_class=status_class(response.status_code),
            ))
            return response
        except Exception as exc:
            record_http_attempt(HTTPAttemptMetrics(
                queue_latency_ms=queue_ms,
                http_latency_ms=(time.perf_counter() - http_started) * 1000.0,
                error_category=transport_error_category(exc),
            ))
            raise
        finally:
            if not handed_off:
                await self._gate.__aexit__()


class ProviderHTTPClient(GatedHTTPClient):
    """ECNU-aware gated client: raw ``post`` consumers inherit output/timeout policy.

    SDK clients construct their own payloads, but use the same ``send`` gate.
    """

    def __init__(
        self, settings: LLMSettings, gate: LLMConcurrencyGate, **kwargs: Any,
    ) -> None:
        self._settings = settings
        super().__init__(gate, **kwargs)

    async def post(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        settings = self._settings
        payload = kwargs.get("json")
        if settings.is_ecnu and isinstance(payload, dict):
            payload = dict(payload)
            payload.pop("max_tokens", None)
            payload.pop("reasoning_effort", None)
            payload.update(ecnu_request_fields(
                model=str(payload.get("model") or settings.model or "ecnu-max"),
                thinking_enabled=settings.thinking_enabled,
                reasoning_effort=settings.reasoning_effort,
                max_tokens=settings.max_output_tokens,
            ))
            if settings.thinking_enabled:
                payload.pop("temperature", None)
            kwargs["json"] = payload
            kwargs["timeout"] = settings.timeout_s
        return await super().post(url, **kwargs)


# ── Shared fallback client (phase 4.3) ────────────────────────────────────────


class SharedHTTPClientPool:
    """Own exactly one gated ``httpx.AsyncClient`` and hand it to every caller.

    The client is built lazily on first use (so a process that never degrades
    to L2 never opens a pool), reused for every later call, and closed exactly
    once through :meth:`aclose`.

    Args:
        gate: Shared concurrency gate — one slot per HTTP attempt.
        timeout_s: Unified request timeout applied by the pool's own client.
        max_retries: Retry budget advertised to callers of :attr:`client`.
        backoff_cap_s: Upper bound for the exponential backoff between retries.
        base_url: Optional base URL; callers may pass absolute URLs instead.
        headers: Optional default headers.
        limits: Connection-pool bounds (defaults to keep-alive friendly limits).
        name: Label used in lifecycle logs.
        factory: Construction seam — called once, with no arguments, when a
            test or tool must supply its own client (e.g. ``MockTransport``).
    """

    def __init__(
        self,
        *,
        gate: LLMConcurrencyGate,
        timeout_s: float = 60.0,
        max_retries: int = 1,
        backoff_cap_s: float = _BACKOFF_CAP_S,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
        limits: httpx.Limits | None = None,
        name: str = "fallback",
        factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._gate = gate
        self.timeout_s = float(timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.backoff_cap_s = float(backoff_cap_s)
        self._base_url = base_url
        self._headers = dict(headers) if headers else None
        self._limits = limits if limits is not None else _DEFAULT_LIMITS
        self._name = name
        self._factory = factory
        self._client: httpx.AsyncClient | None = None
        self._constructions = 0
        self._closes = 0
        self._closed = False

    # ── Introspection (regression tests assert these counters) ────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def constructions(self) -> int:
        """How many clients this pool has built (expected: 0 or 1)."""
        return self._constructions

    @property
    def closes(self) -> int:
        """How many times the owned client has been closed (expected: 0 or 1)."""
        return self._closes

    def peek(self) -> httpx.AsyncClient | None:
        """Return the owned client without constructing it."""
        return self._client

    # ── The shared client ─────────────────────────────────────────────────

    @property
    def client(self) -> httpx.AsyncClient:
        """The shared client, constructed on first access and then reused."""
        if self._closed:
            raise RuntimeError(f"{self._name} HTTP client pool is closed")
        if self._client is None:
            self._client = self._build()
            self._constructions += 1
        return self._client

    def _build(self) -> httpx.AsyncClient:
        if self._factory is not None:
            return self._factory()
        kwargs: dict[str, Any] = {
            "headers": self._headers,
            "timeout": httpx.Timeout(self.timeout_s),
            "limits": self._limits,
            "trust_env": False,
        }
        # ``base_url=None`` is rejected by httpx; only pass it when configured.
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return GatedHTTPClient(self._gate, **kwargs)

    async def aclose(self) -> None:
        """Close the owned client exactly once (idempotent, never raises)."""
        if self._closed:
            return
        self._closed = True
        client, self._client = self._client, None
        if client is None:
            return
        self._closes += 1
        with contextlib.suppress(Exception):
            await client.aclose()

    def snapshot(self) -> dict[str, object]:
        """Lifecycle evidence for diagnostics (no URLs, no headers)."""
        return {
            "name": self._name,
            "constructions": self._constructions,
            "closes": self._closes,
            "closed": self._closed,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
        }


# ── Process-wide default pool ─────────────────────────────────────────────────
#
# ``AnalysisGraph`` builds its ``FallbackRunContext`` from settings (it is owned
# by another workstream and does not see the registry), so the graph's Ollama
# node resolves the registry-owned pool through this slot. Owning the client
# still sits with ``ProviderRegistry``: it installs its pool at construction and
# clears + closes it on shutdown. Tests and standalone scripts may inject a
# client per call instead (``FallbackRunContext.ollama_client``).

_shared_fallback_pool: SharedHTTPClientPool | None = None


def install_shared_fallback_pool(pool: SharedHTTPClientPool) -> None:
    """Install *pool* as the process-wide fallback client owner."""
    global _shared_fallback_pool
    _shared_fallback_pool = pool


def shared_fallback_pool() -> SharedHTTPClientPool | None:
    """Return the installed fallback pool, or ``None`` when nobody owns one."""
    return _shared_fallback_pool


def clear_shared_fallback_pool(pool: SharedHTTPClientPool | None = None) -> None:
    """Clear the installed pool.

    With *pool* given, the slot is only cleared when that exact instance is
    installed — a late ``shutdown()`` of an old registry must not detach a newer
    registry's pool.
    """
    global _shared_fallback_pool
    if pool is None or _shared_fallback_pool is pool:
        _shared_fallback_pool = None
