"""Shared, offline LLM test support: injected HTTP transport + wire recorder.

The optimisation-plan regressions (2.1 policy, 2.5 observability, 4.3 shared
client) all need the same three things, and none of them may touch the network:

* an HTTP transport that produces OpenAI-compatible responses (or injected
  failures) for whichever SDK client ends up sending the request;
* a recorder that keeps the *parsed request bodies*, so a test asserts what
  actually left the process (``reasoning_effort``, ``max_completion_tokens``,
  ``temperature``, …) rather than what the code intended to send;
* in-flight bookkeeping, so a concurrency assertion compares the gate's own
  statistic against the genuinely simultaneous requests the handler saw.

:class:`MockLLMWire` bundles those plus the two installation seams the SDK
forces on us:

* :meth:`MockLLMWire.patch_async_client` — patch ``httpx.AsyncClient.__init__``
  *before* the model is built (the pattern already used by
  ``test_acceptance_provider_regressions``). This is the seam for lazily built
  clients, e.g. ``LangChainGateway``'s on-first-call models.
* :meth:`MockLLMWire.attach` — re-point an already-built client's transport,
  for clients a registry constructs eagerly in its own ``__init__``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain_openai.chat_models import _client_utils

from mindflow.agents.policies import CompletionPolicy
from mindflow.config import LLMSettings

ECNU_BASE_URL = "https://chat.ecnu.edu.cn/open/api/v1"
GENERIC_BASE_URL = "https://api.example-provider.test/v1"

#: Substring identifying the chat-completions endpoint in recorded URLs.
CHAT_COMPLETIONS_PATH = "/chat/completions"

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def make_settings(**overrides: object) -> LLMSettings:
    """ECNU-flavoured settings for tests (override any field by keyword).

    ``ecnu_compat_enabled`` is what makes these settings mean anything: since
    the production L1 is pinned to DeepSeek direct, ``l1_target()`` would
    otherwise replace the campus URL, model and key configured here.
    ``reasoning_effort="max"`` is the campus gateway's own default tier, kept
    explicit because the application default is now DeepSeek's ``"high"``.
    """
    defaults: dict[str, object] = {
        "api_key": "test-key",
        "base_url": ECNU_BASE_URL,
        "model": "ecnu-max",
        "provider": "ecnu",
        "ecnu_compat_enabled": True,
        "reasoning_effort": "max",
        "timeout_s": 30,
        "max_retries": 0,
        "max_output_tokens": 16384,
        "max_concurrent_requests": 1,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)  # type: ignore[arg-type]


def make_generic_settings(**overrides: object) -> LLMSettings:
    """Non-ECNU OpenAI-compatible settings (no thinking fields).

    There are no model *tiers* any more: ``"chat"``/``"reasoner"`` label an
    output policy (JSON vs prose), and both request whichever model is
    resolved for the endpoint.
    """
    defaults: dict[str, object] = {
        "api_key": "test-key",
        "base_url": GENERIC_BASE_URL,
        "model": "deepseek-chat",
        "provider": "generic",
        "timeout_s": 30,
        "max_retries": 0,
        "max_concurrent_requests": 1,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)  # type: ignore[arg-type]


# ── Wire-level recorder ───────────────────────────────────────────────────────


@dataclass
class WireRecorder:
    """What the transport saw: parsed bodies plus in-flight bookkeeping."""

    payloads: list[dict[str, Any]] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    peak_in_flight: int = 0
    in_flight: int = 0
    total_requests: int = 0

    @property
    def last(self) -> dict[str, Any]:
        """The most recent request body (raises when nothing was sent)."""
        return self.payloads[-1]

    def fields(self, *names: str) -> list[dict[str, Any]]:
        """Project every recorded body onto *names*, in arrival order."""
        return [{name: payload.get(name) for name in names} for payload in self.payloads]

    def messages(self) -> list[str]:
        """The concatenated message contents of every recorded body."""
        texts: list[str] = []
        for payload in self.payloads:
            for message in payload.get("messages") or []:
                if isinstance(message, dict):
                    content = message.get("content")
                    texts.append(content if isinstance(content, str) else str(content))
        return texts


# ── Response builders ─────────────────────────────────────────────────────────


def chat_completion(
    content: str = "{}",
    *,
    model: str = "ecnu-max",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    reasoning_tokens: int = 0,
) -> httpx.Response:
    """A 200 OpenAI-compatible chat completion carrying *content*."""
    details: dict[str, int] = {}
    if reasoning_tokens:
        details["reasoning_tokens"] = reasoning_tokens
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": details,
        },
    }
    return httpx.Response(200, json=body)


def error_response(status_code: int, *, retry_after: int | None = None) -> httpx.Response:
    """A non-2xx provider response (sanitized body: status only)."""
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
    return httpx.Response(
        status_code,
        headers=headers,
        json={"error": {"message": "injected failure", "type": "test"}},
    )


class LatencyModel:
    """Deterministic per-request service times for the offline experiment.

    Keyed by ``(scenario, role)`` so every concurrency level serves the *same*
    workload: a comparison between limits would otherwise measure the random
    number generator instead of the gate.
    """

    #: Heavy tail: the moderator is the slowest panel call, mirroring observed
    #: campus-gateway behaviour (high reasoning effort + 1600 output tokens).
    _ROLE_FACTOR: dict[str, float] = {
        "analyst": 1.0,
        "cbt": 1.2,
        "tmt": 1.1,
        "emotion": 1.3,
        "moderator": 2.6,
        "critic": 0.8,
    }

    #: Base service time per scenario, in milliseconds.
    _SCENARIO_BASE_MS: tuple[float, ...] = (60.0, 75.0, 90.0, 110.0, 130.0)

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], float] = {}

    def block_seconds(self, scenario: str, role: str, occurrence: int = 0) -> float:
        """Service time for one call — stable for a given key and occurrence."""
        key = (scenario, role)
        if key not in self._cache:
            index = sum(ord(ch) for ch in scenario) % len(self._SCENARIO_BASE_MS)
            self._cache[key] = self._SCENARIO_BASE_MS[index] * self._ROLE_FACTOR.get(
                role, 1.0
            )
        # Retries of the same call get a shorter service time: the provider is
        # recovering, and re-serving the full block would only inflate wall time.
        return self._cache[key] / 1000.0 / (1.0 + occurrence)

    def timeout_seconds(self, scenario: str, role: str) -> float:
        """A simulated deadline that a timeout-injected call cannot beat."""
        return self.block_seconds(scenario, role) + 0.01


# ── The wire ──────────────────────────────────────────────────────────────────


class MockLLMWire:
    """Recorder + both installation seams over one mocked HTTP transport."""

    def __init__(self, handler: Handler) -> None:
        self.recorder = WireRecorder()
        self._handler = handler
        self.singleton = _ForwardingTransport(self._dispatch)
        self.constructions = 0
        self.clients: list[httpx.AsyncClient] = []
        self.attachments = 0

    async def _dispatch(self, request: httpx.Request) -> httpx.Response:
        self._enter()
        try:
            self.recorder.payloads.append(json.loads(request.content or b"{}"))
            self.recorder.urls.append(str(request.url))
            return await self._handler(request)
        finally:
            self.recorder.in_flight -= 1

    def _enter(self) -> None:
        self.recorder.in_flight += 1
        self.recorder.total_requests += 1
        self.recorder.peak_in_flight = max(
            self.recorder.peak_in_flight, self.recorder.in_flight
        )

    def set_handler(self, handler: Handler) -> None:
        """Swap the response policy for a fresh wire (keeps the counters)."""
        self._handler = handler

    def reset_counters(self) -> None:
        """Forget recorded traffic, keeping the installation intact."""
        self.recorder.payloads.clear()
        self.recorder.urls.clear()
        self.recorder.peak_in_flight = 0
        self.recorder.in_flight = 0
        self.recorder.total_requests = 0
        self.constructions = 0
        self.clients.clear()
        self.attachments = 0

    @property
    def constructions_after_patch(self) -> int:
        """Clients built while :meth:`patch_async_client` was active."""
        return self.constructions

    def attach(self, client: httpx.AsyncClient) -> httpx.AsyncClient:
        """Point an already-built client at the mocked wire."""
        client._transport = self.singleton  # noqa: SLF001 - deliberate test seam
        self.attachments += 1
        return client

    @contextmanager
    def patch_async_client(self) -> Iterator[MockLLMWire]:
        """Mock and count every ``httpx.AsyncClient`` built inside the block.

        The block also means "no test inside it may open a real socket": any
        client constructed afterwards is inspected rather than dispatched, which
        is what makes "exactly one client across all six roles" observable
        instead of inferred.
        """
        original_init = httpx.AsyncClient.__init__

        def patched_init(client: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = _ForwardingTransport(self._dispatch)
            kwargs["trust_env"] = False
            original_init(client, *args, **kwargs)
            self.constructions += 1
            self.clients.append(client)

        # ``langchain_openai`` caches built clients process-wide; a cached one
        # would escape the patch and reach the network.
        _client_utils._cached_async_httpx_client.cache_clear()
        _client_utils._cached_sync_httpx_client.cache_clear()
        httpx.AsyncClient.__init__ = patched_init  # type: ignore[method-assign]
        try:
            yield self
        finally:
            httpx.AsyncClient.__init__ = original_init  # type: ignore[method-assign]


class _ForwardingTransport(httpx.AsyncBaseTransport):
    """A transport whose dispatch callable can be chosen after construction.

    ``httpx.AsyncClient._transport`` cannot be replaced by ``MockTransport`` and
    still be the object a test asserts on, so already-built clients are pointed
    at this forwarder instead.
    """

    def __init__(self, dispatch: Handler) -> None:
        self._dispatch = dispatch

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._dispatch(request)


# ── Blocking helpers for concurrency assertions ───────────────────────────────


class ArrivalLatch:
    """Blocks a handler until *target* requests are simultaneously in flight."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.count = 0
        self.release = asyncio.Event()
        self.all_arrived = asyncio.Event()

    async def arrive_and_wait(self) -> int:
        """Register one arrival; return after *target* peers have arrived too."""
        self.count += 1
        index = self.count
        if self.count >= self.target:
            self.all_arrived.set()
        await self.release.wait()
        return index


# ── Policy helpers ────────────────────────────────────────────────────────────


def overrides_for(policy: CompletionPolicy | None, *, provider: str = "ecnu") -> dict[str, Any]:
    """The per-request kwargs a policy resolves to (``{}`` when ``None``)."""
    return {} if policy is None else policy.request_overrides(provider=provider)


def expected_ecnu_fields(policy: CompletionPolicy) -> dict[str, Any]:
    """The wire fields an ECNU policy must produce (effort + output cap)."""
    expected: dict[str, Any] = {}
    if policy.reasoning_effort is not None:
        expected["reasoning_effort"] = policy.reasoning_effort
    if policy.max_output_tokens is not None:
        expected["max_completion_tokens"] = policy.max_output_tokens
    return expected


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "ECNU_BASE_URL",
    "GENERIC_BASE_URL",
    "ArrivalLatch",
    "LatencyModel",
    "MockLLMWire",
    "WireRecorder",
    "chat_completion",
    "error_response",
    "expected_ecnu_fields",
    "make_generic_settings",
    "make_settings",
    "overrides_for",
]
