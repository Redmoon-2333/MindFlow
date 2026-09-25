"""Regression tests for the shared L2 (Ollama) HTTP client lifecycle (plan 4.3).

Before this change the fallback tier built — and closed — a fresh
``httpx.AsyncClient`` per call, i.e. one connection pool per degraded request.
The contract these tests pin down:

* one construction across many calls, one close at shutdown (idempotent);
* the *same* client object is handed out, so keep-alive connections are reused
  rather than re-established;
* ``graph/fallback_nodes.py``'s ``ollama_node`` resolves that shared client
  instead of constructing one per call, and prefers an explicitly injected one;
* closing the owner really closes the pool (no leaked sockets), and a closed
  pool refuses to hand out a client rather than silently rebuilding one.

Offline: the wire is mocked, so no local Ollama server is required.
"""

from __future__ import annotations

import httpx
import pytest

from mindflow.config import LLMSettings
from mindflow.graph import fallback_nodes
from mindflow.graph.fallback_nodes import (
    FallbackRunContext,
    _ollama_api_call,
    _ollama_transport,
    ollama_node,
)
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.http_client import (
    GatedHTTPClient,
    ProviderHTTPClient,
    SharedHTTPClientPool,
    clear_shared_fallback_pool,
    install_shared_fallback_pool,
    shared_fallback_pool,
)
from mindflow.infrastructure.provider_registry import ProviderRegistry
from tests._llm_test_support import (
    MockLLMWire,
    chat_completion,
    error_response,
    make_settings,
)

_VALID_OLLAMA_CONTENT = (
    '{"procrastination_types": ["impulsivity"], "type_confidence": {"impulsivity": 0.7},'
    ' "cognitive_distortions": [], "cbt_technique": "stimulus_control",'
    ' "response_text": "先做五分钟。", "next_action": "打开文档"}'
)


@pytest.fixture(autouse=True)
def _restore_process_pool_slot() -> None:
    """Leave the process-wide fallback slot exactly as the test found it."""
    previous = shared_fallback_pool()
    yield
    clear_shared_fallback_pool()
    if previous is not None:
        install_shared_fallback_pool(previous)


def _wire(content: str = _VALID_OLLAMA_CONTENT) -> MockLLMWire:
    async def handler(request: httpx.Request) -> httpx.Response:
        return chat_completion(content, model="qwen3:8b", prompt_tokens=40, completion_tokens=20)

    return MockLLMWire(handler)


def _pool(
    wire: MockLLMWire,
    *,
    gate: LLMConcurrencyGate | None = None,
    name: str = "test-ollama",
) -> SharedHTTPClientPool:
    return SharedHTTPClientPool(
        gate=gate if gate is not None else LLMConcurrencyGate(1),
        timeout_s=30.0,
        max_retries=0,
        name=name,
    )


def _attach(wire: MockLLMWire, pool: SharedHTTPClientPool) -> httpx.AsyncClient:
    """Point the pool's lazily built client at the mocked wire (once)."""
    client = wire.attach(pool.client)
    return client


# ═══════════════════════════════════════════════════════════════════════════════
# One construction, one close
# ═══════════════════════════════════════════════════════════════════════════════


async def test_many_calls_construct_the_client_exactly_once() -> None:
    """Twenty fallback calls share one pool — the pre-4.3 behaviour was twenty."""
    wire = _wire()
    pool = _pool(wire)
    client = _attach(wire, pool)

    for _ in range(20):
        assert pool.client is client
        result = await _ollama_api_call(
            "http://localhost:11434", "qwen3:8b", "{}",
            client=pool.client, max_retries=pool.max_retries,
        )
        assert result.cbt_technique == "stimulus_control"

    assert pool.constructions == 1
    assert pool.closes == 0
    assert wire.recorder.total_requests == 20
    assert wire.constructions_after_patch == 0


async def test_local_fallback_ignores_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inherited proxy must not intercept the local Ollama transport."""
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    wire = _wire()
    pool = _pool(wire)
    try:
        client = _attach(wire, pool)
        result = await _ollama_api_call(
            "http://localhost:11434", "qwen3:8b", "{}",
            client=client, max_retries=0,
        )
        assert result.cbt_technique == "stimulus_control"
        assert wire.recorder.total_requests == 1
    finally:
        await pool.aclose()


async def test_client_is_constructed_lazily() -> None:
    """A process that never degrades to L2 must not open a pool at all."""
    wire = _wire()
    pool = _pool(wire)

    assert pool.peek() is None
    assert pool.constructions == 0
    assert wire.attachments == 0
    assert "constructions" in pool.snapshot()


async def test_shutdown_closes_the_owned_client_exactly_once() -> None:
    """``aclose`` is idempotent: the pool is released once, never twice."""
    wire = _wire()
    pool = _pool(wire)
    client = _attach(wire, pool)

    assert client.is_closed is False
    await pool.aclose()
    assert client.is_closed is True
    assert pool.closes == 1
    assert pool.peek() is None

    await pool.aclose()
    await pool.aclose()
    assert pool.closes == 1, "double shutdown must not double-close the pool"
    assert pool.snapshot()["closed"] is True


async def test_closed_pool_refuses_to_hand_out_a_client() -> None:
    """A closed owner must fail loudly rather than quietly build a new pool."""
    wire = _wire()
    pool = _pool(wire)
    _attach(wire, pool)
    await pool.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        _ = pool.client
    assert pool.constructions == 1


async def test_close_before_first_use_is_a_noop() -> None:
    wire = _wire()
    pool = _pool(wire)
    await pool.aclose()  # never constructed anything
    assert pool.closes == 0
    assert pool.snapshot()["closed"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Registry ownership
# ═══════════════════════════════════════════════════════════════════════════════


async def test_registry_owns_one_pool_and_shares_it_with_the_graph() -> None:
    """The graph resolves the registry's pool — the connection reuse seam."""
    registry = ProviderRegistry(make_settings())
    try:
        assert registry.fallback_pool is shared_fallback_pool()
        first = registry.get_fallback_client()
        for _ in range(10):
            assert registry.get_fallback_client() is first
    finally:
        await registry.shutdown()

    assert registry.fallback_pool.constructions == 1
    assert registry.fallback_pool.closes == 1
    assert first.is_closed is True


async def test_registry_publishes_the_pool_for_registry_less_consumers() -> None:
    """``AnalysisGraph`` builds its runtime from settings, never seeing the registry."""
    registry = ProviderRegistry(make_settings())
    try:
        assert shared_fallback_pool() is registry.fallback_pool
        assert shared_fallback_pool() is not None
    finally:
        await registry.shutdown()
    assert shared_fallback_pool() is None, "a closed pool must not stay published"


async def test_shutdown_does_not_detach_a_newer_registry() -> None:
    """A late shutdown of an old registry must not unpublish the live one."""
    old = ProviderRegistry(make_settings())
    new = ProviderRegistry(make_settings())
    try:
        assert shared_fallback_pool() is new.fallback_pool
        await old.shutdown()  # late, out-of-order shutdown
        assert shared_fallback_pool() is new.fallback_pool
    finally:
        await new.shutdown()
    assert shared_fallback_pool() is None


async def test_registry_describe_reports_pool_lifecycle_evidence() -> None:
    """The diagnostics view exposes construction/close counts, never URLs."""
    registry = ProviderRegistry(make_settings())
    try:
        assert registry.get_fallback_client() is registry.fallback_pool.peek()
        before = registry.describe()["fallback_pool"]
        assert isinstance(before, dict)
        assert before["constructions"] == 1
        assert before["closes"] == 0
        assert before["closed"] is False
    finally:
        await registry.shutdown()

    after = registry.describe()["fallback_pool"]
    assert isinstance(after, dict)
    assert after["constructions"] == 1
    assert after["closes"] == 1
    assert after["closed"] is True
    assert after["name"] == "ollama-fallback"
    assert "headers" not in after and "base_url" not in after


# ═══════════════════════════════════════════════════════════════════════════════
# The fallback graph node uses the shared client
# ═══════════════════════════════════════════════════════════════════════════════


async def test_ollama_node_uses_the_shared_client_not_a_per_call_one() -> None:
    """Three node invocations must not build a second HTTP client.

    The pool's own lazy construction is allowed (it is the one shared client);
    what must not happen is one construction *per call*.
    """
    wire = _wire()
    pool = _pool(wire, name="ollama-fallback")
    install_shared_fallback_pool(pool)

    with wire.patch_async_client():
        for _ in range(3):
            state = await ollama_node({
                "summary_json": "{}",
                "runtime": FallbackRunContext(ollama_base_url="http://localhost:11434"),
            })
            assert state["source"] == "ollama"
        shared = pool.client
        built_inside = wire.constructions_after_patch

    assert built_inside == 1, "the shared pool builds its client exactly once"
    assert pool.constructions == 1
    assert len(wire.clients) == 1
    assert wire.clients[0] is shared, "the node must call the pooled client"
    assert wire.recorder.total_requests == 3
    await pool.aclose()


async def test_ollama_node_prefers_an_injected_client() -> None:
    """Tests and tools may inject their own client; the node must honour it."""
    wire = _wire()
    pool = _pool(wire)
    install_shared_fallback_pool(pool)
    injected = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: chat_completion(_VALID_OLLAMA_CONTENT)),
    )
    try:
        state = await ollama_node({
            "summary_json": "{}",
            "runtime": FallbackRunContext(
                ollama_base_url="http://localhost:11434",
                ollama_client=injected,
                ollama_max_retries=0,
            ),
        })
        assert state["source"] == "ollama"
        assert pool.constructions == 0, "an injected client must win over the pool"
    finally:
        await injected.aclose()
        await pool.aclose()


async def test_ollama_node_without_registry_builds_one_local_pool() -> None:
    """A registry-less process (unit tests, scripts) still builds only one pool."""
    wire = _wire()
    clear_shared_fallback_pool()
    original = fallback_nodes._local_fallback_pool
    fallback_nodes._local_fallback_pool = None
    try:
        with wire.patch_async_client():
            for _ in range(3):
                state = await ollama_node({
                    "summary_json": "{}",
                    "runtime": FallbackRunContext(ollama_base_url="http://localhost:11434"),
                })
                assert state["source"] == "ollama"
            local = fallback_nodes._local_fallback_pool
            assert local is not None
            assert local.constructions == 1
            assert wire.constructions_after_patch == 1, "one pool, not one per call"
            assert wire.recorder.total_requests == 3
            assert wire.clients[0] is local.peek()
        await local.aclose()
    finally:
        fallback_nodes._local_fallback_pool = original


def test_ollama_transport_precedence() -> None:
    """injected client → registry pool → process-local pool."""
    injected = httpx.AsyncClient()
    runtime = FallbackRunContext(
        ollama_base_url="http://localhost:11434",
        ollama_client=injected,
        ollama_max_retries=4,
    )
    client, retries, _, owner = _ollama_transport(runtime)
    assert client is injected
    assert retries == 4
    assert owner == "injected"

    pool = _pool(_wire(), name="registry-pool")
    install_shared_fallback_pool(pool)
    try:
        client, retries, cap, owner = _ollama_transport(FallbackRunContext())
        assert client is pool.client
        assert owner == "registry-pool"
        assert retries == pool.max_retries
        assert cap == pool.backoff_cap_s
    finally:
        clear_shared_fallback_pool()


# ═══════════════════════════════════════════════════════════════════════════════
# The shared client is gated, so the fallback tier respects the same budget
# ═══════════════════════════════════════════════════════════════════════════════


def test_pool_builds_a_gated_client_with_unified_policy() -> None:
    """The owned client carries the gate, the timeout and the pool limits."""
    gate = LLMConcurrencyGate(2)
    pool = SharedHTTPClientPool(
        gate=gate, timeout_s=45.0, max_retries=3, backoff_cap_s=12.0, name="p",
    )
    client = pool.client

    assert isinstance(client, GatedHTTPClient)
    assert not isinstance(client, ProviderHTTPClient), "L2 must not inherit ECNU rewriting"
    assert client.timeout == httpx.Timeout(45.0)
    assert gate.limit == 2
    assert pool.timeout_s == 45.0
    assert pool.max_retries == 3
    assert pool.backoff_cap_s == 12.0


async def test_pool_client_acquires_the_shared_gate_per_attempt() -> None:
    """Every L2 attempt takes a slot from the same gate the panel uses."""
    wire = _wire()
    gate = LLMConcurrencyGate(1)
    pool = _pool(wire, gate=gate)
    _attach(wire, pool)

    before = gate.snapshot()
    await _ollama_api_call(
        "http://localhost:11434", "qwen3:8b", "{}", client=pool.client, max_retries=0,
    )
    after = gate.snapshot()

    assert after["acquired"] == before["acquired"] + 1  # type: ignore[operator]
    assert after["current_in_flight"] == 0
    assert after["max_in_flight"] == 1


async def test_retries_reuse_the_same_client_and_pool() -> None:
    """A retried L2 call must not build a second client for the second attempt."""
    attempts = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return error_response(503, retry_after=0)
        return chat_completion(_VALID_OLLAMA_CONTENT, model="qwen3:8b")

    wire = MockLLMWire(handler)
    pool = _pool(wire)
    client = _attach(wire, pool)
    monkey = fallback_nodes.compute_backoff
    fallback_nodes.compute_backoff = lambda attempt, retry_after=None: 0.0  # type: ignore[assignment]
    try:
        result = await _ollama_api_call(
            "http://localhost:11434", "qwen3:8b", "{}",
            client=client, max_retries=1, backoff_cap_s=pool.backoff_cap_s,
        )
    finally:
        fallback_nodes.compute_backoff = monkey  # type: ignore[assignment]

    assert result.cbt_technique == "stimulus_control"
    assert wire.recorder.total_requests == 2
    assert pool.constructions == 1
    await pool.aclose()


async def test_gated_client_returns_the_slot_after_a_transport_error() -> None:
    """A failed attempt must not leak its gate slot (else L2 deadlocks itself)."""
    wire = _wire()
    gate = LLMConcurrencyGate(1)
    pool = SharedHTTPClientPool(gate=gate, name="erroring")
    wire.attach(pool.client)

    async def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("injected connection failure", request=request)

    wire.set_handler(failing)

    with pytest.raises(httpx.HTTPError):
        await _ollama_api_call(
            "http://localhost:11434", "qwen3:8b", "{}", client=pool.client, max_retries=0,
        )

    snapshot = gate.snapshot()
    assert snapshot["current_in_flight"] == 0
    assert snapshot["acquired"] == 1
    assert wire.recorder.total_requests == 1
    await pool.aclose()


def test_registry_settings_carry_the_concurrency_default() -> None:
    """The shipped default stays 1 until the experiment says otherwise."""
    assert LLMSettings().max_concurrent_requests == 1
    registry = ProviderRegistry(make_settings(max_concurrent_requests=3))
    try:
        assert registry.concurrency.limit == 3
        assert registry.describe()["concurrency_limit"] == 3
    finally:
        registry.concurrency = LLMConcurrencyGate(1)


async def test_shutdown_closes_the_fallback_pool_once_even_if_never_used() -> None:
    registry = ProviderRegistry(make_settings())
    await registry.shutdown()
    await registry.shutdown()
    assert registry.fallback_pool.constructions == 0
    assert registry.fallback_pool.closes == 0
