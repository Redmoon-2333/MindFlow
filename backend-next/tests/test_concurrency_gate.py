"""Regression tests for the shared LLM concurrency gate (optimisation plan 4.4).

The gate exists to keep MindFlow inside the campus gateway's per-user in-flight
cap, so its *statistic* has to be true, not decorative. Every test here drives
real HTTP attempts through :class:`GatedHTTPClient` with an injected transport
whose handler blocks until a set number of requests have genuinely arrived
together, then compares:

* the maximum number of simultaneously in-flight requests the transport
  observed,
* against ``gate.snapshot()["max_in_flight"]`` / ``current_in_flight``,
* and the configured ``limit``.

429s and timeouts are counted in the same statistics via their sanitized
categories (``4xx`` / ``timeout``), because those are the two failure modes the
plan's default-concurrency decision is about.

Offline: injected transport only.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.http_client import (
    GatedHTTPClient,
    HTTPAttemptMetrics,
    collect_http_attempts,
    compute_backoff,
    parse_retry_after,
    status_class,
    transport_error_category,
)
from tests._llm_test_support import ArrivalLatch, MockLLMWire, chat_completion, error_response

_URL = "https://chat.ecnu.edu.cn/open/api/v1/chat/completions"


async def _default_ok(request: httpx.Request) -> httpx.Response:
    """The baseline handler: every attempt succeeds."""
    return chat_completion("{}")


# ── Harness ───────────────────────────────────────────────────────────────────


async def _run_gated(
    wire: MockLLMWire,
    gate: LLMConcurrencyGate,
    *,
    requests: int,
) -> list[httpx.Response]:
    """Fire *requests* concurrent attempts through a gated client."""
    client = GatedHTTPClient(gate, transport=wire.singleton)
    try:
        return list(await asyncio.gather(*(
            client.post(_URL, json={"n": index}) for index in range(requests)
        )))
    finally:
        await client.aclose()


def _blocking_wire(
    *, target: int, status: int = 200,
) -> tuple[MockLLMWire, ArrivalLatch]:
    """A wire whose handler blocks until *target* requests are in flight together."""
    latch = ArrivalLatch(target)

    async def handler(request: httpx.Request) -> httpx.Response:
        await latch.arrive_and_wait()
        if status == 200:
            return chat_completion("{}")
        return error_response(status)

    return MockLLMWire(handler), latch


# ═══════════════════════════════════════════════════════════════════════════════
# The statistic equals reality
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_reported_in_flight_equals_observed_concurrency(limit: int) -> None:
    """Gate's peak in-flight count == the peak the transport actually saw."""
    wire, latch = _blocking_wire(target=limit)
    gate = LLMConcurrencyGate(limit)

    task = asyncio.create_task(_run_gated(wire, gate, requests=limit))
    try:
        await asyncio.wait_for(latch.all_arrived.wait(), timeout=5)
        # Everyone the limit allows is inside the handler right now.
        assert wire.recorder.in_flight == limit
        assert gate.snapshot()["current_in_flight"] == limit
        assert wire.recorder.peak_in_flight == limit
    finally:
        latch.release.set()
        responses = await asyncio.wait_for(task, timeout=5)

    assert [response.status_code for response in responses] == [200] * limit
    snapshot = gate.snapshot()
    assert snapshot["limit"] == limit
    assert snapshot["acquired"] == limit
    assert snapshot["max_in_flight"] == limit
    assert snapshot["current_in_flight"] == 0
    assert wire.recorder.peak_in_flight == snapshot["max_in_flight"]


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_concurrency_never_exceeds_the_configured_limit(limit: int) -> None:
    """More callers than slots: the cap holds and every caller is still served."""
    total = limit * 3
    wire, latch = _blocking_wire(target=limit)
    gate = LLMConcurrencyGate(limit)

    task = asyncio.create_task(_run_gated(wire, gate, requests=total))
    try:
        await asyncio.wait_for(latch.all_arrived.wait(), timeout=5)
        # Give any over-admission a chance to happen before releasing.
        for _ in range(5):
            await asyncio.sleep(0)
        assert wire.recorder.in_flight == limit
        assert wire.recorder.peak_in_flight <= limit
        assert gate.snapshot()["current_in_flight"] <= limit
    finally:
        latch.release.set()
        responses = await asyncio.wait_for(task, timeout=10)

    assert len(responses) == total
    assert all(response.status_code == 200 for response in responses)
    snapshot = gate.snapshot()
    assert snapshot["acquired"] == total
    assert snapshot["max_in_flight"] == limit
    assert wire.recorder.peak_in_flight == limit
    assert wire.recorder.total_requests == total


async def test_serial_execution_at_the_shipped_default() -> None:
    """limit=1 is the shipped default: attempts must not overlap at all."""
    wire, latch = _blocking_wire(target=1)
    gate = LLMConcurrencyGate(1)

    task = asyncio.create_task(_run_gated(wire, gate, requests=4))
    try:
        await asyncio.wait_for(latch.all_arrived.wait(), timeout=5)
        for _ in range(10):
            await asyncio.sleep(0)
        # Three callers are queued behind the one inside the handler.
        assert wire.recorder.in_flight == 1
        assert wire.recorder.total_requests == 1
    finally:
        latch.release.set()
        responses = await asyncio.wait_for(task, timeout=10)

    assert all(response.status_code == 200 for response in responses)
    assert wire.recorder.peak_in_flight == 1
    assert gate.snapshot()["max_in_flight"] == 1


async def test_unbounded_gate_reports_real_parallelism() -> None:
    """``limit=None`` means unbounded — and the statistic still tracks reality."""
    wire, latch = _blocking_wire(target=3)
    gate = LLMConcurrencyGate(None)

    assert gate.limit is None
    task = asyncio.create_task(_run_gated(wire, gate, requests=3))
    try:
        await asyncio.wait_for(latch.all_arrived.wait(), timeout=5)
        assert wire.recorder.in_flight == 3
        assert gate.snapshot()["current_in_flight"] == 3
    finally:
        latch.release.set()
        await asyncio.wait_for(task, timeout=5)

    snapshot = gate.snapshot()
    assert snapshot["limit"] is None
    assert snapshot["max_in_flight"] == 3
    assert snapshot["acquired"] == 3
    assert wire.recorder.peak_in_flight == 3


@pytest.mark.parametrize("limit", [0, -3])
async def test_non_positive_limit_is_unbounded(limit: int) -> None:
    """Offline paths pass 0/-1 to mean "no budget to respect"."""
    gate = LLMConcurrencyGate(limit)
    assert gate.limit is None
    async with gate:
        assert gate.snapshot()["current_in_flight"] == 1
    assert gate.snapshot()["current_in_flight"] == 0


async def test_acquire_context_manager_matches_aenter() -> None:
    """The ``acquire()`` form must bookkeep identically to ``async with gate``."""
    wire = MockLLMWire(_default_ok)
    direct = LLMConcurrencyGate(1)
    contextual = LLMConcurrencyGate(1)

    await _run_gated(wire, direct, requests=2)
    client = GatedHTTPClient(contextual, transport=wire.singleton)
    try:
        async with contextual.acquire():
            assert contextual.snapshot()["current_in_flight"] == 1
        await client.post(_URL, json={})
    finally:
        await client.aclose()

    assert contextual.snapshot() == direct.snapshot()


async def test_waiting_callers_are_not_counted_as_in_flight() -> None:
    """Queued callers wait *outside* the counter, so p50 queue latency is real."""
    wire, latch = _blocking_wire(target=1)
    gate = LLMConcurrencyGate(1)

    task = asyncio.create_task(_run_gated(wire, gate, requests=3))
    try:
        await asyncio.wait_for(latch.all_arrived.wait(), timeout=5)
        for _ in range(10):
            await asyncio.sleep(0)
        snapshot = gate.snapshot()
        assert snapshot["acquired"] == 1, "only one caller has entered"
        assert snapshot["current_in_flight"] == 1
        assert wire.recorder.total_requests == 1, "the other two never reached HTTP"
    finally:
        latch.release.set()
        await asyncio.wait_for(task, timeout=10)

    assert wire.recorder.total_requests == 3


async def test_gate_slot_is_released_when_a_caller_is_cancelled() -> None:
    """A cancelled waiter must not consume a slot forever."""
    wire, latch = _blocking_wire(target=1)
    gate = LLMConcurrencyGate(1)

    running = asyncio.create_task(_run_gated(wire, gate, requests=2))
    await asyncio.wait_for(latch.all_arrived.wait(), timeout=5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    # The transport handler is unblocked and the gate's counter is back to zero.
    latch.release.set()
    for _ in range(10):
        await asyncio.sleep(0)
    assert gate.snapshot()["current_in_flight"] == 0
    assert gate.snapshot()["acquired"] >= 1

    # And the gate is usable again.
    wire2, latch2 = _blocking_wire(target=1)
    latch2.release.set()
    responses = await _run_gated(wire2, gate, requests=1)
    assert responses[0].status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# 429s and timeouts are counted
# ═══════════════════════════════════════════════════════════════════════════════


async def test_rate_limit_responses_are_recorded_with_their_status_class() -> None:
    """A 429 still consumes and releases a slot, and is visible as ``4xx``."""
    wire, latch = _blocking_wire(target=2, status=429)
    gate = LLMConcurrencyGate(2)

    with collect_http_attempts() as attempts:
        task = asyncio.create_task(_run_gated(wire, gate, requests=2))
        try:
            await asyncio.wait_for(latch.all_arrived.wait(), timeout=5)
            assert gate.snapshot()["current_in_flight"] == 2
        finally:
            latch.release.set()
            responses = await asyncio.wait_for(task, timeout=5)

    assert [response.status_code for response in responses] == [429, 429]
    assert [attempt.status_class for attempt in attempts] == ["4xx", "4xx"]
    assert all(attempt.error_category == "" for attempt in attempts)
    snapshot = gate.snapshot()
    assert snapshot["acquired"] == 2
    assert snapshot["max_in_flight"] == 2
    assert snapshot["current_in_flight"] == 0


async def test_timeouts_are_counted_and_the_slot_is_returned() -> None:
    """A timed-out attempt is categorised ``timeout`` and does not hold a slot."""
    wire = MockLLMWire(_default_ok)
    gate = LLMConcurrencyGate(1)

    async def timing_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("injected timeout", request=request)

    wire.set_handler(timing_out)
    with collect_http_attempts() as attempts, pytest.raises(httpx.ReadTimeout):
        await _run_gated(wire, gate, requests=1)

    assert len(attempts) == 1
    assert attempts[0].error_category == "timeout"
    assert attempts[0].status_class == ""
    snapshot = gate.snapshot()
    assert snapshot["acquired"] == 1
    assert snapshot["current_in_flight"] == 0, "a timeout must not leak its slot"

    # The gate is immediately reusable after a timeout.
    wire.set_handler(_default_ok)
    responses = await _run_gated(wire, gate, requests=2)
    assert [response.status_code for response in responses] == [200, 200]
    assert gate.snapshot()["acquired"] == 3


async def test_mixed_outcomes_share_one_gate_budget() -> None:
    """429 + timeout + success through one gate: counters stay consistent."""
    wire = MockLLMWire(_default_ok)
    outcomes = {"429": error_response(429), "timeout": None, "ok": chat_completion("{}")}
    index = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        index["n"] += 1
        order = ("429", "timeout", "ok")
        kind = order[(index["n"] - 1) % len(order)]
        if kind == "timeout":
            raise httpx.ConnectTimeout("injected timeout", request=request)
        return outcomes[kind]

    wire.set_handler(handler)
    gate = LLMConcurrencyGate(1)

    with collect_http_attempts() as attempts:
        results = await asyncio.gather(
            *(_one_attempt(wire, gate) for _ in range(6)),
            return_exceptions=True,
        )

    assert [attempt.status_class for attempt in attempts] == ["4xx", "", "2xx"] * 2
    assert [attempt.error_category for attempt in attempts] == ["", "timeout", ""] * 2
    assert sum(isinstance(r, httpx.TimeoutException) for r in results) == 2
    snapshot = gate.snapshot()
    assert snapshot["acquired"] == 6
    assert snapshot["max_in_flight"] == 1
    assert snapshot["current_in_flight"] == 0


async def _one_attempt(wire: MockLLMWire, gate: LLMConcurrencyGate) -> httpx.Response:
    client = GatedHTTPClient(gate, transport=wire.singleton)
    try:
        return await client.post(_URL, json={})
    finally:
        await client.aclose()


# ═══════════════════════════════════════════════════════════════════════════════
# Queue vs HTTP split, and the sanitizers the counters depend on
# ═══════════════════════════════════════════════════════════════════════════════


async def test_queue_latency_is_measured_separately_from_http_latency() -> None:
    """A serialised attempt spends real time waiting; that time is attributed."""
    wire, latch = _blocking_wire(target=1)
    gate = LLMConcurrencyGate(1)

    with collect_http_attempts() as attempts:
        task = asyncio.create_task(_run_gated(wire, gate, requests=2))
        await asyncio.wait_for(latch.all_arrived.wait(), timeout=5)
        await asyncio.sleep(0.05)  # the second caller waits this long
        latch.release.set()
        await asyncio.wait_for(task, timeout=10)

    assert len(attempts) == 2
    assert attempts[0].queue_latency_ms < attempts[1].queue_latency_ms
    assert attempts[1].queue_latency_ms >= 50.0
    assert attempts[1].http_latency_ms >= 0.0


def test_status_class_and_error_category_are_sanitized() -> None:
    """Only coarse classes leave the transport layer — never a status body."""
    assert status_class(200) == "2xx"
    assert status_class(429) == "4xx"
    assert status_class(503) == "5xx"
    assert status_class(99) == ""
    assert status_class(600) == ""

    assert transport_error_category(httpx.ReadTimeout("x")) == "timeout"
    assert transport_error_category(httpx.ConnectError("x")) == "network"
    assert transport_error_category(ValueError("x")) == "error"
    request = httpx.Request("POST", _URL)
    response = httpx.Response(500, request=request)
    assert transport_error_category(
        httpx.HTTPStatusError("x", request=request, response=response)
    ) == "http_error"


def test_backoff_arithmetic_honours_retry_after_and_the_cap() -> None:
    """429 ``Retry-After`` beats jitter, and neither exceeds the ceiling."""
    assert parse_retry_after(httpx.Response(429, headers={"Retry-After": "7"})) == 7
    assert parse_retry_after(httpx.Response(429)) is None
    assert parse_retry_after(httpx.Response(429, headers={"Retry-After": "soon"})) is None

    assert compute_backoff(0, retry_after=7) == 7.0
    assert compute_backoff(0, retry_after=10_000) == 60.0  # capped
    for attempt in range(6):
        delay = compute_backoff(attempt)
        assert 0.0 <= delay <= 60.0
        assert delay >= min(float(2**attempt), 60.0)


def test_attempt_metrics_defaults_are_empty() -> None:
    """A metrics row carries no URL, headers, or body — only numbers and classes."""
    metrics = HTTPAttemptMetrics()
    assert metrics.queue_latency_ms == 0.0
    assert metrics.http_latency_ms == 0.0
    assert metrics.status_class == ""
    assert metrics.error_category == ""
