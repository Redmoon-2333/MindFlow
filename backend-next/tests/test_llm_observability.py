"""Regression tests for aggregated LLM request observability (plan 2.5).

Two contracts are pinned here, and the second one is the important one:

1. **Facts** — a record is emitted for success, for a retried request, for a
   parse failure, and for a fallback, with the documented scalar fields present
   and the p50/p95/mean/failure-rate aggregates arithmetically exact on a fixed
   sample.
2. **Privacy** — *no* allow-listed field can carry prompt text, an API key, or a
   provider response body. Prompts and keys are pushed through the real call
   path (they are in the request), hostile strings are pushed through the
   response body and through every label field, and the recorded metadata plus
   the snapshot are then scanned for the sentinels.

Offline: injected transport only.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import httpx
import pytest

from mindflow.agents.llm_gateway import GatewayAPIError, LangChainGateway
from mindflow.agents.policies import ANALYST_POLICY, MODERATOR_POLICY
from mindflow.infrastructure.llm.http_client import HTTPAttemptMetrics, collect_http_attempts
from mindflow.services.llm_observability import (
    ALLOWED_FIELDS,
    GroupStats,
    LLMObservabilityAggregator,
    LLMRequestRecord,
    current_llm_labels,
    default_aggregator,
    llm_call_context,
    llm_observability_snapshot,
    percentile,
    record_llm_outcome,
    reset_llm_observability,
    summarise_attempts,
)
from tests._llm_test_support import (
    ECNU_BASE_URL,
    MockLLMWire,
    chat_completion,
    error_response,
    make_settings,
)

#: Strings that must never survive into recorded metadata.
PROMPT_SENTINEL = "SENTINEL_PROMPT_不要泄漏_用户窗口标题.docx"
KEY_SENTINEL = "SENTINEL_API_KEY_sk-live-0000"
#: A realistic-length opaque key: no label field can hold it whole (limit 64).
LONG_KEY_SENTINEL = "SENTINEL_LONG_API_KEY_sk-live-" + "0" * 40
BODY_SENTINEL = "SENTINEL_PROVIDER_BODY_reasoning_content"
FORBIDDEN_SENTINEL = "SENTINEL_FORBIDDEN_诊断"


@pytest.fixture(autouse=True)
def _clean_default_aggregator() -> None:
    """The default aggregator is process-wide; never leak records into the suite."""
    reset_llm_observability()
    yield
    reset_llm_observability()


def _gateway(
    wire: MockLLMWire,
    *,
    observability: LLMObservabilityAggregator | None = None,
    **setting_overrides: Any,
) -> LangChainGateway:
    return LangChainGateway(
        api_key=KEY_SENTINEL,
        base_url=ECNU_BASE_URL,
        llm_settings=make_settings(**setting_overrides),
        observability=observability,
    )


def _ok_wire(content: str = "{}") -> MockLLMWire:
    async def handler(request: httpx.Request) -> httpx.Response:
        return chat_completion(content, prompt_tokens=100, completion_tokens=50)

    return MockLLMWire(handler)


# ═══════════════════════════════════════════════════════════════════════════════
# 1a. A record per outcome
# ═══════════════════════════════════════════════════════════════════════════════


async def test_success_emits_one_record_with_documented_fields() -> None:
    """A successful panel call lands as one record: labels, tokens, latency, 2xx."""
    wire = _ok_wire()
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator)

    with wire.patch_async_client(), llm_call_context(
        graph="panel", node=ANALYST_POLICY.node, role=ANALYST_POLICY.role,
    ):
        await gateway.complete("system prompt", "user prompt", policy=ANALYST_POLICY)
        await gateway.close()

    records = aggregator.records()
    assert len(records) == 1
    record = records[0]
    assert record.graph == "panel"
    assert record.node == "analyst"
    assert record.role == "analyst"
    assert record.provider == "ecnu"
    # The tier no longer selects a provider model: every tier records the
    # resolved model id (``deepseek-flash`` under the production pin).
    assert record.model == "ecnu-max"
    assert record.http_status_class == "2xx"
    assert record.input_tokens == 100
    assert record.output_tokens == 50
    assert record.retry_count == 0
    assert record.ok is True
    assert record.failed is False
    assert record.fallback_reason == ""
    assert record.parse_failure is False
    assert record.total_latency_ms > 0.0
    assert record.http_latency_ms > 0.0
    assert record.queue_latency_ms >= 0.0
    assert record.recorded_at > 0.0
    # The documented scalar field set is exactly ALLOWED_FIELDS.
    assert set(record.as_dict()) == ALLOWED_FIELDS


async def test_policy_supplies_role_and_node_when_the_graph_does_not() -> None:
    """A caller without an explicit label context still groups correctly."""
    wire = _ok_wire()
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator)

    with wire.patch_async_client():
        await gateway.complete("s", "u", policy=MODERATOR_POLICY)
        await gateway.close()

    record = aggregator.records()[0]
    assert record.role == "moderator"
    assert record.node == "moderator"


async def test_retry_is_recorded_with_the_retry_count_and_last_status() -> None:
    """A retried request is one record: retry_count=1, last attempt's status."""
    attempts = {"n": 0}

    async def flaky(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return error_response(503)
        return chat_completion("{}")

    wire = MockLLMWire(flaky)
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator, max_retries=1)

    with wire.patch_async_client():
        await gateway.complete("s", "u", policy=ANALYST_POLICY)
        await gateway.close()

    assert attempts["n"] == 2
    records = aggregator.records()
    assert len(records) == 1, "a logical request is one record, not one per attempt"
    record = records[0]
    assert record.retry_count == 1
    assert record.ok is True
    assert record.http_status_class == "2xx", "the deciding attempt's class"
    assert record.failed is False
    # Both attempts' HTTP time is summed: the retried request really spent it.
    assert record.http_latency_ms > 0.0


async def test_transport_failure_is_recorded_as_not_ok() -> None:
    """An exhausted retry budget records ok=False plus the sanitized category."""
    async def always_503(request: httpx.Request) -> httpx.Response:
        return error_response(503)

    wire = MockLLMWire(always_503)
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator, max_retries=0)

    with wire.patch_async_client():
        with pytest.raises(GatewayAPIError):
            await gateway.complete("s", "u", policy=ANALYST_POLICY)
        await gateway.close()

    record = aggregator.records()[0]
    assert record.ok is False
    assert record.failed is True
    assert record.retry_count == 0
    assert record.http_status_class == "5xx"
    assert record.fallback_reason == "exhausted"


async def test_non_retriable_error_is_recorded_without_burning_retries() -> None:
    """A 401 is permanent: one attempt, a named reason, no retry count."""
    attempts = {"n": 0}

    async def unauthorized(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return error_response(401)

    wire = MockLLMWire(unauthorized)
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator, max_retries=3)

    with wire.patch_async_client():
        with pytest.raises(GatewayAPIError):
            await gateway.complete("s", "u", policy=ANALYST_POLICY)
        await gateway.close()

    assert attempts["n"] == 1, "a 401 must not be retried"
    record = aggregator.records()[0]
    assert record.ok is False
    assert record.fallback_reason == "non_retriable_error"
    assert record.http_status_class == "4xx"


async def test_timeout_is_recorded_with_the_timeout_category() -> None:
    async def timing_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("injected", request=request)

    wire = MockLLMWire(timing_out)
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator, max_retries=0)

    with wire.patch_async_client():
        with pytest.raises(GatewayAPIError):
            await gateway.complete("s", "u", policy=ANALYST_POLICY)
        await gateway.close()

    record = aggregator.records()[0]
    assert record.ok is False
    assert record.error_category == "timeout"
    assert record.http_status_class == ""


async def test_parse_failure_is_recorded_when_the_caller_cannot_parse() -> None:
    """A 200 with an unparseable body is a *parse* failure, not a transport one.

    The transport sees a healthy request; the graph knows the payload was
    useless. ``record_llm_outcome`` is the entry point for that fact, and this is
    exactly the shape the panel uses.
    """
    wire = _ok_wire("not json at all")
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator)

    with wire.patch_async_client(), llm_call_context(graph="panel", node="critic", role="critic"):
        raw = await gateway.complete("s", "u", policy=ANALYST_POLICY)
        await gateway.close()

    import json

    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    aggregator.record(LLMRequestRecord(
        graph="panel",
        node=current_llm_labels()[1],
        role="critic",
        parse_failure=True,
        fallback_reason="json_decode_error",
    ))

    transport_record, parse_record = aggregator.records()
    assert transport_record.ok is True and transport_record.parse_failure is False
    assert parse_record.parse_failure is True
    assert parse_record.failed is True
    snapshot = aggregator.snapshot()
    assert snapshot["total"]["parse_failure_rate"] == 0.5
    assert snapshot["total"]["failure_rate"] == 0.5


def test_fallback_and_forbidden_word_outcomes_are_recordable() -> None:
    """Decision-level facts ride the allow-list, and typos are rejected."""
    record = record_llm_outcome(
        graph="fallback",
        node="rule_engine",
        role="rule_engine",
        provider="rule_engine",
        fallback_reason="ollama_failure",
        final_source="rule_engine",
        forbidden_word_failure=True,
        critic_approved=False,
    )
    assert record.fallback_reason == "ollama_failure"
    assert record.final_source == "rule_engine"
    assert record.forbidden_word_failure is True
    assert record.failed is True
    assert record.is_event is True, "no measured request of its own"
    assert record.critic_approved is False

    with pytest.raises(TypeError, match="unknown observability field"):
        record_llm_outcome(prompt="leak me")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        record_llm_outcome(fallback_reason="x", api_key=KEY_SENTINEL)  # type: ignore[call-arg]


async def test_default_aggregator_receives_gateway_records() -> None:
    """A gateway built without an injected aggregator feeds the process one."""
    wire = _ok_wire()
    gateway = _gateway(wire)  # no observability argument

    with wire.patch_async_client(), llm_call_context(graph="chat", node="chat", role="chat"):
        await gateway.complete("s", "u")
        await gateway.close()

    assert gateway._observability is default_aggregator()
    snapshot = llm_observability_snapshot()
    assert snapshot["total"]["requests"] == 1
    assert snapshot["by_graph"]["chat"]["requests"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. Aggregation is arithmetically exact on a fixed sample
# ═══════════════════════════════════════════════════════════════════════════════

_SAMPLE_LATENCIES = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]


def _sample_aggregator(**overrides: Any) -> LLMObservabilityAggregator:
    """Ten analyst + two moderator records with hand-computable statistics."""
    aggregator = LLMObservabilityAggregator()
    for index, latency in enumerate(_SAMPLE_LATENCIES):
        aggregator.record(LLMRequestRecord(
            graph="panel",
            node="analyst",
            role="analyst",
            provider="ecnu",
            model="ecnu-max",
            total_latency_ms=latency,
            http_latency_ms=latency / 2.0,
            queue_latency_ms=latency / 2.0,
            input_tokens=100,
            output_tokens=20,
            reasoning_tokens=5,
            retry_count=0,
            http_status_class="2xx",
            ok=True,
        ))
        # Every other analyst request fails to parse.
        if index % 2 == 0:
            aggregator.record(LLMRequestRecord(
                graph="panel",
                node="analyst",
                role="analyst",
                parse_failure=True,
                total_latency_ms=latency,
                ok=True,
            ))
    # Two moderator requests, one of which was retried once.
    aggregator.record(LLMRequestRecord(
        graph="panel",
        node="moderator",
        role="moderator",
        total_latency_ms=200.0,
        input_tokens=1000,
        output_tokens=100,
        reasoning_tokens=50,
        retry_count=0,
        ok=True,
        **overrides,
    ))
    aggregator.record(LLMRequestRecord(
        graph="panel",
        node="moderator",
        role="moderator",
        total_latency_ms=400.0,
        input_tokens=1000,
        output_tokens=100,
        retry_count=1,
        http_status_class="5xx",
        ok=False,
        fallback_reason="exhausted",
        **overrides,
    ))
    return aggregator


def test_percentile_matches_linear_interpolation() -> None:
    """The percentile helper is the definition the plan's p50/p95 rely on."""
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 0.5) == 30.0
    # position = (5-1)*0.95 = 3.8 -> 40 + (50-40)*0.8 = 48
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 0.95) == 48.0
    # position = (10-1)*0.95 = 8.55 -> 90 + (100-90)*0.55 = 95.5
    assert percentile(_SAMPLE_LATENCIES, 0.95) == pytest.approx(95.5)
    assert percentile(_SAMPLE_LATENCIES, 0.5) == 55.0
    # Unsorted input is fine, and the quantile is clamped to [0, 1].
    assert percentile([50.0, 10.0, 30.0], 0.5) == 30.0
    assert percentile([10.0, 20.0], 2.0) == 20.0
    assert percentile([10.0, 20.0], -1.0) == 10.0


def test_group_stats_are_exact_on_a_fixed_sample() -> None:
    """Every aggregate is asserted against a hand-computed value."""
    snapshot = _sample_aggregator().snapshot()

    analyst: GroupStats = snapshot["by_role"]["analyst"]
    # Ten measured requests (one per sample latency) plus five parse failures.
    assert analyst["count"] == 15
    assert analyst["requests"] == 15
    # Five parse failures out of fifteen.
    assert analyst["parse_failure_rate"] == 5 / 15
    assert analyst["failure_rate"] == 5 / 15
    assert analyst["forbidden_word_failure_rate"] == 0.0
    # Latencies: 10..100 once each, with 10/30/50/70/90 repeated by their parse
    # failures. Sorted: 10,10,20,30,30,40,50,50,60,70,70,80,90,90,100.
    # p50 -> position (15-1)*0.5 = 7 -> 50; p95 -> position 13.3 -> 90+10*0.3 = 93.
    assert analyst["p50_latency_ms"] == 50.0
    assert analyst["p95_latency_ms"] == pytest.approx(93.0)
    assert analyst["mean_latency_ms"] == pytest.approx(800.0 / 15.0)
    assert analyst["input_tokens"] == 1000
    assert analyst["output_tokens"] == 200
    assert analyst["reasoning_tokens"] == 50
    assert analyst["mean_retries"] == 0.0
    assert analyst["retry_rate"] == 0.0
    # Cost only counts records that reported tokens: 10 x (100 + 25x4)/1e6.
    expected_cost = 10 * (100 * 1.0 + (20 + 5) * 4.0) / 1_000_000.0
    assert analyst["mean_cost_units"] == pytest.approx(expected_cost / 10)
    assert analyst["total_cost_units"] == pytest.approx(expected_cost)

    moderator: GroupStats = snapshot["by_role"]["moderator"]
    assert moderator["count"] == 2
    assert moderator["failure_rate"] == 0.5
    assert moderator["p50_latency_ms"] == 300.0  # (200 + 400) / 2
    assert moderator["mean_latency_ms"] == 300.0
    assert moderator["p95_latency_ms"] == 390.0  # 200 + (400-200)*0.95
    assert moderator["mean_retries"] == 0.5
    assert moderator["retry_rate"] == 0.5
    assert moderator["input_tokens"] == 2000
    assert moderator["output_tokens"] == 200
    assert moderator["reasoning_tokens"] == 50

    total: GroupStats = snapshot["total"]
    assert total["count"] == 17
    assert total["requests"] == 17
    assert total["failure_rate"] == 6 / 17
    assert total["p50_latency_ms"] == 60.0
    # 17 samples: position (17-1)*0.95 = 15.2 -> 200 + (400-200)*0.2 = 240 --
    # the slow moderator retry dominates the panel-wide tail.
    assert total["p95_latency_ms"] == pytest.approx(240.0)
    assert total["input_tokens"] == 3000


def test_event_records_are_excluded_from_latency_and_cost_means() -> None:
    """Degradation annotations must not drag p50/p95/mean toward zero."""
    aggregator = LLMObservabilityAggregator()
    aggregator.record(LLMRequestRecord(
        role="analyst", graph="panel", total_latency_ms=100.0, input_tokens=10,
    ))
    aggregator.record(LLMRequestRecord(
        role="analyst", graph="panel", total_latency_ms=300.0, input_tokens=10,
    ))
    event = aggregator.record(LLMRequestRecord(
        role="analyst", graph="panel", retry_count=1, fallback_reason="all_empty_batch",
    ))

    assert event.is_event is True
    stats = aggregator.snapshot()["by_role"]["analyst"]
    assert stats["count"] == 3
    assert stats["requests"] == 2, "the event record measured no request"
    assert stats["p50_latency_ms"] == 200.0
    assert stats["mean_latency_ms"] == 200.0
    # The event still counts toward retries and the failure-rate denominator.
    assert stats["mean_retries"] == 1 / 3
    assert stats["retry_rate"] == 1 / 3


def test_snapshot_groups_only_labelled_records() -> None:
    """An unlabelled caller cannot invent a phantom role or graph group."""
    aggregator = LLMObservabilityAggregator()
    aggregator.record(LLMRequestRecord(total_latency_ms=5.0, input_tokens=1))
    aggregator.record(LLMRequestRecord(role="critic", graph="panel", total_latency_ms=5.0))

    snapshot = aggregator.snapshot()
    assert set(snapshot["by_role"]) == {"critic"}
    assert set(snapshot["by_graph"]) == {"panel"}
    assert snapshot["total"]["count"] == 2


def test_empty_aggregator_reports_zeroes_not_errors() -> None:
    snapshot = LLMObservabilityAggregator().snapshot()
    assert snapshot["total"]["count"] == 0
    assert snapshot["total"]["failure_rate"] == 0.0
    assert snapshot["total"]["p50_latency_ms"] == 0.0
    assert snapshot["total"]["mean_cost_units"] == 0.0
    assert snapshot["by_role"] == {}
    assert snapshot["by_graph"] == {}


def test_ring_buffer_drops_the_oldest_records() -> None:
    """A long-running desktop process must not grow the store without bound."""
    aggregator = LLMObservabilityAggregator(max_records=3)
    for index in range(5):
        record_llm_request_on(aggregator, index)

    records = aggregator.records()
    assert len(records) == 3
    assert [record.node for record in records] == ["n2", "n3", "n4"]


def record_llm_request_on(aggregator: LLMObservabilityAggregator, index: int) -> None:
    aggregator.record(LLMRequestRecord(node=f"n{index}", graph="panel", total_latency_ms=1.0))


def test_summarise_attempts_folds_per_attempt_metrics() -> None:
    """Latency is summed; status class and category come from the last attempt."""
    attempts = [
        HTTPAttemptMetrics(queue_latency_ms=1.0, http_latency_ms=10.0, status_class="5xx"),
        HTTPAttemptMetrics(queue_latency_ms=2.0, http_latency_ms=20.0, status_class="2xx"),
    ]
    assert summarise_attempts(attempts) == (3.0, 30.0, "2xx", "")
    assert summarise_attempts([]) == (0.0, 0.0, "", "")


def test_collect_http_attempts_context_manager() -> None:
    """The collector is usable as a context manager and keeps partial attempts."""
    with collect_http_attempts() as attempts:
        assert attempts == []
    assert attempts == []


def test_record_coercions_reject_non_scalars_and_clamp_labels() -> None:
    """The dataclass is the allow-list boundary: coercion happens on construction."""
    record = LLMRequestRecord(node="  many   spaces  ", total_latency_ms=-5.0, input_tokens=-3)
    assert record.node == "many spaces"
    assert record.total_latency_ms == 0.0
    assert record.input_tokens == 0

    long_label = "x" * 500
    assert len(LLMRequestRecord(final_source=long_label).final_source) == 64

    with pytest.raises(TypeError):
        LLMRequestRecord(node={"not": "a label"})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LLMRequestRecord(input_tokens=["1"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LLMRequestRecord(ok="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LLMRequestRecord(total_latency_ms="fast")  # type: ignore[arg-type]


def test_unknown_record_field_is_a_type_error() -> None:
    """Smuggling free text requires a field name that does not exist."""
    with pytest.raises(TypeError):
        LLMRequestRecord(prompt="user text")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        replace(LLMRequestRecord(), node="x", api_key=KEY_SENTINEL)  # type: ignore[call-arg]


def test_nested_call_labels_override_field_by_field() -> None:
    with llm_call_context(graph="panel", role="analyst"):
        assert current_llm_labels() == ("panel", "", "analyst")
        with llm_call_context(node="moderator"):
            assert current_llm_labels() == ("panel", "moderator", "analyst")
    assert current_llm_labels() == ("", "", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Privacy: sentinels never survive into recorded metadata
# ═══════════════════════════════════════════════════════════════════════════════

_SENTINELS = (PROMPT_SENTINEL, KEY_SENTINEL, BODY_SENTINEL, FORBIDDEN_SENTINEL)


def _assert_no_sentinels(text: str, *, where: str) -> None:
    for sentinel in _SENTINELS:
        assert sentinel not in text, f"{sentinel!r} leaked into {where}"


def _records_as_text(aggregator: LLMObservabilityAggregator) -> str:
    import json

    return json.dumps(
        [record.as_dict() for record in aggregator.records()],
        ensure_ascii=False,
        default=str,
    )


async def test_prompt_and_key_never_reach_the_record_or_the_snapshot() -> None:
    """Sentinels ride the real request and are then absent from all metadata."""
    wire = _ok_wire(f'{{"note": "{BODY_SENTINEL}"}}')
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator)

    with wire.patch_async_client(), llm_call_context(graph="panel", node="analyst", role="analyst"):
        await gateway.complete(
            f"system {PROMPT_SENTINEL}", f"user {PROMPT_SENTINEL}", policy=ANALYST_POLICY,
        )
        await gateway.close()

    # The sentinels really were on the wire, so their absence below is meaningful.
    wire_text = " ".join(wire.recorder.messages())
    assert PROMPT_SENTINEL in wire_text
    assert wire.recorder.payloads[0].get("messages") is not None

    import json

    _assert_no_sentinels(_records_as_text(aggregator), where="records")
    _assert_no_sentinels(
        json.dumps(aggregator.snapshot(), ensure_ascii=False, default=str),
        where="snapshot",
    )
    record = aggregator.records()[0]
    assert record.role == "analyst" and record.node == "analyst"
    assert ALLOWED_FIELDS.isdisjoint({"prompt", "system", "user", "messages", "api_key"})


async def test_provider_error_body_never_reaches_the_record() -> None:
    """A hostile 4xx body is not echoed: only the status class is recorded."""
    wire = MockLLMWire(_ok_wire()._handler)  # type: ignore[attr-defined]

    async def hostile(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={
            "error": {
                "message": f"rejected {PROMPT_SENTINEL}",
                "api_key": KEY_SENTINEL,
                "reasoning_content": BODY_SENTINEL,
                "forbidden": FORBIDDEN_SENTINEL,
            }
        })

    wire.set_handler(hostile)
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator, max_retries=0)

    with wire.patch_async_client():
        with pytest.raises(Exception) as excinfo:
            await gateway.complete(f"s {PROMPT_SENTINEL}", "u", policy=ANALYST_POLICY)
        await gateway.close()

    _assert_no_sentinels(str(excinfo.value), where="the raised error")
    _assert_no_sentinels(_records_as_text(aggregator), where="records")
    record = aggregator.records()[0]
    assert record.http_status_class == "4xx"
    # A provider *rejection* is classified by status class alone; the transport
    # never raised, so there is no transport error category to record.
    assert record.error_category == ""
    assert record.fallback_reason == "non_retriable_error"


async def test_labels_cannot_be_abused_to_carry_free_text() -> None:
    """Every label field is whitespace-collapsed and truncated, so no body fits."""
    wire = _ok_wire()
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator, max_retries=0)

    assert len(LONG_KEY_SENTINEL) > 64
    hostile = f"{PROMPT_SENTINEL} {LONG_KEY_SENTINEL} {BODY_SENTINEL}"
    with wire.patch_async_client():
        with llm_call_context(graph=hostile, node=hostile, role=hostile):
            await gateway.complete("s", "u")
        # A second record carrying a hostile label straight from a policy.
        hostile_policy = replace(MODERATOR_POLICY, node=hostile, role=hostile)
        await gateway.complete("s", "u", policy=hostile_policy)
        await gateway.close()

    assert len(aggregator.records()) == 2
    for record in aggregator.records():
        for field in ("graph", "node", "role", "provider", "model",
                      "http_status_class", "fallback_reason", "final_source",
                      "error_category"):
            value = getattr(record, field)
            assert len(value) <= 64, field
    # Truncation keeps the *head* of the label, so a realistic opaque key cannot
    # survive anywhere: its sentinel is cut off entirely.
    first, second = aggregator.records()
    assert len(first.graph) == 64
    assert first.graph.startswith(PROMPT_SENTINEL[:20])
    for record in (first, second):
        assert LONG_KEY_SENTINEL not in record.graph
        assert LONG_KEY_SENTINEL not in record.node
        assert LONG_KEY_SENTINEL not in record.role
    # The policy-supplied label reaches the record through the same boundary.
    assert second.node == hostile[:64]
    assert BODY_SENTINEL not in second.node
    # Whitespace is collapsed, so a label cannot smuggle a multi-line blob.
    assert "\n" not in LLMRequestRecord(node="a\n\n b\tc").node


async def test_forbidden_word_failure_is_recorded_as_a_flag_not_as_text() -> None:
    """The forbidden word itself is never stored — only the boolean fact."""
    wire = _ok_wire(f'{{"response_text": "你被{ FORBIDDEN_SENTINEL }了"}}')
    aggregator = LLMObservabilityAggregator()
    gateway = _gateway(wire, observability=aggregator)

    with wire.patch_async_client():
        content = await gateway.complete("s", "u", policy=ANALYST_POLICY)
        await gateway.close()

    assert FORBIDDEN_SENTINEL in content  # it *is* in the completion
    aggregator.record(LLMRequestRecord(
        graph="panel", node="critic", role="critic", forbidden_word_failure=True,
    ))

    _assert_no_sentinels(_records_as_text(aggregator), where="records")
    assert aggregator.records()[1].forbidden_word_failure is True
    assert aggregator.snapshot()["total"]["forbidden_word_failure_rate"] == 0.5


def test_api_key_is_absent_from_snapshot_and_describe_style_views() -> None:
    """A key-shaped string in *any* allowed field is truncated out, not stored."""
    aggregator = LLMObservabilityAggregator()
    aggregator.record(LLMRequestRecord(
        provider=KEY_SENTINEL,
        model=KEY_SENTINEL,
        fallback_reason=KEY_SENTINEL,
        final_source=KEY_SENTINEL,
    ))
    record = aggregator.records()[0]
    # 64 characters of a 38-character key is still the key — so the guard that
    # matters is that a *real* key never has a field to travel in.
    assert len(KEY_SENTINEL) <= 64
    assert ALLOWED_FIELDS.isdisjoint({"key", "token", "headers", "authorization"})
    assert not hasattr(record, "api_key")
    assert not hasattr(record, "prompt")
    assert not hasattr(record, "response")
