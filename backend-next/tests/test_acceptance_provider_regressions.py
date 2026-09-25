"""Production-boundary regressions for acceptance A05/A06/A09/A12/A14."""

from __future__ import annotations

import asyncio
import json
import traceback
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from langchain_openai.chat_models import _client_utils
from loguru import logger

from mindflow.agents.llm_gateway import GatewayAPIError
from mindflow.api.deps import get_intervention_service
from mindflow.api.routes.intervention import router
from mindflow.config import LLMSettings
from mindflow.domain.procrastination import (
    CBTTechnique,
    ProcrastinationAssessment,
    ProcrastinationType,
)
from mindflow.infrastructure.notification import LogOnlyNotifier, WindowsNotifier
from mindflow.infrastructure.provider_registry import ProviderRegistry
from mindflow.infrastructure.repositories.intervention import InterventionLogRepository
from mindflow.services.intervention_service import (
    InterventionService,
    _generate_llm_message,
    _generate_ollama_message,
)

_KEY_SENTINEL = "SYNTHETIC_OPAQUE_KEY"
_REASONING_SENTINEL = "SYNTHETIC_PRIVATE_REASONING"


@pytest.fixture
def error_logs():
    messages = []
    sink = logger.add(
        lambda message: messages.append(str(message)),
        format="{message}", backtrace=False, diagnose=False,
    )
    try:
        yield messages
    finally:
        logger.remove(sink)


def _assert_private(text):
    assert _KEY_SENTINEL not in text
    assert _REASONING_SENTINEL not in text


def _error_body():
    return {
        "error": {
            "message": "request rejected",
            "api_key": _KEY_SENTINEL,
            "reasoning_content": _REASONING_SENTINEL,
        },
        "choices": None,
    }


def _settings(**overrides):
    """ECNU-shaped settings for this suite: the legacy triple, explicitly opted in.

    ``ecnu_compat_enabled`` keeps these regressions on the campus wire — the
    production pin (DeepSeek direct) would otherwise replace the URL, model and
    key below. ``reasoning_effort="max"`` is the gateway's own default tier,
    which the application default (DeepSeek's ``"high"``) must not silently
    change under the ECNU wire assertions.
    """
    return LLMSettings(**{
        "api_key": "test-key",
        "base_url": "https://chat.ecnu.edu.cn/open/api/v1",
        "model": "ecnu-max",
        "provider": "ecnu",
        "ecnu_compat_enabled": True,
        "reasoning_effort": "max",
        "timeout_s": 180,
        "max_retries": 0,
        "max_output_tokens": 16384,
        "max_concurrent_requests": 1,
        **overrides,
    })


def _service(notifier, client=None, repo=None):
    return InterventionService(
        intervention_repo=repo if repo is not None else AsyncMock(),
        throttle=AsyncMock(),
        notifier=notifier,
        broadcast_fn=AsyncMock(),
        llm_client=client,
        llm_model="ecnu-max",
        auth_token="test-token",
    )


async def _intervene(service):
    return await service.maybe_intervene(
        assessment=ProcrastinationAssessment(
            types=(ProcrastinationType.TASK_AVERSION,),
            confidence={ProcrastinationType.TASK_AVERSION: 0.8},
            recommended_technique=CBTTechnique.GRADED_EXPOSURE,
            rationale="test",
            source="rule_engine",
        ),
        bypass_throttle=True,
        bypass_deep_work_guard=True,
    )


@pytest.fixture
def wire(monkeypatch):
    """Replace network I/O, retaining the production clients and SDK request path."""
    requests = []
    first_started = asyncio.Event()
    release = asyncio.Event()
    release.set()
    counters = {"active": 0, "peak": 0}

    async def respond(request):
        requests.append(request)
        counters["active"] += 1
        counters["peak"] = max(counters["peak"], counters["active"])
        first_started.set()
        try:
            await release.wait()
            attribution = {
                "procrastination_types": ["task_aversion"],
                "type_confidence": {"task_aversion": 0.8},
                "cognitive_distortions": [],
                "cbt_technique": "graded_exposure",
                "response_text": "Start with one small action.",
                "next_action": "Open the document.",
            }
            payload = json.loads(request.content)
            content = json.dumps(
                attribution if "procrastination_types" in payload["messages"][0]["content"]
                else {"title": "Take a step", "message": "Start with one small action."}
            )
            return httpx.Response(200, json={
                "id": "test-completion",
                "model": "ecnu-max",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }],
            })
        finally:
            counters["active"] -= 1

    original_init = httpx.AsyncClient.__init__

    def init(client, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        kwargs["trust_env"] = False
        original_init(client, *args, **kwargs)

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    _client_utils._cached_async_httpx_client.cache_clear()
    _client_utils._cached_sync_httpx_client.cache_clear()
    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    return requests, counters, first_started, release


@pytest.mark.parametrize("provider", ["ecnu", "generic"])
@pytest.mark.parametrize("limit", [1, 2])
async def test_all_production_entries_share_one_gate(wire, provider, limit):
    requests, counters, first_started, release = wire
    release.clear()
    registry = ProviderRegistry(_settings(provider=provider, max_concurrent_requests=limit))
    client = registry.get_structured_attribution()
    model = registry.get_chat_model()
    assert client is not None and model is not None
    service = _service(LogOnlyNotifier(), client.client)
    tasks = [
        asyncio.create_task(model.ainvoke("hello")),
        asyncio.create_task(registry.get_gateway().complete("system", "user")),
        asyncio.create_task(client.analyze("{}")),
        asyncio.create_task(_intervene(service)),
    ]
    try:
        await asyncio.wait_for(first_started.wait(), 2)
        await asyncio.sleep(0.1)
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert len(requests) == 4
        assert counters["peak"] == limit
        assert registry.concurrency.snapshot()["acquired"] == 4
        assert registry.concurrency.snapshot()["current_in_flight"] == 0
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await registry.shutdown()


async def test_cancelled_active_and_waiting_requests_release_gate(wire):
    requests, _, first_started, release = wire
    release.clear()
    registry = ProviderRegistry(_settings())
    client = registry.get_structured_attribution()
    assert client is not None
    active = asyncio.create_task(client.analyze("{}"))
    waiting = None
    try:
        await asyncio.wait_for(first_started.wait(), 2)
        waiting = asyncio.create_task(registry.get_gateway().complete("system", "user"))
        await asyncio.sleep(0.05)
        assert len(requests) == 1
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert registry.concurrency.snapshot()["current_in_flight"] == 1
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert registry.concurrency.snapshot()["current_in_flight"] == 0
        release.set()
        await asyncio.wait_for(client.analyze("{}"), 2)
        assert registry.concurrency.snapshot()["acquired"] == 2
    finally:
        release.set()
        await asyncio.gather(
            active, *([waiting] if waiting is not None else []), return_exceptions=True,
        )
        await registry.shutdown()


async def test_retry_releases_slot_before_backoff(wire, monkeypatch):
    registry = ProviderRegistry(_settings(max_retries=1))
    client = registry.get_structured_attribution()
    assert client is not None
    transport = client.client._transport
    respond = transport.handle_async_request
    attempts = 0

    async def fail_once(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"error": "temporary"})
        return await respond(request)

    async def backoff(_delay):
        assert registry.concurrency.snapshot()["current_in_flight"] == 0

    monkeypatch.setattr(transport, "handle_async_request", fail_once)
    monkeypatch.setattr("mindflow.infrastructure.llm.client.asyncio.sleep", backoff)
    try:
        await client.analyze("{}")
        assert attempts == 2
        assert registry.concurrency.snapshot()["acquired"] == 2
        assert registry.concurrency.snapshot()["current_in_flight"] == 0
    finally:
        await registry.shutdown()


async def test_stream_holds_gate_until_closed(wire, monkeypatch):
    registry = ProviderRegistry(_settings())
    client = registry.get_structured_attribution()
    assert client is not None

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"chunk"

    async def respond(request):
        return httpx.Response(200, stream=Stream())

    monkeypatch.setattr(client.client._transport, "handle_async_request", respond)
    try:
        async with client.client.stream("POST", "/chat/completions") as response:
            assert registry.concurrency.snapshot()["current_in_flight"] == 1
            assert await response.aread() == b"chunk"
        assert registry.concurrency.snapshot()["current_in_flight"] == 0
        await response.aclose()
        assert registry.concurrency.snapshot()["current_in_flight"] == 0
    finally:
        await registry.shutdown()


@pytest.mark.parametrize("entry", ["attribution", "intervention"])
@pytest.mark.parametrize("thinking", [True, False])
async def test_ecnu_policy_reaches_raw_http_consumers(wire, entry, thinking):
    requests, *_ = wire
    registry = ProviderRegistry(_settings(thinking_enabled=thinking))
    client = registry.get_structured_attribution()
    assert client is not None
    try:
        if entry == "attribution":
            await client.analyze("{}")
        else:
            result = await _intervene(_service(LogOnlyNotifier(), client.client))
            assert result.intervention is not None
            assert result.intervention.title == "Take a step"
        request, = requests
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "enabled" if thinking else "disabled"}
        assert payload.get("reasoning_effort") == ("max" if thinking else None)
        assert payload["max_completion_tokens"] == 16384
        assert "max_tokens" not in payload
        if thinking:
            assert "temperature" not in payload
        assert request.extensions["timeout"]["read"] == 180
    finally:
        await registry.shutdown()


async def test_public_gateway_preserves_chat_json_tier(wire):
    requests, *_ = wire
    registry = ProviderRegistry(_settings())
    try:
        gateway = registry.get_gateway()
        for tier in ("chat", "reasoner", "chat"):
            await gateway.complete("return JSON", "user", model=tier)
        payloads = [json.loads(request.content) for request in requests]
        assert [p.get("response_format") for p in payloads] == [
            {"type": "json_object"}, None, {"type": "json_object"},
        ]
        assert all(p["model"] == "ecnu-max" for p in payloads)
    finally:
        await registry.shutdown()


async def test_explicit_generic_wins_over_ecnu_heuristics(wire):
    requests, *_ = wire
    registry = ProviderRegistry(_settings(provider="generic"))
    try:
        assert registry.describe()["provider"] == "generic"
        model = registry.get_chat_model()
        client = registry.get_structured_attribution()
        assert model is not None and client is not None
        await model.ainvoke("hello")
        await registry.get_gateway().complete("system", "user")
        await client.analyze("{}")
        await _intervene(_service(LogOnlyNotifier(), client.client))
        for request in requests:
            payload = json.loads(request.content)
            # Every L1 entry point now carries the DeepSeek reasoning contract
            # (plan item 2): chat keeps the chat tier's provider-default effort
            # (no explicit fields), the gateway and the structured attribution
            # client send effort + thinking; the ECNU spelling never appears.
            if payload.get("reasoning_effort") is not None:
                assert payload["thinking"] == {"type": "enabled"}
            assert "max_completion_tokens" not in payload
    finally:
        await registry.shutdown()


@pytest.mark.parametrize(
    ("kind", "interactive", "plain", "expected"),
    [
        ("log", False, False, False),
        ("windows", True, False, True),
        ("windows", False, True, True),
        ("windows", False, False, False),
    ],
)
async def test_native_delivery_means_visible_not_interactive(
    kind, interactive, plain, expected,
):
    if kind == "log":
        notifier = LogOnlyNotifier()
    else:
        notifier = WindowsNotifier.__new__(WindowsNotifier)
        notifier._interactive = AsyncMock()
        notifier._interactive.send.return_value = interactive
        backend = AsyncMock()
        backend.send.return_value = plain
        notifier._backends = [backend]
    service = _service(notifier)
    result = await _intervene(service)
    assert result.intervention is not None
    frame = service._broadcast_fn.call_args.args[0]
    assert frame["payload"]["native_delivered"] is expected
    legacy = await notifier.send("title", "body", intervention_id="id", auth_token="token")
    assert legacy is (True if kind == "log" else interactive)


@pytest.mark.parametrize(
    "submissions",
    [
        [("accepted", "human"), ("ignored", "human")],
        [("accepted", "human"), ("ignored", "auto")],
        [("ignored", "auto"), ("accepted", "human"), ("ignored", "auto")],
    ],
)
async def test_response_http_returns_authoritative_record(
    session_factory, create_tables, submissions,
):
    repo = InterventionLogRepository(session_factory=session_factory)
    await repo.log_triggered(
        user_id=1, intervention_type="task_breakdown", intervention_id="test-id",
    )
    service = _service(LogOnlyNotifier(), repo=repo)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_intervention_service] = lambda: service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as client:
        for response, source in submissions:
            result = await client.post(
                "/intervention/test-id/response",
                json={"response": response, "source": source, "latency_s": 1},
            )
            assert result.status_code == 200, result.text
            row = await repo.get_by_id("test-id")
            assert row is not None
            assert result.json()["user_response"] == row["user_response"]


async def test_ecnu_result_error_does_not_expose_provider_body(wire):
    registry = ProviderRegistry(_settings())
    try:
        model = registry.get_chat_model()
        with pytest.raises(Exception) as caught:
            model._create_chat_result(_error_body())
        _assert_private(str(caught.value))
        _assert_private("".join(traceback.format_exception(caught.value)))
    finally:
        await registry.shutdown()


@pytest.mark.parametrize("status", [200, 400, "adapter", "exception"])
async def test_gateway_logs_and_exception_chain_are_allowlisted(
    wire, monkeypatch, error_logs, status,
):
    registry = ProviderRegistry(_settings())
    gateway = registry.get_gateway()
    # The tier label the gateway uses for the structured tier (it is not a model
    # id: both tiers request the resolved model and differ only in JSON mode).
    model = gateway._get_model("chat")

    async def fail(*args, **kwargs):
        if status == "adapter":
            return model._create_chat_result(_error_body())
        raise RuntimeError(json.dumps(_error_body()))

    async def respond(request):
        return httpx.Response(
            status, json=_error_body(),
            headers={"x-private-test": _KEY_SENTINEL},
        )

    monkeypatch.setattr(
        model.http_async_client._transport, "handle_async_request", respond,
    )
    if isinstance(status, str):
        monkeypatch.setattr(type(model), "ainvoke", fail)
    try:
        with pytest.raises(GatewayAPIError) as caught:
            await gateway.complete("system", "user")
        _assert_private("\n".join(error_logs))
        assert any("type=" in message for message in error_logs)
        if status == 400:
            assert any("status=400" in message for message in error_logs)
        _assert_private("".join(traceback.format_exception(caught.value)))
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
    finally:
        await registry.shutdown()


@pytest.mark.parametrize("failure", ["http", "transport", "validation", "malformed"])
async def test_attribution_errors_never_log_or_chain_private_payloads(
    wire, monkeypatch, error_logs, failure,
):
    registry = ProviderRegistry(_settings())
    client = registry.get_structured_attribution()
    assert client is not None

    async def respond(request):
        if failure == "transport":
            raise httpx.ConnectError(json.dumps(_error_body()), request=request)
        if failure == "validation":
            return httpx.Response(200, json={
                "choices": [{"message": {"content": json.dumps({
                    "procrastination_types": [_KEY_SENTINEL],
                    "cbt_technique": _REASONING_SENTINEL,
                })}}],
            })
        return httpx.Response(
            400 if failure == "http" else 200, json=_error_body(),
        )

    monkeypatch.setattr(client.client._transport, "handle_async_request", respond)
    try:
        with pytest.raises(Exception) as caught:
            await client.analyze("{}")
        _assert_private("\n".join(error_logs))
        _assert_private("".join(traceback.format_exception(caught.value)))
    finally:
        await registry.shutdown()


@pytest.mark.parametrize("entry", ["primary", "ollama"])
async def test_intervention_failure_logs_are_allowlisted(
    wire, monkeypatch, error_logs, entry,
):
    registry = ProviderRegistry(_settings())
    client = registry.get_structured_attribution()
    assert client is not None

    async def fail(*args, **kwargs):
        raise httpx.ConnectError(json.dumps(_error_body()))

    monkeypatch.setattr(httpx.AsyncClient, "post", fail)
    try:
        common = {
            "model": "test-model", "summary_json": "{}",
            "intervention_type": "task_breakdown", "intensity": "standard",
        }
        if entry == "primary":
            result = await _generate_llm_message(llm_client=client.client, **common)
        else:
            result = await _generate_ollama_message(
                ollama_base_url="http://offline.test", **common,
            )
        assert result is None
        _assert_private("\n".join(error_logs))
    finally:
        await registry.shutdown()
