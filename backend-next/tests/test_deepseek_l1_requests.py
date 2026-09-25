"""Request-level regressions for the DeepSeek-direct L1 contract (plan item 2).

Three things must be true on the wire, not merely in configuration:

1. **Every L1 entry point uses the resolved model** (`deepseek-flash` under the
   production pin) and the DeepSeek endpoint — a legacy ECNU URL/model/key left
   in the environment must not leak into a request or into the assembled client.
2. **Role-level reasoning and token limits actually reach the request body**:
   the policy's `reasoning_effort`, its `thinking` switch and its output cap are
   per-request fields, and the structured tier keeps its JSON constraint while
   the prose tier drops it.
3. **Thinking-mode tool rounds echo `reasoning_content`**: DeepSeek returns 400
   when a request carries `tools` and the previous assistant turn's
   `reasoning_content` is missing, so the field must survive into the next
   request body alongside the complete tool-call/tool-result pair.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mindflow.agents.llm_gateway import GatewayAPIError, LangChainGateway
from mindflow.agents.policies import (
    ANALYST_POLICY,
    ATTRIBUTION_POLICY,
    CHAT_POLICY,
    MODERATOR_POLICY,
    THINKING_TOKEN_FLOOR,
)
from mindflow.config import LLMSettings, Settings
from mindflow.infrastructure.llm.schemas import LLMAttributionResult
from mindflow.infrastructure.provider_registry import ProviderRegistry
from tests._llm_test_support import MockLLMWire, chat_completion, error_response

PINNED_MODEL = "deepseek-flash"
DEEPSEEK_HOST = "api.deepseek.com"

#: A sentinel credential — never a real key.
SENTINEL_KEY = "sk-sentinel-deepseek"


def _legacy_ecnu_settings(**overrides: object) -> LLMSettings:
    """The worst case: an ECNU URL, an ECNU model, an ECNU provider and its key."""
    defaults: dict[str, object] = {
        "api_key": "sk-ecnu-legacy",
        "base_url": "https://chat.ecnu.edu.cn/open/api/v1",
        "model": "ecnu-max",
        "provider": "ecnu",
        "timeout_s": 30,
        "max_retries": 0,
        "max_concurrent_requests": 1,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Every L1 entry point uses the pinned model/endpoint
# ═══════════════════════════════════════════════════════════════════════════════


def test_registry_chat_model_is_the_pinned_deepseek_model() -> None:
    registry = ProviderRegistry(_legacy_ecnu_settings(deepseek_api_key=SENTINEL_KEY))
    model = registry.get_chat_model()

    assert model is not None
    assert model.model_name == PINNED_MODEL
    assert DEEPSEEK_HOST in str(model.openai_api_base)
    # The ECNU model id must not survive anywhere a request could pick it up.
    assert getattr(model, "model_name", "") != "ecnu-max"


def test_registry_gateway_resolves_the_pinned_target() -> None:
    registry = ProviderRegistry(_legacy_ecnu_settings(deepseek_api_key=SENTINEL_KEY))
    gateway = registry.get_gateway()

    assert gateway._model_id == PINNED_MODEL
    assert gateway._is_ecnu is False
    assert gateway._provider_label == "deepseek"
    assert DEEPSEEK_HOST in gateway._base_url


def test_structured_attribution_client_targets_deepseek() -> None:
    registry = ProviderRegistry(_legacy_ecnu_settings(deepseek_api_key=SENTINEL_KEY))
    client = registry.get_structured_attribution()

    assert client is not None
    # The typed attribution client reads its endpoint/model from the same pin.
    assert DEEPSEEK_HOST in client._base_url
    assert client.model == PINNED_MODEL
    described = registry.describe()
    assert described["model"] == PINNED_MODEL
    assert described["base_url_host"] == DEEPSEEK_HOST
    assert described["provider"] == "generic"
    assert described["provenance"] == "deepseek-direct"


def test_missing_deepseek_credential_leaves_l1_unavailable() -> None:
    """No DeepSeek key → L1 unavailable, and the ECNU key is never borrowed."""
    registry = ProviderRegistry(_legacy_ecnu_settings())

    assert registry.get_structured_attribution() is None
    assert registry.get_chat_model() is None
    assert registry.describe()["credential_present"] is False


def test_describe_never_contains_the_credential() -> None:
    registry = ProviderRegistry(_legacy_ecnu_settings(deepseek_api_key=SENTINEL_KEY))
    rendered = json.dumps(registry.describe(), ensure_ascii=False) + json.dumps(
        registry.l1_target.describe(), ensure_ascii=False,
    )

    assert SENTINEL_KEY not in rendered


def test_ecnu_compat_switch_restores_the_legacy_endpoint() -> None:
    settings = _legacy_ecnu_settings(ecnu_compat_enabled=True)
    target = settings.l1_target()

    assert target.provenance == "ecnu-compat"
    assert target.provider == "ecnu"
    assert target.model == "ecnu-max"
    assert "ecnu.edu.cn" in target.base_url
    assert target.api_key == "sk-ecnu-legacy"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Reasoning / token / JSON constraints on the wire
# ═══════════════════════════════════════════════════════════════════════════════


def _ok_handler(content: str = "{}"):
    """An async handler (the wire awaits handlers) returning a 200 completion."""
    async def _handler(request: httpx.Request) -> httpx.Response:
        return chat_completion(content, model=PINNED_MODEL)
    return _handler


async def _gateway_with_wire(
    wire: MockLLMWire,
    *,
    settings: LLMSettings | None = None,
) -> LangChainGateway:
    """Build a gateway pointed at the mocked wire (the caller sets the handler)."""
    resolved = settings or _legacy_ecnu_settings(deepseek_api_key=SENTINEL_KEY)
    target = resolved.l1_target()
    return LangChainGateway(
        api_key=target.api_key,
        base_url=target.base_url,
        timeout_s=resolved.timeout_s,
        max_retries=0,
        llm_settings=resolved,
        target=target,
    )


async def test_policy_reasoning_and_output_cap_reach_the_wire() -> None:
    wire = MockLLMWire(_ok_handler())
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        try:
            await gateway.complete("system", "user", model="chat", policy=ANALYST_POLICY)
        finally:
            await gateway.close()

    body = wire.recorder.last
    assert body["model"] == PINNED_MODEL
    assert body["reasoning_effort"] == ANALYST_POLICY.reasoning_effort  # "low"
    # Thinking tokens count against the cap, so the request carries the
    # thinking floor rather than the policy's answer-only cap (measured against
    # real DeepSeek: a 1200-token thinking request truncates before the JSON).
    assert body["max_tokens"] == max(
        ANALYST_POLICY.max_output_tokens, THINKING_TOKEN_FLOOR,
    )
    assert body["thinking"] == {"type": "enabled"}
    # The structured tier keeps its JSON constraint.
    assert body["response_format"] == {"type": "json_object"}


async def test_moderator_policy_sends_high_effort_and_its_own_cap() -> None:
    wire = MockLLMWire(_ok_handler())
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        try:
            await gateway.complete("system", "user", model="chat", policy=MODERATOR_POLICY)
        finally:
            await gateway.close()

    body = wire.recorder.last
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == max(
        MODERATOR_POLICY.max_output_tokens, THINKING_TOKEN_FLOOR,
    )


async def test_prose_tier_drops_the_json_constraint() -> None:
    wire = MockLLMWire(_ok_handler("prose"))
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        try:
            await gateway.complete("system", "user", model="reasoner", policy=MODERATOR_POLICY)
        finally:
            await gateway.close()

    body = wire.recorder.last
    assert "response_format" not in body
    assert body["model"] == PINNED_MODEL
    # Thinking/reasoning still travel: the prose tier differs only in the
    # output constraint, never in which model or thinking mode is used.
    assert body["reasoning_effort"] == "high"


async def test_chat_policy_keeps_the_provider_default_effort() -> None:
    wire = MockLLMWire(_ok_handler("hi"))
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        try:
            await gateway.complete("system", "user", model="chat", policy=CHAT_POLICY)
        finally:
            await gateway.close()

    body = wire.recorder.last
    assert body["max_tokens"] == 2048
    # Provider default: no explicit effort is sent, so DeepSeek applies its own.
    assert "reasoning_effort" not in body
    assert "thinking" not in body


async def test_temperature_is_not_sent_alongside_an_explicit_thinking_effort() -> None:
    """Thinking mode ignores temperature, so the request omits it entirely.

    DeepSeek documents that ``temperature`` has no effect while thinking is on
    (setting it is accepted but ignored). The role policy does not contribute
    one, and the DeepSeek model drops the instance default for a thinking
    request, so the wire body carries no sampling parameter that would imply
    otherwise.
    """
    wire = MockLLMWire(_ok_handler())
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        try:
            await gateway.complete("system", "user", model="chat", policy=ANALYST_POLICY)
        finally:
            await gateway.close()

    assert "temperature" not in ANALYST_POLICY.request_overrides(provider="deepseek")
    body = wire.recorder.last
    assert body["reasoning_effort"] == "low"
    assert "temperature" not in body


async def test_temperature_is_kept_when_thinking_is_not_requested() -> None:
    """Without a thinking effort the sampling temperature still applies."""
    wire = MockLLMWire(_ok_handler("hi"))
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        try:
            await gateway.complete("system", "user", model="chat", policy=CHAT_POLICY)
        finally:
            await gateway.close()

    assert wire.recorder.last["temperature"] == pytest.approx(0.2)


async def test_no_legacy_model_or_host_reaches_the_wire() -> None:
    wire = MockLLMWire(_ok_handler())
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        try:
            await gateway.complete("system", "user", model="chat", policy=ANALYST_POLICY)
        finally:
            await gateway.close()

    assert wire.recorder.urls, "no request was sent"
    url = wire.recorder.urls[-1]
    assert DEEPSEEK_HOST in url
    assert "ecnu.edu.cn" not in url
    assert "ecnu" not in json.dumps(wire.recorder.last, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Thinking-mode tool rounds: reasoning_content echo + pairing
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL = [{
    "type": "function",
    "function": {
        "name": "query_evidence",
        "description": "Query behaviour evidence",
        "parameters": {"type": "object", "properties": {}},
    },
}]


async def test_tool_round_echoes_reasoning_content_and_keeps_the_pair() -> None:
    """DeepSeek 400s when a `tools` request omits the previous reasoning."""
    seen: list[dict[str, Any]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content or b"{}"))
        return chat_completion("final answer", model=PINNED_MODEL)

    wire = MockLLMWire(_handler)
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        try:
            # The tool path uses the prose tier: the structured tier sets
            # ``response_format: json_object``, and langchain then tries to
            # auto-parse tool calls (which fails for non-strict functions).
            model = gateway._get_model("reasoner")
            history = [
                SystemMessage(content="system"),
                HumanMessage(content="question"),
                AIMessage(
                    content="",
                    additional_kwargs={"reasoning_content": "thinking trace"},
                    tool_calls=[{
                        "id": "call-1", "name": "query_evidence", "args": {},
                    }],
                ),
                ToolMessage(content='{"evidence": []}', tool_call_id="call-1"),
            ]
            await model.ainvoke(history, tools=_TOOL)
        finally:
            await gateway.close()

    assert seen, "the mocked wire received no request"
    messages = seen[-1]["messages"]
    assistant = next(m for m in messages if m.get("role") == "assistant")
    tool_messages = [m for m in messages if m.get("role") == "tool"]

    # The reasoning trace must be echoed back, or DeepSeek rejects the request.
    assert assistant.get("reasoning_content") == "thinking trace"
    # The tool call and its result must both be present and paired.
    assert assistant.get("tool_calls"), "tool call was dropped"
    assert tool_messages and tool_messages[0]["tool_call_id"] == "call-1"
    assert seen[-1].get("tools"), "tools must be declared on the request"


async def test_non_retriable_4xx_is_not_retried_with_the_same_parameters() -> None:
    """A rejected parameter must not be re-sent identically (plan item 2)."""
    attempts = {"count": 0}

    async def _handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return error_response(400)

    wire = MockLLMWire(_handler)
    with wire.patch_async_client():
        gateway = await _gateway_with_wire(wire)
        wire.set_handler(_handler)
        try:
            with pytest.raises(GatewayAPIError):
                await gateway.complete("system", "user", model="chat", policy=ANALYST_POLICY)
        finally:
            await gateway.close()

    assert attempts["count"] == 1, "a 4xx must not be retried with identical parameters"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. The structured attribution client (raw HTTP L1 entry point)
# ═══════════════════════════════════════════════════════════════════════════════


def _attribution_settings() -> LLMSettings:
    return LLMSettings(
        api_key=SENTINEL_KEY,
        base_url="https://api.deepseek.com",
        model=PINNED_MODEL,
        provider="generic",
        timeout_s=30,
        max_retries=0,
        max_concurrent_requests=1,
    )


def _attribution_payload() -> str:
    return LLMAttributionResult(
        procrastination_types=["impulsivity"],
        type_confidence={"impulsivity": 0.82},
        cbt_technique="stimulus_control",
        response_text="测试回应",
        next_action="测试行动",
    ).model_dump_json()


async def test_attribution_client_carries_the_same_reasoning_contract() -> None:
    """Every L1 entry point sends Flash + effort/thinking/cap + JSON mode."""
    from mindflow.infrastructure.llm.client import DeepSeekClient

    async def _handler(request: httpx.Request) -> httpx.Response:
        return chat_completion(
            _attribution_payload(), model=PINNED_MODEL, reasoning_tokens=77,
        )

    wire = MockLLMWire(_handler)
    with wire.patch_async_client():
        client = DeepSeekClient(_attribution_settings())
        try:
            result = await client.analyze("{}")
        finally:
            await client.close()

    body = wire.recorder.last
    assert body["model"] == PINNED_MODEL
    assert body["response_format"] == {"type": "json_object"}
    assert body["reasoning_effort"] == ATTRIBUTION_POLICY.reasoning_effort
    assert body["thinking"] == {"type": "enabled"}
    assert body["max_tokens"] == max(
        ATTRIBUTION_POLICY.max_output_tokens, THINKING_TOKEN_FLOOR,
    )
    # Provider-reported usage is captured for observability, not discarded.
    assert result.procrastination_types == ["impulsivity"]
    assert client.last_usage is not None
    assert client.last_usage[2] == 77
    assert client.last_usage[0] > 0 and client.last_usage[1] > 0


async def test_attribution_client_reports_missing_usage_as_none() -> None:
    """A provider that omits ``usage`` must not be estimated from text length."""
    from mindflow.infrastructure.llm.client import DeepSeekClient

    async def _handler(request: httpx.Request) -> httpx.Response:
        raw = chat_completion(_attribution_payload(), model=PINNED_MODEL)
        body = json.loads(bytes(raw.content))
        body.pop("usage", None)
        return httpx.Response(200, json=body)

    wire = MockLLMWire(_handler)
    with wire.patch_async_client():
        client = DeepSeekClient(_attribution_settings())
        try:
            await client.analyze("{}")
        finally:
            await client.close()

    assert client.last_usage is None


def test_pinned_settings_keep_both_tiers_on_one_model() -> None:
    """chat/reasoner are output policies, not two model ids."""
    settings = _legacy_ecnu_settings(deepseek_api_key=SENTINEL_KEY)
    target = settings.l1_target()
    gateway = LangChainGateway(
        api_key=target.api_key, base_url=target.base_url, llm_settings=settings, target=target,
    )
    assert gateway._model_id == PINNED_MODEL
    assert gateway._get_model("chat") is not gateway._get_model("reasoner")
    assert gateway._get_model("chat").model_name == PINNED_MODEL
    assert gateway._get_model("reasoner").model_name == PINNED_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# Credential resolution precedence
# ═══════════════════════════════════════════════════════════════════════════════


def _clear_credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MINDFLOW_LLM__DEEPSEEK_API_KEY", raising=False)


def test_prefixed_credential_wins_over_the_bare_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_credential_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-bare")
    monkeypatch.setenv("MINDFLOW_LLM__DEEPSEEK_API_KEY", "sk-prefixed")

    # Standalone construction (tests, diagnostics) …
    assert LLMSettings().deepseek_api_key == "sk-prefixed"
    # … and the nested production path must agree.
    assert Settings(_env_file=None).llm.deepseek_api_key == "sk-prefixed"


def test_bare_credential_is_used_when_the_prefixed_one_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_credential_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-bare")

    assert LLMSettings().deepseek_api_key == "sk-bare"


def test_explicit_credential_argument_beats_both_env_vars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_credential_env(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-bare")
    monkeypatch.setenv("MINDFLOW_LLM__DEEPSEEK_API_KEY", "sk-prefixed")

    assert LLMSettings(deepseek_api_key="sk-explicit").deepseek_api_key == "sk-explicit"


def test_no_credential_anywhere_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_credential_env(monkeypatch)

    assert LLMSettings().deepseek_api_key is None
