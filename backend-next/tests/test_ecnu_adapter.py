"""Regression tests for the ECNU provider adapter.

Covers the adapter's contract with the campus gateway: the thinking fields it
must send, the ``reasoning_content`` round-trip after tool calls, effort-tier
validation (including the disclosure-worthy downgrade case), and the shared
concurrency gate. No network access: the payload is inspected directly, which
is what actually leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.ecnu import (
    ECNUChatModel,
    build_ecnu_model,
    looks_like_ecnu,
    resolve_effort,
    supported_efforts,
)

ECNU_URL = "https://chat.ecnu.edu.cn/open/api/v1"


def _model(**overrides: object) -> ECNUChatModel:
    kwargs: dict[str, object] = {
        "model": "ecnu-max",
        "api_key": "test-key",
        "base_url": ECNU_URL,
        "reasoning_effort": "max",
    }
    kwargs.update(overrides)
    return build_ecnu_model(**kwargs)  # type: ignore[arg-type]


def test_thinking_enabled_sends_enabled_and_effort() -> None:
    """Thinking mode must be requested explicitly on every generation."""
    payload = _model()._get_request_payload(
        [SystemMessage(content="s"), HumanMessage(content="h")]
    )
    assert payload["extra_body"]["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "max"


def test_thinking_disabled_sends_disabled_and_omits_effort() -> None:
    """With thinking off the effort parameter is meaningless, so it is omitted."""
    payload = _model(thinking_enabled=False)._get_request_payload(
        [HumanMessage(content="h")]
    )
    assert payload["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload


def test_output_cap_is_always_set() -> None:
    """Every generation carries an explicit output ceiling."""
    payload = _model(max_tokens=16384)._get_request_payload([HumanMessage(content="h")])
    assert payload["max_completion_tokens"] == 16384


def test_temperature_dropped_while_thinking() -> None:
    """The gateway restricts sampling params in thinking mode."""
    payload = _model(temperature=0.7)._get_request_payload([HumanMessage(content="h")])
    assert "temperature" not in payload


def test_reasoning_content_round_trips_after_tool_call() -> None:
    """Post-tool turns must echo the assistant's reasoning_content back."""
    tool_call = {"name": "get_stats", "args": {"day": "2026-09-18"}, "id": "call_1"}
    assistant = AIMessage(content="", tool_calls=[tool_call])
    assistant.additional_kwargs["reasoning_content"] = "chain of thought"

    payload = _model()._get_request_payload([
        SystemMessage(content="s"),
        HumanMessage(content="u"),
        assistant,
        ToolMessage(content="{}", tool_call_id="call_1", name="get_stats"),
    ])

    assistants = [m for m in payload["messages"] if m.get("role") == "assistant"]
    assert assistants, "assistant message must survive conversion"
    assert assistants[0]["reasoning_content"] == "chain of thought"


def test_reasoning_content_absent_when_not_recorded() -> None:
    """No reasoning is attached for messages that never had any."""
    payload = _model()._get_request_payload([HumanMessage(content="u")])
    assert all(
        "reasoning_content" not in m
        for m in payload["messages"]
        if m.get("role") == "assistant"
    )


@pytest.mark.parametrize(
    ("model", "requested", "expected", "downgraded"),
    [
        ("ecnu-max", "max", "max", False),
        ("ecnu-max", "high", "high", False),
        ("ecnu-max", "low", "low", False),
        # ecnu-plus has no `max`; the adapter must report the downgrade rather
        # than silently sending an unsupported tier.
        ("ecnu-plus", "max", "medium", True),
        ("ecnu-plus", "medium", "medium", False),
    ],
)
def test_effort_resolution_is_explicit_about_downgrades(
    model: str, requested: str, expected: str, downgraded: bool
) -> None:
    assert resolve_effort(model, requested) == (expected, downgraded)


def test_unsupported_effort_records_downgrade_on_instance() -> None:
    """The downgrade is observable on the model, not just in the function."""
    model = _model(model="ecnu-plus", reasoning_effort="max")
    assert model.reasoning_effort == "medium"
    assert model.last_downgraded is True


def test_supported_efforts_documented_tiers() -> None:
    assert supported_efforts("ecnu-max") == ("low", "high", "max")
    assert supported_efforts("ecnu-plus") == ("low", "medium")
    # Unknown names fall back to the strictest set rather than a permissive one.
    assert "max" in supported_efforts("some-new-model")


@pytest.mark.parametrize(
    ("base_url", "model", "expected"),
    [
        (ECNU_URL, "ecnu-max", True),
        (ECNU_URL, "deepseek-chat", True),
        ("https://api.deepseek.com/v1", "ecnu-max", True),
        ("https://api.deepseek.com/v1", "deepseek-chat", False),
        ("http://localhost:11434", "qwen3:8b", False),
        (None, None, False),
    ],
)
def test_looks_like_ecnu(
    base_url: str | None, model: str | None, expected: bool
) -> None:
    assert looks_like_ecnu(base_url, model) is expected


def test_settings_infer_provider_from_base_url() -> None:
    assert LLMSettings(base_url=ECNU_URL, model="ecnu-max").is_ecnu is True
    assert LLMSettings(base_url=ECNU_URL, model="x", provider="generic").is_ecnu is False
    assert LLMSettings(base_url="https://api.deepseek.com", model="deepseek-chat").is_ecnu is False


async def test_concurrency_gate_serialises_at_limit_one() -> None:
    """The default cap is 1, so generations cannot overlap."""
    gate = LLMConcurrencyGate(1)
    order: list[str] = []

    async def worker(name: str, delay: float) -> None:
        async with gate:
            order.append(f"{name}-start")
            await asyncio.sleep(delay)
            order.append(f"{name}-end")

    await asyncio.gather(worker("a", 0.05), worker("b", 0.01))

    assert order == ["a-start", "a-end", "b-start", "b-end"]
    assert gate.snapshot()["max_in_flight"] == 1


async def test_concurrency_gate_allows_parallel_when_raised() -> None:
    gate = LLMConcurrencyGate(3)
    order: list[str] = []

    async def worker(name: str) -> None:
        async with gate:
            order.append(f"{name}-start")
            await asyncio.sleep(0.05)
            order.append(f"{name}-end")

    await asyncio.gather(worker("a"), worker("b"))
    # Both enter before either exits when the limit permits it.
    assert order[:2] == ["a-start", "b-start"]


def test_gateway_json_mode_routes_per_tier() -> None:
    """The gateway's chat tier must request JSON; the reasoner tier must not.

    The panel parses a JSON object from every expert, so losing JSON mode on
    the ECNU path silently broke the panel (observed live: a valid-looking
    response that the orchestrator could not parse). The two tiers also need
    separate model instances — a shared cache would let whichever tier was
    called first fix response_format for all later calls.
    """
    from langchain_core.messages import HumanMessage

    from mindflow.agents.llm_gateway import LangChainGateway

    settings = LLMSettings(
        base_url=ECNU_URL, model="ecnu-max", api_key="k", provider="ecnu",
    )
    gateway = LangChainGateway(api_key="k", base_url=ECNU_URL, llm_settings=settings)

    chat = gateway._get_model("deepseek-chat")
    reasoner = gateway._get_model("deepseek-reasoner")

    assert chat is not reasoner, "tiers must not share one cached instance"
    chat_payload = chat._get_request_payload([HumanMessage(content="x")])
    reasoner_payload = reasoner._get_request_payload([HumanMessage(content="x")])
    assert chat_payload["response_format"] == {"type": "json_object"}
    assert "response_format" not in reasoner_payload
    # Both tiers still carry the thinking protocol.
    assert chat_payload["extra_body"]["thinking"] == {"type": "enabled"}
    assert reasoner_payload["extra_body"]["thinking"] == {"type": "enabled"}
    assert chat_payload["reasoning_effort"] == "max"


def test_explicit_base_url_overrides_ambient_ecnu_settings() -> None:
    """An explicitly targeted endpoint must not inherit the ECNU protocol.

    With MINDFLOW_LLM__MODEL=ecnu-max in the environment, a gateway built for
    some other host used to still classify itself as ECNU (the model name alone
    matched), so it sent `thinking` to a server that does not understand it.
    """
    from mindflow.agents.llm_gateway import LangChainGateway

    gateway = LangChainGateway(api_key="test-key", base_url="https://test.api.example.com")

    assert gateway._is_ecnu is False
    assert gateway._base_url == "https://test.api.example.com"


def test_ecnu_base_url_is_still_detected_when_explicit() -> None:
    from mindflow.agents.llm_gateway import LangChainGateway

    gateway = LangChainGateway(api_key="k", base_url=ECNU_URL)
    assert gateway._is_ecnu is True


async def test_concurrency_gate_unbounded() -> None:
    gate = LLMConcurrencyGate(None)
    assert gate.limit is None
    async with gate:
        pass
    assert gate.snapshot()["limit"] is None
