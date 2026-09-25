"""Configuration-level regressions for the DeepSeek-direct L1 contract.

Production L1 is DeepSeek direct: ``provider="generic"``,
``base_url="https://api.deepseek.com"``, ``model="deepseek-flash"`` and a
credential resolved from ``DEEPSEEK_API_KEY`` (or the prefixed
``MINDFLOW_LLM__DEEPSEEK_API_KEY``). These tests pin the *resolution* rules and
the two invariants that make the switch safe:

* a legacy ECNU triple (campus URL + ``ecnu-*`` model + campus key) is
  overridden, so an ECNU endpoint can never be mixed with a DeepSeek model or
  key — and its key is never sent to DeepSeek;
* a missing DeepSeek credential makes L1 unavailable: the chain degrades to
  L2/L3 and never borrows the campus key or falls back to the campus host.

``ecnu_compat_enabled=True`` is the only way back to the legacy endpoint; the
ECNU adapter's compatibility suites opt in explicitly.

Offline: requests only ever travel over the injected mock wire.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

import httpx
import pytest
from loguru import logger

from mindflow.agents.llm_gateway import GatewayNotConfiguredError
from mindflow.agents.policies import CHAT_POLICY
from mindflow.config import LLMSettings, Settings
from mindflow.infrastructure.llm.ecnu import ECNUChatModel
from mindflow.infrastructure.provider_registry import ProviderRegistry
from tests._llm_test_support import MockLLMWire, chat_completion

ECNU_URL = "https://chat.ecnu.edu.cn/open/api/v1"
DEEPSEEK_HOST = "api.deepseek.com"
PINNED_MODEL = "deepseek-flash"

#: Sentinels only — never a real credential.
ECNU_KEY_SENTINEL = "sk-sentinel-ecnu-legacy"
DEEPSEEK_KEY_SENTINEL = "sk-sentinel-deepseek"
PREFIXED_KEY_SENTINEL = "sk-sentinel-deepseek-prefixed"


@pytest.fixture(autouse=True)
def _no_ambient_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve credentials from this test only, never from the machine's env."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("MINDFLOW_LLM__DEEPSEEK_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _no_ssl_cert_file_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset ``SSL_CERT_FILE`` so SDK client construction never fails."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)


def _legacy_ecnu(**overrides: object) -> LLMSettings:
    """The legacy campus triple: ECNU URL, ``ecnu-max``, provider ``ecnu``."""
    defaults: dict[str, object] = {
        "api_key": ECNU_KEY_SENTINEL,
        "base_url": ECNU_URL,
        "model": "ecnu-max",
        "provider": "ecnu",
        "timeout_s": 30,
        "max_retries": 0,
        "max_concurrent_requests": 1,
    }
    defaults.update(overrides)
    return LLMSettings(**defaults)  # type: ignore[arg-type]


def _wire() -> MockLLMWire:
    async def _handler(request: httpx.Request) -> httpx.Response:
        return chat_completion("{}", model=PINNED_MODEL)

    return MockLLMWire(_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. What a legacy environment resolves to
# ═══════════════════════════════════════════════════════════════════════════════


def test_legacy_ecnu_env_resolves_to_the_deepseek_pin() -> None:
    """An ECNU URL/model/key in the environment cannot survive the pin."""
    settings = _legacy_ecnu()
    target = settings.l1_target()

    assert target.provider == "generic"
    assert target.model == PINNED_MODEL
    assert urlparse(target.base_url).hostname == DEEPSEEK_HOST
    assert target.provenance == "deepseek-direct"
    # The campus key is not borrowed: without a DeepSeek credential L1 is down.
    assert target.api_key is None
    assert target.describe()["credential_present"] is False
    # ``is_ecnu`` keeps its historical meaning ("describes the campus gateway");
    # it is no longer the production selector.
    assert settings.is_ecnu is True
    assert settings.deepseek_credential is None


def test_legacy_ecnu_env_vars_resolve_to_the_deepseek_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lingering ECNU ``.env`` is overridden even through ``Settings``."""
    monkeypatch.setenv("MINDFLOW_LLM__API_KEY", ECNU_KEY_SENTINEL)
    monkeypatch.setenv("MINDFLOW_LLM__BASE_URL", ECNU_URL)
    monkeypatch.setenv("MINDFLOW_LLM__MODEL", "ecnu-max")
    monkeypatch.setenv("MINDFLOW_LLM__PROVIDER", "ecnu")

    settings = Settings(_env_file=None).llm
    assert settings.base_url == ECNU_URL and settings.model == "ecnu-max"
    assert settings.is_ecnu is True

    target = settings.l1_target()
    assert target.provenance == "deepseek-direct"
    assert target.model == PINNED_MODEL
    assert urlparse(target.base_url).hostname == DEEPSEEK_HOST
    # The ECNU-issued key is not borrowed, and nothing falls back to the campus.
    assert target.api_key is None
    assert target.describe()["credential_present"] is False
    assert settings.deepseek_credential is None


def test_deepseek_shaped_legacy_config_keeps_its_key_but_is_pinned() -> None:
    """A config that already pointed at DeepSeek keeps its key, not its model."""
    settings = LLMSettings(
        api_key=DEEPSEEK_KEY_SENTINEL,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
    )
    target = settings.l1_target()

    assert settings.is_ecnu is False
    assert settings.deepseek_api_key is None  # nothing dedicated was configured
    assert settings.deepseek_credential == DEEPSEEK_KEY_SENTINEL
    assert target.api_key == DEEPSEEK_KEY_SENTINEL
    assert target.describe()["credential_present"] is True
    # Still pinned: the model id is the resolved L1 model, not the legacy one.
    assert target.model == PINNED_MODEL
    assert urlparse(target.base_url).hostname == DEEPSEEK_HOST


def test_bare_deepseek_api_key_is_picked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DEEPSEEK_API_KEY`` is honoured without duplicating it into the app env."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", DEEPSEEK_KEY_SENTINEL)

    settings = _legacy_ecnu()
    assert settings.deepseek_api_key == DEEPSEEK_KEY_SENTINEL
    target = settings.l1_target()
    assert target.api_key == DEEPSEEK_KEY_SENTINEL
    assert target.describe()["credential_present"] is True


def test_prefixed_env_var_wins_over_the_bare_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MINDFLOW_LLM__DEEPSEEK_API_KEY`` outranks the conventional variable."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", DEEPSEEK_KEY_SENTINEL)
    monkeypatch.setenv("MINDFLOW_LLM__DEEPSEEK_API_KEY", PREFIXED_KEY_SENTINEL)

    # The prefixed name is the application's own; the bare one is the fallback,
    # so both a standalone ``LLMSettings`` and the nested ``Settings`` take it.
    assert LLMSettings().deepseek_api_key == PREFIXED_KEY_SENTINEL

    settings = Settings(_env_file=None).llm
    assert settings.deepseek_api_key == PREFIXED_KEY_SENTINEL
    assert settings.l1_target().api_key == PREFIXED_KEY_SENTINEL


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Missing credential ⇒ L1 unavailable, no campus fallback
# ═══════════════════════════════════════════════════════════════════════════════


async def test_missing_credential_degrades_without_touching_ecnu() -> None:
    """No DeepSeek key: both L1 accessors are ``None`` and nothing is sent."""
    wire = _wire()
    with wire.patch_async_client():
        registry = ProviderRegistry(_legacy_ecnu())
        try:
            assert registry.get_structured_attribution() is None
            assert registry.get_chat_model() is None
            described = registry.describe()
            assert described["credential_present"] is False
            assert described["provenance"] == "deepseek-direct"
            assert described["base_url_host"] == DEEPSEEK_HOST
            # The key-less gateway stays constructible (degradation reachable),
            # and refuses at call time instead of dialling the campus host.
            with pytest.raises(GatewayNotConfiguredError):
                await registry.get_gateway().complete("system", "user")
        finally:
            await registry.shutdown()

    assert wire.recorder.total_requests == 0
    assert all("ecnu.edu.cn" not in url for url in wire.recorder.urls)


async def test_credential_never_reaches_describe_or_logs() -> None:
    """The resolved credential is really used, yet rendered nowhere."""
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(str(message)),
        format="{message}", backtrace=False, diagnose=False, level="DEBUG",
    )
    rendered = ""
    try:
        registry = ProviderRegistry(_legacy_ecnu(deepseek_api_key=DEEPSEEK_KEY_SENTINEL))
        try:
            # L1 is up, so the absence of the sentinel below means "never
            # rendered" rather than "never configured".
            assert registry.get_structured_attribution() is not None
            assert registry.get_chat_model() is not None
            rendered = json.dumps(registry.describe(), ensure_ascii=False, default=str)
            rendered += json.dumps(
                registry.l1_target.describe(), ensure_ascii=False, default=str,
            )
        finally:
            await registry.shutdown()
    finally:
        logger.remove(sink)

    logs = "\n".join(messages)
    assert any("L1 resolved" in message for message in messages), logs
    assert PINNED_MODEL in logs, "the resolved L1 must be logged"
    assert DEEPSEEK_KEY_SENTINEL not in rendered
    assert DEEPSEEK_KEY_SENTINEL not in logs


# ═══════════════════════════════════════════════════════════════════════════════
# 3. The compat switch, and one model per tier
# ═══════════════════════════════════════════════════════════════════════════════


async def test_ecnu_compat_switch_restores_the_legacy_triple() -> None:
    """``ecnu_compat_enabled=True`` is the documented way back to the campus."""
    settings = _legacy_ecnu(ecnu_compat_enabled=True)
    target = settings.l1_target()

    assert target.provenance == "ecnu-compat"
    assert target.provider == "ecnu"
    assert target.model == "ecnu-max"
    assert target.base_url == ECNU_URL
    assert target.api_key == ECNU_KEY_SENTINEL

    registry = ProviderRegistry(settings)
    try:
        assert isinstance(registry.get_chat_model(), ECNUChatModel)
        assert registry.get_gateway()._is_ecnu is True
        assert registry.describe()["base_url_host"] == "chat.ecnu.edu.cn"
    finally:
        await registry.shutdown()


async def test_both_tiers_request_the_resolved_model_and_differ_only_in_json_mode() -> None:
    """``chat``/``reasoner`` are output policies: one model, JSON vs prose."""
    wire = _wire()
    with wire.patch_async_client():
        registry = ProviderRegistry(_legacy_ecnu(deepseek_api_key=DEEPSEEK_KEY_SENTINEL))
        gateway = registry.get_gateway()
        try:
            await gateway.complete("system", "user", model="chat")
            await gateway.complete("system", "user", model="reasoner")
        finally:
            await registry.shutdown()

    assert wire.recorder.total_requests == 2
    chat_body, prose_body = wire.recorder.payloads
    assert chat_body["model"] == prose_body["model"] == PINNED_MODEL
    assert chat_body["response_format"] == {"type": "json_object"}
    assert "response_format" not in prose_body
    # Both requests went to the pinned host, never to the legacy one.
    assert all(DEEPSEEK_HOST in url for url in wire.recorder.urls)
    assert all("ecnu.edu.cn" not in url for url in wire.recorder.urls)


async def test_chat_policy_sends_only_its_cap_on_the_pinned_wire() -> None:
    """Chat keeps the provider default effort: cap only, no effort/thinking."""
    wire = _wire()
    with wire.patch_async_client():
        registry = ProviderRegistry(_legacy_ecnu(deepseek_api_key=DEEPSEEK_KEY_SENTINEL))
        gateway = registry.get_gateway()
        try:
            await gateway.complete("system", "user", model="chat", policy=CHAT_POLICY)
        finally:
            await registry.shutdown()

    body = wire.recorder.last
    assert body["model"] == PINNED_MODEL
    assert body["max_tokens"] == CHAT_POLICY.max_output_tokens == 2048
    # DeepSeek documents thinking on / effort ``high`` as its own default, so
    # the chat row must not override either.
    assert "reasoning_effort" not in body
    assert "thinking" not in body
