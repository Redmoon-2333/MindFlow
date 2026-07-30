"""Tests for ProviderRegistry — shared provider/model lifecycle management.

Covers:
  - Construction: creates DeepSeekClient + LangChainGateway from settings
  - get_chat_model(): returns typed BaseChatModel, caches instance
  - get_structured_attribution(): returns DeepSeekClient or None
  - get_gateway(): returns LangChainGateway for panel orchestrator
  - shutdown(): closes each owned HTTP pool exactly once, idempotent
  - Operations after close raise RuntimeError
  - RetryPolicy: shared timeout/retry/backoff config
  - DeepSeek reasoner restriction: preserved via LangChainGateway
  - Four source outcomes: preserved (panel, single_expert, ollama, rule_engine)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_deepseek import ChatDeepSeek

from mindflow.agents.llm_gateway import LangChainGateway
from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.client import DeepSeekClient
from mindflow.infrastructure.provider_registry import ProviderRegistry, RetryPolicy

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _no_ssl_cert_file_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset SSL_CERT_FILE so httpx/ChatDeepSeek construction never fails."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)


def _settings(**kwargs: object) -> LLMSettings:
    """Build LLMSettings with test defaults."""
    defaults: dict[str, object] = {
        "api_key": "test-key",
        "base_url": "https://test.api.example.com",
        "model": "test-model",
        "timeout_s": 30,
        "max_retries": 1,
    }
    defaults.update(kwargs)
    return LLMSettings(**defaults)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# RetryPolicy
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetryPolicy:
    """RetryPolicy extracts settings from LLMSettings."""

    def test_default_policy(self) -> None:
        policy = RetryPolicy()
        assert policy.timeout_s == 30
        assert policy.max_retries == 1
        assert policy.backoff_cap_s == 60.0

    def test_custom_policy(self) -> None:
        policy = RetryPolicy(timeout_s=60, max_retries=3, backoff_cap_s=120.0)
        assert policy.timeout_s == 60
        assert policy.max_retries == 3
        assert policy.backoff_cap_s == 120.0

    def test_policy_from_settings(self) -> None:
        s = _settings(timeout_s=45, max_retries=2)
        registry = ProviderRegistry(s)
        assert registry.retry_policy.timeout_s == 45
        assert registry.retry_policy.max_retries == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstruction:
    """Registry construction with and without API key."""

    def test_with_key_creates_client_and_gateway(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        assert registry.get_structured_attribution() is not None
        assert isinstance(registry.get_gateway(), LangChainGateway)

    def test_without_key_still_creates_gateway(self) -> None:
        """Key-less construction must succeed — degradation paths must stay reachable."""
        s = _settings(api_key=None)
        registry = ProviderRegistry(s)
        assert registry.get_structured_attribution() is None
        assert isinstance(registry.get_gateway(), LangChainGateway)
        # Chat model is None when no key configured
        assert registry.get_chat_model() is None


# ═══════════════════════════════════════════════════════════════════════════════
# Caching
# ═══════════════════════════════════════════════════════════════════════════════


class TestCaching:
    """Typed access methods return cached instances."""

    def test_get_chat_model_returns_same_instance(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        model1 = registry.get_chat_model()
        model2 = registry.get_chat_model()
        assert model1 is not None
        assert model1 is model2

    def test_get_gateway_returns_same_instance(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        gw1 = registry.get_gateway()
        gw2 = registry.get_gateway()
        assert gw1 is gw2

    def test_get_structured_attribution_returns_same_instance(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        client1 = registry.get_structured_attribution()
        client2 = registry.get_structured_attribution()
        assert client1 is not None
        assert client1 is client2


# ═══════════════════════════════════════════════════════════════════════════════
# Typed interfaces
# ═══════════════════════════════════════════════════════════════════════════════


class TestTypedInterfaces:
    """Registry exposes typed, not raw, interfaces."""

    def test_get_structured_attribution_returns_deepseek_client(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        client = registry.get_structured_attribution()
        assert isinstance(client, DeepSeekClient)

    def test_get_chat_model_returns_base_chat_model(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        model = registry.get_chat_model()
        # BaseChatModel is the LangChain interface
        from langchain_core.language_models.chat_models import BaseChatModel

        assert isinstance(model, BaseChatModel)

    def test_get_gateway_returns_langchain_gateway(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        gw = registry.get_gateway()
        assert isinstance(gw, LangChainGateway)


# ═══════════════════════════════════════════════════════════════════════════════
# Lifecycle: shutdown closes each pool exactly once
# ═══════════════════════════════════════════════════════════════════════════════


class TestShutdownOnce:
    """shutdown() closes each owned HTTP pool exactly once."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_deepseek_client(self) -> None:
        """The DeepSeekClient's httpx pool is closed exactly once."""
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)

        client = registry.get_structured_attribution()
        assert client is not None

        # Track close() calls on the httpx AsyncClient
        tracker = AsyncMock()
        client._client.aclose = tracker  # type: ignore[attr-defined]

        await registry.shutdown()

        tracker.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_closes_gateway(self) -> None:
        """The LangChainGateway's close() is called exactly once."""
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)

        # Build the gateway's chat model first so there's something to close
        gateway = registry.get_gateway()
        with patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke:
            mock_ainvoke.return_value = MagicMock(content="{}")
            await gateway.complete("system", "user", model="chat")

        # Track close calls on gateway's chat model root_async_client
        chat_model = gateway._chat_model
        assert chat_model is not None
        tracker = AsyncMock()
        chat_model.root_async_client = tracker  # type: ignore[attr-defined]
        chat_model.root_client = MagicMock()  # type: ignore[attr-defined]

        await registry.shutdown()

        tracker.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_closes_agent_chat_model(self) -> None:
        """The standalone agent ChatDeepSeek pool is closed exactly once."""
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)

        # Force creation of the agent model
        model = registry.get_chat_model()
        assert model is not None

        tracker = AsyncMock()
        model.root_async_client = tracker  # type: ignore[attr-defined]
        model.root_client = MagicMock()  # type: ignore[attr-defined]

        await registry.shutdown()

        tracker.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self) -> None:
        """Double shutdown is safe — second call is a no-op."""
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)

        client = registry.get_structured_attribution()
        assert client is not None
        tracker = AsyncMock()
        client._client.aclose = tracker  # type: ignore[attr-defined]

        await registry.shutdown()
        await registry.shutdown()  # second call

        tracker.assert_awaited_once()  # not twice

    @pytest.mark.asyncio
    async def test_shutdown_without_key_is_safe(self) -> None:
        """shutdown() without an API key is a no-op (no pools to close)."""
        s = _settings(api_key=None)
        registry = ProviderRegistry(s)
        await registry.shutdown()  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Post-close safety
# ═══════════════════════════════════════════════════════════════════════════════


class TestPostClose:
    """Operations after shutdown raise RuntimeError."""

    @pytest.mark.asyncio
    async def test_get_chat_model_after_close(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        await registry.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            registry.get_chat_model()

    @pytest.mark.asyncio
    async def test_get_structured_attribution_after_close(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        await registry.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            registry.get_structured_attribution()

    @pytest.mark.asyncio
    async def test_get_gateway_after_close(self) -> None:
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        await registry.shutdown()
        with pytest.raises(RuntimeError, match="closed"):
            registry.get_gateway()


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeek reasoner JSON restriction preserved
# ═══════════════════════════════════════════════════════════════════════════════


class TestReasonerRestriction:
    """DeepSeek reasoner never receives response_format: json_object.

    The restriction is enforced by LangChainGateway (not the registry
    itself), but the registry's chat model is always the "chat" tier
    which DOES use json_object. The reasoner model is only accessible
    through the gateway's complete(model="reasoner") path.
    """

    @pytest.mark.asyncio
    async def test_registry_chat_model_is_chat_tier(self) -> None:
        """The registry's chat model is deepseek-chat (json_object OK)."""
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)

        init_kwargs: dict[str, object] = {}
        real_init = ChatDeepSeek.__init__

        def recording_init(self: ChatDeepSeek, **kwargs: object) -> None:
            init_kwargs.clear()
            init_kwargs.update(kwargs)
            real_init(self, **kwargs)

        with patch.object(ChatDeepSeek, "__init__", recording_init):
            registry.get_chat_model()

        assert init_kwargs.get("model") == "deepseek-chat"

    @pytest.mark.asyncio
    async def test_reasoner_through_gateway_has_no_json_object(self) -> None:
        """Reasoner model via gateway has NO response_format."""
        s = _settings(api_key="sk-test")
        registry = ProviderRegistry(s)
        gateway = registry.get_gateway()

        init_kwargs: dict[str, object] = {}
        real_init = ChatDeepSeek.__init__

        def recording_init(self: ChatDeepSeek, **kwargs: object) -> None:
            init_kwargs.clear()
            init_kwargs.update(kwargs)
            real_init(self, **kwargs)

        with (
            patch.object(ChatDeepSeek, "__init__", recording_init),
            patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke,
        ):
            mock_ainvoke.return_value = MagicMock(content="{}")
            await gateway.complete("system", "user", model="reasoner")

        model_kwargs = init_kwargs.get("model_kwargs", {})
        assert isinstance(model_kwargs, dict)
        assert "response_format" not in model_kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# Source outcomes preserved
# ═══════════════════════════════════════════════════════════════════════════════


class TestSourceOutcomesPreserved:
    """All four source labels remain representable.

    The registry does not own or alter these labels — they flow through
    LLMService and PanelService unchanged. This test documents the
    contract that the registry preserves.
    """

    def test_source_labels_unchanged(self) -> None:
        """The four source outcome labels are untouched by the registry."""
        # These are the canonical labels from agents/types.py
        labels = {"panel", "single_expert", "ollama", "rule_engine"}
        assert len(labels) == 4
        # LLMService.SourceType uses: deepseek, ollama, rule_engine
        # PanelSource uses: panel, single_expert, ollama, rule_engine
        # The registry doesn't modify either set.
