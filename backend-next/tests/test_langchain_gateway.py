"""Tests for LangChainGateway — the ChatDeepSeek-backed LLM gateway.

These tests validate the LangChain migration while keeping the public
``complete(system, user, model) -> str`` interface *and* the ``DeepSeekGateway``
alias intact for existing callers (chat_service, app, eval).

Coverage:
  - Key-less construction is allowed (E2E finding); the error raises at call time.
  - Successful ChatDeepSeek response returns the raw content string.
  - ``model="reasoner"`` does *not* send ``response_format: json_object``
    (unsupported by deepseek-reasoner).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from mindflow.agents.llm_gateway import GatewayNotConfiguredError, LangChainGateway

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _no_ssl_cert_file_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset ``SSL_CERT_FILE`` so real ``ChatDeepSeek`` construction

    (which eagerly builds an httpx client + SSL context via ``trust_env``)
    never fails on a machine where the variable points to an invalid path.
    """
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)


def _make_aimessage(content: str = "{}") -> AIMessage:
    """Build a fake ``AIMessage`` with the given *content*."""
    return AIMessage(content=content)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: construction without key
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoKey:
    """Key-less construction must succeed; error is deferred to call time."""

    async def test_construction_without_key_does_not_raise(self) -> None:
        """``LangChenGateway(api_key="")`` succeeds; ``complete()`` raises."""
        gateway = LangChainGateway(api_key="")
        assert gateway._api_key == ""

        with pytest.raises(GatewayNotConfiguredError, match="API key is not configured"):
            await gateway.complete("system", "user")

    async def test_close_no_key_is_safe(self) -> None:
        """Calling ``close()`` on a key-less gateway is a no-op."""
        gateway = LangChainGateway(api_key="")
        await gateway.close()
        # No exception means pass


# ═══════════════════════════════════════════════════════════════════════════════
# Test: mock ChatDeepSeek response
# ═══════════════════════════════════════════════════════════════════════════════


class TestMockResponse:
    """Simulate a successful LangChain invocation."""

    @pytest.mark.asyncio
    async def test_mock_chat_response(self) -> None:
        """Mock ``ChatDeepSeek.ainvoke`` returns content the gateway passes through."""
        gateway = LangChainGateway(api_key="test-key", base_url="https://test.api.example.com")

        with patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke:
            mock_ainvoke.return_value = _make_aimessage('{"result": "ok"}')

            result = await gateway.complete("Be helpful.", "Hello")

        assert result == '{"result": "ok"}'



# ═══════════════════════════════════════════════════════════════════════════════
# Test: model="reasoner" does not set json_object
# ═══════════════════════════════════════════════════════════════════════════════


class TestClose:
    """C2: close() releases the underlying httpx pools, not just drops refs."""

    @pytest.mark.asyncio
    async def test_close_awaits_root_async_client(self) -> None:
        """close() awaits root_async_client.close() on each built model."""
        gateway = LangChainGateway(api_key="test-key", base_url="https://test.api.example.com")

        # Build the chat model so there's a real ChatDeepSeek to close.
        with patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke:
            mock_ainvoke.return_value = _make_aimessage("{}")
            await gateway.complete("system", "user", model="chat")

        model = gateway._chat_model
        assert model is not None

        # Swap the eagerly-built async client for one whose close() we can track.
        tracker = AsyncMock()
        model.root_async_client = tracker  # type: ignore[attr-defined]
        # Neutralise the sync client so its real close() doesn't run.
        model.root_client = MagicMock()  # type: ignore[attr-defined]

        await gateway.close()

        tracker.close.assert_awaited_once()
        # References are dropped after closing.
        assert gateway._chat_model is None
        assert gateway._reasoner_model is None

    @pytest.mark.asyncio
    async def test_close_with_no_models_is_safe(self) -> None:
        """close() before any model was built is a no-op."""
        gateway = LangChainGateway(api_key="test-key", base_url="https://test.api.example.com")
        assert gateway._chat_model is None
        await gateway.close()  # must not raise


class TestReasonerNoJsonObject:
    """deepseek-reasoner does not support ``response_format: json_object``."""

    @pytest.mark.asyncio
    async def test_reasoner_no_json_object(self) -> None:
        """ChatDeepSeek is created without ``model_kwargs`` for reasoner."""
        gateway = LangChainGateway(api_key="test-key", base_url="https://test.api.example.com")

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
            mock_ainvoke.return_value = _make_aimessage("{}")
            await gateway.complete("system", "user", model="reasoner")

        # The reasoner model should not have response_format in model_kwargs
        model_kwargs = init_kwargs.get("model_kwargs", {})
        assert isinstance(model_kwargs, dict)
        assert "response_format" not in model_kwargs

    @pytest.mark.asyncio
    async def test_chat_has_json_object(self) -> None:
        """Sanity check: the ``chat`` tier *does* set ``response_format``."""
        gateway = LangChainGateway(api_key="test-key", base_url="https://test.api.example.com")

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
            mock_ainvoke.return_value = _make_aimessage("{}")
            await gateway.complete("system", "user", model="chat")

        model_kwargs = init_kwargs.get("model_kwargs", {})
        assert isinstance(model_kwargs, dict)
        assert model_kwargs.get("response_format") == {"type": "json_object"}


# ═══════════════════════════════════════════════════════════════════════════════
# Test: DeepSeekGateway alias
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlias:
    """``DeepSeekGateway`` is a backward-compatible alias for ``LangChainGateway``."""

    def test_alias_is_lang_chain_gateway(self) -> None:
        from mindflow.agents.llm_gateway import DeepSeekGateway

        assert DeepSeekGateway is LangChainGateway
