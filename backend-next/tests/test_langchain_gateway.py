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

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from mindflow.agents.llm_gateway import GatewayNotConfiguredError, LangChainGateway

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


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
