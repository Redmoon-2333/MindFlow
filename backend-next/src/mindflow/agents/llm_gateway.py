"""LLM gateway protocol and implementation for the expert panel.

Provides a generic gateway that is **not** bound to any specific output schema
(unlike the existing ``DeepSeekClient`` which hard-codes ``LLMAttributionResult``).
The panel uses this gateway to call arbitrary experts with arbitrary system prompts.

Design:
  - ``PanelLLMGateway`` is a typing.Protocol — the orchestrator depends on the
    interface, not the implementation, making it trivially testable with mocks.
  - ``LangChainGateway`` wraps ``ChatDeepSeek`` from ``langchain-deepseek``
    and reuses ``Settings.llm`` configuration (api_key, base_url) via the same
    pattern as the legacy ``DeepSeekClient``.
  - ``model_kwargs: {"response_format": {"type": "json_object"}}`` is used for
    ``chat`` model calls; ``reasoner`` model calls omit this (deepseek-reasoner
    does not support it).

Raises:
    ``GatewayNotConfiguredError``: At call time when no API key is available.
    ``GatewayAPIError``: After exhausting the retry budget.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from loguru import logger
from pydantic import SecretStr

from mindflow.config import get_settings

# ── Custom exceptions ──────────────────────────────────────────────────────────


class GatewayNotConfiguredError(RuntimeError):
    """Raised when the gateway is called without an API key being configured."""


class GatewayAPIError(RuntimeError):
    """Raised when the upstream API returns a non-retriable error
    or the retry budget has been exhausted."""


# ── Protocol ───────────────────────────────────────────────────────────────────


@runtime_checkable
class PanelLLMGateway(Protocol):
    """Protocol for LLM gateways used by the expert panel.

    The panel depends on this interface, not on any concrete implementation.
    This makes it straightforward to inject mock gateways in tests.
    """

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",
    ) -> str:
        """Send a completion request and return the response content.

        Args:
            system: System prompt defining the expert's persona and output contract.
            user: User message containing the evidence data and context.
            model: Model tier — "chat" (deepseek-chat) or "reasoner" (deepseek-reasoner).

        Returns:
            The raw response content as a string (expected to be valid JSON).

        Raises:
            GatewayNotConfiguredError: If not configured.
            GatewayAPIError: Non-retriable API error or retries exhausted.
        """
        ...

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        ...


# ── LangChain implementation ──────────────────────────────────────────────────

_DEFAULT_TIMEOUT_S: int = 30
_MAX_RETRIES: int = 1


class LangChainGateway:
    """Async LLM gateway wrapping LangChain's ``ChatDeepSeek``.

    Unlike ``DeepSeekClient`` (which binds to ``LLMAttributionResult``), this
    gateway returns raw response text. The caller (orchestrator) handles parsing.

    Args:
        api_key: DeepSeek API key. If None, reads from ``Settings``.
        base_url: API base URL. If None, reads from ``Settings``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if api_key is None:
            settings = get_settings()
            api_key = settings.llm.api_key
            base_url = base_url or settings.llm.base_url

        # Key-less construction is allowed (E2E finding): the app must be able
        # to assemble PanelService/ChatService without a configured key so the
        # degradation chain (panel->single_expert->rule_engine, chat->safe reply)
        # stays reachable. The raise happens at call time in complete().
        self._api_key = api_key or ""
        self._base_url = (base_url or "https://api.deepseek.com").rstrip("/")

        # Lazy-initialised ChatDeepSeek instances (one per model tier).
        self._chat_model: ChatDeepSeek | None = None
        self._reasoner_model: ChatDeepSeek | None = None

    def _get_model(self, model_id: str) -> ChatDeepSeek:
        """Return a cached ``ChatDeepSeek`` instance for *model_id*.

        The ``chat`` tier (``deepseek-chat``) is created with
        ``response_format: json_object``; the ``reasoner`` tier
        (``deepseek-reasoner``) does not support this parameter.
        """
        if model_id == "deepseek-chat":
            if self._chat_model is None:
                self._chat_model = ChatDeepSeek(
                    model=model_id,
                    api_key=SecretStr(self._api_key) if self._api_key else None,
                    base_url=self._base_url,
                    timeout=_DEFAULT_TIMEOUT_S,
                    max_retries=0,
                    model_kwargs={"response_format": {"type": "json_object"}},
                )
            return self._chat_model

        # model_id == "deepseek-reasoner" (no response_format)
        if self._reasoner_model is None:
            self._reasoner_model = ChatDeepSeek(
                model=model_id,
                api_key=SecretStr(self._api_key) if self._api_key else None,
                base_url=self._base_url,
                timeout=_DEFAULT_TIMEOUT_S,
                max_retries=0,
            )
        return self._reasoner_model

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",
    ) -> str:
        """Send a completion request and return the response content as raw text.

        Raises GatewayNotConfiguredError at call time if no key was supplied
        (deferred from __init__ — E2E finding: the app must assemble services
        without a key so degradation paths stay reachable).

        Args:
            system: System prompt.
            user: User message.
            model: "chat" -> deepseek-chat (with json_object mode),
                   "reasoner" -> deepseek-reasoner (no json_object mode).

        Returns:
            Raw response content string.

        Raises:
            GatewayNotConfiguredError: If not configured.
            GatewayAPIError: After exhausting retries.
        """
        model_id = "deepseek-chat" if model == "chat" else "deepseek-reasoner"

        if not self._api_key:
            raise GatewayNotConfiguredError(
                "DeepSeek API key is not configured — set MINDFLOW_LLM__API_KEY "
                "or add llm.api_key to the .env file"
            )

        chat = self._get_model(model_id)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                result = await chat.ainvoke(messages)
            except Exception as exc:
                logger.warning("LangChain gateway error (attempt {}): {}", attempt + 1, exc)
                last_exc = exc
                continue

            raw_content = result.content
            content: str = raw_content if isinstance(raw_content, str) else ""
            if not content:
                logger.warning("LangChain gateway returned empty content")
                last_exc = GatewayAPIError("Empty content in response")
                continue

            return content

        # All retries exhausted
        raise GatewayAPIError(
            f"LangChain gateway failed after {_MAX_RETRIES + 1} attempts"
        ) from last_exc

    async def close(self) -> None:
        """Release the httpx connection pools held by the ChatDeepSeek models.

        ``ChatDeepSeek`` wraps an ``openai.AsyncOpenAI`` client (exposed as
        ``root_async_client``) that owns a long-lived httpx pool. Dropping the
        reference alone does NOT close it promptly — it lingers until GC, which
        leaks sockets on repeated gateway recreation (tests, eval runs). So we
        await the client's own ``close()`` (a coroutine that shuts the pool)
        before releasing references (review C2 connection leak).
        """
        import contextlib

        for model in (self._chat_model, self._reasoner_model):
            if model is None:
                continue
            # root_async_client is the AsyncOpenAI instance; its close() is a
            # coroutine that releases the underlying httpx AsyncClient pool.
            async_client = getattr(model, "root_async_client", None)
            if async_client is not None and hasattr(async_client, "close"):
                with contextlib.suppress(Exception):
                    await async_client.close()
            # The sync root_client (rarely built) holds a separate pool.
            sync_client = getattr(model, "root_client", None)
            if sync_client is not None and hasattr(sync_client, "close"):
                with contextlib.suppress(Exception):
                    sync_client.close()

        self._chat_model = None
        self._reasoner_model = None


# ── Backward-compatible alias ─────────────────────────────────────────────────
# Callers (chat_service, app, eval) import ``DeepSeekGateway`` by name.
# The rename to ``LangChainGateway`` reflects the new implementation backbone
# while keeping dependent modules untouched.

DeepSeekGateway = LangChainGateway
