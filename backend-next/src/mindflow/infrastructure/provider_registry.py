"""Provider/model registry and lifecycle owner for LLM consumers.

Centralizes model construction, HTTP client pool management, and
retry/backoff/timeout policy shared by chat, panel, and structured
attribution callers.

Created once during application startup (app.py lifespan), injected
into LLMService, ChatService, and PanelService. Owns all HTTP client
pools and ensures each is closed exactly once on shutdown.

Typed access through separate interfaces:
  - ``get_chat_model()`` → ``BaseChatModel`` for LangChain agent usage
  - ``get_structured_attribution()`` → ``DeepSeekClient`` for typed
    ``LLMAttributionResult`` calls (L1 of degradation chain)
  - ``get_gateway()`` → ``LangChainGateway`` for panel orchestrator

Design constraints:
  - DeepSeek reasoner models never receive ``response_format: json_object``
    (preserved via LangChainGateway's own model tier routing).
  - Ollama/RuleEngine fallback chain is owned by LLMService, not by
    the registry — the registry only manages DeepSeek API clients.
  - ``LLMAttributionResult`` remains typed throughout; no raw strings.
  - Source outcome labels (panel, single_expert, ollama, rule_engine)
    are not altered by the registry.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from loguru import logger
from pydantic import SecretStr

from mindflow.agents.llm_gateway import LangChainGateway
from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.client import DeepSeekClient
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.ecnu import build_ecnu_model
from mindflow.infrastructure.llm.http_client import ProviderHTTPClient
from mindflow.infrastructure.llm.safety import safe_error_metadata

# ── Shared policy ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetryPolicy:
    """Shared timeout/retry/backoff configuration for all LLM consumers.

    Attributes:
        timeout_s: Request timeout in seconds (1-300).
        max_retries: Maximum retry attempts (0-10).
        backoff_cap_s: Upper bound for exponential backoff delay.
    """

    timeout_s: int = 180
    max_retries: int = 1
    backoff_cap_s: float = 60.0


def _host_of(url: str | None) -> str | None:
    """Return just the hostname of *url* — never leaks a key or a full path."""
    if not url:
        return None
    from urllib.parse import urlparse

    return urlparse(url).hostname


# ── Registry ──────────────────────────────────────────────────────────────────


class ProviderRegistry:
    """Shared provider/model registry with lifecycle ownership.

    Created once during application startup and injected into services.
    Owns all HTTP client pools; ``shutdown()`` closes each exactly once.

    Args:
        settings: LLM configuration (api_key, base_url, model, retry params).
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self.retry_policy = RetryPolicy(
            timeout_s=settings.timeout_s,
            max_retries=settings.max_retries,
        )
        # Every consumer shares one gate so the campus gateway's per-user
        # concurrency cap is respected no matter which entry point is calling.
        self.concurrency = LLMConcurrencyGate(settings.max_concurrent_requests)

        # ── Structured attribution client (L1 of degradation chain) ──────────
        self._deepseek_client: DeepSeekClient | None = None
        if settings.api_key:
            try:
                self._deepseek_client = DeepSeekClient(settings, self.concurrency)
                logger.debug("ProviderRegistry: structured attribution client created")
            except Exception as exc:
                logger.warning(
                    "ProviderRegistry: failed to create structured client: {}",
                    safe_error_metadata(exc),
                )

        # ── LangChainGateway for panel orchestrator and chat ────────────────────
        # Key-less construction is allowed: the gateway defers the key check to
        # call time so degradation paths stay reachable.
        self._gateway = LangChainGateway(
            api_key=settings.api_key or "",
            base_url=settings.base_url,
            timeout_s=settings.timeout_s,
            max_retries=settings.max_retries,
            llm_settings=settings,
            concurrency=self.concurrency,
        )
        logger.debug(
            "ProviderRegistry: gateway created (provider={}, thinking={}, effort={})",
            "ecnu" if settings.is_ecnu else "generic",
            settings.thinking_enabled,
            settings.reasoning_effort,
        )

        # ── Standalone chat model for ChatService's LangGraph agent ─────────────
        # Built separately from the gateway's own models so the agent can be
        # configured with its own output cap without affecting the gateway.
        self._chat_model: BaseChatModel | None = None

        self._closed = False

    # ── Typed access interfaces ──────────────────────────────────────────────

    def get_chat_model(self) -> BaseChatModel | None:
        """Return a cached chat model for LangGraph/Tool-calling usage.

        Returns ``None`` when no API key is configured so the degradation
        path (LLM-down safe reply) stays reachable.

        For the campus gateway this is an :class:`ECNUChatModel`, which adds
        the thinking fields and preserves ``reasoning_content`` across tool
        calls. Other providers keep the previous ``ChatDeepSeek`` behaviour.
        """
        if self._closed:
            raise RuntimeError("ProviderRegistry is closed")
        if not self._settings.api_key:
            return None
        if self._chat_model is None:
            api_key: str = self._settings.api_key  # known truthy from guard above
            self._chat_model = self._build_chat_model(api_key)
            logger.debug(
                "ProviderRegistry: chat model created ({})",
                type(self._chat_model).__name__,
            )
        return self._chat_model

    def _build_chat_model(self, api_key: str) -> BaseChatModel:
        """Construct the configured chat model (ECNU adapter or DeepSeek)."""
        settings = self._settings
        base_url = (settings.base_url or "https://api.deepseek.com").rstrip("/")
        model = settings.model or "deepseek-chat"

        http_client = ProviderHTTPClient(settings, self.concurrency)
        if settings.is_ecnu:
            return build_ecnu_model(
                model=model,
                api_key=api_key,
                base_url=base_url,
                reasoning_effort=settings.reasoning_effort,
                thinking_enabled=settings.thinking_enabled,
                timeout_s=float(self.retry_policy.timeout_s),
                max_tokens=settings.max_output_tokens,
                http_async_client=http_client,
            )

        return ChatDeepSeek(
            model=model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            timeout=self.retry_policy.timeout_s,
            max_retries=0,  # registry's gateway handles retry at a higher level
            temperature=0.7,
            max_tokens=2048,
            http_async_client=http_client,
        )

    def get_attribution_model(self) -> BaseChatModel | None:
        """Return a chat model suited to structured attribution, or ``None``.

        Unlike :meth:`get_chat_model` this one is the *only* model the ECNU
        configuration accepts for attribution, because the campus gateway has
        no separate "reasoner" tier: thinking is a request flag, not a model.
        Returns ``None`` without a key so L2/L3 stay reachable.
        """
        if self._closed:
            raise RuntimeError("ProviderRegistry is closed")
        return self.get_chat_model()

    @property
    def thinking_effort(self) -> str | None:
        """The reasoning effort this registry sends, or None when disabled."""
        if not self._settings.thinking_enabled:
            return None
        return self._settings.reasoning_effort

    def describe(self) -> dict[str, object]:
        """Provider/thinking evidence for reports and diagnostics.

        Contains no secret material — only the provider kind, model id, and
        the effective thinking configuration.
        """
        settings = self._settings
        model = settings.model or "deepseek-chat"
        is_ecnu = settings.is_ecnu
        downgraded = False
        effective = settings.reasoning_effort
        if is_ecnu:
            from mindflow.infrastructure.llm.ecnu import resolve_effort

            effective, downgraded = resolve_effort(model, settings.reasoning_effort)
        return {
            "provider": "ecnu" if is_ecnu else "generic",
            "model": model,
            "base_url_host": _host_of(settings.base_url),
            "thinking_enabled": settings.thinking_enabled,
            "reasoning_effort_requested": settings.reasoning_effort,
            "reasoning_effort_effective": effective,
            "reasoning_effort_downgraded": downgraded,
            "max_output_tokens": settings.max_output_tokens,
            "concurrency_limit": self.concurrency.limit,
        }

    def get_structured_attribution(self) -> DeepSeekClient | None:
        """Return the ``DeepSeekClient`` for typed ``LLMAttributionResult`` calls.

        Returns ``None`` when no API key was configured. The caller
        (LLMService) uses this for L1 of the degradation chain, falling
        through to Ollama (L2) and RuleEngine (L3).

        The returned client is typed — it returns ``LLMAttributionResult``
        instances, never raw strings or dicts.
        """
        if self._closed:
            raise RuntimeError("ProviderRegistry is closed")
        return self._deepseek_client

    def get_gateway(self) -> LangChainGateway:
        """Return the shared ``LangChainGateway`` for panel orchestrator.

        Always returns a gateway instance (key-less construction is
        allowed). If no API key is configured, ``GatewayNotConfiguredError``
        raises at call time, so the panel degradation chain stays reachable.
        """
        if self._closed:
            raise RuntimeError("ProviderRegistry is closed")
        return self._gateway

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Close all owned HTTP client pools exactly once.

        Idempotent — subsequent calls are no-ops.

        Closes (in order):
          1. DeepSeekClient's httpx pool (structured attribution)
          2. LangChainGateway's ChatDeepSeek pools (panel + chat gateway)
          3. Standalone ChatDeepSeek pool (agent model)
        """
        if self._closed:
            return
        self._closed = True

        # 1. DeepSeekClient (httpx.AsyncClient pool)
        if self._deepseek_client is not None:
            with contextlib.suppress(Exception):
                await self._deepseek_client.close()
            self._deepseek_client = None
            logger.debug("ProviderRegistry: DeepSeekClient pool closed")

        # 2. LangChainGateway (ChatDeepSeek root_async_client pools)
        with contextlib.suppress(Exception):
            await self._gateway.close()
        logger.debug("ProviderRegistry: LangChainGateway pools closed")

        # 3. Standalone ChatDeepSeek agent model
        if self._chat_model is not None:
            async_client = getattr(self._chat_model, "root_async_client", None)
            if async_client is not None and hasattr(async_client, "close"):
                with contextlib.suppress(Exception):
                    await async_client.close()
            sync_client = getattr(self._chat_model, "root_client", None)
            if sync_client is not None and hasattr(sync_client, "close"):
                with contextlib.suppress(Exception):
                    sync_client.close()
            self._chat_model = None
            logger.debug("ProviderRegistry: agent ChatDeepSeek pool closed")

        logger.info("ProviderRegistry shutdown complete")
