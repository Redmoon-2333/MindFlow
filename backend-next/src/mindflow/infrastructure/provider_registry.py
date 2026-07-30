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

# ── Shared policy ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetryPolicy:
    """Shared timeout/retry/backoff configuration for all LLM consumers.

    Attributes:
        timeout_s: Request timeout in seconds (1-300).
        max_retries: Maximum retry attempts (0-10).
        backoff_cap_s: Upper bound for exponential backoff delay.
    """

    timeout_s: int = 30
    max_retries: int = 1
    backoff_cap_s: float = 60.0


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

        # ── DeepSeekClient for structured attribution (L1 of degradation chain) ─
        self._deepseek_client: DeepSeekClient | None = None
        if settings.api_key:
            try:
                self._deepseek_client = DeepSeekClient(settings)
                logger.debug("ProviderRegistry: DeepSeekClient created")
            except Exception as exc:
                logger.warning("ProviderRegistry: failed to create DeepSeekClient: {}", exc)

        # ── LangChainGateway for panel orchestrator and chat ────────────────────
        # Key-less construction is allowed: the gateway defers the key check to
        # call time so degradation paths stay reachable.
        self._gateway = LangChainGateway(
            api_key=settings.api_key or "",
            base_url=settings.base_url,
            timeout_s=settings.timeout_s,
            max_retries=settings.max_retries,
        )
        logger.debug("ProviderRegistry: LangChainGateway created")

        # ── Standalone ChatDeepSeek for ChatService LangChain agent ─────────────
        # Built separately from the gateway's own models so the agent can be
        # configured with temperature/max_tokens without affecting the gateway.
        self._chat_model: ChatDeepSeek | None = None

        self._closed = False

    # ── Typed access interfaces ──────────────────────────────────────────────

    def get_chat_model(self) -> BaseChatModel | None:
        """Return a cached ``ChatDeepSeek`` for LangChain agent usage.

        Returns ``None`` when no API key is configured so the degradation
        path (LLM-down safe reply) stays reachable.
        The ``chat`` tier uses ``response_format: json_object``.
        """
        if self._closed:
            raise RuntimeError("ProviderRegistry is closed")
        if not self._settings.api_key:
            return None
        if self._chat_model is None:
            api_key: str = self._settings.api_key  # known truthy from guard above
            base_url: str = (
                self._settings.base_url or "https://api.deepseek.com"
            ).rstrip("/")
            self._chat_model = ChatDeepSeek(
                model="deepseek-chat",
                api_key=SecretStr(api_key),
                base_url=base_url,
                timeout=self.retry_policy.timeout_s,
                max_retries=0,  # registry's gateway handles retry at a higher level
                temperature=0.7,
                max_tokens=2048,
            )
            logger.debug("ProviderRegistry: ChatDeepSeek agent model created")
        return self._chat_model

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
