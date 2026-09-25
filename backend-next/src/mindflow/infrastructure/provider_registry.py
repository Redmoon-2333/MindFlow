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
  - ``get_fallback_client()`` → shared ``httpx.AsyncClient`` for the L2
    Ollama fallback (one construction, one shutdown, pooled connections)

Design constraints:
  - DeepSeek reasoner models never receive ``response_format: json_object``
    (preserved via LangChainGateway's own model tier routing).
  - Ollama/RuleEngine fallback *routing* is owned by the graphs; the HTTP
    client those graphs use is owned here (optimisation plan 4.3), so the
    fallback tier shares one pool, one timeout, one concurrency gate and one
    shutdown with every other consumer.
  - ``LLMAttributionResult`` remains typed throughout; no raw strings.
  - Source outcome labels (panel, single_expert, ollama, rule_engine)
    are not altered by the registry.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from mindflow.agents.llm_gateway import LangChainGateway
from mindflow.agents.policies import CHAT_POLICY
from mindflow.config import L1Target, LLMSettings
from mindflow.infrastructure.llm.client import DeepSeekClient
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.deepseek import build_deepseek_model
from mindflow.infrastructure.llm.ecnu import build_ecnu_model
from mindflow.infrastructure.llm.http_client import (
    ProviderHTTPClient,
    SharedHTTPClientPool,
    clear_shared_fallback_pool,
    install_shared_fallback_pool,
)
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
        #: Resolved L1 endpoint (plan item 2): production pins DeepSeek direct,
        #: overriding any legacy ECNU URL/model/key left in the environment.
        self._target = settings.l1_target()
        self.retry_policy = RetryPolicy(
            timeout_s=settings.timeout_s,
            max_retries=settings.max_retries,
        )
        # Every consumer shares one gate so the provider's per-user concurrency
        # cap is respected no matter which entry point is calling.
        self.concurrency = LLMConcurrencyGate(settings.max_concurrent_requests)

        # ── Structured attribution client (L1 of degradation chain) ──────────
        self._deepseek_client: DeepSeekClient | None = None
        if self._target.api_key:
            try:
                self._deepseek_client = DeepSeekClient(
                    self._pinned_settings(), self.concurrency,
                )
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
            api_key=self._target.api_key or "",
            base_url=self._target.base_url,
            timeout_s=settings.timeout_s,
            max_retries=settings.max_retries,
            llm_settings=settings,
            concurrency=self.concurrency,
            target=self._target,
        )
        # ── Panel fan-out gateway (targeted serialization fix) ───────────────
        # The panel's parallel attribution/rebuttal batches are the only
        # genuinely concurrent workload; through the shared gate (limit 1) each
        # expert waits for the previous one, so three ~4s calls cost ~19s of
        # queueing. When ``panel_fanout_concurrency`` > 1 the fan-out runs on a
        # dedicated gateway with its own gate, while every other L1 path keeps
        # the global cap. Built lazily-used: None means "reuse the main
        # gateway" so the process still owns exactly one pool by default.
        fanout_limit = max(1, min(settings.panel_fanout_concurrency, 8))
        self._fanout_gateway: LangChainGateway | None = None
        self._fanout_concurrency = fanout_limit
        if fanout_limit > 1:
            self._fanout_gateway = LangChainGateway(
                api_key=self._target.api_key or "",
                base_url=self._target.base_url,
                timeout_s=settings.timeout_s,
                max_retries=settings.max_retries,
                llm_settings=settings,
                concurrency=LLMConcurrencyGate(fanout_limit),
                target=self._target,
            )
            logger.info(
                "ProviderRegistry: panel fan-out gateway created (burst={}, "
                "global={})",
                fanout_limit,
                self.concurrency.limit,
            )
        logger.info(
            "ProviderRegistry: L1 resolved (provenance={}, provider={}, model={}, "
            "credential={}, thinking={}, effort={})",
            self._target.provenance,
            self._target.provider,
            self._target.model,
            "present" if self._target.api_key else "absent",
            self._target.thinking_enabled,
            self._target.reasoning_effort,
        )
        if not self._target.api_key:
            # Make an *unintended* capability drop loud. The production pin
            # overrides a legacy ECNU URL/model/key by design, so a machine that
            # still carries the old ECNU `.env` (and no DeepSeek credential)
            # silently loses L1: say so, with the exact remedy, instead of only
            # logging "credential=absent" at INFO level.
            legacy_present = bool(settings.api_key) or bool(settings.base_url)
            logger.warning(
                "ProviderRegistry: L1 (DeepSeek direct) is DISABLED — no DeepSeek "
                "credential configured{}. The degradation chain will use L2 "
                "(Ollama, if enabled) or L3 (RuleEngine). Set DEEPSEEK_API_KEY "
                "(or MINDFLOW_LLM__DEEPSEEK_API_KEY) to enable L1; a legacy ECNU "
                "key is never reused, by design.",
                " (a legacy ECNU credential/endpoint is present but not reused)"
                if legacy_present else "",
            )

        # ── Standalone chat model for ChatService's LangGraph agent ─────────────
        # Built separately from the gateway's own models so the agent can be
        # configured with its own output cap without affecting the gateway.
        self._chat_model: BaseChatModel | None = None

        # ── Shared fallback (L2 Ollama) HTTP client ────────────────────────────
        # The fallback path used to build — and close — a fresh
        # ``httpx.AsyncClient`` per call, i.e. one connection pool per request.
        # One pool is now created lazily, reused by every fallback call, and
        # closed exactly once in ``shutdown()``. It is also published as the
        # process default because AnalysisGraph builds its FallbackRunContext
        # from settings and never sees the registry.
        self._fallback_http = SharedHTTPClientPool(
            gate=self.concurrency,
            timeout_s=float(self.retry_policy.timeout_s),
            max_retries=self.retry_policy.max_retries,
            backoff_cap_s=self.retry_policy.backoff_cap_s,
            name="ollama-fallback",
        )
        install_shared_fallback_pool(self._fallback_http)

        self._closed = False

    # ── Typed access interfaces ──────────────────────────────────────────────

    @property
    def l1_target(self) -> L1Target:
        """The resolved L1 endpoint (secret-free diagnostics in ``describe()``)."""
        return self._target

    def _pinned_settings(self) -> LLMSettings:
        """A settings view whose credential/endpoint match the resolved target.

        ``DeepSeekClient`` and the standalone chat model read their values from
        an ``LLMSettings`` object, so the pin is applied once here instead of
        every consumer re-deriving it. The pinned copy carries the DeepSeek
        endpoint, so its own ``is_ecnu``/``l1_target`` resolve identically.
        """
        return self._settings.model_copy(update={
            "api_key": self._target.api_key,
            "base_url": self._target.base_url,
            "model": self._target.model,
            "provider": self._target.provider,
            "deepseek_api_key": self._target.api_key,
            "deepseek_model": self._target.model,
            "deepseek_base_url": self._target.base_url,
        })

    def get_chat_model(self) -> BaseChatModel | None:
        """Return a cached chat model for LangGraph/Tool-calling usage.

        Returns ``None`` when no API key is configured so the degradation
        path (LLM-down safe reply) stays reachable.

        For the campus gateway (compat mode only) this is an
        :class:`ECNUChatModel`, which adds the thinking fields and preserves
        ``reasoning_content`` across tool calls. Production uses the pinned
        DeepSeek endpoint, where thinking is requested per call through the
        chat policy (see ``agents/policies.py``).
        """
        if self._closed:
            raise RuntimeError("ProviderRegistry is closed")
        if not self._target.api_key:
            return None
        if self._chat_model is None:
            self._chat_model = self._build_chat_model(self._target.api_key)
            logger.debug(
                "ProviderRegistry: chat model created ({})",
                type(self._chat_model).__name__,
            )
        return self._chat_model

    def _build_chat_model(self, api_key: str) -> BaseChatModel:
        """Construct the configured chat model (pinned DeepSeek, or ECNU compat)."""
        target = self._target
        http_client = ProviderHTTPClient(self._settings, self.concurrency)
        if target.is_ecnu:
            return build_ecnu_model(
                model=target.model,
                api_key=api_key,
                base_url=target.base_url,
                reasoning_effort=target.reasoning_effort,
                thinking_enabled=target.thinking_enabled,
                timeout_s=float(self.retry_policy.timeout_s),
                max_tokens=target.max_output_tokens,
                http_async_client=http_client,
            )

        return build_deepseek_model(
            model=target.model,
            api_key=api_key,
            base_url=target.base_url,
            timeout_s=float(self.retry_policy.timeout_s),
            max_retries=0,  # registry's gateway handles retry at a higher level
            temperature=0.7,
            # The standalone chat model keeps the plan's chat output cap; the
            # panel path applies its own per-role caps per request.
            max_tokens=CHAT_POLICY.max_output_tokens or 2048,
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
        if not self._target.thinking_enabled:
            return None
        return self._target.reasoning_effort

    def describe(self) -> dict[str, object]:
        """Provider/thinking evidence for reports and diagnostics.

        Contains no secret material — only the resolved provider kind, model id,
        provenance and the effective thinking configuration.
        """
        target = self._target
        downgraded = False
        effective = target.reasoning_effort
        if target.is_ecnu:
            from mindflow.infrastructure.llm.ecnu import resolve_effort

            effective, downgraded = resolve_effort(target.model, target.reasoning_effort)
        return {
            "provider": target.provider,
            "model": target.model,
            "base_url_host": _host_of(target.base_url),
            "thinking_enabled": target.thinking_enabled,
            "reasoning_effort_requested": target.reasoning_effort,
            "reasoning_effort_effective": effective,
            "reasoning_effort_downgraded": downgraded,
            "max_output_tokens": target.max_output_tokens,
            "provenance": target.provenance,
            "credential_present": bool(target.api_key),
            "concurrency_limit": self.concurrency.limit,
            "fallback_pool": self._fallback_http.snapshot(),
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

    def get_fanout_gateway(self) -> LangChainGateway:
        """The gateway the panel's parallel batches should use.

        Returns the dedicated fan-out gateway when ``panel_fanout_concurrency``
        > 1, otherwise the shared gateway — callers never need to branch.
        """
        if self._closed:
            raise RuntimeError("ProviderRegistry is closed")
        return self._fanout_gateway or self._gateway

    @property
    def fanout_concurrency(self) -> int:
        """The burst permit count for the panel's parallel batches."""
        return self._fanout_concurrency

    @property
    def fallback_pool(self) -> SharedHTTPClientPool:
        """The registry-owned pool behind the L2 Ollama fallback client."""
        return self._fallback_http

    def get_fallback_client(self) -> httpx.AsyncClient:
        """Return the shared L2 fallback HTTP client (constructed on first use).

        The client carries the registry's timeout, concurrency gate and
        connection-pool limits, so the fallback tier behaves like every other
        LLM consumer instead of opening its own pool per call.
        """
        if self._closed:
            raise RuntimeError("ProviderRegistry is closed")
        return self._fallback_http.client

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Close all owned HTTP client pools exactly once.

        Idempotent — subsequent calls are no-ops.

        Closes (in order):
          1. DeepSeekClient's httpx pool (structured attribution)
          2. LangChainGateway's ChatDeepSeek pools (panel + chat gateway)
          3. Standalone ChatDeepSeek pool (agent model)
          4. Shared fallback (Ollama) pool — also detached from the process
             default slot so no later call can use a closed client
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

        # 2b. Panel fan-out gateway (own pool, only built when burst > 1)
        if self._fanout_gateway is not None:
            with contextlib.suppress(Exception):
                await self._fanout_gateway.close()
            self._fanout_gateway = None
            logger.debug("ProviderRegistry: fan-out gateway pools closed")

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

        # 4. Shared fallback (Ollama) pool — clear the slot first so a
        #    concurrently starting call cannot grab the client mid-close.
        clear_shared_fallback_pool(self._fallback_http)
        with contextlib.suppress(Exception):
            await self._fallback_http.aclose()
        logger.debug(
            "ProviderRegistry: fallback pool closed (constructed={}, closed={})",
            self._fallback_http.constructions, self._fallback_http.closes,
        )

        logger.info("ProviderRegistry shutdown complete")
