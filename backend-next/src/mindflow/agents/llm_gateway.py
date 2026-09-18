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

import asyncio
import random
import time
from typing import Literal, Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from loguru import logger
from pydantic import SecretStr

from mindflow.config import LLMSettings, get_settings
from mindflow.errors import LLMAPIError, LLMNotConfiguredError
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.ecnu import build_ecnu_model, looks_like_ecnu
from mindflow.infrastructure.llm.http_client import ProviderHTTPClient
from mindflow.infrastructure.llm.safety import safe_error_metadata

# ── Custom exceptions ──────────────────────────────────────────────────────────
#
# Historically the gateway defined its own ``Gateway*`` errors that were
# near-identical to the legacy ``DeepSeekClient``'s ``LLM*`` errors. They are
# now reconciled: the gateway names are thin subclasses of the canonical LLM
# exceptions (mindflow.errors). This keeps ``except GatewayAPIError`` and
# ``except LLMAPIError`` both working, and both catch the same failures.


class GatewayNotConfiguredError(LLMNotConfiguredError):
    """Raised when the gateway is called without an API key being configured."""


class GatewayAPIError(LLMAPIError):
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
_BACKOFF_CAP_S: float = 60.0
_LLM_TEMPERATURE: float = 0.2


def _compute_backoff(attempt: int) -> float:
    """Compute exponential backoff with jitter for a retry attempt.

    Formula: ``min(2 ** attempt + random.uniform(0, 1), cap)``.

    Args:
        attempt: Zero-based retry attempt number (0 = first retry).

    Returns:
        Delay in seconds (always ≥ 0).
    """
    jitter: float = random.uniform(0.0, 1.0)
    delay: float = float(2**attempt) + jitter
    capped: float = min(delay, _BACKOFF_CAP_S)
    return capped


#: HTTP statuses that a retry cannot fix: the request itself is rejected.
_NON_RETRIABLE_STATUSES: frozenset[int] = frozenset(
    {400, 401, 403, 404, 405, 415, 422}
)

# Substrings that indicate the request shape is wrong (so a retry is futile).
_NON_RETRIABLE_MARKERS: tuple[str, ...] = (
    "api key",
    "authentication",
    "unauthorized",
    "permission",
    "invalid_request_error",
    "invalid request",
    "does not support",
    "unsupported",
    "bad request",
    "403",
    "401",
)


def _is_non_retriable(exc: Exception | None) -> bool:
    """Return True when retrying *exc* would burn budget without any chance.

    Auth failures (401/403) and parameter/validation errors are permanent for
    a given request. Provider-side 429/5xx and network errors are retriable and
    stay with the caller's retry budget.
    """
    if exc is None:
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _NON_RETRIABLE_STATUSES:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _NON_RETRIABLE_MARKERS)


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
        timeout_s: int | None = None,
        max_retries: int | None = None,
        llm_settings: LLMSettings | None = None,
        concurrency: LLMConcurrencyGate | None = None,
    ) -> None:
        # ``llm_settings`` is the injected LLMSettings (from ProviderRegistry).
        # Falling back to the cached application settings keeps the standalone
        # constructor working for eval scripts and tests.
        settings = llm_settings if llm_settings is not None else get_settings().llm
        explicit_base_url = base_url is not None
        if api_key is None:
            api_key = settings.api_key
        if base_url is None:
            base_url = settings.base_url
        self._timeout_s = timeout_s if timeout_s is not None else settings.timeout_s
        self._max_retries = max_retries if max_retries is not None else settings.max_retries
        self._llm_settings = settings
        self._concurrency = (
            concurrency if concurrency is not None
            else LLMConcurrencyGate(settings.max_concurrent_requests)
        )

        # Key-less construction is allowed (E2E finding): the app must be able
        # to assemble PanelService/ChatService without a configured key so the
        # degradation chain (panel->single_expert->rule_engine, chat->safe reply)
        # stays reachable. The raise happens at call time in complete().
        self._api_key = api_key or ""
        self._base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self._model_id = settings.model or "deepseek-chat"
        # An explicitly supplied base_url decides the provider on its own: the
        # caller is targeting that endpoint regardless of what the ambient
        # settings say. Checking the model name too would let a stale
        # MINDFLOW_LLM__MODEL=ecnu-max make a test/diagnostic gateway pointed at
        # some other host speak the ECNU protocol.
        if settings.provider.strip().lower() == "generic":
            self._is_ecnu = False
        elif llm_settings is not None and settings.provider.strip().lower() == "ecnu":
            self._is_ecnu = settings.is_ecnu
        elif explicit_base_url:
            self._is_ecnu = looks_like_ecnu(self._base_url, None)
        else:
            self._is_ecnu = settings.is_ecnu or looks_like_ecnu(
                self._base_url, self._model_id
            )

        # Lazy-initialised chat model instances (one per model tier).
        self._chat_model: BaseChatModel | None = None
        self._reasoner_model: BaseChatModel | None = None

    def _get_model(self, model_id: str) -> BaseChatModel:
        """Return a cached chat model for *model_id*.

        On the campus gateway there is no separate reasoner model — thinking is
        a request flag — so both tiers resolve to one :class:`ECNUChatModel`.
        Elsewhere the previous two-tier behaviour is preserved: the ``chat``
        tier sends ``response_format: json_object``; the ``reasoner`` tier
        does not, because ``deepseek-reasoner`` rejects that parameter.
        """
        if self._is_ecnu:
            # The campus gateway serves ONE model; thinking is a request flag,
            # not a model choice. The two tiers therefore differ only in
            # whether JSON mode is requested, and each needs its own cached
            # instance — sharing one would let whichever tier was called first
            # decide the response_format for every later call.
            #
            # JSON mode is safe with thinking enabled (probed 2026-09-19:
            # HTTP 200, valid JSON). The "reasoner" tier omits it so a
            # reasoning-heavy generation can return prose; the orchestrator
            # parses that path itself.
            wants_json = model_id == "deepseek-chat"
            if wants_json:
                if self._chat_model is None:
                    self._chat_model = self._build_ecnu_model(json_mode=True)
                return self._chat_model
            if self._reasoner_model is None:
                self._reasoner_model = self._build_ecnu_model(json_mode=False)
            return self._reasoner_model

        # Tier routing for the DeepSeek-compatible endpoint.
        if model_id == "deepseek-chat":
            if self._chat_model is None:
                self._chat_model = ChatDeepSeek(
                    model=model_id,
                    api_key=SecretStr(self._api_key) if self._api_key else None,
                    base_url=self._base_url,
                    timeout=self._timeout_s,
                    max_retries=0,
                    model_kwargs={"response_format": {"type": "json_object"}},
                    temperature=_LLM_TEMPERATURE,
                    http_async_client=ProviderHTTPClient(
                        self._llm_settings, self._concurrency,
                    ),
                )
            return self._chat_model

        # model_id == "deepseek-reasoner" (no response_format)
        if self._reasoner_model is None:
            self._reasoner_model = ChatDeepSeek(
                model=model_id,
                api_key=SecretStr(self._api_key) if self._api_key else None,
                base_url=self._base_url,
                timeout=self._timeout_s,
                max_retries=0,
                temperature=_LLM_TEMPERATURE,
                http_async_client=ProviderHTTPClient(
                    self._llm_settings, self._concurrency,
                ),
            )
        return self._reasoner_model

    def _build_ecnu_model(self, *, json_mode: bool) -> BaseChatModel:
        """Build one ECNU model instance, optionally in JSON-output mode."""
        return build_ecnu_model(
            model=self._model_id,
            api_key=self._api_key,
            base_url=self._base_url,
            reasoning_effort=self._llm_settings.reasoning_effort,
            thinking_enabled=self._llm_settings.thinking_enabled,
            timeout_s=float(self._timeout_s),
            max_tokens=self._llm_settings.max_output_tokens,
            http_async_client=ProviderHTTPClient(
                self._llm_settings, self._concurrency,
            ),
            model_kwargs=(
                {"response_format": {"type": "json_object"}} if json_mode else None
            ),
        )

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
                "LLM API key is not configured — set MINDFLOW_LLM__API_KEY "
                "or add llm.api_key to the .env file"
            )

        chat = self._get_model(model_id)
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        started_at = time.perf_counter()

        # Retry only transient failures. A 401/403 or a bad-parameter 4xx cannot
        # succeed on a second identical attempt and would just burn budget.
        non_retriable = _is_non_retriable(exc=None)

        for attempt in range(self._max_retries + 1):
            try:
                result = await chat.ainvoke(messages)
                logger.debug(
                    "LangChain gateway complete model={} latency_ms={:.1f}",
                    model_id, (time.perf_counter() - started_at) * 1000.0,
                )
            except Exception as exc:
                non_retriable = _is_non_retriable(exc)
                logger.warning(
                    "LangChain gateway error (attempt {}, retriable={}): {}",
                    attempt + 1, not non_retriable, safe_error_metadata(exc),
                )
                if non_retriable:
                    break
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue

            raw_content = result.content
            content: str = raw_content if isinstance(raw_content, str) else ""
            if not content:
                logger.warning("LangChain gateway returned empty content")
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue

            return content

        # All retries exhausted (or a non-retriable error stopped the loop)
        raise GatewayAPIError(
            f"LangChain gateway failed after {attempt + 1} attempt(s)"
            + (" (non-retriable error)" if non_retriable else "")
        ) from None

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
