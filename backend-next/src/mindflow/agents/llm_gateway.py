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
import inspect
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from mindflow.agents.policies import CompletionPolicy
from mindflow.config import L1Target, LLMSettings, get_settings
from mindflow.errors import LLMAPIError, LLMNotConfiguredError
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.deepseek import build_deepseek_model
from mindflow.infrastructure.llm.ecnu import build_ecnu_model, looks_like_ecnu
from mindflow.infrastructure.llm.http_client import (
    HTTPAttemptMetrics,
    ProviderHTTPClient,
    collect_http_attempts,
)
from mindflow.infrastructure.llm.safety import safe_error_metadata
from mindflow.services.llm_observability import (
    LLMObservabilityAggregator,
    LLMRequestRecord,
    current_llm_labels,
    default_aggregator,
    summarise_attempts,
)

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
        policy: CompletionPolicy | None = None,
    ) -> str:
        """Send a completion request and return the response content.

        Args:
            system: System prompt defining the expert's persona and output contract.
            user: User message containing the evidence data and context.
            model: Model tier — "chat" (deepseek-chat) or "reasoner" (deepseek-reasoner).
            policy: Optional role-level :class:`CompletionPolicy`. It is applied as
                per-request overrides on the shared model instance (no new model
                and no new HTTP client per role); ``None`` preserves the
                provider-configured defaults exactly.

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

#: Tier labels accepted by ``complete(model=...)``. Both request the same
#: provider model (DeepSeek Flash in production); the tier only selects the
#: output constraint. ``"chat"`` = structured JSON output mode, ``"reasoner"`` =
#: prose (the caller parses it).
_STRUCTURED_TIER: str = "chat"
_PROSE_TIER: str = "reasoner"


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


#: Exception type names for which an identical retry is futile. Truncation is
#: included: the answer hit the token cap, so re-sending the same cap truncates
#: again (observed against real DeepSeek on 2026-09-25 and fixed by raising the
#: thinking-mode floor in ``agents/policies.py``).
_NON_RETRIABLE_EXCEPTIONS: frozenset[str] = frozenset({
    "LengthFinishReasonError",
    "ContentFilterFinishReasonError",
})


def _is_non_retriable(exc: Exception | None) -> bool:
    """Return True when retrying *exc* would burn budget without any chance.

    Auth failures (401/403), parameter/validation errors and **length
    truncation** are permanent for a given request. Provider-side 429/5xx and
    network errors are retriable and stay with the caller's retry budget.
    """
    if exc is None:
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _NON_RETRIABLE_STATUSES:
        return True
    if type(exc).__name__ in _NON_RETRIABLE_EXCEPTIONS:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _NON_RETRIABLE_MARKERS)


def gateway_accepts_policy(gateway: object) -> bool:
    """Probe whether ``gateway.complete`` accepts a ``policy`` argument.

    Gateways written before ``CompletionPolicy`` existed — and the test doubles
    used across the suite — keep the three-argument signature. Callers (the
    panel helpers) must keep working with them, so the capability is probed
    instead of assumed: a gateway that cannot take a policy simply gets none.
    """
    complete = getattr(gateway, "complete", None)
    if complete is None or not callable(complete):
        return False
    try:
        parameters = inspect.signature(complete).parameters
    except (TypeError, ValueError):  # builtins / exotic callables
        return False
    if "policy" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


# ── Token accounting ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _TokenUsage:
    """Provider-reported token counts (all zero when the provider omits them)."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


def _token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _extract_usage(message: object) -> _TokenUsage:
    """Read token usage off a LangChain message, tolerating provider shapes.

    ``usage_metadata`` is the LangChain v1 field; older/proxied providers only
    populate ``response_metadata["token_usage"]``. Reasoning tokens arrive under
    ``output_token_details`` (OpenAI-compatible) or ``completion_tokens_details``.
    """
    usage: object = getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        metadata = getattr(message, "response_metadata", None)
        usage = None
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
    if not isinstance(usage, dict):
        return _TokenUsage()

    input_tokens = _token_count(usage.get("input_tokens")) or _token_count(
        usage.get("prompt_tokens")
    )
    output_tokens = _token_count(usage.get("output_tokens")) or _token_count(
        usage.get("completion_tokens")
    )

    reasoning_tokens = 0
    details = usage.get("output_token_details")
    if isinstance(details, dict):
        reasoning_tokens = _token_count(details.get("reasoning"))
    if not reasoning_tokens:
        reasoning_tokens = _token_count(usage.get("reasoning_tokens"))
    if not reasoning_tokens:
        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning_tokens = _token_count(completion_details.get("reasoning_tokens"))
    return _TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


class LangChainGateway:
    """Async LLM gateway wrapping LangChain's ``ChatDeepSeek``.

    Unlike ``DeepSeekClient`` (which binds to ``LLMAttributionResult``), this
    gateway returns raw response text. The caller (orchestrator) handles parsing.

    Args:
        api_key: DeepSeek API key. If None, reads from ``Settings``.
        base_url: API base URL. If None, reads from ``Settings``.
        observability: Aggregator for LLM request metadata. Defaults to the
            process-wide aggregator; inject one to isolate a test or harness.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: int | None = None,
        max_retries: int | None = None,
        llm_settings: LLMSettings | None = None,
        concurrency: LLMConcurrencyGate | None = None,
        observability: LLMObservabilityAggregator | None = None,
        target: L1Target | None = None,
    ) -> None:
        # ``llm_settings`` is the injected LLMSettings (from ProviderRegistry).
        # Falling back to the cached application settings keeps the standalone
        # constructor working for eval scripts and tests.
        settings = llm_settings if llm_settings is not None else get_settings().llm
        explicit_base_url = base_url is not None
        # ``target`` is the caller's *resolved* L1 endpoint (plan item 2). The
        # production assembly passes the DeepSeek pin so a legacy ECNU URL/model
        # still sitting in the environment cannot select the campus protocol;
        # direct constructions without a target keep the historical inference.
        if target is not None:
            api_key = target.api_key if api_key is None else api_key
            base_url = target.base_url if base_url is None else base_url
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
        self._model_id = target.model if target is not None else (
            settings.model or settings.deepseek_model
        )
        self._target = target
        # An explicitly supplied base_url decides the provider on its own: the
        # caller is targeting that endpoint regardless of what the ambient
        # settings say. Checking the model name too would let a stale
        # MINDFLOW_LLM__MODEL=ecnu-max make a test/diagnostic gateway pointed at
        # some other host speak the ECNU protocol.
        if target is not None:
            # The caller resolved the endpoint for us: trust it verbatim.
            self._is_ecnu = target.is_ecnu
        elif settings.provider.strip().lower() == "generic":
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

        # Provider label used in observability records and to pick the
        # request-body spelling for role policies ("ecnu" | "deepseek").
        self._provider_label = "ecnu" if self._is_ecnu else "deepseek"
        self._observability = (
            observability if observability is not None else default_aggregator()
        )

    def _get_model(self, model_id: str) -> BaseChatModel:
        """Return a cached chat model for the requested *tier*.

        ``model_id`` is now a **tier label**, not a provider model id: the
        production pin sends one model (``deepseek-flash``) for every call.
        ``"chat"`` is the structured tier (JSON output mode, used by the panel
        experts and the critic); ``"reasoner"`` is the prose tier (no JSON
        constraint, used where the orchestrator parses free text itself).

        On the campus gateway thinking is a request flag rather than a model, so
        both tiers resolve to one :class:`ECNUChatModel` per JSON mode. The
        DeepSeek endpoint behaves the same way — the tier differs only in
        whether ``response_format`` is requested.
        """
        wants_json = model_id == _STRUCTURED_TIER
        provider_model = self._model_id

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
            if wants_json:
                if self._chat_model is None:
                    self._chat_model = self._build_ecnu_model(json_mode=True)
                return self._chat_model
            if self._reasoner_model is None:
                self._reasoner_model = self._build_ecnu_model(json_mode=False)
            return self._reasoner_model

        # DeepSeek direct: one model id for both tiers; JSON mode distinguishes
        # the structured tier. ``max_tokens`` is the documented output cap
        # (``max_completion_tokens`` is not a DeepSeek parameter).
        if wants_json:
            if self._chat_model is None:
                self._chat_model = self._build_deepseek_model(
                    provider_model, json_mode=True,
                )
            return self._chat_model
        if self._reasoner_model is None:
            self._reasoner_model = self._build_deepseek_model(
                provider_model, json_mode=False,
            )
        return self._reasoner_model

    def _deepseek_max_tokens(self) -> int:
        """Output cap for a DeepSeek request: the resolved target, else settings."""
        if self._target is not None:
            return self._target.max_output_tokens
        return self._llm_settings.max_output_tokens

    def _build_deepseek_model(self, provider_model: str, *, json_mode: bool) -> BaseChatModel:
        """Build one DeepSeek chat model, optionally in JSON-output mode.

        Uses :class:`DeepSeekThinkingModel` rather than plain ``ChatDeepSeek``:
        DeepSeek's thinking mode requires the previous turns'
        ``reasoning_content`` to be echoed back on any request carrying tools
        (otherwise HTTP 400), which LangChain's converter drops.
        """
        return build_deepseek_model(
            model=provider_model,
            api_key=self._api_key,
            base_url=self._base_url,
            timeout_s=float(self._timeout_s),
            max_retries=0,
            max_tokens=self._deepseek_max_tokens(),
            temperature=_LLM_TEMPERATURE,
            http_async_client=ProviderHTTPClient(
                self._llm_settings, self._concurrency,
            ),
            model_kwargs=(
                {"response_format": {"type": "json_object"}} if json_mode else None
            ),
        )

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
        policy: CompletionPolicy | None = None,
    ) -> str:
        """Send a completion request and return the response content as raw text.

        Raises GatewayNotConfiguredError at call time if no key was supplied
        (deferred from __init__ — E2E finding: the app must assemble services
        without a key so degradation paths stay reachable).

        Args:
            system: System prompt.
            user: User message.
            model: Tier label — "chat" (structured: JSON output mode) or
                "reasoner" (prose: no JSON constraint). Both tiers request the
                same provider model (the resolved L1 model, DeepSeek Flash in
                production); the tier only decides the output constraint.
            policy: Role-level completion policy, applied as *per-request*
                overrides on the cached model instance. ``None`` sends nothing
                extra, preserving the provider-configured defaults.

        Returns:
            Raw response content string.

        Raises:
            GatewayNotConfiguredError: If not configured.
            GatewayAPIError: After exhausting retries.
        """
        tier = _STRUCTURED_TIER if model == "chat" else _PROSE_TIER
        if not self._api_key:
            raise GatewayNotConfiguredError(
                "LLM API key is not configured — set DEEPSEEK_API_KEY "
                "(or MINDFLOW_LLM__DEEPSEEK_API_KEY) to enable L1"
            )

        chat = self._get_model(tier)
        provider_model = self._model_id
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        # Per-request policy overrides ride on the call kwargs, which
        # langchain-openai merges into the request payload. The model instance
        # (and its HTTP client) is reused for every role.
        overrides: dict[str, Any] = (
            policy.request_overrides(provider=self._provider_label)
            if policy is not None else {}
        )
        started_at = time.perf_counter()

        # Retry only transient failures. A 401/403 or a bad-parameter 4xx cannot
        # succeed on a second identical attempt and would just burn budget.
        non_retriable = _is_non_retriable(exc=None)
        last_attempts: list[HTTPAttemptMetrics] = []
        retries_used = 0
        failure_reason = "exhausted"

        for attempt in range(self._max_retries + 1):
            attempt_metrics: list[HTTPAttemptMetrics] = []
            retries_used = attempt
            try:
                with collect_http_attempts() as attempt_metrics:
                    result = await chat.ainvoke(messages, **overrides)
                logger.debug(
                    "LangChain gateway complete tier={} model={} latency_ms={:.1f}",
                    tier, provider_model, (time.perf_counter() - started_at) * 1000.0,
                )
            except Exception as exc:
                last_attempts = attempt_metrics
                non_retriable = _is_non_retriable(exc)
                logger.warning(
                    "LangChain gateway error (attempt {}, retriable={}): {}",
                    attempt + 1, not non_retriable, safe_error_metadata(exc),
                )
                if non_retriable:
                    failure_reason = "non_retriable_error"
                    break
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                    continue
                break

            raw_content = result.content
            content: str = raw_content if isinstance(raw_content, str) else ""
            if not content:
                last_attempts = attempt_metrics
                logger.warning("LangChain gateway returned empty content")
                failure_reason = "empty_content"
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue

            self._record_request(
                model_id=provider_model,
                policy=policy,
                attempts=attempt_metrics,
                retry_count=attempt,
                total_latency_ms=(time.perf_counter() - started_at) * 1000.0,
                ok=True,
                usage=_extract_usage(result),
            )
            return content

        # All retries exhausted (or a non-retriable error stopped the loop)
        self._record_request(
            model_id=provider_model,
            policy=policy,
            attempts=last_attempts,
            retry_count=retries_used,
            total_latency_ms=(time.perf_counter() - started_at) * 1000.0,
            ok=False,
            fallback_reason=failure_reason,
        )
        raise GatewayAPIError(
            f"LangChain gateway failed after {retries_used + 1} attempt(s)"
            + (" (non-retriable error)" if non_retriable else "")
        ) from None

    # ── Observability (phase 2.5) ─────────────────────────────────────────

    def _record_request(
        self,
        *,
        model_id: str,
        policy: CompletionPolicy | None,
        attempts: Sequence[HTTPAttemptMetrics],
        retry_count: int,
        total_latency_ms: float,
        ok: bool,
        fallback_reason: str = "",
        usage: _TokenUsage | None = None,
    ) -> None:
        """Emit one aggregated metadata record for a logical LLM request.

        Aggregated only: latency, token counts, status class, retry count and
        sanitized categories. Prompts, completions, headers and keys never reach
        this path — :class:`LLMRequestRecord` has no field that could hold them.
        """
        graph, node, role = current_llm_labels()
        if policy is not None:
            role = role or policy.role
            node = node or policy.node
        queue_ms, http_ms, status, error_category = summarise_attempts(attempts)
        token_usage = usage if usage is not None else _TokenUsage()
        self._observability.record(LLMRequestRecord(
            graph=graph,
            node=node,
            role=role,
            provider=self._provider_label,
            model=model_id,
            queue_latency_ms=queue_ms,
            http_latency_ms=http_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            reasoning_tokens=token_usage.reasoning_tokens,
            retry_count=retry_count,
            http_status_class=status,
            error_category=error_category,
            fallback_reason=fallback_reason,
            ok=ok,
        ))

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
