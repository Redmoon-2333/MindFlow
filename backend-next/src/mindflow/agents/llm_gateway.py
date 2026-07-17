"""LLM gateway protocol and implementation for the expert panel.

Provides a generic gateway that is **not** bound to any specific output schema
(unlike the existing ``DeepSeekClient`` which hard-codes ``LLMAttributionResult``).
The panel uses this gateway to call arbitrary experts with arbitrary system prompts.

Design:
  - ``PanelLLMGateway`` is a typing.Protocol — the orchestrator depends on the
    interface, not the implementation, making it trivially testable with mocks.
  - ``DeepSeekGateway`` reuses ``Settings.llm`` configuration (api_key, base_url)
    via the same pattern as ``DeepSeekClient``, but does NOT share the same
    connection pool (separate httpx.AsyncClient for clarity).
  - ``response_format: {"type": "json_object"}`` is used for ``chat`` model calls;
    ``reasoner`` model calls omit this (deepseek-reasoner does not support it).

Raises:
    httpx.TimeoutException: After 30s with no response (1 retry on network errors).
    ``GatewayAPIError``: For non-retriable API errors (4xx, auth failure, …).
"""

from __future__ import annotations

import json
from typing import Literal, Protocol, runtime_checkable

import httpx
from loguru import logger

from mindflow.config import get_settings

# ── Custom exceptions ──────────────────────────────────────────────────────────


class GatewayNotConfiguredError(RuntimeError):
    """Raised when the gateway is constructed without an API key."""


class GatewayAPIError(RuntimeError):
    """Raised when the upstream API returns a non-retriable error."""


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
            httpx.TimeoutException: Request timed out.
            GatewayAPIError: Non-retriable API error.
        """
        ...

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        ...


# ── DeepSeek implementation ────────────────────────────────────────────────────

_DEFAULT_TIMEOUT_S: int = 30
_MAX_RETRIES: int = 1


class DeepSeekGateway:
    """Async HTTP client for DeepSeek Chat API (OpenAI-compatible).

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

        if not api_key:
            raise GatewayNotConfiguredError(
                "DeepSeek API key is not configured — set MINDFLOW_LLM__API_KEY "
                "or add llm.api_key to the .env file"
            )

        self._base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT_S),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def complete(
        self,
        system: str,
        user: str,
        model: Literal["chat", "reasoner"] = "chat",
    ) -> str:
        """Send a completion request and return the response content as raw text.

        Args:
            system: System prompt.
            user: User message.
            model: "chat" → deepseek-chat (with json_object mode),
                   "reasoner" → deepseek-reasoner (no json_object mode).

        Returns:
            Raw response content string.

        Raises:
            GatewayNotConfiguredError: If not configured.
            httpx.TimeoutException: Request timed out.
            GatewayAPIError: Non-retriable API error.
        """
        model_id = "deepseek-chat" if model == "chat" else "deepseek-reasoner"

        payload: dict[str, object] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        # deepseek-reasoner does not support response_format
        if model == "chat":
            payload["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                )
            except httpx.TimeoutException:
                logger.warning("DeepSeek gateway timeout (attempt {})", attempt + 1)
                last_exc = httpx.TimeoutException("DeepSeek gateway timed out after 30s")
                continue
            except httpx.HTTPError as exc:
                logger.warning("DeepSeek gateway HTTP error (attempt {}): {}", attempt + 1, exc)
                last_exc = exc
                continue

            if response.status_code == 429:
                logger.warning("DeepSeek gateway rate limited (attempt {})", attempt + 1)
                last_exc = GatewayAPIError(f"Rate limited: {response.status_code}")
                continue

            if response.status_code >= 500:
                logger.warning(
                    "DeepSeek gateway server error {} (attempt {})",
                    response.status_code,
                    attempt + 1,
                )
                last_exc = GatewayAPIError(f"Server error: {response.status_code}")
                continue

            if response.status_code != 200:
                msg = f"DeepSeek gateway error {response.status_code}: {response.text[:200]}"
                logger.error(msg)
                raise GatewayAPIError(msg)

            # Parse response body
            try:
                body = response.json()
            except json.JSONDecodeError as exc:
                logger.warning("DeepSeek gateway returned non-JSON response: {}", exc)
                last_exc = exc
                continue

            content: str = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                logger.warning("DeepSeek gateway returned empty content")
                last_exc = GatewayAPIError("Empty content in response")
                continue

            return content

        # All retries exhausted
        raise GatewayAPIError(
            f"DeepSeek gateway failed after {_MAX_RETRIES + 1} attempts"
        ) from last_exc

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.aclose()
