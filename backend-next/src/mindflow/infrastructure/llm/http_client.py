"""HTTP-boundary policy shared by SDK models and raw completion consumers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate
from mindflow.infrastructure.llm.ecnu import ecnu_request_fields


class _GatedStream(httpx.AsyncByteStream):
    """Keep a streaming generation's slot until the response is closed."""

    def __init__(self, stream: httpx.AsyncByteStream, gate: LLMConcurrencyGate) -> None:
        self._stream = stream
        self._gate = gate
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._stream.aclose()
        finally:
            await self._gate.__aexit__()


class ProviderHTTPClient(httpx.AsyncClient):
    """Acquire once per HTTP attempt, never around a workflow or retry loop.

    Raw ``post`` consumers also inherit ECNU's output and timeout budget.
    SDK clients construct their own payloads, but use the same ``send`` gate.
    """

    def __init__(
        self, settings: LLMSettings, gate: LLMConcurrencyGate, **kwargs: Any,
    ) -> None:
        self._settings = settings
        self._gate = gate
        super().__init__(**kwargs)

    async def post(self, url: httpx.URL | str, **kwargs: Any) -> httpx.Response:
        settings = self._settings
        payload = kwargs.get("json")
        if settings.is_ecnu and isinstance(payload, dict):
            payload = dict(payload)
            payload.pop("max_tokens", None)
            payload.pop("reasoning_effort", None)
            payload.update(ecnu_request_fields(
                model=str(payload.get("model") or settings.model or "ecnu-max"),
                thinking_enabled=settings.thinking_enabled,
                reasoning_effort=settings.reasoning_effort,
                max_tokens=settings.max_output_tokens,
            ))
            if settings.thinking_enabled:
                payload.pop("temperature", None)
            kwargs["json"] = payload
            kwargs["timeout"] = settings.timeout_s
        return await super().post(url, **kwargs)

    async def send(
        self, request: httpx.Request, *, stream: bool = False, **kwargs: Any,
    ) -> httpx.Response:
        await self._gate.__aenter__()
        handed_off = False
        try:
            response = await super().send(request, stream=stream, **kwargs)
            if stream and not response.is_closed:
                assert isinstance(response.stream, httpx.AsyncByteStream)
                response.stream = _GatedStream(response.stream, self._gate)
                handed_off = True
            return response
        finally:
            if not handed_off:
                await self._gate.__aexit__()
