"""DeepSeek API client via httpx AsyncClient (OpenAI-compatible).

L1 of the three-tier degradation chain (Architecture §3.3, ADR-003).

Design decisions:
  - httpx.AsyncClient with connection pooling and a configurable timeout
    (``LLMSettings.timeout_s``, default 30 seconds).
  - Retry budget from ``LLMSettings.max_retries`` (default 1) on
    network-level errors (connect, timeout, 5xx) —
    validation errors and 4xx are NOT retried (they won't succeed).
  - Exponential backoff with jitter before retries (P0-3):
    delay = min(2^attempt + random.uniform(0, 1), 60).
    ``Retry-After`` header (integer only) preferred when available.
    No delay before the final attempt.
  - ``response_format: {"type": "json_object"}`` instructs the API to
    return valid JSON — the caller still validates via Pydantic.
  - System prompt encodes the CBT-coach persona, safety boundaries,
    and JSON schema constraints per llm-cbt.md §2.

Raises:
    LLMNotConfiguredError: If no api_key is available at construction time.
    httpx.TimeoutException: After ``timeout_s`` with no response.
    LLMAttributionResult.ValidationError: If the response JSON is structurally
        valid but semantically invalid (forbidden words, type mismatches, …).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from loguru import logger
from pydantic import ValidationError

from mindflow.agents.policies import ATTRIBUTION_POLICY, THINKING_TOKEN_FLOOR
from mindflow.config import LLMSettings

# Canonical LLM exceptions now live in ``mindflow.errors`` (unified taxonomy);
# re-exported here (``... as ...`` marks the intentional re-export for mypy)
# so ``from mindflow.infrastructure.llm.client import LLMAPIError`` keeps working.
from mindflow.errors import LLMAPIError as LLMAPIError
from mindflow.errors import LLMNotConfiguredError as LLMNotConfiguredError
from mindflow.infrastructure.llm.concurrency import LLMConcurrencyGate

# Retry/backoff arithmetic has one implementation (http_client) shared with the
# fallback tier; the private aliases below keep this module's historical names.
from mindflow.infrastructure.llm.http_client import ProviderHTTPClient
from mindflow.infrastructure.llm.http_client import compute_backoff as _compute_backoff
from mindflow.infrastructure.llm.http_client import parse_retry_after as _parse_retry_after
from mindflow.infrastructure.llm.safety import safe_error_metadata
from mindflow.infrastructure.llm.schemas import LLMAttributionResult

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = (
    "你是一个基于认知行为疗法(CBT)的拖延干预教练。"
    "你的角色是分析用户的行为数据并提供温和、鼓励但不纵容的反馈。\n\n"
    "## 交互协议\n"
    "1. 镜像确认：基于行为数据描述观察到的情况\n"
    "2. 归因探索：识别拖延类型和认知扭曲\n"
    "3. 行动约定：提供最小下一步建议\n\n"
    "## 安全边界\n"
    "- 你不冒充心理治疗师或医生\n"
    "- 检测到严重拖延模式时，回应以温和建议\n"
    '- 永远不要使用"诊断"、"治疗"、"患者"、"处方"等医疗用语\n\n'
    "## 输出要求\n"
    "你必须以 JSON 对象格式输出，包含以下字段：\n"
    "  procrastination_types: 检测到的拖延类型数组(1-3个)，可选值："
    '"task_aversion","impulsivity","decisional","perfectionism","emotional_regulation"\n'
    "  type_confidence: 对象，key 为拖延类型，value 为置信度(0-1)\n"
    "  cognitive_distortions: 认知扭曲列表\n"
    "  cbt_technique: 推荐的CBT技术，可选值："
    '"behavioral_experiment","cognitive_restructuring","stimulus_control","goal_setting","graded_exposure","mindfulness"\n'
    "  response_text: 对用户的回应文本(中文，不超过500字)\n"
    "  next_action: 下一个最小可执行建议\n\n"
    "请确保输出是合法的 JSON 对象，不包含 markdown 代码块标记。"
)

# ``_compute_backoff`` / ``_parse_retry_after`` are imported above from
# ``http_client`` (the single implementation shared with the fallback tier),
# keeping this module's historical private names working for existing callers.


class DeepSeekClient:
    """Async HTTP client for DeepSeek Chat API (OpenAI-compatible).

    Args:
        settings: LLM configuration (api_key, base_url, model).
            If ``settings.api_key`` is None, raises ``LLMNotConfiguredError``.
    """

    def __init__(
        self, settings: LLMSettings, concurrency: LLMConcurrencyGate | None = None,
    ) -> None:
        if not settings.api_key:
            raise LLMNotConfiguredError(
                "DeepSeek API key is not configured — set DEEPSEEK_API_KEY "
                "(or MINDFLOW_LLM__DEEPSEEK_API_KEY) to enable L1"
            )

        self._base_url = (settings.base_url or "https://api.deepseek.com").rstrip("/")
        self._model = settings.model or "deepseek-chat"
        self._timeout_s: int = settings.timeout_s
        self._max_retries: int = settings.max_retries
        #: Provider-reported token usage of the last successful call:
        #: ``(input_tokens, output_tokens, reasoning_tokens)``, or ``None`` when
        #: the provider omitted it or the last call failed. Observability reads
        #: this after :meth:`analyze` — never estimated from text length.
        self.last_usage: tuple[int, int, int] | None = None
        self._client = ProviderHTTPClient(
            settings,
            concurrency if concurrency is not None
            else LLMConcurrencyGate(settings.max_concurrent_requests),
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout_s),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def client(self) -> httpx.AsyncClient:
        """The underlying shared HTTP client (owned by ProviderRegistry).

        Exposed so consumers like InterventionService can reuse the same
        connection pool instead of creating a second one (audit report —
        second httpx pool outside ProviderRegistry).
        """
        return self._client

    @property
    def model(self) -> str:
        """Model id this client sends (used for observability records)."""
        return self._model

    # ── Public API ────────────────────────────────────────────────────

    async def analyze(
        self,
        summary_json: str,
    ) -> LLMAttributionResult:
        """Send a behavior summary to DeepSeek and return a parsed result.

        Args:
            summary_json: JSON-serialized behavior summary
                (from :func:`build_behavior_summary`).

        Returns:
            A validated ``LLMAttributionResult``.

        Raises:
            LLMNotConfiguredError: If the client was not configured with
                an API key.
            httpx.TimeoutException: Request timed out.
            LLMAPIError: Non-retriable API error (4xx).
            ValidationError: Response JSON failed semantic validation.
        """
        self.last_usage = None
        # Every L1 entry point carries the same reasoning contract as the
        # gateway (plan item 2): JSON constraint for this structured path, the
        # attribution policy's effort and a cap that leaves room for the
        # thinking trace (thinking tokens are billed against ``max_tokens``).
        policy = ATTRIBUTION_POLICY
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"请分析以下行为数据，输出结构化归因结果：\n\n{summary_json}",
                },
            ],
            "response_format": {"type": "json_object"},
        }
        if policy.reasoning_effort is not None:
            payload["reasoning_effort"] = policy.reasoning_effort
            payload["thinking"] = {"type": "enabled"}
        if policy.max_output_tokens is not None:
            payload["max_tokens"] = (
                max(policy.max_output_tokens, THINKING_TOKEN_FLOOR)
                if policy.reasoning_effort is not None
                else policy.max_output_tokens
            )

        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                )
            except httpx.TimeoutException:
                logger.warning("DeepSeek API timeout (attempt {})", attempt + 1)
                last_exc = httpx.TimeoutException(
                    f"DeepSeek API timed out after {self._timeout_s}s"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue
            except httpx.HTTPError as exc:
                metadata = safe_error_metadata(exc)
                logger.warning("DeepSeek API HTTP error (attempt {}): {}", attempt + 1, metadata)
                last_exc = LLMAPIError(metadata)
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue

            if response.status_code == 429:
                logger.warning("DeepSeek rate limited (attempt {})", attempt + 1)
                last_exc = LLMAPIError(f"Rate limited: {response.status_code}")
                if attempt < self._max_retries:
                    retry_after = _parse_retry_after(response)
                    await asyncio.sleep(_compute_backoff(attempt, retry_after))
                continue

            if response.status_code >= 500:
                logger.warning(
                    "DeepSeek server error {} (attempt {})", response.status_code, attempt + 1
                )
                last_exc = LLMAPIError(f"Server error: {response.status_code}")
                if attempt < self._max_retries:
                    retry_after = _parse_retry_after(response)
                    await asyncio.sleep(_compute_backoff(attempt, retry_after))
                continue

            if response.status_code != 200:
                msg = f"DeepSeek API error status={response.status_code}"
                logger.error(msg)
                raise LLMAPIError(msg)

            # Parse response
            try:
                body = response.json()
            except json.JSONDecodeError as exc:
                metadata = safe_error_metadata(exc)
                logger.warning("DeepSeek returned non-JSON response: {}", metadata)
                last_exc = LLMAPIError(metadata)
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue

            self.last_usage = _extract_usage(body)

            try:
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            except (AttributeError, IndexError, TypeError) as exc:
                raise LLMAPIError(
                    f"DeepSeek malformed response ({safe_error_metadata(exc)})"
                ) from None
            if not content:
                logger.warning("DeepSeek returned empty content")
                last_exc = LLMAPIError("Empty content in response")
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue

            # Parse and validate via Pydantic strict mode
            try:
                return LLMAttributionResult.model_validate_json(content)
            except ValidationError as exc:
                logger.warning(
                    "DeepSeek response validation failed: {}", safe_error_metadata(exc)
                )
                # Preserve the degradation contract without retaining provider inputs.
                validation_error = ValidationError.from_exception_data(
                    "LLMAttributionResult",
                    [{
                        "type": "value_error",
                        "loc": (),
                        "input": None,
                        "ctx": {"error": ValueError("Invalid attribution response")},
                    }],
                    hide_input=True,
                )
            raise validation_error

        # All retries exhausted
        raise LLMAPIError(
            f"DeepSeek API failed after {self._max_retries + 1} attempts"
        ) from last_exc

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.aclose()


def _extract_usage(body: Any) -> tuple[int, int, int] | None:
    """Pull ``(input, output, reasoning)`` from an OpenAI-compatible usage block.

    Returns ``None`` when the provider omitted usage entirely — the observability
    layer then reports "not reported" instead of estimating from text length.
    """
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else 0
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    if not (input_tokens or output_tokens):
        return None
    return (input_tokens, output_tokens, int(reasoning or 0))
