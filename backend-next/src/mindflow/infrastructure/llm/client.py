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
import random

import httpx
from loguru import logger

from mindflow.config import LLMSettings

# Canonical LLM exceptions now live in ``mindflow.errors`` (unified taxonomy);
# re-exported here (``... as ...`` marks the intentional re-export for mypy)
# so ``from mindflow.infrastructure.llm.client import LLMAPIError`` keeps working.
from mindflow.errors import LLMAPIError as LLMAPIError
from mindflow.errors import LLMNotConfiguredError as LLMNotConfiguredError
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

_BACKOFF_CAP_S: float = 60.0


def _compute_backoff(attempt: int, retry_after: int | None = None) -> float:
    """Compute the backoff delay for a retry attempt.

    If *retry_after* is a positive integer, use it (capped at ``_BACKOFF_CAP_S``).
    Otherwise, use exponential backoff with jitter:
    ``min(2 ** attempt + random.uniform(0, 1), cap)``.

    Args:
        attempt: Zero-based retry attempt number (0 = first retry).
        retry_after: Optional ``Retry-After`` header value in seconds (integer only).

    Returns:
        Delay in seconds (always ≥ 0).
    """
    if retry_after is not None and retry_after > 0:
        capped_ra: float = min(float(retry_after), _BACKOFF_CAP_S)
        return capped_ra

    jitter: float = random.uniform(0.0, 1.0)
    delay: float = float(2**attempt) + jitter
    capped_exp: float = min(delay, _BACKOFF_CAP_S)
    return capped_exp


def _parse_retry_after(response: httpx.Response) -> int | None:
    """Parse the ``Retry-After`` header as an integer number of seconds.

    Returns ``None`` when the header is absent, non-integer, or unparseable.
    """
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return int(header)
    except (ValueError, TypeError):
        return None


class DeepSeekClient:
    """Async HTTP client for DeepSeek Chat API (OpenAI-compatible).

    Args:
        settings: LLM configuration (api_key, base_url, model).
            If ``settings.api_key`` is None, raises ``LLMNotConfiguredError``.
    """

    def __init__(self, settings: LLMSettings) -> None:
        if not settings.api_key:
            raise LLMNotConfiguredError(
                "DeepSeek API key is not configured — set MINDFLOW_LLM__API_KEY "
                "or add llm.api_key to the .env file"
            )

        self._base_url = (settings.base_url or "https://api.deepseek.com").rstrip("/")
        self._model = settings.model or "deepseek-chat"
        self._timeout_s: int = settings.timeout_s
        self._max_retries: int = settings.max_retries
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout_s),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
        )

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
        payload = {
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
                logger.warning("DeepSeek API HTTP error (attempt {}): {}", attempt + 1, exc)
                last_exc = exc
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
                msg = f"DeepSeek API error {response.status_code}: {response.text[:200]}"
                logger.error(msg)
                raise LLMAPIError(msg)

            # Parse response
            try:
                body = response.json()
            except json.JSONDecodeError as exc:
                logger.warning("DeepSeek returned non-JSON response: {}", exc)
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue

            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                logger.warning("DeepSeek returned empty content")
                last_exc = LLMAPIError("Empty content in response")
                if attempt < self._max_retries:
                    await asyncio.sleep(_compute_backoff(attempt))
                continue

            # Parse and validate via Pydantic strict mode
            try:
                return LLMAttributionResult.model_validate_json(content)
            except Exception as exc:
                logger.warning("DeepSeek response validation failed: {}", exc)
                last_exc = exc
                # Don't retry validation failures — the model's output won't
                # change on a retry with the same input.
                raise  # noqa: TRY201 — intentional re-raise to route to L2/L3

        # All retries exhausted
        raise LLMAPIError(
            f"DeepSeek API failed after {self._max_retries + 1} attempts"
        ) from last_exc

    async def close(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.aclose()
