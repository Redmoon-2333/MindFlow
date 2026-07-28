"""P0-3: Exponential retry backoff with jitter — focused acceptance tests.

Covers DeepSeekClient (HTTP mock transport) and LangChainGateway
(ChatDeepSeek patching) backoff behaviour:

  - Integer Retry-After preferred, capped at 60s
  - Malformed/absent Retry-After → fallback exponential + jitter
  - Exponential: ``min(2^attempt + random.uniform(0,1), 60)``
  - No sleep after final exhausted attempt (attempt < max_retries guard)
  - Validation errors and non-retryable 4xx: fast-fail, no delay
  - Jitter bounds: delay ∈ [2^n, 2^n+1] when uniform pinned

All randomness is patched deterministically; no flaky statistical assertions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import ValidationError

from mindflow.agents.llm_gateway import GatewayAPIError, LangChainGateway
from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.client import DeepSeekClient, LLMAPIError

# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixtures and helpers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _no_ssl_cert_file_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)


def _make_settings(api_key: str = "test-key") -> LLMSettings:
    return LLMSettings(api_key=api_key, base_url="https://test.api.example.com", model="test-model")


def _mock_openai_response(content: str, model: str = "test-model") -> httpx.Response:
    body = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    return httpx.Response(200, json=body)


_VALID_LLM_RESPONSE = json.dumps(
    {
        "procrastination_types": ["impulsivity"],
        "type_confidence": {"impulsivity": 0.82},
        "cognitive_distortions": ["all-or-nothing thinking"],
        "cbt_technique": "stimulus_control",
        "response_text": (
            "\u4f60\u4eca\u5929\u7684\u4e13\u6ce8\u6a21\u5f0f"
            "\u53cd\u6620\u4e86\u51b2\u52a8\u5206\u5fc3\u7684\u503e\u5411\u3002"
        ),
        "next_action": "\u8bbe\u7f6e\u4e00\u4e2a\u756a\u8304\u949f",
    },
    ensure_ascii=False,
)


async def _make_mock_client(handler) -> DeepSeekClient:
    settings = _make_settings()
    client = DeepSeekClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=client._base_url,
        timeout=httpx.Timeout(30),
        headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
    )
    return client


def _make_aimessage(content: str = "{}") -> AIMessage:
    return AIMessage(content=content)


async def _run_client_with_patched_sleep(handler, monkeypatch) -> list[float]:
    """Run DeepSeekClient.analyze() capturing every asyncio.sleep delay.

    random.uniform is pinned to 0.0.  Returns the list of delay arguments.
    """
    sleep_calls: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleep_calls.append(d)

    client = await _make_mock_client(handler)
    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)

    with contextlib.suppress(LLMAPIError, ValidationError):
        await client.analyze('{"test": "data"}')
    await client.close()
    return sleep_calls


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeekClient backoff tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestClientRetryBackoff:
    """Client-level exponential backoff with jitter and Retry-After parsing."""

    # ── 429 with integer Retry-After ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_429_with_retry_after_uses_header_value(self, monkeypatch) -> None:
        """429 with ``Retry-After: 5`` → sleep ~5s before retry."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "5"})

        delays = await _run_client_with_patched_sleep(_handler, monkeypatch)
        assert len(delays) >= 1, f"Expected at least one sleep call, got {delays}"
        assert 4.5 <= delays[0] <= 5.5, f"Expected delay ~5s, got {delays[0]}"

    @pytest.mark.asyncio
    async def test_429_with_large_retry_after_is_capped_at_60(self, monkeypatch) -> None:
        """``Retry-After: 999`` → capped at 60s."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "999"})

        delays = await _run_client_with_patched_sleep(_handler, monkeypatch)
        assert len(delays) >= 1
        assert delays[0] <= 60.0, f"Expected capped ≤60s, got {delays[0]}"

    @pytest.mark.asyncio
    async def test_429_with_malformed_retry_after_falls_back(self, monkeypatch) -> None:
        """Malformed ``Retry-After: \"soon\"`` → fallback exponential + jitter."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "soon"})

        delays = await _run_client_with_patched_sleep(_handler, monkeypatch)
        assert len(delays) >= 1
        # Fallback: attempt=0 → 2^0 + 0.0 = 1.0
        assert 0.9 <= delays[0] <= 1.1, f"Expected fallback ~1.0s, got {delays[0]}"

    # ── 5xx / timeout / HTTP error → exponential + jitter ───────────────

    @pytest.mark.asyncio
    async def test_5xx_without_retry_after_uses_exponential_jitter(self, monkeypatch) -> None:
        """500 without ``Retry-After`` → exponential+jitter delay."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        delays = await _run_client_with_patched_sleep(_handler, monkeypatch)
        assert len(delays) >= 1
        assert 0.9 <= delays[0] <= 1.1, f"Expected ~1.0s, got {delays[0]}"

    @pytest.mark.asyncio
    async def test_timeout_uses_exponential_jitter(self, monkeypatch) -> None:
        """Timeout → exponential+jitter delay before retry."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out")

        delays = await _run_client_with_patched_sleep(_handler, monkeypatch)
        assert len(delays) >= 1
        assert 0.9 <= delays[0] <= 1.1, f"Expected ~1.0s, got {delays[0]}"

    # ── No delay on final attempt ───────────────────────────────────────

    @pytest.mark.asyncio
    async def test_no_delay_after_final_attempt(self, monkeypatch) -> None:
        """After last retry exhausted, no additional sleep fires."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        delays = await _run_client_with_patched_sleep(_handler, monkeypatch)
        # _MAX_RETRIES=1 → 2 attempts (0,1), 1 delay before attempt 1
        assert len(delays) == 1, (
            f"Expected exactly 1 sleep (before the retry), got {len(delays)}"
        )

    # ── Fast-fail: non-retryable 4xx ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_4xx_non_retryable_no_delay(self, monkeypatch) -> None:
        """400/401/403 raise immediately — no sleep, no retry."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad request"})

        sleep_calls: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        client = await _make_mock_client(_handler)
        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        with pytest.raises(LLMAPIError, match="400"):
            await client.analyze('{"test": "data"}')
        await client.close()

        assert sleep_calls == [], f"Expected no sleep for 4xx, got {sleep_calls}"

    # ── Fast-fail: validation error ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_validation_error_no_delay(self, monkeypatch) -> None:
        """Validation errors raise immediately — no sleep, no retry."""
        invalid_content = json.dumps(
            {
                "procrastination_types": ["invalid_type"],
                "type_confidence": {"invalid_type": 0.5},
                "cbt_technique": "stimulus_control",
                "response_text": "test",
                "next_action": "test",
            }
        )

        async def _handler(_: httpx.Request) -> httpx.Response:
            return _mock_openai_response(invalid_content)

        sleep_calls: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        client = await _make_mock_client(_handler)
        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

        with pytest.raises(ValidationError):
            await client.analyze('{"test": "data"}')
        await client.close()

        assert sleep_calls == [], f"Expected no sleep for validation error, got {sleep_calls}"

    # ── Deterministic jitter bounds ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_jitter_stays_within_bounds(self, monkeypatch) -> None:
        """uniform pinned to 0.0 → delay = 2^0 + 0.0 = 1.0."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        monkeypatch.setattr("random.uniform", lambda a, b: 0.0)
        delays = await _run_client_with_patched_sleep(_handler, monkeypatch)
        assert len(delays) == 1
        assert 0.9 <= delays[0] <= 1.1, f"Expected ~1.0 (jitter=0), got {delays[0]}"

    @pytest.mark.asyncio
    async def test_jitter_upper_bound_is_2n_plus_1(self, monkeypatch) -> None:
        """uniform pinned to 1.0 → delay = 2^0 + 1.0 = 2.0."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        sleep_calls: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        client = await _make_mock_client(_handler)
        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
        monkeypatch.setattr("random.uniform", lambda a, b: 1.0)

        with contextlib.suppress(LLMAPIError):
            await client.analyze('{"test": "data"}')
        await client.close()

        assert len(sleep_calls) == 1
        assert 1.9 <= sleep_calls[0] <= 2.1, f"Expected ~2.0 (jitter=1), got {sleep_calls[0]}"


# ═══════════════════════════════════════════════════════════════════════════════
# LangChainGateway backoff tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGatewayRetryBackoff:
    """Gateway exponential backoff + jitter on retryable errors."""

    @pytest.mark.asyncio
    async def test_retryable_error_sleeps_before_retry(self, monkeypatch) -> None:
        """When ainvoke raises, sleep with exponential+jitter before retry."""
        sleep_calls: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        gateway = LangChainGateway(
            api_key="test-key", base_url="https://test.api.example.com", max_retries=1
        )

        with patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke:
            mock_ainvoke.side_effect = RuntimeError("simulated failure")
            monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
            monkeypatch.setattr("random.uniform", lambda a, b: 0.0)

            with pytest.raises(GatewayAPIError, match="failed after"):
                await gateway.complete("system", "user")

        assert len(sleep_calls) >= 1, f"Expected at least one sleep, got {sleep_calls}"
        assert 0.9 <= sleep_calls[0] <= 1.1, f"Expected ~1.0s, got {sleep_calls[0]}"

    @pytest.mark.asyncio
    async def test_no_delay_after_final_attempt(self, monkeypatch) -> None:
        """After max_retries exhausted, no extra sleep fires."""
        sleep_calls: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        gateway = LangChainGateway(
            api_key="test-key", base_url="https://test.api.example.com", max_retries=1
        )

        with patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke:
            mock_ainvoke.side_effect = RuntimeError("persistent failure")
            monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
            monkeypatch.setattr("random.uniform", lambda a, b: 0.0)

            with pytest.raises(GatewayAPIError):
                await gateway.complete("system", "user")

        assert len(sleep_calls) == 1, (
            f"Expected exactly 1 sleep (before the retry), got {len(sleep_calls)}"
        )

    @pytest.mark.asyncio
    async def test_no_delay_on_success(self, monkeypatch) -> None:
        """Successful response → no sleep at all."""
        sleep_calls: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        gateway = LangChainGateway(
            api_key="test-key", base_url="https://test.api.example.com", max_retries=1
        )

        with patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke:
            mock_ainvoke.return_value = _make_aimessage('{"ok": true}')
            monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
            monkeypatch.setattr("random.uniform", lambda a, b: 0.0)

            result = await gateway.complete("system", "user")

        assert result == '{"ok": true}'
        assert sleep_calls == [], f"Expected no sleep on success, got {sleep_calls}"

    @pytest.mark.asyncio
    async def test_delay_grows_exponentially_with_retries(self, monkeypatch) -> None:
        """max_retries=2 → delays: ~1.0s, ~2.0s."""
        sleep_calls: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        gateway = LangChainGateway(
            api_key="test-key", base_url="https://test.api.example.com", max_retries=2
        )

        with patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke:
            mock_ainvoke.side_effect = RuntimeError("persistent failure")
            monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
            monkeypatch.setattr("random.uniform", lambda a, b: 0.0)

            with pytest.raises(GatewayAPIError):
                await gateway.complete("system", "user")

        assert len(sleep_calls) == 2, f"Expected 2 sleeps, got {len(sleep_calls)}"
        assert 0.9 <= sleep_calls[0] <= 1.1, f"First delay should be ~1.0s, got {sleep_calls[0]}"
        assert 1.9 <= sleep_calls[1] <= 2.1, f"Second delay should be ~2.0s, got {sleep_calls[1]}"
        assert sleep_calls[1] > sleep_calls[0], "Delays should increase exponentially"

    @pytest.mark.asyncio
    async def test_delay_capped_at_60(self, monkeypatch) -> None:
        """Exponential with max jitter capped at 60s."""
        sleep_calls: list[float] = []

        async def _fake_sleep(d: float) -> None:
            sleep_calls.append(d)

        gateway = LangChainGateway(
            api_key="test-key", base_url="https://test.api.example.com", max_retries=7
        )

        with patch.object(ChatDeepSeek, "ainvoke", new=AsyncMock()) as mock_ainvoke:
            mock_ainvoke.side_effect = RuntimeError("persistent failure")
            monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
            monkeypatch.setattr("random.uniform", lambda a, b: 1.0)

            with pytest.raises(GatewayAPIError):
                await gateway.complete("system", "user")

        for i, d in enumerate(sleep_calls):
            assert d <= 60.0, f"Delay[{i}]={d} exceeded cap of 60s"
