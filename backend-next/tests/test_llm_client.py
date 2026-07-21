"""Tests for DeepSeekClient — HTTP client for OpenAI-compatible API.

All tests use ``httpx.MockTransport`` to avoid real network calls.

Coverage:
  - Successful response returns parsed LLMAttributionResult
  - Timeout raises TimeoutException → caught by degradation chain
  - Non-JSON response raises appropriate error
  - 429 rate limit retries once
  - 500 server error retries once
  - Missing API key raises LLMNotConfiguredError
  - Validation error in response raises through
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from mindflow.config import LLMSettings
from mindflow.infrastructure.llm.client import DeepSeekClient, LLMAPIError, LLMNotConfiguredError


@pytest.fixture(autouse=True)
def _no_ssl_cert_file_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset ``SSL_CERT_FILE`` so real ``httpx.AsyncClient`` construction

    (which eagerly builds an SSL context via ``trust_env``) never fails on a
    machine where the variable points to a path invalid for this interpreter.
    """
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)


def _make_settings(api_key: str = "test-key") -> LLMSettings:
    return LLMSettings(api_key=api_key, base_url="https://test.api.example.com", model="test-model")


def _mock_openai_response(content: str, model: str = "test-model") -> httpx.Response:
    """Build a mock OpenAI-compatible chat completion response."""
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
        "response_text": "你今天的专注模式反映了冲动分心的倾向。",
        "next_action": "设置一个番茄钟",
    },
    ensure_ascii=False,
)


async def _make_mock_client(handler) -> DeepSeekClient:
    """Create a DeepSeekClient with a MockTransport handler.

    Constructs the real client (which validates the API key and sets up
    headers), then swaps in a mock transport for the test.
    """
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


class TestDeepSeekClient:
    """DeepSeekClient HTTP tests."""

    def test_no_key_raises_at_construction(self) -> None:
        """Constructing without an API key should raise."""
        settings = LLMSettings(api_key=None)
        with pytest.raises(LLMNotConfiguredError, match="API key"):
            DeepSeekClient(settings)

    @pytest.mark.asyncio
    async def test_successful_response(self) -> None:
        """A 200 with valid JSON should return a parsed result."""

        async def _handler(_: httpx.Request) -> httpx.Response:
            return _mock_openai_response(_VALID_LLM_RESPONSE)

        client = await _make_mock_client(_handler)

        result = await client.analyze('{"test": "data"}')
        await client.close()

        assert result.cbt_technique == "stimulus_control"
        assert result.procrastination_types == ["impulsivity"]
        assert len(result.response_text) <= 500

    @pytest.mark.asyncio
    async def test_timeout_raises(self) -> None:
        """A timeout should retry then raise LLMAPIError for the degradation chain."""

        async def _handler(_: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Request timed out")

        client = await _make_mock_client(_handler)

        with pytest.raises(LLMAPIError, match="failed after"):
            await client.analyze('{"test": "data"}')
        await client.close()

    @pytest.mark.asyncio
    async def test_non_json_response(self) -> None:
        """Non-JSON response body should raise after retries."""

        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json at all")

        client = await _make_mock_client(_handler)

        with pytest.raises(LLMAPIError):
            await client.analyze('{"test": "data"}')
        await client.close()

    @pytest.mark.asyncio
    async def test_rate_limit_retry(self) -> None:
        """429 should retry once, then raise."""

        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(429, json={"error": "rate_limit"})

        client = await _make_mock_client(_handler)

        with pytest.raises(LLMAPIError, match="failed after"):
            await client.analyze('{"test": "data"}')
        await client.close()

    @pytest.mark.asyncio
    async def test_server_error_retry(self) -> None:
        """500 should retry once, then raise."""
        async def _handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "internal"})

        client = await _make_mock_client(_handler)

        with pytest.raises(LLMAPIError, match="failed after"):
            await client.analyze('{"test": "data"}')
        await client.close()

    @pytest.mark.asyncio
    async def test_validation_error_raises_immediately(self) -> None:
        """Validation errors should not be retried — raise immediately."""

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

        client = await _make_mock_client(_handler)

        with pytest.raises(ValidationError):
            await client.analyze('{"test": "data"}')
        await client.close()

    @pytest.mark.asyncio
    async def test_empty_content_in_response(self) -> None:
        """Empty content from the API should raise after retries."""

        async def _handler(_: httpx.Request) -> httpx.Response:
            body = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
            }
            return httpx.Response(200, json=body)

        client = await _make_mock_client(_handler)

        with pytest.raises(LLMAPIError):
            await client.analyze('{"test": "data"}')
        await client.close()

    @pytest.mark.asyncio
    async def test_auth_header_present(self) -> None:
        """Verify the Authorization header is set as Bearer token."""

        async def _handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["Authorization"] == "Bearer test-key"
            return _mock_openai_response(_VALID_LLM_RESPONSE)

        client = await _make_mock_client(_handler)

        result = await client.analyze('{"test": "data"}')
        await client.close()

        assert result.cbt_technique == "stimulus_control"
