"""DeepSeek-direct chat model: OpenAI-compatible plus thinking-mode protocol.

Production L1 is DeepSeek itself (``https://api.deepseek.com``,
``model=deepseek-flash``).  DeepSeek's thinking mode is a *request* flag rather
than a separate model, and it has the same round-trip requirement as the campus
gateway that preceded it:

* ``reasoning_content`` from previous assistant turns must be echoed whenever a
  request carries ``tools`` — otherwise the API answers 400;
* ``temperature`` has no effect in thinking mode, so it is dropped from a
  request that asks for thinking rather than sent and silently ignored.

Request-body spelling follows the official docs
(https://api-docs.deepseek.com/api/create-chat-completion): the output cap is
``max_tokens``, thinking is ``extra_body={"thinking": {"type": "enabled"}}`` and
the intensity is the top-level ``reasoning_effort`` (``low`` / ``high`` /
``max``, with ``none`` disabling thinking).  Role-level policies supply those
values per request, so one model instance serves every role.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from mindflow.infrastructure.llm.thinking import (
    restore_reasoning_content,
    thinking_requested,
)

#: DeepSeek's documented output-cap field.
MAX_TOKENS_FIELD = "max_tokens"


class DeepSeekThinkingModel(ChatDeepSeek):
    """``ChatDeepSeek`` that keeps the thinking-mode protocol intact.

    The model itself is stateless with respect to roles: reasoning intensity and
    the output cap arrive as per-request kwargs from ``agents/policies.py``.
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Thinking mode ignores temperature (the docs are explicit: setting it
        # is accepted but has no effect). Dropping it keeps the wire body honest
        # instead of implying a sampling setting that is not applied.
        if thinking_requested(payload):
            payload.pop("temperature", None)

        # Re-attach the previous turns' chain-of-thought: mandatory for any
        # request that carries `tools`, ignored (harmlessly) otherwise.
        restore_reasoning_content(self, input_, payload)
        return payload


def build_deepseek_model(
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout_s: float | None = 180.0,
    max_retries: int = 0,
    max_tokens: int | None = None,
    temperature: float | None = None,
    http_async_client: Any = None,
    model_kwargs: dict[str, Any] | None = None,
) -> DeepSeekThinkingModel:
    """Construct a DeepSeek chat model with the thinking protocol applied.

    Mirrors ``build_ecnu_model`` so the gateway and the registry can build both
    providers the same way; only the request-body spelling differs.
    """
    init: dict[str, Any] = {}
    if max_tokens is not None:
        init[MAX_TOKENS_FIELD] = max_tokens
    if temperature is not None:
        init["temperature"] = temperature
    if http_async_client is not None:
        init["http_async_client"] = http_async_client
    # ``response_format`` must travel inside ``model_kwargs``: passing it
    # top-level makes langchain-openai warn and move it there anyway.
    if model_kwargs:
        init["model_kwargs"] = dict(model_kwargs)
    return DeepSeekThinkingModel(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        timeout=timeout_s,
        max_retries=max_retries,
        **init,
    )


__all__ = ["DeepSeekThinkingModel", "MAX_TOKENS_FIELD", "build_deepseek_model"]
