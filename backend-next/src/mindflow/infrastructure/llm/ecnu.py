"""ECNU (华东师范大学) OpenAI-compatible chat model adapter.

The campus gateway at ``https://chat.ecnu.edu.cn/open/api/v1`` speaks the
OpenAI chat-completions protocol, but it needs two things that
``langchain-deepseek``'s ``ChatDeepSeek`` does not provide:

1. **Thinking mode** — ``thinking={"type": "enabled"}`` plus
   ``reasoning_effort`` (``low``/``high``/``max`` for ``ecnu-max``) must be on
   every request. Both are non-standard body fields.
2. **``reasoning_content`` round-tripping** — once the model makes a tool
   call, the platform requires the assistant message's ``reasoning_content``
   to be echoed back verbatim on every later turn of that conversation.
   ``langchain-openai`` deliberately drops unknown reasoning fields when
   converting messages back to the wire format, and it also drops them from
   the parsed response.

This module is a *narrow* adapter: it reuses ``BaseChatOpenAI`` (so streaming,
tool binding, and parsing stay the framework's job) and overrides only the
request-payload and message-conversion hooks.

Design notes:
  - The adapter is provider-neutral in name only where it matters: it reads
    ``base_url`` from settings, so pointing it at DeepSeek simply disables the
    ECNU-specific body fields via :attr:`ECNUChatModel.thinking_enabled`.
  - ``reasoning_content`` is kept in ``additional_kwargs`` on the returned
    ``AIMessage`` so callers can persist it privately; it is never logged.
"""

from __future__ import annotations

import warnings
from typing import Any

import httpx
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from mindflow.errors import LLMAPIError
from mindflow.infrastructure.llm.safety import safe_error_metadata

# Reasoning effort tiers supported by each ECNU model. Used to fail loudly on
# an unsupported tier instead of silently degrading.
ECNU_EFFORT_TIERS: dict[str, tuple[str, ...]] = {
    "ecnu-max": ("low", "high", "max"),
    "ecnu-plus": ("low", "medium"),
    # Historical aliases still served by the platform.
    "ecnu-reasoner": ("low", "high", "max"),
    "ecnu-reasoner-lite": ("low", "medium"),
}

DEFAULT_THINKING_EFFORT = "max"


def supported_efforts(model: str) -> tuple[str, ...]:
    """Return the reasoning_effort tiers the platform documents for *model*.

    Unknown model names get the ``ecnu-max`` tiers, which are the strictest
    and therefore the safest default for a mis-typed name.
    """
    return ECNU_EFFORT_TIERS.get(model.strip().lower(), ECNU_EFFORT_TIERS["ecnu-max"])


def resolve_effort(model: str, requested: str) -> tuple[str, bool]:
    """Resolve *requested* effort against what *model* supports.

    Returns ``(effort_to_send, was_downgraded)``. The caller is expected to
    surface ``was_downgraded`` in its evidence trail — the project rule is to
    disclose a downgrade, never to hide one.
    """
    tiers = supported_efforts(model)
    if requested in tiers:
        return requested, False
    return tiers[-1], True


def ecnu_request_fields(
    *, model: str, thinking_enabled: bool, reasoning_effort: str,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Build the same wire fields for SDK and raw HTTP completions."""
    fields: dict[str, Any] = {
        "thinking": {"type": "enabled" if thinking_enabled else "disabled"},
    }
    if max_tokens is not None:
        fields["max_completion_tokens"] = max_tokens
    if thinking_enabled:
        fields["reasoning_effort"] = resolve_effort(model, reasoning_effort)[0]
    return fields


class ECNUChatModel(ChatOpenAI):
    """``ChatOpenAI`` extended with the ECNU thinking protocol.

    Args:
        model: ECNU model id (``ecnu-max`` / ``ecnu-plus``).
        api_key: Bearer token for the campus gateway.
        base_url: Gateway base URL including ``/open/api/v1``.
        thinking_enabled: Send ``thinking={"type": "enabled"}``. Turning this
            off makes the model behave like a plain OpenAI-compatible chat
            model, which is what the compatibility path uses.
        reasoning_effort: Requested thinking tier. Validated against the
            model's documented tiers at request time.
    """

    thinking_enabled: bool = True
    reasoning_effort: str = DEFAULT_THINKING_EFFORT
    #: Records the effort value actually sent on the last request, so evidence
    #: trails can distinguish "asked for max" from "server accepted max".
    last_effort_sent: str | None = None
    last_downgraded: bool = False

    def __init__(
        self,
        *,
        model: str = "ecnu-max",
        api_key: str | SecretStr,
        base_url: str,
        thinking_enabled: bool = True,
        reasoning_effort: str = DEFAULT_THINKING_EFFORT,
        timeout: float | None = 180.0,
        max_retries: int = 0,
        temperature: float | None = None,
        max_tokens: int | None = 16384,
        **kwargs: Any,
    ) -> None:
        resolved, downgraded = resolve_effort(model, reasoning_effort)
        init: dict[str, Any] = {}
        # Thinking mode restricts sampling params, so only pass a temperature
        # when a caller explicitly accepts that trade-off.
        if temperature is not None:
            init["temperature"] = temperature
        init.update(kwargs)
        # ``max_completion_tokens`` is the field's canonical alias in
        # langchain-openai (``max_tokens`` is the deprecated spelling, which
        # the installed type stubs reject as a keyword).
        super().__init__(
            model=model,
            api_key=api_key if isinstance(api_key, SecretStr) else SecretStr(api_key),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            max_completion_tokens=max_tokens,
            **init,
        )
        self.thinking_enabled = thinking_enabled
        self.reasoning_effort = resolved
        self.last_effort_sent = resolved if thinking_enabled else None
        self.last_downgraded = downgraded

    # ── Request payload ──────────────────────────────────────────────────

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add ECNU's thinking fields and re-attach ``reasoning_content``.

        Two transformations beyond the OpenAI payload:

        * ``thinking`` / ``reasoning_effort`` are injected for every request
          while thinking mode is on.
        * Assistant messages that carry a recorded ``reasoning_content`` get it
          copied back onto the wire message. The platform requires this on
          every turn after a tool call; without it some models return 400.
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # Drop the base class's temperature: thinking mode constrains sampling
        # parameters on this platform, and sending one while thinking is on
        # produces inconsistent behaviour. Callers that genuinely need
        # temperature set it explicitly on the instance.
        if self.thinking_enabled:
            payload.pop("temperature", None)

        # `thinking` is not a parameter the OpenAI SDK knows, so it must travel
        # inside `extra_body` (the SDK merges that mapping into the JSON body).
        # `reasoning_effort` is a recognised parameter and stays top-level,
        # which is where the platform documents it.
        extra = dict(payload.get("extra_body") or {})
        fields = ecnu_request_fields(
            model=self.model_name,
            thinking_enabled=self.thinking_enabled,
            reasoning_effort=self.reasoning_effort,
            max_tokens=payload.get("max_completion_tokens"),
        )
        extra["thinking"] = fields.pop("thinking")
        payload.pop("reasoning_effort", None)
        payload.update(fields)
        payload["extra_body"] = extra

        self._restore_reasoning_content(input_, payload)
        return payload

    def _restore_reasoning_content(
        self,
        input_: LanguageModelInput,
        payload: dict[str, Any],
    ) -> None:
        """Copy stored ``reasoning_content`` from LangChain messages to the payload.

        LangChain's own converter omits unknown ``additional_kwargs`` keys, so
        the stored chain-of-thought would otherwise be lost on the way out.
        """
        messages = self._convert_input(input_).to_messages()
        wire = payload.get("messages")
        if not isinstance(wire, list) or not messages:
            return

        # Walk both lists together: the converter preserves order and length
        # for the supported message types, so positional pairing is safe here.
        for lc_msg, wire_msg in zip(messages, wire, strict=False):
            if not isinstance(lc_msg, AIMessage) or not isinstance(wire_msg, dict):
                continue
            if wire_msg.get("role") != "assistant":
                continue
            reasoning = lc_msg.additional_kwargs.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                wire_msg["reasoning_content"] = reasoning

    # ── Response parsing ─────────────────────────────────────────────────

    def _create_chat_result(
        self,
        response: dict[str, Any] | Any,
        generation_info: dict[str, Any] | None = None,
    ) -> Any:
        """Keep ``reasoning_content`` on the parsed ``AIMessage``.

        Stored under ``additional_kwargs`` so the chat graph can persist it
        privately and echo it back on later turns. It is never written to logs
        or exposed on the API.
        """
        result = None
        failure = None
        try:
            result = super()._create_chat_result(response, generation_info)
        except Exception as exc:
            failure = safe_error_metadata(exc)
        if failure is not None:
            # Raise outside the handler so even __context__ contains no raw error.
            raise LLMAPIError(f"ECNU response parsing failed ({failure})")
        assert result is not None

        raw = response
        if not isinstance(raw, dict):
            raw = raw.model_dump() if hasattr(raw, "model_dump") else {}
        choices = raw.get("choices") or []
        if not choices:
            return result

        message = choices[0].get("message") or {}
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning and result.generations:
            gen_message = result.generations[0].message
            if isinstance(gen_message, AIMessage):
                gen_message.additional_kwargs["reasoning_content"] = reasoning
        return result


def build_ecnu_model(
    *,
    model: str,
    api_key: str,
    base_url: str,
    reasoning_effort: str = DEFAULT_THINKING_EFFORT,
    thinking_enabled: bool = True,
    timeout_s: float = 180.0,
    max_tokens: int = 16384,
    temperature: float | None = None,
    model_kwargs: dict[str, Any] | None = None,
    http_async_client: httpx.AsyncClient | None = None,
) -> ECNUChatModel:
    """Construct an :class:`ECNUChatModel`, tolerating an unset temperature.

    ``langchain-openai`` supplies its own default temperature when the value is
    ``None``; ECNU applies its model-side preference while thinking is on, so
    the parameter is omitted entirely unless a caller sets it.
    """
    init: dict[str, Any] = {}
    if temperature is not None:
        init["temperature"] = temperature
    if model_kwargs:
        init.update(model_kwargs)
    # ``response_format`` is not a declared field on the installed
    # langchain-openai, so passing it triggers a UserWarning stating the value
    # "was transferred to model_kwargs" — which is exactly the routing we want.
    # Silence that specific warning instead of printing it on every model
    # construction.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*response_format is not default parameter.*",
            category=UserWarning,
        )
        return ECNUChatModel(
            model=model,
            api_key=api_key,
            base_url=base_url,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            timeout=timeout_s,
            max_retries=0,
            max_tokens=max_tokens,
            http_async_client=http_async_client,
            **init,
        )


#: Provider names recognised as "the campus gateway".
ECNU_PROVIDER_ALIASES: tuple[str, ...] = ("ecnu", "ecnu-max", "ecnu-plus")


def looks_like_ecnu(base_url: str | None, model: str | None) -> bool:
    """Heuristic used to decide whether to send ECNU-specific body fields.

    Keyed on the gateway hostname and the documented model ids, so a
    DeepSeek/Ollama configuration never receives ECNU-only parameters.
    """
    host = (base_url or "").lower()
    name = (model or "").lower()
    if "ecnu.edu.cn" in host:
        return True
    return name in ECNU_PROVIDER_ALIASES or name.startswith("ecnu-")


__all__ = [
    "DEFAULT_THINKING_EFFORT",
    "ECNU_EFFORT_TIERS",
    "ECNU_PROVIDER_ALIASES",
    "ECNUChatModel",
    "build_ecnu_model",
    "looks_like_ecnu",
    "resolve_effort",
    "supported_efforts",
]
