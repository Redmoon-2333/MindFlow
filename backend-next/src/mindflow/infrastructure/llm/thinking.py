"""Shared thinking-mode helpers for OpenAI-compatible chat models.

DeepSeek's thinking mode (and the campus gateway that mimics it) has one
protocol requirement that LangChain's converter does not satisfy on its own:
once a request carries ``tools``, **every** later request must echo the previous
assistant turn's ``reasoning_content`` back to the API, otherwise the provider
rejects the call with HTTP 400.

LangChain keeps unknown response fields (we store ``reasoning_content`` in
``additional_kwargs``) but its outgoing message converter only serialises the
known keys, so the chain-of-thought would be dropped on the way back.  This
module holds the single implementation both adapters share.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage

#: The wire field the provider expects on assistant messages.
REASONING_CONTENT_FIELD = "reasoning_content"


def restore_reasoning_content(
    model: Any,
    input_: LanguageModelInput,
    payload: dict[str, Any],
) -> None:
    """Copy stored ``reasoning_content`` from LangChain messages onto *payload*.

    Args:
        model: Any ``ChatOpenAI``-style model (it must expose
            ``_convert_input``); passed explicitly so this stays a free function
            shared by the ECNU and DeepSeek adapters.
        input_: The messages handed to the model for this request.
        payload: The already-built request body, mutated in place.

    The converter preserves order and length for the supported message types, so
    positional pairing is safe here; anything unexpected is skipped rather than
    guessed at.
    """
    messages = model._convert_input(input_).to_messages()  # noqa: SLF001 - shared adapter seam
    wire = payload.get("messages")
    if not isinstance(wire, list) or not messages:
        return

    for lc_msg, wire_msg in zip(messages, wire, strict=False):
        if not isinstance(lc_msg, AIMessage) or not isinstance(wire_msg, dict):
            continue
        if wire_msg.get("role") != "assistant":
            continue
        reasoning = lc_msg.additional_kwargs.get(REASONING_CONTENT_FIELD)
        if isinstance(reasoning, str) and reasoning:
            wire_msg[REASONING_CONTENT_FIELD] = reasoning


def thinking_requested(payload: dict[str, Any]) -> bool:
    """True when *payload* asks for thinking mode on this request.

    Either an explicit ``reasoning_effort`` or a ``thinking`` object in
    ``extra_body`` counts; both are per-request fields the role policies set.
    """
    if isinstance(payload.get("reasoning_effort"), str) and payload["reasoning_effort"]:
        return True
    extra = payload.get("extra_body")
    return bool(isinstance(extra, dict) and extra.get("thinking"))


__all__ = [
    "REASONING_CONTENT_FIELD",
    "restore_reasoning_content",
    "thinking_requested",
]
