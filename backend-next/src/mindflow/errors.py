"""Root exception taxonomy for MindFlow.

Before this module, seven exception classes across four modules each
subclassed ``RuntimeError`` directly, with two near-identical pairs
(``GatewayNotConfiguredError``/``LLMNotConfiguredError`` and
``GatewayAPIError``/``LLMAPIError`` — the legacy ``DeepSeekClient`` vs the
new ``LangChainGateway``). This module unifies them under a single root so
callers can catch by category (``LLMError``, ``PanelError``, …) instead of
by concrete leaf type.

Design decisions:
  - Root is ``MindFlowError(Exception)``, NOT ``RuntimeError``. The API
    layer (``api/errors.py``) registers a catch-all handler for both
    ``RuntimeError`` and ``Exception``, so an ``Exception`` root is still
    caught and rendered as RFC 9457 ``internal-error``. No code path does
    ``isinstance(x, RuntimeError)`` on these exceptions (verified across
    src/ and tests/), so widening the root to ``Exception`` is safe.
  - The canonical LLM exceptions live here so the two legacy/new pairs
    share one identity. ``infrastructure/llm/client.py`` re-exports them
    unchanged, and ``agents/llm_gateway.py`` defines its ``Gateway*`` names
    as subclasses — so ``except GatewayAPIError`` and ``except LLMAPIError``
    both keep working, and either import site keeps resolving.
  - Panel/Collector categories are defined here; the concrete leaf classes
    stay in their original modules (``agents/types.py``,
    ``infrastructure/collectors/base.py``) and now inherit from these
    categories instead of ``RuntimeError``.
"""

from __future__ import annotations


class MindFlowError(Exception):
    """Root of the MindFlow exception hierarchy.

    Subclass ``Exception`` (not ``RuntimeError``): the ``api/errors.py``
    catch-all handler covers ``Exception``, and nothing relies on these
    being ``RuntimeError`` instances.
    """


# ── LLM tier ─────────────────────────────────────────────────────────────────


class LLMError(MindFlowError):
    """Base for errors originating from an LLM provider or gateway."""


class LLMNotConfiguredError(LLMError):
    """Raised when an LLM client/gateway is used without an API key.

    Canonical home for what used to be two separate classes: the legacy
    ``DeepSeekClient``'s ``LLMNotConfiguredError`` and the LangChain
    gateway's ``GatewayNotConfiguredError`` (now a subclass of this).
    """


class LLMAPIError(LLMError):
    """Raised on a non-retriable upstream error or exhausted retry budget.

    Canonical home for the former ``LLMAPIError`` / ``GatewayAPIError`` pair
    (the gateway's ``GatewayAPIError`` is now a subclass of this).
    """


# ── Expert panel tier ──────────────────────────────────────────────────────────


class PanelError(MindFlowError):
    """Base for expert-panel deliberation failures (degradation chain)."""


# ── Collector tier ─────────────────────────────────────────────────────────────


class CollectorError(MindFlowError):
    """Base for platform-specific event-collector failures."""


# ── Service-layer domain errors ────────────────────────────────────────────────


class NoActivityDataError(MindFlowError):
    """Raised by services when no activity events exist for the request.

    Lets the service layer signal "nothing to analyse" without importing the
    HTTP layer (``api/errors``). The API boundary maps this to an RFC 9457
    404 via a registered handler, so the dependency points downward
    (api → services) instead of the reverse.

    Attributes:
        resource: Human-readable Chinese description of what was missing,
            used verbatim as the 404 ``detail`` (``未找到{resource}``).
    """

    def __init__(self, resource: str = "活动数据") -> None:
        super().__init__(resource)
        self.resource = resource
