"""Privacy-safe OpenTelemetry span definitions for MindFlow workflows.

Span hierarchy:
    workflow_run > graph_node > model_call / tool_call
    workflow_run > routing_decision / retry / fallback / persistence

Allowlisted attributes: duration_ms, call_count, token_estimate, source,
status, graph_version, error_category.

REDACTED (never recorded): raw messages, prompts, window_titles, evidence JSON,
API keys, full model output.

Sanitized error categories: network_timeout, rate_limited, invalid_response,
parser_error, unavailable.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal, cast

import opentelemetry.trace as trace_api
from opentelemetry.trace import Span, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

# ═══════════════════════════════════════════════════════════════════════════════
# Module-level tracer (created once, reused everywhere)
# ═══════════════════════════════════════════════════════════════════════════════

_tracer: Tracer | None = None


def get_mindflow_tracer() -> Tracer:
    """Return the module-level MindFlow tracer (lazy-init).

    Falls back to a no-op tracer when no TracerProvider is configured,
    preventing recursion in the default ProxyTracerProvider.
    """
    global _tracer
    if _tracer is None:
        try:
            provider = trace_api.get_tracer_provider()
            # If the provider is a ProxyTracerProvider with no delegate,
            # get_tracer can recurse.  Guard against that by providing
            # a local NoOpTracerProvider fallback.
            _tracer = provider.get_tracer("mindflow")
        except (RecursionError, RuntimeError):
            _tracer = trace_api.NoOpTracerProvider().get_tracer("mindflow")
    return _tracer


# ═══════════════════════════════════════════════════════════════════════════════
# Allowlisted attribute keys
# ═══════════════════════════════════════════════════════════════════════════════

_ALLOWLISTED_ATTRS: frozenset[str] = frozenset({
    "duration_ms",
    "call_count",
    "token_estimate",
    "source",
    "status",
    "graph_version",
    "error_category",
    # OpenTelemetry gen_ai semantic convention keys (allowlisted subset)
    "gen_ai.operation.name",
    "gen_ai.request.model",
    "gen_ai.response.id",
})

# Attribute names that are NEVER allowed and trigger redaction warnings.
_REDACTED_KEYS: frozenset[str] = frozenset({
    "message",
    "prompt",
    "window_title",
    "evidence",
    "evidence_json",
    "api_key",
    "model_output",
    "full_output",
})

# Sanitized error categories per ADR-003.
ErrorCategory = Literal[
    "network_timeout",
    "rate_limited",
    "invalid_response",
    "parser_error",
    "unavailable",
]

_VALID_ERROR_CATEGORIES: frozenset[str] = frozenset({
    "network_timeout",
    "rate_limited",
    "invalid_response",
    "parser_error",
    "unavailable",
})

# ═══════════════════════════════════════════════════════════════════════════════
# Attribute sanitization
# ═══════════════════════════════════════════════════════════════════════════════


def _sanitize_attributes(attrs: dict[str, object]) -> dict[str, AttributeValue]:
    """Strip redacted keys from attributes, keeping only allowlisted entries.

    Unknown keys are silently dropped (default-deny for privacy).
    """
    safe: dict[str, AttributeValue] = {}
    for key, value in attrs.items():
        if key in _REDACTED_KEYS:
            continue
        if key in _ALLOWLISTED_ATTRS:
            safe[key] = cast(AttributeValue, value)
    return safe


def _sanitize_error_category(category: str | None) -> str | None:
    """Validate and normalize an error category string.

    Returns None for unrecognized categories (silently dropped).
    """
    if category is None:
        return None
    return category if category in _VALID_ERROR_CATEGORIES else None


# ═══════════════════════════════════════════════════════════════════════════════
# Span context managers
# ═══════════════════════════════════════════════════════════════════════════════


def _set_safe_attrs(span: Span, attrs: dict[str, object]) -> None:
    """Set allowlisted attributes on a span."""
    safe = _sanitize_attributes(attrs)
    span.set_attributes(safe)


def _end_span(
    span: Span,
    error: Exception | None = None,
    error_category: str | None = None,
) -> None:
    """Finalize a span: record exception (sanitized) and set status."""
    if error is not None:
        cat = _sanitize_error_category(error_category)
        if cat is not None:
            span.set_attribute("error_category", cat)
        span.set_status(StatusCode.ERROR, str(type(error).__name__))
        span.record_exception(error)
    span.end()


@contextmanager
def workflow_run_span(
    *,
    source: str = "",
    graph_version: int = 1,
    **attrs: object,
) -> Generator[Span, None, None]:
    """Top-level workflow_run span — root of the trace hierarchy.

    Args:
        source: Origin of the workflow (e.g. "scheduler", "api", "chat").
        graph_version: Schema version of the graph being executed.
        **attrs: Additional allowlisted attributes.
    """
    tracer = get_mindflow_tracer()
    with tracer.start_as_current_span("workflow_run") as span:
        merged: dict[str, object] = {"source": source, "graph_version": graph_version, **attrs}
        _set_safe_attrs(span, merged)
        try:
            yield span
            span.set_status(StatusCode.OK)
        except Exception as exc:
            _end_span(span, exc, _categorize(exc))
            raise
        else:
            span.end()


@contextmanager
def graph_node_span(
    node_name: str,
    *,
    graph_version: int = 1,
    **attrs: object,
) -> Generator[Span, None, None]:
    """Span for a single graph node execution.

    Args:
        node_name: The graph node identifier (e.g. "analyst", "moderator").
        graph_version: Schema version of the graph being executed.
        **attrs: Additional allowlisted attributes.
    """
    tracer = get_mindflow_tracer()
    with tracer.start_as_current_span(f"graph_node.{node_name}") as span:
        merged: dict[str, object] = {"graph_version": graph_version, **attrs}
        _set_safe_attrs(span, merged)
        try:
            yield span
            span.set_status(StatusCode.OK)
        except Exception as exc:
            _end_span(span, exc, _categorize(exc))
            raise
        else:
            span.end()


@contextmanager
def model_call_span(
    *,
    model_name: str = "",
    call_count: int = 1,
    token_estimate: int = 0,
    **attrs: object,
) -> Generator[Span, None, None]:
    """Span for an LLM model inference call.

    Args:
        model_name: The model identifier (e.g. "deepseek-chat", "qwen3:8b").
        call_count: Number of API calls in this batch.
        token_estimate: Estimated token usage (input + output).
        **attrs: Additional allowlisted attributes.
    """
    tracer = get_mindflow_tracer()
    with tracer.start_as_current_span("model_call") as span:
        merged: dict[str, object] = {
            "call_count": call_count,
            "token_estimate": token_estimate,
            "gen_ai.request.model": model_name,
            "gen_ai.operation.name": "chat",
            **attrs,
        }
        _set_safe_attrs(span, merged)
        try:
            yield span
            span.set_status(StatusCode.OK)
        except Exception as exc:
            _end_span(span, exc, _categorize(exc))
            raise
        else:
            span.end()


@contextmanager
def tool_call_span(
    tool_name: str,
    *,
    call_count: int = 1,
    **attrs: object,
) -> Generator[Span, None, None]:
    """Span for a tool invocation (e.g. "run_panel", "get_evidence").

    Args:
        tool_name: The tool/function name being called.
        call_count: Number of tool invocations in this batch.
        **attrs: Additional allowlisted attributes.
    """
    tracer = get_mindflow_tracer()
    with tracer.start_as_current_span(f"tool_call.{tool_name}") as span:
        merged: dict[str, object] = {"call_count": call_count, **attrs}
        _set_safe_attrs(span, merged)
        try:
            yield span
            span.set_status(StatusCode.OK)
        except Exception as exc:
            _end_span(span, exc, _categorize(exc))
            raise
        else:
            span.end()


@contextmanager
def routing_decision_span(
    route: str,
    *,
    source: str = "",
    **attrs: object,
) -> Generator[Span, None, None]:
    """Span for a routing decision within the graph.

    Args:
        route: The chosen route (e.g. "panel", "single_expert", "rule_engine").
        source: Which component made the routing decision.
        **attrs: Additional allowlisted attributes.
    """
    tracer = get_mindflow_tracer()
    with tracer.start_as_current_span(f"routing_decision.{route}") as span:
        merged: dict[str, object] = {"source": source, **attrs}
        _set_safe_attrs(span, merged)
        try:
            yield span
        finally:
            span.end()


@contextmanager
def retry_span(
    attempt: int = 1,
    *,
    error_category: str | None = None,
    **attrs: object,
) -> Generator[Span, None, None]:
    """Span for a retry attempt after a failure.

    Args:
        attempt: 1-based retry attempt number.
        error_category: Sanitized error category string.
        **attrs: Additional allowlisted attributes.
    """
    tracer = get_mindflow_tracer()
    with tracer.start_as_current_span("retry") as span:
        cat = _sanitize_error_category(error_category)
        merged: dict[str, object] = {"call_count": attempt}
        if cat is not None:
            merged["error_category"] = cat
        merged.update(attrs)
        _set_safe_attrs(span, merged)
        try:
            yield span
        finally:
            span.end()


@contextmanager
def fallback_span(
    from_tier: str = "",
    to_tier: str = "",
    *,
    error_category: str | None = None,
    **attrs: object,
) -> Generator[Span, None, None]:
    """Span for a degradation/fallback event in the LLM tier chain.

    Args:
        from_tier: The tier that failed (e.g. "L1_deepseek").
        to_tier: The tier being fallen back to (e.g. "L2_ollama").
        error_category: Sanitized error category string.
        **attrs: Additional allowlisted attributes.
    """
    tracer = get_mindflow_tracer()
    span_name = f"fallback.{from_tier}_to_{to_tier}" if from_tier and to_tier else "fallback"
    with tracer.start_as_current_span(span_name) as span:
        cat = _sanitize_error_category(error_category)
        merged: dict[str, object] = {}
        if from_tier:
            merged["source"] = from_tier
        if cat is not None:
            merged["error_category"] = cat
        merged.update(attrs)
        _set_safe_attrs(span, merged)
        try:
            yield span
        finally:
            span.end()


@contextmanager
def persistence_span(
    operation: str,
    *,
    token_estimate: int = 0,
    **attrs: object,
) -> Generator[Span, None, None]:
    """Span for a persistence operation (save/load/checkpoint).

    Args:
        operation: The persistence operation type (e.g. "save", "load", "checkpoint").
        token_estimate: Estimated data size in bytes/records.
        **attrs: Additional allowlisted attributes.
    """
    tracer = get_mindflow_tracer()
    with tracer.start_as_current_span(f"persistence.{operation}") as span:
        merged: dict[str, object] = {"token_estimate": token_estimate, **attrs}
        _set_safe_attrs(span, merged)
        try:
            yield span
            span.set_status(StatusCode.OK)
        except Exception as exc:
            _end_span(span, exc, _categorize(exc))
            raise
        else:
            span.end()


# ═══════════════════════════════════════════════════════════════════════════════
# Error categorization
# ═══════════════════════════════════════════════════════════════════════════════


def _categorize(exc: Exception) -> str | None:
    """Map known exception types to sanitized error categories.

    Falls back to None for unknown exceptions (silently dropped).
    """
    exc_type_name = type(exc).__name__
    msg = str(exc).lower()

    if "timeout" in msg or "TimeoutError" in exc_type_name:
        return "network_timeout"
    if "rate" in msg and ("limit" in msg or "429" in msg):
        return "rate_limited"
    if "parse" in msg or "json" in msg or "pydantic" in exc_type_name.lower():
        return "parser_error"
    if "connect" in msg or "refused" in msg or "unavailable" in msg:
        return "unavailable"
    if "invalid" in msg or "400" in msg or "422" in msg:
        return "invalid_response"

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Trace ID extraction for Loguru context
# ═══════════════════════════════════════════════════════════════════════════════


def current_trace_id() -> str | None:
    """Return the current OpenTelemetry trace ID as a hex string, or None.

    Safe to call when no span is active — returns None without raising.
    """
    try:
        from opentelemetry.trace import get_current_span
    except ImportError:
        return None

    span = get_current_span()
    ctx = span.get_span_context()
    if ctx is None or ctx.trace_id == 0:
        return None
    return _format_trace_id(ctx.trace_id)


def _format_trace_id(trace_id: int) -> str:
    return format(trace_id, "032x")
