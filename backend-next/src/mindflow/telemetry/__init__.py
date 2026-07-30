"""MindFlow OpenTelemetry instrumentation — local-only, privacy-safe.

Exports:
  - ``get_mindflow_tracer()`` — module-level tracer used by all services.
  - ``setup_telemetry()`` — one-shot tracer provider initialisation.
  - Span context managers: ``workflow_run_span``, ``graph_node_span``,
    ``model_call_span``, ``tool_call_span``, ``routing_decision_span``,
    ``retry_span``, ``fallback_span``, ``persistence_span``.
  - ``current_trace_id()`` — extract active trace ID for Loguru context.
  - Exporters: ``InMemoryExporter``, ``ConsoleExporter``, ``SQLiteExporter``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from mindflow.telemetry.exporters import (
    ConsoleExporter,
    InMemoryExporter,
    SQLiteExporter,
)
from mindflow.telemetry.tracing import (
    current_trace_id,
    fallback_span,
    get_mindflow_tracer,
    graph_node_span,
    model_call_span,
    persistence_span,
    retry_span,
    routing_decision_span,
    tool_call_span,
    workflow_run_span,
)

__all__ = [
    "get_mindflow_tracer",
    "setup_telemetry",
    "current_trace_id",
    "workflow_run_span",
    "graph_node_span",
    "model_call_span",
    "tool_call_span",
    "routing_decision_span",
    "retry_span",
    "fallback_span",
    "persistence_span",
    "InMemoryExporter",
    "ConsoleExporter",
    "SQLiteExporter",
]


def setup_telemetry(
    *,
    exporter: str = "console",
    db_path: str | Path | None = None,
    provider: TracerProvider | None = None,
) -> TracerProvider:
    """One-shot setup of the global OpenTelemetry tracer provider.

    Args:
        exporter: One of ``"console"``, ``"in_memory"``, or ``"sqlite"``.
            - ``"console"``: prints spans to stderr (development default).
            - ``"in_memory"``: spans collected in-memory for testing.
            - ``"sqlite"``: persists spans to a local SQLite file.
        db_path: Required when ``exporter="sqlite"`` — path to the SQLite file.
        provider: Existing TracerProvider to configure (for testing).
            When None, creates a new provider and sets it as global.

    Returns:
        The configured TracerProvider instance.
    """
    if provider is None:
        provider = TracerProvider()

    span_exporter: Any
    if exporter == "in_memory":
        span_exporter = InMemoryExporter()
    elif exporter == "sqlite":
        if db_path is None:
            raise ValueError("db_path is required for sqlite exporter")
        span_exporter = SQLiteExporter(db_path)
    else:
        span_exporter = ConsoleExporter()

    # Use SimpleSpanProcessor for immediate export (tests/dev); BatchSpanProcessor
    # for production SQLite to avoid blocking on each span.
    if exporter == "sqlite":
        processor: Any = BatchSpanProcessor(span_exporter)
    else:
        processor = SimpleSpanProcessor(span_exporter)

    provider.add_span_processor(processor)

    return provider
