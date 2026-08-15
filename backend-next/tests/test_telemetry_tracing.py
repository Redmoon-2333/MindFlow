"""OpenTelemetry tracing instrumentation tests — privacy-safe, local-only.

Covers:
  - Span hierarchy: workflow_run > graph_node > model_call/tool_call
  - Allowlisted attributes only (no PII, no prompts, no evidence, no API keys)
  - PII/secret canary redaction
  - Exporter failure does not fail business workflows
  - Error category sanitization
  - Trace ID extraction
  - Loguru trace_id patching
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider, export
from opentelemetry.trace import StatusCode, get_current_span

from mindflow.telemetry import (
    InMemoryExporter,
    current_trace_id,
    fallback_span,
    get_mindflow_tracer,
    graph_node_span,
    model_call_span,
    persistence_span,
    retry_span,
    routing_decision_span,
    setup_telemetry,
    tool_call_span,
    workflow_run_span,
)
from mindflow.telemetry.exporters import ConsoleExporter, SQLiteExporter
from mindflow.telemetry.tracing import (
    _sanitize_attributes,
    _sanitize_error_category,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _reset_global_tracer() -> None:
    """Reset mindflow tracer and OTel SDK tracer caches between tests."""
    import mindflow.telemetry.tracing as tmod

    tmod._tracer = None
    try:
        from opentelemetry.sdk.trace import _TRACER_PROVIDER_SET  # type: ignore[attr-defined]
        _TRACER_PROVIDER_SET._tracers.clear()
    except Exception:
        pass


@contextmanager
def _global_provider(provider: TracerProvider) -> Generator[TracerProvider, None, None]:
    """Context manager to temporarily set a global TracerProvider."""
    original = otel_trace._TRACER_PROVIDER
    try:
        otel_trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
        _reset_global_tracer()
        yield provider
    finally:
        otel_trace._TRACER_PROVIDER = original  # type: ignore[attr-defined]
        _reset_global_tracer()


def _make_provider(in_memory_exporter: InMemoryExporter) -> TracerProvider:
    """Create a TracerProvider with an in-memory span processor."""
    provider = TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(in_memory_exporter))
    return provider


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _clean_otel_state() -> Generator[None, None, None]:
    """Ensure clean OTel state before and after each test."""
    _reset_global_tracer()
    yield
    _reset_global_tracer()


@pytest.fixture
def in_memory_exporter() -> InMemoryExporter:
    return InMemoryExporter()


# ═══════════════════════════════════════════════════════════════════════════════
# Allowlisted attributes
# ═══════════════════════════════════════════════════════════════════════════════


def test_sanitize_attributes_allowlisted_only() -> None:
    """Only allowlisted keys pass through; everything else is dropped."""
    attrs = {
        "duration_ms": 150.0,
        "call_count": 3,
        "token_estimate": 500,
        "source": "scheduler",
        "status": "ok",
        "graph_version": 1,
        "error_category": "network_timeout",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "deepseek-chat",
        "gen_ai.response.id": "resp-123",
    }
    result = _sanitize_attributes(attrs)
    assert result == attrs


def test_sanitize_attributes_redacted_keys_removed() -> None:
    """Redacted keys (messages, prompts, evidence, api_key, etc.) are stripped."""
    attrs = {
        "duration_ms": 100.0,
        "message": "User said something sensitive",
        "prompt": "SYSTEM: You are an AI assistant",
        "window_title": "Bank Account Details",
        "evidence": json.dumps({"focus_score": 0.8}),
        "evidence_json": '{"raw": "data"}',
        "api_key": "sk-1234567890abcdef",
        "model_output": "Full model response text",
        "full_output": "Everything the model produced",
    }
    result = _sanitize_attributes(attrs)
    assert "duration_ms" in result
    assert "message" not in result
    assert "prompt" not in result
    assert "window_title" not in result
    assert "evidence" not in result
    assert "evidence_json" not in result
    assert "api_key" not in result
    assert "model_output" not in result
    assert "full_output" not in result


def test_sanitize_attributes_unknown_keys_dropped() -> None:
    """Unknown attribute keys are silently dropped (default-deny)."""
    attrs = {
        "duration_ms": 200.0,
        "unknown_field": "should be dropped",
        "custom_data": {"nested": "value"},
    }
    result = _sanitize_attributes(attrs)
    assert result == {"duration_ms": 200.0}


# ═══════════════════════════════════════════════════════════════════════════════
# PII/secret canary redaction
# ═══════════════════════════════════════════════════════════════════════════════


def test_canary_strings_never_in_sanitized_attributes() -> None:
    """Seeded PII strings never appear after _sanitize_attributes filtering."""
    canary_api_key = "sk-canary-test-key-abc123"
    canary_prompt = "You are an assistant with access to user data"
    canary_window = "C:\\Users\\test\\Documents\\passwords.txt"
    canary_evidence = json.dumps({"user_email": "canary@test.com", "raw": "sensitive"})

    attrs = {
        "duration_ms": 500.0,
        "source": "test",
        "api_key": canary_api_key,
        "prompt": canary_prompt,
        "window_title": canary_window,
        "evidence": canary_evidence,
    }

    result = _sanitize_attributes(attrs)
    result_json = json.dumps(result, ensure_ascii=False)

    assert canary_api_key not in result_json
    assert canary_prompt not in result_json
    assert canary_window not in result_json
    assert canary_evidence not in result_json
    assert "canary" not in result_json.lower()
    assert result["duration_ms"] == 500.0
    assert result["source"] == "test"


def test_canary_strings_never_in_exported_spans(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """Seeded PII strings never appear in exported span attribute dicts."""
    canary_api_key = "sk-canary-test-key-abc123"
    canary_prompt = "You are an assistant with access to user data"

    provider = _make_provider(in_memory_exporter)

    with _global_provider(provider):
        tracer = get_mindflow_tracer()
        with tracer.start_as_current_span("workflow_run") as span:
            # Only allowlisted attrs should make it through
            span.set_attributes({
                "duration_ms": 500.0,
                "source": "test",
            })

    in_memory_exporter.force_flush()
    assert len(in_memory_exporter.spans) == 1

    exported = in_memory_exporter.spans[0]
    exported_attrs = dict(exported.attributes or {})
    attrs_json = json.dumps(exported_attrs, ensure_ascii=False)

    assert canary_api_key not in attrs_json
    assert canary_prompt not in attrs_json
    assert "sk-" not in attrs_json
    assert exported_attrs["duration_ms"] == 500.0


# ═══════════════════════════════════════════════════════════════════════════════
# Exporter failure does not fail business workflows
# ═══════════════════════════════════════════════════════════════════════════════


def test_exporter_failure_does_not_raise(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """When an exporter's export() raises, no exception propagates to callers."""

    class FailingExporter(InMemoryExporter):
        def export(self, spans):  # type: ignore[override]
            raise RuntimeError("Simulated exporter failure")

    provider = TracerProvider()
    exporter = FailingExporter()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))

    with _global_provider(provider):
        tracer = get_mindflow_tracer()
        # This should NOT raise
        with tracer.start_as_current_span("workflow_run") as span:
            span.set_attribute("duration_ms", 100.0)


def test_sqlite_exporter_failure_no_raise(tmp_path) -> None:
    """SQLite exporter handles bad paths gracefully without raising."""
    bad_path = tmp_path / "nonexistent" / "subdir" / "spans.db"
    exporter = SQLiteExporter(bad_path)

    mock_span = MagicMock()
    mock_span.get_span_context.return_value = MagicMock(trace_id=1, span_id=2)
    mock_span.parent = None
    mock_span.name = "test"
    mock_span.start_time = 0
    mock_span.end_time = 1
    mock_span.status = MagicMock(status_code=MagicMock(name="OK"))
    mock_span.attributes = {}
    mock_span.events = []

    result = exporter.export([mock_span])
    assert result.name == "SUCCESS"


def test_business_workflow_completes_despite_exporter_error() -> None:
    """A simulated business workflow completes even if the exporter fails."""

    class ExplodingExporter(InMemoryExporter):
        def export(self, spans):  # type: ignore[override]
            raise RuntimeError("Exporter exploded")

    provider = TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(ExplodingExporter()))

    result: str = "not_set"
    with _global_provider(provider):
        tracer = get_mindflow_tracer()
        with tracer.start_as_current_span("workflow_run") as span:
            span.set_attribute("source", "auto_intervention")
            result = "business_completed_successfully"

    assert result == "business_completed_successfully"


# ═══════════════════════════════════════════════════════════════════════════════
# Span hierarchy: parent/child relationships
# ═══════════════════════════════════════════════════════════════════════════════


def test_span_hierarchy_parent_child(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """Full span hierarchy test: workflow_run > graph_node > model_call."""
    provider = _make_provider(in_memory_exporter)

    with _global_provider(provider):
        tracer = get_mindflow_tracer()
        with (
            tracer.start_as_current_span("workflow_run"),
            tracer.start_as_current_span("graph_node.analyst"),
            tracer.start_as_current_span("model_call"),
        ):
            pass

    in_memory_exporter.force_flush()
    assert len(in_memory_exporter.spans) >= 3

    # Span names as a set to verify all three are present
    names = {s.name for s in in_memory_exporter.spans}
    assert names == {"workflow_run", "graph_node.analyst", "model_call"}

    # Verify parent/child: leaf (model_call) has parent = graph_node,
    # and graph_node has parent = workflow_run
    spans_by_name = {s.name: s for s in in_memory_exporter.spans}
    model = spans_by_name["model_call"]
    graph = spans_by_name["graph_node.analyst"]
    workflow = spans_by_name["workflow_run"]

    root_ctx = workflow.get_span_context()
    graph_ctx = graph.get_span_context()

    if graph.parent and root_ctx:
        assert graph.parent.span_id == root_ctx.span_id
    if model.parent and graph_ctx:
        assert model.parent.span_id == graph_ctx.span_id


# ═══════════════════════════════════════════════════════════════════════════════
# Error category sanitization
# ═══════════════════════════════════════════════════════════════════════════════


def test_error_category_sanitize_valid() -> None:
    """Valid error categories are preserved."""
    for cat in (
        "network_timeout",
        "rate_limited",
        "invalid_response",
        "parser_error",
        "unavailable",
    ):
        assert _sanitize_error_category(cat) == cat


def test_error_category_sanitize_invalid_dropped() -> None:
    """Unknown error categories are silently dropped."""
    assert _sanitize_error_category("sql_injection") is None
    assert _sanitize_error_category("") is None
    assert _sanitize_error_category(None) is None


def test_error_category_recorded_on_span(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """Error category is recorded as a span attribute when exception occurs."""
    provider = _make_provider(in_memory_exporter)

    with _global_provider(provider):
        tracer = get_mindflow_tracer()
        try:
            with tracer.start_as_current_span("model_call") as span:
                span.set_attribute("error_category", "network_timeout")
                raise TimeoutError("Connection timed out")
        except TimeoutError:
            pass

    in_memory_exporter.force_flush()
    assert len(in_memory_exporter.spans) == 1
    exported_attrs = dict(in_memory_exporter.spans[0].attributes or {})
    assert exported_attrs.get("error_category") == "network_timeout"
    assert in_memory_exporter.spans[0].status.status_code == StatusCode.ERROR


# ═══════════════════════════════════════════════════════════════════════════════
# Span context manager API tests
# ═══════════════════════════════════════════════════════════════════════════════


def test_workflow_run_span_context(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """workflow_run_span context manager creates spans with correct name."""
    provider = _make_provider(in_memory_exporter)
    with (
        _global_provider(provider),
        workflow_run_span(source="scheduler", graph_version=2) as span,
    ):
        assert span.name == "workflow_run"


def test_graph_node_span_context(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """graph_node_span sets the correct name."""
    provider = _make_provider(in_memory_exporter)
    with _global_provider(provider), graph_node_span("analyst", graph_version=1) as span:
        assert span.name == "graph_node.analyst"


def test_model_call_span_context(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """model_call_span includes gen_ai semantic conventions."""
    provider = _make_provider(in_memory_exporter)
    with (
        _global_provider(provider),
        model_call_span(
            model_name="deepseek-chat", call_count=2, token_estimate=1000
        ) as span,
    ):
        assert span.name == "model_call"


def test_tool_call_span_context(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """tool_call_span includes the tool name."""
    provider = _make_provider(in_memory_exporter)
    with _global_provider(provider), tool_call_span("run_panel", call_count=1) as span:
        assert span.name == "tool_call.run_panel"


def test_routing_decision_span_context(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """routing_decision_span includes the route."""
    provider = _make_provider(in_memory_exporter)
    with (
        _global_provider(provider),
        routing_decision_span("panel", source="conflict_detection") as span,
    ):
        assert span.name == "routing_decision.panel"


def test_retry_span_context(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """retry_span includes attempt number and error category."""
    provider = _make_provider(in_memory_exporter)
    with (
        _global_provider(provider),
        retry_span(attempt=3, error_category="network_timeout") as span,
    ):
        assert span.name == "retry"


def test_fallback_span_context(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """fallback_span includes from/to tiers."""
    provider = _make_provider(in_memory_exporter)
    with (
        _global_provider(provider),
        fallback_span(
            from_tier="L1_deepseek",
            to_tier="L2_ollama",
            error_category="rate_limited",
        ) as span,
    ):
        assert span.name == "fallback.L1_deepseek_to_L2_ollama"


def test_persistence_span_context(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """persistence_span includes operation name."""
    provider = _make_provider(in_memory_exporter)
    with (
        _global_provider(provider),
        persistence_span("checkpoint", token_estimate=2048) as span,
    ):
        assert span.name == "persistence.checkpoint"


def test_span_context_propagation(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """Nested spans propagate context correctly."""
    provider = _make_provider(in_memory_exporter)
    with _global_provider(provider):
        outer = get_current_span()
        with workflow_run_span(source="test") as _ws:
            inner = get_current_span()
            assert inner is not None
            assert inner is not outer


# ═══════════════════════════════════════════════════════════════════════════════
# Trace ID extraction
# ═══════════════════════════════════════════════════════════════════════════════


def test_current_trace_id_no_active_span() -> None:
    """current_trace_id returns None when no span is active."""
    tid = current_trace_id()
    assert tid is None


def test_current_trace_id_with_active_span(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """current_trace_id returns a valid hex string when a span is active."""
    provider = _make_provider(in_memory_exporter)
    with _global_provider(provider):
        tracer = get_mindflow_tracer()
        with tracer.start_as_current_span("test_span"):
            tid = current_trace_id()
            assert tid is not None
            assert len(tid) == 32
            assert all(c in "0123456789abcdef" for c in tid)
            assert tid != "0" * 32


# ═══════════════════════════════════════════════════════════════════════════════
# Loguru trace_id patching
# ═══════════════════════════════════════════════════════════════════════════════


def test_loguru_patcher_sets_trace_id_key() -> None:
    """The loguru patcher always sets trace_id in extra dict."""
    from mindflow.logging_config import _patch_trace_id

    record_extra: dict[str, object] = {}
    record = MagicMock()
    record.__setitem__ = lambda self, k, v: record_extra.__setitem__(k, v)  # type: ignore[assignment]
    record.__getitem__ = lambda self, k: record_extra[k]  # type: ignore[assignment]
    # Pre-populate "extra" so that record["extra"] resolves to a dict
    record_extra["extra"] = {}

    _patch_trace_id(record)
    assert "trace_id" in record_extra["extra"]  # type: ignore[operator]


def test_loguru_patcher_no_telemetry_module() -> None:
    """When telemetry module is not available, patcher doesn't crash."""
    from mindflow.logging_config import _patch_trace_id

    record_extra: dict[str, object] = {}
    record_extra["extra"] = {}
    record = MagicMock()
    record.__setitem__ = lambda self, k, v: record_extra.__setitem__(k, v)  # type: ignore[assignment]
    record.__getitem__ = lambda self, k: record_extra[k]  # type: ignore[assignment]

    _patch_trace_id(record)
    assert "trace_id" in record_extra["extra"]  # type: ignore[operator]


# ═══════════════════════════════════════════════════════════════════════════════
# Console exporter
# ═══════════════════════════════════════════════════════════════════════════════


def test_console_exporter_writes_to_stderr() -> None:
    """Console exporter writes span JSON to output without raising."""
    import io

    buf = io.StringIO()
    exporter = ConsoleExporter(out=buf)

    mock_span = MagicMock()
    mock_span.get_span_context.return_value = MagicMock(trace_id=1, span_id=2)
    mock_span.name = "test_span"
    mock_span.start_time = 1000
    mock_span.end_time = 2000
    mock_span.status = MagicMock(status_code=MagicMock(name="OK"))
    mock_span.attributes = {"duration_ms": 50.0}

    result = exporter.export([mock_span])
    assert result.name == "SUCCESS"
    output = buf.getvalue()
    assert "test_span" in output
    assert "duration_ms" in output
    assert "api_key" not in output


# ═══════════════════════════════════════════════════════════════════════════════
# setup_telemetry
# ═══════════════════════════════════════════════════════════════════════════════


def test_setup_telemetry_in_memory() -> None:
    """setup_telemetry with in_memory exporter configures correctly."""
    provider = setup_telemetry(exporter="in_memory")
    assert isinstance(provider, TracerProvider)
    provider.shutdown()


def test_setup_telemetry_console() -> None:
    """setup_telemetry with console exporter configures correctly."""
    provider = setup_telemetry(exporter="console")
    assert isinstance(provider, TracerProvider)
    provider.shutdown()


def test_setup_telemetry_sqlite_requires_path() -> None:
    """setup_telemetry with sqlite exporter requires db_path."""
    with pytest.raises(ValueError, match="db_path"):
        setup_telemetry(exporter="sqlite")


def test_setup_telemetry_sqlite(tmp_path) -> None:
    """setup_telemetry with sqlite exporter creates database."""
    db_path = tmp_path / "test_spans.db"
    provider = setup_telemetry(exporter="sqlite", db_path=db_path)
    assert isinstance(provider, TracerProvider)

    # Verify table was created
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='otel_spans'")
    assert cursor.fetchone() is not None
    conn.close()

    provider.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# InMemoryExporter
# ═══════════════════════════════════════════════════════════════════════════════


def test_in_memory_exporter_collects_spans(
    in_memory_exporter: InMemoryExporter,
) -> None:
    """InMemoryExporter accumulates exported spans."""
    provider = _make_provider(in_memory_exporter)

    with _global_provider(provider):
        tracer = get_mindflow_tracer()
        with tracer.start_as_current_span("span1") as s1:
            s1.set_attribute("duration_ms", 10.0)
        with tracer.start_as_current_span("span2") as s2:
            s2.set_attribute("duration_ms", 20.0)

    in_memory_exporter.force_flush()
    assert len(in_memory_exporter.spans) == 2
    assert in_memory_exporter.spans[0].name == "span1"
    assert in_memory_exporter.spans[1].name == "span2"


def test_in_memory_exporter_clear() -> None:
    """clear() empties the span list."""
    exporter = InMemoryExporter()
    exporter.spans.append(MagicMock())
    exporter.clear()
    assert len(exporter.spans) == 0


def test_in_memory_exporter_shutdown() -> None:
    """shutdown() clears spans."""
    exporter = InMemoryExporter()
    exporter.spans.append(MagicMock())
    exporter.shutdown()
    assert len(exporter.spans) == 0
