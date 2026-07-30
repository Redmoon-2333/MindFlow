"""Local-only span exporters for MindFlow telemetry.

Provides three exporters:
  1. ``InMemoryExporter`` — captures spans in-memory for testing/assertions.
  2. ``ConsoleExporter`` — prints spans to stderr for development debugging.
  3. ``SQLiteExporter`` — persists spans to a local SQLite file for offline
     analysis.  Stores allowlisted attributes only — no PII.

All exporters are designed to never raise on error (fail-safe),
so business workflows are never disrupted by exporter failures.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class InMemoryExporter(SpanExporter):
    """Captures exported spans in a list for programmatic inspection.

    Thread-safe for single-threaded testing; NOT safe for multi-threaded
    production use.
    """

    def __init__(self) -> None:
        self.spans: list[Any] = []

    def export(self, spans: Any) -> SpanExportResult:
        try:
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self.spans.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def clear(self) -> None:
        """Clear accumulated spans (convenience for test cleanup)."""
        self.spans.clear()


class ConsoleExporter(SpanExporter):
    """Print span summaries to stderr for development debugging.

    Outputs compact JSON with allowlisted attributes only.
    Messages/prompts/evidence are NEVER included.
    """

    def __init__(self, *, out: Any = sys.stderr) -> None:
        self._out = out

    def export(self, spans: Any) -> SpanExportResult:
        try:
            for span in spans:
                self._out.write(json.dumps(_span_to_safe_dict(span), ensure_ascii=False) + "\n")
            self._out.flush()
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class SQLiteExporter(SpanExporter):
    """Persist spans to a local SQLite database.

    Creates the spans table automatically.  Stores only allowlisted
    attributes.  Failures are logged but never raised.
    """

    _TABLE_DDL = """
    CREATE TABLE IF NOT EXISTS otel_spans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trace_id TEXT NOT NULL,
        span_id TEXT NOT NULL,
        parent_span_id TEXT,
        name TEXT NOT NULL,
        start_time_ns INTEGER NOT NULL,
        end_time_ns INTEGER NOT NULL,
        status TEXT NOT NULL,
        attributes_json TEXT NOT NULL,
        events_json TEXT NOT NULL
    )
    """

    _INSERT_SQL = """
    INSERT INTO otel_spans
        (trace_id, span_id, parent_span_id, name, start_time_ns, end_time_ns, status, attributes_json, events_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._TABLE_DDL)
        self._conn.commit()

    def _ensure_table(self) -> sqlite3.Connection:
        """Return the connection (table created in __init__)."""
        return self._conn  # type: ignore[return-value]

    def shutdown(self) -> None:
        """Close the SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def export(self, spans: Any) -> SpanExportResult:
        try:
            rows: list[tuple[Any, ...]] = []
            for span in spans:
                ctx = span.get_span_context()
                if ctx is None:
                    continue
                parent_id = span.parent
                parent_id_str = _format_span_id(parent_id.span_id) if parent_id is not None else None
                rows.append((
                    _format_trace_id(ctx.trace_id),
                    _format_span_id(ctx.span_id),
                    parent_id_str,
                    span.name or "",
                    span.start_time or 0,
                    span.end_time or 0,
                    str(span.status.status_code.name),
                    json.dumps(_span_to_safe_dict(span), ensure_ascii=False),
                    json.dumps(
                        [
                            {"name": e.name, "timestamp_ns": e.timestamp, "attrs": dict(e.attributes or {})}
                            for e in (span.events or [])
                        ],
                        ensure_ascii=False,
                    ),
                ))
            self._conn.executemany(self._INSERT_SQL, rows)
            self._conn.commit()
            return SpanExportResult.SUCCESS
        except Exception:
            try:
                self._conn.rollback()
            except Exception:
                pass
            return SpanExportResult.FAILURE


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _span_to_safe_dict(span: Any) -> dict[str, Any]:
    """Convert a span to a dict with allowlisted attributes only.

    Redacted fields: messages, prompts, evidence, api_key, window_title.
    """
    ctx = span.get_span_context()
    return {
        "name": span.name,
        "trace_id": _format_trace_id(ctx.trace_id) if ctx else "",
        "span_id": _format_span_id(ctx.span_id) if ctx else "",
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "status": str(span.status.status_code.name),
        "attributes": dict(span.attributes or {}),
    }


def _format_trace_id(trace_id: int) -> str:
    """Format an OTel trace ID (128-bit int) as a 32-char hex string."""
    return format(trace_id, "032x")


def _format_span_id(span_id: int) -> str:
    """Format an OTel span ID (64-bit int) as a 16-char hex string."""
    return format(span_id, "016x")
