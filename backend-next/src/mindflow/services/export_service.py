"""Data export service — CSV/JSON export of user activity data.

Generates an archive of events, focus sessions, and daily reports for a
given date range.  Empty ranges produce an empty archive (not an error).

Exports never include window titles (privacy NF-S3a) — only app names and
aggregated metrics.

Design:
  - CSV uses stdlib ``csv`` + ``io.StringIO`` (zero extra dependencies).
  - JSON is a structured dict with three keys: ``events``, ``focus_sessions``,
    ``daily_reports``.
  - All timestamps are ISO8601 strings (timezone-aware UTC).
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from mindflow.domain.events import ActivityEvent
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.infrastructure.repositories.report import (
    SQLAlchemyDailyReportRepository,
)

ExportFormat = Literal["csv", "json"]


@dataclass(frozen=True)
class ExportStreamResult:
    """Metadata and byte iterator for a streaming export."""

    content: AsyncIterator[bytes]
    filename: str
    media_type: str


# ── CSV column headers ───────────────────────────────────────────────────

_EVENTS_CSV_HEADERS: list[str] = [
    "id",
    "user_id",
    "timestamp_utc",
    "duration_s",
    "event_type",
    "app_name",
    "process_name",
    "is_idle",
]

_FOCUS_CSV_HEADERS: list[str] = [
    "id",
    "user_id",
    "date",
    "start_time",
    "end_time",
    "session_type",
    "dominant_app",
    "focus_score",
    "switch_count",
]

_REPORTS_CSV_HEADERS: list[str] = [
    "id",
    "user_id",
    "date",
    "total_focus_min",
    "total_distraction_min",
    "focus_score",
    "switch_frequency",
    "pattern_summary",
]


def _csv_safe(value: str) -> str:
    """Neutralise CSV/DDE formula injection for a single field value.

    Excel/LibreOffice execute a cell as a formula when it starts with
    ``=``, ``+``, ``-``, ``@``, or a tab/CR — a malicious/misbehaving app
    naming itself e.g. ``=cmd|'/c calc'!A1`` would run when the exported
    CSV is opened. Prefixing with a single quote makes affected spreadsheet
    apps treat the cell as literal text instead (the standard mitigation;
    OWASP calls this "CSV Injection"). Only apply to collector/user
    -influenceable strings (app_name, process_name, dominant_app,
    pattern_summary) — NOT to fields we fully control (ids, dates, numeric
    aggregates, the fixed session_type enum), since those can never carry
    attacker-chosen content.
    """
    if value and value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _event_payload(event: ActivityEvent) -> dict[str, Any]:
    """Return the export-allowlisted event fields.

    Shared by the CSV and JSON streamers so both formats expose exactly the
    same field set (mirroring the CSV headers) and never leak snapshot
    internals such as ``window_title`` (privacy NF-S3a). Values are raw —
    each format applies its own escaping/typing (``_csv_safe`` in CSV, native
    JSON types in JSON).
    """
    return {
        "id": event.id,
        "user_id": event.user_id,
        "timestamp_utc": event.timestamp_utc.isoformat(),
        "duration_s": event.duration_s,
        "event_type": event.event_type,
        "app_name": event.data.app_name,
        "process_name": event.data.process_name,
        "is_idle": event.data.is_idle,
    }


# ── Public API ───────────────────────────────────────────────────────────


class ExportService:
    """Data export service — generates CSV/JSON archives.

    Args:
        activity_repo: Activity event repository.
        focus_repo: Focus session repository.
        report_repo: Daily report repository.
        user_id: Default user identifier (default 1 for single-user mode).
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        focus_repo: SQLAlchemyFocusSessionRepository,
        report_repo: SQLAlchemyDailyReportRepository,
        user_id: int = 1,
    ) -> None:
        self._activity_repo = activity_repo
        self._focus_repo = focus_repo
        self._report_repo = report_repo
        self._user_id = user_id

    # ── Main export entry point ──────────────────────────────────────

    async def stream_events(
        self,
        start: datetime,
        end: datetime,
        fmt: ExportFormat = "csv",
        *,
        chunk_size: int = 1000,
    ) -> ExportStreamResult:
        """Return a bounded-memory export stream."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        focus_sessions, daily_reports = await asyncio.gather(
            self._focus_repo.query_range(self._user_id, start.date(), end.date()),
            self._report_repo.query_range(self._user_id, start.date(), end.date()),
        )
        date_suffix = f"{start.date().isoformat()}_{end.date().isoformat()}"
        if fmt == "csv":
            content = self._stream_csv(
                start,
                end,
                focus_sessions,
                daily_reports,
                chunk_size=chunk_size,
            )
            return ExportStreamResult(
                content=content,
                filename=f"mindflow_export_{date_suffix}.csv",
                media_type="text/csv; charset=utf-8",
            )
        content = self._stream_json(
            start,
            end,
            focus_sessions,
            daily_reports,
            chunk_size=chunk_size,
        )
        return ExportStreamResult(
            content=content,
            filename=f"mindflow_export_{date_suffix}.json",
            media_type="application/json; charset=utf-8",
        )

    async def _stream_csv(
        self,
        start: datetime,
        end: datetime,
        focus_sessions: list[dict[str, Any]],
        daily_reports: list[dict[str, Any]],
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes]:
        output = io.StringIO()
        output.write("\ufeff# Events\n")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(_EVENTS_CSV_HEADERS)
        yield output.getvalue().encode("utf-8")

        async for events in self._activity_repo.iter_range_chunks(
            self._user_id,
            start,
            end,
            chunk_size=chunk_size,
        ):
            output = io.StringIO()
            writer = csv.writer(output, lineterminator="\n")
            for event in events:
                payload = _event_payload(event)
                writer.writerow([
                    payload["id"],
                    payload["user_id"],
                    payload["timestamp_utc"],
                    payload["duration_s"],
                    payload["event_type"],
                    _csv_safe(payload["app_name"]),
                    _csv_safe(payload["process_name"]),
                    "1" if payload["is_idle"] else "0",
                ])
            yield output.getvalue().encode("utf-8")

        output = io.StringIO()
        output.write("# Focus Sessions\n")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(_FOCUS_CSV_HEADERS)
        for session in focus_sessions:
            writer.writerow([
                session.get("id", ""),
                session.get("user_id", ""),
                session.get("date", ""),
                session.get("start_time", ""),
                session.get("end_time", ""),
                session.get("session_type", ""),
                _csv_safe(session.get("dominant_app") or ""),
                session.get("focus_score", ""),
                session.get("switch_count", ""),
            ])
        output.write("# Daily Reports\n")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(_REPORTS_CSV_HEADERS)
        for report in daily_reports:
            writer.writerow([
                report.get("id", ""),
                report.get("user_id", ""),
                report.get("date", ""),
                report.get("total_focus_min", ""),
                report.get("total_distraction_min", ""),
                report.get("focus_score", ""),
                report.get("switch_frequency", ""),
                _csv_safe(report.get("pattern_summary") or ""),
            ])
        yield output.getvalue().encode("utf-8")

    async def _stream_json(
        self,
        start: datetime,
        end: datetime,
        focus_sessions: list[dict[str, Any]],
        daily_reports: list[dict[str, Any]],
        *,
        chunk_size: int,
    ) -> AsyncIterator[bytes]:
        yield b'{"events":['
        first = True
        async for events in self._activity_repo.iter_range_chunks(
            self._user_id,
            start,
            end,
            chunk_size=chunk_size,
        ):
            for event in events:
                prefix = b"" if first else b","
                first = False
                yield prefix + json.dumps(
                    _event_payload(event),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
        suffix = (
            '],"focus_sessions":'
            + json.dumps(focus_sessions, ensure_ascii=False, separators=(",", ":"))
            + ',"daily_reports":'
            + json.dumps(daily_reports, ensure_ascii=False, separators=(",", ":"))
            + "}"
        )
        yield suffix.encode("utf-8")

    # ── Format validation ───────────────────────────────────────────

    @staticmethod
    def validate_format(fmt: str) -> ExportFormat:
        """Validate and normalise an export format string.

        Args:
            fmt: The format string (``"csv"`` or ``"json"``).

        Returns:
            The normalised format literal.

        Raises:
            ValueError: If *fmt* is not ``"csv"`` or ``"json"``.
        """
        normalised = fmt.strip().lower()
        if normalised in ("csv", "json"):
            from typing import cast
            result: Literal["csv", "json"] = cast("Literal['csv', 'json']", normalised)
            return result
        msg = f"Unsupported export format: {fmt!r}. Must be 'csv' or 'json'."
        raise ValueError(msg)
