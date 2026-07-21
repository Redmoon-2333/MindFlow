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
class ExportResult:
    """Result of an export operation.

    Attributes:
        content: The exported data as bytes.
        filename: Suggested filename for download.
        media_type: MIME type of the content.
    """

    content: bytes
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

    async def export_events(
        self,
        start: datetime,
        end: datetime,
        fmt: ExportFormat = "csv",
    ) -> ExportResult:
        """Export activity data for the given time range.

        Args:
            start: Inclusive start of the time range (timezone-aware UTC).
            end: Inclusive end of the time range (timezone-aware UTC).
            fmt: Output format — ``"csv"`` or ``"json"``.

        Returns:
            An ``ExportResult`` with content, filename, and media type.
            Empty ranges produce an empty archive (not an error).
        """
        # These three reads are independent (no data dependency) — fetch
        # them concurrently instead of three sequential round-trips.
        events, focus_sessions, daily_reports = await asyncio.gather(
            self._activity_repo.query_range(self._user_id, start, end),
            self._focus_repo.query_range(self._user_id, start.date(), end.date()),
            self._report_repo.query_range(self._user_id, start.date(), end.date()),
        )

        date_suffix = f"{start.date().isoformat()}_{end.date().isoformat()}"

        if fmt == "csv":
            content = self._build_csv(events, focus_sessions, daily_reports)
            return ExportResult(
                content=content.encode("utf-8-sig"),
                filename=f"mindflow_export_{date_suffix}.csv",
                media_type="text/csv; charset=utf-8",
            )

        content = self._build_json(events, focus_sessions, daily_reports)
        return ExportResult(
            content=content.encode("utf-8"),
            filename=f"mindflow_export_{date_suffix}.json",
            media_type="application/json; charset=utf-8",
        )

    # ── CSV builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_csv(
        events: list[ActivityEvent],
        focus_sessions: list[dict[str, Any]],
        daily_reports: list[dict[str, Any]],
    ) -> str:
        """Build a CSV string with three sections.

        Sections are separated by comment lines (``# Section Name``).
        """
        output = io.StringIO()
        output.write("﻿")  # BOM for Excel compatibility

        # ── Events section ─────────────────────────────────────────
        output.write("# Events\n")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(_EVENTS_CSV_HEADERS)
        for ev in events:
            writer.writerow([
                ev.id,
                ev.user_id,
                ev.timestamp_utc.isoformat(),
                ev.duration_s,
                ev.event_type,
                _csv_safe(ev.data.app_name),
                _csv_safe(ev.data.process_name),
                "1" if ev.data.is_idle else "0",
            ])

        # ── Focus sessions section ──────────────────────────────────
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

        # ── Daily reports section ───────────────────────────────────
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

        return output.getvalue()

    # ── JSON builder ────────────────────────────────────────────────

    @staticmethod
    def _build_json(
        events: list[ActivityEvent],
        focus_sessions: list[dict[str, Any]],
        daily_reports: list[dict[str, Any]],
    ) -> str:
        """Build a JSON string with three top-level keys."""
        import json

        data: dict[str, Any] = {
            "events": [ev.to_dict() for ev in events],
            "focus_sessions": focus_sessions,
            "daily_reports": daily_reports,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

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
