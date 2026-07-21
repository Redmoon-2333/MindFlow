"""Tests for services/export_service.py — CSV/JSON export logic.

Covers:
  - CSV format: correct column headers, BOM, section separation
  - JSON format: correct structure with events/focus_sessions/daily_reports
  - Empty date range: returns empty sections (not an error)
  - Boundary dates: inclusive range filtering
  - Format validation
  - F4: CSV formula-injection escaping for collector-sourced text fields
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from mindflow.domain.events import make_event
from mindflow.services.export_service import ExportResult, ExportService, _csv_safe


class TestExportService:
    """ExportService end-to-end tests with mocked repositories."""

    @pytest.fixture
    def events(self) -> list[Any]:
        """Sample events for testing."""
        base = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
        return [
            make_event(
                user_id=1,
                timestamp_utc=base + __import__("datetime").timedelta(seconds=i * 10),
                duration_s=10.0,
                app_name="Code.exe",
                process_name="Code.exe",
            )
            for i in range(5)
        ]

    @pytest.fixture
    def focus_sessions(self) -> list[dict[str, Any]]:
        """Sample focus sessions."""
        return [
            {
                "id": "fs-1",
                "user_id": 1,
                "date": "2026-07-17",
                "start_time": "2026-07-17T08:00:00+00:00",
                "end_time": "2026-07-17T08:30:00+00:00",
                "session_type": "focus",
                "dominant_app": "Code.exe",
                "focus_score": 85.0,
                "switch_count": 5,
            },
            {
                "id": "fs-2",
                "user_id": 1,
                "date": "2026-07-17",
                "start_time": "2026-07-17T09:00:00+00:00",
                "end_time": "2026-07-17T09:45:00+00:00",
                "session_type": "focus",
                "dominant_app": "Terminal",
                "focus_score": 72.0,
                "switch_count": 8,
            },
        ]

    @pytest.fixture
    def daily_reports(self) -> list[dict[str, Any]]:
        """Sample daily reports."""
        return [
            {
                "id": "dr-1",
                "user_id": 1,
                "date": "2026-07-17",
                "total_focus_min": 120.0,
                "total_distraction_min": 30.0,
                "focus_score": 78.5,
                "switch_frequency": 12.0,
                "pattern_summary": "上午专注度较高，下午有所下降",
            },
        ]

    @pytest.fixture
    def mock_activity_repo(self, events: list[Any]) -> AsyncMock:
        repo = AsyncMock()
        repo.query_range = AsyncMock(return_value=events)
        return repo

    @pytest.fixture
    def mock_focus_repo(self, focus_sessions: list[dict[str, Any]]) -> AsyncMock:
        repo = AsyncMock()
        repo.query_range = AsyncMock(return_value=focus_sessions)
        return repo

    @pytest.fixture
    def mock_report_repo(self, daily_reports: list[dict[str, Any]]) -> AsyncMock:
        repo = AsyncMock()
        repo.query_range = AsyncMock(return_value=daily_reports)
        return repo

    @pytest.fixture
    def service(
        self,
        mock_activity_repo: AsyncMock,
        mock_focus_repo: AsyncMock,
        mock_report_repo: AsyncMock,
    ) -> ExportService:
        return ExportService(
            activity_repo=mock_activity_repo,
            focus_repo=mock_focus_repo,
            report_repo=mock_report_repo,
        )

    # ── CSV tests ──────────────────────────────────────────────────

    async def test_csv_has_correct_headers(self, service: ExportService) -> None:
        """CSV export should include all three section headers with correct columns."""
        start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        result: ExportResult = await service.export_events(start, end, fmt="csv")

        assert isinstance(result.content, bytes)
        assert result.media_type.startswith("text/csv")
        assert result.filename.endswith(".csv")

        body = result.content.decode("utf-8-sig")
        lines = body.splitlines()

        # Check section headers
        assert "# Events" in lines[0] or "# Events" in body
        assert "# Focus Sessions" in body
        assert "# Daily Reports" in body

        # Check column headers
        assert "id,user_id,timestamp_utc,duration_s" in body
        assert "event_type,app_name,process_name,is_idle" in body
        assert "start_time,end_time,session_type,dominant_app" in body
        assert "focus_score,switch_count" in body
        assert "total_focus_min,total_distraction_min,focus_score" in body
        assert "switch_frequency,pattern_summary" in body

    async def test_csv_contains_data_rows(self, service: ExportService, events: list[Any]) -> None:
        """CSV export should include data rows from all three sections."""
        start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        result: ExportResult = await service.export_events(start, end, fmt="csv")

        body = result.content.decode("utf-8-sig")

        # Events data rows
        assert "Code.exe" in body
        # Header + 5 data rows = events section
        assert body.count("Code.exe") >= 5  # At least one per row

    async def test_csv_empty_range(
        self, mock_activity_repo: AsyncMock, mock_focus_repo: AsyncMock, mock_report_repo: AsyncMock
    ) -> None:
        """Empty date range should return CSV with only headers (no data rows)."""
        mock_activity_repo.query_range = AsyncMock(return_value=[])
        mock_focus_repo.query_range = AsyncMock(return_value=[])
        mock_report_repo.query_range = AsyncMock(return_value=[])
        svc = ExportService(
            activity_repo=mock_activity_repo,
            focus_repo=mock_focus_repo,
            report_repo=mock_report_repo,
        )

        start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC)
        result: ExportResult = await svc.export_events(start, end, fmt="csv")

        body = result.content.decode("utf-8-sig")
        # Headers present but no data rows beyond headers
        assert "# Events" in body
        assert "Code.exe" not in body

    # ── JSON tests ─────────────────────────────────────────────────

    async def test_json_structure(self, service: ExportService) -> None:
        """JSON export should have events/focus_sessions/daily_reports keys."""
        start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        result: ExportResult = await service.export_events(start, end, fmt="json")

        assert result.media_type.startswith("application/json")
        assert result.filename.endswith(".json")

        import json

        data = json.loads(result.content.decode("utf-8"))
        assert "events" in data
        assert "focus_sessions" in data
        assert "daily_reports" in data

    async def test_json_events_use_to_dict(self, service: ExportService, events: list[Any]) -> None:
        """JSON events should be serialised via ActivityEvent.to_dict."""
        start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        result: ExportResult = await service.export_events(start, end, fmt="json")

        import json

        data = json.loads(result.content.decode("utf-8"))
        assert len(data["events"]) == len(events)
        assert "id" in data["events"][0]
        assert "data" in data["events"][0]
        assert "app_name" in data["events"][0]["data"]

    async def test_json_empty_range(
        self, mock_activity_repo: AsyncMock, mock_focus_repo: AsyncMock, mock_report_repo: AsyncMock
    ) -> None:
        """Empty date range should return JSON with empty lists."""
        mock_activity_repo.query_range = AsyncMock(return_value=[])
        mock_focus_repo.query_range = AsyncMock(return_value=[])
        mock_report_repo.query_range = AsyncMock(return_value=[])
        svc = ExportService(
            activity_repo=mock_activity_repo,
            focus_repo=mock_focus_repo,
            report_repo=mock_report_repo,
        )

        start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC)
        result: ExportResult = await svc.export_events(start, end, fmt="json")

        import json

        data = json.loads(result.content.decode("utf-8"))
        assert data["events"] == []
        assert data["focus_sessions"] == []
        assert data["daily_reports"] == []

    # ── Format validation ──────────────────────────────────────────

    @pytest.mark.parametrize(
        ("fmt", "expected"),
        [
            ("csv", "csv"),
            ("json", "json"),
            (" CSV ", "csv"),
            ("JSON", "json"),
        ],
    )
    def test_validate_format_valid(self, fmt: str, expected: str) -> None:
        """Valid format strings should be normalised correctly."""
        assert ExportService.validate_format(fmt) == expected

    @pytest.mark.parametrize(
        "fmt",
        ["xml", "yaml", "pdf", ""],
    )
    def test_validate_format_invalid(self, fmt: str) -> None:
        """Invalid format strings should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            ExportService.validate_format(fmt)

    # ── Filename includes date range ───────────────────────────────

    async def test_filename_contains_dates(self, service: ExportService) -> None:
        """Filename should include start and end dates."""
        start = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        result: ExportResult = await service.export_events(start, end, fmt="csv")
        assert "2026-07-01" in result.filename
        assert "2026-07-17" in result.filename

    # ── Boundary: repositories called with correct range ───────────

    async def test_export_calls_repos_with_range(
        self, mock_activity_repo: AsyncMock, mock_focus_repo: AsyncMock, mock_report_repo: AsyncMock
    ) -> None:
        """Repositories should be called with the requested date range."""
        svc = ExportService(
            activity_repo=mock_activity_repo,
            focus_repo=mock_focus_repo,
            report_repo=mock_report_repo,
        )

        start = datetime(2026, 7, 15, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        await svc.export_events(start, end, fmt="json")

        mock_activity_repo.query_range.assert_awaited_once_with(1, start, end)
        mock_focus_repo.query_range.assert_awaited_once_with(1, start.date(), end.date())
        mock_report_repo.query_range.assert_awaited_once_with(1, start.date(), end.date())


class TestCsvSafe:
    """F4: unit tests for the CSV formula-injection escape helper."""

    @pytest.mark.parametrize(
        "dangerous",
        [
            "=1+1",
            "=cmd|'/c calc'!A1",
            "+1+1",
            "-1+1",
            "@SUM(A1:A9)",
            "\tstarts with tab",
            "\rstarts with CR",
        ],
    )
    def test_dangerous_prefix_gets_quoted(self, dangerous: str) -> None:
        """Values starting with =, +, -, @, tab, or CR get a leading single quote."""
        assert _csv_safe(dangerous) == "'" + dangerous

    @pytest.mark.parametrize(
        "safe",
        ["Code.exe", "chrome", "文档编辑器", "", "notepad-plus-plus"],
    )
    def test_safe_value_unchanged(self, safe: str) -> None:
        """Ordinary values (including empty string) pass through untouched."""
        assert _csv_safe(safe) == safe

    def test_dash_in_middle_not_escaped(self) -> None:
        """A dash that isn't the first character must not trigger escaping."""
        assert _csv_safe("notepad-plus-plus") == "notepad-plus-plus"


class TestCsvInjectionInExport:
    """F4: end-to-end — a malicious app_name/dominant_app/pattern_summary
    must come out prefixed with a quote in the actual CSV output, not just
    in the _csv_safe unit tests above."""

    @pytest.fixture
    def malicious_events(self) -> list[Any]:
        base = datetime(2026, 7, 17, 8, 0, 0, tzinfo=UTC)
        return [
            make_event(
                user_id=1,
                timestamp_utc=base,
                duration_s=10.0,
                app_name="=cmd|'/c calc'!A1",
                process_name="+2+3",
            )
        ]

    @pytest.fixture
    def malicious_focus_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "fs-1",
                "user_id": 1,
                "date": "2026-07-17",
                "start_time": "2026-07-17T08:00:00+00:00",
                "end_time": "2026-07-17T08:30:00+00:00",
                "session_type": "focus",
                "dominant_app": "=1+1",
                "focus_score": 85.0,
                "switch_count": 5,
            },
        ]

    @pytest.fixture
    def malicious_daily_reports(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "dr-1",
                "user_id": 1,
                "date": "2026-07-17",
                "total_focus_min": 120.0,
                "total_distraction_min": 30.0,
                "focus_score": 78.5,
                "switch_frequency": 12.0,
                "pattern_summary": "@SUM(1,2)",
            },
        ]

    @pytest.fixture
    def malicious_service(
        self,
        malicious_events: list[Any],
        malicious_focus_sessions: list[dict[str, Any]],
        malicious_daily_reports: list[dict[str, Any]],
    ) -> ExportService:
        activity_repo = AsyncMock()
        activity_repo.query_range = AsyncMock(return_value=malicious_events)
        focus_repo = AsyncMock()
        focus_repo.query_range = AsyncMock(return_value=malicious_focus_sessions)
        report_repo = AsyncMock()
        report_repo.query_range = AsyncMock(return_value=malicious_daily_reports)
        return ExportService(
            activity_repo=activity_repo,
            focus_repo=focus_repo,
            report_repo=report_repo,
        )

    async def test_malicious_app_name_escaped_in_csv(
        self, malicious_service: ExportService
    ) -> None:
        """A formula-shaped app_name must appear quote-prefixed in the CSV, not raw."""
        start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        result: ExportResult = await malicious_service.export_events(start, end, fmt="csv")
        body = result.content.decode("utf-8-sig")

        # The raw (unescaped) field must not appear right after a comma —
        # note the escaped form "'=cmd..." legitimately *contains* the raw
        # substring, so this must check the delimiter boundary, not just
        # substring presence.
        assert ",=cmd|'/c calc'!A1," not in body
        assert "'=cmd|'/c calc'!A1" in body
        assert "'+2+3" in body

    async def test_malicious_dominant_app_escaped_in_csv(
        self, malicious_service: ExportService
    ) -> None:
        """A formula-shaped dominant_app must be escaped in the Focus Sessions section."""
        start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        result: ExportResult = await malicious_service.export_events(start, end, fmt="csv")
        body = result.content.decode("utf-8-sig")

        assert "'=1+1" in body

    async def test_malicious_pattern_summary_escaped_in_csv(
        self, malicious_service: ExportService
    ) -> None:
        """A formula-shaped pattern_summary must be escaped in the Daily Reports section."""
        start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        result: ExportResult = await malicious_service.export_events(start, end, fmt="csv")
        body = result.content.decode("utf-8-sig")

        assert "'@SUM(1,2)" in body

    async def test_json_export_leaves_raw_value_untouched(
        self, malicious_service: ExportService
    ) -> None:
        """JSON export is not spreadsheet-rendered, so no escaping is applied there."""
        import json

        start = datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)
        end = datetime(2026, 7, 17, 23, 59, 59, tzinfo=UTC)
        result: ExportResult = await malicious_service.export_events(start, end, fmt="json")
        data = json.loads(result.content.decode("utf-8"))

        assert data["events"][0]["data"]["app_name"] == "=cmd|'/c calc'!A1"
