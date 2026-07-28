from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, patch

from mindflow.services.analysis_service import AnalysisService
from mindflow.services.report_service import ReportService


async def test_analysis_queries_local_business_day_as_utc_range() -> None:
    activity_repo = AsyncMock()
    activity_repo.query_range = AsyncMock(return_value=[])
    focus_repo = AsyncMock()
    service = AnalysisService(
        activity_repo=activity_repo,
        focus_repo=focus_repo,
        timezone="Asia/Shanghai",
    )

    await service.identify_focus_sessions(1, date(2026, 7, 17))

    activity_repo.query_range.assert_awaited_once_with(
        1,
        datetime(2026, 7, 16, 16, 0, tzinfo=UTC),
        datetime(2026, 7, 17, 15, 59, 59, 999999, tzinfo=UTC),
    )


async def test_report_queries_local_business_day_as_utc_range() -> None:
    activity_repo = AsyncMock()
    activity_repo.query_range = AsyncMock(return_value=[])
    focus_repo = AsyncMock()
    focus_repo.get_by_date = AsyncMock(return_value=[])
    report_repo = AsyncMock()
    report_repo.get_by_date = AsyncMock(return_value=None)
    report_repo.upsert = AsyncMock(side_effect=lambda report: report)
    service = ReportService(
        activity_repo=activity_repo,
        focus_repo=focus_repo,
        report_repo=report_repo,
        timezone="Asia/Shanghai",
    )

    await service.generate_daily_report(1, date(2026, 7, 17))

    activity_repo.query_range.assert_awaited_once_with(
        1,
        datetime(2026, 7, 16, 16, 0, tzinfo=UTC),
        datetime(2026, 7, 17, 15, 59, 59, 999999, tzinfo=UTC),
    )


async def test_scheduled_service_wrappers_use_local_business_date() -> None:
    target = date(2026, 7, 17)
    analysis = AnalysisService(AsyncMock(), AsyncMock(), timezone="Asia/Shanghai")
    report = ReportService(AsyncMock(), AsyncMock(), AsyncMock(), timezone="Asia/Shanghai")
    analysis.identify_focus_sessions = AsyncMock(return_value=[])  # type: ignore[method-assign]
    report.generate_daily_report = AsyncMock(return_value={})  # type: ignore[method-assign]

    with (
        patch("mindflow.services.analysis_service.business_today", return_value=target),
        patch("mindflow.services.report_service.business_today", return_value=target),
    ):
        await analysis.identify_all_today()
        await report.generate_daily_for_all()

    analysis.identify_focus_sessions.assert_awaited_once_with(1, target)
    report.generate_daily_report.assert_awaited_once_with(
        1,
        target,
        refresh=False,
    )
