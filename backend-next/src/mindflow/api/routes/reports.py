"""API routes for daily and weekly reports.

Endpoints:
  - GET /reports/daily (get or generate a daily report)
  - GET /reports/weekly (weekly summary with 7-day trend)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query  # noqa: B008
from loguru import logger

from mindflow.api.deps import get_report_service
from mindflow.api.errors import _not_found
from mindflow.services.report_service import ReportService

router = APIRouter(tags=["reports"])


@router.get("/reports/daily")
async def get_daily_report(
    report_date: date | None = Query(  # noqa: B008
        None, alias="date", description="Report date (YYYY-MM-DD, default today)"
    ),
    report_svc: ReportService = Depends(get_report_service),  # noqa: B008
) -> dict[str, Any]:
    """Return a daily report for the given date (generates if missing)."""
    target = report_date or date.today()

    report = await report_svc.generate_daily_report(1, target)
    if not report:
        raise _not_found(f"日期 {target.isoformat()} 的报告")

    logger.debug("Daily report returned for {}", target)
    return report


@router.get("/reports/weekly")
async def get_weekly_report(
    week_start: date | None = Query(  # noqa: B008
        None, alias="week_start", description="Week start (YYYY-MM-DD, ISO week start Monday)"
    ),
    report_svc: ReportService = Depends(get_report_service),  # noqa: B008
) -> dict[str, Any]:
    """Return a weekly report with 7-day trend and week-over-week comparison.

    If *week_start* is omitted, the current ISO week is used.
    """
    if week_start:
        start = week_start
    else:
        today = date.today()
        # ISO week start (Monday)
        start = today - timedelta(days=today.weekday())

    report = await report_svc.weekly_report(1, start)
    if not report or not report.get("daily_reports"):
        raise _not_found(f"周报 {start.isoformat()} 的数据")

    return report
