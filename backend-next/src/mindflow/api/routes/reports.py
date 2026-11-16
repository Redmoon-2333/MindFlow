"""API routes for daily, weekly and period reports.

Endpoints:
  - GET  /reports/daily (get or generate a daily report)
  - GET  /reports/weekly (weekly summary with 7-day trend)
  - POST /reports/period-review (facts + optional AI narration for 3/7/14 days)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query, Request  # noqa: B008
from loguru import logger

from mindflow.api.deps import get_period_review_service, get_report_service
from mindflow.api.errors import _not_found
from mindflow.api.schemas import (
    DailyReportResponse,
    PeriodReviewRequest,
    PeriodReviewResponse,
    WeeklyReportResponse,
)
from mindflow.services.period_review_service import PeriodReviewService
from mindflow.services.report_service import ReportService
from mindflow.time_utils import business_today

router = APIRouter(tags=["reports"])


@router.get("/reports/daily", response_model=DailyReportResponse)
async def get_daily_report(
    request: Request,
    report_date: date | None = Query(  # noqa: B008
        None, alias="date", description="Report date (YYYY-MM-DD, default today)"
    ),
    report_svc: ReportService = Depends(get_report_service),  # noqa: B008
) -> dict[str, Any]:
    """Return a daily report for the given date (generates if missing)."""
    settings = getattr(request.app.state, "settings", None)
    target = report_date or business_today(getattr(settings, "timezone", "local"))

    report = await report_svc.generate_daily_report(1, target)
    if not report:
        raise _not_found(f"日期 {target.isoformat()} 的报告")

    logger.debug("Daily report returned for {}", target)
    return report


@router.get("/reports/weekly", response_model=WeeklyReportResponse)
async def get_weekly_report(
    request: Request,
    week_start: date | None = Query(  # noqa: B008
        None, alias="week_start", description="Week start (YYYY-MM-DD, ISO week start Monday)"
    ),
    report_svc: ReportService = Depends(get_report_service),  # noqa: B008
) -> dict[str, Any]:
    """Return a weekly report with 7-day trend and week-over-week comparison.

    If *week_start* is omitted, the current ISO week is used.  Expected
    empty/future weeks are typed 200 responses carrying an explicit
    ``data_state`` — never a 404.
    """
    if week_start:
        start = week_start
    else:
        settings = getattr(request.app.state, "settings", None)
        today = business_today(getattr(settings, "timezone", "local"))
        # ISO week start (Monday)
        start = today - timedelta(days=today.weekday())

    return await report_svc.weekly_report(1, start)


@router.post("/reports/period-review", response_model=PeriodReviewResponse)
async def post_period_review(
    payload: PeriodReviewRequest,
    period_svc: PeriodReviewService = Depends(get_period_review_service),  # noqa: B008
) -> PeriodReviewResponse:
    """Summarise the last 3 / 7 / 14 local days.

    The facts come from re-scoring the stored 5-minute windows; the narrative is
    AI-written only when asked for *and* the data supports it, and falls back to
    the local template otherwise (``narrative.fallback_reason`` says why).

    ``user_id`` is fixed to 1, like the daily and weekly routes.
    """
    result = await period_svc.review(1, payload.days, payload.use_ai)
    return PeriodReviewResponse(
        digest=result.digest,
        llm_payload=result.llm_payload,
        ai_requested=result.ai_requested,
        narrative=result.narrative,
        generated_at=result.generated_at,
    )
