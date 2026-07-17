"""API route for data export — /api/v1/export.

Returns a downloadable CSV or JSON file containing events, focus sessions,
and daily reports for the requested date range.

Design:
  - StreamingResponse with Content-Disposition attachment for browser download.
  - Default range is the last 30 days (matching event retention period).
  - Ranges exceeding 90 days are rejected (422) to prevent memory abuse.
  - Empty ranges return an empty archive (not an error).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query  # noqa: B008
from fastapi.responses import StreamingResponse

from mindflow.api.deps import (
    get_activity_repo,
    get_focus_repo,
    get_report_repo,
)
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.infrastructure.repositories.report import (
    SQLAlchemyDailyReportRepository,
)
from mindflow.services.export_service import ExportService

router = APIRouter(tags=["export"])

_MAX_EXPORT_DAYS: int = 90


@router.get("/export")
async def export_data(
    fmt: str = Query("csv", description="Export format: csv or json"),  # noqa: B008
    start: str | None = Query(None, description="Start date (ISO8601, default 30 days ago)"),  # noqa: B008
    end: str | None = Query(None, description="End date (ISO8601, default now)"),  # noqa: B008
    activity_repo: SQLAlchemyActivityRepository = Depends(get_activity_repo),  # noqa: B008
    focus_repo: SQLAlchemyFocusSessionRepository = Depends(get_focus_repo),  # noqa: B008
    report_repo: SQLAlchemyDailyReportRepository = Depends(get_report_repo),  # noqa: B008
) -> StreamingResponse:
    """Export activity data as CSV or JSON.

    Returns a streaming file download with Content-Disposition attachment.

    Args:
        fmt: Output format — ``"csv"`` or ``"json"``.
        start: Start of the date range (ISO8601, inclusive).
            Defaults to 30 days ago.
        end: End of the date range (ISO8601, inclusive).
            Defaults to now.

    Returns:
        A ``StreamingResponse`` with the exported data as an attachment.

    Raises:
        HTTPException 422: If the date range exceeds 90 days or the format
            is invalid.
    """
    now = datetime.now(UTC)

    # ── Parse date range ─────────────────────────────────────────────
    try:
        start_dt = (
            datetime.fromisoformat(start).replace(tzinfo=UTC)
            if start
            else now - timedelta(days=30)
        )
        end_dt = (
            datetime.fromisoformat(end).replace(tzinfo=UTC)
            if end
            else now
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"无效的日期格式: {exc}",
        ) from exc

    # ── Validate date range ──────────────────────────────────────────
    if end_dt < start_dt:
        raise HTTPException(
            status_code=422,
            detail="结束日期不能早于开始日期",
        )

    range_days = (end_dt - start_dt).days
    if range_days > _MAX_EXPORT_DAYS:
        raise HTTPException(
            status_code=422,
            detail=f"导出范围不能超过 {_MAX_EXPORT_DAYS} 天（当前: {range_days} 天）",
        )

    # ── Validate format ──────────────────────────────────────────────
    try:
        resolved_fmt = ExportService.validate_format(fmt)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # ── Generate export ──────────────────────────────────────────────
    service = ExportService(
        activity_repo=activity_repo,
        focus_repo=focus_repo,
        report_repo=report_repo,
    )
    result = await service.export_events(start_dt, end_dt, fmt=resolved_fmt)

    # Stream the content as a file download
    async def _stream() -> AsyncGenerator[bytes, None]:
        yield result.content

    # Build Content-Disposition filename (sanitised for URL safety)
    disposition = f'attachment; filename="{result.filename}"'

    return StreamingResponse(
        content=_stream(),
        media_type=result.media_type,
        headers={"Content-Disposition": disposition},
    )
