"""Activity event query endpoints — /api/v1/activities.

Provides:
  - GET /activities: Paginated activity events with optional date filtering
  - GET /activities/current: Most recent activity snapshot

All timestamps are returned as ISO8601 UTC strings.
Dates are optional and default to today if omitted.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query

from mindflow.api.deps import get_activity_repo
from mindflow.api.errors import ProblemDetail
from mindflow.domain.events import ActivityEvent
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)

router = APIRouter(tags=["activities"])


@router.get("/activities")
async def list_activities(
    activity_repo: SQLAlchemyActivityRepository = Depends(get_activity_repo),  # noqa: B008
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),  # noqa: B008
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),  # noqa: B008
    start_date: str | None = Query(  # noqa: B008
        default=None,
        description="Start date (YYYY-MM-DD, inclusive). Defaults to 7 days ago.",
    ),
    end_date: str | None = Query(  # noqa: B008
        default=None,
        description="End date (YYYY-MM-DD, inclusive). Defaults to today.",
    ),
) -> dict[str, Any]:
    """Return paginated activity events within a date range.

    Results are ordered by timestamp descending (most recent first).
    """
    today = date.today()

    try:
        start = (
            datetime.fromisoformat(start_date).replace(tzinfo=UTC)
            if start_date
            else datetime(today.year, today.month, today.day, tzinfo=UTC) - timedelta(days=7)
        )
        end = (
            datetime.fromisoformat(end_date).replace(tzinfo=UTC)
            if end_date
            else datetime(today.year, today.month, today.day, tzinfo=UTC) + timedelta(days=1)
        )
    except (ValueError, TypeError):
        raise ProblemDetail(
            type_slug="validation-error",
            title="Validation Error",
            status=422,
            detail="日期格式无效，请使用 YYYY-MM-DD 格式",
        ) from None

    # If same day, extend end to end of that day
    if start.date() == end.date():
        end = end + timedelta(days=1)

    if start >= end:
        raise ProblemDetail(
            type_slug="validation-error",
            title="Validation Error",
            status=422,
            detail="开始日期必须早于结束日期",
        )

    events = await activity_repo.query_range(user_id=1, start=start, end=end)

    # Reverse to show most recent first, then paginate
    events.reverse()
    offset = (page - 1) * page_size
    page_events = events[offset : offset + page_size]

    return {
        "items": [_event_to_dict(e) for e in page_events],
        "page": page,
        "page_size": page_size,
        "total": len(events),
        "has_more": (offset + page_size) < len(events),
    }


@router.get("/activities/current")
async def get_current_activity(
    activity_repo: SQLAlchemyActivityRepository = Depends(get_activity_repo),  # noqa: B008
) -> dict[str, Any]:
    """Return the most recent activity snapshot, or a 404 if none exist."""
    event = await activity_repo.last_event(user_id=1)

    if event is None:
        raise ProblemDetail(
            type_slug="not-found",
            title="Not Found",
            status=404,
            detail="暂无活动记录",
        )

    return _event_to_dict(event)


def _event_to_dict(event: ActivityEvent) -> dict[str, Any]:
    """Convert an ActivityEvent to a JSON-safe dict for API responses."""
    return {
        "id": event.id,
        "user_id": event.user_id,
        "timestamp": event.timestamp_utc.isoformat(),
        "duration_s": event.duration_s,
        "event_type": event.event_type,
        "data": {
            "app_name": event.data.app_name,
            "window_title": event.data.window_title,
            "process_name": event.data.process_name,
            "is_idle": event.data.is_idle,
        },
    }
