"""Activity event query endpoints — /api/v1/activities.

Provides:
  - GET /activities: Paginated activity events with optional date filtering
  - GET /activities/current: Most recent activity snapshot

All timestamps are returned as ISO8601 UTC strings.
Dates are optional and default to today if omitted.
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from mindflow.api.deps import get_activity_repo
from mindflow.api.errors import ProblemDetail
from mindflow.domain.events import ActivityEvent
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.time_utils import TimezoneLike, business_day_bounds_utc, business_today

router = APIRouter(tags=["activities"])


def _activity_date_bounds(
    start_date: str | None,
    end_date: str | None,
    *,
    timezone: TimezoneLike,
) -> tuple[datetime, datetime]:
    """Return inclusive UTC bounds for local business-date parameters."""
    today = business_today(timezone)
    start_day = date.fromisoformat(start_date) if start_date else today - timedelta(days=7)
    end_day = date.fromisoformat(end_date) if end_date else today
    if start_day > end_day:
        raise ValueError("start date is after end date")
    start, _ = business_day_bounds_utc(start_day, timezone)
    _, end = business_day_bounds_utc(end_day, timezone)
    return start, end


@router.get("/activities")
async def list_activities(
    request: Request,
    activity_repo: SQLAlchemyActivityRepository = Depends(get_activity_repo),  # noqa: B008
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),  # noqa: B008
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),  # noqa: B008
    cursor: str | None = Query(  # noqa: B008
        default=None,
        description="Opaque cursor returned by the previous response",
    ),
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
    settings = getattr(request.app.state, "settings", None)
    timezone: TimezoneLike = getattr(settings, "timezone", "local")
    try:
        start, end = _activity_date_bounds(
            start_date, end_date, timezone=timezone
        )
    except (ValueError, TypeError) as exc:
        detail = (
            "开始日期不能晚于结束日期"
            if str(exc) == "start date is after end date"
            else "日期格式必须为 YYYY-MM-DD"
        )
        raise ProblemDetail(
            type_slug="validation-error",
            title="Validation Error",
            status=422,
            detail=detail,
        ) from None

    decoded_cursor = _decode_cursor(cursor) if cursor is not None else None
    offset = None if decoded_cursor is not None else (page - 1) * page_size
    total = (
        None
        if decoded_cursor is not None
        else await activity_repo.count_range(user_id=1, start=start, end=end)
    )

    events = await activity_repo.query_range(
        user_id=1,
        start=start,
        end=end,
        limit=page_size + 1,
        offset=offset,
        descending=True,
        cursor=decoded_cursor,
    )

    has_more = len(events) > page_size
    page_events = events[:page_size]
    next_cursor = (
        _encode_cursor(page_events[-1])
        if has_more and page_events
        else None
    )

    return {
        "items": [_event_to_dict(e) for e in page_events],
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_more": has_more,
        "next_cursor": next_cursor,
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



def _encode_cursor(event: ActivityEvent) -> str:
    payload = json.dumps(
        [event.timestamp_utc.isoformat(), event.id],
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(value: str) -> tuple[str, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded).decode())
        if (
            not isinstance(decoded, list)
            or len(decoded) != 2
            or not all(isinstance(item, str) and item for item in decoded)
        ):
            raise ValueError
        datetime.fromisoformat(decoded[0])
        return decoded[0], decoded[1]
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError, binascii.Error):
        raise ProblemDetail(
            type_slug="validation-error",
            title="Validation Error",
            status=422,
            detail="分页游标无效",
        ) from None

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
