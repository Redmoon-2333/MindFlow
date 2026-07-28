from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, tzinfo
from zoneinfo import ZoneInfo

TimezoneLike = str | tzinfo


def resolve_timezone(value: TimezoneLike = "local") -> tzinfo:
    """Resolve ``local`` or an IANA name to a timezone object."""
    if not isinstance(value, str):
        return value
    if value == "local":
        return datetime.now().astimezone().tzinfo or UTC
    return ZoneInfo(value)


def utc_today() -> date:
    return datetime.now(UTC).date()


def business_today(
    timezone: TimezoneLike = "local",
    *,
    now_utc: datetime | None = None,
) -> date:
    """Return the calendar date in the configured business timezone."""
    now = now_utc or datetime.now(UTC)
    if now.tzinfo is None:
        msg = "now_utc must be timezone-aware"
        raise ValueError(msg)
    return now.astimezone(resolve_timezone(timezone)).date()


def business_day_bounds_utc(
    target_date: date, timezone: TimezoneLike = "local"
) -> tuple[datetime, datetime]:
    """Return inclusive UTC bounds for one local business calendar day."""
    local_timezone = resolve_timezone(timezone)
    start_local = datetime.combine(target_date, time.min, tzinfo=local_timezone)
    next_day_local = datetime.combine(
        target_date + timedelta(days=1), time.min, tzinfo=local_timezone
    )
    return (
        start_local.astimezone(UTC),
        next_day_local.astimezone(UTC) - timedelta(microseconds=1),
    )
