from datetime import UTC, date, datetime

from mindflow.time_utils import (
    business_day_bounds_utc,
    business_today,
    resolve_timezone,
    utc_today,
)


def test_utc_today_matches_utc_calendar_date() -> None:
    assert utc_today() == datetime.now(UTC).date()


def test_business_today_uses_configured_timezone() -> None:
    now_utc = datetime(2026, 7, 16, 16, 30, tzinfo=UTC)

    assert business_today("Asia/Shanghai", now_utc=now_utc) == date(2026, 7, 17)


def test_business_day_bounds_are_stored_as_utc() -> None:
    start, end = business_day_bounds_utc(date(2026, 7, 17), "Asia/Shanghai")

    assert start == datetime(2026, 7, 16, 16, 0, tzinfo=UTC)
    assert end == datetime(2026, 7, 17, 15, 59, 59, 999999, tzinfo=UTC)


def test_business_day_bounds_follow_dst_transition() -> None:
    start, end = business_day_bounds_utc(date(2026, 3, 8), "America/New_York")

    assert start == datetime(2026, 3, 8, 5, 0, tzinfo=UTC)
    assert end == datetime(2026, 3, 9, 3, 59, 59, 999999, tzinfo=UTC)


def test_resolve_timezone_local_returns_aware_timezone() -> None:
    assert resolve_timezone("local").utcoffset(datetime.now()) is not None
