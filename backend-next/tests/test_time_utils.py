from datetime import UTC, datetime

from mindflow.time_utils import utc_today


def test_utc_today_matches_utc_calendar_date() -> None:
    assert utc_today() == datetime.now(UTC).date()
