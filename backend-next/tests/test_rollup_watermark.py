"""Phase 4.2 regressions: incremental telemetry rollup watermark.

The rollup used to recompute the whole trailing window on every tick.  It now
keeps a watermark of the last *successful* rollup and re-covers only the missing
buckets plus one bucket of overlap.  The invariants that must hold:

  * a fresh service behaves exactly like before (full trailing window);
  * a successful rollup advances the watermark and shrinks the next range;
  * one bucket of overlap is always re-covered (late events are not lost);
  * the trailing window is still the upper bound (a stale watermark cannot make
    the service recompute hours of history);
  * a failed rollup does NOT advance the watermark, so the next attempt
    re-covers the whole missing range instead of silently skipping it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mindflow.services.telemetry_service import (
    _RECENT_ROLLUP_WINDOW_HOURS,
    _ROLLUP_OVERLAP_MINUTES,
    TelemetryService,
)

_NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def _service(tmp_path: Path) -> tuple[TelemetryService, AsyncMock]:
    """A service whose rollup is captured instead of executed."""
    repository = AsyncMock()
    service = TelemetryService(
        repository=repository,
        preferences_repository=AsyncMock(),
        data_dir=tmp_path,
    )
    rollup = AsyncMock(return_value=0)
    service.rollup_feature_windows = rollup  # type: ignore[method-assign]
    return service, rollup


def _range(rollup: AsyncMock, call: int = -1) -> tuple[datetime, datetime]:
    start, end = rollup.await_args_list[call].args[:2]
    return start, end


async def test_first_rollup_covers_the_full_trailing_window(tmp_path: Path) -> None:
    service, rollup = _service(tmp_path)
    assert service.last_successful_rollup is None

    await service.rollup_recent(_NOW)

    start, end = _range(rollup)
    assert end == _NOW
    assert start == _NOW - timedelta(hours=_RECENT_ROLLUP_WINDOW_HOURS)
    assert service.last_successful_rollup == _NOW


async def test_second_rollup_only_covers_the_missing_buckets(
    tmp_path: Path,
) -> None:
    service, rollup = _service(tmp_path)
    await service.rollup_recent(_NOW)

    later = _NOW + timedelta(minutes=15)
    await service.rollup_recent(later)

    start, end = _range(rollup)
    assert end == later
    assert start == _NOW - timedelta(minutes=_ROLLUP_OVERLAP_MINUTES)
    assert service.last_successful_rollup == later


async def test_overlap_re_covers_the_newest_bucket(tmp_path: Path) -> None:
    """Late-arriving events for the previous bucket are still folded in."""
    service, rollup = _service(tmp_path)
    await service.rollup_recent(_NOW)
    await service.rollup_recent(_NOW + timedelta(minutes=15))

    start, _ = _range(rollup)
    assert start < _NOW, "the previous bucket must be re-covered by the overlap"


async def test_trailing_window_is_still_the_upper_bound(tmp_path: Path) -> None:
    """A long idle gap cannot make the service recompute more than the window."""
    service, rollup = _service(tmp_path)
    await service.rollup_recent(_NOW)

    much_later = _NOW + timedelta(days=2)
    await service.rollup_recent(much_later)

    start, end = _range(rollup)
    assert end == much_later
    assert start == much_later - timedelta(hours=_RECENT_ROLLUP_WINDOW_HOURS)


async def test_failed_rollup_does_not_advance_the_watermark(tmp_path: Path) -> None:
    service, rollup = _service(tmp_path)
    await service.rollup_recent(_NOW)
    assert service.last_successful_rollup == _NOW

    rollup.side_effect = RuntimeError("db down")
    later = _NOW + timedelta(minutes=15)
    with pytest.raises(RuntimeError):
        await service.rollup_recent(later)

    assert service.last_successful_rollup == _NOW, "a failure must not lose coverage"

    # The retry therefore re-covers everything the failure left behind.
    rollup.side_effect = None
    rollup.return_value = 0
    await service.rollup_recent(later)
    start, end = _range(rollup)
    assert end == later
    assert start == _NOW - timedelta(minutes=_ROLLUP_OVERLAP_MINUTES)


async def test_watermark_never_moves_backwards(tmp_path: Path) -> None:
    """Clock skew (a rollup asked for an earlier 'now') must not shrink coverage."""
    service, rollup = _service(tmp_path)
    await service.rollup_recent(_NOW)

    earlier = _NOW - timedelta(minutes=5)
    await service.rollup_recent(earlier)

    start, end = _range(rollup)
    assert end == earlier
    assert start <= end
    # The watermark stays at the furthest point already covered, so the next
    # rollup still starts from there rather than re-covering less history.
    assert service.last_successful_rollup == _NOW


async def test_rollup_is_delegated_to_the_idempotent_seam(tmp_path: Path) -> None:
    """The watermark only narrows the range; upsert/baseline logic is untouched."""
    service, rollup = _service(tmp_path)
    await service.rollup_recent(_NOW, user_id=7)

    assert rollup.await_args.kwargs["user_id"] == 7
