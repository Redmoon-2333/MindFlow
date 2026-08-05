"""CollectorIntervalLifecycle open/close and sleep-recovery helpers tests.

Covers the lifecycle seam directly with a recording fake (asserting the
exact arguments the lifecycle issues to the repository) plus one real
SQLite round-trip proving the repository's guarded close keeps the sleep
interval idempotent across repeated ``close_sleep`` calls.

  - ``open()`` / ``close()`` lock the unchanged run-interval behaviour.
  - ``open_sleep()`` opens a ``sleep=True`` interval with a fixed backoff
    reason and no failure/manual-stop flags.
  - ``close_sleep()`` closes the same interval keeping ``sleep=True``,
    recording ``manual_stop`` only when actually requested.
  - Unwired (``repository is None``) — the sleep helpers are no-ops.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mindflow.infrastructure.repositories.collector_intervals import (
    CollectorIntervalsRepository,
)
from mindflow.infrastructure.schema import collector_intervals
from mindflow.ports import CollectorIntervalRecord
from mindflow.services.collector_interval_lifecycle import (
    CollectorIntervalLifecycle,
)


class RecordingIntervalsPort:
    """Minimal intervals port that records open/close calls verbatim."""

    def __init__(self) -> None:
        self.opens: list[CollectorIntervalRecord] = []
        self.closes: list[CollectorIntervalRecord] = []

    async def open(
        self,
        user_id: int,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord:
        record = CollectorIntervalRecord(
            id=f"open-{len(self.opens) + 1}",
            user_id=user_id,
            started_at=(now or datetime.now(UTC)).isoformat(),
            ended_at=None,
            reason=reason,
            manual_stop=manual_stop,
            failure=failure,
            sleep=sleep,
            last_error=None,
        )
        self.opens.append(record)
        return record

    async def close(
        self,
        interval_id: str,
        *,
        reason: str | None = None,
        manual_stop: bool = False,
        failure: bool = False,
        sleep: bool = False,
        last_error: str | None = None,
        now: datetime | None = None,
    ) -> CollectorIntervalRecord | None:
        record = CollectorIntervalRecord(
            id=interval_id,
            user_id=1,
            started_at="",
            ended_at=(now or datetime.now(UTC)).isoformat(),
            reason=reason,
            manual_stop=manual_stop,
            failure=failure,
            sleep=sleep,
            last_error=last_error,
        )
        self.closes.append(record)
        return record


async def _create_table(engine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(collector_intervals.metadata.create_all)


# ── Unchanged run-interval behaviour (regression guard) ───────────────


async def test_open_and_close_run_interval_unchanged() -> None:
    port = RecordingIntervalsPort()
    lifecycle = CollectorIntervalLifecycle(port, user_id=1)

    interval_id = await lifecycle.open()
    await lifecycle.close(
        interval_id,
        degraded=True,
        stop_requested=True,
        last_error="boom",
    )

    assert interval_id == "open-1"
    assert port.opens[0].reason == "collector run started"
    assert port.opens[0].sleep is False
    # Degraded classification wins over a requested stop.
    closed = port.closes[0]
    assert closed.failure is True
    assert closed.manual_stop is False
    assert closed.last_error == "boom"


# ── open_sleep ────────────────────────────────────────────────────────


async def test_open_sleep_records_backoff_with_sleep_and_no_failure_manual_stop() -> None:
    port = RecordingIntervalsPort()
    lifecycle = CollectorIntervalLifecycle(port, user_id=1)

    interval_id = await lifecycle.open_sleep()

    assert interval_id == "open-1"
    assert len(port.opens) == 1
    opened = port.opens[0]
    assert opened.sleep is True
    assert opened.failure is False
    assert opened.manual_stop is False
    assert opened.reason == "collector backoff (system sleep)"


# ── close_sleep ───────────────────────────────────────────────────────


async def test_close_sleep_without_stop_request_keeps_sleep_and_marks_external() -> None:
    port = RecordingIntervalsPort()
    lifecycle = CollectorIntervalLifecycle(port, user_id=1)
    interval_id = await lifecycle.open_sleep()

    await lifecycle.close_sleep(interval_id, stop_requested=False)

    assert len(port.closes) == 1
    closed = port.closes[0]
    assert closed.id == interval_id
    assert closed.sleep is True
    assert closed.failure is False
    assert closed.manual_stop is False
    assert closed.last_error is None
    assert closed.reason == "system sleep"


async def test_close_sleep_with_stop_requested_marks_manual_stop_and_keeps_sleep() -> None:
    port = RecordingIntervalsPort()
    lifecycle = CollectorIntervalLifecycle(port, user_id=1)
    interval_id = await lifecycle.open_sleep()

    await lifecycle.close_sleep(interval_id, stop_requested=True)

    assert len(port.closes) == 1
    closed = port.closes[0]
    assert closed.manual_stop is True
    assert closed.sleep is True
    assert closed.failure is False
    assert closed.reason == "manual stop"


async def test_sleep_helpers_unwired_are_noops() -> None:
    lifecycle = CollectorIntervalLifecycle(None, user_id=1)

    assert await lifecycle.open_sleep() is None
    # Must not raise even though no interval was ever opened.
    await lifecycle.close_sleep(None, stop_requested=False)


# ── Real-repository round-trip + idempotency ──────────────────────────


async def test_open_sleep_close_sleep_persist_and_second_close_is_idempotent(
    engine, session_factory,
) -> None:
    await _create_table(engine)
    repository = CollectorIntervalsRepository(session_factory)
    lifecycle = CollectorIntervalLifecycle(repository, user_id=1)

    interval_id = await lifecycle.open_sleep()
    await lifecycle.close_sleep(interval_id, stop_requested=False)
    # A second close must not rewrite the terminal facts.
    await lifecycle.close_sleep(interval_id, stop_requested=True)

    listed = await repository.list_by_user(1)
    assert len(listed) == 1
    record = listed[0]
    assert record.ended_at is not None
    assert record.sleep is True
    assert record.failure is False
    assert record.manual_stop is False
    assert record.reason == "system sleep"
