"""Collector recovery state and backoff — focused unit tests.

Covers the exact retry sequence (5/15/30/60 seconds, capped at 60),
UTC ``next_retry_at`` computation, and error retention/reset semantics
of :class:`mindflow.services.collector_recovery.RecoveryState`.

All times are injected fixed-UTC datetimes — no wall-clock dependence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mindflow.services.collector_recovery import (
    MAX_RETRY_DELAY_SECONDS,
    RETRY_DELAY_SECONDS,
    RecoveryState,
    retry_delay_seconds,
)

# Fixed UTC clock — shared by every test so transitions are reproducible.
_NOW = datetime(2026, 8, 5, 12, 0, 0, tzinfo=UTC)

# ═══════════════════════════════════════════════════════════════════════════════
# Backoff schedule
# ═══════════════════════════════════════════════════════════════════════════════


def test_retry_delay_sequence_attempts_1_to_6() -> None:
    """Attempts 1..6 map to 5/15/30/60/60/60 seconds (capped at 60)."""
    assert [retry_delay_seconds(i) for i in range(1, 7)] == [5.0, 15.0, 30.0, 60.0, 60.0, 60.0]


def test_retry_delay_schedule_is_exact_constant() -> None:
    """The published schedule is exactly 5/15/30/60, capped at 60."""
    assert RETRY_DELAY_SECONDS == (5.0, 15.0, 30.0, 60.0)
    assert MAX_RETRY_DELAY_SECONDS == 60.0


def test_retry_delay_clamps_below_first_attempt() -> None:
    """Attempts < 1 (0, negative) clamp to the first delay of 5s."""
    assert retry_delay_seconds(0) == 5.0
    assert retry_delay_seconds(-3) == 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# record_failure transitions
# ═══════════════════════════════════════════════════════════════════════════════


def test_record_failure_delay_sequence_via_state() -> None:
    """Consecutive failures produce the exact 5/15/30/60/60/60 schedule."""
    state = RecoveryState()
    delays = [state.record_failure(_NOW, f"error-{i}") for i in range(1, 7)]
    assert delays == [5.0, 15.0, 30.0, 60.0, 60.0, 60.0]
    assert state.recovery_attempts == 6


def test_next_retry_at_is_utc_now_plus_delay() -> None:
    """First failure: next_retry_at == injected now + 5s, tzinfo == UTC."""
    state = RecoveryState()
    delay = state.record_failure(_NOW, "boom")
    assert delay == 5.0
    assert state.next_retry_at == _NOW + timedelta(seconds=5.0)
    assert state.next_retry_at is not None
    assert state.next_retry_at.tzinfo is UTC
    assert state.next_retry_at.utcoffset() == timedelta(0)


def test_next_retry_at_grows_then_caps_at_60() -> None:
    """next_retry_at walks 5/15/30/60s ahead of now, never more than 60s."""
    state = RecoveryState()
    for attempt in range(1, 7):
        state.record_failure(_NOW, "boom")
        offset = state.next_retry_at - _NOW
        assert offset == timedelta(seconds=retry_delay_seconds(attempt))
        assert offset <= timedelta(seconds=60.0)


def test_last_error_retained_and_replaced_on_subsequent_failures() -> None:
    """Later failures replace last_error; earlier errors are not kept."""
    state = RecoveryState()
    state.record_failure(_NOW, "first error")
    assert state.last_error == "first error"
    state.record_failure(_NOW, "second error")
    assert state.last_error == "second error"
    assert state.recovery_attempts == 2


# ═══════════════════════════════════════════════════════════════════════════════
# record_success reset semantics
# ═══════════════════════════════════════════════════════════════════════════════


def test_fresh_state_defaults() -> None:
    """A new RecoveryState is idle: no retry time, no error, zero attempts."""
    state = RecoveryState()
    assert state.recovery_attempts == 0
    assert state.next_retry_at is None
    assert state.last_error is None


def test_record_success_resets_all_fields() -> None:
    """After success, attempts/error/next_retry_at all return to idle."""
    state = RecoveryState()
    state.record_failure(_NOW, "boom")
    assert state.next_retry_at is not None
    assert state.last_error is not None

    state.record_success()

    assert state.recovery_attempts == 0
    assert state.next_retry_at is None
    assert state.last_error is None


def test_recovery_restarts_schedule_after_success() -> None:
    """A success resets the counter, so the next failure starts at 5s again."""
    state = RecoveryState()
    assert state.record_failure(_NOW, "one") == 5.0
    assert state.record_failure(_NOW, "two") == 15.0
    state.record_success()
    assert state.recovery_attempts == 0
    assert state.record_failure(_NOW, "three") == 5.0
    assert state.recovery_attempts == 1
