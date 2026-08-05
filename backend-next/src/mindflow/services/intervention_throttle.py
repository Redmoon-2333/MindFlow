"""Intervention throttle — rate-limiting state machine (Wave 7, C3).

Rules (from 03-requirements.md §C3):
  1. Daily cap: ≤3 interventions per user per day.
  2. Cooldown: ≥2 hours since the last intervention (any type).
  3. Type cap: ≤2 interventions of the same type per day.
  4. Fatigue detection: if the 7-day ignore rate exceeds 60%, reduce
     the daily cap to 1 for the current day (resets at midnight).
  5. All counts reset at the start of each calendar day (UTC).

The throttle is the ONLY gate for automated interventions. Manual
trigger (POST /api/v1/intervention/trigger) bypasses throttle but
still counts toward rate limits for future automated checks.

Design:
  - ``can_intervene()`` returns a ``ThrottleDecision`` (no exceptions).
  - ``clock`` is injectable for deterministic testing.
  - State is STATELESS — all state lives in the DB (intervention_logs).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import StrEnum

from mindflow.infrastructure.repositories.intervention import (
    Clock,
    InterventionLogRepository,
    UTCCLock,
)


class ThrottleReason(StrEnum):
    """Reason code for a denied intervention.

    OK — intervention allowed.
    DAILY_CAP — daily total limit reached.
    COOLDOWN — too soon since last intervention.
    TYPE_CAP — same-type daily limit reached.
    FATIGUE — high ignore rate triggered stricter limit.
    """

    OK = "ok"
    DAILY_CAP = "daily_cap"
    COOLDOWN = "cooldown"
    TYPE_CAP = "type_cap"
    FATIGUE = "fatigue"
    ANNOYING = "annoying"


# ── Configuration defaults ──────────────────────────────────────────────

_DEFAULT_DAILY_LIMIT: int = 3
_DEFAULT_TYPE_LIMIT: int = 2
_DEFAULT_COOLDOWN_H: float = 2.0
_DEFAULT_IGNORE_RATE_THRESHOLD: float = 0.6
_DEFAULT_FATIGUE_DAILY_LIMIT: int = 1
_DEFAULT_ANNOYING_THRESHOLD: int = 3


class ThrottleDecision:
    """Result of a throttle check.

    Attributes:
        allowed: True if the intervention may proceed.
        reason: ``ThrottleReason`` enum — ``OK`` when allowed, otherwise
            the specific blocking reason.
        detail: Human-readable explanation (Chinese, for debugging/logging).
    """

    def __init__(self, reason: ThrottleReason, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail

    @property
    def allowed(self) -> bool:
        return self.reason == ThrottleReason.OK

    def __repr__(self) -> str:
        return f"ThrottleDecision({self.reason}, allowed={self.allowed})"


# ── Throttle implementation ────────────────────────────────────────────


class InterventionThrottle:
    """Rate-limiting throttle for automated interventions.

    All state is read from the DB via ``InterventionLogRepository`` —
    no in-memory counters.  This guarantees correctness across service
    restarts and concurrent access.

    Args:
        repo: Intervention log repository.
        clock: Injectable clock (defaults to UTCClock for production).
        daily_limit: Max interventions per day (default 3).
        type_limit: Max interventions of same type per day (default 2).
        cooldown_h: Min hours since last intervention (default 2).
        ignore_rate_threshold: Ignore rate above which fatigue kicks in
            (default 0.6 = 60%).
        fatigue_daily_limit: Reduced daily cap when fatigued (default 1).
    """

    def __init__(
        self,
        repo: InterventionLogRepository,
        clock: Clock | None = None,
        daily_limit: int = _DEFAULT_DAILY_LIMIT,
        type_limit: int = _DEFAULT_TYPE_LIMIT,
        cooldown_h: float = _DEFAULT_COOLDOWN_H,
        ignore_rate_threshold: float = _DEFAULT_IGNORE_RATE_THRESHOLD,
        fatigue_daily_limit: int = _DEFAULT_FATIGUE_DAILY_LIMIT,
        annoying_threshold: int = _DEFAULT_ANNOYING_THRESHOLD,
    ) -> None:
        self._repo = repo
        self._clock = clock or UTCCLock()
        self._daily_limit = daily_limit
        self._type_limit = type_limit
        self._cooldown_h = cooldown_h
        self._ignore_rate_threshold = ignore_rate_threshold
        self._fatigue_daily_limit = fatigue_daily_limit
        self._annoying_threshold = annoying_threshold

    async def can_intervene(
        self,
        user_id: int,
        intervention_type: str,
    ) -> ThrottleDecision:
        """Check whether an intervention of *type* is allowed for *user_id*.

        Rule evaluation order (short-circuit: first rejection wins):
          1. Daily cap (adjusted for fatigue)
          2. Cooldown check
          3. Type cap
          4. Fatigue check — if ignore rate is high, reduce daily cap

        P2-1 optimisation: all five throttle metrics are fetched in a single
        ``get_throttle_stats`` DB round-trip; behaviour is unchanged.

        Returns:
            ``ThrottleDecision`` — ``allowed=True`` with ``reason=OK`` if
            the intervention may proceed, otherwise the specific rejection.
        """
        now = self._clock.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_7d = now - timedelta(days=7)
        cooldown_lower_bound = now - timedelta(hours=self._cooldown_h * 2)

        # ── Single-query aggregate (P2-1) ──────────────────────────────
        stats = await self._repo.get_throttle_stats(
            user_id,
            intervention_type,
            now=now,
            today_start=today_start,
            cutoff_7d=cutoff_7d,
            cooldown_lower_bound=cooldown_lower_bound,
        )

        # ── 1. Fatigue check (affects daily limit) ─────────────────────
        effective_daily_limit = (
            self._fatigue_daily_limit
            if stats.ignore_rate > self._ignore_rate_threshold
            else self._daily_limit
        )

        # ── 2. Daily cap ──────────────────────────────────────────────
        if stats.today_count >= effective_daily_limit:
            return ThrottleDecision(
                reason=ThrottleReason.DAILY_CAP,
                detail=f"今日干预已达上限 ({stats.today_count}/{effective_daily_limit})",
            )

        # ── 3. Cooldown ───────────────────────────────────────────────
        # stats.last_triggered_at is the most recent triggered_at within
        # [cooldown_lower_bound, now] — same as old query_range semantic.
        if stats.last_triggered_at is not None:
            try:
                last_ts = datetime.fromisoformat(stats.last_triggered_at)
                elapsed_h = (now - last_ts).total_seconds() / 3600.0
                if elapsed_h < self._cooldown_h:
                    remaining_m = round((self._cooldown_h - elapsed_h) * 60)
                    return ThrottleDecision(
                        reason=ThrottleReason.COOLDOWN,
                        detail=f"距上次干预仅 {elapsed_h:.1f}h，需等待 {remaining_m} 分钟",
                    )
            except (ValueError, TypeError):
                pass  # Parse error — allow through (defensive)

        # ── 4. Type cap (adjusted for annoying feedback) ────────────────
        # If 3+ "annoying" ratings in 7 days for this type, reduce daily limit to 1
        effective_type_limit = (
            1 if stats.annoying_count_by_type >= self._annoying_threshold
            else self._type_limit
        )

        if stats.today_count_by_type >= effective_type_limit:
            reason = (
                ThrottleReason.ANNOYING
                if stats.annoying_count_by_type >= self._annoying_threshold
                else ThrottleReason.TYPE_CAP
            )
            return ThrottleDecision(
                reason=reason,
                detail=(
                    f"今日 {intervention_type} 类型干预已达上限 "
                    f"({stats.today_count_by_type}/{effective_type_limit})"
                    + (
                        f"（近7日 {stats.annoying_count_by_type} 条负面反馈）"
                        if stats.annoying_count_by_type >= self._annoying_threshold
                        else ""
                    )
                ),
            )

        return ThrottleDecision(
            reason=ThrottleReason.OK,
            detail="节流检查通过",
        )

    # ── Atomic daily slot reservation (Wave 18) ───────────────────────

    async def reserve_slot(
        self,
        user_id: int,
        intervention_type: str,
        *,
        slot_index: int | None = None,
        date_str: str | None = None,
    ) -> int | None:
        """Atomically reserve a daily intervention slot.

        Uses ``INSERT … ON CONFLICT DO NOTHING`` via
        ``InterventionLogRepository.try_reserve_daily_slot()`` — the same
        atomic pattern as ``BudgetReservationRepository.try_reserve()``.

        Callers should first check ``can_intervene()`` to determine the
        current count, then call ``reserve_slot()`` with the next available
        slot index.  If the reservation succeeds, the slot is claimed and
        the caller may proceed to log the intervention.  If it fails,
        another concurrent caller beat us to the slot.

        Args:
            user_id: User identifier.
            intervention_type: Type of intervention for audit tracking.
            slot_index: Specific slot to reserve (1-based).  If None,
                auto-computes from the current daily count + 1.
            date_str: Optional date override (ISO8601 date).  When provided,
                it is threaded through to the repository so the reservation
                can later be released with the exact same date.

        Returns:
            The reserved ``slot_index`` (1-based) on success, or ``None``
            if the slot was already taken.
        """
        if slot_index is None:
            now = self._clock.now()
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_7d = now - timedelta(days=7)
            cooldown_lower_bound = now - timedelta(hours=self._cooldown_h * 2)
            stats = await self._repo.get_throttle_stats(
                user_id,
                intervention_type,
                now=now,
                today_start=today_start,
                cutoff_7d=cutoff_7d,
                cooldown_lower_bound=cooldown_lower_bound,
            )
            slot_index = stats.today_count + 1

        reserved = await self._repo.try_reserve_daily_slot(
            user_id, slot_index, intervention_type, date_str=date_str,
        )
        return slot_index if reserved else None
