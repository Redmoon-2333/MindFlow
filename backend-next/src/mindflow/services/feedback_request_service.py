"""Proactive-learning feedback requests (contract §5).

The system asks for a label only when the answer carries information — the
decision itself lives in ``services/feedback_sampling.py`` (a pure function).
This module owns the *lifecycle* around that decision:

  * one in-flight request per user, so the UI never stacks prompts;
  * a cooldown and a per-day cap, so asking cannot become nagging;
  * a two-hour expiry, so a stale prompt cannot label behaviour the user no
    longer remembers;
  * the write into ``behavior_feature_windows.label``, the same column the V2
    training pipeline reads (weight ``WINDOW_LABEL_WEIGHT`` = 0.8, below
    explicit per-session feedback at 1.0).

Only windows whose ``label`` is still NULL are written. A user-answered label
must never be overwritten by a later prompt about an overlapping window, and
that rule is enforced in the UPDATE's WHERE clause rather than by a prior
read — a concurrent rollup cannot slip between the check and the write.

The store is deliberately in-memory: a pending prompt is worthless after a
restart (the window has moved on), and persisting it would mean storing a
question the user never sees. Losing the cooldown across a restart can at
worst produce one extra prompt.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final, Literal

from loguru import logger

from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.time_utils import TimezoneLike, resolve_timezone

#: A prompt older than this is no longer answerable: the user's memory of a
#: 15-minute window two hours ago is not a label, it is a guess.
REQUEST_TTL: Final = timedelta(hours=2)

#: Minimum gap between two prompts. Long enough that a prompt never interrupts
#: twice in one work block, short enough to collect a handful of labels a day.
MIN_ASK_INTERVAL: Final = timedelta(minutes=45)

#: Per-local-day cap. Six labels is roughly one every two waking hours — the
#: point where a prompt is still an occasional question rather than a tax.
MAX_ASKS_PER_DAY: Final = 6

#: How far back a request's window reaches. Matches the prediction service's
#: "recent behaviour" horizon and covers three 5-minute feature windows.
REQUEST_WINDOW: Final = timedelta(minutes=15)

#: Answer vocabulary. ``skip`` is a dismissal, not a label, and is never written.
LABEL_FOCUS: Final = "focus"
LABEL_DISTRACTED: Final = "distracted"
LABEL_SKIP: Final = "skip"

#: The two labels the training pipeline maps to classes
#: (``train/pipeline.py``: focus -> 1, distracted -> 0).
WRITABLE_LABELS: Final = frozenset({LABEL_FOCUS, LABEL_DISTRACTED})

#: Chinese trigger descriptions. The sampling module emits stable English
#: identifiers; the user reads these instead.
REASON_LABELS: Final[dict[str, str]] = {
    "model_uncertain": "模型不确定",
    "rule_ml_conflict": "规则与模型判断不一致",
    "panel_disagreement": "专家意见分歧",
    "task_context_mismatch": "任务上下文与行为不符",
    "data_drift": "近期行为相对训练数据发生漂移",
}

#: Default question. One sentence, no jargon, answerable from memory.
DEFAULT_PROMPT: Final = "刚才 15 分钟你在专注吗？"

AnswerStatus = Literal["recorded", "skipped", "expired"]

BroadcastFn = Callable[[dict[str, Any]], Awaitable[int]]

#: Clock the store reads when a caller does not inject an instant.
#: Injected once at construction so every lifecycle decision (expiry,
#: cooldown, daily cap) is measured against the *same* clock. A test that
#: injects a fixed ``now`` into ``maybe_create`` and lets the lifecycle
#: read the wall clock instead would see the two disagree mid-run.
ClockFn = Callable[[], datetime]


@dataclass
class FeedbackRequest:
    """One in-flight question about a specific time window."""

    request_id: str
    user_id: int
    window_start: datetime
    window_end: datetime
    probability: float | None
    reasons: list[str]
    prompt: str
    created_at: datetime

    def is_expired(self, now: datetime) -> bool:
        return now - self.created_at >= REQUEST_TTL

    def to_payload(self) -> dict[str, Any]:
        """The contract §5 body (``data`` in the WS frame, an item over HTTP)."""
        return {
            "request_id": self.request_id,
            "window_start": self.window_start.astimezone(UTC).isoformat(),
            "window_end": self.window_end.astimezone(UTC).isoformat(),
            "probability": self.probability,
            "reasons": [REASON_LABELS.get(r, r) for r in self.reasons],
            "prompt": self.prompt,
        }


@dataclass
class AnswerResult:
    """Outcome of an answer, mirroring the contract's response body."""

    request_id: str
    labeled_windows: int
    status: AnswerStatus


@dataclass
class _AskLedger:
    """Per-user ask history used by the cooldown and the daily cap."""

    last_asked_at: datetime | None = None
    #: Local business date of the last ask (``Settings.timezone``), not UTC:
    #: a cap of "six per day" must reset at the user's midnight.
    day: date | None = None
    count: int = 0


@dataclass
class FeedbackRequestStore:
    """In-memory pending-request store with cooldown and daily cap.

    Every mutable field is guarded by one ``asyncio.Lock``: the scheduler job
    and the HTTP routes both reach this object from the same event loop, and
    a check-then-create across an ``await`` would otherwise let two prompts
    through. The lock is held only for the bookkeeping, never across the
    prediction call or the database write.
    """

    telemetry_repository: TelemetryRepository
    prediction_service: FocusPredictionService | None = None
    activity_repository: SQLAlchemyActivityRepository | None = None
    broadcast_fn: BroadcastFn | None = None
    timezone: TimezoneLike = "local"
    #: Source of "now" for every lifecycle decision. Defaults to the real
    #: clock; tests inject a fixed one so expiry/cooldown never depend on
    #: when the suite happens to run.
    clock: ClockFn = field(default=lambda: datetime.now(UTC))
    _pending: dict[int, FeedbackRequest] = field(default_factory=dict)
    _ledger: dict[int, _AskLedger] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ── Creation ────────────────────────────────────────────────────────

    async def maybe_create(
        self,
        *,
        user_id: int = 1,
        now: datetime | None = None,
    ) -> FeedbackRequest | None:
        """Create a request when the sampling decision says a label is worth it.

        Returns None — without asking — when any guard declines: no
        prediction, a non-``ready`` status, a request already pending, the
        cooldown, the daily cap, or a deep-work session. The guards are
        ordered cheapest-first so a declined tick does no database work.
        """
        now = now or self.clock()

        async with self._lock:
            if self._has_live_pending_locked(user_id, now):
                return None
            ledger = self._ledger.setdefault(user_id, _AskLedger())
            if not self._within_caps(ledger, now):
                return None

        decision = await self._decide(user_id, now)
        if decision is None:
            return None
        probability, reasons, window_start, window_end = decision

        if await self._in_deep_work(user_id, now):
            logger.debug("Feedback request skipped: user is in deep work")
            return None

        request = FeedbackRequest(
            request_id=uuid.uuid4().hex,
            user_id=user_id,
            window_start=window_start,
            window_end=window_end,
            probability=probability,
            reasons=reasons,
            prompt=DEFAULT_PROMPT,
            created_at=now,
        )

        async with self._lock:
            # Re-check under the lock: the prediction call above yielded, so a
            # concurrent tick or route may have created or answered a request.
            if self._has_live_pending_locked(user_id, now):
                return None
            ledger = self._ledger.setdefault(user_id, _AskLedger())
            if not self._within_caps(ledger, now):
                return None
            self._pending[user_id] = request
            ledger.last_asked_at = now
            today = self._local_today(now)
            if ledger.day != today:
                ledger.day = today
                ledger.count = 1
            else:
                ledger.count += 1

        await self._broadcast(request)
        logger.info(
            "Feedback request created ({} reasons, probability={})",
            len(request.reasons),
            probability,
        )
        return request

    async def _decide(
        self, user_id: int, now: datetime
    ) -> tuple[float | None, list[str], datetime, datetime] | None:
        """Run the prediction + sampling decision, or None when not to ask."""
        if self.prediction_service is None:
            return None
        try:
            prediction = await self.prediction_service.predict_latest(
                user_id=user_id, now=now
            )
        except Exception as exc:
            logger.warning("Feedback request: prediction failed: {}", exc)
            return None
        if prediction.status != "ready":
            return None

        # Imported here: ``feedback_sampling`` is pure stdlib, but keeping the
        # decision next to its use makes the dependency obvious.
        from mindflow.services.feedback_sampling import (
            decide_feedback_request,
            signals_from_outputs,
        )

        # Only the probability is cheaply available at this tick: the rule
        # engine, the panel and the drift detector all cost a full analysis
        # run. The uncertainty band alone is the trigger; the other four
        # signals stay "unavailable" and therefore never fire (see
        # ``FeedbackSignals``).
        signals = signals_from_outputs(
            prediction={"focus_probability": prediction.focus_probability}
        )
        decision = decide_feedback_request(signals)
        if not decision.should_request:
            return None

        window_end = now
        newest = prediction.newest_window_start_utc
        if newest:
            try:
                start = datetime.fromisoformat(str(newest).replace("Z", "+00:00"))
                if start.tzinfo is None:
                    start = start.replace(tzinfo=UTC)
                window_end = start.astimezone(UTC) + timedelta(minutes=5)
            except (TypeError, ValueError):
                # Keep ``now`` rather than guessing a window boundary.
                logger.debug("Feedback request: unparseable window start {!r}", newest)
        window_start = window_end - REQUEST_WINDOW
        return prediction.focus_probability, list(decision.reasons), window_start, window_end

    async def _in_deep_work(self, user_id: int, now: datetime) -> bool:
        """Reuse the intervention service's deep-work rule, when available.

        The rule reads raw activity events, so this needs the activity
        repository. Without it the guard is skipped (never assumed True): a
        missing dependency must not silently disable feedback collection.
        """
        if self.activity_repository is None:
            return False
        try:
            events = await self.activity_repository.query_range(
                user_id, now - timedelta(minutes=45), now
            )
        except Exception as exc:
            logger.warning("Feedback request: activity query failed: {}", exc)
            return False
        from mindflow.services.intervention_service import _deep_work_guard

        return _deep_work_guard(events)

    # ── Answering ───────────────────────────────────────────────────────

    async def answer(
        self,
        request_id: str,
        label: str,
        *,
        now: datetime | None = None,
    ) -> AnswerResult | None:
        """Record an answer, or report why it could not be recorded.

        Returns None when *request_id* is unknown — the route maps that to 404
        so a client polling a request it already answered gets a clear signal
        rather than a silent success.
        """
        now = now or self.clock()
        async with self._lock:
            found = next(
                (
                    (uid, req)
                    for uid, req in self._pending.items()
                    if req.request_id == request_id
                ),
                None,
            )
            if found is None:
                return None
            user_id, request = found
            self._pending.pop(user_id, None)
            expired = request.is_expired(now)

        if expired:
            return AnswerResult(request_id=request_id, labeled_windows=0, status="expired")
        if label == LABEL_SKIP:
            return AnswerResult(request_id=request_id, labeled_windows=0, status="skipped")
        if label not in WRITABLE_LABELS:
            # The route validates the vocabulary; this is the belt-and-braces
            # path so an unvalidated caller cannot write an arbitrary string
            # into the training label column.
            raise ValueError(f"unsupported feedback label: {label!r}")

        written = await self.telemetry_repository.label_feature_windows(
            user_id=user_id,
            start=request.window_start,
            end=request.window_end,
            label=label,
        )
        return AnswerResult(
            request_id=request_id, labeled_windows=written, status="recorded"
        )

    async def pending(
        self, *, user_id: int = 1, now: datetime | None = None
    ) -> list[FeedbackRequest]:
        """The outstanding request for *user_id*, if any.

        At most one, because ``maybe_create`` refuses to stack them. An
        expired request is not returned — a client polling after the TTL must
        not be shown a prompt it can no longer answer. It is also not
        *forgotten*: answering it must yield ``status="expired"`` (the
        contract's third status), not the 404 that an evicted id would produce.
        """
        now = now or self.clock()
        async with self._lock:
            request = self._pending.get(user_id)
        if request is None or request.is_expired(now):
            return []
        return [request]

    # ── Internals ───────────────────────────────────────────────────────

    def _has_live_pending_locked(self, user_id: int, now: datetime) -> bool:
        """True while an unanswered, unexpired request occupies the slot.

        An expired request releases the slot but stays addressable so a late
        answer reports ``expired`` instead of 404.
        """
        request = self._pending.get(user_id)
        return request is not None and not request.is_expired(now)

    def _within_caps(self, ledger: _AskLedger, now: datetime) -> bool:
        """False when the cooldown or the per-day cap still forbids asking."""
        cooling_down = (
            ledger.last_asked_at is not None
            and now - ledger.last_asked_at < MIN_ASK_INTERVAL
        )
        day_exhausted = (
            ledger.day == self._local_today(now) and ledger.count >= MAX_ASKS_PER_DAY
        )
        return not (cooling_down or day_exhausted)

    def _local_today(self, now: datetime) -> date:
        """The user's calendar date — the cap resets at local midnight."""
        return now.astimezone(resolve_timezone(self.timezone)).date()

    async def _broadcast(self, request: FeedbackRequest) -> None:
        """Best-effort WS push; the polling endpoint is the fallback path."""
        if self.broadcast_fn is None:
            return
        try:
            await self.broadcast_fn(
                {"type": "feedback_request", "data": request.to_payload()}
            )
        except Exception as exc:
            logger.warning("Feedback request broadcast failed: {}", exc)


__all__ = [
    "DEFAULT_PROMPT",
    "LABEL_DISTRACTED",
    "LABEL_FOCUS",
    "LABEL_SKIP",
    "MAX_ASKS_PER_DAY",
    "MIN_ASK_INTERVAL",
    "REASON_LABELS",
    "REQUEST_TTL",
    "REQUEST_WINDOW",
    "AnswerResult",
    "FeedbackRequest",
    "FeedbackRequestStore",
]
