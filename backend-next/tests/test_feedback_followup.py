"""Appendix B: an ignored/rejected reminder is followed up once (active learning).

The reminder response itself says nothing about whether the model was right —
"ignored" can mean "I was focused, you interrupted me" or "I was distracted and
did not care". So the response is never a label; the *answer* to one follow-up
question is, and it lands in ``behavior_feature_windows.label`` exactly like an
uncertainty-driven answer does.

These tests pin the follow-up lifecycle: which reminder qualifies, which does
not, that one reminder is asked about at most once, and that the guards
(cooldown, pending slot, deep work) are the existing ones rather than a second
set of rules.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from mindflow.domain.events import make_event
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.domain.prediction import FocusPrediction
from mindflow.infrastructure.repositories.intervention import InterventionLogRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import behavior_feature_windows
from mindflow.services.feedback_request_service import (
    FOLLOWUP_PROMPT,
    MIN_ASK_INTERVAL,
    REQUEST_TTL,
    REQUEST_WINDOW,
    FeedbackRequestStore,
)

_NOW = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)


class _StubPredictionService:
    """Uncertain prediction — the original, non-follow-up path."""

    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability
        self.calls = 0

    async def predict_latest(
        self, *, user_id: int = 1, now: datetime | None = None
    ) -> FocusPrediction:
        self.calls += 1
        return FocusPrediction(
            status="ready",
            focus_probability=self.probability,
            newest_window_start_utc=(_NOW - timedelta(minutes=5)).isoformat(),
        )


@pytest.fixture
async def telemetry_repo(engine, session_factory) -> TelemetryRepository:
    async with engine.begin() as connection:
        await connection.run_sync(behavior_feature_windows.metadata.create_all)
    return TelemetryRepository(session_factory)


@pytest.fixture
async def intervention_repo(engine, session_factory) -> InterventionLogRepository:
    async with engine.begin() as connection:
        await connection.run_sync(behavior_feature_windows.metadata.create_all)
    return InterventionLogRepository(session_factory=session_factory)


def _store(
    telemetry_repo: TelemetryRepository,
    intervention_repo: InterventionLogRepository | None,
    *,
    prediction_service: Any | None = None,
    activity_repository: Any | None = None,
    now: datetime = _NOW,
) -> FeedbackRequestStore:
    return FeedbackRequestStore(
        telemetry_repository=telemetry_repo,
        prediction_service=prediction_service or _StubPredictionService(),
        activity_repository=activity_repository,
        intervention_repository=intervention_repo,
        clock=lambda: now,
    )


async def _reminder(
    repo: InterventionLogRepository,
    *,
    response: str | None,
    triggered_at: datetime,
    intervention_id: str,
) -> None:
    await repo.log_triggered(
        1,
        "gentle_nudge",
        intervention_id=intervention_id,
        triggered_at=triggered_at,
    )
    if response is not None:
        await repo.update_response(intervention_id, response)  # type: ignore[arg-type]


async def _save_window(
    repo: TelemetryRepository, start: datetime, *, label: str | None = None
) -> None:
    await repo.save_feature_window(
        user_id=1,
        window_start_utc=start,
        window_end_utc=start + timedelta(minutes=5),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features_json=json.dumps({"app_switch_count": 2}),
        label=label,
    )


async def _labels(repo: TelemetryRepository) -> dict[str, str | None]:
    async with repo._session_factory() as session:  # noqa: SLF001
        result = await session.execute(
            sa.select(
                behavior_feature_windows.c.window_start_utc,
                behavior_feature_windows.c.label,
            ).where(behavior_feature_windows.c.user_id == 1)
        )
        return {row.window_start_utc: row.label for row in result.fetchall()}


class TestFollowupCreation:
    """Which reminder produces a follow-up, and what the question looks like."""

    async def test_ignored_reminder_is_followed_up_with_its_own_window(
        self, telemetry_repo: TelemetryRepository, intervention_repo: InterventionLogRepository
    ) -> None:
        triggered = _NOW - timedelta(minutes=10)
        await _reminder(
            intervention_repo,
            response="ignored",
            triggered_at=triggered,
            intervention_id="reminder-1",
        )
        store = _store(telemetry_repo, intervention_repo)

        request = await store.maybe_create(now=_NOW)

        assert request is not None
        assert request.prompt == FOLLOWUP_PROMPT
        assert request.reasons == ["reminder_ignored"]
        assert request.probability is None
        # The question is about the 15 minutes *before* the reminder fired, not
        # about "now": that is the stretch the reminder was judging.
        assert request.window_end == triggered
        assert request.window_start == triggered - REQUEST_WINDOW

    async def test_dismissed_reminder_uses_the_dismissed_reason(
        self, telemetry_repo: TelemetryRepository, intervention_repo: InterventionLogRepository
    ) -> None:
        await _reminder(
            intervention_repo,
            response="dismissed",
            triggered_at=_NOW - timedelta(minutes=5),
            intervention_id="reminder-2",
        )
        store = _store(telemetry_repo, intervention_repo)

        request = await store.maybe_create(now=_NOW)

        assert request is not None
        assert request.reasons == ["reminder_dismissed"]
        assert request.to_payload()["reasons"] == ["你选了“不适合我”"]

    async def test_same_reminder_is_asked_about_only_once(
        self, telemetry_repo: TelemetryRepository, intervention_repo: InterventionLogRepository
    ) -> None:
        await _reminder(
            intervention_repo,
            response="ignored",
            triggered_at=_NOW - timedelta(minutes=10),
            intervention_id="reminder-1",
        )
        store = _store(telemetry_repo, intervention_repo)

        first = await store.maybe_create(now=_NOW)
        assert first is not None
        assert first.reasons == ["reminder_ignored"]
        await store.answer(first.request_id, "skip", now=_NOW)

        # Past the cooldown, so only "already asked" can explain a second
        # request having a different basis. The original uncertainty path is
        # what remains, which is the point: the reminder is not re-asked.
        second = await store.maybe_create(now=_NOW + MIN_ASK_INTERVAL)

        assert second is not None
        assert second.reasons != ["reminder_ignored"]
        assert "model_uncertain" in second.reasons

    async def test_accepted_reminder_does_not_trigger_a_followup(
        self, telemetry_repo: TelemetryRepository, intervention_repo: InterventionLogRepository
    ) -> None:
        await _reminder(
            intervention_repo,
            response="accepted",
            triggered_at=_NOW - timedelta(minutes=10),
            intervention_id="reminder-3",
        )
        store = _store(telemetry_repo, intervention_repo)

        request = await store.maybe_create(now=_NOW)

        assert request is not None
        assert "model_uncertain" in request.reasons
        assert request.prompt != FOLLOWUP_PROMPT

    async def test_reminder_older_than_the_ttl_is_ignored(
        self, telemetry_repo: TelemetryRepository, intervention_repo: InterventionLogRepository
    ) -> None:
        await _reminder(
            intervention_repo,
            response="ignored",
            triggered_at=_NOW - REQUEST_TTL - timedelta(minutes=1),
            intervention_id="reminder-old",
        )
        store = _store(telemetry_repo, intervention_repo)

        request = await store.maybe_create(now=_NOW)

        assert request is not None
        assert "model_uncertain" in request.reasons

    async def test_cooldown_still_governs_followups(
        self, telemetry_repo: TelemetryRepository, intervention_repo: InterventionLogRepository
    ) -> None:
        await _reminder(
            intervention_repo,
            response="ignored",
            triggered_at=_NOW - timedelta(minutes=10),
            intervention_id="reminder-1",
        )
        store = _store(telemetry_repo, intervention_repo)
        first = await store.maybe_create(now=_NOW)
        assert first is not None
        await store.answer(first.request_id, "skip", now=_NOW)

        await _reminder(
            intervention_repo,
            response="ignored",
            triggered_at=_NOW + timedelta(minutes=1),
            intervention_id="reminder-2",
        )
        too_soon = await store.maybe_create(
            now=_NOW + MIN_ASK_INTERVAL - timedelta(minutes=1)
        )

        assert too_soon is None

    async def test_deep_work_still_suppresses_a_followup(
        self, telemetry_repo: TelemetryRepository, intervention_repo: InterventionLogRepository
    ) -> None:
        await _reminder(
            intervention_repo,
            response="ignored",
            triggered_at=_NOW - timedelta(minutes=10),
            intervention_id="reminder-1",
        )

        class _BusyActivityRepository:
            async def query_range(
                self, user_id: int, start: datetime, end: datetime
            ) -> list[Any]:
                base = _NOW - timedelta(minutes=40)
                return [
                    make_event(
                        user_id=1,
                        timestamp_utc=base + timedelta(seconds=index * 10),
                        duration_s=10.0,
                        process_name="Code.exe",
                    )
                    for index in range(85)
                ]

        store = _store(
            telemetry_repo,
            intervention_repo,
            activity_repository=_BusyActivityRepository(),
        )

        assert await store.maybe_create(now=_NOW) is None

    async def test_missing_repository_keeps_the_original_path(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        store = _store(telemetry_repo, None)

        request = await store.maybe_create(now=_NOW)

        assert request is not None
        assert "model_uncertain" in request.reasons

    async def test_repository_failure_falls_back_to_uncertainty(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        class _BrokenRepository:
            async def query_range(
                self, user_id: int, start: datetime, end: datetime
            ) -> list[dict[str, Any]]:
                raise RuntimeError("database unavailable")

        store = _store(telemetry_repo, _BrokenRepository())  # type: ignore[arg-type]

        request = await store.maybe_create(now=_NOW)

        assert request is not None
        assert "model_uncertain" in request.reasons


class TestFollowupAnswer:
    """The answer to a follow-up is the label — the reminder response is not."""

    async def test_answering_focus_labels_the_preceding_window(
        self, telemetry_repo: TelemetryRepository, intervention_repo: InterventionLogRepository
    ) -> None:
        triggered = _NOW - timedelta(minutes=10)
        await _reminder(
            intervention_repo,
            response="ignored",
            triggered_at=triggered,
            intervention_id="reminder-1",
        )
        # Three windows inside the request range; the middle one already
        # carries the user's own answer and must not be overwritten.
        await _save_window(telemetry_repo, triggered - timedelta(minutes=15))
        await _save_window(
            telemetry_repo, triggered - timedelta(minutes=10), label="distracted"
        )
        await _save_window(telemetry_repo, triggered - timedelta(minutes=5))
        store = _store(telemetry_repo, intervention_repo)

        request = await store.maybe_create(now=_NOW)
        assert request is not None
        result = await store.answer(request.request_id, "focus", now=_NOW)

        assert result is not None
        assert result.status == "recorded"
        assert result.labeled_windows == 2
        labels = await _labels(telemetry_repo)
        assert labels[(triggered - timedelta(minutes=15)).isoformat()] == "focus"
        assert labels[(triggered - timedelta(minutes=10)).isoformat()] == "distracted"
        assert labels[(triggered - timedelta(minutes=5)).isoformat()] == "focus"
