"""Tests for the proactive-learning feedback request store (contract §5).

Covers the lifecycle guards (one pending slot, 45-minute cooldown, 6-per-day
cap, 2-hour expiry), the NULL-only label write into
``behavior_feature_windows``, the WS frame shape, and both HTTP routes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.telemetry import router as telemetry_router
from mindflow.domain.events import make_event
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.domain.prediction import FocusPrediction
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import behavior_feature_windows, metadata
from mindflow.services.feedback_request_service import (
    MAX_ASKS_PER_DAY,
    MIN_ASK_INTERVAL,
    REQUEST_TTL,
    REQUEST_WINDOW,
    FeedbackRequestStore,
)
from mindflow.services.scheduler import _IntervalTrigger, build_scheduler


def _prediction(
    probability: float | None = 0.5,
    *,
    status: str = "ready",
    newest: datetime | None = None,
) -> FocusPrediction:
    return FocusPrediction(
        status=status,  # type: ignore[arg-type]
        focus_probability=probability,
        newest_window_start_utc=(newest or datetime(2026, 11, 10, 9, 55, tzinfo=UTC)).isoformat(),
    )


class _StubPredictionService:
    """Minimal stand-in for FocusPredictionService."""

    def __init__(self, prediction: FocusPrediction) -> None:
        self.prediction = prediction
        self.calls = 0

    async def predict_latest(
        self, *, user_id: int = 1, now: datetime | None = None
    ) -> FocusPrediction:
        self.calls += 1
        return self.prediction


class _StubActivityRepository:
    def __init__(self, events: list[Any]) -> None:
        self.events = events

    async def query_range(self, user_id: int, start: datetime, end: datetime) -> list[Any]:
        return self.events


@pytest.fixture
async def telemetry_repo(engine, session_factory) -> TelemetryRepository:
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
    return TelemetryRepository(session_factory)


def _store(
    telemetry_repo: TelemetryRepository,
    prediction: FocusPrediction | None = None,
    *,
    activity_repository: Any | None = None,
    broadcast: Any | None = None,
) -> FeedbackRequestStore:
    return FeedbackRequestStore(
        telemetry_repository=telemetry_repo,
        prediction_service=(
            _StubPredictionService(prediction) if prediction is not None else None
        ),
        activity_repository=activity_repository,
        broadcast_fn=broadcast,
    )


async def _labels(telemetry_repo: TelemetryRepository, user_id: int = 1) -> dict[str, str | None]:
    async with telemetry_repo._session_factory() as session:  # noqa: SLF001
        result = await session.execute(
            sa.select(
                behavior_feature_windows.c.window_start_utc,
                behavior_feature_windows.c.label,
            ).where(behavior_feature_windows.c.user_id == user_id)
        )
        return {row.window_start_utc: row.label for row in result.fetchall()}


async def _save_window(
    telemetry_repo: TelemetryRepository,
    start: datetime,
    *,
    label: str | None = None,
) -> None:
    await telemetry_repo.save_feature_window(
        user_id=1,
        window_start_utc=start,
        window_end_utc=start + timedelta(minutes=5),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        features_json=json.dumps({"app_switch_count": 2}),
        label=label,
    )


class TestCreationGuards:
    """When the store is allowed to ask at all."""

    async def test_no_prediction_service_never_asks(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        store = _store(telemetry_repo)

        assert await store.maybe_create(now=datetime(2026, 11, 10, 9, 0, tzinfo=UTC)) is None
        assert await store.pending() == []

    async def test_non_ready_prediction_never_asks(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        for status in ("no_data", "no_model", "stale", "inference_error"):
            store = _store(telemetry_repo, _prediction(0.5, status=status))

            assert await store.maybe_create(now=datetime(2026, 11, 10, 9, 0, tzinfo=UTC)) is None

    async def test_confident_prediction_does_not_ask(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        """Outside the uncertainty band no label carries information."""
        store = _store(telemetry_repo, _prediction(0.95))

        assert await store.maybe_create(now=datetime(2026, 11, 10, 9, 0, tzinfo=UTC)) is None

    async def test_uncertain_prediction_creates_request(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        now = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
        newest = datetime(2026, 11, 10, 9, 55, tzinfo=UTC)
        store = _store(telemetry_repo, _prediction(0.5, newest=newest))

        request = await store.maybe_create(now=now)

        assert request is not None
        # The window ends at the newest feature window, not at "now": the label
        # must describe behaviour that actually has features.
        assert request.window_end == newest + timedelta(minutes=5)
        assert request.window_start == request.window_end - REQUEST_WINDOW
        assert request.probability == 0.5
        assert "model_uncertain" in request.reasons
        assert request.prompt

    async def test_payload_uses_contract_keys_and_chinese_reasons(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        store = _store(telemetry_repo, _prediction(0.5))

        request = await store.maybe_create(now=datetime(2026, 11, 10, 10, 0, tzinfo=UTC))
        assert request is not None
        payload = request.to_payload()

        assert set(payload) == {
            "request_id",
            "window_start",
            "window_end",
            "probability",
            "reasons",
            "prompt",
        }
        assert payload["reasons"] == ["模型不确定"]
        assert datetime.fromisoformat(payload["window_start"]).tzinfo is not None

    async def test_pending_slot_blocks_a_second_request(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        store = _store(telemetry_repo, _prediction(0.5))
        first = await store.maybe_create(now=datetime(2026, 11, 10, 10, 0, tzinfo=UTC))
        assert first is not None

        # 30 minutes later: still inside the TTL, so only the pending slot can
        # be the reason a second request is refused.
        second = await store.maybe_create(now=datetime(2026, 11, 10, 10, 30, tzinfo=UTC))

        assert second is None
        assert [r.request_id for r in await store.pending()] == [first.request_id]

    async def test_cooldown_blocks_until_45_minutes_elapsed(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        store = _store(telemetry_repo, _prediction(0.5))
        t0 = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
        first = await store.maybe_create(now=t0)
        assert first is not None
        await store.answer(first.request_id, "skip", now=t0)

        too_soon = await store.maybe_create(now=t0 + MIN_ASK_INTERVAL - timedelta(minutes=1))
        assert too_soon is None

        allowed = await store.maybe_create(now=t0 + MIN_ASK_INTERVAL)
        assert allowed is not None

    async def test_daily_cap_limits_asks(self, telemetry_repo: TelemetryRepository) -> None:
        store = _store(telemetry_repo, _prediction(0.5))
        cursor = datetime(2026, 11, 10, 8, 0, tzinfo=UTC)

        created = 0
        for _ in range(MAX_ASKS_PER_DAY):
            request = await store.maybe_create(now=cursor)
            assert request is not None
            created += 1
            await store.answer(request.request_id, "skip", now=cursor)
            cursor += MIN_ASK_INTERVAL

        assert created == MAX_ASKS_PER_DAY
        assert await store.maybe_create(now=cursor) is None

        # A new local day resets the cap.
        assert await store.maybe_create(now=cursor + timedelta(days=1)) is not None

    async def test_daily_cap_counts_the_local_day_not_utc(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        """Six asks in one UTC day can span two local days and vice versa."""
        # Asia/Shanghai is UTC+8: 2026-11-10T18:00Z is already 11-11 locally.
        store = FeedbackRequestStore(
            telemetry_repository=telemetry_repo,
            prediction_service=_StubPredictionService(_prediction(0.5)),
            timezone="Asia/Shanghai",
        )
        cursor = datetime(2026, 11, 10, 18, 0, tzinfo=UTC)  # 02:00 local 11-11

        for _ in range(MAX_ASKS_PER_DAY):
            request = await store.maybe_create(now=cursor)
            assert request is not None
            await store.answer(request.request_id, "skip", now=cursor)
            cursor += MIN_ASK_INTERVAL

        assert await store.maybe_create(now=cursor) is None
        # Still 2026-11-10 in UTC, but 2026-11-11 06:30 locally: same local day,
        # so the cap holds. The next local midnight is 2026-11-11T16:00Z.
        assert await store.maybe_create(now=datetime(2026, 11, 11, 15, 59, tzinfo=UTC)) is None
        assert (
            await store.maybe_create(now=datetime(2026, 11, 11, 16, 0, tzinfo=UTC))
            is not None
        )

    async def test_deep_work_session_is_not_interrupted(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        base = datetime(2026, 11, 10, 8, 0, tzinfo=UTC)
        events = [
            make_event(
                user_id=1,
                timestamp_utc=base + timedelta(seconds=i * 10),
                duration_s=10.0,
                process_name="Code.exe",
            )
            for i in range(85)
        ]
        store = _store(
            telemetry_repo,
            _prediction(0.5),
            activity_repository=_StubActivityRepository(events),
        )

        assert await store.maybe_create(now=datetime(2026, 11, 10, 10, 0, tzinfo=UTC)) is None

    async def test_missing_activity_repository_does_not_block(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        """A missing dependency must not silently disable feedback collection."""
        store = _store(telemetry_repo, _prediction(0.5))

        assert await store.maybe_create(now=datetime(2026, 11, 10, 10, 0, tzinfo=UTC)) is not None

    async def test_broadcast_sends_contract_frame(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        frames: list[dict[str, Any]] = []

        async def _broadcast(frame: dict[str, Any]) -> int:
            frames.append(frame)
            return 1

        store = _store(telemetry_repo, _prediction(0.5), broadcast=_broadcast)

        request = await store.maybe_create(now=datetime(2026, 11, 10, 10, 0, tzinfo=UTC))

        assert request is not None
        assert len(frames) == 1
        assert frames[0]["type"] == "feedback_request"
        assert frames[0]["data"]["request_id"] == request.request_id
        assert frames[0]["data"]["reasons"] == ["模型不确定"]


class TestAnswering:
    """Writing the label back into the feature windows."""

    async def test_answer_labels_only_null_windows(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        newest = datetime(2026, 11, 10, 9, 55, tzinfo=UTC)
        now = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
        # Three windows inside the request's 15-minute range, one already
        # labelled by explicit per-session feedback (must not be overwritten).
        await _save_window(telemetry_repo, newest - timedelta(minutes=10), label="focus")
        await _save_window(telemetry_repo, newest - timedelta(minutes=5))
        await _save_window(telemetry_repo, newest)
        # Outside the range: untouched.
        await _save_window(telemetry_repo, newest - timedelta(minutes=40))

        store = _store(telemetry_repo, _prediction(0.5, newest=newest))
        request = await store.maybe_create(now=now)
        assert request is not None

        result = await store.answer(request.request_id, "distracted", now=now)

        assert result is not None
        assert result.status == "recorded"
        assert result.labeled_windows == 2
        labels = await _labels(telemetry_repo)
        assert labels[(newest - timedelta(minutes=10)).isoformat()] == "focus"
        assert labels[(newest - timedelta(minutes=5)).isoformat()] == "distracted"
        assert labels[newest.isoformat()] == "distracted"
        assert labels[(newest - timedelta(minutes=40)).isoformat()] is None

    async def test_second_answer_is_not_found(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        now = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
        store = _store(telemetry_repo, _prediction(0.5))
        request = await store.maybe_create(now=now)
        assert request is not None

        assert await store.answer(request.request_id, "focus", now=now) is not None
        assert await store.answer(request.request_id, "focus", now=now) is None

    async def test_skip_writes_nothing(self, telemetry_repo: TelemetryRepository) -> None:
        newest = datetime(2026, 11, 10, 9, 55, tzinfo=UTC)
        now = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
        await _save_window(telemetry_repo, newest)
        store = _store(telemetry_repo, _prediction(0.5, newest=newest))
        request = await store.maybe_create(now=now)
        assert request is not None

        result = await store.answer(request.request_id, "skip", now=now)

        assert result is not None
        assert result.status == "skipped"
        assert result.labeled_windows == 0
        assert (await _labels(telemetry_repo))[newest.isoformat()] is None

    async def test_expired_request_labels_nothing(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        newest = datetime(2026, 11, 10, 7, 55, tzinfo=UTC)
        created = datetime(2026, 11, 10, 8, 0, tzinfo=UTC)
        await _save_window(telemetry_repo, newest)
        store = _store(telemetry_repo, _prediction(0.5, newest=newest))
        request = await store.maybe_create(now=created)
        assert request is not None

        result = await store.answer(
            request.request_id, "focus", now=created + REQUEST_TTL
        )

        assert result is not None
        assert result.status == "expired"
        assert result.labeled_windows == 0
        assert (await _labels(telemetry_repo))[newest.isoformat()] is None

    async def test_pending_drops_expired_request(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        created = datetime(2026, 11, 10, 8, 0, tzinfo=UTC)
        store = _store(telemetry_repo, _prediction(0.5))
        assert await store.maybe_create(now=created) is not None

        assert await store.pending(now=created + REQUEST_TTL) == []

    async def test_late_answer_to_an_expired_prompt_reports_expired(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        """A client that polls first and answers late must see ``expired``.

        Dropping the id when it expires would answer 404 instead, and the
        contract's third status would be unreachable over HTTP.
        """
        created = datetime(2026, 11, 10, 8, 0, tzinfo=UTC)
        store = _store(telemetry_repo, _prediction(0.5))
        request = await store.maybe_create(now=created)
        assert request is not None
        late = created + REQUEST_TTL + timedelta(minutes=1)

        assert await store.pending(now=late) == []
        result = await store.answer(request.request_id, "focus", now=late)

        assert result is not None
        assert result.status == "expired"

    async def test_expired_prompt_releases_the_slot(
        self, telemetry_repo: TelemetryRepository
    ) -> None:
        created = datetime(2026, 11, 10, 8, 0, tzinfo=UTC)
        store = _store(telemetry_repo, _prediction(0.5))
        assert await store.maybe_create(now=created) is not None

        # Past the TTL and past the cooldown: the user may be asked again.
        assert await store.maybe_create(now=created + REQUEST_TTL) is not None

    async def test_unknown_label_raises(self, telemetry_repo: TelemetryRepository) -> None:
        now = datetime(2026, 11, 10, 10, 0, tzinfo=UTC)
        store = _store(telemetry_repo, _prediction(0.5))
        request = await store.maybe_create(now=now)
        assert request is not None

        with pytest.raises(ValueError):
            await store.answer(request.request_id, "mixed", now=now)


class TestRoutes:
    """Both contract §5 endpoints."""

    @pytest.fixture
    def app(self, telemetry_repo: TelemetryRepository) -> FastAPI:
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(telemetry_router, prefix="/api/v1")
        app.state.feedback_request_store = _store(telemetry_repo, _prediction(0.5))
        return app

    async def test_pending_then_answer_roundtrip(self, app: FastAPI) -> None:
        store: FeedbackRequestStore = app.state.feedback_request_store
        request = await store.maybe_create(now=datetime(2026, 11, 10, 10, 0, tzinfo=UTC))
        assert request is not None
        client = TestClient(app)

        pending = client.get("/api/v1/telemetry/feedback-requests/pending")
        assert pending.status_code == 200
        items = pending.json()["items"]
        assert len(items) == 1
        assert items[0]["request_id"] == request.request_id
        assert items[0]["reasons"] == ["模型不确定"]

        answered = client.post(
            f"/api/v1/telemetry/feedback-requests/{request.request_id}/answer",
            json={"label": "focus"},
        )
        assert answered.status_code == 200
        assert answered.json() == {
            "request_id": request.request_id,
            "labeled_windows": 0,
            "status": "recorded",
        }

        assert client.get("/api/v1/telemetry/feedback-requests/pending").json()["items"] == []

    def test_pending_is_empty_by_default(self, app: FastAPI) -> None:
        response = TestClient(app).get("/api/v1/telemetry/feedback-requests/pending")

        assert response.status_code == 200
        assert response.json() == {"items": []}

    def test_unknown_request_id_is_404(self, app: FastAPI) -> None:
        response = TestClient(app).post(
            "/api/v1/telemetry/feedback-requests/does-not-exist/answer",
            json={"label": "focus"},
        )

        assert response.status_code == 404
        assert response.json()["type"].endswith("feedback-request-not-found")

    def test_invalid_label_is_rejected(self, app: FastAPI) -> None:
        response = TestClient(app).post(
            "/api/v1/telemetry/feedback-requests/whatever/answer",
            json={"label": "mixed"},
        )

        assert response.status_code == 422


class TestSchedulerJob:
    """The dedicated 15-minute interval job (B5: not the intervention tick)."""

    async def test_registered_as_its_own_15min_interval_job(self) -> None:
        store = FeedbackRequestStore(telemetry_repository=MagicMock())

        scheduler = build_scheduler(feedback_request_store=store)

        job = scheduler.get_job("feedback_request_check")
        assert job is not None
        assert isinstance(job.trigger, _IntervalTrigger)
        assert job.trigger.interval.total_seconds() == 900.0
        # Asking for a label is not an intervention: the two must not share a
        # job, or one feature's gates would silently govern the other.
        assert scheduler.get_job("auto_intervention_check") is None

    async def test_absent_store_registers_no_job(self) -> None:
        scheduler = build_scheduler()

        assert scheduler.get_job("feedback_request_check") is None

    async def test_job_failure_is_logged_not_raised(self) -> None:
        store = FeedbackRequestStore(telemetry_repository=MagicMock())
        store.maybe_create = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        scheduler = build_scheduler(feedback_request_store=store)
        coro = next(j["coro"] for j in scheduler._jobs if j["name"] == "feedback_request_check")  # noqa: SLF001

        await coro()

        store.maybe_create.assert_awaited_once_with(user_id=1)
