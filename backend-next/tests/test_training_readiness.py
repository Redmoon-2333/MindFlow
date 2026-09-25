"""Training-readiness endpoint integration tests.

Covers: empty state, partial data, trainable state, and temporal-overlap
matching semantics.  Uses real SQLite DB with activity events, focus
sessions, feedback, and V2 feature windows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.analytics import router as analytics_router
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.baseline import (
    BaselineRepository,
    baseline_models,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
    focus_sessions,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import (
    behavior_feature_windows,
    focus_session_feedback,
)

# ── V2 features JSON (24-dim) ────────────────────────────────────────────

_V2_FEATURES_JSON = json.dumps({
    "app_switch_count": 3.0, "domain_switch_count": 2.0,
    "longest_segment_ratio": 0.6, "idle_ratio": 0.1,
    "keypress_rate_per_min": 25.0, "mouse_click_rate_per_min": 12.0,
    "scroll_rate_per_min": 5.0, "mouse_distance_per_min": 200.0,
    "input_active_ratio": 0.7, "interaction_bursts_per_min": 2.0,
    "click_key_ratio": 0.5, "browser_ratio": 0.3,
    "audible_browser_ratio": 0.1, "active_seconds_ratio": 0.8,
    "top_app_ratio": 0.7, "top_domain_ratio": 0.5,
    "interaction_interval_mean_s": 10.0,
    "interaction_interval_std_s": 5.0,
    "interaction_interval_cv": 0.5,
    "hour_sin": 0.5, "hour_cos": 0.5,
    "weekday_sin": 0.5, "weekday_cos": 0.5,
    "task_type_code": 0.0,
})


# ── Seed helpers ───────────────────────────────────────────────────────────


def _v2_window(
    user_id: int, start: datetime, end: datetime,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "window_start_utc": start,
        "window_end_utc": end,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features_json": _V2_FEATURES_JSON,
        "label": None,
    }


async def _seed_windows(
    telemetry_repo: TelemetryRepository, rows: list[dict[str, Any]],
) -> None:
    await telemetry_repo.upsert_feature_windows(rows)


async def _seed_feedback(
    engine: Any, rows: list[dict[str, Any]],
) -> None:
    """Direct-insert feedback rows into focus_session_feedback table."""
    async with engine.begin() as conn:
        for row in rows:
            await conn.execute(
                focus_session_feedback.insert().values(
                    id=row.get("id", f"fb-{row['session_id']}"),
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    label=row["label"],
                    score=row["score"],
                    task_type=row.get("task_type"),
                    created_at=row.get("created_at", datetime.now(UTC).isoformat()),
                )
            )


async def _seed_focus_sessions(
    engine: Any, rows: list[dict[str, Any]],
) -> None:
    """Direct-insert focus sessions (with start_time/end_time for matching)."""
    async with engine.begin() as conn:
        for row in rows:
            await conn.execute(
                focus_sessions.insert().values(
                    id=row["id"],
                    user_id=row["user_id"],
                    date=row["date"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    session_type=row.get("session_type", "focus"),
                    dominant_app=row.get("dominant_app"),
                    focus_score=row.get("focus_score"),
                    switch_count=row.get("switch_count"),
                )
            )


async def _seed_activity_events(
    engine: Any, count: int, user_id: int = 1,
) -> None:
    """Insert minimal activity events for the aggregate summary."""
    base = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
    async with engine.begin() as conn:
        for i in range(count):
            ts = base + timedelta(minutes=i * 2)
            await conn.execute(
                activity_events.insert().values(
                    id=f"act-{i:04d}",
                    user_id=user_id,
                    timestamp=ts.isoformat(),
                    duration_s=60.0,
                    data_json=json.dumps({"app_name": "Code.exe", "is_idle": False}),
                    event_type="window_snapshot",
                )
            )


# ── App factory ────────────────────────────────────────────────────────────


def _make_app(engine: Any, session_factory: async_sessionmaker[AsyncSession]) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    telemetry_repo = TelemetryRepository(session_factory=session_factory)
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
    baseline_repo = BaselineRepository(session_factory=session_factory)

    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.telemetry_repository = telemetry_repo
    app.state.focus_repository = focus_repo
    app.state.activity_repository = activity_repo
    app.state.baseline_repository = baseline_repo
    app.state.v2_model_manager = None
    app.state.v2_training_mode = "rule_engine_only"
    app.include_router(analytics_router, prefix="/api/v1")
    return app


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
async def tables(engine) -> None:
    """Create all tables needed for training readiness tests."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(focus_session_feedback.metadata.create_all)
        await conn.run_sync(behavior_feature_windows.metadata.create_all)
        await conn.run_sync(baseline_models.metadata.create_all)


# ── Tests: empty state ─────────────────────────────────────────────────────


class TestEmptyDatabase:
    """When no data exists, all counts are zero."""

    async def test_empty_returns_all_zeros(
        self, engine, session_factory, tables,
    ) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        assert body["raw_events"]["total_events"] == 0
        assert body["raw_events"]["coverage_days"] == 0
        assert body["raw_events"]["oldest_timestamp"] is None
        assert body["raw_events"]["newest_timestamp"] is None

        assert body["v2_windows"]["total"] == 0
        assert body["v2_windows"]["eligible_count"] == 0
        assert body["v2_windows"]["matched_focus_count"] == 0
        assert body["v2_windows"]["matched_distract_count"] == 0

        assert body["feedback_labels"]["total"] == 0

        assert body["trainable"] is False
        assert body["trainable_window_count"] == 0
        assert body["trainable_class_count"] == 0
        assert body["evaluable"] is False
        assert body["evaluable_explicit_count"] == 0
        assert body["evaluable_date_count"] == 0

        gates = {g["key"]: g for g in body["gates"]}
        assert gates["minimum_days"]["status"] == "failed"
        assert gates["minimum_explicit_feedback"]["status"] == "failed"
        assert gates["minimum_class_feedback"]["status"] == "failed"
        assert gates["balanced_accuracy"]["status"] == "not_evaluated"
        assert gates["minority_f1"]["status"] == "not_evaluated"
        assert gates["calibration_better_than_rule"]["status"] == "not_evaluated"
        assert gates["stable_date_folds"]["status"] == "not_evaluated"
        assert body["v2_windows"]["schema_version"] == FEATURE_SCHEMA_VERSION
        assert gates["minimum_days"]["threshold"] == ">= 7"

        assert len(body["blockers"]) >= 2
        assert body["current_training_job"] is None


# ── Tests: activity events tracked ─────────────────────────────────────────


class TestActivityEventsSummary:
    async def test_activity_events_reflected_in_raw_events(
        self, engine, session_factory, tables,
    ) -> None:
        await _seed_activity_events(engine, 50)

        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        re = body["raw_events"]
        assert re["total_events"] == 50
        assert re["coverage_days"] >= 1
        assert re["oldest_timestamp"] is not None
        assert re["newest_timestamp"] is not None


# ── Tests: temporal overlap vs non-overlap ────────────────────────────────


class TestTemporalOverlap:
    """Feedback counts alone do NOT make data trainable; time overlap must match."""

    async def test_no_overlap_not_trainable(
        self, engine, session_factory, tables,
    ) -> None:
        """20 feedback rows with zero temporal overlap → NOT trainable."""
        telemetry_repo = TelemetryRepository(session_factory=session_factory)

        # Feature windows on day 1 (2026-07-25)
        base = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
        windows = [_v2_window(1, base + timedelta(minutes=i * 5),
                              base + timedelta(minutes=(i + 1) * 5))
                   for i in range(40)]
        await _seed_windows(telemetry_repo, windows)

        # Focus sessions + feedback on day 10 (2026-08-04) — no overlap
        fb_base = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
        fcs_rows: list[dict[str, Any]] = []
        fb_rows: list[dict[str, Any]] = []
        for i in range(15):
            sid = f"focus-sess-{i}"
            fcs_rows.append({
                "id": sid, "user_id": 1,
                "date": "2026-08-04",
                "start_time": (fb_base + timedelta(minutes=i * 10)).isoformat(),
                "end_time": (fb_base + timedelta(minutes=(i + 1) * 10 - 1)).isoformat(),
                "session_type": "focus",
            })
            fb_rows.append({
                "user_id": 1, "session_id": sid, "label": "focus",
                "score": 4, "task_type": "coding",
                "created_at": (fb_base + timedelta(minutes=i * 10)).isoformat(),
            })
        for i in range(10):
            sid = f"dist-sess-{i}"
            fcs_rows.append({
                "id": sid, "user_id": 1,
                "date": "2026-08-04",
                "start_time": (fb_base + timedelta(minutes=(i + 15) * 10)).isoformat(),
                "end_time": (fb_base + timedelta(minutes=(i + 16) * 10 - 1)).isoformat(),
                "session_type": "focus",
            })
            fb_rows.append({
                "user_id": 1, "session_id": sid, "label": "distracted",
                "score": 2, "task_type": "browsing",
                "created_at": (fb_base + timedelta(minutes=(i + 15) * 10)).isoformat(),
            })

        await _seed_focus_sessions(engine, fcs_rows)
        await _seed_feedback(engine, fb_rows)

        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        # Raw counts look healthy...
        assert body["v2_windows"]["total"] == 40
        assert body["feedback_labels"]["focus"] == 15
        assert body["feedback_labels"]["distract"] == 10
        assert body["feedback_labels"]["total"] == 25

        # ...but ZERO temporal overlap → not trainable
        assert body["v2_windows"]["eligible_count"] == 0
        assert body["v2_windows"]["matched_focus_count"] == 0
        assert body["v2_windows"]["matched_distract_count"] == 0
        assert body["trainable"] is False
        assert body["trainable_window_count"] == 0
        assert body["evaluable"] is False

        blockers = {b["code"] for b in body["blockers"]}
        assert "insufficient_eligible_windows" in blockers

    async def test_overlap_makes_trainable(
        self, engine, session_factory, tables,
    ) -> None:
        """Feedback with temporal overlap with windows → trainable."""
        telemetry_repo = TelemetryRepository(session_factory=session_factory)

        # Feature windows spanning 3 days for overlap with all feedback
        base = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
        windows = []
        for d in range(3):
            day_base = base + timedelta(days=d)
            for i in range(14):
                windows.append(_v2_window(
                    1, day_base + timedelta(minutes=i * 5),
                    day_base + timedelta(minutes=(i + 1) * 5),
                ))
        await _seed_windows(telemetry_repo, windows)

        # Focus sessions + feedback on SAME day ranges → overlap
        # Spread across 3 days so evaluable (needs >=3 distinct dates)
        fcs_rows: list[dict[str, Any]] = []
        fb_rows: list[dict[str, Any]] = []
        for i in range(5):
            sid = f"of-sess-{i}"
            start = base + timedelta(minutes=i * 5)
            end = start + timedelta(minutes=4)
            fcs_rows.append({
                "id": sid, "user_id": 1,
                "date": "2026-07-25",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "session_type": "focus",
            })
            fb_rows.append({
                "user_id": 1, "session_id": sid, "label": "focus",
                "score": 4, "task_type": "coding",
                "created_at": start.isoformat(),
            })
        for i in range(5):
            sid = f"of-sess-d2-{i}"
            start = base + timedelta(days=1, minutes=i * 5)
            end = start + timedelta(minutes=4)
            fcs_rows.append({
                "id": sid, "user_id": 1,
                "date": "2026-07-26",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "session_type": "focus",
            })
            fb_rows.append({
                "user_id": 1, "session_id": sid, "label": "focus",
                "score": 4, "task_type": "coding",
                "created_at": start.isoformat(),
            })
        for i in range(5):
            sid = f"of-sess-d3-{i}"
            start = base + timedelta(days=2, minutes=i * 5)
            end = start + timedelta(minutes=4)
            fcs_rows.append({
                "id": sid, "user_id": 1,
                "date": "2026-07-27",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "session_type": "focus",
            })
            fb_rows.append({
                "user_id": 1, "session_id": sid, "label": "focus",
                "score": 4, "task_type": "coding",
                "created_at": start.isoformat(),
            })
        for i in range(5):
            sid = f"od-sess-d3-{i}"
            start = base + timedelta(days=2, minutes=(i + 5) * 5)
            end = start + timedelta(minutes=4)
            fcs_rows.append({
                "id": sid, "user_id": 1,
                "date": "2026-07-27",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "session_type": "focus",
            })
            fb_rows.append({
                "user_id": 1, "session_id": sid, "label": "distracted",
                "score": 2, "task_type": "browsing",
                "created_at": start.isoformat(),
            })

        await _seed_focus_sessions(engine, fcs_rows)
        await _seed_feedback(engine, fb_rows)

        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        # Time overlap → eligible windows detected
        assert body["v2_windows"]["eligible_count"] >= 10
        assert body["v2_windows"]["matched_focus_count"] >= 5
        assert body["v2_windows"]["matched_distract_count"] >= 5

        assert body["trainable"] is True
        assert body["trainable_window_count"] >= 10
        assert body["trainable_class_count"] >= 2

        assert body["evaluable"] is True
        assert body["evaluable_explicit_count"] >= 10

        # Three days allow an offline evaluation, not model activation.
        gates = {g["key"]: g for g in body["gates"]}
        assert gates["minimum_days"]["status"] == "failed"
        assert gates["minimum_days"]["threshold"] == ">= 7"
        assert gates["minimum_explicit_feedback"]["status"] == "passed"
        assert gates["minimum_class_feedback"]["status"] == "passed"


# ── Tests: partial / insufficient ──────────────────────────────────────────


@pytest.mark.parametrize("bridge_dates", [False, True])
@pytest.mark.parametrize("extra_weak_days", [False, True])
async def test_evaluable_requires_independent_explicit_date_groups(
    engine, session_factory, tables, bridge_dates: bool, extra_weak_days: bool,
) -> None:
    from mindflow.train.v2 import evaluate_v2_candidates, prepare_v2_training_data

    base = datetime(2026, 9, 1, 9, tzinfo=UTC)
    windows, sessions, feedback = [], [], []

    def add_session(
        sid: str, start: datetime, end: datetime, focus: bool,
        window_starts: list[datetime],
    ) -> None:
        sessions.append({
            "id": sid, "user_id": 1, "date": start.date().isoformat(),
            "start_time": start.isoformat(), "end_time": end.isoformat(),
        })
        feedback.append({
            "session_id": sid, "user_id": 1,
            "label": "focus" if focus else "distracted", "score": 5 if focus else 1,
            "start_time": start.isoformat(), "end_time": end.isoformat(),
        })
        windows.extend(
            _v2_window(1, window_start, window_start + timedelta(minutes=5))
            for window_start in window_starts
        )

    for day in range(3):
        for slot in range(4):
            start = base + timedelta(days=day, minutes=slot * 10)
            add_session(
                f"day-{day}-session-{slot}", start, start + timedelta(minutes=5),
                bool(slot % 2), [start],
            )
    if bridge_dates:
        for day in range(2):
            start = base.replace(hour=23, minute=55) + timedelta(days=day)
            add_session(
                f"overnight-{day}", start, start + timedelta(minutes=20),
                True, [start, start + timedelta(minutes=10)],
            )
    if extra_weak_days:
        for day in (3, 4):
            start = base + timedelta(days=day)
            windows.append(_v2_window(1, start, start + timedelta(minutes=5)))

    prepared = prepare_v2_training_data(windows, feedback)
    explicit_groups = {
        group for group, explicit in zip(prepared.group_ids, prepared.explicit_mask, strict=True)
        if explicit
    }
    expected_groups = 1 if bridge_dates else 3
    assert len(explicit_groups) == expected_groups
    assert len(set(prepared.group_ids)) == expected_groups + (2 if extra_weak_days else 0)
    if bridge_dates:
        evaluation = evaluate_v2_candidates(prepared)
        assert evaluation["status"] == "insufficient_data"
        assert evaluation["date_group_count"] == 1

    await _seed_windows(TelemetryRepository(session_factory=session_factory), windows)
    await _seed_focus_sessions(engine, sessions)
    await _seed_feedback(engine, feedback)
    with TestClient(_make_app(engine, session_factory)) as client:
        response = client.get("/api/v1/analytics/training-readiness")
    assert response.status_code == 200
    body = response.json()
    assert body["trainable"] is True
    assert body["evaluable_explicit_count"] >= 10
    assert body["evaluable_date_count"] == 3
    assert body["evaluable"] is (not bridge_dates)
    assert body["v2_windows"]["schema_version"] == FEATURE_SCHEMA_VERSION
    assert len(body["gates"]) == 7
    blockers = {blocker["code"]: blocker for blocker in body["blockers"]}
    if bridge_dates:
        blocker = blockers["insufficient_independent_date_groups"]
        assert "独立日期组" in blocker["message"]
        assert "1" in blocker["message"] and "3" in blocker["message"]
    else:
        assert "insufficient_independent_date_groups" not in blockers


class TestPartialMatches:
    async def test_windows_no_feedback_no_match(
        self, engine, session_factory, tables,
    ) -> None:
        """Feature windows exist, no feedback → zero eligible."""
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        base = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
        await _seed_windows(telemetry_repo, [
            _v2_window(1, base + timedelta(minutes=i * 5),
                       base + timedelta(minutes=(i + 1) * 5))
            for i in range(15)
        ])

        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        assert body["v2_windows"]["total"] == 15
        assert body["v2_windows"]["eligible_count"] == 0
        assert body["trainable"] is False

    async def test_insufficient_classes(
        self, engine, session_factory, tables,
    ) -> None:
        """>=10 feedback with overlap but only 1 class → not trainable."""
        telemetry_repo = TelemetryRepository(session_factory=session_factory)
        base = datetime(2026, 7, 25, 10, 0, tzinfo=UTC)
        windows = [_v2_window(1, base + timedelta(minutes=i * 5),
                              base + timedelta(minutes=(i + 1) * 5))
                   for i in range(30)]
        await _seed_windows(telemetry_repo, windows)

        fcs_rows: list[dict[str, Any]] = []
        fb_rows: list[dict[str, Any]] = []
        for i in range(25):
            sid = f"of-sess-{i}"
            start = base + timedelta(minutes=i * 5)
            end = start + timedelta(minutes=4)
            fcs_rows.append({
                "id": sid, "user_id": 1,
                "date": "2026-07-25",
                "start_time": start.isoformat(),
                "end_time": end.isoformat(),
                "session_type": "focus",
            })
            fb_rows.append({
                "user_id": 1, "session_id": sid, "label": "focus",
                "score": 4, "task_type": "coding",
                "created_at": start.isoformat(),
            })
        await _seed_focus_sessions(engine, fcs_rows)
        await _seed_feedback(engine, fb_rows)

        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        assert body["v2_windows"]["eligible_count"] >= 10
        assert body["v2_windows"]["matched_focus_count"] >= 5
        assert body["v2_windows"]["matched_distract_count"] == 0
        assert body["trainable"] is False
        assert body["trainable_class_count"] < 2

        gates = {g["key"]: g for g in body["gates"]}
        assert gates["minimum_class_feedback"]["status"] == "failed"


# ── Contract tests ─────────────────────────────────────────────────────────


class TestContract:
    async def test_response_has_all_required_keys(
        self, engine, session_factory, tables,
    ) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        required = {
            "raw_events", "v2_windows", "feedback_labels",
            "trainable", "trainable_window_count", "trainable_class_count",
            "evaluable", "evaluable_explicit_count", "evaluable_date_count",
            "baseline_ready", "current_mode",
            "gates", "blockers", "current_training_job",
        }
        assert set(body.keys()) == required

    async def test_gate_items_have_all_fields(
        self, engine, session_factory, tables,
    ) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        for gate in body["gates"]:
            required = {
                "key", "label", "passed", "status",
                "actual", "threshold", "message", "blocker_code",
            }
            assert set(gate.keys()) == required, f"Gate {gate['key']} missing"

    async def test_blockers_have_code_and_message(
        self, engine, session_factory, tables,
    ) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        body = resp.json()

        for blocker in body["blockers"]:
            assert isinstance(blocker["code"], str)
            assert isinstance(blocker["message"], str)

    async def test_gate_count_is_seven(
        self, engine, session_factory, tables,
    ) -> None:
        app = _make_app(engine, session_factory)
        client = TestClient(app)
        resp = client.get("/api/v1/analytics/training-readiness")
        assert resp.status_code == 200
        assert len(resp.json()["gates"]) == 7


# ── Training-report gate overrides ─────────────────────────────────────────


class TestTrainingReportGateOverride:
    """Real post-training values replace hard-coded not_implemented gates."""

    @staticmethod
    def _service_with_report(report: dict[str, Any] | None) -> Any:
        from mindflow.services.training_readiness_service import (
            TrainingReadinessService,
        )

        svc = TrainingReadinessService.__new__(TrainingReadinessService)
        svc._training_report = report
        return svc

    def test_no_report_returns_empty(self) -> None:
        svc = self._service_with_report(None)
        assert svc._report_gate_override() == {}

    def test_report_exposes_real_gate_values(self) -> None:
        report = {
            "quality_gate": {
                "checks": {
                    "balanced_accuracy": False,
                    "minority_f1": False,
                    "calibration_better_than_rule": False,
                    "stable_date_folds": False,
                },
            },
            "evaluation": {
                "candidate": {
                    "balanced_accuracy": 0.336492,
                    "minority_f1": 0.083102,
                    "brier_score": 0.37435,
                },
                "rule_baseline": {"brier_score": 0.314571},
                "fold_stability": {
                    "passed": False,
                    "min_balanced_accuracy": 0.224359,
                    "range": 0.406583,
                    "min_test_size": 117,
                },
            },
        }
        overrides = self._service_with_report(report)._report_gate_override()

        assert set(overrides) == {
            "balanced_accuracy",
            "minority_f1",
            "calibration_better_than_rule",
            "stable_date_folds",
        }
        assert overrides["balanced_accuracy"][0] == "failed"
        assert "0.336" in overrides["balanced_accuracy"][1]
        assert overrides["minority_f1"][0] == "failed"
        assert "0.083" in overrides["minority_f1"][1]
        assert overrides["calibration_better_than_rule"][0] == "failed"
        assert "0.374" in overrides["calibration_better_than_rule"][1]
        assert overrides["stable_date_folds"][0] == "failed"
        assert "0.224" in overrides["stable_date_folds"][1]

    @staticmethod
    def _passing_report() -> dict[str, Any]:
        return {
            "quality_gate": {
                "checks": {
                    "balanced_accuracy": True,
                    "minority_f1": True,
                    "calibration_better_than_rule": True,
                    "calibration_available": True,
                    "stable_date_folds": True,
                },
            },
            "evaluation": {
                "status": "evaluated",
                "calibration": {"method": "sigmoid", "status": "fitted"},
                "candidate": {
                    "balanced_accuracy": 0.72,
                    "minority_f1": 0.65,
                    "brier_score": 0.18,
                },
                "rule_baseline": {"brier_score": 0.30},
                "fold_stability": {
                    "passed": True,
                    "min_balanced_accuracy": 0.62,
                    "range": 0.18,
                    "min_test_size": 40,
                },
            },
        }
    def test_report_passing_gates_reported_as_passed(self) -> None:
        overrides = self._service_with_report(self._passing_report())._report_gate_override()

        assert overrides["balanced_accuracy"][0] == "passed"
        assert overrides["minority_f1"][0] == "passed"
        assert overrides["calibration_better_than_rule"][0] == "passed"
        assert overrides["stable_date_folds"][0] == "passed"
        for key in overrides:
            assert overrides[key][3] == ""  # no failure message when passed

    @pytest.mark.parametrize(
        ("available", "calibration"),
        [
            (False, {"method": "sigmoid", "status": "fitted"}),
            (None, {"method": "sigmoid", "status": "fitted"}),
            (True, None),
            (True, {"method": "sigmoid", "status": "unavailable"}),
            (True, {"status": "not_requested"}),
            (True, {"method": "sigmoid", "status": "not_requested"}),
        ],
    )
    def test_calibration_fails_closed_despite_passing_brier(
        self, available: bool | None, calibration: dict[str, Any] | None,
    ) -> None:
        report = self._passing_report()
        if available is None:
            del report["quality_gate"]["checks"]["calibration_available"]
        else:
            report["quality_gate"]["checks"]["calibration_available"] = available
        if calibration is None:
            del report["evaluation"]["calibration"]
        else:
            report["evaluation"]["calibration"] = calibration
        overrides = self._service_with_report(report)._report_gate_override()
        gate = overrides["calibration_better_than_rule"]
        assert gate[0] == "failed"
        assert "0.180" in gate[1]
        assert "校准" in gate[3]

    @pytest.mark.parametrize("report", [
        {},
        {"evaluation": {"calibration": {
            "method": "sigmoid", "status": "unavailable", "reason": "no grouped split",
        }}},
    ])
    def test_report_without_metrics_still_reports_missing_calibration(
        self, report: dict[str, Any],
    ) -> None:
        overrides = self._service_with_report(report)._report_gate_override()
        gate = overrides["calibration_better_than_rule"]
        assert gate[0] == "failed"
        assert gate[1] == "-"
        assert "校准" in gate[3]

    @pytest.mark.parametrize("calibration", [
        {"method": "sigmoid", "status": "fitted"},
        {"method": "isotonic", "status": "fitted"},
        {"method": None, "status": "not_requested"},
        {"method": "sigmoid", "status": "not_requested"},
        {"status": "not_requested"},
        {"method": "sigmoid", "status": "unavailable"},
        {},
    ])
    def test_calibration_semantics_match_training(self, calibration: dict[str, Any]) -> None:
        from mindflow.train.v2 import evaluate_v2_quality_gate

        report = self._passing_report()
        report["evaluation"]["calibration"] = calibration
        report["quality_gate"] = evaluate_v2_quality_gate(
            report["evaluation"], explicit_feedback_count=28, explicit_focus_count=14,
            explicit_distract_count=14, distinct_feedback_days=7,
        )
        expected = report["quality_gate"]["checks"]["calibration_available"]
        overrides = self._service_with_report(report)._report_gate_override()
        gate = overrides["calibration_better_than_rule"]
        assert (gate[0] == "passed") is expected
        if calibration.get("status") == "not_requested" and expected:
            assert "未请求校准" in gate[1]

    def test_deployment_calibration_failure_overrides_evaluation_success(self) -> None:
        report = self._passing_report()
        report["classifier"] = {"calibration": {
            "method": "sigmoid", "status": "unavailable", "reason": "deployment split failed",
        }}
        overrides = self._service_with_report(report)._report_gate_override()
        gate = overrides["calibration_better_than_rule"]
        assert gate[0] == "failed"
        assert "deployment split failed" in gate[3]

    def test_not_requested_does_not_bypass_brier_comparison(self) -> None:
        report = self._passing_report()
        report["evaluation"]["calibration"] = {"method": None, "status": "not_requested"}
        report["evaluation"]["candidate"]["brier_score"] = 0.8
        report["quality_gate"]["checks"]["calibration_better_than_rule"] = False
        overrides = self._service_with_report(report)._report_gate_override()
        gate = overrides["calibration_better_than_rule"]
        assert gate[0] == "failed"
        assert "未请求校准" in gate[1]
        assert gate[3] == "候选模型校准不优于规则引擎"

    @pytest.mark.parametrize("available", [True, False, None])
    async def test_http_preserves_seven_gates_and_blocks_unavailable_calibration(
        self, engine, session_factory, tables, monkeypatch, available: bool | None,
    ) -> None:
        from mindflow.api.routes import analytics

        report = self._passing_report()
        if available is None:
            del report["quality_gate"]["checks"]["calibration_available"]
        else:
            report["quality_gate"]["checks"]["calibration_available"] = available
        monkeypatch.setattr(analytics, "_load_training_report", lambda request: report)
        base = datetime(2026, 9, 1, 9, tzinfo=UTC)
        windows, sessions, feedback = [], [], []
        for index in range(28):
            start = base + timedelta(days=index // 4, minutes=10 * (index % 4))
            end = start + timedelta(minutes=5)
            sid = f"calibration-{index}"
            windows.append(_v2_window(1, start, end))
            sessions.append({
                "id": sid, "user_id": 1, "date": start.date().isoformat(),
                "start_time": start.isoformat(), "end_time": end.isoformat(),
            })
            feedback.append({
                "session_id": sid, "user_id": 1,
                "label": "focus" if index % 2 else "distracted",
                "score": 5 if index % 2 else 1,
            })
        await _seed_windows(TelemetryRepository(session_factory=session_factory), windows)
        await _seed_focus_sessions(engine, sessions)
        await _seed_feedback(engine, feedback)

        with TestClient(_make_app(engine, session_factory)) as client:
            response = client.get("/api/v1/analytics/training-readiness")
        assert response.status_code == 200
        body = response.json()
        assert body["trainable"] is True
        assert body["v2_windows"]["schema_version"] == FEATURE_SCHEMA_VERSION
        gates = {gate["key"]: gate for gate in body["gates"]}
        assert len(gates) == 7
        assert gates["minimum_days"]["threshold"] == ">= 7"
        assert gates["minimum_days"]["actual"] == "7"
        assert sum(gate["passed"] for gate in gates.values()) == (7 if available else 6)
        if not available:
            gate = gates["calibration_better_than_rule"]
            assert gate["status"] == "failed"
            assert {"code": gate["blocker_code"], "message": gate["message"]} in body["blockers"]
