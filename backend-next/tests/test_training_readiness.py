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
        "feature_schema_version": 2,
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
        assert gates["calibration_better_than_rule"]["status"] == "not_implemented"
        assert gates["stable_date_folds"]["status"] == "not_implemented"

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

        # Gates for minimums should pass
        gates = {g["key"]: g for g in body["gates"]}
        assert gates["minimum_days"]["status"] == "passed"
        assert gates["minimum_explicit_feedback"]["status"] == "passed"
        assert gates["minimum_class_feedback"]["status"] == "passed"


# ── Tests: partial / insufficient ──────────────────────────────────────────


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
