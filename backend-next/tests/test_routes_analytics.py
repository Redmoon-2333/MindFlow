"""Tests for /api/v1/analytics endpoints.

Covers:
  - GET /analytics/patterns: pattern analysis with/without data
  - GET /analytics/baseline: placeholder response
  - GET /analytics/profile: behavioural profile with/without data
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.analytics import router as analytics_router
from mindflow.domain.baseline import BaselineModel
from mindflow.domain.events import make_event
from mindflow.domain.feature_schema import V2_FEATURE_NAMES
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
from mindflow.services import analysis_service as analysis_service_module
from mindflow.services.analysis_service import AnalysisService
from mindflow.train.models import ModelManager


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE = _utc("2026-07-17T08:00:00")


@pytest.fixture
async def seeded_app(engine, session_factory) -> FastAPI:
    """Test app with seed data for analytics."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(baseline_models.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True

    activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    analysis_svc = AnalysisService(
        activity_repo=activity_repo, focus_repo=focus_repo
    )

    # Seed events
    for i in range(20):
        ev = make_event(
            user_id=1,
            timestamp_utc=_BASE + timedelta(seconds=i * 5),
            duration_s=5.0,
            process_name="Code.exe",
        )
        await activity_repo.append_event(ev)

    # Seed sessions
    await focus_repo.save_sessions(1, [
        {
            "date": "2026-07-17",
            "start_time": _utc("2026-07-17T08:00:00").isoformat(),
            "end_time": _utc("2026-07-17T08:30:00").isoformat(),
            "session_type": "focus",
            "dominant_app": "Code.exe",
            "focus_score": 85.0,
            "switch_count": 0,
        },
    ])

    app.state.activity_repository = activity_repo
    app.state.analysis_service = analysis_svc
    app.state.focus_repository = focus_repo
    app.state.baseline_repository = BaselineRepository(session_factory=session_factory)
    return app


@pytest.fixture
async def empty_app(engine, session_factory) -> FastAPI:
    """Test app with no data."""
    async with engine.begin() as conn:
        await conn.run_sync(focus_sessions.metadata.create_all)
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(baseline_models.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.collector_service = None
    app.state.migration_applied = True

    activity_repo = SQLAlchemyActivityRepository(session_factory=session_factory)
    focus_repo = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    analysis_svc = AnalysisService(
        activity_repo=activity_repo, focus_repo=focus_repo
    )

    app.state.activity_repository = activity_repo
    app.state.analysis_service = analysis_svc
    app.state.focus_repository = focus_repo
    app.state.baseline_repository = BaselineRepository(session_factory=session_factory)
    return app


class TestPatterns:
    """Pattern analysis endpoint tests."""

    def test_patterns_success(self, seeded_app, monkeypatch):
        """GET /analytics/patterns should return pattern data."""
        monkeypatch.setattr(
            analysis_service_module,
            "business_today",
            lambda _timezone: date(2026, 7, 17),
        )
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/analytics/patterns")
        assert resp.status_code == 200
        data = resp.json()
        assert "high_switch_periods" in data
        assert "trigger_apps" in data
        assert "heatmap" in data
        assert data["total_sessions"] >= 1

    def test_patterns_empty(self, empty_app):
        """GET /analytics/patterns with no data should return 404."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/analytics/patterns")
        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]


class TestBaseline:
    """Baseline endpoint tests."""

    def test_baseline_not_found_no_data(self, seeded_app):
        """GET /analytics/baseline with no baseline data should return 404."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]

    def test_baseline_not_found_empty(self, empty_app):
        """GET /analytics/baseline with no data should return 404."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]


class TestProfile:
    """Behavioural profile endpoint tests."""

    def test_profile_success(self, seeded_app):
        """GET /analytics/profile should return profile data."""
        client = TestClient(seeded_app)
        resp = client.get("/api/v1/analytics/profile")
        assert resp.status_code == 200
        data = resp.json()
        assert "peak_focus_hours" in data
        assert "top_apps" in data
        assert "avg_focus_block_min" in data
        assert data["total_events_analysed"] >= 1

    def test_profile_empty(self, empty_app):
        """GET /analytics/profile with no data should return 404."""
        client = TestClient(empty_app)
        resp = client.get("/api/v1/analytics/profile")
        assert resp.status_code == 404
        data = resp.json()
        assert "not-found" in data["type"]


def test_model_status_reports_unready_classifier(tmp_path) -> None:
    app = FastAPI()
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.v2_model_manager = ModelManager(models_dir=tmp_path / "models" / "v2")

    response = TestClient(app).get("/api/v1/analytics/model-status")

    assert response.status_code == 200
    data = response.json()
    assert data["loaded"] is True
    assert data["ready"] is False
    assert data["mode"] == "rule_engine_only"
    assert "classifier_not_fitted" in data["reasons"]


def test_model_status_reports_shadow_mode_without_active_model() -> None:
    app = FastAPI()
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.v2_model_manager = None
    app.state.v2_training_mode = "shadow"

    data = TestClient(app).get("/api/v1/analytics/model-status").json()

    assert data["ready"] is False
    assert data["mode"] == "shadow"


def test_model_status_prefers_ready_v2_model() -> None:
    class ReadyV2Manager:
        current_version_tag = "20260724"

        @staticmethod
        def readiness_status():
            return {"ready": True, "reasons": []}

        @staticmethod
        def list_versions():
            return ["20260724"]

    v2_manager = ReadyV2Manager()
    app = FastAPI()
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.model_manager = None
    app.state.v2_model_manager = v2_manager

    data = TestClient(app).get("/api/v1/analytics/model-status").json()

    assert data["mode"] == "ready"
    assert data["feature_schema_version"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline weighted-mean route tests
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def baseline_with_data_app(engine, session_factory) -> FastAPI:
    """Test app with a persisted baseline containing unequal bucket counts."""
    async with engine.begin() as conn:
        await conn.run_sync(baseline_models.metadata.create_all)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(analytics_router, prefix="/api/v1")
    app.state.collector_service = None

    # Build a BaselineModel with two unequal V2 buckets:
    #   bucket (9, 0): 10 samples, app_switch_count mean=20, active_seconds_ratio mean=0.8
    #   bucket (14, 3): 30 samples, app_switch_count mean=10, active_seconds_ratio mean=0.4
    # Weighted app_switch_count = (10*20 + 30*10) / 40 = 12.5
    # Weighted active_seconds_ratio = (10*0.8 + 30*0.4) / 40 = 0.5
    model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
    model.update([
        *_rows(10, hour=9, dow=0, app_switch_count=20.0, active_seconds_ratio=0.8),
        *_rows(30, hour=14, dow=3, app_switch_count=10.0, active_seconds_ratio=0.4),
    ])

    app.state.baseline_repository = BaselineRepository(session_factory=session_factory)

    # Persist the baseline directly into the DB
    import sqlalchemy as sa

    async with session_factory() as session:
        await session.execute(
            sa.insert(baseline_models).values(
                id="test-baseline-1",
                user_id=1,
                model_json=json.dumps(model.to_dict(), ensure_ascii=False),
                training_events_count=40,
                created_at=model.created_at.isoformat(),
                updated_at=model.updated_at.isoformat(),
            )
        )
        await session.commit()

    return app


def _rows(
    n: int,
    hour: int,
    dow: int,
    app_switch_count: float = 15.0,
    active_seconds_ratio: float = 0.5,
) -> list[dict]:
    """Generate *n* V2 feature-window rows bucketed at local (hour, dow).

    ``window_start_utc`` is chosen so the Asia/Shanghai local time is
    (hour, dow); ``features_json`` carries the flattened V2 vocabulary.
    """
    local = datetime.combine(
        date(2026, 7, 27), time(hour=hour), tzinfo=ZoneInfo("Asia/Shanghai"),
    )
    shift = (dow - local.weekday()) % 7
    start = (local + timedelta(days=shift)).astimezone(UTC)
    features = {name: 0.0 for name in V2_FEATURE_NAMES}
    features.update({
        "app_switch_count": app_switch_count,
        "active_seconds_ratio": active_seconds_ratio,
    })
    return [
        {"window_start_utc": start, "features_json": json.dumps(features)}
        for _ in range(n)
    ]


class TestBaselineWeightedMean:
    """Baseline route exposes a typed, sample-weighted V2 summary.

    Todo 10 contract: the response is the Pydantic ``BaselineSummary`` schema
    carrying the canonical V2 means (``mean_app_switch_count``,
    ``mean_active_seconds_ratio``, ``mean_idle_ratio``) with one-to-one
    compatibility aliases only (``switch_frequency == mean_app_switch_count``,
    ``productivity_ratio == mean_active_seconds_ratio``). ``features`` is
    exactly the 24-name V2 vocabulary and an empty repository stays 404.
    """

    def test_baseline_exposes_v2_feature_vocabulary(
        self,
        baseline_with_data_app,
    ):
        client = TestClient(baseline_with_data_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["features"] == list(V2_FEATURE_NAMES)
        assert data["total_samples"] == 40 * len(V2_FEATURE_NAMES)

    def test_baseline_canonical_means_are_sample_weighted(
        self,
        baseline_with_data_app,
    ):
        """Canonical V2 means are exposed with the weighted values.

        Weighted app_switch_count = (10*20 + 30*10) / 40 = 12.5 (bucket 9,0
        has 10 samples with mean 20; bucket 14,3 has 30 samples with mean
        10), weighted active_seconds_ratio = (10*0.8 + 30*0.4) / 40 = 0.5.
        ``idle_ratio`` is 0.0 in every fixture row so its mean is 0.0.
        """
        client = TestClient(baseline_with_data_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mean_app_switch_count"] == 12.5
        assert data["mean_app_switch_count"] != 15.0  # not the unweighted mean
        assert data["mean_active_seconds_ratio"] == 0.5
        assert data["mean_idle_ratio"] == 0.0

    def test_baseline_compat_aliases_equal_canonical_means(
        self,
        baseline_with_data_app,
    ):
        """Compatibility aliases are exact copies of the canonical means."""
        client = TestClient(baseline_with_data_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["switch_frequency"] == data["mean_app_switch_count"] == 12.5
        assert data["productivity_ratio"] == data["mean_active_seconds_ratio"] == 0.5

    def test_baseline_response_matches_typed_schema(
        self,
        baseline_with_data_app,
    ):
        """The 200 body is exactly the BaselineSummary schema field set.

        No legacy V1 fields (``avg_focus_min``, ``avg_switches_per_day``,
        ``productivity_score``) leak into the typed contract.
        """
        client = TestClient(baseline_with_data_app)
        resp = client.get("/api/v1/analytics/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert set(data) == {
            "user_id",
            "created_at",
            "updated_at",
            "total_days",
            "total_samples",
            "features",
            "mean_app_switch_count",
            "mean_active_seconds_ratio",
            "mean_idle_ratio",
            "switch_frequency",
            "productivity_ratio",
        }
        assert "avg_focus_min" not in data
        assert "avg_switches_per_day" not in data
        assert "productivity_score" not in data

    def test_baseline_model_weighted_mean_is_not_unweighted(self) -> None:
        """Model-level weighted means stay correct on V2 features.

        Weighted app_switch_count = (10*20 + 30*10) / 40 = 12.5, while the
        naive unweighted mean of bucket means would be (20 + 10) / 2 = 15.0.
        """
        model = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        model.update([
            *_rows(10, hour=9, dow=0, app_switch_count=20.0, active_seconds_ratio=0.8),
            *_rows(30, hour=14, dow=3, app_switch_count=10.0, active_seconds_ratio=0.4),
        ])
        assert model.overall_mean("app_switch_count") == 12.5
        assert model.overall_mean("app_switch_count") != 15.0
        assert model.overall_mean("active_seconds_ratio") == 0.5
