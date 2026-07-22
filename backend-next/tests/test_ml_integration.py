"""Integration tests for ML model wiring in the evidence pipeline.

Covers:
  - EvidenceBundleBuilder with mock ML models produces ML items.
  - EvidenceBundleBuilder without ML models falls back to rule-only.
  - ML inference failure degrades gracefully to rule-based only.
  - GET /api/v1/analytics/model-status reports loaded vs unloaded.
  - BehaviorFeatureExtractor delegates title analysis to domain/features.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes.analytics import router as analytics_router
from mindflow.domain.events import make_event
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
    activity_events,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
    intervention_logs,
)
from mindflow.services.evidence_service import EvidenceBundleBuilder, baseline_models_metadata
from mindflow.train.features import BehaviorFeatureExtractor


def _utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


_BASE_TS = _utc("2026-07-18T08:00:00")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(count: int, process_name: str = "Code.exe") -> list[dict[str, Any]]:
    """Generate *count* non-idle events spaced 2 min apart (span > 30 min)."""
    events = []
    for i in range(count):
        events.append(
            {
                "user_id": 1,
                "timestamp_utc": _BASE_TS + timedelta(minutes=i * 2),
                "duration_s": 60.0,
                "process_name": process_name,
                "is_idle": False,
            }
        )
    return events


async def _insert_events(repo: SQLAlchemyActivityRepository, events: list[dict]) -> None:
    for ev in events:
        await repo.append_event(make_event(**ev))


def _make_mock_model_manager(
    focus_proba: float = 0.75,
    cluster_id: int = 0,
    predict_proba_side_effect: Exception | None = None,
    predict_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock ModelManager with classifier and clustering.

    The mock dynamically matches the input matrix shape so it works
    regardless of how many feature rows the extractor produces.
    """
    mm = MagicMock()
    mm.classifier = MagicMock()
    mm.clustering = MagicMock()

    if predict_proba_side_effect is not None:
        mm.classifier.predict_proba.side_effect = predict_proba_side_effect
    else:
        def _dynamic_proba(matrix):
            n = matrix.shape[0]
            return np.column_stack([np.full(n, 1.0 - focus_proba), np.full(n, focus_proba)])
        mm.classifier.predict_proba.side_effect = _dynamic_proba

    if predict_side_effect is not None:
        mm.clustering.predict.side_effect = predict_side_effect
    else:
        def _dynamic_cluster(matrix):
            n = matrix.shape[0]
            return np.full(n, cluster_id, dtype=int)
        mm.clustering.predict.side_effect = _dynamic_cluster

    mm.current_version_tag = "20260718"
    mm.list_versions.return_value = ["20260718"]
    return mm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def all_tables(engine):
    """Create all tables needed by the evidence builder."""
    async with engine.begin() as conn:
        await conn.run_sync(activity_events.metadata.create_all)
        await conn.run_sync(intervention_logs.metadata.create_all)
        await conn.run_sync(baseline_models_metadata.create_all)
    yield


@pytest.fixture
async def activity_repo(session_factory, all_tables):
    return SQLAlchemyActivityRepository(session_factory=session_factory, pulsetime_s=10)


@pytest.fixture
async def intervention_repo(session_factory, all_tables):
    return InterventionLogRepository(session_factory=session_factory)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvidenceBundleWithMLModels:
    """Verify ML items appear when ModelManager is provided."""

    async def test_evidence_bundle_with_ml_models(
        self, activity_repo, intervention_repo, session_factory
    ):
        """ML items (ml_focus_probability, ml_behavior_cluster) appear in the bundle."""
        events = _make_events(15, process_name="Code.exe")
        await _insert_events(activity_repo, events)

        events = _make_events(20, process_name="Code.exe")
        await _insert_events(activity_repo, events)

        mock_mm = _make_mock_model_manager(focus_proba=0.82, cluster_id=0)

        builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            model_manager=mock_mm,
        )

        bundle = await builder.build(
            user_id=1,
            window_start=_BASE_TS,
            window_end=_BASE_TS + timedelta(minutes=60),
        )

        metrics = [item.metric for item in bundle.items]

        # Rule-based items should still be present
        assert "focus_score" in metrics
        assert "switch_rate" in metrics

        # ML enrichment items should be present
        assert "ml_focus_probability" in metrics
        assert "ml_behavior_cluster" in metrics

        # Verify ML focus probability value
        ml_focus = next(i for i in bundle.items if i.metric == "ml_focus_probability")
        assert ml_focus.source == "rf_classifier"
        assert 0.0 <= float(ml_focus.value) <= 1.0

        # Verify ML cluster item
        ml_cluster = next(i for i in bundle.items if i.metric == "ml_behavior_cluster")
        assert ml_cluster.source == "dbscan_clustering"
        assert ml_cluster.severity == "info"

        # classifier.predict_proba should have been called at least once
        mock_mm.classifier.predict_proba.assert_called()
        # clustering.predict should have been called at least once
        mock_mm.clustering.predict.assert_called()


class TestEvidenceBundleWithoutMLModels:
    """Verify rule-only fallback when no ModelManager is provided."""

    async def test_evidence_bundle_without_ml_models(
        self, activity_repo, intervention_repo, session_factory
    ):
        """Bundle works with only rule-based items when model_manager=None."""
        events = _make_events(20, process_name="Code.exe")
        await _insert_events(activity_repo, events)

        builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            model_manager=None,
        )

        bundle = await builder.build(
            user_id=1,
            window_start=_BASE_TS,
            window_end=_BASE_TS + timedelta(minutes=60),
        )

        metrics = [item.metric for item in bundle.items]

        # Rule-based items should be present
        assert "focus_score" in metrics
        assert "switch_rate" in metrics
        assert "longest_block" in metrics
        assert "top_apps" in metrics

        # No ML items
        assert "ml_focus_probability" not in metrics
        assert "ml_behavior_cluster" not in metrics


class TestMLInferenceFailureGracefulDegradation:
    """Verify ML failure does not break the evidence pipeline."""

    async def test_ml_inference_failure_graceful_degradation(
        self, activity_repo, intervention_repo, session_factory
    ):
        """When classifier raises, bundle still contains rule-based items only."""
        events = _make_events(20, process_name="Code.exe")
        await _insert_events(activity_repo, events)

        # Both classifier and clustering raise to verify full ML degradation
        mock_mm = _make_mock_model_manager(
            predict_proba_side_effect=RuntimeError("model corrupted"),
            predict_side_effect=RuntimeError("model corrupted"),
        )

        builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            model_manager=mock_mm,
        )

        bundle = await builder.build(
            user_id=1,
            window_start=_BASE_TS,
            window_end=_BASE_TS + timedelta(minutes=60),
        )

        metrics = [item.metric for item in bundle.items]

        # Rule-based items should still be present
        assert "focus_score" in metrics
        assert "switch_rate" in metrics
        assert "longest_block" in metrics
        assert "top_apps" in metrics

        # ML items should NOT be present (degraded gracefully)
        assert "ml_focus_probability" not in metrics
        assert "ml_behavior_cluster" not in metrics


class TestModelStatusEndpoint:
    """Verify the model-status endpoint returns correct status."""

    async def test_model_status_endpoint(self, engine, session_factory):
        """GET /api/v1/analytics/model-status reports correct loaded state."""
        async with engine.begin() as conn:
            await conn.run_sync(activity_events.metadata.create_all)

        # --- Case 1: models not loaded (model_manager=None) ---
        app_none = FastAPI()
        register_exception_handlers(app_none)
        app_none.include_router(analytics_router, prefix="/api/v1")
        app_none.state.model_manager = None

        client_none = TestClient(app_none)
        resp_none = client_none.get("/api/v1/analytics/model-status")
        assert resp_none.status_code == 200
        body_none = resp_none.json()
        assert body_none["loaded"] is False
        assert body_none["mode"] == "rule_engine_only"
        assert "ML models not available" in body_none["message"]

        # --- Case 2: models loaded (mock ModelManager) ---
        mock_mm = _make_mock_model_manager()

        app_loaded = FastAPI()
        register_exception_handlers(app_loaded)
        app_loaded.include_router(analytics_router, prefix="/api/v1")
        app_loaded.state.model_manager = mock_mm

        client_loaded = TestClient(app_loaded)
        resp_loaded = client_loaded.get("/api/v1/analytics/model-status")
        assert resp_loaded.status_code == 200
        body_loaded = resp_loaded.json()
        assert body_loaded["loaded"] is True
        assert body_loaded["mode"] == "ml_enriched"
        assert body_loaded["version"] == "20260718"
        assert body_loaded["available_versions"] == ["20260718"]
        assert "enriched" in body_loaded["message"]


class TestFeatureExtractorDelegatesToDomain:
    """Verify BehaviorFeatureExtractor uses domain/features.py functions."""

    def test_feature_extractor_delegates_to_domain(self):
        """TitleAnalyzer calls domain.features.title_features, converting booleans to floats."""
        extractor = BehaviorFeatureExtractor(window_minutes=30)
        result = extractor.title_analyzer.analyze("main.py - VS Code")

        # Code editor title should produce is_code_editor=1.0
        assert result["is_code_editor"] == 1.0
        # Other flags should be 0.0
        assert result["is_document"] == 0.0
        assert result["is_meeting"] == 0.0

    def test_feature_extractor_title_delegation_url(self):
        """TitleAnalyzer correctly delegates URL detection to domain."""
        extractor = BehaviorFeatureExtractor()
        # Use .org to avoid false-positive on ".c" substring in ".com"
        result = extractor.title_analyzer.analyze("https://example.org/page")

        assert result["is_browser"] == 1.0
        assert result["is_code_editor"] == 0.0

    def test_feature_extractor_title_delegation_meeting(self):
        """TitleAnalyzer correctly delegates meeting detection to domain."""
        extractor = BehaviorFeatureExtractor()
        result = extractor.title_analyzer.analyze("Daily standup - Zoom Meeting")

        assert result["is_meeting"] == 1.0

    def test_feature_extractor_empty_title(self):
        """TitleAnalyzer handles empty title gracefully."""
        extractor = BehaviorFeatureExtractor()
        result = extractor.title_analyzer.analyze("")
        assert all(v == 0.0 for v in result.values())

    def test_feature_extractor_get_feature_names(self):
        """BehaviorFeatureExtractor exposes 14 feature names."""
        extractor = BehaviorFeatureExtractor()
        names = extractor.get_feature_names()
        assert len(names) == 14
        assert "unique_app_count" in names
        assert "title_code_ratio" in names

    def test_feature_extractor_extract_session_features_uses_domain(self):
        """extract_session_features produces features when given valid events."""
        extractor = BehaviorFeatureExtractor(window_minutes=30)

        # Create events spanning >30 min so the extractor produces ≥2 boundaries
        events = []
        for i in range(20):
            ev = make_event(
                user_id=1,
                timestamp_utc=_BASE_TS + timedelta(minutes=i * 2),
                duration_s=60.0,
                process_name="Code.exe",
                is_idle=False,
            )
            events.append(ev)

        rows = extractor.extract_session_features(events)
        # Should produce at least one feature row
        assert len(rows) >= 1
        # Each row should have the 14 feature keys
        for name in BehaviorFeatureExtractor.FEATURE_NAMES:
            assert name in rows[0]
