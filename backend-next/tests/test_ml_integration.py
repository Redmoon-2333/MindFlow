"""Integration tests for ML model wiring in the evidence pipeline.

Covers:
  - EvidenceBundleBuilder with mock ML models produces ML items.
  - EvidenceBundleBuilder without ML models falls back to rule-only.
  - ML inference failure degrades gracefully to rule-based only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

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


def _make_mock_prediction_service(
    focus_proba: float = 0.75,
) -> MagicMock:
    """Create a mock FocusPredictionService returning predictable predictions."""
    from mindflow.domain.prediction import FocusPrediction

    ps = MagicMock()

    async def _predict_range(user_id, start, end):
        return FocusPrediction(
            status="ready",
            focus_probability=focus_proba,
            uncertainty=round(1.0 - abs(2.0 * focus_proba - 1.0), 6),
            distracted_window_ratio=round(1.0 - focus_proba, 6),
            window_count=4,
            coverage_ratio=0.8,
            data_age_s=60.0,
            model_version="20260718_v2",
            feature_schema_version=3,
            top_factors=[
                {"feature": "idle_ratio", "value": 0.05, "importance": 0.3},
                {"feature": "app_switch_count", "value": 3.0, "importance": 0.25},
                {"feature": "keypress_rate_per_min", "value": 40.0, "importance": 0.2},
            ],
            explanation_method="global_importance_times_observation",
            reason="",
        )
    ps.predict_range = _predict_range

    async def _predict_latest(user_id):
        return await _predict_range(user_id, None, None)
    ps.predict_latest = _predict_latest

    return ps


_make_mock_prediction_service.__no_coverage__ = True


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
        """ML items (ml_focus_probability, ml_distracted_window_ratio) appear in the bundle."""
        events = _make_events(15, process_name="Code.exe")
        await _insert_events(activity_repo, events)

        events = _make_events(20, process_name="Code.exe")
        await _insert_events(activity_repo, events)

        mock_ps = _make_mock_prediction_service(focus_proba=0.82)

        builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            prediction_service=mock_ps,
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

        # v2 ML enrichment items should be present
        assert "ml_focus_probability" in metrics
        assert "ml_distracted_window_ratio" in metrics
        assert "ml_uncertainty" in metrics
        assert "ml_feature_coverage" in metrics

        # Verify ML focus probability value
        ml_focus = next(i for i in bundle.items if i.metric == "ml_focus_probability")
        assert ml_focus.source == "rf_classifier_v2"
        assert 0.0 <= float(ml_focus.value) <= 1.0

        # Verify ml_distracted_window_ratio
        ml_distracted = next(i for i in bundle.items if i.metric == "ml_distracted_window_ratio")
        assert ml_distracted.source == "rf_classifier_v2"

        # Verify ml_uncertainty
        ml_uncertainty = next(i for i in bundle.items if i.metric == "ml_uncertainty")
        assert ml_uncertainty.source == "rf_classifier_v2"


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
            prediction_service=None,
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
        """When prediction service returns error, bundle contains rule-based items only."""
        from mindflow.domain.prediction import FocusPrediction

        events = _make_events(20, process_name="Code.exe")
        await _insert_events(activity_repo, events)

        # Mock prediction service that returns inference_error
        mock_ps = MagicMock()

        async def _predict_range_error(user_id, start, end):
            return FocusPrediction(
                status="inference_error",
                focus_probability=None,
                uncertainty=None,
                window_count=0,
                coverage_ratio=0.0,
                data_age_s=None,
                model_version=None,
                feature_schema_version=3,
                top_factors=[],
                explanation_method="",
                reason="模拟推理失败",
            )
        mock_ps.predict_range = _predict_range_error

        builder = EvidenceBundleBuilder(
            activity_repo=activity_repo,
            intervention_repo=intervention_repo,
            session_factory=session_factory,
            prediction_service=mock_ps,
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
