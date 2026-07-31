"""Tests for FocusPredictionService — unified ML prediction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from mindflow.domain.prediction import (
    MIN_COVERAGE_RATIO,
    STALE_THRESHOLD_S,
    FocusPrediction,
    FocusPredictionStatus,
)
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.train.v2 import V2_FEATURE_NAMES


def _make_mock_model_manager(focus_proba: float = 0.75) -> MagicMock:
    """Create a mock ModelManager for testing."""
    mm = MagicMock()
    mm.current_version_tag = "20260726_v2"
    mm.classifier = MagicMock()
    mm.classifier._is_fitted = True

    def _dynamic_proba(matrix):
        n = matrix.shape[0]
        return np.column_stack([np.full(n, 1.0 - focus_proba), np.full(n, focus_proba)])
    mm.classifier.predict_proba.side_effect = _dynamic_proba
    mm.classifier.get_feature_importance.return_value = {
        "idle_ratio": 0.3,
        "app_switch_count": 0.25,
        "keypress_rate_per_min": 0.2,
        "longest_segment_ratio": 0.15,
        "active_seconds_ratio": 0.1,
    }
    return mm


def _make_feature_window(
    window_start: datetime,
    window_end: datetime | None = None,
    features: dict | None = None,
) -> dict:
    if window_end is None:
        window_end = window_start + timedelta(minutes=5)
    if features is None:
        features = {
            "app_switch_count": 2,
            "domain_switch_count": 1,
            "longest_segment_ratio": 0.8,
            "idle_ratio": 0.05,
            "keypress_rate_per_min": 30.0,
            "mouse_click_rate_per_min": 5.0,
            "scroll_rate_per_min": 10.0,
            "mouse_distance_per_min": 200.0,
            "input_active_ratio": 0.9,
            "interaction_bursts_per_min": 0.5,
            "click_key_ratio": 0.2,
            "browser_ratio": 0.3,
            "audible_browser_ratio": 0.0,
            "active_seconds_ratio": 0.95,
            "top_app_ratio": 0.7,
            "top_domain_ratio": 0.5,
            "interaction_interval_mean_s": 2.0,
            "interaction_interval_std_s": 1.5,
            "interaction_interval_cv": 0.75,
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "weekday_sin": 0.0,
            "weekday_cos": 1.0,
            "task_type_code": 0.0,
        }
    return {
        "window_start_utc": window_start.isoformat(),
        "window_end_utc": window_end.isoformat(),
        "features_json": str(features).replace("'", '"'),
        "feature_schema_version": 3,
    }


class TestFocusPredictionService:
    """Tests for the unified ML prediction service."""

    async def test_no_model_returns_no_model_status(self):
        """When model_manager is None, status is no_model."""
        repo = MagicMock()
        service = FocusPredictionService(telemetry_repository=repo)
        result = await service.predict_latest(user_id=1)
        assert result.status == "no_model"
        assert result.focus_probability is None

    async def test_classifier_not_fitted_returns_no_model(self):
        """When classifier is not fitted, treat as no_model."""
        repo = MagicMock()
        mm = MagicMock()
        mm.current_version_tag = None
        mm.classifier._is_fitted = False
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1)
        assert result.status == "no_model"

    async def test_no_data_returns_no_data_status(self):
        """When no feature windows found, status is no_data."""
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=[])
        mm = _make_mock_model_manager()
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1)
        assert result.status == "no_data"
        assert result.focus_probability is None

    async def test_predict_latest_returns_ready_status(self):
        """When data and model are available, returns ready prediction."""
        now = datetime.now(UTC)
        # 24 windows over 2 hours = full coverage
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i))
            for i in range(23, 0, -1)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.82)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "ready"
        assert result.focus_probability is not None
        assert result.focus_probability > 0.5
        assert result.window_count == 23
        assert len(result.top_factors) == 3
        assert result.model_version == "20260726_v2"
        assert result.feature_schema_version == 3

    async def test_predict_range_returns_ready(self):
        """Predict for a specific time range works."""
        now = datetime.now(UTC)
        start = now - timedelta(hours=2)
        end = now
        windows = [
            _make_feature_window(start + timedelta(minutes=5 * i))
            for i in range(23)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.65)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_range(user_id=1, start=start, end=end, now=now)
        assert result.status == "ready"
        assert 0.0 <= result.focus_probability <= 1.0
        assert result.window_count >= 20

    async def test_stale_data_detection(self):
        """Data older than STALE_THRESHOLD_S is reported as stale."""
        now = datetime.now(UTC)
        # All windows end at least 30 minutes before now (> STALE_THRESHOLD_S)
        base = now - timedelta(minutes=150)
        windows = [
            _make_feature_window(base + timedelta(minutes=5 * i))
            for i in range(24)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "stale"
        assert result.data_age_s is not None
        assert result.data_age_s > STALE_THRESHOLD_S
        assert result.focus_probability is not None  # Still returns value

    async def test_schema_mismatch_handling(self):
        """Missing features in feature windows produce still usable prediction."""
        now = datetime.now(UTC)
        bad_features = {"app_switch_count": 2, "idle_ratio": 0.1}
        # Need enough windows to pass coverage check
        windows = [
            {
                "window_start_utc": (now - timedelta(minutes=5 * (24 - i))).isoformat(),
                "window_end_utc": (now - timedelta(minutes=5 * (24 - i - 1))).isoformat(),
                "features_json": str(bad_features).replace("'", '"'),
                "feature_schema_version": 3,
            }
            for i in range(24)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        # Expect ready since missing features default to 0.0
        assert result.status == "ready"

    async def test_inference_error_from_empty_windows(self):
        """All windows failing to parse returns no_data."""
        now = datetime.now(UTC)
        windows = [
            {
                "window_start_utc": (now - timedelta(minutes=5)).isoformat(),
                "window_end_utc": now.isoformat(),
                "features_json": "not valid json",
                "feature_schema_version": 3,
            },
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "no_data"

    async def test_attach_model_manager_lazy(self):
        """Model manager can be attached after construction."""
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=[])
        service = FocusPredictionService(telemetry_repository=repo)
        assert (await service.predict_latest()).status == "no_model"

        mm = _make_mock_model_manager()
        service.attach_model_manager(mm)
        assert service._model_manager is not None

    async def test_check_health_returns_dict(self):
        """Health check returns structured status."""
        repo = MagicMock()
        mm = _make_mock_model_manager()
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        health = await service.check_health()
        assert "status" in health
        assert "model_version" in health
        assert "feature_schema_version" in health
        assert health["status"] == "ready"
        assert health["feature_schema_version"] == 3

    async def test_predict_range_filters_correctly(self):
        """predict_range only returns windows within the requested range."""
        now = datetime.now(UTC)
        all_windows = [
            _make_feature_window(now - timedelta(hours=3)),  # outside
            _make_feature_window(now - timedelta(hours=1)),  # inside
            _make_feature_window(now - timedelta(minutes=30)),  # inside
            _make_feature_window(now + timedelta(hours=1)),  # outside (future)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=all_windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_range(
            user_id=1,
            start=now - timedelta(hours=2),
            end=now,
            now=now,
        )
        assert result.window_count == 2


class TestFocusPredictionStatusContract:
    """Characterization: pin the six-status FocusPrediction contract.

    Locks the domain Literal set and the service's per-status payload
    semantics (probability null for non-ready/coerced paths, numeric for
    ready/stale) so later contract work cannot silently add or re-map
    status values.
    """

    def test_domain_defaults(self) -> None:
        prediction = FocusPrediction()
        assert prediction.status == "no_model"
        assert prediction.focus_probability is None
        assert prediction.feature_schema_version == 3
        assert prediction.window_count == 0
        assert prediction.reason == ""

    def test_all_six_statuses_constructible(self) -> None:
        statuses: list[FocusPredictionStatus] = [
            "ready",
            "no_model",
            "no_data",
            "stale",
            "schema_mismatch",
            "inference_error",
        ]
        for status in statuses:
            prediction = FocusPrediction(status=status)
            assert prediction.status == status

    def test_domain_is_frozen(self) -> None:
        prediction = FocusPrediction()
        with pytest.raises(FrozenInstanceError):
            prediction.__setattr__("status", "ready")

    async def test_predict_proba_failure_is_inference_error(self) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager()
        mm.classifier.predict_proba.side_effect = RuntimeError("model crash")
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "inference_error"
        assert result.focus_probability is None
        assert result.window_count == 23

    async def test_db_query_failure_is_inference_error(self) -> None:
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(side_effect=RuntimeError("db down"))
        mm = _make_mock_model_manager()
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1)
        assert result.status == "inference_error"
        assert result.focus_probability is None

    async def test_model_feature_names_mismatch_is_schema_mismatch(self) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager()
        mm.classifier.feature_names_ = ["old_feature"]
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "schema_mismatch"
        assert result.focus_probability is None
        assert result.window_count == 23

    async def test_low_coverage_returns_stale_with_probability(self) -> None:
        now = datetime.now(UTC)
        # One window ending exactly at now -> coverage 1/24 below the 0.3 floor.
        windows = [_make_feature_window(now - timedelta(minutes=5))]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "stale"
        assert result.coverage_ratio < MIN_COVERAGE_RATIO
        assert result.focus_probability is not None

    async def test_realistic_list_feature_names_is_ready(self) -> None:
        """A classifier storing ``feature_names_`` as a list must be ready.

        The training pipeline passes ``list(V2_FEATURE_NAMES)`` (a list) to
        ``FocusClassifier.fit``, so the in-memory attribute is a list. The
        schema guard must compare names by content, not by container type.
        """
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager()
        mm.classifier.feature_names_ = list(V2_FEATURE_NAMES)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "ready"
        assert result.focus_probability is not None

    async def test_ready_payload_field_contract(self) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.82)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "ready"
        assert result.focus_probability is not None
        assert 0.0 <= result.focus_probability <= 1.0
        assert result.coverage_ratio <= 1.0
        assert result.feature_schema_version == 3
        assert len(result.top_factors) == 3
        assert result.model_version == "20260726_v2"
