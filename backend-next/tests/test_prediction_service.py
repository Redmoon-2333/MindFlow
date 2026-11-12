"""Tests for FocusPredictionService — unified ML prediction."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.domain.prediction import (
    MIN_COVERAGE_RATIO,
    STALE_THRESHOLD_S,
    FocusPrediction,
    FocusPredictionStatus,
)
from mindflow.services.prediction_service import (
    _MAX_PREDICT_WINDOWS,
    FocusPredictionService,
)
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
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
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
        repo.list_feature_windows_in_range = AsyncMock(return_value=[])
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
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.82)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "ready"
        assert result.focus_probability is not None
        assert result.focus_probability > 0.5
        assert result.window_count == 23
        assert len(result.top_factors) == 3
        assert result.model_version == "20260726_v2"
        assert result.feature_schema_version == FEATURE_SCHEMA_VERSION

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
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
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
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
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
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            }
            for i in range(24)
        ]
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
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
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            },
        ]
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "no_data"

    async def test_attach_model_manager_lazy(self):
        """Model manager can be attached after construction."""
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=[])
        service = FocusPredictionService(telemetry_repository=repo)
        assert (await service.predict_latest()).status == "no_model"

        mm = _make_mock_model_manager()
        service.attach_model_manager(mm)
        assert service._model_manager is not None

    async def test_detach_model_manager_returns_to_no_model(self):
        """Detaching the active manager immediately disables inference."""
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=[])
        service = FocusPredictionService(
            telemetry_repository=repo,
            model_manager=_make_mock_model_manager(),
        )

        service.detach_model_manager()

        result = await service.predict_latest(user_id=1)
        assert result.status == "no_model"
        repo.list_feature_windows_in_range.assert_not_awaited()

    async def test_detach_during_feature_query_stops_inference(self):
        """A privacy wipe racing with a query cannot use the detached model."""
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i))
            for i in range(23, 0, -1)
        ]
        repo = MagicMock()
        manager = _make_mock_model_manager()
        service = FocusPredictionService(repo, manager)

        async def detach_before_return(**_kwargs: Any) -> list[dict]:
            service.detach_model_manager()
            return windows

        repo.list_feature_windows_in_range = AsyncMock(side_effect=detach_before_return)

        result = await service.predict_latest(user_id=1, now=now)

        assert result.status == "no_model"
        manager.classifier.predict_proba.assert_not_called()

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
        assert health["feature_schema_version"] == FEATURE_SCHEMA_VERSION

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
        repo.list_feature_windows_in_range = AsyncMock(return_value=all_windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_range(
            user_id=1,
            start=now - timedelta(hours=2),
            end=now,
            now=now,
        )
        assert result.window_count == 2


class TestFocusPredictionRangeBoundedRetrieval:
    """Regression: prediction retrieval must be range-bounded.

    ``predict_latest``/``predict_range`` must query via
    ``list_feature_windows_in_range`` (never load the full window history) and
    must never build a feature matrix larger than ``_MAX_PREDICT_WINDOWS`` —
    the matrix rows and the reported ``window_count`` must always agree.
    """

    def _make_matrix_probe(self) -> tuple[dict[str, float], Any]:
        """Return a (probe, side_effect) pair capturing matrix shape/ordering."""
        probe: dict[str, float] = {}

        def _probe(matrix: np.ndarray) -> np.ndarray:
            probe["rows"] = float(matrix.shape[0])
            probe["first_app_switch"] = float(matrix[0, 0])
            probe["last_app_switch"] = float(matrix[-1, 0])
            n = matrix.shape[0]
            return np.column_stack([np.full(n, 0.25), np.full(n, 0.75)])

        return probe, _probe

    def _oversized_windows(
        self,
        *,
        spacing_s: int,
        latest_end: datetime,
    ) -> list[dict]:
        """Build >_MAX_PREDICT_WINDOWS ascending windows, most recent first by index."""
        total = 3 * _MAX_PREDICT_WINDOWS
        return [
            _make_feature_window(
                latest_end
                - timedelta(minutes=5)
                - timedelta(seconds=spacing_s * (total - 1 - i)),
                features={"app_switch_count": i},
            )
            for i in range(total)
        ]

    async def test_predict_latest_queries_range_bounded(self):
        """predict_latest queries the repo by range, never full history."""
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i))
            for i in range(23, 0, -1)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock()  # legacy path must not be used
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.82)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "ready"
        repo.list_feature_windows_in_range.assert_awaited_once()
        call = repo.list_feature_windows_in_range.await_args
        assert call.kwargs["user_id"] == 1
        assert call.kwargs["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        assert call.kwargs["end"] == now
        assert (now - call.kwargs["start"]).total_seconds() == pytest.approx(7200.0)
        repo.list_feature_windows.assert_not_awaited()

    async def test_predict_range_queries_range_bounded(self):
        """predict_range passes the requested range through to the repo."""
        now = datetime.now(UTC)
        start = now - timedelta(hours=2)
        end = now
        windows = [
            _make_feature_window(start + timedelta(minutes=5 * i))
            for i in range(23)
        ]
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock()  # legacy path must not be used
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.65)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_range(user_id=1, start=start, end=end, now=now)
        assert result.status == "ready"
        repo.list_feature_windows_in_range.assert_awaited_once()
        call = repo.list_feature_windows_in_range.await_args
        assert call.kwargs["start"] == start
        assert call.kwargs["end"] == end
        assert call.kwargs["user_id"] == 1
        assert call.kwargs["feature_schema_version"] == FEATURE_SCHEMA_VERSION
        repo.list_feature_windows.assert_not_awaited()

    async def test_predict_latest_bounds_matrix_to_max_windows(self):
        """>_MAX_PREDICT_WINDOWS in the lookback never over-builds the matrix."""
        now = datetime.now(UTC)
        total = 3 * _MAX_PREDICT_WINDOWS
        windows = self._oversized_windows(
            spacing_s=3,
            latest_end=now,  # newest window ends exactly at *now*
        )
        probe, side_effect = self._make_matrix_probe()
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)  # legacy path
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        mm.classifier.predict_proba.side_effect = side_effect
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        # The retained overlapping rows cover 1797 seconds, not 500 full windows.
        assert result.status == "stale"
        assert result.coverage_ratio == 0.2496
        repo.list_feature_windows_in_range.assert_awaited_once()
        repo.list_feature_windows.assert_not_awaited()
        assert probe["rows"] == _MAX_PREDICT_WINDOWS
        assert result.window_count == _MAX_PREDICT_WINDOWS
        assert probe["rows"] == result.window_count
        # Recency preserved: the most recent _MAX_PREDICT_WINDOWS windows.
        assert probe["first_app_switch"] == float(total - _MAX_PREDICT_WINDOWS)
        assert probe["last_app_switch"] == float(total - 1)

    async def test_predict_range_bounds_matrix_to_max_windows(self):
        """A range wider than _MAX_PREDICT_WINDOWS never over-builds the matrix."""
        now = datetime.now(UTC)
        total = 3 * _MAX_PREDICT_WINDOWS
        start = now - timedelta(hours=13)
        end = now
        windows = self._oversized_windows(
            spacing_s=30,
            latest_end=now,  # newest window ends exactly at *now*
        )
        probe, side_effect = self._make_matrix_probe()
        repo = MagicMock()
        repo.list_feature_windows = AsyncMock(return_value=windows)  # legacy path
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.75)
        mm.classifier.predict_proba.side_effect = side_effect
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_range(user_id=1, start=start, end=end, now=now)
        assert result.status == "ready"
        repo.list_feature_windows_in_range.assert_awaited_once()
        call = repo.list_feature_windows_in_range.await_args
        assert call.kwargs["start"] == start
        assert call.kwargs["end"] == end
        assert probe["rows"] == _MAX_PREDICT_WINDOWS
        assert result.window_count == _MAX_PREDICT_WINDOWS
        assert probe["rows"] == result.window_count
        # Recency preserved: the most recent _MAX_PREDICT_WINDOWS windows.
        assert probe["first_app_switch"] == float(total - _MAX_PREDICT_WINDOWS)
        assert probe["last_app_switch"] == float(total - 1)


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
        assert prediction.feature_schema_version == FEATURE_SCHEMA_VERSION
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
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager()
        mm.classifier.predict_proba.side_effect = RuntimeError("model crash")
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "inference_error"
        assert result.focus_probability is None
        assert result.window_count == 23

    async def test_db_query_failure_is_inference_error(self) -> None:
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(side_effect=RuntimeError("db down"))
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
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
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
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
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
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
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
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        mm = _make_mock_model_manager(focus_proba=0.82)
        service = FocusPredictionService(telemetry_repository=repo, model_manager=mm)
        result = await service.predict_latest(user_id=1, now=now)
        assert result.status == "ready"
        assert result.focus_probability is not None
        assert 0.0 <= result.focus_probability <= 1.0
        assert result.coverage_ratio <= 1.0
        assert result.feature_schema_version == FEATURE_SCHEMA_VERSION
        assert len(result.top_factors) == 3
        assert result.model_version == "20260726_v2"


class TestLocalExplanationWiring:
    """Contract §1: ``explanation`` on a ready prediction.

    Uses a real fitted classifier (not a mock) because the whole point of the
    field is that the contributions add up to the reported probability — a
    mock ``predict_proba`` cannot produce a decomposition.
    """

    @staticmethod
    def _fitted_manager() -> MagicMock:
        """Real RF+XGB ensemble wrapped in a manager-shaped mock."""
        from mindflow.train.models.ensemble import EnsembleClassifier

        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(300, len(V2_FEATURE_NAMES)))
        labels = (-matrix[:, 0] - matrix[:, 3] + rng.normal(scale=0.6, size=300) > 0)
        model = EnsembleClassifier(calibration=None)
        model.fit(matrix, labels.astype(int), list(V2_FEATURE_NAMES))

        manager = MagicMock()
        manager.current_version_tag = "20261110_v4"
        manager.classifier = model
        return manager

    @staticmethod
    def _windows(now: datetime, n: int = 24) -> list[dict]:
        rng = np.random.default_rng(4)
        windows = []
        for i in range(n, 0, -1):
            features = {
                name: float(rng.normal())
                for name in V2_FEATURE_NAMES
            }
            windows.append(
                _make_feature_window(now - timedelta(minutes=5 * i), features=features)
            )
        return windows

    async def test_ready_prediction_carries_additive_explanation(self) -> None:
        now = datetime.now(UTC)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=self._windows(now))
        service = FocusPredictionService(
            telemetry_repository=repo, model_manager=self._fitted_manager()
        )

        result = await service.predict_latest(user_id=1, now=now)

        assert result.status == "ready"
        explanation = result.explanation
        assert explanation is not None
        assert explanation["method"] == "tree_shap"
        assert explanation["output_space"] == "probability"
        assert 1 <= len(explanation["contributions"]) <= 8
        # The contract's promise: base + Σ contributions ≈ focus_probability.
        total = explanation["base_value"] + sum(
            item["contribution"] for item in explanation["contributions"]
        )
        # Only the top 8 are reported, so the sum is a lower bound on the total
        # magnitude; assert the *reported* probability is within the plausible
        # band rather than an exact equality.
        assert total == pytest.approx(result.focus_probability, abs=0.25)
        for item in explanation["contributions"]:
            assert set(item) == {
                "feature",
                "label",
                "value",
                "baseline_value",
                "contribution",
            }
            assert item["feature"] in V2_FEATURE_NAMES
            # Chinese label from the shared map, not the raw column name.
            assert item["label"] != item["feature"]

    async def test_explanation_reports_baseline_values_when_available(self) -> None:
        """A stored baseline contributes per-feature ``baseline_value``."""
        from mindflow.domain.baseline import BaselineModel

        now = datetime.now(UTC)
        windows = self._windows(now)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)

        baseline = BaselineModel(user_id=1, timezone="Asia/Shanghai")
        rng = np.random.default_rng(9)
        for day in range(3):
            for window in windows[:6]:
                start = datetime.fromisoformat(window["window_start_utc"])
                baseline.update(
                    [
                        {
                            "window_start_utc": start - timedelta(days=7 * day),
                            "features_json": json.dumps(
                                {
                                    name: float(rng.normal())
                                    for name in V2_FEATURE_NAMES
                                }
                            ),
                        }
                    ]
                )
        baseline_repo = MagicMock()
        baseline_repo.get_latest = AsyncMock(return_value=baseline)

        service = FocusPredictionService(
            telemetry_repository=repo,
            model_manager=self._fitted_manager(),
            baseline_repository=baseline_repo,
        )
        result = await service.predict_latest(user_id=1, now=now)

        assert result.explanation is not None
        baselined = [
            item
            for item in result.explanation["contributions"]
            if item["baseline_value"] is not None
        ]
        assert baselined, "baseline_repository supplied but no baseline_value reported"

    async def test_unexplainable_model_reports_no_explanation(self) -> None:
        """A classifier that cannot be decomposed leaves the field null."""
        now = datetime.now(UTC)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=self._windows(now))
        service = FocusPredictionService(
            telemetry_repository=repo, model_manager=_make_mock_model_manager()
        )

        result = await service.predict_latest(user_id=1, now=now)

        assert result.status == "ready"
        assert result.explanation is None

    async def test_predict_range_skips_the_explanation(self) -> None:
        """Panel evidence does not read it; SHAP must not be paid for there."""
        now = datetime.now(UTC)
        windows = self._windows(now)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        service = FocusPredictionService(
            telemetry_repository=repo, model_manager=self._fitted_manager()
        )

        result = await service.predict_range(
            user_id=1, start=now - timedelta(hours=2), end=now, now=now
        )

        assert result.status == "ready"
        assert result.explanation is None

    async def test_explainer_is_cached_per_model_version(self) -> None:
        now = datetime.now(UTC)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=self._windows(now))
        manager = self._fitted_manager()
        service = FocusPredictionService(telemetry_repository=repo, model_manager=manager)

        await service.predict_latest(user_id=1, now=now)
        first = dict(service._explainers)  # noqa: SLF001
        await service.predict_latest(user_id=1, now=now)

        assert set(first) == {"20261110_v4"}
        assert service._explainers["20261110_v4"] is first["20261110_v4"]  # noqa: SLF001


class TestPredictionTimeRange:
    @pytest.mark.parametrize("minutes", [5, 30, 120, 360])
    async def test_complete_range_has_full_coverage(self, minutes: int) -> None:
        end = datetime(2026, 11, 12, 8, tzinfo=UTC)
        start = end - timedelta(minutes=minutes)
        windows = [
            _make_feature_window(start + timedelta(minutes=5 * i))
            for i in range(minutes // 5)
        ]
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
        service = FocusPredictionService(repo, _make_mock_model_manager())

        result = await service.predict_range(1, start, end, now=end)

        assert result.status == "ready"
        assert result.coverage_ratio == 1.0

    async def test_historical_range_is_fresh_at_its_own_end(self) -> None:
        now = datetime(2026, 11, 12, 8, tzinfo=UTC)
        end = now - timedelta(days=1)
        start = end - timedelta(hours=1)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=[
            _make_feature_window(start + timedelta(minutes=5 * i)) for i in range(12)
        ])
        service = FocusPredictionService(repo, _make_mock_model_manager())

        result = await service.predict_range(1, start, end, now=now)

        assert result.status == "ready"
        assert result.coverage_ratio == 1.0
        assert result.data_age_s == 0.0

    async def test_missing_hours_are_not_hidden_by_recent_data(self) -> None:
        end = datetime(2026, 11, 12, 8, tzinfo=UTC)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=[
            _make_feature_window(end - timedelta(minutes=5 * i))
            for i in range(6, 0, -1)
        ])
        service = FocusPredictionService(repo, _make_mock_model_manager())

        result = await service.predict_range(1, end - timedelta(hours=6), end, now=end)

        assert result.status == "stale"
        assert result.coverage_ratio == pytest.approx(0.0833)

    async def test_latest_coverage_includes_gap_after_last_window(self) -> None:
        now = datetime(2026, 11, 12, 8, tzinfo=UTC)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=[
            _make_feature_window(now - timedelta(minutes=5 * i))
            for i in range(24, 18, -1)
        ])
        service = FocusPredictionService(repo, _make_mock_model_manager())

        result = await service.predict_latest(1, now=now)

        assert result.status == "stale"
        assert result.coverage_ratio == 0.25

    async def test_partial_boundary_windows_use_clipped_duration(self) -> None:
        end = datetime(2026, 11, 12, 8, tzinfo=UTC)
        repo = MagicMock()
        repo.list_feature_windows_in_range = AsyncMock(return_value=[
            _make_feature_window(end - timedelta(minutes=10)),
            _make_feature_window(end - timedelta(minutes=5)),
        ])
        service = FocusPredictionService(repo, _make_mock_model_manager())

        result = await service.predict_range(
            1, end - timedelta(minutes=7), end - timedelta(minutes=2), now=end,
        )

        assert result.status == "ready"
        assert result.coverage_ratio == 1.0
        assert result.data_age_s == 0.0


@pytest.mark.parametrize("invalid", [
    [[float("nan"), 0.7]],
    [[0.3, float("inf")]],
    [[-0.2, 1.2]],
    [[0.3, 0.3]],
    [[0.3, 0.7], [0.2, 0.8]],
    [[0.3, 0.7, 0.0]],
])
async def test_invalid_probabilities_degrade_without_exposing_numbers(invalid: list) -> None:
    now = datetime(2026, 11, 12, 8, tzinfo=UTC)
    repo = MagicMock()
    repo.list_feature_windows_in_range = AsyncMock(
        return_value=[_make_feature_window(now - timedelta(minutes=5))],
    )
    manager = _make_mock_model_manager()
    manager.classifier.predict_proba.side_effect = None
    manager.classifier.predict_proba.return_value = np.asarray(invalid)
    service = FocusPredictionService(repo, manager)

    result = await service.predict_range(1, now - timedelta(minutes=5), now, now=now)

    assert result.status == "inference_error"
    assert result.focus_probability is None
    assert result.uncertainty is None


@pytest.mark.parametrize("observed_windows", [0, 1])
async def test_empty_collection_windows_do_not_become_behavior(observed_windows: int) -> None:
    now = datetime(2026, 11, 12, 8, tzinfo=UTC)
    windows = [
        _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(24, 0, -1)
    ]
    for index, window in enumerate(windows):
        observed = index >= 24 - observed_windows
        window["quality_json"] = json.dumps({
            "sources_available": {"activity": observed},
            "coverage": {"activity": 1.0 if observed else 0.0},
            "quality_tier": "full" if observed else "empty",
        })
    repo = MagicMock()
    repo.list_feature_windows_in_range = AsyncMock(return_value=windows)
    service = FocusPredictionService(repo, _make_mock_model_manager())

    result = await service.predict_latest(1, now=now)

    assert result.window_count == observed_windows
    assert result.status == ("stale" if observed_windows else "no_data")
    assert result.coverage_ratio == pytest.approx(0.0417 if observed_windows else 0.0)


def test_model_replacement_with_same_tag_refreshes_explanation() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    matrix = np.zeros((20, len(V2_FEATURE_NAMES)))
    matrix[:, 0] = np.arange(20)
    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    query = matrix[-1:]
    service = FocusPredictionService(MagicMock())
    explanations = []
    for reverse in (False, True):
        labels = np.arange(20) > 9
        classifier = LogisticRegression().fit(scaled, ~labels if reverse else labels)
        manager = _make_mock_model_manager()
        manager.classifier.model = classifier
        manager.classifier.scaler = scaler
        service.attach_model_manager(manager)
        probabilities = classifier.predict_proba(scaler.transform(query))[:, 1]
        explanations.append(service._explain(manager, query, probabilities, None))

    first, second = explanations
    assert first is not None and second is not None
    assert first["contributions"][0]["contribution"] > 0
    assert second["contributions"][0]["contribution"] < 0
