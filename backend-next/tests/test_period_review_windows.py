"""Tests for ``FocusPredictionService.score_windows`` (period-review input).

The one property that matters most here is *no truncation*: a 14-day review
spans far more than ``_MAX_PREDICT_WINDOWS`` (500) windows, and silently
dropping the tail would understate the period. The rest guards the degenerate
states and the driver-selection rules.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.domain.period_review import NON_BEHAVIORAL_FEATURES
from mindflow.services.prediction_service import (
    FocusPredictionService,
    WindowScoring,
)

BASE = datetime(2026, 11, 10, 0, 0, tzinfo=UTC)


class _Classifier:
    """Fitted 2-class classifier that records how many rows it was given."""

    def __init__(self, p_distraction: float = 0.9) -> None:
        self._is_fitted = True
        self._p = p_distraction
        self.feature_names_ = list(V2_FEATURE_NAMES)
        self.batch_sizes: list[int] = []

    def predict_proba(self, matrix: Any) -> Any:
        self.batch_sizes.append(int(matrix.shape[0]))
        rows = matrix.shape[0]
        return np.column_stack([np.full(rows, self._p), np.full(rows, 1.0 - self._p)])

    def get_feature_importance(self) -> dict[str, float]:
        return {}


class _Manager:
    def __init__(self, p_distraction: float = 0.9, version: str = "20260726_v2") -> None:
        self.classifier = _Classifier(p_distraction)
        self.current_version_tag = version


class _Repo:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.error: Exception | None = None

    async def list_feature_windows_in_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
        feature_schema_version: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.error is not None:
            raise self.error
        return list(self._rows)


def _row(index: int, *, features: dict[str, float] | None = None) -> dict[str, Any]:
    start = BASE + timedelta(minutes=5 * index)
    payload = {name: 1.0 for name in V2_FEATURE_NAMES}
    payload["task_type_code"] = 1.0  # coding
    if features:
        payload.update(features)
    return {
        "window_start_utc": start.isoformat(),
        "window_end_utc": (start + timedelta(minutes=5)).isoformat(),
        "features_json": json.dumps(payload),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "quality_json": None,
    }


async def test_more_than_max_windows_are_all_scored() -> None:
    """1,200 windows: every one is scored, in 500-row batches, none dropped."""
    total = 1200
    manager = _Manager()
    service = FocusPredictionService(
        telemetry_repository=_Repo([_row(i) for i in range(total)]),
        model_manager=manager,
    )
    end = BASE + timedelta(minutes=5 * total)
    result = await service.score_windows(1, BASE, end)

    assert result.status == "ready"
    assert len(result.windows) == total
    assert sum(manager.classifier.batch_sizes) == total
    assert manager.classifier.batch_sizes == [500, 500, 200]
    # Windows keep their identity: ascending, distinct starts.
    starts = [window.start_utc for window in result.windows]
    assert starts == sorted(starts)
    assert len(set(starts)) == total


async def test_no_model_returns_no_model() -> None:
    service = FocusPredictionService(telemetry_repository=_Repo([]))
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert result.status == "no_model"
    assert result.windows == ()


async def test_unfitted_classifier_returns_no_model() -> None:
    manager = _Manager()
    manager.classifier._is_fitted = False
    service = FocusPredictionService(
        telemetry_repository=_Repo([_row(0)]), model_manager=manager
    )
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert result.status == "no_model"


async def test_empty_range_returns_no_data() -> None:
    service = FocusPredictionService(
        telemetry_repository=_Repo([]), model_manager=_Manager()
    )
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert result.status == "no_data"


async def test_db_failure_returns_inference_error() -> None:
    repo = _Repo([])
    repo.error = RuntimeError("db down")
    service = FocusPredictionService(telemetry_repository=repo, model_manager=_Manager())
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert result.status == "inference_error"
    assert "db down" in result.reason


async def test_rows_without_activity_are_skipped() -> None:
    rows = [_row(0), _row(1)]
    rows[1]["quality_json"] = json.dumps(
        {"quality_tier": "empty", "sources_available": {"activity": False}}
    )
    service = FocusPredictionService(
        telemetry_repository=_Repo(rows), model_manager=_Manager()
    )
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert result.status == "ready"
    assert len(result.windows) == 1


async def test_unparseable_features_are_skipped() -> None:
    rows = [_row(0), {**_row(1), "features_json": "not json"}]
    service = FocusPredictionService(
        telemetry_repository=_Repo(rows), model_manager=_Manager()
    )
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert len(result.windows) == 1


async def test_model_feature_name_mismatch_is_schema_mismatch() -> None:
    manager = _Manager()
    manager.classifier.feature_names_ = ["wrong"]
    service = FocusPredictionService(
        telemetry_repository=_Repo([_row(0)]), model_manager=manager
    )
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert result.status == "schema_mismatch"


async def test_task_context_is_read_from_the_window() -> None:
    from mindflow.domain.task_context import TASK_TYPE_CODES

    rows = [
        _row(0, features={"task_type_code": TASK_TYPE_CODES["coding"]}),
        _row(1, features={"task_type_code": TASK_TYPE_CODES["entertainment"]}),
    ]
    service = FocusPredictionService(
        telemetry_repository=_Repo(rows), model_manager=_Manager()
    )
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert [window.task_context for window in result.windows] == ["coding", "entertainment"]


async def test_window_scoring_never_raises_on_bad_probabilities() -> None:
    manager = _Manager()
    manager.classifier.predict_proba = lambda matrix: np.full((matrix.shape[0], 2), 0.7)  # type: ignore[method-assign]
    service = FocusPredictionService(
        telemetry_repository=_Repo([_row(0)]), model_manager=manager
    )
    result = await service.score_windows(1, BASE, BASE + timedelta(days=1))
    assert result.status == "inference_error"


async def test_drivers_need_three_windows_per_group() -> None:
    service = FocusPredictionService(
        telemetry_repository=_Repo([_row(i) for i in range(5)]),
        model_manager=_Manager(),
    )
    result = await service.score_windows(1, BASE, BASE + timedelta(minutes=30))
    # Every window is distracted → the focused group is empty.
    assert result.drivers == ()


async def test_drivers_exclude_non_behavioral_features() -> None:
    """A clock column that separates the groups must not become a "signal"."""
    rows: list[dict[str, Any]] = []
    for index in range(20):
        distracted = index < 10
        rows.append(
            _row(
                index,
                features={
                    # Only the hour encoding differs between the two groups.
                    "hour_sin": 1.0 if distracted else -1.0,
                    "app_switch_count": 4.0,
                },
            )
        )
    manager = _Manager(p_distraction=0.9)
    service = FocusPredictionService(telemetry_repository=_Repo(rows), model_manager=manager)

    # Alternate probabilities by row so both groups have members.
    probabilities = np.array([0.9 if i % 2 == 0 else 0.1 for i in range(20)], dtype=float)
    matrix = np.asarray(
        [[1.0 if i % 2 == 0 else -1.0] + [4.0] * (len(V2_FEATURE_NAMES) - 1) for i in range(20)],
        dtype=np.float64,
    )
    drivers = service._distraction_drivers(manager, matrix, probabilities)  # type: ignore[arg-type]
    assert drivers, "a separating behavioural feature exists (app_switch_count ties)"
    features = {driver.feature for driver in drivers}
    assert not features & NON_BEHAVIORAL_FEATURES


def test_window_scoring_default_reason_is_empty() -> None:
    scoring = WindowScoring(status="ready", model_version="v", windows=(), drivers=())
    assert scoring.reason == ""


@pytest.mark.parametrize("status", ["no_model", "no_data", "schema_mismatch", "inference_error"])
async def test_degenerate_statuses_carry_no_windows(status: str) -> None:
    from mindflow.services.prediction_service import WindowScoring

    scoring = WindowScoring(status=status, model_version=None, windows=(), drivers=(), reason="x")  # type: ignore[arg-type]
    assert scoring.windows == () and scoring.drivers == ()
