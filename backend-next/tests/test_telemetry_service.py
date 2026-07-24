from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np

from mindflow.services.telemetry_service import TelemetryService
from mindflow.train.v2 import V2_FEATURE_NAMES


class _Classifier:
    def predict_proba(self, features):
        assert features.shape == (1, len(V2_FEATURE_NAMES))
        return np.array([[0.2, 0.8]])

    def get_feature_importance(self):
        return {name: (index + 1) / 100 for index, name in enumerate(V2_FEATURE_NAMES)}


class _Manager:
    classifier = _Classifier()
    current_version_tag = "20260724"


async def test_predict_latest_focus_returns_probability_uncertainty_and_top_factors(
    tmp_path,
) -> None:
    repository = AsyncMock()
    repository.list_feature_windows.return_value = [{
        "window_start_utc": "2026-07-24T08:00:00+00:00",
        "features_json": "{\"idle_ratio\": 0.1, \"top_app_ratio\": 0.9}",
    }]
    service = TelemetryService(
        repository=repository,
        preferences_repository=AsyncMock(),
        data_dir=tmp_path,
    )
    service.attach_model_manager(_Manager())

    prediction = await service.predict_latest_focus()

    assert prediction["mode"] == "ready"
    assert prediction["focus_probability"] == 0.8
    assert prediction["uncertainty"] == 0.4
    assert prediction["feature_schema_version"] == 2
    assert len(prediction["top_factors"]) == 3
