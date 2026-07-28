from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import numpy as np

from mindflow.services.telemetry_service import TelemetryService
from mindflow.train.v2 import V2_FEATURE_NAMES


def _make_service(tmp_path):
    prefs_repo = AsyncMock()
    prefs_repo.get.return_value = {"telemetry": {}}
    prefs_repo.set.return_value = None
    return TelemetryService(
        repository=AsyncMock(),
        preferences_repository=prefs_repo,
        data_dir=tmp_path,
    )


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
    repository.latest_feature_window.return_value = {
        "window_start_utc": "2026-07-24T08:00:00+00:00",
        "features_json": "{\"idle_ratio\": 0.1, \"top_app_ratio\": 0.9}",
    }
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


# --- _cleanup_expired_pairing_codes unit tests ---


def test_cleanup_removes_past_expiry(tmp_path) -> None:
    """Given a code with expires_at in the past, When cleanup runs, Then it is removed."""
    service = _make_service(tmp_path)
    now = datetime.now(UTC)
    service._pairing_codes["past"] = now - timedelta(minutes=1)

    service._cleanup_expired_pairing_codes(now)

    assert "past" not in service._pairing_codes


def test_cleanup_removes_exactly_expired(tmp_path) -> None:
    """Given a code with expires_at == now, When cleanup runs, Then it is removed."""
    service = _make_service(tmp_path)
    now = datetime.now(UTC)
    service._pairing_codes["exact"] = now

    service._cleanup_expired_pairing_codes(now)

    assert "exact" not in service._pairing_codes


def test_cleanup_keeps_future(tmp_path) -> None:
    """Given a code with expires_at in the future, When cleanup runs, Then it survives."""
    service = _make_service(tmp_path)
    now = datetime.now(UTC)
    future_at = now + timedelta(hours=1)
    service._pairing_codes["future"] = future_at

    service._cleanup_expired_pairing_codes(now)

    assert "future" in service._pairing_codes
    assert service._pairing_codes["future"] is future_at


def test_cleanup_bulk_1000(tmp_path) -> None:
    """Given 1000 expired codes and 1 future, When cleanup runs, Then all 1000
    are removed and the future code survives."""
    service = _make_service(tmp_path)
    now = datetime.now(UTC)
    expired_at = now - timedelta(hours=2)
    for i in range(1000):
        service._pairing_codes[f"expired-{i:04d}"] = expired_at
    future_at = now + timedelta(hours=1)
    service._pairing_codes["future"] = future_at

    service._cleanup_expired_pairing_codes(now)

    assert len(service._pairing_codes) == 1
    assert "future" in service._pairing_codes


# --- create_pairing_code integration tests ---


async def test_create_pairing_code_removes_expired_and_keeps_future(tmp_path) -> None:
    """Given past, exactly-expired, and future codes,
    When create_pairing_code is called,
    Then past+exact are removed, future survives, new code is added."""
    service = _make_service(tmp_path)
    now = datetime.now(UTC)
    past_at = now - timedelta(minutes=10)
    exact_at = now
    future_at = now + timedelta(hours=1)

    service._pairing_codes["past"] = past_at
    service._pairing_codes["exact"] = exact_at
    service._pairing_codes["future"] = future_at

    result = await service.create_pairing_code()

    assert "past" not in service._pairing_codes
    assert "exact" not in service._pairing_codes
    assert "future" in service._pairing_codes
    assert service._pairing_codes["future"] is future_at
    new_code = result["code"]
    assert new_code in service._pairing_codes
    assert service._pairing_codes[new_code] > now


async def test_create_pairing_code_bulk_1000_cleanup(tmp_path) -> None:
    """Given 1000 expired codes seeded, When create_pairing_code is called,
    Then all 1000 are removed, 1 future + 1 new code remain."""
    service = _make_service(tmp_path)
    now = datetime.now(UTC)
    expired_at = now - timedelta(minutes=30)
    for i in range(1000):
        service._pairing_codes[f"expired-{i:04d}"] = expired_at
    future_at = now + timedelta(hours=2)
    service._pairing_codes["future"] = future_at

    result = await service.create_pairing_code()

    assert len(service._pairing_codes) == 2
    assert "future" in service._pairing_codes
    new_code = result["code"]
    assert new_code in service._pairing_codes
    assert service._pairing_codes[new_code] > now
