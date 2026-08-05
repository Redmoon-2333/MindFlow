from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from test_prediction_service import _make_feature_window, _make_mock_model_manager

import mindflow.services.telemetry_service as telemetry_service_module
from mindflow.services.prediction_service import FocusPredictionService
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
    assert prediction["feature_schema_version"] == 3
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


# ── predict_latest_focus normalization (real FocusPredictionService) ────────

_MISSING = MagicMock()  # sentinel: replaced with a real mock model manager


def _prediction_service_for(
    windows: list[dict],
    *,
    focus_proba: float = 0.75,
    model_manager: MagicMock | None = _MISSING,
    classifier_overrides: dict | None = None,
) -> FocusPredictionService:
    """Wire a real ``FocusPredictionService`` against a mocked telemetry repo."""
    repository = AsyncMock()
    repository.list_feature_windows_in_range.return_value = windows
    if model_manager is _MISSING:
        model_manager = _make_mock_model_manager(focus_proba=focus_proba)
    if model_manager is not None and classifier_overrides:
        for key, value in classifier_overrides.items():
            setattr(model_manager.classifier, key, value)
    return FocusPredictionService(
        telemetry_repository=repository,
        model_manager=model_manager,
    )


class TestPredictLatestFocusNormalization:
    """Real FocusPredictionService -> ``TelemetryService.predict_latest_focus``.

    The telemetry service is the single boundary that maps the domain result
    to the API payload: ``focus_probability`` is numeric only for ``ready``,
    present-and-null for every non-ready status, while ``status``/``mode``/
    ``reason`` pass through unchanged.
    """

    async def _predict_latest(
        self,
        tmp_path,
        *,
        windows: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        prediction_service = _prediction_service_for(windows or [], **kwargs)
        service = TelemetryService(
            repository=AsyncMock(),
            preferences_repository=AsyncMock(),
            data_dir=tmp_path,
            prediction_service=prediction_service,
        )
        return await service.predict_latest_focus()

    async def test_ready_returns_numeric_probability(self, tmp_path) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]

        payload = await self._predict_latest(tmp_path, windows=windows)

        assert payload["status"] == "ready"
        assert payload["focus_probability"] == pytest.approx(0.75)
        assert payload["mode"] == "ready"
        assert payload["reason"] == ""

    async def test_old_window_stale_returns_null_probability(self, tmp_path) -> None:
        now = datetime.now(UTC)
        base = now - timedelta(minutes=150)
        windows = [
            _make_feature_window(base + timedelta(minutes=5 * i)) for i in range(24)
        ]

        payload = await self._predict_latest(tmp_path, windows=windows)

        assert payload["status"] == "stale"
        assert payload["focus_probability"] is None
        assert payload["mode"] == "ready"
        assert payload["reason"], "stale must keep its human-readable reason"

    async def test_low_coverage_stale_returns_null_probability(self, tmp_path) -> None:
        now = datetime.now(UTC)
        windows = [_make_feature_window(now - timedelta(minutes=5))]

        payload = await self._predict_latest(tmp_path, windows=windows)

        assert payload["status"] == "stale"
        assert payload["focus_probability"] is None
        assert payload["reason"], "stale must keep its human-readable reason"

    async def test_schema_mismatch_returns_null_probability(self, tmp_path) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]

        payload = await self._predict_latest(
            tmp_path,
            windows=windows,
            classifier_overrides={"feature_names_": ["old_feature"]},
        )

        assert payload["status"] == "schema_mismatch"
        assert payload["focus_probability"] is None
        assert payload["mode"] == "rule_engine_only"
        assert payload["reason"]

    async def test_inference_error_returns_null_probability(self, tmp_path) -> None:
        now = datetime.now(UTC)
        windows = [
            _make_feature_window(now - timedelta(minutes=5 * i)) for i in range(23, 0, -1)
        ]
        model_manager = _make_mock_model_manager()
        model_manager.classifier.predict_proba.side_effect = RuntimeError("model crash")

        payload = await self._predict_latest(
            tmp_path, windows=windows, model_manager=model_manager
        )

        assert payload["status"] == "inference_error"
        assert payload["focus_probability"] is None
        assert payload["mode"] == "rule_engine_only"
        assert payload["reason"]

    async def test_no_data_returns_null_probability(self, tmp_path) -> None:
        payload = await self._predict_latest(tmp_path, windows=[])

        assert payload["status"] == "no_data"
        assert payload["focus_probability"] is None
        assert payload["mode"] == "ready"
        assert payload["reason"]

    async def test_no_model_returns_null_probability(self, tmp_path) -> None:
        payload = await self._predict_latest(tmp_path, windows=[], model_manager=None)

        assert payload["status"] == "no_model"
        assert payload["focus_probability"] is None
        assert payload["mode"] == "rule_engine_only"
        assert payload["reason"]


# ── Activity-retention lifecycle: preference authoritative, env default ──────


async def test_get_preferences_falls_back_to_env_event_retention_days_when_preference_missing(
    tmp_path, monkeypatch,
) -> None:
    """Missing ``activity_retention_days`` preference → env startup default.

    The environment ``event_retention_days`` is the default only when the
    user preference is absent; it must not override a stored preference.
    """
    monkeypatch.setattr(
        telemetry_service_module,
        "get_settings",
        lambda: SimpleNamespace(event_retention_days=45),
    )
    prefs_repo = AsyncMock()
    prefs_repo.get.return_value = {"telemetry": {"input_telemetry_enabled": True}}
    service = TelemetryService(
        repository=AsyncMock(),
        preferences_repository=prefs_repo,
        data_dir=tmp_path,
    )

    prefs = await service.get_preferences(1)

    assert prefs["activity_retention_days"] == 45


async def test_get_preferences_preference_overrides_env_event_retention_days(
    tmp_path, monkeypatch,
) -> None:
    """Stored preference wins over the env startup default."""
    monkeypatch.setattr(
        telemetry_service_module,
        "get_settings",
        lambda: SimpleNamespace(event_retention_days=30),
    )
    prefs_repo = AsyncMock()
    prefs_repo.get.return_value = {"telemetry": {"activity_retention_days": 60}}
    service = TelemetryService(
        repository=AsyncMock(),
        preferences_repository=prefs_repo,
        data_dir=tmp_path,
    )

    prefs = await service.get_preferences(1)

    assert prefs["activity_retention_days"] == 60


async def test_cleanup_retained_data_activity_cutoff_follows_preference(
    tmp_path, monkeypatch,
) -> None:
    """``cleanup_retained_data`` derives the raw-event cutoff from the preference."""
    monkeypatch.setattr(
        telemetry_service_module,
        "get_settings",
        lambda: SimpleNamespace(event_retention_days=45),
    )
    repository = AsyncMock()
    repository.cleanup_old_telemetry = AsyncMock(return_value=7)
    prefs_repo = AsyncMock()
    prefs_repo.get.return_value = {"telemetry": {"activity_retention_days": 60}}
    service = TelemetryService(
        repository=repository,
        preferences_repository=prefs_repo,
        data_dir=tmp_path,
    )

    await service.cleanup_retained_data(1)

    kwargs = repository.cleanup_old_telemetry.await_args.kwargs
    interaction_cutoff = kwargs["interaction_cutoff"]
    activity_cutoff = kwargs["activity_cutoff"]
    feature_cutoff = kwargs["feature_cutoff"]
    assert interaction_cutoff - activity_cutoff == timedelta(days=60 - 7)
    assert activity_cutoff - feature_cutoff == timedelta(days=180 - 60)


async def test_cleanup_retained_data_activity_cutoff_falls_back_to_env(
    tmp_path, monkeypatch,
) -> None:
    """Missing preference → env ``event_retention_days`` drives the activity cutoff."""
    monkeypatch.setattr(
        telemetry_service_module,
        "get_settings",
        lambda: SimpleNamespace(event_retention_days=45),
    )
    repository = AsyncMock()
    repository.cleanup_old_telemetry = AsyncMock(return_value=7)
    prefs_repo = AsyncMock()
    prefs_repo.get.return_value = {"telemetry": {"input_telemetry_enabled": True}}
    service = TelemetryService(
        repository=repository,
        preferences_repository=prefs_repo,
        data_dir=tmp_path,
    )

    await service.cleanup_retained_data(1)

    kwargs = repository.cleanup_old_telemetry.await_args.kwargs
    interaction_cutoff = kwargs["interaction_cutoff"]
    activity_cutoff = kwargs["activity_cutoff"]
    assert interaction_cutoff - activity_cutoff == timedelta(days=45 - 7)
