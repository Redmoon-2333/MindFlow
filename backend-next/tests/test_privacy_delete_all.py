"""Privacy wipe — DELETE /telemetry/data?scope=all.

Approved semantics for "behavior data all":

  Deletes (per authenticated user):
    - raw activity events (``activity_events``)
    - input telemetry buckets (``interaction_buckets``)
    - browser segments (``browser_segments``)
    - focus sessions and focus feedback (``focus_sessions``,
      ``focus_session_feedback``)
    - daily reports and analytics artifacts (``daily_reports``,
      ``procrastination_analyses``)
    - intervention logs / checks / slot state (``intervention_logs``,
      ``intervention_checks``, ``intervention_slot_reservations``)
    - collector lifecycle audit rows (``collector_intervals``)
    - derived baseline/model metadata (``baseline_models``)
    - derived feature windows (``behavior_feature_windows``)
    - user-owned local training/model artifact files under the configured
      ``models_dir/v2`` root

  Revokes (does not delete): browser pairing tokens (``browser_tokens``).

  Preserves: ``chat_messages``, ``user_preferences``,
  ``app_classification_rules``, authentication credential files (``token``),
  and backup files (``backups/*``).

Narrower scopes (interaction/browser/feedback) keep their existing semantics.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.routes.telemetry import router
from mindflow.infrastructure.repositories.activity import activity_events
from mindflow.infrastructure.repositories.focus import focus_sessions
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.report import daily_reports
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.schema import (
    app_classification_rules,
    baseline_models,
    behavior_feature_windows,
    browser_segments,
    browser_tokens,
    chat_messages,
    collector_intervals,
    focus_session_feedback,
    interaction_buckets,
    intervention_checks,
    intervention_logs,
    intervention_slot_reservations,
    metadata,
    procrastination_analyses,
    user_preferences,
)
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.services.telemetry_service import TelemetryClearResult, TelemetryService
from mindflow.train.models.manager import ModelManager
from mindflow.train.v2 import V2_FEATURE_NAMES

# Tables whose rows are deleted for the current user by scope=all.
IN_SCOPE_TABLES: list[sa.Table] = [
    activity_events,
    interaction_buckets,
    browser_segments,
    focus_sessions,
    focus_session_feedback,
    daily_reports,
    procrastination_analyses,
    intervention_logs,
    intervention_checks,
    intervention_slot_reservations,
    collector_intervals,
    behavior_feature_windows,
    baseline_models,
]


@pytest.fixture
async def all_tables_engine(engine):
    """Create every table used by the delete-all surface on the temp DB."""
    async with engine.begin() as connection:
        await connection.run_sync(metadata.create_all)
        await connection.run_sync(activity_events.metadata.create_all)
        await connection.run_sync(focus_sessions.metadata.create_all)
        await connection.run_sync(daily_reports.metadata.create_all)
    return engine


async def _insert(session_factory: Any, table: sa.Table, **values: Any) -> None:
    async with session_factory() as session, session.begin():
        await session.execute(table.insert().values(**values))


async def _count(session_factory: Any, table: sa.Table, user_id: int = 1) -> int:
    async with session_factory() as session:
        return int(
            (
                await session.execute(
                    sa.select(sa.func.count())
                    .select_from(table)
                    .where(table.c.user_id == user_id)
                )
            ).scalar()
            or 0
        )


async def _count_all_users(session_factory: Any, table: sa.Table) -> int:
    async with session_factory() as session:
        return int(
            (await session.execute(sa.select(sa.func.count()).select_from(table))).scalar()
            or 0
        )


def _seed_artifact_dir(models_dir: Path) -> Path:
    """Create a realistic set of MindFlow-owned v2 model artifacts."""
    v2 = models_dir / "v2"
    v2.mkdir(parents=True, exist_ok=True)
    (v2 / "classifier-20260724_120000_abc123.pkl").write_bytes(b"classifier-payload")
    (v2 / "clustering-20260724_120000_abc123.pkl").write_bytes(b"clustering-payload")
    (v2 / "hmm-20260724_120000_abc123.pkl").write_bytes(b"hmm-payload")
    (v2 / "classifier-20260724_120000_abc123.pkl.hmac").write_bytes(b"hmac")
    (v2 / "manifest.json").write_text('{"version": "x"}', encoding="utf-8")
    latest = '{"classifier": "classifier-20260724_120000_abc123.pkl"}'
    (v2 / "latest.json").write_text(latest, encoding="utf-8")
    (v2 / "training_report.json").write_text('{"model_mode": "ready"}', encoding="utf-8")
    (v2 / "signing_key").write_bytes(b"key-material")
    return v2


def _seed_backup_and_token(data_dir: Path) -> None:
    """Create backup files (preserved) and the auth token file (preserved)."""
    backup_dir = data_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / "mindflow-2026-07-24.db").write_bytes(b"sqlite-backup-sentinel")
    (backup_dir / "mindflow-2026-07-23.db").write_bytes(b"older-backup-sentinel")
    (data_dir / "token").write_text("auth-token-sentinel", encoding="utf-8")


def _make_service(session_factory: Any, tmp_path: Path) -> TelemetryService:
    """Construct the production service with only the pre-existing API surface.

    Model artifacts are seeded under ``tmp_path/models/v2`` — the default
    artifact root anchored under ``data_dir`` (mirrors ``Settings.models_dir``).
    """
    return TelemetryService(
        TelemetryRepository(session_factory),
        PreferencesRepository(session_factory),
        data_dir=tmp_path,
    )


def _make_route_app(service: TelemetryService) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.telemetry_service = service
    return app


async def _seed_user_1(session_factory: Any, tmp_path: Path) -> dict[str, int]:
    """Seed one row per in-scope table for user 1 plus 2 browser tokens.

    Returns the expected per-table deleted count (1 per table; tokens are
    revoked, not deleted).
    """
    now = datetime(2026, 7, 24, 8, 0, tzinfo=UTC).isoformat()
    telemetry = TelemetryRepository(session_factory)

    await _insert(session_factory, interaction_buckets,
                  id="ib-1", user_id=1, window_start_utc=now, duration_s=30.0,
                  context_key="code.exe:abc", keypress_count=1,
                  mouse_click_count=1, scroll_delta=0, mouse_distance_px=0.0,
                  input_active_s=1.0, interaction_burst_count=1, created_at=now)
    await _insert(session_factory, browser_segments,
                  id="bs-1", user_id=1, timestamp=now, duration_s=5.0,
                  browser_name="edge", domain="example.com", audible=False,
                  context_key="edge:example.com", created_at=now)
    await _insert(session_factory, focus_sessions,
                  id="fs-1", user_id=1, date="2026-07-24", start_time=now,
                  end_time=now, session_type="focus", created_at=now)
    await _insert(session_factory, focus_session_feedback,
                  id="ff-1", user_id=1, session_id="fs-1", label="focus",
                  score=5, task_type="coding", created_at=now)
    await _insert(session_factory, behavior_feature_windows,
                  id="fw-1", user_id=1, window_start_utc=now,
                  window_end_utc=now, feature_schema_version=3,
                  features_json="{}", label=None, created_at=now)
    await _insert(session_factory, activity_events,
                  id="evt-1", user_id=1, timestamp=now, duration_s=30.0,
                  data_json=json.dumps({
                      "app_name": "Code",
                      "process_name": "code.exe",
                      "window_title": "main.py",
                      "is_idle": False,
                  }), event_type="window_snapshot")
    await _insert(session_factory, intervention_checks,
                  id="ic-1", user_id=1, checked_at=now, reason="low_confidence",
                  confidence=0.4, source="rule_engine", ml_status="ready")
    await _insert(session_factory, intervention_logs,
                  id="il-1", user_id=1, triggered_at=now,
                  intervention_type="popup")
    await _insert(session_factory, intervention_slot_reservations,
                  id="isr-1", user_id=1, date="2026-07-24", slot_index=1,
                  intervention_type="popup")
    await _insert(session_factory, collector_intervals,
                  id="ci-1", user_id=1, started_at=now, ended_at=now,
                  reason="manual stop", manual_stop=1, failure=0, sleep=0,
                  last_error=None)
    await _insert(session_factory, daily_reports,
                  id="dr-1", user_id=1, date="2026-07-24", total_focus_min=60.0,
                  total_distraction_min=30.0, focus_score=0.7,
                  top_apps_json="{}", switch_frequency=5.0,
                  pattern_summary="focused", created_at=now)
    await _insert(session_factory, procrastination_analyses,
                  id="pa-1", user_id=1, date="2026-07-24",
                  analysis_kind="daily_attribution", source="panel")
    await _insert(session_factory, baseline_models,
                  id="bm-1", user_id=1, model_json="{}",
                  training_events_count=5, created_at=now, updated_at=now)
    await telemetry.save_browser_token(1, "token-user1-1")
    await telemetry.save_browser_token(1, "token-user1-2")
    return {table.name: 1 for table in IN_SCOPE_TABLES}


async def _seed_user_2(session_factory: Any) -> None:
    """Rows belonging to a different user — must survive a user-1 wipe."""
    now = datetime(2026, 7, 24, 8, 0, tzinfo=UTC).isoformat()
    await _insert(session_factory, activity_events,
                  id="evt-u2", user_id=2, timestamp=now, duration_s=30.0,
                  data_json=json.dumps({"app_name": "Mail",
                                        "process_name": "mail.exe",
                                        "is_idle": False}),
                  event_type="window_snapshot")
    await _insert(session_factory, interaction_buckets,
                  id="ib-u2", user_id=2, window_start_utc=now, duration_s=30.0,
                  context_key="mail.exe:def", keypress_count=1,
                  mouse_click_count=1, scroll_delta=0, mouse_distance_px=0.0,
                  input_active_s=1.0, interaction_burst_count=1, created_at=now)
    await _insert(session_factory, focus_sessions,
                  id="fs-u2", user_id=2, date="2026-07-24", start_time=now,
                  end_time=now, session_type="focus", created_at=now)
    await _insert(session_factory, baseline_models,
                  id="bm-u2", user_id=2, model_json="{}",
                  training_events_count=3, created_at=now, updated_at=now)
    await _insert(session_factory, collector_intervals,
                  id="ci-u2", user_id=2, started_at=now, ended_at=now,
                  reason="service shutdown", manual_stop=0, failure=0,
                  sleep=0, last_error=None)
    await TelemetryRepository(session_factory).save_browser_token(2, "token-user2-1")


async def _seed_preserved(session_factory: Any, tmp_path: Path) -> None:
    """Seed chat, preferences, classification rules, backups and auth token."""
    now = datetime(2026, 7, 24, 8, 0, tzinfo=UTC).isoformat()
    await _insert(session_factory, user_preferences,
                  id="pref-1", user_id=1,
                  preferences_json=json.dumps({"telemetry": {"input_telemetry_enabled": True}}),
                  updated_at=now)
    await _insert(session_factory, chat_messages,
                  id="chat-1", user_id=1, session_id="sess-1",
                  role="user", content="keep me")
    await _insert(session_factory, app_classification_rules,
                  id="rule-1", user_id=1, process_name="code.exe",
                  category="code", priority=0, created_at=now, updated_at=now)
    _seed_backup_and_token(tmp_path)


# ── RED/GREEN: service-level delete-all ──────────────────────────────────


async def test_scope_all_clears_in_scope_tables_and_preserves_sentinels(
    all_tables_engine, session_factory, tmp_path,
) -> None:
    """Given: a fully seeded user with every in-scope table, tokens, artifacts,
    and preserved sentinels. When: clear_data("all", user_id=1) runs.
    Then: every in-scope table is empty for user 1, tokens are revoked in place,
    other-user rows survive, and chat/preferences/rules/backups/token/artifacts
    follow the approved preserve/delete semantics."""
    expected = await _seed_user_1(session_factory, tmp_path)
    await _seed_user_2(session_factory)
    await _seed_preserved(session_factory, tmp_path)
    models_dir = tmp_path / "models"
    _seed_artifact_dir(models_dir)

    service = _make_service(session_factory, tmp_path)
    deleted = await service.clear_data("all", user_id=1)

    assert deleted == sum(expected.values())
    for table in IN_SCOPE_TABLES:
        assert await _count(session_factory, table, user_id=1) == 0, (
            f"{table.name} still has rows for user 1"
        )

    # Browser tokens are revoked, not deleted; user 2's token untouched.
    async with session_factory() as session:
        rows = (
            await session.execute(
                sa.select(browser_tokens.c.token_hash, browser_tokens.c.revoked_at)
                .where(browser_tokens.c.user_id == 1)
            )
        ).fetchall()
        user2_rows = (
            await session.execute(
                sa.select(browser_tokens.c.token_hash, browser_tokens.c.revoked_at)
                .where(browser_tokens.c.user_id == 2)
            )
        ).fetchall()
    assert len(rows) == 2
    assert all(row.revoked_at is not None for row in rows)
    assert len(user2_rows) == 1
    assert user2_rows[0].revoked_at is None

    # Other-user rows survive the user-1 wipe.
    for table in (activity_events, interaction_buckets, focus_sessions, baseline_models):
        assert await _count(session_factory, table, user_id=2) == 1

    # Preserved data.
    assert await _count(session_factory, user_preferences, user_id=1) == 1
    assert await _count(session_factory, chat_messages, user_id=1) == 1
    assert await _count(session_factory, app_classification_rules, user_id=1) == 1
    assert (tmp_path / "token").read_text(encoding="utf-8") == "auth-token-sentinel"
    assert (tmp_path / "backups" / "mindflow-2026-07-24.db").exists()
    assert (tmp_path / "backups" / "mindflow-2026-07-23.db").exists()

    # User-owned model artifacts removed.
    assert not (models_dir / "v2").exists()


async def test_scope_all_unloads_shared_manager_and_stops_runtime_inference(
    tmp_path: Path,
) -> None:
    """Disk deletion also invalidates app-state and service-held model references."""
    now = datetime(2026, 8, 5, 8, 0, tzinfo=UTC)
    windows = []
    for index in range(23):
        window_start = now - timedelta(minutes=5 * (24 - index))
        features = {name: float(index) for name in V2_FEATURE_NAMES}
        windows.append(
            {
                "window_start_utc": window_start.isoformat(),
                "window_end_utc": (window_start + timedelta(minutes=5)).isoformat(),
                "features_json": json.dumps(features),
                "feature_schema_version": 3,
            }
        )

    repository = AsyncMock()
    repository.list_feature_windows_in_range.return_value = windows
    repository.delete_scope.return_value = 0
    repository.revoke_browser_tokens.return_value = 0
    manager = ModelManager(models_dir=tmp_path / "models" / "v2", use_ensemble=False)
    training_features = np.arange(480, dtype=np.float64).reshape(20, 24)
    manager.classifier.fit(
        training_features,
        np.array([0, 1] * 10, dtype=np.int32),
        list(V2_FEATURE_NAMES),
    )
    manager.clustering.model = SimpleNamespace()
    manager.hmm._is_fitted = True

    prediction_service = FocusPredictionService(repository, manager)
    service = TelemetryService(
        repository,
        AsyncMock(),
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        prediction_service=prediction_service,
    )
    service.attach_model_manager(manager)
    app_state_manager = manager
    classifier = manager.classifier
    clustering = manager.clustering
    hmm = manager.hmm

    assert (await prediction_service.predict_latest(now=now)).status == "ready"
    repository.list_feature_windows_in_range.reset_mock()

    await service.clear_data("all", user_id=1)

    assert app_state_manager.readiness_status()["ready"] is False
    assert manager.classifier is classifier
    assert manager.clustering is clustering
    assert manager.hmm is hmm
    assert classifier._is_fitted is False
    assert clustering.model is None
    assert hmm._is_fitted is False
    assert (await prediction_service.predict_latest(now=now)).status == "no_model"
    telemetry_prediction = await service.predict_latest_focus(user_id=1)
    assert telemetry_prediction["status"] == "no_model"
    repository.list_feature_windows_in_range.assert_not_awaited()


async def test_scope_all_detaches_fake_manager_without_unload(tmp_path: Path) -> None:
    """Test doubles without ModelManager.unload remain supported."""
    repository = AsyncMock()
    repository.delete_scope.return_value = 0
    repository.revoke_browser_tokens.return_value = 0
    fake_manager = SimpleNamespace(
        classifier=SimpleNamespace(_is_fitted=True),
        current_version_tag="fake-v1",
    )
    prediction_service = FocusPredictionService(repository, fake_manager)
    service = TelemetryService(
        repository,
        AsyncMock(),
        data_dir=tmp_path,
        prediction_service=prediction_service,
    )
    service.attach_model_manager(fake_manager)

    result = await service.clear_data("all", user_id=1)

    assert result.partial is False
    assert (await prediction_service.predict_latest()).status == "no_model"
    assert (await service.predict_latest_focus())["status"] == "no_model"


# ── RED/GREEN: real endpoint through ASGI transport ──────────────────────


async def test_scope_all_through_delete_endpoint(
    all_tables_engine, session_factory, tmp_path,
) -> None:
    """The actual DELETE /api/v1/telemetry/data?scope=all route is driven through
    ASGI (TestClient); afterwards the real SQLite state and the filesystem are
    inspected."""
    expected = await _seed_user_1(session_factory, tmp_path)
    await _seed_preserved(session_factory, tmp_path)
    models_dir = tmp_path / "models"
    _seed_artifact_dir(models_dir)

    app = _make_route_app(_make_service(session_factory, tmp_path))
    response = TestClient(app).delete("/api/v1/telemetry/data", params={"scope": "all"})

    assert response.status_code == 200
    assert response.json() == {"deleted": sum(expected.values())}

    for table in IN_SCOPE_TABLES:
        assert await _count(session_factory, table, user_id=1) == 0, (
            f"{table.name} still has rows after the real endpoint call"
        )
    async with session_factory() as session:
        token_rows = (
            await session.execute(
                sa.select(browser_tokens.c.revoked_at).where(browser_tokens.c.user_id == 1)
            )
        ).fetchall()
    assert len(token_rows) == 2
    assert all(row.revoked_at is not None for row in token_rows)
    assert await _count(session_factory, chat_messages, user_id=1) == 1
    assert await _count(session_factory, user_preferences, user_id=1) == 1
    assert (tmp_path / "backups" / "mindflow-2026-07-24.db").exists()
    assert not (models_dir / "v2").exists()


async def test_scope_all_reports_partial_when_post_delete_cleanup_fails(
    all_tables_engine, session_factory, tmp_path, monkeypatch,
) -> None:
    """DB rows are deleted, but token/artifact failures remain explicit."""
    expected = await _seed_user_1(session_factory, tmp_path)
    service = _make_service(session_factory, tmp_path)
    attempts: list[str] = []

    async def fail_revoke(user_id: int) -> int:
        attempts.append(f"revoke:{user_id}")
        raise OSError("token store unavailable")

    def fail_artifacts() -> None:
        attempts.append("artifacts")
        raise OSError("model directory unavailable")

    monkeypatch.setattr(service._repository, "revoke_browser_tokens", fail_revoke)
    monkeypatch.setattr(service, "_delete_local_artifacts", fail_artifacts)

    result = await service.clear_data("all", user_id=1)

    assert isinstance(result, TelemetryClearResult)
    assert result == sum(expected.values())
    assert result.partial is True
    assert result.failures == ("browser_tokens", "model_artifacts")
    assert attempts == ["revoke:1", "artifacts"]
    for table in IN_SCOPE_TABLES:
        assert await _count(session_factory, table, user_id=1) == 0


async def test_scope_all_endpoint_exposes_partial_cleanup_result(
    all_tables_engine, session_factory, tmp_path, monkeypatch,
) -> None:
    """The legacy ``deleted`` field remains while partial failures are visible."""
    expected = await _seed_user_1(session_factory, tmp_path)
    service = _make_service(session_factory, tmp_path)

    async def fail_revoke(user_id: int) -> int:
        raise OSError("token store unavailable")

    monkeypatch.setattr(service._repository, "revoke_browser_tokens", fail_revoke)
    monkeypatch.setattr(
        service,
        "_delete_local_artifacts",
        lambda: (_ for _ in ()).throw(OSError("model directory unavailable")),
    )

    response = TestClient(_make_route_app(service)).delete(
        "/api/v1/telemetry/data", params={"scope": "all"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "deleted": sum(expected.values()),
        "partial": True,
        "failures": ["browser_tokens", "model_artifacts"],
    }


# ── Regression: narrower scopes keep their existing semantics ────────────


async def test_narrower_scopes_keep_current_semantics(
    all_tables_engine, session_factory, tmp_path,
) -> None:
    """interaction/browser/feedback scopes only touch their historical tables."""
    now = datetime(2026, 7, 24, 8, 0, tzinfo=UTC).isoformat()
    telemetry = TelemetryRepository(session_factory)
    await _insert(session_factory, interaction_buckets,
                  id="ib-s", user_id=1, window_start_utc=now, duration_s=30.0,
                  context_key="code.exe:abc", keypress_count=1,
                  mouse_click_count=1, scroll_delta=0, mouse_distance_px=0.0,
                  input_active_s=1.0, interaction_burst_count=1, created_at=now)
    await _insert(session_factory, browser_segments,
                  id="bs-s", user_id=1, timestamp=now, duration_s=5.0,
                  browser_name="edge", domain="example.com", audible=False,
                  context_key="edge:example.com", created_at=now)
    await _insert(session_factory, focus_session_feedback,
                  id="ff-s", user_id=1, session_id="fs-s", label="focus",
                  score=5, task_type="coding", created_at=now)
    await telemetry.save_browser_token(1, "token-scope-test")

    service = _make_service(session_factory, tmp_path)

    await service.clear_data("interaction", user_id=1)
    assert await _count(session_factory, interaction_buckets, 1) == 0
    assert await _count(session_factory, browser_segments, 1) == 1
    assert await _count(session_factory, focus_session_feedback, 1) == 1
    async with session_factory() as session:
        token = (
            await session.execute(
                sa.select(browser_tokens.c.revoked_at).where(
                    browser_tokens.c.token_hash == "token-scope-test"
                )
            )
        ).fetchone()
    assert token is not None and token.revoked_at is None

    await service.clear_data("browser", user_id=1)
    assert await _count(session_factory, browser_segments, 1) == 0
    assert await _count(session_factory, focus_session_feedback, 1) == 1
    async with session_factory() as session:
        token_rows = (
            await session.execute(
                sa.select(browser_tokens.c.token_hash).where(browser_tokens.c.user_id == 1)
            )
        ).fetchall()
    assert token_rows == []  # browser scope deletes tokens (unchanged semantics)

    await service.clear_data("feedback", user_id=1)
    assert await _count(session_factory, focus_session_feedback, 1) == 0
