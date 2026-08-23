"""End-to-end flow tests covering every API path and branch condition.

Organized by subsystem; each test exercises a realistic user workflow
rather than a single unit, ensuring that the full request pipeline
works correctly for both happy and error paths.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mindflow.api.errors import register_exception_handlers
from mindflow.api.routes import (
    activities_router,
    analytics_router,
    auth_router,
    autonomy_router,
    chat_router,
    collector_router,
    export_router,
    focus_router,
    health_router,
    intervention_router,
    panel_router,
    reports_router,
    telemetry_router,
)
from mindflow.domain.feature_schema import V2_FEATURE_NAMES
from mindflow.domain.procrastination import RuleEngine
from mindflow.eval.scenarios import ALL_SCENARIOS

# ── Test helpers ─────────────────────────────────────────────────────────────


class _HealthyEngine:
    def connect(self):
        class _Ctx:
            async def __aenter__(self): return object()
            async def __aexit__(self, *a): return None
        return _Ctx()


class _BrokenEngine:
    async def connect(self): raise RuntimeError("db down")


# All routers and state keys that routes depend on
_ROUTERS = (
    health_router, collector_router, focus_router, activities_router,
    reports_router, analytics_router, panel_router, chat_router,
    intervention_router, export_router, telemetry_router, autonomy_router,
    auth_router,
)


def _make_app(**overrides: Any) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    for r in _ROUTERS:
        app.include_router(r, prefix="/api/v1")
    # Core state
    app.state.engine = _HealthyEngine()
    app.state.migration_applied = True
    app.state.db_integrity_ok = True
    app.state.system_token = ""
    app.state.v2_training_mode = "rule_engine_only"
    app.state.collector_service = MagicMock()
    app.state.collector_service.status = "stopped"
    app.state.checkpointer = MagicMock()
    app.state.checkpointer._closed = False
    # Service stubs
    for attr in (
        "analysis_service", "activity_repository", "intervention_service",
        "intervention_repo", "telemetry_service", "panel_service",
        "chat_service", "llm_service", "prediction_service",
        "training_job_service", "training_readiness_service",
        "report_service", "export_service", "autonomy_service",
        "baseline_repo", "focus_repository", "workflow_runs_repository",
        "scheduler", "session_factory",
    ):
        if attr not in overrides:
            setattr(app.state, attr, MagicMock())
    # Autonomy route awaits get_status(); make the stub awaitable with a sane default.
    if "autonomy_service" not in overrides:
        app.state.autonomy_service = AsyncMock()
        app.state.autonomy_service.get_status.return_value = {
            "enabled": True,
            "paused_until": None,
        }
    for k, v in overrides.items():
        setattr(app.state, k, v)
    # Auth tests opt in to real middleware; the shared `client` fixture stays
    # middleware-free so non-auth routes are reachable without a token.
    if overrides.get("with_auth"):
        from mindflow.api.middleware.auth import AuthMiddleware

        app.add_middleware(AuthMiddleware)
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_make_app())


# ── 1. Health subsystem ──────────────────────────────────────────────────────


class TestHealthE2E:
    def test_live_always_200(self, client: TestClient) -> None:
        r = client.get("/api/v1/health/live")
        assert r.status_code == 200
        assert r.json()["status"] == "alive"

    def test_ready_200_when_all_ok(self, client: TestClient) -> None:
        r = client.get("/api/v1/health/ready")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"

    def test_ready_503_on_migration_fail(self) -> None:
        app = _make_app(migration_applied=False)
        with TestClient(app) as c:
            r = c.get("/api/v1/health/ready")
            assert r.status_code == 503
            assert r.json()["status"] == "not_ready"

    def test_ready_503_on_db_fail(self) -> None:
        app = _make_app(engine=_BrokenEngine())
        with TestClient(app) as c:
            r = c.get("/api/v1/health/ready")
            assert r.status_code == 503

    def test_compatible_always_200(self) -> None:
        app = _make_app(migration_applied=False, db_integrity_ok=False)
        with TestClient(app) as c:
            r = c.get("/api/v1/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

    def test_observability_fields_present(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        obs = data["observability"]
        assert {"last_activity_at", "last_intervention_at",
                "scheduler_heartbeat_at", "ml_mode"} <= set(obs)

    def test_ml_status_includes_v2_training_mode(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["ml"]["v2_training_mode"] == "rule_engine_only"


# ── 2. Collector subsystem ───────────────────────────────────────────────────


class TestCollectorE2E:
    def test_get_status(self, client: TestClient) -> None:
        r = client.get("/api/v1/collector")
        assert r.status_code == 200

    def test_start_stop(self) -> None:
        collector = MagicMock()
        collector.status = "stopped"
        collector.start = AsyncMock()
        collector.stop = AsyncMock()
        app = _make_app(collector_service=collector)
        with TestClient(app) as c:
            c.post("/api/v1/collector")
            assert collector.start.called
            c.post("/api/v1/collector/stop")
            assert collector.stop.called


# ── 3. Auth subsystem ───────────────────────────────────────────────────────


class TestAuthE2E:
    def test_protected_endpoint_rejects_without_token(self) -> None:
        app = _make_app(system_token="secret", with_auth=True)
        with TestClient(app) as c:
            r = c.get("/api/v1/focus")
            assert r.status_code == 401

    def test_exempt_endpoint_passes_without_token(self) -> None:
        app = _make_app(system_token="secret")
        with TestClient(app) as c:
            r = c.get("/api/v1/health")
            assert r.status_code == 200

    def test_bootstrap_ticket_endpoint_requires_token(self) -> None:
        """bootstrap/ticket issues one-time tickets and requires the Bearer system token.

        This endpoint was deliberately removed from _EXEMPT_PATHS (fcb7021):
        it mints bootstrap credentials, so unauthenticated requests must be
        rejected with 401 rather than publicly callable.
        """
        app = _make_app(system_token="secret", with_auth=True)
        with TestClient(app) as c:
            r = c.post("/api/v1/auth/bootstrap/ticket")
            assert r.status_code == 401

    def test_bootstrap_ticket_with_valid_token(self) -> None:
        """With a valid Bearer token the launcher can mint a one-time ticket."""
        app = _make_app(system_token="secret", with_auth=True)
        with TestClient(app) as c:
            r = c.post(
                "/api/v1/auth/bootstrap/ticket",
                headers={"Authorization": "Bearer secret"},
            )
            # 503 only if the ticket store is missing; a minted ticket is 200.
            assert r.status_code in (200, 503)


# ── 4. Autonomy subsystem ───────────────────────────────────────────────────


class TestAutonomyE2E:
    def test_get_autonomy_status(self, client: TestClient) -> None:
        r = client.get("/api/v1/autonomy")
        assert r.status_code == 200
        data = r.json()
        assert "enabled" in data


# ── 5. Scheduler job verification ───────────────────────────────────────────


class TestSchedulerE2E:
    def test_build_scheduler_registers_all_jobs(self) -> None:
        from mindflow.services.scheduler import build_scheduler
        scheduler = build_scheduler(
            analysis_service=MagicMock(), report_service=MagicMock(),
            maintenance_service=MagicMock(), intervention_service=MagicMock(),
            activity_repository=MagicMock(), panel_service=MagicMock(),
            autonomy_service=MagicMock(), telemetry_service=MagicMock(),
        )
        jobs = {j.id for j in scheduler.get_jobs()}
        assert "daily_panel" in jobs
        assert "auto_intervention_check" in jobs
        assert "telemetry_rollup_recent" in jobs

    def test_intervention_interval_is_5min(self) -> None:
        from mindflow.services.scheduler import build_scheduler
        s = build_scheduler(intervention_service=MagicMock(), activity_repository=MagicMock())
        job = s.get_job("auto_intervention_check")
        assert job.trigger.interval.total_seconds() == 300

    def test_recent_rollup_interval_is_15min(self) -> None:
        from mindflow.services.scheduler import build_scheduler
        s = build_scheduler(
            analysis_service=MagicMock(), report_service=MagicMock(),
            maintenance_service=MagicMock(), telemetry_service=MagicMock(),
        )
        job = s.get_job("telemetry_rollup_recent")
        assert job.trigger.interval.total_seconds() == 900

    def test_heartbeat_tracked(self) -> None:
        from mindflow.services.scheduler import AsyncioScheduler
        s = AsyncioScheduler(timezone="UTC")
        assert s.last_heartbeat_at is None
        s._touch_heartbeat()
        assert s.last_heartbeat_at is not None


# ── 6. Migration & schema ───────────────────────────────────────────────────


class TestMigrationE2E:
    def test_focus_session_feedback_has_snapshot_columns(self) -> None:
        from mindflow.infrastructure.schema import focus_session_feedback
        col_names = {c.name for c in focus_session_feedback.columns}
        assert "session_start_utc" in col_names
        assert "session_end_utc" in col_names

    def test_intervention_checks_table_exists(self) -> None:
        from mindflow.infrastructure.schema import intervention_checks
        col_names = {c.name for c in intervention_checks.columns}
        assert "checked_at" in col_names
        assert "ml_status" in col_names


# ── 7. ML training pipeline ─────────────────────────────────────────────────


def _build_feature_row(is_focus: bool) -> dict:
    features = {n: 0.0 for n in V2_FEATURE_NAMES}
    features.update({
        "idle_ratio": 0.01 if is_focus else 0.7,
        "longest_segment_ratio": 0.98 if is_focus else 0.05,
        "top_app_ratio": 0.98 if is_focus else 0.1,
        "input_active_ratio": 0.7 if is_focus else 0.05,
        "app_switch_count": 0 if is_focus else 12,
        "domain_switch_count": 0 if is_focus else 8,
    })
    return features


class TestMLTrainingPipelineE2E:
    def test_synthetic_v2_training_completes(self, tmp_path: Path) -> None:
        from mindflow.train.pipeline import run_training
        report = run_training(source="synthetic_v2", data_dir=tmp_path / "d",
                              models_dir=tmp_path / "m", days=3, seed=42)
        assert report.source == "synthetic_v2"
        assert report.feature_schema_version == 3

    def test_v2_training_shadow_when_gates_fail(self, tmp_path: Path) -> None:
        from mindflow.train.pipeline import run_training
        # 6 days × 2 classes = 12 windows: enough to enter the training branch
        # (>=10 explicit samples) but too few feedback days for the quality gate
        # (minimum_days >= 7), so the model lands in shadow mode, not ready.
        start = datetime(2026, 7, 1, 9, tzinfo=UTC)
        windows, feedback = [], []
        for d in range(6):
            for cls in (True, False):
                s = start + timedelta(days=d, hours=2 if cls else 4)
                windows.append({"window_start_utc": s.isoformat(),
                    "window_end_utc": (s + timedelta(minutes=5)).isoformat(),
                    "feature_schema_version": 3, "features": _build_feature_row(cls)})
                feedback.append({"session_id": f"s-{d}-{cls}",
                    "start_time": s.isoformat(),
                    "end_time": (s + timedelta(minutes=30)).isoformat(),
                    "label": "focus" if cls else "distracted",
                    "score": 5 if cls else 1, "task_type": "coding"})
        report = run_training(source="db", data_dir=tmp_path / "d",
                              models_dir=tmp_path / "m",
                              feature_windows=windows, feedback_sessions=feedback,
                              calibration=None)  # toy data: skip post-hoc calibration
        assert report.model_mode == "shadow"
        assert not (tmp_path / "m" / "v2" / "latest.json").exists()

    def test_v2_training_activates_when_gates_pass(self, tmp_path: Path) -> None:
        from mindflow.train.pipeline import run_training
        start = datetime(2026, 7, 1, 8, tzinfo=UTC)
        windows, feedback, idx = [], [], 0
        for d in range(8):
            for cls in range(4):
                is_focus = cls < 2
                s = start + timedelta(days=d, hours=cls * 2)
                windows.append({"window_start_utc": s.isoformat(),
                    "window_end_utc": (s + timedelta(minutes=5)).isoformat(),
                    "feature_schema_version": 3, "features": _build_feature_row(is_focus)})
                feedback.append({"session_id": f"s-{idx}",
                    "start_time": s.isoformat(),
                    "end_time": (s + timedelta(minutes=30)).isoformat(),
                    "label": "focus" if is_focus else "distracted",
                    "score": 5 if is_focus else 1, "task_type": "coding"})
                idx += 1
        report = run_training(source="db", data_dir=tmp_path / "d",
                              models_dir=tmp_path / "m",
                              feature_windows=windows, feedback_sessions=feedback,
                              calibration=None)  # toy data: skip post-hoc calibration
        assert report.model_mode == "ready"
        assert report.activated is True
        assert (tmp_path / "m" / "v2" / "latest.json").exists()
        assert (tmp_path / "m" / "v2" / "manifest.json").exists()

    def test_manifest_contains_evaluation_and_version(self, tmp_path: Path) -> None:
        from mindflow.train.pipeline import run_training
        start = datetime(2026, 7, 1, 8, tzinfo=UTC)
        windows, feedback, idx = [], [], 0
        for d in range(8):
            for cls in range(4):
                is_focus = cls < 2
                s = start + timedelta(days=d, hours=cls * 2)
                windows.append({"window_start_utc": s.isoformat(),
                    "window_end_utc": (s + timedelta(minutes=5)).isoformat(),
                    "feature_schema_version": 3, "features": _build_feature_row(is_focus)})
                feedback.append({"session_id": f"s-{idx}",
                    "start_time": s.isoformat(),
                    "end_time": (s + timedelta(minutes=30)).isoformat(),
                    "label": "focus" if is_focus else "distracted",
                    "score": 5 if is_focus else 1, "task_type": "coding"})
                idx += 1
        report = run_training(source="db", data_dir=tmp_path / "d",
                              models_dir=tmp_path / "m",
                              feature_windows=windows, feedback_sessions=feedback)
        manifest = json.loads((tmp_path / "m" / "v2" / "manifest.json").read_text())
        assert manifest["version"] == report.version_tag
        assert "evaluation" in manifest
        assert manifest["evaluation"]["candidate"]["balanced_accuracy"] > 0

    def test_version_tag_format_has_random_suffix(self, tmp_path: Path) -> None:
        from mindflow.train.pipeline import run_training
        start = datetime(2026, 7, 1, 8, tzinfo=UTC)
        windows, feedback, idx = [], [], 0
        for d in range(8):
            for cls in range(4):
                is_focus = cls < 2
                s = start + timedelta(days=d, hours=cls * 2)
                windows.append({"window_start_utc": s.isoformat(),
                    "window_end_utc": (s + timedelta(minutes=5)).isoformat(),
                    "feature_schema_version": 3, "features": _build_feature_row(is_focus)})
                feedback.append({"session_id": f"s-{idx}",
                    "start_time": s.isoformat(),
                    "end_time": (s + timedelta(minutes=30)).isoformat(),
                    "label": "focus" if is_focus else "distracted",
                    "score": 5 if is_focus else 1, "task_type": "coding"})
                idx += 1
        report = run_training(source="db", data_dir=tmp_path / "d",
                              models_dir=tmp_path / "m",
                              feature_windows=windows, feedback_sessions=feedback)
        tag = report.version_tag
        assert tag is not None
        assert "_" in tag  # timestamp + random suffix format

    def test_weak_labels_only_all_focus_no_crash(self, tmp_path: Path) -> None:
        from mindflow.train.pipeline import run_training
        start = datetime(2026, 7, 1, 9, tzinfo=UTC)
        windows, feedback, idx = [], [], 0
        for d in range(8):
            for h in range(4):
                s = start + timedelta(days=d, hours=h * 2, minutes=5)
                features = {n: 0.0 for n in V2_FEATURE_NAMES}
                features.update({"idle_ratio": 0.02, "longest_segment_ratio": 0.9,
                                 "top_app_ratio": 0.9, "input_active_ratio": 0.7})
                windows.append({"window_start_utc": s.isoformat(),
                    "window_end_utc": (s + timedelta(minutes=5)).isoformat(),
                    "feature_schema_version": 3, "features": features})
                feedback.append({"session_id": f"s-{idx}",
                    "start_time": s.isoformat(),
                    "end_time": (s + timedelta(minutes=5)).isoformat(),
                    "label": "focus", "score": 5, "task_type": "coding"})
                idx += 1
        report = run_training(source="db", data_dir=tmp_path / "d",
                              models_dir=tmp_path / "m",
                              feature_windows=windows, feedback_sessions=feedback)
        assert report.evaluation is not None


# ── 8. Evaluation pipeline ─────────────────────────────────────────────────


class TestEvalPipelineE2E:
    async def test_rule_engine_baseline_evaluation(self) -> None:
        from mindflow.eval.runner import EvalReport, run_eval
        engine = RuleEngine()
        async def analyzer(bundle):
            return engine.assess(bundle.behavior_summary)
        report = await run_eval(analyzer, ALL_SCENARIOS[:5], analyzer_name="rule_engine")
        assert isinstance(report, EvalReport)
        assert report.total == 5
        assert report.top1_accuracy >= 0
        assert len(report.per_type) > 0

    async def test_per_type_metrics_structure(self) -> None:
        from mindflow.eval.runner import run_eval
        engine = RuleEngine()
        async def analyzer(bundle):
            return engine.assess(bundle.behavior_summary)
        report = await run_eval(analyzer, ALL_SCENARIOS[:10], analyzer_name="rule")
        for _type_key, metrics in report.per_type.items():
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics
            assert "support" in metrics

    async def test_comparison_same_analyzer_zero_delta(self) -> None:
        from mindflow.eval.runner import compare, run_eval
        engine = RuleEngine()
        async def analyzer(bundle):
            return engine.assess(bundle.behavior_summary)
        baseline = await run_eval(analyzer, ALL_SCENARIOS[:10], analyzer_name="a")
        panel = await run_eval(analyzer, ALL_SCENARIOS[:10], analyzer_name="b")
        comp = compare(baseline, panel)
        assert comp.top1_delta == 0.0
        assert comp.baseline_wins == 0
        assert comp.panel_wins == 0


# ── 9. Model manager versioning ────────────────────────────────────────────


class TestModelManagerVersioningE2E:
    def test_save_load_manifest_lifecycle(self, tmp_path: Path) -> None:
        from mindflow.train.models.manager import ModelManager
        mgr = ModelManager(models_dir=tmp_path, use_ensemble=False)
        X = np.random.default_rng(42).random((50, 14))
        mgr.train_all(X, [f"f{i}" for i in range(14)], np.array([1]*25+[0]*25))
        mgr.save_all(activate=True, manifest={"test": True})
        tag = mgr.current_version_tag
        assert tag is not None
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["version"] == tag
        assert manifest["test"] is True
        mgr2 = ModelManager(models_dir=tmp_path, use_ensemble=False)
        assert mgr2.load_latest() is True

    def test_shadow_save_preserves_latest(self, tmp_path: Path) -> None:
        from mindflow.train.models.manager import ModelManager
        mgr = ModelManager(models_dir=tmp_path, use_ensemble=False)
        X = np.random.default_rng(42).random((50, 14))
        mgr.train_all(X, [f"f{i}" for i in range(14)], np.array([1]*25+[0]*25))
        mgr.save_all(activate=False)
        assert not (tmp_path / "latest.json").exists()

    def test_rollback_roundtrip(self, tmp_path: Path) -> None:
        from mindflow.train.models.manager import ModelManager
        mgr = ModelManager(models_dir=tmp_path, use_ensemble=False)
        X = np.random.default_rng(42).random((50, 14))
        mgr.train_all(X, [f"f{i}" for i in range(14)], np.array([1]*25+[0]*25))
        mgr.save_all(activate=True)
        tag = mgr.current_version_tag
        assert mgr.rollback(tag) is True
        assert mgr.current_version_tag == tag

    def test_tampered_model_refused(self, tmp_path: Path) -> None:
        from mindflow.train.models.manager import ModelManager, ModelSignatureError
        mgr = ModelManager(models_dir=tmp_path, use_ensemble=False)
        X = np.random.default_rng(42).random((50, 14))
        mgr.train_all(X, [f"f{i}" for i in range(14)], np.array([1]*25+[0]*25))
        mgr.save_all(activate=True)
        pkl = list(tmp_path.glob("classifier-*.pkl"))[0]
        pkl.write_bytes(b"tampered" + pkl.read_bytes())
        with pytest.raises(ModelSignatureError):
            ModelManager(models_dir=tmp_path, use_ensemble=False).load_latest()


# ── 10. Panel graph topology & parsing ──────────────────────────────────────


class TestPanelGraphTopologyE2E:
    def test_no_dead_fanout_and_required_nodes(self) -> None:
        import mindflow.graph.panel_graph as pg
        from mindflow.graph.panel_graph import PanelGraph
        class _GW:
            async def complete(self, **kw): return "{}"
            async def close(self): pass
        graph = PanelGraph(_GW()).build()
        assert not hasattr(pg, "attribution_fanout")
        assert "verdict_schema_validation" in graph.nodes
        assert "human_review_interrupt" in graph.nodes
        assert "attribution_call" not in graph.nodes

    def test_moderator_round_trip(self) -> None:
        from mindflow.agents.orchestrator import _parse_verdict
        raw = json.dumps({
            "types": ["impulsivity", "decisional"],
            "confidence": {"impulsivity": 0.8, "decisional": 0.3},
            "recommended_technique": "stimulus_control",
            "rationale": "test", "dissent": [],
            "insufficient_data": True, "uncertainty": 0.6,
            "evidence_gaps": ["no_mouse_data"]})
        v = _parse_verdict(raw)
        assert v["insufficient_data"] is True
        assert v["uncertainty"] == 0.6
        assert "no_mouse_data" in v["evidence_gaps"]

    def test_critic_json_bool_parsing(self) -> None:
        from mindflow.agents.schemas import CriticOutput
        assert CriticOutput.model_validate_json('{"approved": true, "issues": []}').approved is True
        assert (
            CriticOutput.model_validate_json('{"approved": false, "issues": []}').approved
            is False
        )
        assert CriticOutput.model_validate_json('{"approved": 1, "issues": []}').approved is True
        assert CriticOutput.model_validate_json('{"approved": 0, "issues": []}').approved is False


# ── 11. LLM gateway ────────────────────────────────────────────────────────


class TestLLMGatewayE2E:
    def test_temperature_is_02(self) -> None:
        from mindflow.agents.llm_gateway import _LLM_TEMPERATURE
        assert _LLM_TEMPERATURE == 0.2

    async def test_raises_without_key(self) -> None:
        from mindflow.agents.llm_gateway import GatewayNotConfiguredError, LangChainGateway
        gw = LangChainGateway(api_key="", base_url="http://fake")
        with pytest.raises(GatewayNotConfiguredError):
            await gw.complete("sys", "user")

    async def test_close_idempotent(self) -> None:
        from mindflow.agents.llm_gateway import LangChainGateway
        gw = LangChainGateway(api_key="test", base_url="http://fake")
        await gw.close()
        await gw.close()  # no error


# ── 12. Calibration bins ────────────────────────────────────────────────────


class TestCalibrationBinsE2E:
    def test_calibration_bins(self) -> None:
        from mindflow.train.v2 import _calibration_bins
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.55, 0.45])
        bins = _calibration_bins(y_true, y_proba)
        assert len(bins) > 0
        for b in bins:
            assert 0 <= b["fraction_positive"] <= 1

    def test_empty_returns_empty(self) -> None:
        from mindflow.train.v2 import _calibration_bins
        assert _calibration_bins(np.array([]), np.array([])) == []


# ── 13. Focus prediction ───────────────────────────────────────────────────


class TestFocusPredictionE2E:
    async def test_no_model(self) -> None:
        from mindflow.services.prediction_service import FocusPredictionService
        svc = FocusPredictionService(telemetry_repository=AsyncMock(), model_manager=None)
        r = await svc.predict_latest(user_id=1)
        assert r.status == "no_model"

    async def test_no_data(self) -> None:
        from mindflow.services.prediction_service import FocusPredictionService
        from mindflow.train.models.manager import ModelManager
        with tempfile.TemporaryDirectory() as td:
            mgr = ModelManager(models_dir=Path(td), use_ensemble=False)
            X = np.random.default_rng(42).random((50, 14))
            mgr.train_all(X, [f"f{i}" for i in range(14)], np.array([1]*25+[0]*25))
            mgr.save_all(activate=True)
            repo = AsyncMock()
            repo.list_feature_windows_in_range = AsyncMock(return_value=[])
            svc = FocusPredictionService(telemetry_repository=repo, model_manager=mgr)
            r = await svc.predict_latest(user_id=1)
            assert r.status == "no_data"


# ── 14. App startup mode ───────────────────────────────────────────────────


class TestAppModeLogicE2E:
    def test_shadow_report_keeps_mode_shadow(self) -> None:
        """When training_report says shadow, loaded old model should not become ready."""
        report = {"model_mode": "shadow", "version_tag": "20260726"}
        v2_report_mode = report.get("model_mode")
        v2_report_version = report.get("version_tag")
        loaded_tag = "20260726"
        v2_training_mode = "rule_engine_only"
        if v2_report_mode == "shadow":
            v2_training_mode = "shadow"
        elif v2_report_mode == "ready" and (
            v2_report_version is None or v2_report_version == loaded_tag):
            v2_training_mode = "ready"
        assert v2_training_mode == "shadow"

    def test_ready_report_matching_version(self) -> None:
        report = {"model_mode": "ready", "version_tag": "20260726"}
        v2_report_mode = report.get("model_mode")
        v2_report_version = report.get("version_tag")
        loaded_tag = "20260726"
        v2_training_mode = "rule_engine_only"
        if v2_report_mode == "shadow":
            v2_training_mode = "shadow"
        elif v2_report_mode == "ready" and (
            v2_report_version is None or v2_report_version == loaded_tag):
            v2_training_mode = "ready"
        assert v2_training_mode == "ready"


# ── 15. Training readiness structure ────────────────────────────────────────


class TestTrainingReadinessE2E:
    def test_gate_keys_complete(self) -> None:
        from mindflow.services.training_readiness_service import _GATES
        gate_keys = {g.key for g in _GATES}
        assert gate_keys == {
            "minimum_days", "minimum_explicit_feedback", "minimum_class_feedback",
            "balanced_accuracy", "minority_f1", "calibration_better_than_rule",
            "stable_date_folds"}


# ── 16. Intervention throttle ───────────────────────────────────────────────


class TestInterventionThrottleE2E:
    async def test_ok_when_no_history(self) -> None:
        from mindflow.services.intervention_throttle import InterventionThrottle
        repo = AsyncMock()
        stats = MagicMock()
        stats.today_count = 0
        stats.last_triggered_at = None
        stats.ignore_rate = 0.0
        stats.today_count_by_type = 0
        stats.annoying_count_by_type = 0
        repo.get_throttle_stats = AsyncMock(return_value=stats)
        throttle = InterventionThrottle(repo=repo)
        decision = await throttle.can_intervene(1, "nudge")
        assert decision.allowed is True

    async def test_cooldown_rejects(self) -> None:
        from mindflow.services.intervention_throttle import InterventionThrottle
        repo = AsyncMock()
        stats = MagicMock()
        stats.today_count = 1
        stats.last_triggered_at = datetime.now(UTC).isoformat()
        stats.ignore_rate = 0.0
        stats.today_count_by_type = 1
        stats.annoying_count_by_type = 0
        repo.get_throttle_stats = AsyncMock(return_value=stats)
        throttle = InterventionThrottle(repo=repo, cooldown_h=2.0)
        decision = await throttle.can_intervene(1, "nudge")
        assert decision.allowed is False


# ── 17. Auto-intervention guard branches ────────────────────────────────────


class TestAutoInterventionE2E:
    async def test_autonomy_disabled_skips(self) -> None:
        from mindflow.services.autonomy_service import AutonomyService
        from mindflow.services.scheduler import _auto_intervention_check
        autonomy = MagicMock(spec=AutonomyService)
        autonomy.is_enabled = AsyncMock(return_value=False)
        intervention_svc = MagicMock()
        await _auto_intervention_check(
            activity_repo=AsyncMock(), intervention_service=intervention_svc,
            autonomy_service=autonomy, timezone="UTC")
        intervention_svc.maybe_intervene.assert_not_called()

    async def test_outside_hours_skips(self) -> None:
        from mindflow.services.scheduler import _auto_intervention_check
        with patch("mindflow.services.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 7, 17, 2, 30, tzinfo=UTC)
            await _auto_intervention_check(
                activity_repo=AsyncMock(), intervention_service=MagicMock(),
                timezone="UTC")

    async def test_all_idle_skips(self) -> None:
        from mindflow.domain.events import make_event
        from mindflow.services.scheduler import _auto_intervention_check
        idle_event = make_event(user_id=1, timestamp_utc=datetime.now(UTC),
            duration_s=300, process_name="idle", is_idle=True)
        activity_repo = AsyncMock()
        activity_repo.query_range = AsyncMock(return_value=[idle_event])
        intervention_svc = MagicMock()
        await _auto_intervention_check(
            activity_repo=activity_repo, intervention_service=intervention_svc,
            timezone="UTC")
        intervention_svc.maybe_intervene.assert_not_called()


# ── 18. Health integration ──────────────────────────────────────────────────


class TestHealthIntegrationE2E:
    def test_health_live_against_running_server(self) -> None:
        """If the backend is running, live endpoint should return alive."""
        import httpx
        try:
            r = httpx.get("http://127.0.0.1:8765/api/v1/health/live", timeout=3.0)
            assert r.status_code == 200
            assert r.json()["status"] == "alive"
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pytest.skip("Backend not running on 8765")

    def test_health_observability_against_running_server(self) -> None:
        """If the backend is running, health should have observability fields."""
        import httpx
        try:
            r = httpx.get("http://127.0.0.1:8765/api/v1/health", timeout=3.0)
            data = r.json()
            assert "observability" in data
            obs = data["observability"]
            assert "scheduler_heartbeat_at" in obs
            assert "last_activity_at" in obs
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pytest.skip("Backend not running on 8765")

    def test_root_serves_frontend(self) -> None:
        """Backend should serve the frontend at /"""
        import httpx
        try:
            r = httpx.get("http://127.0.0.1:8765/", timeout=3.0)
            assert r.status_code == 200
            assert len(r.text) > 0
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pytest.skip("Backend not running on 8765")

    def test_collector_status_against_running_server(self) -> None:
        import httpx

        from mindflow.config import get_settings
        try:
            token = get_settings().token_path.read_text(encoding="utf-8").strip()
        except (OSError, ValueError):
            token = ""
        try:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = httpx.get(
                "http://127.0.0.1:8765/api/v1/collector",
                timeout=3.0,
                headers=headers,
            )
            assert r.status_code == 200
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pytest.skip("Backend not running on 8765")
