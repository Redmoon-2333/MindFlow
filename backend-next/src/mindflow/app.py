"""FastAPI application factory — ``create_app(settings) -> FastAPI``.

Wires together:
  - Lifespan: migration → integrity check → token loading → CollectorService
    → Wave 5 services (analysis, report, maintenance) → Wave 6 LLM service
    → scheduler
  - Middleware: logging → host → rate-limit → auth (per §3.5 addition
    order; actual request-processing order is the reverse, so auth gates
    before rate-limit meters — see create_app for the full LIFO rationale)
  - Routes: health, collector, activities, preferences, Wave 5+6 endpoints
  - WebSocket: /api/v1/ws
  - Exception handlers: RFC 9457 ProblemDetail (8 error codes)
  - Security headers: X-MindFlow-Version, X-Content-Type-Options

No global singletons — all shared state lives on ``app.state``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from mindflow import __version__

# Agent imports (G002, G003)
from mindflow.api.errors import register_exception_handlers
from mindflow.api.middleware import (
    AuthMiddleware,
    HostValidationMiddleware,
    RateLimitMiddleware,
    StructuredLoggingMiddleware,
)
from mindflow.api.routes import register_routes
from mindflow.api.websocket import broadcast, close_all_connections
from mindflow.api.websocket import router as websocket_router
from mindflow.config import Settings
from mindflow.graph.analysis_graph import AnalysisGraph
from mindflow.graph.panel_graph import PanelGraph
from mindflow.infrastructure.checkpointer import create_checkpointer
from mindflow.infrastructure.collectors.base import EventCollector, create_collector
from mindflow.infrastructure.database import (
    create_engine,
    create_session_factory,
    integrity_check,
)
from mindflow.infrastructure.migrations import run_migrations
from mindflow.infrastructure.notification import create_notifier
from mindflow.infrastructure.provider_registry import ProviderRegistry
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.analysis import (
    SQLAlchemyProcrastinationAnalysisRepository,
)
from mindflow.infrastructure.repositories.app_classification import (
    AppClassificationRulesRepository,
)
from mindflow.infrastructure.repositories.baseline import (
    BaselineRepository,
)
from mindflow.infrastructure.repositories.chat import (
    ChatRepository,
)
from mindflow.infrastructure.repositories.collector_intervals import (
    CollectorIntervalsRepository,
)
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.infrastructure.repositories.preferences import (
    PreferencesRepository,
)
from mindflow.infrastructure.repositories.report import (
    SQLAlchemyDailyReportRepository,
)
from mindflow.infrastructure.repositories.scheduled_jobs import ScheduledJobRunsRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.infrastructure.repositories.workflow_runs import (
    BudgetReservationRepository,
    WorkflowRunsRepository,
)
from mindflow.infrastructure.security.crisis_detector import CrisisDetector
from mindflow.infrastructure.security.token_manager import (
    BootstrapTicketStore,
    SessionTokenStore,
    load_or_create_token,
)
from mindflow.logging_config import setup_logging
from mindflow.ports import AnalysisWorkflowPort
from mindflow.runtime import RuntimeServices
from mindflow.services.analysis_service import AnalysisService
from mindflow.services.autonomy_service import AutonomyService
from mindflow.services.chat_service import ChatService
from mindflow.services.collector_service import CollectorService
from mindflow.services.effectiveness_service import EffectivenessService
from mindflow.services.evidence_service import EvidenceBundleBuilder
from mindflow.services.input_telemetry_service import InputTelemetryService
from mindflow.services.intervention_service import InterventionService
from mindflow.services.intervention_throttle import InterventionThrottle
from mindflow.services.llm_service import LLMService
from mindflow.services.maintenance_service import MaintenanceService
from mindflow.services.panel_service import PanelService
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.services.report_service import ReportService
from mindflow.services.scheduler import build_scheduler
from mindflow.services.telemetry_service import TelemetryService
from mindflow.services.training_job_service import TrainingJobService

# ── Lifespan ────────────────────────────────────────────────────────────────


class SPAStaticFiles(StaticFiles):
    """Serve built frontend assets and fall back to index.html for SPA routes."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if (
                exc.status_code != 404
                or str(scope.get("path", "")).startswith("/api/")
                or "." in Path(path).name
            ):
                raise
            return await super().get_response("index.html", scope)
        if (
            response.status_code == 404
            and not str(scope.get("path", "")).startswith("/api/")
            and "." not in Path(path).name
        ):
            return await super().get_response("index.html", scope)
        return response


def _frontend_dist_dir() -> Path:
    """Resolve frontend assets in source checkouts and PyInstaller bundles."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        return Path(bundle_root) / "frontend"
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _publish_runtime_state(app: FastAPI, runtime: RuntimeServices) -> None:
    """Publish the aggregate while preserving legacy ``app.state`` aliases."""
    app.state.runtime = runtime
    for name in runtime.__dataclass_fields__:
        setattr(app.state, name, getattr(runtime, name))


async def _start_runtime_services(
    runtime: RuntimeServices, *, input_telemetry_enabled: bool
) -> None:
    if runtime.settings.run_collectors:
        if runtime.collector_service is not None:
            await runtime.collector_service.start()
        if input_telemetry_enabled and runtime.input_telemetry_service is not None:
            await runtime.input_telemetry_service.start()
    if runtime.settings.run_scheduler and runtime.scheduler is not None:
        runtime.scheduler.start()


async def _shutdown_runtime_services(runtime: RuntimeServices) -> None:
    if runtime.scheduler is not None:
        try:
            await runtime.scheduler.shutdown()
        except Exception as exc:
            logger.warning("Scheduler shutdown error: {}", exc)
    # Close provider registry first — owns all HTTP pools shared across
    # LLMService, ChatService, and PanelService. Individual service aclose()
    # calls are no-ops when a registry is injected.
    if runtime.provider_registry is not None:
        try:
            await runtime.provider_registry.shutdown()
        except Exception as exc:
            logger.warning("ProviderRegistry shutdown error: {}", exc)
    for name, service in (
        ("PanelService", runtime.panel_service),
        ("ChatService", runtime.chat_service),
        ("LLMService", runtime.llm_service),
    ):
        if service is not None:
            try:
                await service.aclose()
            except Exception as exc:
                logger.warning("{} close error: {}", name, exc)
    if runtime.input_telemetry_service is not None:
        try:
            await runtime.input_telemetry_service.stop()
        except Exception as exc:
            logger.warning("Input telemetry stop error: {}", exc)
    if runtime.collector_service is not None:
        try:
            await asyncio.wait_for(runtime.collector_service.stop(), timeout=3.0)
        except TimeoutError:
            logger.warning("Collector stop timed out, forcing")
        except Exception as exc:
            logger.warning("Collector stop error: {}", exc)
    if runtime.checkpointer is not None:
        try:
            await runtime.checkpointer.aclose()
        except Exception as exc:
            logger.warning("Checkpointer close error: {}", exc)
    try:
        await asyncio.wait_for(runtime.engine.dispose(), timeout=3.0)
    except TimeoutError:
        logger.warning("Engine dispose timed out")
    except Exception as exc:
        logger.warning("Engine dispose error: {}", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup initialisation, shutdown cleanup.

    Startup sequence:
      1. Run Alembic migrations (fail fast on failure)
      2. Integrity check (attempt VACUUM recovery on failure)
      3. Load/create auth token
      4. Create repositories (activity, preferences, focus, report)
      5. Create CollectorService (not started yet — caller must start)
      6. Create Wave 5 services (analysis, report, maintenance)
      7. Start APScheduler (cron jobs: identify, report, cleanup, backup)
      8. Inject everything into app.state

    Shutdown sequence (reverse order):
      1. Stop scheduler (Wave 5 cron jobs)
      2. Close WebSocket connections
      3. Stop collector
      4. Dispose engine
    """
    # ── Extract settings ─────────────────────────────────────────────
    settings: Settings = app.state.settings
    data_dir = settings.data_dir
    token_path = settings.token_path

    # ── Database engine ───────────────────────────────────────────────
    engine = create_engine(settings.db_url)
    runtime: RuntimeServices | None = None
    checkpointer: Any = None
    checkpointer_ctx: Any = None
    try:
        session_factory = create_session_factory(engine)

        # ── 0b. Checkpointer (LangGraph persistence, same DB file) ──────────
        checkpointer_ctx = create_checkpointer(settings)
        checkpointer = await checkpointer_ctx.__aenter__()
        logger.debug(
            "Checkpointer created (enabled={})", settings.checkpointing_enabled
        )

        # ── 1. Migrations ─────────────────────────────────────────────────
        migration_applied = await run_migrations(settings.db_url)
        if not migration_applied:
            raise RuntimeError(
                "Database migration failed; refusing to start on an incompatible schema"
            )

        # ── 2. Integrity check ────────────────────────────────────────────
        db_ok = await integrity_check(engine)
        if not db_ok:
            logger.critical("Database integrity check failed after recovery attempt")
        else:
            logger.info("Database integrity check passed")

        # ── 3. Auth token ─────────────────────────────────────────────────
        system_token = load_or_create_token(token_path)
        bootstrap_tickets = BootstrapTicketStore()
        browser_sessions = SessionTokenStore()
        logger.debug("Auth token loaded from {}", token_path)

        # ── 4. Repositories ───────────────────────────────────────────────
        activity_repository = SQLAlchemyActivityRepository(
            session_factory=session_factory,
            pulsetime_s=settings.heartbeat_pulsetime_s,
        )
        preferences_repository = PreferencesRepository(
            session_factory=session_factory,
        )
        telemetry_repository = TelemetryRepository(
            session_factory=session_factory,
        )
        classification_rules_repository = AppClassificationRulesRepository(
            session_factory=session_factory,
        )
        focus_repository = SQLAlchemyFocusSessionRepository(
            session_factory=session_factory,
        )
        report_repository = SQLAlchemyDailyReportRepository(
            session_factory=session_factory,
        )
        analysis_repository = SQLAlchemyProcrastinationAnalysisRepository(
            session_factory=session_factory,
        )
        baseline_repository = BaselineRepository(
            session_factory=session_factory,
        )
        chat_repository = ChatRepository(
            session_factory=session_factory,
        )
        scheduled_job_runs_repository = ScheduledJobRunsRepository(session_factory)

        # ── 4b. Wave 7: Intervention repository ───────────────────────────
        intervention_repository = InterventionLogRepository(
            session_factory=session_factory,
        )

        # ── 4c. G005: Autonomy service ──────────────────────────────────
        autonomy_service = AutonomyService(
            preferences_repo=preferences_repository,
        )

        # ── 5. Collector ──────────────────────────────────────────────────
        collector: EventCollector | None = None
        collector_service: CollectorService | None = None
        try:
            collector = create_collector()
            collector_service = CollectorService(
                collector=collector,
                repository=activity_repository,
                interval_repository=CollectorIntervalsRepository(session_factory),
                interval_s=float(settings.collect_interval_s),
            )
            logger.info("CollectorService created (not started)")
        except Exception as exc:
            logger.warning("Failed to create collector: {}", exc)

        input_telemetry_service = InputTelemetryService(
            telemetry_repository=telemetry_repository,
            activity_repository=activity_repository,
        )

        # ── 7-ext. ML prediction service (unified online inference) ─────────
        prediction_service = FocusPredictionService(
            telemetry_repository=telemetry_repository,
        )

        telemetry_service = TelemetryService(
            repository=telemetry_repository,
            preferences_repository=preferences_repository,
            data_dir=data_dir,
            models_dir=settings.models_dir,
            activity_repository=activity_repository,
            prediction_service=prediction_service,
            baseline_repository=baseline_repository,
            session_factory=session_factory,
        )
        telemetry_service.attach_input_watcher(input_telemetry_service)
        telemetry_preferences = await telemetry_service.get_preferences()

        # ── 6. Notifier ───────────────────────────────────────────────────
        notification_host = settings.host
        if notification_host in {"0.0.0.0", "::", ""}:
            notification_host = "127.0.0.1"
        if ":" in notification_host and not notification_host.startswith("["):
            notification_host = f"[{notification_host}]"
        notifier = create_notifier(
            api_base_url=f"http://{notification_host}:{settings.port}"
        )

        # ── 7. Wave 7: Effectiveness service (needed by report service) ────
        effectiveness_service = EffectivenessService(
            activity_repo=activity_repository,
            intervention_repo=intervention_repository,
        )

        # ── 7-ext. ML models (scikit-learn behaviour analysis) ─────────────
        # Three-tier degradation chain mirrors LLM's DeepSeek -> Ollama -> RuleEngine:
        #   Tier 1: Trained ML models available  -> ML + rule engine enrichment
        #   Tier 2: Models not found / load fail -> rule engine only (current)
        #   Tier 3: ML inference fails at runtime -> log warning, rule-only fallback
        from mindflow.train.models.manager import ModelManager  # noqa: PLC0415

        v2_model_manager: ModelManager | None = None
        v2_training_mode = "rule_engine_only"
        model_base_dir = settings.models_dir

        # V1 model loading removed — only V2 (24-dim feature schema) is supported.

        v2_report_path = model_base_dir / "v2" / "training_report.json"
        v2_report_mode: str | None = None
        v2_report_version: str | None = None
        if v2_report_path.exists():
            try:
                v2_report = json.loads(v2_report_path.read_text(encoding="utf-8"))
                if v2_report.get("model_mode") == "shadow":
                    v2_training_mode = "shadow"
                v2_report_mode = v2_report.get("model_mode")
                v2_report_version = v2_report.get("version_tag")
            except (json.JSONDecodeError, OSError) as exc:
                logger.opt(exception=True).warning(
                    "Failed to parse training report {}: {}", v2_report_path, exc
                )

        try:
            _v2_model_manager = ModelManager(
                models_dir=model_base_dir / "v2",
                use_ensemble=False,
            )
            if _v2_model_manager.load_latest():
                v2_model_manager = _v2_model_manager
                prediction_service.attach_model_manager(v2_model_manager)
                telemetry_service.attach_model_manager(v2_model_manager)
                v2_training_mode = "ready"
                loaded_tag = _v2_model_manager.current_version_tag
                if v2_report_mode == "shadow":
                    v2_training_mode = "shadow"
                elif v2_report_mode == "ready" and (
                    v2_report_version is None or v2_report_version == loaded_tag
                ):
                    v2_training_mode = "ready"
                elif v2_report_mode == "ready":
                    v2_training_mode = "shadow"
                else:
                    v2_training_mode = "ready"
                logger.info(
                    "Feature schema v2 model loaded (version: {})",
                    v2_model_manager.current_version_tag,
                )
        except Exception as exc:
            logger.warning("Failed to load feature schema v2 model: {}", exc)

        # ── 7-ext. Training job service (manual V2 model training) ──────────
        training_job_service = TrainingJobService(
            telemetry_repo=telemetry_repository,
            focus_repo=focus_repository,
            user_id=1,
        )

        # ── 7a. Wave 5 Services ────────────────────────────────────────────
        analysis_service = AnalysisService(
            activity_repo=activity_repository,
            focus_repo=focus_repository,
            timezone=settings.timezone,
        )
        report_service = ReportService(
            activity_repo=activity_repository,
            focus_repo=focus_repository,
            report_repo=report_repository,
            effectiveness_svc=effectiveness_service,
            timezone=settings.timezone,
        )
        maintenance_service = MaintenanceService(
            engine=engine,
            session_factory=session_factory,
            notifier=notifier,
            data_dir=data_dir,
            preferences_repository=preferences_repository,
        )

        # ── 7b. Wave 6: Provider registry + LLM service ───────────────────
        provider_registry = ProviderRegistry(settings.llm)
        llm_service: LLMService | None = None
        try:
            llm_service = LLMService(
                activity_repo=activity_repository,
                analysis_repo=analysis_repository,
                ollama_base_url=(
                    settings.llm.ollama_base_url
                    if settings.llm.ollama_enabled
                    else None
                ),
                ollama_model=settings.llm.ollama_model,
                timezone=settings.timezone,
                provider_registry=provider_registry,
            )
            logger.info(
                "LLMService created (L1: {}, L2: {})",
                "yes" if provider_registry.get_structured_attribution() else "no",
                settings.llm.ollama_enabled,
            )
        except Exception as exc:
            logger.warning("Failed to create LLMService: {}", exc)

        # ── 7c. Wave 7: Intervention service ───────────────────────────────
        intervention_throttle = InterventionThrottle(
            repo=intervention_repository,
            daily_limit=settings.throttle_daily_limit,
            type_limit=settings.throttle_type_limit,
            cooldown_h=settings.throttle_cooldown_hours,
            ignore_rate_threshold=settings.throttle_ignore_rate_threshold,
            fatigue_daily_limit=settings.throttle_fatigue_daily_limit,
            annoying_threshold=settings.throttle_annoying_threshold,
        )

        # LLM client for AI-generated intervention messages
        intervention_llm_client: httpx.AsyncClient | None = None
        intervention_llm_model = "deepseek-chat"
        if settings.llm.api_key:
            llm_base_url = (settings.llm.base_url or "https://api.deepseek.com").rstrip("/")
            intervention_llm_client = httpx.AsyncClient(
                base_url=llm_base_url,
                timeout=httpx.Timeout(10.0),
                headers={
                    "Authorization": f"Bearer {settings.llm.api_key}",
                    "Content-Type": "application/json",
                },
            )
            intervention_llm_model = settings.llm.model or "deepseek-chat"
            logger.info("Intervention LLM client created for AI message generation")
        else:
            logger.info("No LLM API key - intervention messages will use templates")

        intervention_service = InterventionService(
            intervention_repo=intervention_repository,
            throttle=intervention_throttle,
            notifier=notifier,
            activity_repo=activity_repository,
            broadcast_fn=broadcast,
            llm_client=intervention_llm_client,
            llm_model=intervention_llm_model,
            auth_token=system_token,
            ollama_base_url=(
                settings.llm.ollama_base_url
                if settings.llm.ollama_enabled
                else None
            ),
            ollama_model=settings.llm.ollama_model,
        )

        # ── 7d. G003: Panel service ──────────────────────────────────────────
        panel_service: PanelService | None = None
        analysis_workflow_port: AnalysisWorkflowPort | None = None
        shared_evidence_builder: EvidenceBundleBuilder | None = None
        if llm_service is not None:
            try:
                gateway = provider_registry.get_gateway()
                # Create shared EvidenceBundleBuilder for Panel + Chat
                shared_evidence_builder = EvidenceBundleBuilder(
                    activity_repo=activity_repository,
                    intervention_repo=intervention_repository,
                    session_factory=session_factory,
                    effectiveness_service=effectiveness_service,
                    baseline_repo=baseline_repository,
                    prediction_service=prediction_service,
                )

                # ── Create AnalysisGraph (v2 is the only analysis path) ──
                workflow_run_repo: Any = WorkflowRunsRepository(session_factory)
                workflow_budget_repo: Any = BudgetReservationRepository(session_factory)

                # RuleEngine is a lightweight deterministic engine — safe to
                # create here (no DB state, no HTTP clients).
                from mindflow.domain.procrastination import RuleEngine as _RuleEngine

                deepseek_client: Any | None = provider_registry.get_structured_attribution()

                # ── G002: Explicit PanelGraph wired into AnalysisGraph ──
                # Shares the gateway/provider lifecycle; the compiled graph
                # is built lazily on first access and reused across calls.
                panel_graph = PanelGraph(gateway=gateway)

                analysis_graph = AnalysisGraph(
                    analysis_repo=analysis_repository,
                    workflow_run_repo=workflow_run_repo,
                    budget_repo=workflow_budget_repo,
                    evidence_builder=shared_evidence_builder,
                    crisis_detector=CrisisDetector(),
                    panel_graph=panel_graph,
                    deepseek_client=deepseek_client,
                    ollama_base_url=(
                        settings.llm.ollama_base_url
                        if settings.llm.ollama_enabled
                        else None
                    ),
                    ollama_model=settings.llm.ollama_model,
                    rule_engine=_RuleEngine(),
                    timezone=settings.timezone,
                )
                analysis_workflow_port = analysis_graph
                logger.info("AnalysisGraph created as shared AnalysisWorkflowPort")

                panel_service = PanelService(
                    activity_repo=activity_repository,
                    intervention_repo=intervention_repository,
                    session_factory=session_factory,
                    llm_service=llm_service,
                    effectiveness_service=effectiveness_service,
                    timezone=settings.timezone,
                    analysis_repository=analysis_repository,
                    evidence_builder=shared_evidence_builder,
                    workflow_port=analysis_workflow_port,
                )
                logger.info("PanelService created with v2 AnalysisGraph workflow port")
            except Exception as exc:
                logger.warning("Failed to create PanelService: {}", exc)
        else:
            logger.warning("LLMService not available, skipping PanelService creation")

        # ── 7e. G004: Chat service ────────────────────────────────────────────
        chat_service: ChatService | None = None
        try:
            crisis_detector = CrisisDetector()
            chat_gateway = provider_registry.get_gateway()
            # Use the shared evidence builder from panel section, or create one
            shared_evidence = shared_evidence_builder
            if shared_evidence is None:
                shared_evidence = EvidenceBundleBuilder(
                    activity_repo=activity_repository,
                    intervention_repo=intervention_repository,
                    session_factory=session_factory,
                    effectiveness_service=effectiveness_service,
                    baseline_repo=baseline_repository,
                    prediction_service=prediction_service,
                )
            chat_service = ChatService(
                session_factory=session_factory,
                crisis_detector=crisis_detector,
                llm_gateway=chat_gateway,
                analysis_repo=analysis_repository,
                panel_service=panel_service,
                intervention_repo=intervention_repository,
                evidence_builder=shared_evidence,
                chat_repo=chat_repository,
                max_history_rounds=settings.max_history_rounds,
                timezone=settings.timezone,
                model=provider_registry.get_chat_model(),
                provider_registry=provider_registry,
            )
            logger.info("ChatService created for G004 conversational assistant")
        except Exception as exc:
            logger.warning("Failed to create ChatService: {}", exc)

        # ── Extract ChatGraph from ChatService for runtime state ──────────
        chat_graph = (
            getattr(chat_service, "_chat_graph", None)
            if chat_service is not None
            else None
        )
        if chat_graph is not None:
            logger.info("ChatGraph attached to runtime (v2 chat graph)")

        # ── 8. Scheduler (Wave 5 cron jobs) ───────────────────────────────
        scheduler = build_scheduler(
            analysis_service=analysis_service,
            report_service=report_service,
            maintenance_service=maintenance_service,
            intervention_service=intervention_service,
            activity_repository=activity_repository,
            panel_service=panel_service,
            autonomy_service=autonomy_service,
            telemetry_service=telemetry_service,
            scheduled_job_runs_repository=scheduled_job_runs_repository,
            workflow_port=analysis_workflow_port,
            event_retention_days=settings.event_retention_days,
            min_confidence=settings.auto_intervention_min_confidence,
            panel_confidence=settings.auto_intervention_panel_confidence,
            start_hour=settings.intervention_start_hour,
            end_hour=settings.intervention_end_hour,
            timezone=settings.timezone,
        )
        runtime = RuntimeServices(
            settings=settings,
            engine=engine,
            session_factory=session_factory,
            scheduled_job_runs_repository=scheduled_job_runs_repository,
            scheduler=scheduler,
            collector_service=collector_service,
            input_telemetry_service=input_telemetry_service,
            panel_service=panel_service,
            chat_service=chat_service,
            chat_graph=chat_graph,
            llm_service=llm_service,
            prediction_service=prediction_service,
            evidence_builder=shared_evidence_builder,
            provider_registry=provider_registry,
            workflow_port=analysis_workflow_port,
            checkpointer=checkpointer,
        )
        await _start_runtime_services(
            runtime,
            input_telemetry_enabled=bool(
                telemetry_preferences.get("input_telemetry_enabled", False)
            ),
        )
        _publish_runtime_state(app, runtime)
        logger.info(
            "Runtime services started (scheduler={}, collectors={})",
            settings.run_scheduler,
            settings.run_collectors,
        )

        # ── Inject into app.state ─────────────────────────────────────────
        app.state.engine = engine
        app.state.session_factory = session_factory
        app.state.activity_repository = activity_repository
        app.state.preferences_repository = preferences_repository
        app.state.telemetry_repository = telemetry_repository
        app.state.telemetry_service = telemetry_service
        app.state.input_telemetry_service = input_telemetry_service
        app.state.classification_rules_repository = classification_rules_repository
        app.state.collector_service = collector_service
        app.state.system_token = system_token
        app.state.bootstrap_tickets = bootstrap_tickets
        app.state.browser_sessions = browser_sessions
        app.state.migration_applied = migration_applied
        app.state.db_integrity_ok = db_ok
        app.state.notifier = notifier
        app.state.focus_repository = focus_repository
        app.state.report_repository = report_repository
        app.state.analysis_repository = analysis_repository
        app.state.baseline_repository = baseline_repository
        app.state.analysis_service = analysis_service
        app.state.scheduled_job_runs_repository = scheduled_job_runs_repository
        app.state.report_service = report_service
        app.state.maintenance_service = maintenance_service
        app.state.scheduler = scheduler
        app.state.llm_service = llm_service
        app.state.panel_service = panel_service
        app.state.intervention_repository = intervention_repository
        app.state.intervention_service = intervention_service
        app.state.intervention_llm_client = intervention_llm_client
        app.state.effectiveness_service = effectiveness_service
        app.state.v2_model_manager = v2_model_manager
        app.state.v2_training_mode = v2_training_mode
        app.state.training_job_service = training_job_service
        app.state.chat_service = chat_service
        app.state.chat_graph = chat_graph
        app.state.autonomy_service = autonomy_service
        app.state.prediction_service = prediction_service
        app.state.checkpointer = checkpointer
        app.state.workflow_port = analysis_workflow_port
        if shared_evidence_builder is not None:
            app.state.shared_evidence_builder = shared_evidence_builder

        logger.info("MindFlow v{} startup complete", __version__)

        yield  # ── Application runs here ──
    finally:
        logger.info("Shutting down MindFlow...")
        # Cancel any active training job before tearing down services.
        _ts = getattr(app.state, "training_job_service", None)
        if _ts is not None:
            try:
                await _ts.shutdown()
            except Exception as exc:
                logger.warning("Training job service shutdown error: {}", exc)
        if runtime is not None:
            await _shutdown_runtime_services(runtime)
        else:
            if checkpointer is not None:
                try:
                    await checkpointer.aclose()
                except Exception as exc:
                    logger.warning("Checkpointer close error during failed startup: {}", exc)
            try:
                await engine.dispose()
            except Exception as exc:
                logger.warning("Engine dispose error during failed startup: {}", exc)
        # Exit the checkpointer async context manager (cleanup the generator).
        if checkpointer_ctx is not None:
            try:
                await checkpointer_ctx.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("Checkpointer ctx exit error: {}", exc)
        try:
            n_closed = await close_all_connections()
            logger.debug("Closed {} active WebSocket connection(s)", n_closed)
        except Exception as exc:
            logger.warning("WebSocket close error: {}", exc)
        # Close intervention LLM client if created
        _ilc = getattr(app.state, "intervention_llm_client", None)
        if _ilc is not None:
            try:
                await _ilc.aclose()
            except Exception as exc:
                logger.warning("Intervention LLM client close error: {}", exc)
        logger.info("MindFlow shutdown complete")


# ── App factory ────────────────────────────────────────────────────────────


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure a MindFlow FastAPI application instance.

    Args:
        settings: Application settings. If None, loads from defaults.

    Returns:
        A fully configured FastAPI application ready to serve.
    """
    if settings is None:
        from mindflow.config import get_settings

        settings = get_settings()

    # Configure logging
    setup_logging(settings)

    app = FastAPI(
        title="MindFlow API",
        description="Local-first intelligent focus assistant",
        version=__version__,
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Store settings for lifespan access
    app.state.settings = settings

    # ── Exception handlers (wraps everything) ─────────────────────────
    register_exception_handlers(app)

    # ── Middleware (§3.5) ────────────────────────────────────────────────
    # Starlette's add_middleware is LIFO: the LAST middleware added becomes
    # the OUTERMOST layer and therefore runs FIRST on each request. The list
    # below is in ADDITION order; actual request-processing order is the
    # reverse (Auth → RateLimit → Host → Logging → route).
    #
    # F2 fix: Auth is added AFTER RateLimit (so Auth processes the request
    # BEFORE RateLimit does). Previously Auth was added before RateLimit,
    # which made RateLimit the outer/first-run layer — an unauthenticated
    # request would consume rate-limit budget before ever being rejected by
    # Auth, letting a token-less local process exhaust the global bucket (and
    # tiny per-endpoint daily caps, e.g. panel's daily_hard_limit=3) and lock
    # out the legitimate authenticated client. Auth must gate before
    # metering: an unauthenticated request should get a cheap 401 without
    # touching any bucket.

    # 1. StructuredLoggingMiddleware (request_id + timing)
    app.add_middleware(StructuredLoggingMiddleware)

    # 2. HostValidationMiddleware (localhost only)
    app.add_middleware(HostValidationMiddleware)

    # 3. RateLimitMiddleware (token bucket) — added before Auth so Auth
    # ends up outer and runs first (see LIFO note above).
    app.add_middleware(RateLimitMiddleware)

    # 4. AuthMiddleware (token check, exempt /health and /docs)
    # Token is read from app.state.system_token at request time,
    # so it doesn't need to be set during construction.
    app.add_middleware(AuthMiddleware)

    # ── Routes ─────────────────────────────────────────────────────────
    register_routes(app)

    # ── WebSocket ──────────────────────────────────────────────────────
    app.include_router(websocket_router, prefix="/api/v1")

    # ── Startup security headers (via middleware) ──────────────────────

    @app.middleware("http")
    async def add_security_headers(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Add security headers to every response."""
        response = await call_next(request)
        response.headers["X-MindFlow-Version"] = __version__
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Hardens the HTML docs pages against XSS (security audit L1);
        # inline allowances are required by Swagger UI.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: "
            "https://fastapi.tiangolo.com"
        )
        return response

    frontend_dist = _frontend_dist_dir()
    if (frontend_dist / "index.html").is_file():
        app.mount(
            "/",
            SPAStaticFiles(directory=frontend_dist, html=True),
            name="frontend",
        )
    else:
        logger.warning("Frontend build not found at {}; API-only mode", frontend_dist)

    logger.info("MindFlow app created (v{})", __version__)
    return app
