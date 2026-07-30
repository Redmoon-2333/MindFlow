# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Active Backend

Current active backend is **`backend-next/`** (FastAPI rewrite, layered architecture + LangChain/LangGraph).
Legacy `backend/` (Phase 0, sync SQLAlchemy, no LLM layer) has been deleted; `backend-next` has zero dependency on it.

This document focuses on the active backend. Frontend-specific commands are maintained separately.

## Build & Test Commands (uv-managed)

All commands run from `mindflow-app/backend-next/`. Python 3.11+ required. Dependency management uses **uv** (not pip/conda).

```bash
cd mindflow-app/backend-next

# Install all dependencies (dev + ML extras)
uv sync --extra dev --extra ml

# Activate the venv (Windows: .venv\Scripts\activate, macOS/Linux: source .venv/bin/activate)
# Or prefix all commands with `uv run`

# Start production service (watchdog auto-restarts on crash)
uv run python -m mindflow.main
# Note: create_app(settings) is a parameterised factory — not compatible with `uvicorn --factory`.
# Hot-reload: Ctrl+C then re-run `uv run python -m mindflow.main` (startup <2s).

# Generate local bootstrap login link (one-time ticket)
uv run python -m mindflow.bootstrap

# Run full test suite (1956 passed, 12 skipped, 1 warning as of 2026-07-29)
uv run python -m pytest tests/ -q

# Run a single test file / single test case
uv run python -m pytest tests/test_llm_client.py -v
uv run python -m pytest tests/test_features.py::test_calculate_focus_score -v

# Lint — Ruff (94 findings currently open)
uv run python -m ruff check src tests

# Type check — strict mypy (158 errors in 16 files currently open)
uv run python -m mypy --strict src/mindflow

# Database migrations
uv run alembic history          # show migration chain
uv run alembic current          # show current version
uv run alembic upgrade head     # apply all pending migrations
# WARNING: `alembic downgrade` on SQLite must use an isolated/backup DB.
# SQLite has limited ALTER TABLE support; always backup first:
#   sqlite3 mindflow.db ".backup mindflow_pre_migration.db"

# Train ML models (synthetic data / real data / version management)
uv run python -m mindflow.train --source synthetic_v2
uv run python -m mindflow.train --source db
uv run python -m mindflow.train --list-versions

# Evaluation harness (mock deterministic replay — no API key needed)
uv run python -m mindflow.eval --mode both
# NOTE: --mode "mock" is invalid. "both" = rule engine + mock panel gateway.
# Real LLM evaluation (requires API key + confirmation):
uv run python -m mindflow.eval --mode both --live --yes
```

Windows runtime dependencies (`psutil`, `pywin32` on Win32) are managed by uv via `pyproject.toml` platform markers.

## Architecture

MindFlow is a local-first intelligent focus assistant: monitors computer usage behaviour, analyses patterns, and generates personalised anti-procrastination interventions.

```
Frontend (React/TS) <-> Backend (FastAPI :8765) <-> Collector (cross-platform activity collection)
                              |
                         SQLite (WAL mode, local)
```

**Layered dependency direction**: `domain` -> `infrastructure` -> `services` -> `api` / `agents` (one-way, irreversible).

| Layer | Path | Responsibility |
|-------|------|----------------|
| `config` | `src/mindflow/config.py` | Pydantic BaseSettings from `.env`/env vars; `{data_dir}` placeholder resolution |
| `domain` | `src/mindflow/domain/` | Pure domain models: events, features, baseline, deviation, procrastination types, evidence contracts. Zero framework dependencies (stdlib + typing only) |
| `infrastructure` | `src/mindflow/infrastructure/` | Collectors (Win32/macOS/X11/Wayland), SQLAlchemy repositories, LLM client, security (token/crisis detection), notification, provider registry |
| `services` | `src/mindflow/services/` | Business orchestration: analysis, report, intervention, throttle, evidence building, panel, chat, scheduler, maintenance, export, **training_readiness** (V2 data assessment + 7 quality gates), **training_job** (async in-process training lifecycle) |
| `agents` | `src/mindflow/agents/` | Multi-expert LLM panel (LangGraph StateGraph): orchestrator + 5 experts + conflict detection + LangChain gateway |
| `graph` | `src/mindflow/graph/` | AnalysisGraph (framework-neutral workflow port), PanelGraph, ChatGraph definitions; ADR-004 boundaries |
| `api` | `src/mindflow/api/` | REST routes + WebSocket + middleware (auth/host/ratelimit/logging) + RFC 9457 error handling |
| `train` | `src/mindflow/train/` | ML training pipeline (synthetic/real data, clustering, classification, HMM, version management). Previously purely offline CLI; now also callable from `TrainingJobService` via `asyncio.to_thread(run_training, ...)`. The V2 pipeline reads feature windows + focus feedback and produces `TrainingReport` with quality gate results. |
| `eval` | `src/mindflow/eval/` | Evaluation suite (30 scenarios) + mock/real LLM comparison runner |

### Orchestration Architecture (ADR-001, ADR-002, ADR-004)

- **Framework-neutral ports** (`src/mindflow/ports.py`): Protocol interfaces (`AnalysisWorkflowPort`, `WorkflowRunStorePort`, `BudgetReservationPort`) decouple the outer scheduler from the inner analysis graph. LangGraph can be replaced without touching the scheduler.
- **AnalysisGraph** (`src/mindflow/graph/analysis_graph.py`): Daily analysis composition root implementing `AnalysisWorkflowPort`; owns idempotency, budget, crisis gating, persistence, and fallback routing.
- **PanelGraph** (`src/mindflow/graph/panel_graph.py`): Explicit AnalysisGraph subgraph for expert deliberation (Analyst -> 3x Attribution parallel -> validation -> Moderator -> Critic). The legacy `PanelOrchestrator` remains available when the v2 route is disabled.
- **ChatGraph** (`src/mindflow/graph/chat_graph.py`): Explicit chat lifecycle StateGraph, independent from analysis. The legacy LangChain `create_agent` path remains the default until `new_chat_graph=True`.
- **ProviderRegistry** (`src/mindflow/infrastructure/provider_registry.py`): Manages LLM provider lifecycle (L1 DeepSeek, L2 Ollama, L3 RuleEngine). Single HTTP session pool shut down atomically.
- **SQLite checkpointer**: LangGraph checkpoint persistence (off by default via `checkpointing_enabled=False`). Shares the same DB file.
- **Workflow run store**: `workflow_runs` + `workflow_node_events` tables track every analysis run with status, timing, call count, and degradation metadata. Exposed read-only via `/api/v1/ai/runs`.
- **Local OTel**: OpenTelemetry SDK configured with local SQLite exporter. No external OTLP/gRPC export. Span attributes never include raw window titles, file paths, or PII (per ADR-003).
- **Unified entry-point routing**: All analysis triggers (scheduler, API, chat tool, auto-intervention) converge through a single `AnalysisWorkflowPort` instance.

### Key Design Decisions

- **All-local data**: SQLite WAL mode, no cloud upload, privacy-first.
- **No global singletons**: Shared state on `app.state`, `create_app(settings)` factory assembly, dependency injection throughout.
- **Three-tier LLM degradation** (`config.LLMSettings`): L1 DeepSeek (key required) -> L2 Ollama local -> L3 RuleEngine (always available).
- **LLM output treated as untrusted**: Pydantic v2 strict + `extra="forbid"` + forbidden-word validators (NF-S7), citation code-enforced, independent crisis detector gates before LLM calls.
- **Async SQLAlchemy**: Each repository method opens its own `async with session_factory()`, no cross-request session sharing.
- **Timezone**: UTC everywhere internally.
- **Public API zero breakage**: All refactoring maintains backward-compatible interfaces.

## Feature Flags (ADR-005)

All flags live in `Settings` (Pydantic BaseSettings) with `MINDFLOW_` env-var prefix. **All default to legacy-safe paths.**

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `graph_version` | int | `1` | Reserved graph-version metadata. It does not select an implementation today. |
| `checkpointing_enabled` | bool | `False` | Use SQLite-backed LangGraph checkpoints instead of the in-memory checkpointer. |
| `new_analysis_graph` | bool | `False` | Route daily panel analysis through v2 AnalysisGraph instead of direct PanelOrchestrator |
| `new_chat_graph` | bool | `False` | Route chat through explicit ChatGraph StateGraph instead of LangChain create_agent |
| `shadow_mode_chat` | bool | `False` | Run both legacy and new chat paths, compare, return legacy output |

**Legacy-safe rollback**: Set all flags to defaults to restore pre-refactoring behaviour. No code revert needed. See ADR-005 for full migration/rollback procedure.

## Health & Diagnostics Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/api/v1/health/live` | GET | No | Process liveness (no dependencies touched) |
| `/api/v1/health/ready` | GET | No | Readiness: migration status + DB connection + integrity check + checkpoint/run-store probe. Returns 503 if not ready. |
| `/api/v1/health` | GET | No | Legacy health check (always 200). Includes collector, DB, ML, checkpoint, and run-store status. |
| `/api/v1/ai/runs` | GET | Yes | Paginated list of workflow runs (metadata only; allowlisted — no prompts, evidence, or PII) |
| `/api/v1/ai/runs/{run_id}` | GET | Yes | Single run detail with sanitised node events |

Auth uses a bootstrap-token/session model: a local root token generates short-lived bootstrap tickets exchanged for `HttpOnly`, `SameSite=Strict` session cookies. The `/health` and `/docs` endpoints are exempt from auth.

## Model Center & V2 Training Endpoints

| Endpoint | Method | Auth | Status | Description |
|----------|--------|------|--------|-------------|
| `/api/v1/analytics/training-readiness` | GET | Yes | 200 | Training readiness assessment: raw events, V2 windows, feedback labels, trainability (>=10 matched windows, >=2 classes), evaluability (>=10 explicit samples, >=3 distinct days), baseline readiness (>=30 samples), 7 quality gates, blockers. Injects `current_training_job` from `TrainingJobService`. |
| `/api/v1/analytics/training-jobs` | POST | Yes | 202/409/412 | Start a training job. Reads readiness gate first; returns 412 if not trainable, 409 if another job active. Returns `CreateTrainingJobResponse` with `job_id` + `status`. |
| `/api/v1/analytics/training-jobs/{job_id}` | GET | Yes | 200/404 | Full job lifecycle status: `status`, `source`, `model_mode`, `activated`, `version_tag`, `feature_schema_version`, `quality_gate`, `evaluation`, `error`. |
| `/api/v1/analytics/training-jobs/{job_id}/cancel` | POST | Yes | 200/404/409 | Cancel a pending/preparing job. Returns 409 once `training` phase started. |
| `/api/v1/analytics/baseline` | GET | Yes | 200/404 | Welford online baseline: `total_days`, `total_samples`, `features`, timestamps. |
| `/api/v1/analytics/model-status` | GET | Yes | 200 | V2 model manager status: `loaded`, `ready`, `mode`, `v2_mode`, `version`, `available_versions`, `reasons`. |

See full API contracts in [`docs/api/model-training.md`](docs/api/model-training.md).

## V2 Training Architecture Caveats

- **Data presence != trainability**: raw activity events must roll up into V2 feature windows (schema_version=2) via telemetry; explicit feedback timestamps must overlap window ranges.
- **Baseline and ML share one UI route** (`/model-center`) but remain separate backend lifecycles: baseline is Welford online incremental, ML is batch offline.
- **Job state is in-memory**: `TrainingJobService` holds `_current: _JobState | None`; restart loses job observation. No SQLite persistence for training job state.
- **One job per process**: `asyncio.Lock` guards creation; duplicate `start_job` returns 409.
- **Cancel window**: only before CPU training (`pending` / `preparing_data`). Once status = `training`, cancel returns 409 because the offloaded thread may call `save_all(activate=True)`.
- **Shadow never replaces active**: `shadow` model_mode updates `app.state.v2_training_mode` only; `v2_model_manager` and attached services unchanged.
- **Ready publication failure == job failure**: if quality gate passes but `_refresh_ready_manager()` raises, job.status = `failed`, not `succeeded`.
- **No auto-retraining**: scheduler has no training cron (hardened by test).
- **Two gates hardcoded as not_implemented**: `calibration_better_than_rule` and `stable_date_folds` are `_NotImplementedGate`; they expose `not_implemented` status in readiness and should not be interpreted as green passes.

## Focused Verification: Model Center / Training (2026-07-30)

The following verification was run after the model-center implementation, not as a full re-acceptance of all 1956 tests. It supplements the 2026-07-29 acceptance evidence above.

- **Backend readiness + jobs tests**: 33 passed (test_training_readiness.py + test_training_jobs.py, all cases)
- **Broader backend related tests**: 56 passed, 1 skipped (training_jobs + training_readiness + ml_integration + prediction_service + app_lifespan_runtime)
- **Ruff (focused)**: passed on training-related source (`training_readiness_service.py`, `training_job_service.py`, `analytics.py`)
- **Frontend build**: passed (`npm run build`)
- **Model center Playwright (E2E)**: 9/9 passed
- **Visual QA**: dual Oracle PASS, 375px no overflow
- **Frontend lint**: 3 pre-existing warnings in old E2E files (unrelated to model center)
- **Endpoints verified against source**: training-readiness (200 with full schema), training-jobs (202/409/412), training-jobs/{id} (200/404), training-jobs/{id}/cancel (200/404/409), baseline (200/404), model-status (200)
- **OpenAPI schema**: matches Pydantic models for all request/response bodies

## Quality Debt (Transparent)

The following commands are **required visibility gates** but are **not currently green**:

- **Ruff**: 94 findings (run `uv run python -m ruff check src tests`). Planned cleanup under slop-reduction workflow.
- **Mypy (strict)**: 158 errors across 16 files (run `uv run python -m mypy --strict src/mindflow`). Strict mode enforcement is ongoing.

Do NOT claim lint or type-clean status. These are tracked as known debt, not regressions.

## Dataset Context

`data/datasets/` contains external datasets for local training only, not committed to Git:
- `manictime/`: 44 real user activity CSV exports (ManicTime, contains PII)
- `awt-labelled/`: Academic Work Tracker labelled data + preprocessing notebook

Synthetic data generator in `train/synthetic_data.py`, models real weekday/weekend behaviour patterns.

## Docs

- `backend-next/README.md` — Backend quickstart and architecture
- `../docs/architecture/ADR-001..005` — Architecture Decision Records
- `../docs/handbook/` — Full-stack handbook (6 chapters)
- `../docs/redesign/` — Redesign documents (requirements, architecture, testing, technology usage, agent upgrades)
