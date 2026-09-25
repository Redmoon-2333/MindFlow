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

# Lint — Ruff (green since 2026-08-16)
uv run python -m ruff check src tests

# Type check — strict mypy (green since 2026-08-16)
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
- **PanelGraph** (`src/mindflow/graph/panel_graph.py`): Explicit AnalysisGraph subgraph for expert deliberation (Analyst -> 3x Attribution parallel -> validation -> Moderator -> Critic). The legacy `PanelOrchestrator` class was removed at v2 cutover; v2 is the production route.
- **ChatGraph** (`src/mindflow/graph/chat_graph.py`): Explicit chat lifecycle StateGraph, independent from analysis. It is the only production chat path; the former LangChain `create_agent` path has been removed.
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

All flags live in `Settings` (Pydantic BaseSettings) with `MINDFLOW_` env-var prefix.
The backend is v2-only for analysis and chat. The graph-selection fields remain as
deprecated compatibility inputs for older `.env` files, but they do not change routing.

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `checkpointing_enabled` | bool | `False` | Use SQLite-backed LangGraph checkpoints instead of the in-memory checkpointer. |
| `training_use_window_labels` | bool | `True` | Also train on user-calibrated `behavior_feature_windows.label` (weight 0.8; feedback still wins; quality-gate counts stay feedback-only). Measured 2026-08-20: BA 0.46→0.64, Brier 0.40→0.23 — activated the model. |
| `new_analysis_graph` | bool | `True` | Deprecated compatibility input; v2 AnalysisGraph is always active |
| `new_chat_graph` | bool | `True` | Deprecated compatibility input; v2 ChatGraph is always active |
| `panel_fast_path_enabled` | bool | `False` | Panel fast path (plan 2.2). Must stay `False`: the offline replay measured 60% top-1 agreement on the 5/30 scenarios it triggered (target ≥95%). |
| `deepseek_api_key` | str | unset | Production L1 credential (`DEEPSEEK_API_KEY` / `MINDFLOW_LLM__DEEPSEEK_API_KEY`). Absent → L1 unavailable, degrade to L2/L3; the legacy ECNU key is never reused. |
| `ecnu_compat_enabled` | bool | `False` | Honour the legacy ECNU triple instead of the DeepSeek pin. Compatibility tests/diagnostics only; **must stay `False` in production**. |

## L1 = DeepSeek Direct (2026-09-25)

- Production L1 is DeepSeek itself: `provider=generic`,
  `base_url=https://api.deepseek.com`, `model=deepseek-flash`, resolved once by
  `LLMSettings.l1_target()` and consumed by `ProviderRegistry` (structured client,
  gateway, chat model). Legacy ECNU URL/model/provider/key values are overridden
  and never mixed; a missing DeepSeek credential means L1 is unavailable — there
  is no ECNU fallback.
- `chat`/`reasoner` are **output policies** (JSON mode vs prose), not two model
  ids: both tiers request the same model. Role-level `reasoning_effort`,
  `thinking` and `max_tokens` travel per request via `CompletionPolicy`
  (official spelling: `max_tokens`, `extra_body={"thinking": {...}}`).
- Thinking mode requires the previous assistant turns' `reasoning_content` to be
  echoed on any request carrying `tools` (DeepSeek answers 400 otherwise);
  `infrastructure/llm/thinking.py` is the shared implementation used by both the
  ECNU and the new DeepSeek adapters. `temperature` is omitted for thinking
  requests. Non-retriable 4xx responses are never retried with identical
  parameters.

**2026-08-20 additions**: production training defaults to Platt(sigmoid) calibration (`run_training(calibration="sigmoid")` makes `evaluate_v2_candidates` and `ModelManager` share it; `make_v2_classifier()` public default stays raw — small/toy datasets pass `calibration=None`); `_PANEL_WORKFLOW_TIMEOUT_S=120` (was 8, which always timed out against real DeepSeek); critic prompt capped (`critique_detail ≤300字`) to avoid the 8192-token truncation that broke JSON parsing.

**Rollback**: The pre-v2 implementations are no longer shipped. Roll back by deploying a
previous application revision; changing the deprecated graph flags only preserves config
file compatibility and does not restore legacy behavior.

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

- **Data presence != trainability**: raw activity events must roll up into V3 feature windows (schema_version=3) via telemetry; explicit feedback timestamps must overlap window ranges.
- **Baseline and ML share one UI route** (`/model-center`) but remain separate backend lifecycles: baseline is Welford online incremental, ML is batch offline.
- **Job state is in-memory**: `TrainingJobService` holds `_current: _JobState | None`; restart loses job observation. No SQLite persistence for training job state.
- **One job per process**: `asyncio.Lock` guards creation; duplicate `start_job` returns 409.
- **Cancel window**: only before CPU training (`pending` / `preparing_data`). Once status = `training`, cancel returns 409 because the offloaded thread may call `save_all(activate=True)`.
- **Shadow never replaces active**: `shadow` model_mode updates `app.state.v2_training_mode` only; `v2_model_manager` and attached services unchanged.
- **Ready publication failure == job failure**: if quality gate passes but `_refresh_ready_manager()` raises, job.status = `failed`, not `succeeded`.
- **No auto-retraining**: scheduler has no training cron (hardened by test).
- **Quality gates now implemented (2026-07-31)**: `calibration_better_than_rule` and `stable_date_folds` are real checks; readiness no longer reports them as `not_implemented`.

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

## 2026-07-31 ML/LangGraph 升级摘要

- Feature schema 已升级到 **v4**（28 列：v3 的 24 列 + task-context 4 列）；切换计数必须使用 `count_confirmed_switches()`（驻留 10 秒 + 瞬时进程忽略）。
- ML 质量门使用唯一反馈会话数（>=20 条）、7 个反馈日、每类 >=5 条；forward-chaining 内计算 rule baseline 与校准。
- `POST /panel/today` 支持 `force`/`retry_if_degraded`，缓存命中保留降级元数据。
- `PanelGraph` 是唯一活动面板图；旧 `PanelOrchestrator` 类已在 v2 cutover 中删除，解析/校验 helper 保留在 `agents/orchestrator.py`。
- 实验统一使用 `scripts/run_experiments.py`，最终报告见 `data/experiments/20260731_final/`。

## 2026-09-25 分阶段优化方案落地摘要

分阶段优化方案（正确性 → 降本 → ML → 基础设施）在本轮落地，关键设计：

- **面板终态**：`panel_finalize` 单终态节点写入 `critic_approved`/`panel_rejected`/`panel_terminal`/`rejection_reason`；只有「裁决存在 + schema 通过 + critic 批准」才是 `source="panel"`，驳回一律走 L1→L2→L3 降级链，`panel_degradation_marker` 前缀进 `degradation_path`。
- **训练激活策略**：`run_training(allow_activation=...)`、`TrainingJobService.start_job(allow_activation=...)`；auto 训练固定 shadow-only，手动/独立发布任务才可移动 active 指针（`TrainingReport.activation_allowed/activation_suppressed_reason`，响应体 `allow_activation`，CLI `--no-activate`）。
- **Prompt/schema 统一**：`extra="forbid"` + 代码级语义校验（空论据、无合法类型、无证据、置信度越界、insufficient_data 无 gaps 均无效）。
- **Claim Ledger**：`agents/claims.py`（校验、冲突摘要、共识惩罚、verdict 置信度上限）；Moderator 只吃校验过的 claim 表 + 证据表 + 冲突摘要。
- **证据压缩**：默认压缩 payload（稳定项折叠 `stable_summary`，catalog 为位置元组 `[id, 中文标签, 类型]`）。实测：稳定型真实窗口 −32.8%（达标 ≥30%），生产形态 9 项窗口 −20.1%，30 个评估场景合计 −12.0% 且全部变小；citation id 集合与 legacy 完全一致。`compressed=False` 仅供 A/B。
- **LLM 调用策略/可观测性**：`agents/policies.py` 角色级 `CompletionPolicy`（analyst 1200 / attribution 1000 / rebuttal 800 / moderator 1600 / critic 500 / chat 2048，moderator=high reasoning）；`services/llm_observability.py` 记录延迟/token/重试/解析失败/降级原因等聚合元数据，不含 prompt、密钥或供应商正文。
- **ML 协议**：主评估 forward-chaining（leave-future-days-out），窗口标签只训练不进评估；session/date 级指标、PR-AUC/Brier/ECE/reliability、按 session 重采样的 bootstrap CI、abstention coverage；候选固定五选一（rule/LogReg/RF/XGB/RF+XGB soft voting），复杂模型必须稳定胜出；HMM 退出默认发布链；质量门不因口径调整而放宽。
- **基础设施**：SQLite 热查询经 `EXPLAIN QUERY PLAN` 验证走索引（`tests/test_sqlite_query_plans.py`）；`rollup_recent()` watermark 增量重算（失败不推进）；Ollama/HTTP client 由 ProviderRegistry 单例持有；并发门默认仍为 1，压测脚本 `scripts/experiment_concurrency.py`。
- **主动反馈**：`services/feedback_sampling.py` 仅在不确定/规则-ML 冲突/Panel 分歧/任务上下文不一致/数据漂移时请求标注，并记录触发原因与信息增益代理指标。
- **快速路径**：`MINDFLOW_PANEL_FAST_PATH_ENABLED` 默认 `False`；离线回放 `scripts/experiment_fast_path.py` 实测快速路径 5/30 场景、top-1 一致率 60%（<95% 门槛），**因此保持关闭**。

## Quality Gates (Green since 2026-08-16)

The following commands are **required visibility gates** and are **green**:

- **Ruff**: 0 findings (`uv run python -m ruff check src tests` → `All checks passed!`).
- **Mypy (strict)**: 0 errors across 185 source files (`uv run python -m mypy --strict src/mindflow` → `Success`).
- **Pytest**: 3120 passing, 4 skipped, 22 warnings (`uv run python -m pytest tests/ -q`).

**Live DeepSeek smoke (2026-09-25, `scripts/smoke_live_llm.py --yes --max-requests 16`, isolated data dir)**: all four targets PASS — panel 9 requests / 9-9 schema-valid / 7-7 valid citations / 0% failure / p95 26.7s; chat 1 request / p95 2.1s; tools 2 requests (a real tool call plus a successful follow-up turn, proving the `reasoning_content` echo works); attribution 1 request / 1-1 schema / p95 7.5s. 13 requests total, 20,786 input tokens, 12,755 output tokens of which 7,075 were reasoning (55%) — every response HTTP 2xx, no degradation, no 4xx retry. Artifacts: `data/experiments/20260925_051755/{summary.json,report.md}` (no keys, prompts or bodies).

## Gray-Release Round (2026-09-25 下午)

Six pre-launch items closed; the full suite was **3116 passing, 4 skipped** at that acceptance checkpoint. The latest full run on 2026-09-25 is **3120 passing, 4 skipped, 22 warnings**; ruff and mypy strict are green.

1. **Attribution usage fixed**: `DeepSeekClient.analyze()` captures the provider's `usage` into `last_usage`, `single_expert_node` records it, and the raw client now sends the same reasoning contract as the gateway (`reasoning_effort` / `thinking` / `max_tokens`). Live re-test: `usage_reported=true` (3 calls, reasoning = 85% of output).
2. **Panel fan-out optimization (targeted)**: new `panel_fanout_concurrency` (default **3**) routes only the panel's parallel attribution/rebuttal batches through a dedicated registry gateway with its own burst gate; every other L1 path keeps the global limit 1. Real A/B on 2 scenarios × 9 calls: wall time 114.5s → 60.5s (**-47%**), serial queueing 82.5s → 0ms (**-100%**), identical verdict types and escalation shape, zero errors. Set `MINDFLOW_LLM__PANEL_FANOUT_CONCURRENCY=1` to revert.
3. **Multi-scenario / repeat smoke**: `--scenario-count N --repeat M` (panel over the first N eval scenarios; per-target aggregation of schema, citations, failure rate, p95 and usage). Gray artifacts: `data/experiments/gray_multiscenario_c1`, `gray_fanout_c3_v2`, `gray_repeat_chat_tools` — all PASS.
4. **v4 backfill accepted on a copy of the real DB**: 6265 v4 windows rebuilt / written, 2401 v3 labels inherited, 0 failed; v2 (1726/198) and v3 (8198/2981) row and label counts identical before/after; re-run idempotent (6265/2401 unchanged); 153 blocks beyond raw-event retention are reported, not synthesized. **Production apply still pending your explicit run** (backup first).
5. **Production DB was migrated to 0027 during the rehearsal** (see the incident note in the final report): alembic 0024→0027 applied additively; integrity ok; row counts unchanged; pre-upgrade byte copy retained at `%TEMP%\mf_backfill_acceptance.db`.
6. **Publication-guard acceptance — all three paths PASS** (`scripts/acceptance_publication_guard.py`, idempotent, exit 0): Run A mismatch → shadow + `REASON_MISMATCH` in report and manifest; Run B (engineered dataset where the evaluation itself selects the ensemble: 896 windows, drift PSI 0.21 < 0.25, ensemble promoted over RF by +0.0247 with 3/4 fold wins) → consistent → **activated, `model_mode="ready"`, `latest.json` moved**; Run C `allow_activation=False` → shadow-only regardless of the verdict. The activation path requires xgboost installed (absent → `REASON_XGB_FALLBACK` by design).

Any change that regresses one of these gates must be fixed before the work is
considered done.

## Dataset Context

`data/datasets/` contains external datasets for local training only, not committed to Git:
- `manictime/`: 44 real user activity CSV exports (ManicTime, contains PII)
- `awt-labelled/`: Academic Work Tracker labelled data + preprocessing notebook

V2 synthetic feature-window generation lives in `train/synthetic_v2.py` and uses the
student archetypes from `train/user_profiles.py`. Model artifacts and
`training_report.json` are written under `data/models/v2/`.

## Docs

- `backend-next/README.md` — Backend quickstart and architecture
- `../docs/architecture/ADR-001..005` — Architecture Decision Records
- `../docs/handbook/` — Full-stack handbook (6 chapters)
- `../docs/redesign/` — Redesign documents (requirements, architecture, testing, technology usage, agent upgrades)
