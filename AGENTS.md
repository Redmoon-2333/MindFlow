# AGENTS.md — MindFlow Backend Execution Guide

Concise reference for agents working on the MindFlow backend (`mindflow-app/backend-next/`).
This guide intentionally omits frontend-specific workflows.

## Scope

- **Active backend**: `backend-next/` (FastAPI, LangGraph, SQLite). Legacy `backend/` deleted.
- **Python**: 3.11+.
- **Package manager**: **uv** only. Never use pip, conda, or poetry.
- **Windows deps**: `psutil` (always), `pywin32` (platform-marked, Win32 only). All managed by uv.

## Canonical Commands

Run all from `mindflow-app/backend-next/`:

| Action | Command |
|--------|---------|
| Install deps | `uv sync --extra dev --extra ml` |
| Start server | `uv run python -m mindflow.main` |
| Run tests | `uv run python -m pytest tests/ -q` |
| Single test | `uv run python -m pytest tests/test_foo.py -v` |
| Ruff lint | `uv run python -m ruff check src tests` |
| Mypy strict | `uv run python -m mypy --strict src/mindflow` |
| Migrate up | `uv run alembic upgrade head` |
| Eval (mock) | `uv run python -m mindflow.eval --mode both` |

**Eval note**: `--mode both` = rule engine + mock panel gateway (default mock, no API key needed). `--mode "mock"` is invalid. Use `--live --yes` for real LLM calls.

**Migration warning**: `alembic downgrade` on SQLite must use an isolated/backup DB. Always `sqlite3 mindflow.db ".backup backup.db"` first.

## Architecture Boundaries (ADR-001, ADR-004)

- **Framework-neutral ports** (`ports.py`): `AnalysisWorkflowPort`, `WorkflowRunStorePort`, `BudgetReservationPort`. LangGraph can be swapped without touching the scheduler.
- **AnalysisGraph**: Composition root for daily analysis. Implements `AnalysisWorkflowPort`. Contains PanelGraph as subgraph.
- **PanelGraph**: Explicit expert-deliberation subgraph (Analyst -> 3x parallel attribution -> validation -> Moderator -> Critic). `PanelOrchestrator` is a thin compatibility adapter; do not reintroduce its old inline graph.
- **ChatGraph**: Explicit chat lifecycle StateGraph. The legacy `create_agent` route remains the default while `new_chat_graph=False`.
- **ProviderRegistry**: LLM provider lifecycle (L1 DeepSeek, L2 Ollama, L3 RuleEngine). Single HTTP session pool.
- **Scheduler**: No graph. Owns time, claims, heartbeats (`scheduled_job_runs` table). Two-layer design per ADR-001.
- **Local OTel**: Traces to local SQLite; no external export. No PII in span attributes (ADR-003).
- **Unified entry point**: All analysis triggers (scheduler, API, chat, auto-intervention) go through one `AnalysisWorkflowPort`.

## Testing Rules

- **asyncio_mode = auto** (pytest.ini). Fixture loop scope defaults to function.
- Do not share sessions across tests. Each test gets its own async session.
- Mock LLM calls for unit tests. Integration tests use mock gateways.
- Dirty worktree is fine — do not commit, push, or create PRs unless explicitly asked.

## 2026-07-31 规则（保持这些不变式）

- 特征 schema 当前为 v3（`FEATURE_SCHEMA_VERSION=3`）；不要静默改回 v2，也不要绕过 `count_confirmed_switches()` 手写切换计数。
- 切换计数必须满足驻留阈值（默认 10 秒）并忽略 `TRANSIENT_PROCESSES`；同一应用内点击不算切换。
- ML 质量门统计“唯一反馈会话数”，不是重叠窗口数；低于 7 个反馈日时模型只能 shadow。
- `POST /panel/today` 缓存命中必须保留 `source/degraded/degradation_path`；降级结果重试使用 `retry_if_degraded`。
- LangGraph 主持人输出必须通过 `validate_verdict_schema()` 后再交给 critic；新增类型必须同步到 `TYPE_ALIASES` 和 `experts.py` 的枚举说明。
- 实验统一走 `scripts/run_experiments.py`，产物写入 `data/experiments/<run-id>/`；不要在实验目录外留下临时报告。

## Feature Flags (ADR-005)

All default to **legacy-safe old paths**. Rollback is a config change, not a code revert.

| Flag | Default | Effect When True |
|------|---------|-----------------|
| `MINDFLOW_GRAPH_VERSION` | `1` | Reserved metadata; does not select an implementation today |
| `MINDFLOW_CHECKPOINTING_ENABLED` | `False` | Use SQLite-backed instead of in-memory checkpoints |
| `MINDFLOW_NEW_ANALYSIS_GRAPH` | `False` | Route panel through v2 AnalysisGraph |
| `MINDFLOW_NEW_CHAT_GRAPH` | `False` | Route chat through ChatGraph StateGraph |
| `MINDFLOW_SHADOW_MODE_CHAT` | `False` | Run both legacy+new chat, return legacy |

## Real QA Expectations

- Real LLM calls cost money (~180 calls for full eval). Default is mock-only.
- Health endpoints: `/api/v1/health/live` (no auth), `/ready` (deep probe), `/health` (legacy).
- Diagnostics: `/api/v1/ai/runs` and `/api/v1/ai/runs/{run_id}` (authenticated, read-only, sanitised).
- Auth uses bootstrap tokens + session cookies. `/health` and `/docs` are exempt.
- Always test against an isolated temp data dir and port, never the user's production DB.

## Known Quality Debt

- **Ruff**: 94 findings (not clean; run `uv run python -m ruff check src tests` to see).
- **Mypy strict**: 158 errors in 16 files (not clean; run `uv run python -m mypy --strict src/mindflow`).
- These are visibility gates, tracked as known debt. Do not claim green.

## Pointers

- `CLAUDE.md` — Full build/test commands, architecture tables, feature flag details, acceptance evidence
- `backend-next/README.md` — Product overview, API table, config reference
- `../docs/architecture/ADR-001..005` — Architecture Decision Records
