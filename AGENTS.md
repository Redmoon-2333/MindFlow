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
- **PanelGraph**: Explicit expert-deliberation subgraph (Analyst -> 3x parallel attribution -> validation -> Moderator -> Critic). The old `PanelOrchestrator` class was removed at v2 cutover; keep its parsing helpers module-level and do not reintroduce the inline graph.
- **ChatGraph**: Explicit chat lifecycle StateGraph. The v2 graph is now the only production chat path; the former `create_agent` route has been removed.
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

The backend completed the v2 graph cutover. `checkpointing_enabled` remains active;
the two graph-selection flags are retained only so older environment files still
parse, but changing them no longer selects a legacy implementation.

| Flag | Default | Effect When True |
|------|---------|-----------------|
| `MINDFLOW_CHECKPOINTING_ENABLED` | `False` | Use SQLite-backed instead of in-memory checkpoints |
| `MINDFLOW_NEW_ANALYSIS_GRAPH` | `True` | Deprecated compatibility flag; v2 AnalysisGraph is always active |
| `MINDFLOW_NEW_CHAT_GRAPH` | `True` | Deprecated compatibility flag; v2 ChatGraph is always active |
| `MINDFLOW_TRAINING_USE_WINDOW_LABELS` | `True` | Also train on user-calibrated `behavior_feature_windows.label` (weight 0.8; feedback still wins; quality-gate counts stay feedback-only). Measured 2026-08-20: BA 0.46→0.64, Brier 0.40→0.23, folds stable — activated the model. Set `0` to disable. |

## 2026-08-20 补充

- **生产训练默认带 Platt(sigmoid)校准**:`run_training(calibration="sigmoid")`(默认)令评估 `evaluate_v2_candidates` 与部署 `ModelManager` 一致;校准器随 `to_dict/from_dict` 序列化。`make_v2_classifier()` 公开默认保持原始;合成/小数据集显式传 `calibration=None`(校准只在大而干净的数据上有效)。
- **面板/归因真实 LLM 跑通的先决条件**:`_PANEL_WORKFLOW_TIMEOUT_S` 为 120s(此前 8s 在真实 DeepSeek ~4s/次多次调用下必然超时);critic 提示词强制 `critique_detail ≤300字` 防撞 8192 token 截断。
- 面板并行专家调用在 DeepSeek 瞬时连接错误时会整链退化到 rule_engine(待后续加 per-call 隔离)。**2026-08-20 已加整批重试**:`_fanout_raw_with_batch_retry` 在并行专家批全部返回空(瞬时连接故障特征)时整批重试一次(3s 回退,预算兜底);`tests/test_panel_batch_retry.py` 覆盖。

## Real QA Expectations

- Real LLM calls cost money (~180 calls for full eval). Default is mock-only.
- Health endpoints: `/api/v1/health/live` (no auth), `/ready` (deep probe), `/health` (legacy).
- Diagnostics: `/api/v1/ai/runs` and `/api/v1/ai/runs/{run_id}` (authenticated, read-only, sanitised).
- Auth uses bootstrap tokens + session cookies. `/health` and `/docs` are exempt.
- Always test against an isolated temp data dir and port, never the user's production DB.

## Quality Gates

- **Ruff**: 0 findings (`uv run python -m ruff check src tests` → `All checks passed!`, 2026-08-16).
- **Mypy strict**: 0 errors in 163 source files (`uv run python -m mypy --strict src/mindflow` → `Success`, 2026-08-16).
- **Pytest**: 2201+ passing (baseline 2026-08-16).
- Keep these green: any edit that regresses them must be fixed before finishing.

## Pointers

- `CLAUDE.md` — Full build/test commands, architecture tables, feature flag details, acceptance evidence
- `backend-next/README.md` — Product overview, API table, config reference
- `../docs/architecture/ADR-001..005` — Architecture Decision Records
