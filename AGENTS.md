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

- 特征 schema 当前为 v4（`FEATURE_SCHEMA_VERSION=4`，28 列：v3 的 24 列 + `task_type_entropy`、`task_type_dominant_ratio`、`task_context_transition`、`task_unknown_ratio`）；不要静默改回 v2/v3，也不要绕过 `count_confirmed_switches()` 手写切换计数。
- 切换计数必须满足驻留阈值（默认 10 秒）并忽略 `TRANSIENT_PROCESSES`；同一应用内点击不算切换。
- ML 质量门统计“唯一反馈会话数”，不是重叠窗口数；低于 7 个反馈日时模型只能 shadow。
- `POST /panel/today` 缓存命中必须保留 `source/degraded/degradation_path`；降级结果重试使用 `retry_if_degraded`。
- LangGraph 主持人输出必须通过 `validate_verdict_schema()` 后再交给 critic；新增类型必须同步到 `TYPE_ALIASES`（现在位于 `agents/schemas.py`）和 `experts.py` 的枚举说明。
- 实验统一走 `scripts/run_experiments.py`，产物写入 `data/experiments/<run-id>/`；不要在实验目录外留下临时报告。

## 2026-09-25 规则（优化方案落地后的不变式）

- **L1 是 DeepSeek 直连**：`LLMSettings.l1_target()` 在生产（`ecnu_compat_enabled=False`，默认）固定 `provider=generic`、`base_url=https://api.deepseek.com`、`model=deepseek-flash`，凭证取 `DEEPSEEK_API_KEY` / `MINDFLOW_LLM__DEEPSEEK_API_KEY`；遗留的 ECNU URL/model/provider/密钥会被**覆盖且不混用**（ECNU 形状的遗留配置不会把它的密钥交给 DeepSeek）。缺凭证时 L1 不可用，走既有 L2/L3，**不回退 ECNU**。面板的 `chat`/`reasoner` 只是输出策略（JSON vs 散文），两个 tier 都请求同一模型；`reasoning_effort`/`thinking`/`max_tokens` 是**每次请求**下发（`agents/policies.py`）。ECNU 适配代码保留，仅通过 `ecnu_compat_enabled=True` 供兼容测试与诊断使用。
- **thinking 模式必须回传 `reasoning_content`**：任何携带 `tools` 的请求都要把历史 assistant 回合的 `reasoning_content` 原样回传，否则 DeepSeek 返回 400。`infrastructure/llm/thinking.py` 提供共享实现，ECNU 与 DeepSeek 两个适配器都用它；思考模式下 `temperature` 无效且不再下发。4xx 与「长度截断」不做同参重试（`LengthFinishReasonError` 已列入不可重试集合）。
- **思考模式下 token 上限有下限**：DeepSeek 的 thinking token **计入** `max_tokens`。2026-09-25 真实请求实测：attribution 角色 1000 token 上限时每次都被截断（`LengthFinishReasonError`，JSON 从未产出）。因此 `agents/policies.py` 的 `THINKING_TOKEN_FLOOR=4096` 只在请求开启思考时生效（`max(角色上限, 4096)`），关闭思考时仍精确使用角色上限；面板路径真实推理 token 占输出约 55%（13 次请求 12755 输出 token / 7075 推理 token），这是下限存在的依据。
- **训练发布拦截**：质量门通过后、移动 active 指针前，`train/publication.py` 比对「评估选中候选」与「实际分类器」。只有二者确认一致（当前仅 `rf_xgb_soft_voting` ↔ `EnsembleClassifier`）才允许激活；不一致、评估无候选、未知制品、XGBoost 缺失导致的 RF-only 回退一律保存为 shadow 并记录 `activation_blocked_reason`（报告与 manifest 都带 `publication` 块）。`allow_activation=False` 的自动训练策略继续生效，且本轮不扩展为发布所有候选。**验收**（`scripts/acceptance_publication_guard.py`，三路径端到端全 PASS）：A 不一致→shadow+原因；B 数据工程让评估选中集成（896 窗口、漂移 PSI 0.21 过门）→ 一致→激活（latest.json 移动、ready）；C `allow_activation=False`→shadow。激活路径要求 xgboost 已安装（缺失即 REASON_XGB_FALLBACK）。
- **v4 特征回填**：`uv run python -m mindflow.telemetry rebuild-features --start <UTC日期> --end <UTC日期> --user-id 1 [--apply]`。默认只预览；`--apply` 按日块写入并输出 JSON 汇总（块级 rebuilt/written/missing_raw_data/failed）。范围受 180 天边界限制（越界会被 clamp 并报告），只从仍留存的原始事件生成 v4（绝不从 v3 推造），按 `window_start_utc` 继承 v3 非空标签但**绝不覆盖**已有 v4 标签，保留 v3 行不删。同一 `(user_id, window_start_utc, feature_schema_version)` 的 upsert 现在同时刷新 `features_json` 与 `f01..f28`。
- **面板终态语义**：`PanelGraph` 只有一个终态节点 `panel_finalize`，写入 `critic_approved` / `panel_rejected` / `panel_terminal` / `rejection_reason`。`AnalysisGraph.panel_graph_node` 只有在「存在 `moderator_verdict` + `validate_verdict_schema()` 通过 + `critic_approved=True`」时才允许 `panel_succeeded=True` / `source="panel"`；critic 连续驳回必须走 `single_expert → ollama → rule_engine`，不得把主持人旧裁决当成功结果。`panel_degradation_marker` 会作为 `degradation_path` 的前缀持久化，cache 读取时禁止重新标成 panel。
- **自动训练永不激活**：`run_training(..., allow_activation=...)`；`TrainingJobService.auto_train_if_due()` 固定传 `allow_activation=False`，只能产出 shadow candidate；只有手动训练（`allow_activation=True`）或独立发布任务才允许移动 active 指针。候选即使通过质量门，被抑制时也只写 shadow 版本并记录 `activation_suppressed_reason`。
- **Prompt ⇄ schema 对齐**：所有生产 schema 都是 `extra="forbid"`；prompt 提到的字段必须存在于 schema（`top_concerns` / `cognitive_distortions` / `tmt_factors` / `emotion_pattern` / `is_emotion_driven` / `critique_detail`）。语义校验由代码执行：空论据、无合法类型、无证据引用、置信度越界、`insufficient_data=true` 但无 `evidence_gaps` 的意见一律不是有效意见。
- **Claim Ledger**：专家输出优先使用 `claims`（type/confidence/evidence_ids/support/alternative）；旧格式会被 `agents/claims.py` 合成为等价 ledger。Moderator 只接收校验过的 claim 表 + 证据表 + 冲突摘要，并叠加代码级共识保护（低共识自动降置信度或弃权）。
- **证据压缩**：`to_prompt_json(bundle)` 默认压缩（异常/显著偏离/近期干预项保留全量，稳定项折叠为 `stable_summary`；catalog 是位置元组 `[id, 中文标签, 类型]`，不是对象）。实测：稳定型真实窗口 −32.8%（达标 ≥30%），9 项生产形态窗口 −20.1%，异常为主的 panel fixture −12.9%，30 个评估场景全部变小（合计 −12.0%）；citation id 集合与 legacy 完全一致。`compressed=False` 仅用于 A/B 对比，不是生产模式。
- **ML 评估协议**：主评估是 forward-chaining / leave-future-days-out；每个 fold 只在与声明 `train_groups` 完全一致的行上训练（禁止“除测试组以外的所有支持行”，那会把未来日期块泄进训练）。窗口标签只训练、绝不进入评估掩码；mixed/冲突窗口排除；校准器只用训练折内部的独立校准组。
- **rollup watermark**：`TelemetryService.rollup_recent()` 只重算缺失的桶 + 一个桶的 overlap；失败不推进 watermark。完整 2 小时窗口仍是上界。
- **快速路径默认关闭**：`MINDFLOW_PANEL_FAST_PATH_ENABLED=False`。离线回放（`scripts/experiment_fast_path.py`，2026-09-24 运行）显示快速路径仅触发 5/30 场景，关键结论 top-1 一致率 60%（集合一致率 80%），**低于方案要求的 95%**，因此开关必须保持关闭，直到在固定场景集上重新验证达标。
- **全局并发门仍为 1**：`scripts/experiment_concurrency.py`（MockTransport，离线）实测 c=2/c=3 的 p95 分别比 c=1 低 24.4%/30.0%，错误率不变，但 8 个场景块下 c=2 与 c=3 的排序不稳定，因此**全局门**默认保持 `1`，脚本只做测量。
- **面板扇出默认 burst=3**（2026-09-25 实测后新增，仅作用于 panel 的并行 attribution/rebuttal 批）：`MINDFLOW_LLM__PANEL_FANOUT_CONCURRENCY`，经 registry 专用 fan-out gateway（自带并发门）下发；其余 L1 路径（chat/干预/结构化归因）仍走全局门。真实对照（2 场景 × 9 调用）：墙钟 114.5s→60.5s（-47%），串行排队 82.5s→0ms（-100%），两场景裁决类型与升级形态**完全一致**、零错误；设为 1 即回到串行。
- **Attribution usage 已补齐**：`DeepSeekClient.analyze()` 现捕获 `usage`（`last_usage`），`single_expert_node` 把它写进观测记录；该 raw L1 入口同样下发 `reasoning_effort`/`thinking`/`max_tokens`（与网关同契约）。真实复测 usage_reported=True，3 次调用推理 token 占输出 85%（token floor 的又一依据）。
- **多场景/重复烟测**：`scripts/smoke_live_llm.py --scenario-count N --repeat M`（panel 跑前 N 个 eval 场景，各目标重复 M 次并聚合 schema/引用/失败率/p95/usage）。灰度产物：`data/experiments/gray_multiscenario_c1`、`gray_fanout_c3_v2`、`gray_repeat_chat_tools`。
- **v4 回填已在真实数据副本上验收**：生产库升级到 0027 后，副本上 apply 重建 6265 个 v4 窗口、继承 2401 个 v3 标签、0 失败；v2（1726/198）与 v3（8198/2981）行数与标签数前后一致；同参复跑幂等（6265/2401 不变）。153 个块因原始事件过保留期无法重建（设计使然：v4 只从留存的原始事件生成）。**生产库执行 --apply 前先备份**，命令见 README。

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
| `MINDFLOW_PANEL_FAST_PATH_ENABLED` | `False` | Panel fast path (plan 2.2): skip the LLM panel when evidence coverage/quality is high, the rule engine is confident and nothing conflicts (route to rule engine or a single expert); return `insufficient_data` without any LLM call when collectors are missing or coverage is too low. **Must stay `False`**: the 2026-09-24 offline replay measured only 60% top-1 agreement on the 5 scenarios it did trigger (target ≥95%). |
| `MINDFLOW_LLM__DEEPSEEK_API_KEY` / `DEEPSEEK_API_KEY` | unset | Production L1 credential for DeepSeek direct (`deepseek-flash`). Without it L1 is unavailable and the chain degrades to L2/L3 — the legacy ECNU key is never borrowed. |
| `MINDFLOW_LLM__ECNU_COMPAT_ENABLED` | `False` | Honour the legacy ECNU triple (provider/base_url/model/api_key) instead of the DeepSeek pin. For the ECNU adapter's compatibility tests and diagnostics only — **must stay `False` in production**. |

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

- **Ruff**: 0 findings (`uv run python -m ruff check src tests` → `All checks passed!`, 2026-09-25).
- **Mypy strict**: 0 errors in 185 source files (`uv run python -m mypy --strict src/mindflow` → `Success`, 2026-09-25).
- **Pytest**: 3120 passing, 4 skipped, 22 warnings (`uv run python -m pytest tests/ -q`, 2026-09-25).
- **真实 DeepSeek 烟测**（2026-09-25，`scripts/smoke_live_llm.py --yes --max-requests 16`，隔离数据目录）：四目标全 PASS——panel 9 请求 / 9-9 schema / 引用 7-7 有效 / 失败率 0% / p95 26.7s；chat 1 请求 / p95 2.1s；tools 2 请求（真实工具调用 + 后续轮次成功，证明 `reasoning_content` 回传可用）/ p95 0.9s；attribution 1 请求 / 1-1 schema / p95 7.5s。合计 13 次请求、输入 20786 token、输出 12755 token（其中推理 7075，占 55%），全部 2xx、无降级、无 4xx 重试。产物：`data/experiments/20260925_051755/{summary.json,report.md}`（不含任何密钥或提示词）。
- Keep these green: any edit that regresses them must be fixed before finishing.

## Pointers

- `CLAUDE.md` — Full build/test commands, architecture tables, feature flag details, acceptance evidence
- `backend-next/README.md` — Product overview, API table, config reference
- `../docs/architecture/ADR-001..005` — Architecture Decision Records
