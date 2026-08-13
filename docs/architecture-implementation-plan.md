# MindFlow 后端架构优化技术文档与实施计划

> **文档编号**: architecture-implementation-plan
> **日期**: 2026-08-14 · **状态**: ✅ 全部实施完成（A→J，2201 tests passed）
> **前置**: 审计修复已合并（2196 tests / mypy 117 / ruff 已知债）
> **配套**: docs/backend-architecture-optimization-evaluation.md（评估报告）、docs/project-audit-report-2026-08-14.md（审计报告）

---

## 1. 目标

将架构评估报告中"本周（🟢）+ 本月（🟡）"共 10 项优化按序落地，每项配套测试，最终全量验证并重启服务。本文档为唯一技术依据，包含每项的设计动机、改动文件、接口变更、数据流、测试方案与回滚策略。

## 2. 实施顺序与依赖

| # | 项 | 档位 | 后端 | 前端 | 迁移 | 预计 |
|---|----|------|------|------|------|------|
| A | 1.1 持久化 Checkpointer 接线 | 🟢 | ✅ | - | - | 0.5d |
| B | 3.1 采集器健康可见性 + 3.3 覆盖缺口 | 🟢 | ✅ | ✅ | - | 1d |
| C | 5.2 首启引导 + 5.1 AI 用量卡片 | 🟢 | ✅ | ✅ | - | 1d |
| D | 4.3 schema.py 双源合并 | 🟢 | ✅ | - | - | 0.5d |
| E | 2.1 ML 冷启动渐进激活 + 隐式反馈 + 特征 v4 | 🟡 | ✅ | ✅ | 0022 | 2d |
| F | 2.2 自动增量训练 | 🟡 | ✅ | ✅ | - | 1d |
| G | 1.2 图拓扑可视化 + 1.3 阶段预算 | 🟡 | ✅ | ✅ | - | 1d |
| H | 3.2 采集频率自适应 | 🟡 | ✅ | - | - | 1d |
| I | 4.1 特征窗口 JSON 拆列 | 🟡 | ✅ | - | 0023 | 1.5d |
| J | 2.5 解释性文案 + 5.3 AI 报告解读 | 🟡 | ✅ | ✅ | - | 1.5d |

**依赖**: E 依赖 D（activity_events 并入 schema）与 I 的表层；D/I 独立；H 依赖 B 的 health_summary 附带字段；其余独立。

## 3. 逐项设计

（每项含：现状 → 改动 → 接口 → 测试 → 回滚）

### A. 1.1 持久化 Checkpointer 接线

**动机**: AnalysisGraph 编译时未接 checkpointer（analysis_graph.py:1261 `graph.compile()`），即使 `checkpointing_enabled=True` 也无效果——LLM 调用无法跨崩溃续跑（付费成本翻倍），human_review 依赖 MemorySaver 丢失于重启。

**改动**:
1. `graph/analysis_graph.py`: `__init__` 增 `checkpointer: Any = None`；`_build_compiled_graph()` → `graph.compile(checkpointer=self._checkpointer.saver if self._checkpointer is not None else None)`。
2. `app.py:566` 组装 AnalysisGraph 时传 `checkpointer=checkpointer`（app.py 已创建 ApplicationCheckpointer）。
3. `run_analysis()` 中 `graph.ainvoke(initial_state, config=...)`，仅当 checkpointer 存在时传 `config={"configurable": {"thread_id": f"analysis_{user_id}_{target_date}"}}`。
4. 默认 `checkpointing_enabled=False` 走 InMemory 分支（行为不变）；`human_review_enabled=True` 自动启用 SQLite saver。

**接口**: 无对外 API 变更。

**测试**: test_analysis_graph.py 新增 `test_checkpointing_enabled_resumes_state`（构造含 checkpointer 的 AnalysisGraph，跑图中断后重放，断言关键状态续接）；保留默认关闭的行为不变测试。

**回滚**: 不传 checkpointer 参数即可。

### B. 3.1 采集器健康可见性 + 3.3 覆盖缺口检测

**动机**: 采集器失败有自愈但用户无感知；特征窗口因采集缺口稀疏时只标 stale 无原因。

**改动**:
1. `services/collector_service.py` 新增 `health_summary() -> dict`: 从 `collector_intervals` 聚合 `{uptime_s, last_failure_at, recovery_count, last_reason}`。
2. `api/routes/health.py` /health 的 collector 段扩展上述字段（保持 `status` 兼容）。
3. `services/telemetry_service.py` `rollup_feature_windows`: 期望窗口数（跨度/300s）vs 实际，缺口 >20% 时写入 `collector_intervals` failure（reason="coverage_gap"）。
4. 前端 `Dashboard.tsx` 采集器卡片: 显示 uptime / 最近失败时间 / 恢复次数。

**接口**: /health collector 对象新增可选字段（向后兼容）。

**测试**: collector_service 单测（health_summary 聚合逻辑，mock intervals）；telemetry rollup 缺口单测；health 端点测试更新。

### C. 5.2 首启引导 + 5.1 AI 用量卡片

**动机**: 新用户首日全 "--" 无预期管理；API key 用户看不到 LLM 花费。

**改动**:
1. 后端 `api/routes/analytics.py` 新增 `GET /analytics/usage` → `{llm_calls_30d, llm_cost_usd_30d, panel_count_30d, mode}`（聚合 procrastination_analyses.llm_cost_usd + workflow_runs）。
2. 前端 `Dashboard.tsx` 空态渲染"引导时间线"（采集时长/报告明日/画像 7 天/模型中心链接）。
3. 前端 `Settings.tsx` 新增"AI 用量"卡片；rule_engine 模式显示"本地模式，零成本"。

**接口**: 新端点 /api/v1/analytics/usage。

**测试**: analytics 路由测试（注入分析记录断言聚合）；前端 build。

### D. 4.3 schema.py 双源合并

**动机**: activity_events 表定义散落在 repositories/activity.py，schema.py 标注"后续移入"，alembic autogenerate 看不到全部表。

**改动**: 将 activity_events（含 Computed 列）迁移至 `infrastructure/schema.py`；activity.py 改从 schema 导入。无需新迁移（0001/0008 已覆盖表结构）。

**测试**: test_migrations + test_activity_repository 全绿。

### E. 2.1 ML 冷启动渐进激活 + 隐式反馈 + 特征 v4

**动机**: 质量门全过才 ready → 前两周 ML 不可用；只依赖显式反馈冷启动太慢；特征无任务类型维度。

**改动**:
1. **分档部署**（train/v2.py + training_readiness_service.py）: `deployment_tier` ∈ {full_ready, low_confidence, shadow}；low_confidence 条件: minimum_days≥3 且 balanced_accuracy≥0.55。model-status 与 readiness 返回 tier。
2. **隐式反馈**（train/v2.py）: `derive_implicit_labels(feature_windows)`——top_app_ratio>0.9 且 idle_ratio<0.1 → focus；switch_count>8 且 input_active_ratio<0.2 → distract；sample_weight 0.5（显式 1.0）。
3. **特征 v4**（domain/feature_schema.py + 迁移 0022）: 新增 `task_type_code`（app_classification 推断，回填）、`window_coverage_ratio`；FEATURE_SCHEMA_VERSION=4；同步 `_parse_window`、prediction_service 的 V2_FEATURE_NAMES。
4. 前端 ModelCenter: 显示 tier（"早期模式（低置信度）" / "已就绪" / "影子模式"）。

**接口**: model-status / training-readiness 新增 `deployment_tier`。

**测试**: train_v2 单测（隐式标签/分档/新特征）；readiness 更新；ml_integration 冒烟。迁移 0022 upgrade/downgrade。

### F. 2.2 自动增量训练

**动机**: 训练手动触发；新反馈不自动进模型。

**改动**:
1. `services/training_job_service.py` 新增 `auto_train_if_due() -> bool`: 24h 新增显式反馈 ≥5 且距上次训练 ≥24h → start_job（shadow，不 activate）。
2. `services/scheduler.py` build_scheduler 增 `interval_minutes(60, _auto_train_check)`（注入 training_job_service）；app.py 传参。
3. 前端 ModelCenter 显示"上次自动训练时间"。

**接口**: 内部 scheduler 任务；无 API 变更。

**测试**: training_job_service 单测（反馈计数/间隔守卫）；scheduler 注册测试。

### G. 1.2 图拓扑可视化 + 1.3 阶段预算

**改动**:
1. `api/routes/ai_diagnostics.py` 新增 `GET /ai/graph`: 从 `analysis_graph._get_compiled_graph().get_graph()` 提取 nodes/edges 返回 JSON（mermaid 文本亦可）。
2. 前端 Diagnostics 页新增"分析图拓扑"面板（SVG/表格渲染 nodes+edges）。
3. `graph/panel_graph.py`: `_MAX_CALLS=12` → `_PHASE_BUDGETS`（analyst=1, attribution=3, moderator=3, critic=2, rebuttal=1 累计 10）；`_call_with_budget` 按 role 查阶段预算。

**接口**: 新端点 /api/v1/ai/graph。

**测试**: ai/graph 端点结构测试；panel_graph 预算测试（超限时机变化）。

### H. 3.2 采集频率自适应

**动机**: 固定 5s tick，空闲时浪费电量/事件量。

**改动**:
1. `services/collector_service.py` tick 循环: 检测最近输入（input_telemetry last_input_at 或最近事件 is_idle）——60s 无输入 → sleep `idle_collect_interval_s=30`；恢复后回 5s。
2. config.py 增 `idle_collect_interval_s: int = 30`；health_summary 附带 current_interval_s。

**接口**: 配置项新增；/health collector 附带字段。

**测试**: collector_service 单测（mock sleep/输入状态，断言切换逻辑；无真实 sleep）。

### I. 4.1 特征窗口 JSON 拆列

**动机**: features_json 全量 JSON 解析，数据增长后训练/推理变慢。

**改动**:
1. 迁移 0023: behavior_feature_windows ADD COLUMN f01..f24 REAL（24 列 nullable）。
2. `repositories/telemetry.py`: 写时同时写 features_json + 新列；读时优先新列回退 JSON。
3. `train/v2.py`、`prediction_service.py` 读取优先新列（免 JSON 解析）。
4. 保留 features_json（不删，兼容回滚）。

**接口**: 表结构扩展（向后兼容）。

**测试**: telemetry repository 写读一致；prediction_service 新列路径。

### J. 2.5 解释性文案 + 5.3 AI 报告解读

**动机**: 干预提醒无理由（数字不解释）；报告是数字表格。

**改动**:
1. `services/intervention_service.py`: `_LLM_SYSTEM_PROMPT` 加"引用基线对比"约束；模板 fallback 加基线偏差句（EvidenceBundle.deviation）。
2. `services/report_service.py`: `generate_daily_report` 增 `ai_insight` 字段——LLM 降级链生成 2-3 句解读；无 key 用模板（改造 pattern_summary）。
3. 前端 Reports 渲染 ai_insight 段。

**接口**: daily report 响应新增可选 `ai_insight`。

**测试**: report_service 单测（mock LLM/无 key 模板）；intervention 模板测试。

## 4. 验收标准

| 关卡 | 标准 |
|------|------|
| 单元测试 | pytest 全量 ≥2196 且 0 失败 |
| 迁移 | 0022/0023 upgrade head 成功、可 downgrade |
| 前端 | npm run build + lint 通过 |
| mypy | 不新增错误（≤117） |
| ruff | 不新增错误 |
| 服务 | /health 全绿、采集器 running、新端点（usage/ai/graph）可用 |

## 5. 交付物

1. 本文档（技术文档）
2. 10 项实现 + 测试（A→J，独立 commit）
3. 迁移 0022/0023
4. 架构评估文档状态更新
5. 全量验证 + 服务重启

---

*文档结束。实施从 A 项开始。*
