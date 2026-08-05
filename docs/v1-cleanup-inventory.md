# V1 残留清理清单（MindFlow backend-next）

> 生成日期：2026-08-04
> 目标：彻底清理 V1（原始事件级特征提取 / legacy 训练路径），只保留 V2（24 维 schema v3）最新管线。
> 判定口径：基于实际 `grep` 引用，不猜。`可删` = 生产代码零引用、V2 不依赖；`谨慎` = V2 部分依赖或由运行时 flag 控制行为；`保留` = V2 在用 / API 契约不能动。

> **状态更新（2026-08-05）**：本清单生成后，分析与聊天已完成 v2-only cutover。
> `new_analysis_graph` / `new_chat_graph` 仍可解析旧配置，但已经是 deprecated
> no-op；旧 `create_agent` 聊天路径及相关 rollback 说明不再适用。以下涉及运行时
> flag 的条目以当前代码为准，历史判断仅保留作清理记录。

汇总：**可删 10 条 · 谨慎 9 条 · 保留 16 条**。

---

## 一、可删（确认 V2 不依赖，生产代码零引用）

| # | 位置 | 是什么 | 判定依据 |
|---|------|--------|----------|
| 1 | `src/mindflow/train/features.py`（整文件） | V1 17 特征提取器 `BehaviorFeatureExtractor` + `TitleAnalyzer`（30 分钟窗口，含 PII） | 生产仅被 `train/__init__.py:12` 再导出和注释引用；证据服务已改用 `FocusPredictionService`（`evidence_service.py:465` 只是注释）。V2 特征窗口由 `telemetry_features.build_v2_feature_window` 产出。删除时需同步改 `train/__init__.py` |
| 2 | `src/mindflow/train/synthetic_data.py`（整文件，含 `_generate_legacy` 于 :125） | V1 原始事件合成数据生成器；`generate_synthetic_data()` 仅被测试调用 | V2 合成路径是 `train/synthetic_v2.py:generate_v2_synthetic_data`（`pipeline.py:137`）。`_generate_legacy` 只被本模块 `synthetic_data.py:94` 调用 |
| 3 | `src/mindflow/train/qa_pipeline.py`（整文件） | V1 合成数据的 3-agent QA 管线（`QAPipeline` 等） | 生产零引用；仅 `tests/test_qa_pipeline.py` 测试。V2 质量门在 `train/v2.py:evaluate_v2_quality_gate` |
| 4 | `src/mindflow/train/__main__.py:47-96` | `load_database_events()`：加载 V1 原始事件 | `run_training(source="db")` 只用 `feature_windows`（`pipeline.py:120-127`），`events` 参数被忽略；`__main__.py:348` 的调用与打印需一并删除 |
| 5 | `src/mindflow/train/pipeline.py:240-407` | V1 时代死代码：`_is_trainable_window`、`_evaluate_quality_gate`、`_build_feature_matrix`、`_enrich_with_process`、`_print_clustering_summary`、`_print_classifier_summary`、`_print_hmm_summary` | 全仓库（含测试）零引用。V2 路径 `_run_v2_training`（:164）不使用它们 |
| 6 | `src/mindflow/train/__init__.py:12,26` | 对 `BehaviorFeatureExtractor` 的再导出 | `train/features.py` 删除后成为孤儿导出；`src` 下无人 `from mindflow.train import BehaviorFeatureExtractor` |
| 7 | `src/mindflow/domain/labeling.py`（整文件） | V1 弱监督标记 `ConsensusLabeler` + 6 个 signal 类 | 生产零引用（`pipeline.py:95`、`classifier.py:19` 只是注释）；仅 `tests/test_labeling.py`。V2 弱标签在 `train/v2.py:_weak_label`（:364） |
| 8 | `src/mindflow/domain/features.py:29-30` | 常量 `MIN_ACTIVITY_THRESHOLD`（注释自述 "Legacy ... retained"） | 仅 `tests/test_features.py:91` 引用；`domain/features.py` 其余函数（`focus_score`、`count_confirmed_switches` 等）被 `evidence_service` 在用，**保留** |
| 9 | `src/mindflow/config.py` 的旧 `Settings.graph_version` 字段 | 已在 v2 cutover 清理；注意区分 `graph/state.py` 与 `workflow_runs.graph_version` 列（那是活的状态 schema 字段，见保留 #14） |
| 10 | `src/mindflow/graph/fallback_nodes.py:716-808` | `run_fallback_pipeline()` 顺序执行器（注释自述 "used by legacy adapter"） | 生产零引用；`llm_service.py:298-304` 与 `analysis_graph.py:29-31` 直接调用各节点，不走此函数。仅 `tests/test_fallback_nodes.py` 集成测试 |

---

## 二、谨慎（V2 部分依赖 / 运行时 flag 控制行为 / 需先改引用）

| # | 位置 | 是什么 | 判定依据 |
|---|------|--------|----------|
| 1 | `src/mindflow/config.py` `new_analysis_graph` | Deprecated compatibility input | v2 AnalysisGraph is unconditionally wired; changing the value no longer selects a legacy path |
| 2 | `src/mindflow/config.py` `new_chat_graph` | Deprecated compatibility input | v2 ChatGraph is unconditionally used; the former `create_agent` path was removed |
| 3 | `src/mindflow/config.py` `shadow_mode_chat` | Historical-only flag | The shadow comparison path was removed with the legacy chat implementation |
| 4 | `src/mindflow/services/chat_service.py` | `_ShadowChatRepo` 影子内存仓库 | 已随 legacy chat 路径一并删除 |
| 5 | `src/mindflow/services/chat_service.py` | `_ask_shadow_mode()` 双路径对比 | 已随 legacy chat 路径一并删除 |
| 6 | `src/mindflow/train/explain.py`（整文件） | `ModelExplainer`（SHAP 可解释性） | 被 `manager.py:207` 在 `train_all(use_explainer=True)` 时懒加载；但 V2 管线从不传 `use_explainer=True`（`pipeline.py:202`）。默认关闭的活代码路径 |
| 7 | `src/mindflow/services/telemetry_service.py:479-532` | `predict_latest_focus` 的 `_model_manager` 直连回退分支（注释 "Legacy fallback"） | 仅当 `_prediction_service is None` 时进入；app.py 永远注入 `prediction_service`（app.py:345-347）→ 防御性死路径 |
| 8 | `src/mindflow/domain/baseline.py:93` | `_window_start_local` 接受 legacy `window_start` 别名键 | 为旧 V1 行兼容而保留；V2 写入用 `window_start_utc`。属无害兼容 |
| 9 | `alembic/versions/0017_create_ml_shadow_predictions.py` + `src/mindflow/infrastructure/schema.py:238-252` | `ml_shadow_predictions` 表（V1 时代 shadow 预测留档） | 全仓库无任何仓库层读写（grep 仅命中 schema 定义 + migration）。删除需迁移链手术：`0018` 的 `down_revision` 指向 `0017`，或新增 drop 表 migration |

---

## 三、保留（V2 在用 / API 契约不能动）

| # | 位置 | 是什么 | 判定依据 |
|---|------|--------|----------|
| 1 | `src/mindflow/train/v2.py` | V2 训练数据准备 / 评估 / 质量门 | 被 `pipeline.py`、`training_readiness_service.py`、`prediction_service.py`、`telemetry_service.py` 引用 |
| 2 | `src/mindflow/train/synthetic_v2.py`、`user_profiles.py`、`config.py` | V2 合成数据 + 30 原型 + 超参 | `pipeline.py:137`、`__main__.py:336`、`v2.py:32` 在用 |
| 3 | `src/mindflow/train/models/manager.py`、`ensemble.py`、`clustering.py`、`hmm.py`、`types.py`、`serialization.py` | V2 `ModelManager` 模型族（含 HMAC 签名） | `pipeline.py:201` 训练、`app.py:382-411` 加载、`prediction_service` / `training_job_service` 推理均在用 |
| 4 | `src/mindflow/train/models/classifier.py` `FocusClassifier` | RF 分类器（V2 特征上训练） | 仍是 `ModelManager` 的活回退（`manager.py:81,94,429`）；app.py 以 `use_ensemble=False` 加载 V2 模型（app.py:408），RF-only 构件是真实路径 |
| 5 | `src/mindflow/train/pipeline.py:73-237`（`run_training` + `_run_v2_training`） | V2 训练入口 | CLI（`__main__.py:364`）、`training_job_service.py:32`、e2e 测试在用 |
| 6 | `src/mindflow/config.py:220` `checkpointing_enabled` | LangGraph 检查点持久化开关 | `checkpointer.py:267` + `app.py:261` 读取，属现役基础设施，非 V1 专属 |
| 7 | `src/mindflow/domain/feature_schema.py` | `FEATURE_SCHEMA_VERSION` + `V2_FEATURE_NAMES`（24 维） | 唯一权威特征词汇表，domain/train/telemetry 共享 |
| 8 | `src/mindflow/domain/baseline.py:282-288` | `from_dict` 对 V1 payload 的 schema 版本检测（默认 1，无映射器） | 刻意保留以便 `telemetry_service.rebuild_baseline_if_needed` 识别并重建旧 baseline（:395-458） |
| 9 | `src/mindflow/api/routes/health.py:97-194` | legacy `/health` 组件 payload 端点 | 注释自述 "always keep HTTP 200 compatibility"，是运维/前端依赖的 API 契约 |
| 10 | `src/mindflow/agents/orchestrator.py` 模块级 helper | 专家面板解析/校验 helper | `PanelGraph` 已接管唯一生产面板路径；旧 `PanelOrchestrator` 类已在 v2 cutover 中删除，helper 仍被 `PanelGraph` 懒加载引用 |
| 11 | `src/mindflow/graph/panel_graph.py` `PanelGraph` | V2 显式面板子图 | 由始终启用的 `AnalysisGraph` 使用（app.py:567）；旧 flag 仅保留为 deprecated no-op |
| 12 | `src/mindflow/graph/analysis_graph.py` + `fallback_nodes.py` 的节点（除 `run_fallback_pipeline`） | V2 AnalysisGraph + 三档降级节点 | `llm_service.py:298-304` 与 `analysis_graph.py:29-31` 直接用；`app.py:569` 构造 |
| 13 | `src/mindflow/services/chat_service.py` | V2 ChatGraph service adapter | 旧 `_ask_serialized` / `create_agent` 路径已删除；`ChatService` 统一委托 `ChatGraph` |
| 14 | `src/mindflow/graph/state.py` `graph_version` + `src/mindflow/infrastructure/schema.py:294` `workflow_runs.graph_version` 列 + `api/schemas.py:130,146` | 状态 schema 版本元数据 + 运行记录列 + API 响应字段 | 与 config.py 的 `graph_version` 字段不同，这些是活的 schema/契约字段 |
| 15 | `src/mindflow/infrastructure/repositories/telemetry.py`（`interaction_buckets`/`browser_segments`/`behavior_feature_windows`） | V2 特征窗口的数据源 | `telemetry_service.rollup_feature_windows`（:252-393）直接消费；**绝不能标可删** |
| 16 | `alembic/versions/0001-0016, 0018` | 现役表/列迁移（baseline、telemetry、workflow、feedback 快照、intervention_checks 等） | 全部被 schema.py / 仓库层使用；`baseline_models`、`behavior_feature_windows` 为 v1/v2 共享表 |

---

## 四、清理顺序建议

### 阶段 1：纯死代码（零行为影响，先删这批）
1. **删 `domain/labeling.py`** + `tests/test_labeling.py`（ConsensusLabeler 无生产引用）。
2. **删 `train/features.py`** + `tests/test_train_features.py`；删 `train/__init__.py:12,26` 的再导出；从 `tests/test_ml_integration.py` 移除 `BehaviorFeatureExtractor` 相关用例（保留 V2 集成部分）。
3. **删 `train/synthetic_data.py`** + `tests/test_train_synthetic.py`、`tests/test_synthetic_enhanced.py`。
4. **删 `train/qa_pipeline.py`** + `tests/test_qa_pipeline.py`。
5. **删 `domain/features.py:29-30` `MIN_ACTIVITY_THRESHOLD`** + 修 `tests/test_features.py:91`。
6. **删 `pipeline.py:240-407` 死函数**；删完检查 `pipeline.py` 顶部 import 是否孤儿（`math`、`timedelta`、`npt`、`Mapping`、`Sequence` 等）。
7. **删 `config.py:219` `graph_version` 字段** + 修 `tests/test_checkpointer.py:90,107`。
8. **删 `fallback_nodes.py:716-808` `run_fallback_pipeline`** + 精简 `tests/test_fallback_nodes.py` 集成用例（节点单测保留）。

**验证**：`uv run python -m pytest tests/ -q`（预计 1956 通过，删除对应测试文件后数量下降但全绿）+ `uv run python -m ruff check src tests` 确认无新增。

### 阶段 2：需先决策 flag 方向再删
9. **Chat shadow 三件套**（`_ShadowChatRepo`、`_ask_shadow_mode`、`_shadow_graph` 构建）已随 legacy chat 路径删除；`shadow_mode_chat` 仅作为历史配置记录。
10. **`new_chat_graph` / `new_analysis_graph`**：已完成 v2-only cutover；字段保留为 deprecated no-op 以兼容历史配置，旧实现不再随开关恢复。
11. **`train/explain.py`**：要么在 V2 训练接上 `use_explainer=True`，要么删（当前默认关闭）。
12. **`telemetry_service.py:479-532` legacy 回退分支**：确认 app.py 永远注入 `prediction_service` 后删除。
13. **`ml_shadow_predictions` 表**：新增 drop 表 migration（推荐，避免改动 0017/0018 迁移链），或删 0017 并把 0018 的 `down_revision` 指向 0016。同步删 `schema.py:238-252` 定义。

**验证**：全量测试 + `uv run alembic upgrade head`（在临时库上）+ 启动 `uv run python -m mindflow.main` 检查 `/health`、`/api/v1/analytics/model-status`。

### 阶段 3：回归验证清单
- `uv run python -m pytest tests/ -q`
- `uv run python -m mindflow.train --source synthetic_v2`（V2 训练仍可跑通）
- `uv run python -m alembic history && alembic upgrade head`（在备份库上，勿直接 downgrade SQLite）
- 启动服务后 `GET /api/v1/health`、`GET /api/v1/analytics/model-status`、`GET /api/v1/analytics/training-readiness`
- 前端 `npm run build` + model-center Playwright E2E（若前端涉及 `/model-center`）
