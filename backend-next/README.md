# MindFlow

**本地优先的智能专注助手** —— 基于行为分析的抗拖延系统

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)](https://fastapi.tiangolo.com)

---

## 项目简介

MindFlow 是一款桌面端专注力管理工具，通过实时采集电脑使用行为数据，结合机器学习模型和认知行为疗法（CBT）技术，帮助用户识别拖延模式、保持专注、提升效率。

**核心理念**：所有数据本地存储，隐私优先；分析引擎在本地运行，无需联网即可获得智能反馈。

### 功能概览

| 模块 | 功能 | 状态 |
|------|------|------|
| 行为采集 | 主动窗口监测（Win/Mac/Linux），5 秒采集间隔，心跳合并 | stable |
| 专注分析 | 会话识别，专注评分，基线偏差检测，拖延类型分类 | stable |
| 数据报告 | 日报/周报生成，App 使用统计，趋势分析 | stable |
| LLM 分析 | DeepSeek / Ollama 三层降级，CBT 行为分析 | stable |
| 智能干预 | 基于规则的干预生成，节流控制，深度工作不打扰 | stable |
| 数据导出 | CSV / JSON 导出，日期范围筛选 | stable |
| ML 训练 | 合成数据生成，HMM 训练，聚类分析 CLI | stable |
| AI 诊断 | 工作流运行记录与节点事件查询 (`/api/v1/ai/runs`) | stable |
| 图编排 | AnalysisGraph (AnalysisWorkflowPort) + ChatGraph + feature flags | active dev |

---

## 架构

```
┌──────────────────────────────────────────────────┐
│                MindFlow App                         │
│  ┌──────────────────────┐  ┌──────────────────┐   │
│  │  FastAPI (REST :8765) │  │ WebSocket /api/v1/ws │ │
│  └──────┬───────────────┘  └──────┬───────────┘   │
│         │                          │                │
│  ┌──────┴──────────────────────────┴───────┐      │
│  │        Services Layer                     │      │
│  │  analysis · report · intervention · llm  │      │
│  └──────┬──────────────────────────┬───────┘      │
│         │                          │                │
│  ┌──────┴──────┐  ┌───────────────┴────────┐     │
│  │  Repos      │  │  RuleEngine / ML Models  │     │
│  └──────┬──────┘  └────────────────────────┘     │
│         │                                          │
│  ┌──────┴──────────────────────────────────┐      │
│  │  SQLite (aiosqlite, WAL mode)            │      │
│  └─────────────────────────────────────────┘      │
│                                                    │
│  ┌──────────────────────────────────────────┐     │
│  │  Collector (asyncio tick loop, 5s)       │      │
│  │  Win32 / macOS / X11 / Wayland           │      │
│  └──────────────────────────────────────────┘     │
└──────────────────────────────────────────────────┘
```

### 技术栈

- **运行时**: Python 3.11+, uvicorn (async ASGI)
- **Web 框架**: FastAPI 0.115+
- **数据库**: SQLite + SQLAlchemy (async) + Alembic 迁移
- **编排**: LangGraph StateGraph (AnalysisGraph, PanelGraph, ChatGraph) + framework-neutral ports via `AnalysisWorkflowPort`
- **LLM 集成**: ProviderRegistry 管理 DeepSeek / Ollama / RuleEngine 三层降级
- **调度**: 双层架构 — 外层纯 asyncio Scheduler + SQLite claims/heartbeats，内层 LangGraph reasoning
- **可观测性**: 本地 OpenTelemetry (SQLite exporter)，无外部导出；工作流运行记录 (`workflow_runs` 表)
- **ML**: scikit-learn, hmmlearn (本地训练/预测)
- **打包**: PyInstaller (单文件桌面应用)
- **包管理**: uv (取代 pip/conda)

---

## 快速开始

### 环境准备

```bash
# 1. 安装依赖（Python 3.11+，包管理使用 uv）
cd backend-next
uv sync --extra dev --extra ml

# 2. 启动服务（生产入口，含崩溃自动重启 watchdog — E2E 实测验证的启动方式）
uv run python -m mindflow.main

# 3. 另开终端生成一次性本地登录链接
uv run python -m mindflow.bootstrap

# 注意：create_app(settings) 是带参工厂，不适用 `uvicorn --factory` 直启。
# 需要热重载的开发场景，修改代码后 Ctrl+C 重启即可（启动 <2s）。
# Windows 依赖 (psutil, pywin32) 由 uv 通过 pyproject.toml 平台标记自动管理。
```

启动时 Alembic 迁移和 SQLite 完整性检查必须成功；迁移失败会终止启动，不会在不兼容
schema 上降级运行。SQLite ALTER TABLE 能力有限，回滚迁移前务必先备份数据库：
`sqlite3 mindflow.db ".backup mindflow_pre_migration.db"`。

### 本地认证

- 启动器持有本地 root token，并通过回环地址申请 60 秒、单次使用的 bootstrap ticket。
- 浏览器在 URL fragment 中取得 ticket，交换为 `HttpOnly`、`SameSite=Strict` 的
  `mindflow_session` Cookie；网页脚本和 URL 均不会接触 root token。
- `/api/*` REST 请求与 `/api/v1/ws` WebSocket 都使用该会话 Cookie。
- 旧的用户名/密码 `/auth/login` 接口已移除；请通过 `python -m mindflow.bootstrap`
  或桌面启动器进入界面。

### 训练 V2 模型

#### Web UI（推荐）

访问前端 `/model-center` 页面。该页面提供四个标签页：

1. **数据准备** — 原始事件数、V2 特征窗口数、反馈分布、可训练性/可评估性指标、7 项质量门禁状态、阻塞项列表
2. **个人基线** — Welford 在线基线统计（数据天数、样本数、特征维度）
3. **模型训练** — 启动/监控训练任务，查看训练结果（影子模式/ready 激活）
4. **模型状态** — V2 ML 模型加载状态、版本、就绪度

#### CLI 训练

```bash
# 合成数据端到端（种子 42 可复现）
uv run python -m mindflow.train --source synthetic_v2
# 真数据训练
uv run python -m mindflow.train --source db
# 模型版本管理
uv run python -m mindflow.train --list-versions
uv run python -m mindflow.train --rollback 20260717
```

#### V2 训练关键阈值

| 指标 | 阈值 | 说明 |
|------|------|------|
| `trainable` | >= 10 个合格窗口 且 >= 2 类标签 | 时间重叠匹配后的显式反馈窗口 |
| `evaluable` | >= 10 个显式样本 且 >= 3 个不同日期 | GroupKFold 需要至少 3 个日期 |
| `baseline_ready` | >= 30 个总样本 | Welford 在线基线 |
| 激活门禁（全部 7 项通过） | — | 见下方 |

#### 7 项激活质量门禁

| 门禁键 | 阈值 | 当前实现状态 |
|--------|------|-------------|
| `minimum_days` | >= 1 天 | 正常评估 |
| `minimum_explicit_feedback` | >= 20 条显式反馈 | 正常评估 |
| `minimum_class_feedback` | 专注 >= 5 且 分心 >= 5 | 正常评估 |
| `balanced_accuracy` | >= 0.50 | `not_evaluated`（需训练后才有报告） |
| `minority_f1` | >= 0.30 | `not_evaluated`（需训练后才有报告） |
| `calibration_better_than_rule` | 训练报告提供证据 | **`not_implemented`** — 硬编码为通过，不可视为绿色 |
| `stable_date_folds` | 训练报告提供证据 | **`not_implemented`** — 硬编码为通过，不可视为绿色 |

后两项 `not_implemented` 在 readiness 响应中暴露为 `not_implemented` 状态，不应解释为通过。

#### 训练任务语义

- **数据来源**：从 `telemetry_repo.list_feature_windows()` 获取 V2 窗口，从 `focus_repo.list_all()` + `telemetry_repo.list_focus_feedback()` 获取带时间戳的反馈，执行时间重叠匹配
- **异步执行**：`TrainingJobService.start_job()` 立即返回 `202 Accepted`，训练在后台 `asyncio.create_task` 中运行
- **状态机**：`pending` → `preparing_data` → `training` → `succeeded` / `failed` / `cancelled`
- **取消规则**：仅允许在 `pending` 或 `preparing_data` 阶段取消；进入 `training` 后拒绝（409），因为线程可能已写入激活的模型制品
- **并发限制**：每个进程最多一个活跃训练任务；已有活跃任务时请求返回 409
- **状态持久化**：运行时状态在内存中，仅保留当前/最新任务；服务重启后丢失历史任务记录
- **无自动重训**：调度器不包含训练 cron 任务（已验证 `test_no_auto_scheduler`）
- **影子模式不替换活跃模型**：`shadow` 训练完成后仅更新 `app.state.v2_training_mode`，不替换已加载的 `v2_model_manager`
- **Ready 发布失败 = 任务失败**：如果 quality gate 通过但 `_refresh_ready_manager()` 抛出异常，任务状态转为 `failed`（非 `succeeded`）
- **数据目录**：制品路径从 `settings.models_dir` / `settings.data_dir` 解析；无 `app.state` 时默认到 `data/models/`

### 多专家智能体会诊

```bash
# 触发当日会诊（5专家+主持人+批评家, ~6-12次LLM调用）
curl -X POST -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/api/v1/panel/today

# 查看最后一次会诊结果
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/api/v1/panel

# 对话式问答（4工具 agent loop）
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -d '{"message":"我今天为什么分心？"}' http://127.0.0.1:8765/api/v1/chat

# 自主行动体开关
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/api/v1/autonomy
```

无 API key 时自动降级（panel -> 单专家 -> 规则引擎，chat -> 规则式安全回复）。

### 评估集（单专家 vs 专家团对比）

```bash
# mock 模式（确定性回放，无需 API key — 默认行为）
uv run python -m mindflow.eval --mode both
# 注意：--mode "mock" 是无效参数。--mode both = 规则引擎 + mock panel 管线验证。

# 真实 LLM（需 DeepSeek API key，~180 调用，需 --yes 确认成本）
uv run python -m mindflow.eval --mode both --live --yes
```

### 运行测试

```bash
# 全量测试（1956 passed, 12 skipped, 1 warning as of 2026-07-29）
uv run python -m pytest tests/ -v

# 带覆盖率报告
uv run python -m pytest --cov=src/mindflow --cov-report=term-missing

# 类型检查（注意：strict mypy 目前有 158 个已知错误，见质量债务说明）
uv run python -m mypy --strict src/mindflow

# 代码风格（注意：Ruff 目前有 94 个已知发现，见质量债务说明）
uv run python -m ruff check src tests

# 数据库迁移
uv run alembic history          # 查看迁移链
uv run alembic upgrade head     # 应用所有待定迁移
# 回滚警告：SQLite ALTER 能力有限，回滚前务必备份：
#   sqlite3 mindflow.db ".backup mindflow_pre_migration.db"
#   alembic downgrade -1
```

---

## API 概览

启动服务后访问 http://localhost:8765/docs 查看完整的 Swagger 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health/live` | GET | 存活检查（免认证） |
| `/api/v1/health/ready` | GET | 就绪检查；迁移、DB 连接、完整性、checkpoint/run-store 状态（503 表示未就绪） |
| `/api/v1/health` | GET | 兼容健康检查（始终 200），含采集器/DB/ML/checkpoint/run-store 状态 |
| `/api/v1/ai/runs` | GET | 工作流运行记录列表（分页，已脱敏 — 不含 prompt/证据/PII，需认证） |
| `/api/v1/ai/runs/{run_id}` | GET | 单条运行详情（含节点事件，需认证） |
| `/api/v1/activities` | GET/POST | 活动事件流 |
| `/api/v1/focus/sessions` | GET | 专注会话列表 |
| `/api/v1/reports/daily` | GET | 日报查询/生成 |
| `/api/v1/reports/weekly` | GET | 周报查询 |
| `/api/v1/analytics/profile` | GET | 行为画像 |
| `/api/v1/analytics/baseline` | GET | 个人行为基线 |
| `/api/v1/analytics/model-status` | GET | V2 ML 模型加载状态与版本 |
| `/api/v1/analytics/training-readiness` | GET | 训练就绪评估（含 7 项质量门禁），[API 文档](../docs/api/model-training.md) |
| `/api/v1/analytics/training-jobs` | POST | 启动训练任务（202 / 409 / 412） |
| `/api/v1/analytics/training-jobs/{job_id}` | GET | 训练任务生命周期状态与报告 |
| `/api/v1/analytics/training-jobs/{job_id}/cancel` | POST | 取消待定/准备中的训练任务 |
| `/api/v1/intervention/trigger` | POST | 手动触发干预 |
| `/api/v1/intervention/history` | GET | 干预历史 |
| `/api/v1/export` | GET | 数据导出（CSV/JSON） |
| `/api/v1/ws` | WS | 实时 WebSocket（会话 Cookie + Host/Origin 校验） |

---

## 配置说明

通过环境变量或 `.env` 文件配置（优先级：环境变量 > `.env` > 默认值）。

`.env` 文件放置在平台数据目录（`platformdirs`），默认为：
- **Windows**: `%LOCALAPPDATA%/mindflow/.env`
- **macOS**: `~/Library/Application Support/mindflow/.env`
- **Linux**: `~/.local/share/mindflow/.env`

### 主要配置项

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MINDFLOW_DB_URL` | `sqlite+aiosqlite:///{data_dir}/mindflow.db` | 数据库连接 URL |
| `MINDFLOW_HOST` | `127.0.0.1` | 服务绑定地址 |
| `MINDFLOW_PORT` | `8765` | 服务端口 |
| `MINDFLOW_COLLECT_INTERVAL_S` | `5` | 采集间隔（秒） |
| `MINDFLOW_HEARTBEAT_PULSETIME_S` | `10` | 心跳合并窗口（秒） |
| `MINDFLOW_EVENT_RETENTION_DAYS` | `30` | 事件数据保留天数（7-90） |
| `MINDFLOW_TIMEZONE` | `local` | 业务时区，可设为 IANA 名称，如 `Asia/Shanghai` |
| `MINDFLOW_RUN_SCHEDULER` | `true` | 是否在当前进程运行调度任务 |
| `MINDFLOW_RUN_COLLECTORS` | `true` | 是否在当前进程运行行为采集器 |
| `MINDFLOW_LOG__LEVEL` | `DEBUG` | 日志级别 |
| `MINDFLOW_LOG__JSON_FORMAT` | `false` | JSON 日志格式 |
| `MINDFLOW_LLM__API_KEY` | — | LLM API 密钥（DeepSeek） |
| `MINDFLOW_LLM__OLLAMA_ENABLED` | `false` | 启用 Ollama 本地模型 |

---

## 编排架构

### 双层设计 (ADR-001)

```
┌──────────────────────────────────────────────────────────┐
│  外层：持久化执行壳 (Scheduler)                           │
│  拥有：时间、任务 claim、心跳、恢复、运行生命周期           │
│  技术：纯 asyncio Scheduler + SQLite scheduled_job_runs 表│
│  不包含 graph/LLM/推理状态                               │
├──────────────────────────────────────────────────────────┤
│  内层：分析推理图 (LangGraph StateGraph)                  │
│  拥有：专家面板逻辑、证据引用、降级链                      │
│  技术：AnalysisGraph / PanelGraph                          │
│  不包含任务 claim/心跳/时间管理                            │
└──────────────────────────────────────────────────────────┘
```

### 图边界 (ADR-004)

- **AnalysisGraph** (`src/mindflow/graph/analysis_graph.py`): 每日分析组成根，实现 `AnalysisWorkflowPort`
- **PanelGraph** (`src/mindflow/graph/panel_graph.py`): AnalysisGraph 的显式子图，封装专家审议（分析师 -> 3 路并行归因 -> 校验 -> 调节器 -> 评论家）
- **ChatGraph** (`src/mindflow/graph/chat_graph.py`): 显式聊天生命周期 StateGraph；v2 图是当前唯一生产聊天路径，旧版 `create_agent` 已移除
- **框架无关端口** (`src/mindflow/ports.py`): `AnalysisWorkflowPort`, `WorkflowRunStorePort`, `BudgetReservationPort` — LangGraph 可替换而不影响调度器
- **ProviderRegistry**: DeepSeek / Ollama / RuleEngine 三层 LLM 降级管理，统一 HTTP 会话池
- **本地 OTel**: OpenTelemetry 写入本地 SQLite，不导出外部（无 PII 在 span 属性中，ADR-003）

---

## 功能标记 (Feature Flags, ADR-005)

分析和聊天已完成 v2 cutover。旧版图选择开关仅为兼容历史 `.env` 保留，修改它们不会恢复旧实现。

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `MINDFLOW_CHECKPOINTING_ENABLED` | bool | `False` | 使用 SQLite 持久化 checkpoint；关闭时使用内存实现 |
| `MINDFLOW_NEW_ANALYSIS_GRAPH` | bool | `True` | 已弃用的兼容字段；始终使用 v2 AnalysisGraph |
| `MINDFLOW_NEW_CHAT_GRAPH` | bool | `True` | 已弃用的兼容字段；始终使用 v2 ChatGraph |

---

## 质量债务

以下命令是**必需的可见性门禁**，但**当前未通过**：

- **Ruff**: 94 个发现（`uv run python -m ruff check src tests`）
- **Mypy (strict)**: 158 个错误，涉及 16 个文件（`uv run python -m mypy --strict src/mindflow`）

这些是已知债务，而非回归问题。不要声称代码风格或类型检查为绿色。

---

## 隐私声明

- **本地存储优先**：所有行为数据存储在本地 SQLite 数据库中，不会上传到云端。
- **LLM 隐私保护**：LLM 分析仅发送聚合后的行为摘要（无窗口标题、文件路径等敏感信息）。摘要中包含的是匿名化的指标数据（切换频率、专注时长比例等）。
- **可选网络功能**：LLM 增强分析需要网络连接（DeepSeek API 或 Ollama），用户可以随时关闭。
- **数据控制权**：用户可通过导出功能随时获取完整数据副本，并可通过数据保留设置控制存储周期。

---

## 项目结构

```
backend-next/
├── alembic/                 # 数据库迁移
│   ├── versions/            # 迁移版本
│   └── env.py               # 异步迁移配置
├── alembic.ini              # Alembic 配置
├── mindflow.spec            # PyInstaller 打包配置
├── pyproject.toml           # 项目元数据与工具配置
├── src/
│   ├── mindflow/
│   │   ├── app.py           # FastAPI 应用工厂
│   │   ├── config.py        # Pydantic Settings 配置
│   │   ├── main.py          # 入口文件
│   │   ├── api/             # API 层
│   │   │   ├── routes/      # 路由模块（含 health, ai_diagnostics, panel, chat 等）
│   │   │   ├── middleware/  # 中间件（认证、日志、限流）
│   │   │   ├── deps.py      # 依赖注入
│   │   │   ├── schemas.py   # API 响应模型（含 DiagnosticsListResponse 等）
│   │   │   └── errors.py    # 错误处理（RFC 9457 ProblemDetail）
│   │   ├── ports.py          # 框架无关端口协议 (AnalysisWorkflowPort 等)
│   │   ├── runtime.py        # RuntimeServices 聚合
│   │   ├── domain/           # 领域模型
│   │   │   ├── events.py     # 事件溯源模型
│   │   │   ├── features.py   # 特征计算
│   │   │   ├── procrastination.py  # 拖延类型分类 + 规则引擎
│   │   │   └── intervention.py     # 干预模型
│   │   ├── infrastructure/   # 基础设施
│   │   │   ├── collectors/   # 平台采集器
│   │   │   ├── repositories/ # 数据访问层
│   │   │   ├── llm/          # LLM 集成
│   │   │   ├── provider_registry.py  # LLM 供应商注册
│   │   │   ├── checkpointer.py       # LangGraph 检查点
│   │   │   ├── database.py   # 数据库引擎
│   │   │   └── notification.py  # 桌面通知
│   │   ├── graph/            # 图编排 (AnalysisGraph, PanelGraph, ChatGraph)
│   │   ├── services/         # 业务服务层
│   │   │   ├── analysis_service.py
│   │   │   ├── report_service.py
│   │   │   ├── intervention_service.py
│   │   │   ├── scheduler.py
│   │   │   ├── panel_service.py
│   │   │   ├── chat_service.py
│   │   │   ├── export_service.py
│   │   │   ├── training_readiness_service.py
│   │   │   ├── training_job_service.py
│   │   │   └── ...
│   │   └── telemetry/        # 本地 OpenTelemetry
│   └── main.py
└── tests/                   # 测试套件
```

---

## 开发规范

- **包管理**：使用 `uv sync --extra dev --extra ml` 安装依赖，不使用 pip/conda/poetry
- **TDD 驱动**：所有新功能先写测试，再实现
- **严格类型**：`mypy --strict` 强制类型标注（当前 158 个已知错误，见质量债务）
- **代码风格**：`ruff` 自动检查（行宽 100，Python 3.11 目标，当前 94 个已知发现）
- **提交规范**：遵循 Conventional Commits（`feat:` / `fix:` / `refactor:`）
- **回滚安全**：功能标记全部默认关闭（旧路径），无需代码回滚（见 ADR-005）

---

## 许可证

MIT License © 2026 RedMoon (胡淙煜)

---

*MindFlow — 理解你的专注，守护你的效率*


## 2026-07-31 ML/LangGraph 实施更新

- 特征 schema v3：`count_confirmed_switches()` 为唯一切换计数实现，默认驻留 10 秒，忽略 `TRANSIENT_PROCESSES`。
- ML 质量门：唯一反馈会话数、7 个反馈日、日期 GroupKFold、规则基线在折内计算；`make_v2_classifier()` 同时用于评估与生产。
- Panel：`POST /panel/today` 支持 `retry_if_degraded`；缓存命中保留 `source/degraded/degradation_path`。
- LangGraph：`PanelGraph` 为唯一活动图；新增确定性 schema 校验、共识强度、`insufficient_data/uncertainty/evidence_gaps` 与 `workflow_node_events` trace。
- 实验：`python scripts/run_experiments.py`；结果见 `data/experiments/20260731_final/`。
