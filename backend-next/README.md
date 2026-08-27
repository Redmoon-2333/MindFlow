# MindFlow Backend (`backend-next`)

MindFlow 当前活动后端：FastAPI + LangGraph + SQLite + 本地 ML/LLM 推理。它运行在本机 `127.0.0.1:8765`，由桌面端或前端开发服务器访问。

## 当前状态（2026-08-27）

- **Feature schema**：V3，24 维窗口特征。
- **输入统计**：30 秒聚合桶记录键盘敲击、鼠标点击、滚动、鼠标移动距离、输入活跃时长和交互突发次数；5 分钟 rollup 产生 `keypress_rate_per_min`、`mouse_click_rate_per_min`、`click_key_ratio` 等特征。
- **用户标注**：支持 `focus` / `distracted` / `mixed` 会话反馈与用户确认的窗口标签。
- **用户软件分类**：`app_classification_rules` + `/api/v1/app-classifications`，支持进程名、窗口标题模式和优先级。
- **LLM**：DeepSeek / Ollama / RuleEngine 三层降级；AnalysisGraph、PanelGraph、ChatGraph 为生产图路径。
- **ML**：训练默认使用窗口标签增广与 Platt sigmoid 校准；质量门通过后才发布 `ready` 模型。
- **最新已验证模型**：`20260827_173356_a48c0c`，`training_report.json` 为 `ready/activated=true`，7 项质量门全部通过。
- **最近验证**：后端 `2250 passed`、Ruff 通过、mypy strict 163 个源文件 0 错误；前端 `oxlint` 0 error、Vite 构建成功。

> 模型文件写在平台用户数据目录。新模型写入磁盘后，已经运行的后端进程需要重启才能加载；`model-status` 显示的是当前进程内存中的状态，不等同于磁盘上是否存在最新制品。

## 目录结构

```text
backend-next/
├── alembic/                         # Alembic 迁移
├── scripts/                         # 实验与运维脚本
├── src/mindflow/
│   ├── api/routes/                  # REST 路由
│   ├── api/middleware/              # auth / host / logging / ratelimit
│   ├── agents/                      # Expert 定义、LLM gateway、解析 helper
│   ├── domain/                      # 纯领域模型、特征与规则
│   ├── graph/                       # AnalysisGraph / PanelGraph / ChatGraph
│   ├── infrastructure/
│   │   ├── collectors/              # Win32、macOS、X11、Wayland 采集器
│   │   ├── repositories/            # SQLAlchemy Core 仓库
│   │   ├── provider_registry.py     # LLM provider 生命周期
│   │   ├── checkpointer.py          # 内存/SQLite checkpoint adapter
│   │   └── notification.py           # 桌面通知
│   ├── services/                    # 业务编排与训练任务
│   └── train/                       # V3 训练、评估、模型版本
├── tests/                           # pytest
├── pyproject.toml
└── README.md
```

分层依赖方向：`domain → infrastructure → services → api/agents`。调度器只负责时间、claim、heartbeat 和生命周期；分析/聊天推理由图实现，通过 framework-neutral port 接入。

## 安装与启动

```bash
cd mindflow-app/backend-next
uv sync --extra dev --extra ml
uv run python -m mindflow.main
```

生产入口包含 watchdog。不要使用 `uvicorn --factory` 启动 `create_app(settings)`；需要换配置或加载新模型时，停止后重新执行上述命令。

本地登录：

```bash
uv run python -m mindflow.bootstrap
```

`.env` 位于 `platformdirs.user_data_dir("mindflow")` 下的用户数据目录，环境变量优先级高于 `.env` 和默认值。API key 只应放在用户数据目录，不应进入仓库。

## 数据流水线

```text
activity_events
    ↓ 5 秒活动窗口采集
interaction_buckets（默认 30 秒输入聚合）
    ↓ telemetry rollup
behavior_feature_windows（schema_version=3，24 维）
    ↓ 与 focus_session_feedback 按时间重叠匹配
V2TrainingData
    ↓ 日期 GroupKFold、规则基线、校准与稳定性检查
ModelManager → shadow / ready
```

### 输入与隐私边界

`interaction_buckets` 只保存计数和聚合时长：

- `keypress_count`
- `mouse_click_count`
- `scroll_delta`
- `mouse_distance_px`
- `input_active_s`
- `interaction_burst_count`

不保存按键字符、鼠标坐标或逐事件输入轨迹。浏览器 telemetry 只保存规范化域名、时长、外放标记和哈希化 context key。LLM 只接收聚合行为摘要，不接收原始窗口标题、路径或文件内容。

## 用户软件分类

分类规则存储在 `app_classification_rules`：

```text
process_name             精确进程名，不区分大小写
window_title_pattern     可选 SQL-LIKE 模式，% 为通配符
category                 code / document / browser_work / communication /
                         entertainment / social / other
priority                 0-100，越高越先匹配
```

API：

```text
GET    /api/v1/app-classifications
POST   /api/v1/app-classifications
PUT    /api/v1/app-classifications
DELETE /api/v1/app-classifications/{rule_id}
GET    /api/v1/app-classifications/unknown-apps
```

前端设置页把常用语义映射为：

```text
工作软件   → code
娱乐软件   → entertainment
不一定     → other / 不建立强规则，回退自动判断
```

`UserAppClassifier` 的解析顺序是：用户规则 → Bilibili 学习标题启发式 → 内置分类器。分类规则是实时分析的强语义入口；但“工作软件一定完全不打扰”仍取决于当前活动上下文，不能把单个软件规则等同于全局静默策略。

## 训练与模型发布

### 命令

```bash
# 真实数据；窗口 label 作为附加训练信号，显式反馈仍优先
uv run python -m mindflow.train --source db --use-window-labels

# 合成 V3 数据
uv run python -m mindflow.train --source synthetic_v2

# 版本管理
uv run python -m mindflow.train --list-versions
uv run python -m mindflow.train --rollback <version-tag>
```

`MINDFLOW_TRAINING_USE_WINDOW_LABELS=True` 时，用户确认的窗口标签权重为 0.8，显式反馈权重为 1.0；质量门的反馈统计仍按**唯一反馈会话**计算，而不是重叠窗口数。生产训练默认传 `calibration="sigmoid"`，评估和最终部署使用同一校准配置。

### 训练质量门

`evaluate_v2_quality_gate()` 要求以下检查全部通过：

| 检查 | 阈值 |
|------|------|
| `minimum_days` | ≥ 7 个反馈日 |
| `minimum_explicit_feedback` | ≥ 20 个唯一反馈会话 |
| `minimum_class_feedback` | focus ≥ 5 且 distracted ≥ 5 |
| `balanced_accuracy` | ≥ 0.55 |
| `minority_f1` | ≥ 0.40 |
| `calibration_better_than_rule` | candidate Brier ≤ rule Brier + 0.01 |
| `stable_date_folds` | 日期分组折稳定性通过 |

通过后写入 `ready` 模型并更新 `latest.json`；未通过只产生 `shadow` 制品，不替换当前活跃模型。发布失败也会把训练任务标记为失败，不会伪造 ready 状态。

## LLM 图与降级

### AnalysisGraph

统一承接 scheduler、API、chat tool 和 auto-intervention 的分析请求，负责幂等、预算、危机门、证据构建、PanelGraph、降级和持久化。活动 runtime 通过 ContextVar 传递，不进入 checkpointable state，避免 msgpack 序列化仓库/客户端对象。

### PanelGraph

```text
Analyst
  → 3 路 Attribution（并行）
  → validation / citation / forbidden-word checks
  → conflict detection
  → rebuttal（并行）
  → Moderator（deepseek-reasoner）
  → Critic
```

当前真实 provider 参数：

- 面板总 wall-clock 预算：120 秒
- critic 输出限制：`critique_detail ≤ 300` 字
- 并行专家批全部返回空时：3 秒回退后整批重试一次
- 部分失败或 provider 不可用：按设计降级到单专家/Ollama/RuleEngine

### ChatGraph

聊天使用显式生命周期图，支持工具调用、证据引用和降级安全回复。聊天端点不能替代面板：聊天适合交互式解释，面板适合完整的多专家会诊。

## 干预与工作态

自动干预顺序：

```text
选择干预类型
    → 工作态/深度工作保护
    → throttle 与 daily slot
    → LLM 消息或模板
    → safety guard
    → 持久化、WebSocket、桌面通知
```

工作态保护的目标是避免在“代码编辑 + 教学视频 + 持续输入”的场景中打扰用户。当前代码使用保守的深度工作与工作类应用信号组合，并保留手动触发绕过能力；`intervention_work_suppress_*` 配置项用于后续把用户分类规则、焦点阈值、键盘频率阈值和浏览器工作语义统一接入运行时门。

## 主要 API

访问 `http://localhost:8765/docs` 查看 OpenAPI。

| 路径 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health/live` | GET | 存活检查，免认证 |
| `/api/v1/health/ready` | GET | 迁移、数据库、checkpoint、run-store 就绪检查 |
| `/api/v1/collector` | GET/POST | 查看/启动采集器 |
| `/api/v1/collector/stop` | POST | 停止采集器 |
| `/api/v1/telemetry/status` | GET | 输入/浏览器 telemetry 状态 |
| `/api/v1/telemetry/preferences` | PATCH | 修改 telemetry 开关与保留期 |
| `/api/v1/telemetry/data` | DELETE | 按 scope 清理 telemetry 数据 |
| `/api/v1/app-classifications` | GET/POST/PUT | 软件分类规则 |
| `/api/v1/app-classifications/{rule_id}` | DELETE | 删除分类规则 |
| `/api/v1/app-classifications/unknown-apps` | GET | 未分类进程列表 |
| `/api/v1/analytics/attribution` | POST | 单日归因 |
| `/api/v1/panel/today` | POST | 当日多专家面板 |
| `/api/v1/panel` | GET | 读取已持久化面板结果 |
| `/api/v1/chat` | POST | 对话与工具调用 |
| `/api/v1/analytics/model-status` | GET | 模型加载/ready/version |
| `/api/v1/analytics/training-readiness` | GET | 训练就绪度和质量门 |
| `/api/v1/analytics/training-jobs` | POST | 启动后台训练 |
| `/api/v1/analytics/training-jobs/{job_id}` | GET | 训练状态与报告 |
| `/api/v1/ai/runs` | GET | 脱敏工作流运行记录 |
| `/api/v1/intervention/trigger` | POST | 手动干预 |
| `/api/v1/intervention/history` | GET | 干预历史/反馈 |

## 测试命令

```bash
cd mindflow-app/backend-next
uv run python -m ruff check src tests
uv run python -m mypy --strict src/mindflow
uv run python -m pytest tests/ -q
```

前端：

```bash
cd mindflow-app/frontend
npm run lint
npm run build
```

当前质量基线为后端 `2250 passed`、Ruff 通过、mypy strict 163 文件 0 错误；前端 lint 0 error（现有 E2E 文件的 unused-variable 警告仍存在）且生产构建成功。

## 配置速查

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MINDFLOW_HOST` | `127.0.0.1` | 服务地址 |
| `MINDFLOW_PORT` | `8765` | 服务端口 |
| `MINDFLOW_COLLECT_INTERVAL_S` | `5` | 活动采集间隔 |
| `MINDFLOW_RUN_SCHEDULER` | `True` | 是否运行调度器 |
| `MINDFLOW_RUN_COLLECTORS` | `True` | 是否运行活动采集器 |
| `MINDFLOW_CHECKPOINTING_ENABLED` | `False` | 是否使用 SQLite checkpoint |
| `MINDFLOW_TRAINING_USE_WINDOW_LABELS` | `True` | 是否把用户窗口标签纳入训练 |
| `MINDFLOW_LLM__TIMEOUT_S` | `30` | 单次 LLM 请求超时 |
| `MINDFLOW_LLM__MAX_RETRIES` | `1` | LLM 重试预算 |
| `MINDFLOW_LLM__OLLAMA_ENABLED` | `False` | 是否启用本地 Ollama |
| `MINDFLOW_INTERVENTION_WORK_SUPPRESS_ENABLED` | `True` | 工作态保护总开关（当前仍在持续接入用户规则） |

数据库迁移回滚前必须先备份 SQLite：

```bash
sqlite3 mindflow.db ".backup mindflow_pre_migration.db"
```

## 开发约束

- 只用 uv 管理 Python 依赖。
- 保持 `FEATURE_SCHEMA_VERSION=3` 和 `count_confirmed_switches()` 的唯一切换计数实现。
- 质量门统计唯一反馈会话，不用重叠窗口冒充反馈量。
- 主持人输出必须通过 `validate_verdict_schema()` 后才交给 critic。
- 真实行为数据、token、模型制品和含个人行为明细的报告不提交 Git。
- 未明确要求时不自动 commit/push；新增功能先写测试并保持三条质量门绿色。

## License

MIT License © 2026 RedMoon (胡淙煜)
