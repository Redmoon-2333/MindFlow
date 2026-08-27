# MindFlow - 智能专注助手

MindFlow 是一个**本地优先的智能专注助手**：记录电脑使用行为的聚合指标，识别专注与拖延模式，结合规则、机器学习和大模型生成可解释的反馈与干预。

> 设计原则：数据默认留在本机；用户可以控制采集、浏览器追踪、自动干预和数据清理；模型必须通过可复现的质量门禁后才能进入 `ready`。

## 核心能力

| 能力 | 说明 | 当前状态 |
|------|------|---------|
| 行为采集 | 活动窗口、空闲状态、5 秒窗口快照 | 已实现 |
| 输入统计 | 30 秒聚合的键盘敲击、鼠标点击、滚动、移动距离、输入活跃时长和交互突发次数 | 已实现 |
| 专注分析 | 会话识别、专注评分、切换频率、个人基线与趋势 | 已实现 |
| 手动软件分类 | 用户把软件设为工作、娱乐或不确定，并支持进程名/窗口标题模式 | API/UI 已实现 |
| 智能干预 | 深度工作保护、节流、内容安全、任务排序、站点拦截 | 已实现；工作态规则仍在持续收敛 |
| LLM 分析 | DeepSeek → Ollama → RuleEngine 三层降级，支持归因、面板和聊天 | 已实现 |
| ML 训练 | V3 特征窗口、用户窗口标签、显式反馈、分类器/聚类/HMM、7 项质量门禁 | 已实现 |
| 数据控制 | 导出、按范围清理、输入/浏览器追踪开关、采集器启停 | 已实现 |
| 可观测性 | 工作流运行、节点事件、本地 OpenTelemetry SQLite exporter | 已实现 |

## 当前推荐判断策略

MindFlow 不把“应用切换次数多”直接等同于分心，而是采用分层判断：

1. **用户规则**：手动分类优先于内置启发式，覆盖长尾软件。
2. **软件语义**：工作类、娱乐类、通信类和不确定类分别处理。
3. **输入行为**：键盘敲击、鼠标点击、输入活跃比、空闲比和切换频率共同判断。
4. **模型预测**：使用经过日期分组评估和校准的 ML 分类器。
5. **干预前置保护**：工作态/深度工作时优先不打扰；不确定时宁可降低提醒，也不把一次切换直接判成拖延。

因此，“一边看教学视频、一边在代码编辑器敲代码”的分屏场景不应仅凭切换次数判定为分心；用户可以把相关工具设为工作软件，模型训练也可使用用户确认过的窗口标签。

## 技术栈

| 层 | 技术 |
|------|------|
| 后端 | Python 3.11+ / FastAPI / SQLAlchemy 2 / SQLite / Alembic |
| 编排 | LangGraph `AnalysisGraph`、`PanelGraph`、`ChatGraph` |
| LLM | DeepSeek、Ollama、本地 RuleEngine；ProviderRegistry 统一生命周期 |
| ML | scikit-learn / XGBoost（可选）/ hmmlearn / Platt sigmoid 校准 |
| 前端 | React 19 / TypeScript / Vite / `openapi-fetch` / plain CSS |
| 包管理 | **uv**（后端）；npm（前端） |

## 快速开始

### 环境要求

- Python 3.11+
- uv
- Node.js 18+
- Windows 10/11；采集器另支持 macOS/Linux 的对应实现

### 启动后端

```bash
cd mindflow-app/backend-next
uv sync --extra dev --extra ml
uv run python -m mindflow.main
```

后端默认监听 `http://127.0.0.1:8765`。启动时会执行 Alembic 迁移和 SQLite 完整性检查；失败时不会在不兼容 schema 上继续运行。

### 本地登录

```bash
cd mindflow-app/backend-next
uv run python -m mindflow.bootstrap
```

启动器使用本地 root token 换取一次性 bootstrap ticket，浏览器最终使用 `HttpOnly`、`SameSite=Strict` 的 session cookie；root token 不进入 URL 或网页脚本。

### 启动前端开发服务器

```bash
cd mindflow-app/frontend
npm install
npm run dev
```

生产桌面入口仍由本地后端/桌面启动脚本提供；Vite 开发服务器不是独立的桌面应用。

## 数据与隐私

### 数据流水线

```text
活动窗口/空闲状态
        ↓ 5 秒采集
交互输入聚合桶（默认 30 秒，仅计数）
        ↓ 5 分钟 rollup
behavior_feature_windows（FEATURE_SCHEMA_VERSION=3，24 维）
        ↓ 反馈时间重叠匹配 + 用户窗口标签
V2TrainingData
        ↓ 日期 GroupKFold + 7 项质量门禁
shadow / ready 模型
```

V3 特征包含 `keypress_rate_per_min`、`mouse_click_rate_per_min`、`scroll_rate_per_min`、`mouse_distance_per_min`、`input_active_ratio`、`interaction_bursts_per_min`、`click_key_ratio`、应用切换和空闲等指标。输入统计不保存按键字符、鼠标坐标或原始轨迹。

浏览器追踪若开启，只保存规范化域名、时长和是否外放，不保存完整 URL。LLM 分析发送的是聚合行为摘要，不发送原始窗口标题、文件路径或个人文件内容。

所有原始数据、反馈和模型制品默认存储在本机。设置页支持关闭输入统计/浏览器追踪、导出数据和按 `interaction`、`browser`、`feedback`、`all` 范围清理数据。

## 用户手动软件分类

后端分类规则表为 `app_classification_rules`，支持：

- 精确进程名匹配（不区分大小写）
- 可选 SQL-LIKE 风格窗口标题模式（`%` 表示通配）
- `priority` 0-100，数值越大越先匹配

用户界面提供三档常用语义：

| 用户选择 | 存储映射 | 用途 |
|---------|---------|------|
| 工作软件 | `code` | 作为工作语义与训练标注来源 |
| 娱乐软件 | `entertainment` | 作为娱乐语义与训练标注来源 |
| 不一定 | `other` / 不建立强规则 | 回退自动分类和模型判断 |

API：

```text
GET    /api/v1/app-classifications
POST   /api/v1/app-classifications
PUT    /api/v1/app-classifications
DELETE /api/v1/app-classifications/{rule_id}
GET    /api/v1/app-classifications/unknown-apps
```

设置页可以先获取未知应用，再点击应用名填入进程名，选择三档分类后保存。

## ML 训练与当前制品

### 训练命令

```bash
cd mindflow-app/backend-next

# 真实数据库训练：默认使用用户窗口标签与 sigmoid 校准
uv run python -m mindflow.train --source db --use-window-labels

# 合成数据训练
uv run python -m mindflow.train --source synthetic_v2

# 版本管理
uv run python -m mindflow.train --list-versions
uv run python -m mindflow.train --rollback <version-tag>
```

`MINDFLOW_TRAINING_USE_WINDOW_LABELS=True` 时，用户确认过的 `behavior_feature_windows.label` 作为附加训练信号，权重为 0.8；显式会话反馈权重为 1.0，且质量门统计仍以唯一反馈会话为准。生产训练默认使用 `calibration="sigmoid"`；小型合成测试可以显式传 `calibration=None`。

### 质量门禁

模型必须同时满足：

- 至少 7 个反馈日
- 至少 20 个唯一显式反馈会话
- focus 和 distracted 各至少 5 个反馈会话
- balanced accuracy ≥ 0.55
- minority F1 ≥ 0.40
- candidate Brier ≤ rule Brier + 0.01
- 日期 GroupKFold 稳定性通过

截至 2026-08-27，正式模型目录中最新已验证制品为 `20260827_173356_a48c0c`：训练报告为 `ready`、`activated=true`、质量门 7/7 通过，训练样本为 focus 2440、distracted 955。运行中的后端进程需要重启后才会重新加载磁盘上的最新模型。

## LLM 与干预

### LLM 降级顺序

```text
DeepSeek / PanelGraph
        ↓ 失败、超时或解析失败
DeepSeek 单专家 / Ollama
        ↓ 仍不可用
RuleEngine / 模板干预
```

面板采用分析师 → 三路归因 → 校验 → 主持人 → 批评家的显式图；真实 DeepSeek 调用使用 120 秒面板总预算，批评家输出有限长，并在并行专家全部失败时做一次整批重试。任何 LLM 输出都会经过 schema、禁词、危机和证据边界校验。

### 工作态保护

自动干预在深度工作或明显工作态时会优先跳过提醒，手动触发仍可绕过。当前工作态判定是保守实现，用户分类规则、输入阈值和运行时抑制策略仍在持续融合；不要把“某个软件被标为工作”理解为所有上下文下都绝对静默。

## API 与诊断

启动后访问 `http://localhost:8765/docs` 查看完整 OpenAPI。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health/live` | GET | 存活检查，免认证 |
| `/api/v1/health/ready` | GET | 迁移、数据库、checkpoint/run-store 就绪检查 |
| `/api/v1/collector` | GET/POST | 查看/启动采集器 |
| `/api/v1/collector/stop` | POST | 停止采集器 |
| `/api/v1/telemetry/status` | GET | 输入/浏览器追踪状态与桶数量 |
| `/api/v1/panel/today` | POST | 触发当日专家面板，可 `force/retry_if_degraded` |
| `/api/v1/analytics/attribution` | POST | 触发单日归因，可指定日期 |
| `/api/v1/chat` | POST | 对话式分析 |
| `/api/v1/analytics/model-status` | GET | ML 加载状态、版本和部署层级 |
| `/api/v1/analytics/training-readiness` | GET | 训练就绪度与质量门状态 |
| `/api/v1/analytics/training-jobs` | POST | 启动后台训练任务 |
| `/api/v1/ai/runs` | GET | 脱敏工作流运行记录 |
| `/api/v1/intervention/trigger` | POST | 手动触发干预 |
| `/api/v1/intervention/history` | GET | 干预历史和用户反馈 |

成功响应为类型化 JSON；错误遵循 RFC 9457 Problem Details。工作流诊断只暴露运行元数据和脱敏节点事件，不重复存储 prompt、原始证据或 PII。

## 测试与质量门

```bash
cd mindflow-app/backend-next
uv run python -m ruff check src tests
uv run python -m mypy --strict src/mindflow
uv run python -m pytest tests/ -q

cd ../frontend
npm run lint
npm run build
```

最近一次完整后端验收结果：`2250 passed`、Ruff 通过、mypy strict 在 163 个源文件中 0 错误；前端 lint 为 0 error（现有 E2E 文件有 10 条 unused-variable warning），生产构建成功。

## 项目结构

```text
mindflow-app/
├── backend-next/
│   ├── alembic/                 # 数据库迁移
│   ├── src/mindflow/
│   │   ├── api/                # REST、WebSocket、中间件
│   │   ├── domain/             # 领域模型与纯规则
│   │   ├── graph/              # AnalysisGraph / PanelGraph / ChatGraph
│   │   ├── infrastructure/     # 采集器、仓库、LLM、通知
│   │   ├── services/           # 分析、干预、调度、训练服务
│   │   └── train/              # V3 训练、评估、版本管理
│   └── tests/                  # 后端 pytest 套件
├── frontend/                   # React + TypeScript + Vite
├── docs/                       # 架构、API、实验和设计文档
├── start.bat                   # 桌面启动入口
├── AGENTS.md                   # 后端执行约束
└── CLAUDE.md                   # 开发指南
```

## 开发规范

- 后端依赖只使用 uv，不使用 pip、conda 或 poetry。
- 新功能先写测试，修改后必须跑 Ruff、mypy、pytest 和受影响的前端构建。
- 维持 V3 schema、确认切换计数、反馈会话统计、LLM schema 校验和隐私边界。
- 不把真实数据库、token、模型制品或含个人行为明细的报告提交到 Git。
- 使用 Conventional Commits；未明确要求时不要自动 commit/push。

## License

MIT License © 2026 RedMoon (胡淙煜)
