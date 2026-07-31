# MindFlow - 智能专注助手

本地优先的智能专注力追踪应用，监控计算机使用行为，分析注意力模式，生成个性化的抗拖延干预策略。

## 技术栈

| 层 | 技术 |
|------|------|
| 后端 | Python 3.11+ / FastAPI / SQLAlchemy 2.0 / SQLite / LangGraph |
| 前端 | React 19 / TypeScript / Vite / react-router-dom / openapi-fetch / plain CSS |
| ML | scikit-learn / hmmlearn / Welford 在线基线 / 弱监督学习 |
| 包管理 | **uv**（取代 pip/conda） |

## 快速开始

### 环境要求

- Windows 10/11（采集器支持 macOS/Linux）
- Python 3.11+
- 前端需要 Node.js 18+

### 一键启动（推荐）

双击 `start.bat`，选择"系统托盘模式"。

### 手动启动

```bash
cd mindflow-app/backend-next

# 安装依赖（dev + ML extras）
uv sync --extra dev --extra ml

# 启动后端
uv run python -m mindflow.main

# 生成一次性本地登录链接
uv run python -m mindflow.bootstrap

# 浏览器打开
# http://localhost:8765/docs
```

启动时 Alembic 迁移和 SQLite 完整性检查必须成功；迁移失败会终止启动。

### 前端启动

```bash
cd mindflow-app/frontend
npm install
npm run dev
# http://localhost:5173
```

## 2026-07-31 升级与实验

- 特征 schema 已升级到 v3：应用切换采用“确认切换”计数（默认驻留 10 秒），系统瞬时窗口不计数；历史窗口通过回填脚本重建。
- ML 质量门改为统计唯一反馈会话数，要求至少 7 个反馈日、20 条唯一反馈、每类 5 条；评估与部署共用同一 EnsembleClassifier，规则基线在日期 GroupKFold 折内计算。
- `POST /panel/today` 支持 `force` / `retry_if_degraded`；降级结果再次点击会重新尝试 DeepSeek，缓存命中保留真实的 `source/degraded/degradation_path`。
- LangGraph `PanelGraph` 为唯一活动编排；新增确定性 schema 校验、共识强度注入、`insufficient_data/uncertainty/evidence_gaps` 输出和 `workflow_node_events` trace。
- 实验与报告：`backend-next/scripts/run_experiments.py` 可一键跑 3 轮 ML 和 3 轮 LangGraph，产物落在 `data/experiments/`。
- 历史 `docs/optimization-plan-codex-review.md` 已删除，相关结论并入本文档与架构文档。

## 数据流水线

```
activity_events（原始活动事件）
    ↓ 5s 采集 + 心跳合并
telemetry rollup（V2 特征窗口，schema_version=2）
telemetry rollup（V3 特征窗口，schema_version=3）
    ↓ 时间窗口重叠匹配
focus_session_feedback（用户显式标注）
    ↓ join focus_sessions 得到带时间戳的反馈
prepare_v2_training_data()
    ↓ 时间重叠过滤 + 显式反馈优先
V2TrainingData（合格窗口 + 标签）
    ↓ 质量门禁 7 项检查
训练 → shadow / ready 模型
```

**核心概念**：
- **基线（Baseline）**：Welford 在线统计，日常行为的实时对比基准，非 ML 模型
- **ML 训练**：批量离线训练（分类器 + 聚类 + HMM），通过模型中心或 CLI 触发
- **数据存在 ≠ 可训练**：原始事件需先经 telemetry rollup 成 V2 特征窗口；显式反馈的时间必须与窗口范围重叠

## 项目结构

```
mindflow-app/
├── backend-next/               # 当前活动后端（FastAPI + LangGraph）
│   ├── src/mindflow/
│   │   ├── main.py             # 入口
│   │   ├── app.py              # FastAPI 应用工厂
│   │   ├── api/routes/         # 路由（含 analytics/training-readiness 等）
│   │   ├── services/           # 业务服务层
│   │   │   ├── training_readiness_service.py
│   │   │   └── training_job_service.py
│   │   ├── train/              # ML 训练流水线
│   │   └── domain/             # 领域模型
│   └── tests/                  # pytest 测试套件
├── frontend/                   # React / Vite / TypeScript
│   └── src/pages/
│       └── ModelCenter.tsx     # 模型中心页面（/model-center）
├── docs/                       # 设计文档 + API 规范 + ADR
├── start.bat                   # 一键启动
└── CLAUDE.md
```

**注意**：遗留 `backend/` 目录（Phase 0）已删除。`backend-next` 与之零依赖。

## API 端点一览

### 基础 & 健康

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health/live` | 存活检查（免认证） |
| GET | `/api/v1/health/ready` | 就绪检查（503 表示未就绪） |
| GET | `/api/v1/health` | 兼容健康检查 |

### 活动采集

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/activities` | 活动事件流（分页、过滤） |
| GET | `/api/v1/activities/current` | 当前活动窗口 |

### 专注分析

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/focus` | 专注会话列表 |
| GET | `/api/v1/focus/trend` | N 天专注趋势 |
| POST | `/api/v1/focus/{session_id}/feedback` | 提交会话反馈标注 |
| GET | `/api/v1/reports/daily` | 日报查询/生成 |
| GET | `/api/v1/reports/weekly` | 周报 |
| GET | `/api/v1/analytics/patterns` | 专注时段分析 |
| GET | `/api/v1/analytics/profile` | 行为画像 |
| GET | `/api/v1/analytics/baseline` | 个人行为基线 |
| GET | `/api/v1/analytics/model-status` | ML 模型状态 |

### 模型中心（V2 训练）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/analytics/training-readiness` | 训练就绪评估（含 7 项质量门禁） |
| POST | `/api/v1/analytics/training-jobs` | 启动训练任务（202 / 409 / 412） |
| GET | `/api/v1/analytics/training-jobs/{job_id}` | 训练任务状态与报告 |
| POST | `/api/v1/analytics/training-jobs/{job_id}/cancel` | 取消待定/准备中的任务 |

更多端点详见 [`docs/api/model-training.md`](docs/api/model-training.md) 和 `backend-next/README.md`。

### WebSocket

| 路径 | 说明 |
|------|------|
| `/api/v1/ws` | 实时 WebSocket（需会话 Cookie） |

> **响应格式**：成功返回类型化 JSON 模型（无统一信封）；错误遵循 RFC 9457 Problem Details（`type`, `title`, `status`, `detail`, `instance` + 合并的额外字段）。

## 训练 ML 模型

### Web UI

访问前端 `/model-center` 页面查看训练就绪状态、质量门禁结果，启动和监控训练任务。

### CLI

```bash
cd backend-next
# 合成数据训练
uv run python -m mindflow.train --source synthetic_v2
# 真实数据训练
uv run python -m mindflow.train --source db
# 模型版本管理
uv run python -m mindflow.train --list-versions
```

### 模型模式

| 模式 | 说明 |
|------|------|
| `rule_engine_only` | 仅使用规则引擎，无 ML 模型 |
| `shadow` | 训练完成但不替换活跃模型（评估期） |
| `ready` | 训练完成且通过质量门禁，替换为当前活跃模型 |

## 隐私

- **所有数据存储在本地 SQLite 文件**，不会上传到任何服务器
- LLM 分析仅发送聚合后的行为摘要（无窗口标题、文件路径等敏感信息）
- 用户可通过导出功能获取完整数据副本，通过数据保留设置控制存储周期

## 架构决策

### 本地桌面应用

采用前后端分离架构，但本质是本地桌面应用——"后端"是本地分析引擎，"前端"连的是 localhost。系统托盘模式对其进行了封装。

### 双层编排设计

外层调度器（纯 asyncio + SQLite claims/heartbeats）与内层 LangGraph 分析图分离，通过框架无关端口解耦。详见 [`backend-next/README.md`](backend-next/README.md) 和 ADR-001。

### 时区

所有 datetime 值在内部使用 timezone-aware UTC；业务边界通过 `MINDFLOW_TIMEZONE` 配置（默认 `local` 或 IANA 名称如 `Asia/Shanghai`）转换显示。

## 团队

| 成员 | 职责 |
|------|------|
| 胡淙煜 | 后端架构、数据采集、ML、LLM 集成 |
| 张皓 | 前端 Dashboard、数据可视化 |
| 杨智杰 | 前端组件、数据清洗、API 对接 |

## License

MIT
