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
| 行为采集 | 主动窗口监测（Win/Mac/Linux），5 秒采集间隔，心跳合并 | Wave 3 |
| 专注分析 | 会话识别，专注评分，基线偏差检测，拖延类型分类 | Wave 5 |
| 数据报告 | 日报/周报生成，App 使用统计，趋势分析 | Wave 5 |
| LLM 分析 | DeepSeek / Ollama 三层降级，CBT 行为分析 | Wave 6 |
| 智能干预 | 基于规则的干预生成，节流控制，深度工作不打扰 | Wave 7 |
| 数据导出 | CSV / JSON 导出，日期范围筛选 | Wave 8b |
| ML 训练 | 合成数据生成，HMM 训练，聚类分析 CLI | Wave 8a |

---

## 架构

```
┌──────────────────────────────────────────────────┐
│                MindFlow App                         │
│  ┌──────────────────────┐  ┌──────────────────┐   │
│  │  FastAPI (REST :8765) │  │  WebSocket /ws    │   │
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

- **运行时**: Python 3.11+, uv icon (async ASGI)
- **Web 框架**: FastAPI 0.115+
- **数据库**: SQLite + SQLAlchemy (async) + Alembic 迁移
- **调度**: APScheduler (cron 任务 + 定时干预检查)
- **ML**: scikit-learn, hmmlearn (本地训练/预测)
- **打包**: PyInstaller (单文件桌面应用)

---

## 快速开始

### 环境准备

```bash
# 1. 创建 conda 环境
conda create -n mindflow python=3.11
conda activate mindflow

# 2. 安装依赖
cd backend-next
pip install -e ".[dev]"

# 3. 启动服务（生产入口，含崩溃自动重启 watchdog — E2E 实测验证的启动方式）
python -m mindflow.main

# 注意：create_app(settings) 是带参工厂，不适用 `uvicorn --factory` 直启。
# 需要热重载的开发场景，修改代码后 Ctrl+C 重启 python -m mindflow.main 即可（启动 <2s）。
```

### 训练 ML 模型

```bash
# 使用合成数据训练专注分析模型
python -m mindflow.analyzer.train
```

### 运行测试

```bash
# 全量测试
pytest -v

# 带覆盖率报告
pytest --cov=src/mindflow --cov-report=term-missing

# 类型检查
mypy src/mindflow

# 代码风格
ruff check src/mindflow
```

---

## API 概览

启动服务后访问 http://localhost:8765/docs 查看完整的 Swagger 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查（免认证） |
| `/api/v1/activities` | GET/POST | 活动事件流 |
| `/api/v1/focus/sessions` | GET | 专注会话列表 |
| `/api/v1/reports/daily` | GET | 日报查询/生成 |
| `/api/v1/reports/weekly` | GET | 周报查询 |
| `/api/v1/analytics/profile` | GET | 行为画像 |
| `/api/v1/intervention/trigger` | POST | 手动触发干预 |
| `/api/v1/intervention/history` | GET | 干预历史 |
| `/api/v1/export` | GET | 数据导出（CSV/JSON） |
| `/api/v1/ws` | WS | 实时 WebSocket |

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
| `MINDFLOW_LOG__LEVEL` | `DEBUG` | 日志级别 |
| `MINDFLOW_LOG__JSON_FORMAT` | `false` | JSON 日志格式 |
| `MINDFLOW_LLM__API_KEY` | — | LLM API 密钥（DeepSeek） |
| `MINDFLOW_LLM__OLLAMA_ENABLED` | `false` | 启用 Ollama 本地模型 |

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
│   │   │   ├── routes/      # 路由模块
│   │   │   ├── middleware/  # 中间件（认证、日志、限流）
│   │   │   ├── deps.py      # 依赖注入
│   │   │   └── errors.py    # 错误处理
│   │   ├── domain/          # 领域模型
│   │   │   ├── events.py    # 事件溯源模型
│   │   │   ├── features.py  # 特征计算
│   │   │   ├── procrastination.py  # 拖延类型分类 + 规则引擎
│   │   │   └── intervention.py     # 干预模型
│   │   ├── infrastructure/  # 基础设施
│   │   │   ├── collectors/  # 平台采集器
│   │   │   ├── repositories/ # 数据访问层
│   │   │   ├── llm/         # LLM 集成
│   │   │   ├── database.py  # 数据库引擎
│   │   │   └── notification.py  # 桌面通知
│   │   └── services/        # 业务服务层
│   │       ├── analysis_service.py
│   │       ├── report_service.py
│   │       ├── intervention_service.py
│   │       ├── scheduler.py
│   │       ├── export_service.py
│   │       └── ...
│   └── main.py
└── tests/                   # 测试套件
```

---

## 开发规范

- **TDD 驱动**：所有新功能先写测试，再实现
- **严格类型**：`mypy --strict` 强制类型标注
- **代码风格**：`ruff` 自动检查（行宽 100，Python 3.11 目标）
- **提交规范**：遵循 Conventional Commits（`feat:` / `fix:` / `refactor:`）

---

## 许可证

MIT License © 2026 RedMoon (胡淙煜)

---

*MindFlow — 理解你的专注，守护你的效率*
