# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 活跃后端

当前活跃后端是 **`backend-next/`**（FastAPI 重写版，分层架构 + LangChain/LangGraph）。
旧版 `backend/`（Phase 0，同步 SQLAlchemy，无 LLM 层）已删除，`backend-next` 完全不依赖它。

## Build & Test Commands

```bash
# 后端（Python 3.11+，conda 环境: mindflow）
cd mindflow-app/backend-next

# 安装依赖（开发模式）
pip install -e ".[dev]"

# 启动生产服务（含崩溃自动重启 watchdog — E2E 实测验证的启动方式）
python -m mindflow.main
# 注意：create_app(settings) 是带参工厂，不适用 `uvicorn --factory` 直启。
# 热重载开发场景：修改代码后 Ctrl+C 重启 python -m mindflow.main 即可（启动 <2s）。

# 跑全部测试
python -m pytest tests/ -q

# 跑单个测试文件 / 单个用例
python -m pytest tests/test_llm_client.py -v
python -m pytest tests/test_features.py::test_calculate_focus_score -v

# Lint / 类型检查
python -m ruff check src tests
python -m mypy --strict src/mindflow

# 数据库迁移
alembic upgrade head

# 训练 ML 模型（合成数据 / 真数据 / 版本管理）
python -m mindflow.train --source synthetic
python -m mindflow.train --source db
python -m mindflow.train --list-versions

# 评估集（mock 确定性回放 / 真实 LLM）
python -m mindflow.eval --mode both
python -m mindflow.eval --mode both --live --yes

# 前端（独立目录，就绪时）
cd mindflow-app/frontend && npm install && npm run dev
```

API 文档：启动后端后访问 `http://localhost:8765/docs`。

## Architecture

MindFlow 是本地优先的智能专注助手：监测电脑使用行为、分析行为模式、生成个性化抗拖延干预。

```
Frontend (React/TS) ←→ Backend (FastAPI :8765) ←→ Collector (跨平台活动采集)
                              ↓
                         SQLite (WAL mode, 本地)
```

**分层依赖方向**：`domain` → `infrastructure` → `services` → `api` / `agents`（单向，不可逆）。

| 层 | 路径 | 职责 |
|------|------|------|
| `config` | `src/mindflow/config.py` | Pydantic BaseSettings，从 `.env`/环境变量加载，`{data_dir}` 占位符解析 |
| `domain` | `src/mindflow/domain/` | 纯领域模型：事件、特征、基线、偏差、拖延类型、证据合同。零框架依赖（纯 stdlib + typing） |
| `infrastructure` | `src/mindflow/infrastructure/` | 采集器（Win32/macOS/X11/Wayland）、SQLAlchemy 仓库、LLM 客户端、安全（token/危机检测）、通知 |
| `services` | `src/mindflow/services/` | 业务编排：分析、报告、干预、节流、证据构建、面板、聊天、调度、维护、导出 |
| `agents` | `src/mindflow/agents/` | 多专家 LLM 面板（LangGraph StateGraph）：orchestrator + 5 专家 + 冲突检测 + LangChain 网关 |
| `api` | `src/mindflow/api/` | REST 路由 + WebSocket + 中间件（auth/host/ratelimit/logging）+ RFC 9457 错误处理 |
| `train` | `src/mindflow/train/` | ML 训练 CLI（合成数据 / 聚类 / 分类 / HMM / 版本管理），离线用，不接入运行时 |
| `eval` | `src/mindflow/eval/` | 评估集（30 场景）+ mock/real LLM 对比 runner |

## Key Design Decisions

- **全本地数据**：SQLite WAL 模式，无云端上传，隐私优先。
- **无全局单例**：所有共享状态挂在 `app.state`，`create_app(settings)` 工厂装配，依赖注入贯穿。
- **三层 LLM 降级**（`config.LLMSettings`）：L1 DeepSeek（需 key）→ L2 Ollama 本地 → L3 规则引擎（永远可用）。
- **LLM 输出当不可信**：Pydantic v2 strict + `extra="forbid"` + 禁词校验器（NF-S7），citation 代码强制校验，独立危机检测器在 LLM 前硬门控。
- **异步 SQLAlchemy**：每个仓库方法开独立 `async with session_factory()`，不跨请求共享 session。
- **时区**：全链路 timezone-aware UTC。
- **公共 API 零破坏**：所有重构保持对外接口不变（已有 LangChain 迁移先例）。

## Dataset Context

`data/datasets/` 下两套外部数据集（经 Git LFS 管理，不入 git 常规存储）：
- `manictime/`：44 个真实用户活动记录 CSV（ManicTime 导出，含 PII）
- `awt-labelled/`：Academic Work Tracker 标注数据 + 预处理 notebook

合成数据生成器在 `train/synthetic_data.py`，建模真实工作日/周末行为模式。

## Docs

- `backend-next/README.md` — 后端快速开始与架构
- `docs/handbook/` — 全栈手册（6 章）
- `docs/redesign/` — 重设计文档（需求/架构/测试/技术使用/智能体升级）
