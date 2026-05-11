# MindFlow - 智能专注助手

本地优先的智能专注力追踪应用，监控计算机使用行为，分析注意力模式，生成个性化的抗拖延干预策略。

## 技术栈

| 层 | 技术 |
|------|------|
| 后端 | Python 3.11+ / FastAPI / SQLAlchemy 2.0 / SQLite |
| 前端 | React 19 / TypeScript / Vite / Ant Design / Recharts |
| ML | scikit-learn / 弱监督学习 / HMM 状态推断 / Welford 在线基线 |

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.11+（推荐 conda 环境）
- 前端需要 Node.js 18+

### 一键启动（推荐）

双击 `start.bat`，选择"系统托盘模式"。系统托盘图标会出现在任务栏右下角，右键可操作。

### 手动启动

```bash
cd mindflow-app/backend

# 安装依赖
pip install -r requirements.txt

# 复制配置文件
cp .env.example .env

# 启动后端
uvicorn mindflow.main:app --reload --host 127.0.0.1 --port 8765

# 浏览器打开 Dashboard
# http://localhost:8765/docs
```

配置文件 `.env` 可调整采集间隔、空闲阈值等参数。

### 数据采集

```bash
# 启动采集器
curl -X POST http://localhost:8765/api/v1/collector/start

# 或在 http://localhost:8765/docs 页面操作
```

## 项目结构

```
mindflow-app/
├── backend/
│   ├── mindflow/
│   │   ├── main.py              # FastAPI 应用入口 + CORS
│   │   ├── config.py             # Pydantic Settings（从 .env 加载）
│   │   ├── logging_config.py     # 日志配置（控制台 + 文件轮转）
│   │   ├── tray.py              # 系统托盘程序
│   │   ├── models/
│   │   │   ├── database.py       # SQLAlchemy 引擎 + 会话 + WAL 模式
│   │   │   └── schemas.py        # ORM 模型：User, ActivityLog, FocusSession, DailyReport
│   │   ├── collector/
│   │   │   ├── tracker.py        # Win32 活动窗口 + 空闲检测
│   │   │   └── scheduler.py      # APScheduler 后台定时采集
│   │   ├── analyzer/
│   │   │   ├── features.py       # 专注评分 + 应用排名 + 切换频率
│   │   │   ├── patterns.py       # 专注会话识别 + 日报生成
│   │   │   ├── baseline.py       # 个人行为基线（Welford 在线统计）
│   │   │   ├── deviation.py      # 多特征 Z 分数偏差检测
│   │   │   ├── data_pipeline.py  # 特征工程 + AppClassifier（三层分类）
│   │   │   ├── labeling.py       # 弱监督标注（6 信号共识）
│   │   │   ├── title_analyzer.py # 窗口标题特征提取
│   │   │   ├── ml_models.py      # DBSCAN/KMeans 聚类 + RF 分类器 + HMM
│   │   │   ├── context_packer.py # LLM 上下文打包
│   │   │   └── train.py          # 训练流水线（合成数据 / 真实数据）
│   │   ├── api/
│   │   │   ├── routes.py         # 18 个 REST 端点（前缀 /api/v1）
│   │   │   └── websocket.py      # WebSocket 实时推送（含分析快照）
│   │   ├── llm/                  # LLM 归因分析（Phase 3）
│   │   └── intervention/         # 智能干预（Phase 4）
│   ├── tests/                    # 66 个 pytest 测试
│   ├── data/
│   │   ├── mindflow.db           # SQLite 数据库
│   │   ├── logs/                 # 日志文件
│   │   └── models/               # 训练好的 ML 模型
│   └── .env.example
├── frontend/                     # React Dashboard（待开发）
├── docs/                         # 设计文档 + 实施计划
├── start.bat                     # 一键启动脚本
└── CLAUDE.md
```

## API 端点一览

### 基础

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查（DB + 采集器 + 模型状态） |
| GET | `/api/v1/status` | 采集器状态 + 设置 |
| GET | `/api/v1/data/summary` | 数据隐私面板（记录数、DB 大小、不云端上传） |

### 采集

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/collector/start` | 启动采集 |
| POST | `/api/v1/collector/stop` | 停止采集 |
| GET | `/api/v1/activities/current` | 当前活动窗口 |
| GET | `/api/v1/activities/today` | 今日活动摘要 |

### 专注分析

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/focus/today` | 今日专注报告 |
| GET | `/api/v1/focus/trend` | N 天专注趋势（默认 7，最多 90） |
| GET | `/api/v1/reports/weekly` | 周报 |
| GET | `/api/v1/preferences` | 用户偏好 |
| PUT | `/api/v1/preferences` | 更新偏好（间隔、阈值等） |

### ML 分析

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/analytics/patterns` | 今日专注时段 |
| GET | `/api/v1/analytics/deviation` | 行为偏差检测（异常告警） |
| GET | `/api/v1/analytics/clusters` | 行为模式聚类分布 |
| GET | `/api/v1/analytics/risk` | 分心风险预测（HMM 状态转移） |

### WebSocket

| 路径 | 说明 |
|------|------|
| `/ws/activities` | 每 2 秒推送当前活动 + 分析快照 |

### 标准响应格式

```json
{
  "code": 0,
  "message": "success",
  "data": { ... },
  "timestamp": 1709251200
}
```

## 用真实数据训练

```bash
# 采集至少半天数据后
cd backend
python -m mindflow.analyzer.train --from-db
```

这会从 `activity_logs` 表读取真实数据，走完整流水线：特征提取 → 弱监督标注 → 聚类 → 分类器 → HMM，完成后保存模型到 `data/models/`。

## 隐私

- **所有数据存储在本地 SQLite 文件**，不会上传到任何服务器
- 采集的窗口标题可能包含文件名、URL 等信息
- 可在 `GET /api/v1/data/summary` 查看数据概况
- 删除 `data/mindflow.db` 即可清除所有数据

## 架构决策

### 应用分类三层递进

1. **标题特征分析（TitleAnalyzer）**：基于 URL 域名、文件扩展名、会议关键词
2. **用户标记（UserAppLabel）**：用户在 Dashboard 中手动标记的应用分类
3. **行为推断（ImplicitSignal）**：基于使用时长、时间段、切换模式自动推断

### 本地桌面应用

虽然采用前后端分离架构，但本质是本地桌面应用——"后端"是本地分析引擎，"前端"连的是 localhost。系统托盘模式对其进行了封装。

### 北京时间

所有时间戳使用系统本地时间（北京时间），不再使用 UTC。

## 团队

| 成员 | 职责 |
|------|------|
| 胡淙煜 | 后端架构、数据采集、ML、LLM 集成 |
| 张皓 | 前端 Dashboard、数据可视化 |
| 杨智杰 | 前端组件、数据清洗、API 对接 |

## License

MIT
