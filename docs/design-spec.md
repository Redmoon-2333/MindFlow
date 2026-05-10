# MindFlow 系统设计规格文档

> **项目名称**: MindFlow——基于行为建模与LLM的智能专注助手
> **版本**: v0.1.0-draft
> **日期**: 2026-05-10
> **状态**: 设计阶段

---

## 1. 项目概述

### 1.1 项目背景

拖延行为在大学生群体中普遍存在，约80%-95%的大学生存在不同程度的拖延行为（Steel, 2007）。现有专注类应用（Forest、番茄钟等）主要采用被动管理方式——计时、锁机——无法识别分心的深层心理动因，且统一化规则难以适应个体差异。

### 1.2 项目目标

研发一款基于用户行为建模与大语言模型的智能专注助手，通过无感采集电脑使用行为数据，结合机器学习算法与认知行为疗法（CBT）理论框架，精准识别用户的拖延类型与分心动因，动态生成个性化干预策略。

### 1.3 核心价值主张

> 从"被动时间管理"升级为"主动行为改善"——不只告诉你分心了，而是理解你为什么分心，并帮你从根源解决。

---

## 2. 产品需求规格

### 2.1 用户角色

| 角色 | 描述 |
|------|------|
| 普通用户 | 高校学生/年轻白领，希望改善拖延行为、提升专注力 |
| 管理员 | （未来）可查看群体行为统计的分析人员 |

### 2.2 功能需求

#### F1: 无感数据采集（P0 - 核心基础）

- **F1.1** 后台采集当前活动窗口标题与进程名
- **F1.2** 记录应用使用时长（按进程聚合）
- **F1.3** 统计窗口切换频率（每分钟切换次数）
- **F1.4** 记录活跃时段分布（按小时聚合）
- **F1.5** 采集频率可配置（默认每5秒采样一次）
- **F1.6** 用户可随时暂停/恢复采集
- **F1.7** 所有数据本地存储，不上传云端

#### F2: 行为模式分析（P1 - Demo重点）

- **F2.1** 识别专注时段：连续30分钟以上单一应用使用
- **F2.2** 识别分心模式：频繁切换窗口的时段
- **F2.3** 统计每日各应用使用时长排名
- **F2.4** 计算每日专注得分（0-100）
- **F2.5** 生成周度行为趋势报告

#### F3: LLM归因分析（P2 - 后续迭代）

- **F3.1** 基于行为数据生成用户画像
- **F3.2** 结合CBT框架识别拖延类型（任务畏惧型、信息过载型、决策困难型等）
- **F3.3** 生成个性化的归因解释

#### F4: 智能干预（P2 - 后续迭代）

- **F4.1** 任务拆解：将复杂任务分解为微任务
- **F4.2** 环境优化：检测到分心时提供温和提醒
- **F4.3** 智能排序：基于截止日期与精力状态推荐任务优先级
- **F4.4** 干预强度可调节（温和/标准/严格）

#### F5: Dashboard可视化（P0）

- **F5.1** 实时显示当前专注状态
- **F5.2** 今日应用使用时间饼图
- **F5.3** 专注趋势折线图（日/周）
- **F5.4** 分心模式展示
- **F5.5** 设置页面（采集开关、干预强度、API配置）

### 2.3 非功能需求

| 类别 | 要求 |
|------|------|
| 性能 | 数据采集CPU占用 < 2%，内存 < 50MB |
| 隐私 | 所有行为数据本地存储，不上传云端 |
| 可用性 | 后台静默运行，系统托盘控制 |
| 可扩展性 | 模块化架构，各模块可独立开发测试 |
| 兼容性 | Windows 10/11（Phase 1），macOS（Phase 2） |

---

## 3. 系统架构设计

### 3.1 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    MindFlow System                       │
│                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │  Frontend    │    │  Backend     │    │  Collector   │ │
│  │  (React)     │◄──►│  (FastAPI)   │◄──►│  (Win32 API) │ │
│  │  :5173       │ WS │  :8765       │    │  background  │ │
│  └─────────────┘    └──────┬───────┘    └─────────────┘ │
│                             │                            │
│                      ┌──────┴───────┐                   │
│                      │   SQLite DB   │                   │
│                      │  (data/*.db)  │                   │
│                      └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 模块划分

| 模块 | 目录 | 职责 | 依赖 |
|------|------|------|------|
| collector | `backend/mindflow/collector/` | 窗口监控、数据采集、本地存储 | 无 |
| analyzer | `backend/mindflow/analyzer/` | 特征提取、专注度计算、模式识别 | collector |
| llm | `backend/mindflow/llm/` | LLM调用封装、归因分析 | analyzer |
| intervention | `backend/mindflow/intervention/` | 干预策略引擎 | analyzer, llm |
| api | `backend/mindflow/api/` | RESTful API、WebSocket推送 | 所有业务模块 |
| models | `backend/mindflow/models/` | SQLAlchemy ORM模型 | 无 |
| frontend | `frontend/` | React SPA Dashboard | api |

### 3.3 数据流

```
用户操作行为
    │
    ▼
[Collector] ──原始数据──► [SQLite]
    │                        │
    │                        ▼
    │                  [Analyzer]
    │                        │
    │                        ▼
    │              [行为特征 + 专注评分]
    │                        │
    │                        ▼
    │                   [LLM Module] ←── CBT Framework
    │                        │
    │                        ▼
    │              [归因结果 + 干预建议]
    │                        │
    │                        ▼
    │                 [Intervention]
    │                        │
    ▼                        ▼
[API Server] ◄──────────────┘
    │
    ▼ (WebSocket real-time)
[React Dashboard]
```

### 3.4 技术栈

| 层 | 技术 | 选型理由 |
|----|------|----------|
| 数据采集 | Python 3.11 + psutil + win32gui | Windows原生API，Python生态成熟 |
| 后端框架 | FastAPI + uvicorn | 高性能异步、自动OpenAPI文档、WebSocket支持 |
| ORM | SQLAlchemy 2.0 | Python事实标准、SQLite支持好 |
| 数据存储 | SQLite | 零配置、单文件、适合本地桌面应用 |
| 数据分析 | pandas + scikit-learn | 行为数据分析与聚类 |
| LLM集成 | LangChain + OpenAI兼容API | 模型无关抽象，可切换国产大模型 |
| 定时任务 | APScheduler | 后台定时采集调度 |
| 前端框架 | React 19 + TypeScript + Vite | 团队前端技能匹配 |
| UI组件 | Ant Design | 成熟中文生态、图表组件完善 |
| 图表 | recharts | React原生图表库 |
| 打包 | PyInstaller（未来） | 打包为独立exe |

---

## 4. 数据库设计

### 4.1 ER图（核心实体）

```
User ──1:N──► ActivityLog
                │
                ├── timestamp
                ├── process_name
                ├── window_title
                ├── window_class
                └── duration_seconds

User ──1:N──► FocusSession
                │
                ├── start_time
                ├── end_time
                ├── focus_score
                └── session_type (focus/distraction/neutral)

User ──1:N──► DailyReport
                │
                ├── date
                ├── total_focus_minutes
                ├── total_distraction_minutes
                ├── focus_score
                └── top_apps (JSON)

User ──1:N──► InterventionLog
                │
                ├── triggered_at
                ├── intervention_type
                ├── context_data (JSON)
                └── user_response (accepted/ignored/dismissed)
```

### 4.2 表结构

#### users
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences JSON  -- 干预强度、采集频率等配置
);
```

#### activity_logs
```sql
CREATE TABLE activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    process_name TEXT NOT NULL,      -- 进程名: "chrome.exe"
    window_title TEXT,               -- 窗口标题
    window_class TEXT,               -- 窗口类名
    duration_seconds REAL DEFAULT 0,  -- 距上次采样间隔
    is_idle INTEGER DEFAULT 0,       -- 是否检测为空闲
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE INDEX idx_activity_user_time ON activity_logs(user_id, timestamp);
```

#### focus_sessions
```sql
CREATE TABLE focus_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    focus_score REAL,
    session_type TEXT,               -- 'focus', 'distraction', 'neutral'
    dominant_app TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

#### daily_reports
```sql
CREATE TABLE daily_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date DATE NOT NULL,
    total_focus_minutes REAL DEFAULT 0,
    total_distraction_minutes REAL DEFAULT 0,
    focus_score REAL DEFAULT 0,
    top_apps JSON,                   -- [{"app": "vscode", "minutes": 120}, ...]
    switch_frequency REAL DEFAULT 0, -- 平均每小时切换次数
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

---

## 5. API设计

### 5.1 RESTful端点

| 方法 | 路径 | 说明 | Phase |
|------|------|------|-------|
| GET | `/api/v1/status` | 采集器运行状态 | P0 |
| POST | `/api/v1/collector/start` | 启动采集 | P0 |
| POST | `/api/v1/collector/stop` | 停止采集 | P0 |
| GET | `/api/v1/activities/today` | 今日活动摘要 | P0 |
| GET | `/api/v1/activities/current` | 当前活动窗口 | P0 |
| GET | `/api/v1/focus/today` | 今日专注报告 | P1 |
| GET | `/api/v1/focus/trend?days=7` | 专注趋势（N日） | P1 |
| GET | `/api/v1/analytics/patterns` | 分心模式识别 | P1 |
| GET | `/api/v1/reports/weekly` | 周度报告 | P1 |
| POST | `/api/v1/llm/attribution` | LLM归因分析 | P2 |
| POST | `/api/v1/intervention/trigger` | 触发干预 | P2 |
| GET | `/api/v1/preferences` | 用户偏好 | P0 |
| PUT | `/api/v1/preferences` | 更新偏好 | P0 |

### 5.2 响应格式

```json
{
  "code": 0,
  "message": "success",
  "data": {},
  "timestamp": 1715335200
}
```

### 5.3 WebSocket端点

| 路径 | 说明 |
|------|------|
| `ws://localhost:8765/ws/activities` | 实时推送当前活动与状态变更 |

---

## 6. Demo范围（Phase 0 - 2026年5月）

### 6.1 Demo交付物

1. **数据采集器**: 后台采集窗口标题、进程名、每秒采样
2. **基础分析器**: 专注时段识别、每日应用使用统计
3. **FastAPI后端**: 核心API端点 + WebSocket实时推送
4. **React Dashboard**: 实时状态卡片、今日使用饼图、设置页
5. **SQLite数据层**: 完整ORM模型与自动建表
6. **Conda环境**: 隔离Python环境 + requirements.txt

### 6.2 Demo不包含（预留给后续阶段）

- LLM归因分析模块（需要API Key和更成熟的行为数据）
- 智能干预策略（需要归因模块先行）
- 系统托盘图标（Demo阶段用命令行启动）
- macOS支持
- PyInstaller打包

---

## 7. 参考文献

[1] Steel P. The nature of procrastination. Psychological Bulletin, 2007.
[2] Hofmann S G, et al. The efficacy of cognitive behavioral therapy. Cognitive Therapy and Research, 2012.
[3] Zhao W X, et al. A survey of large language models. arXiv:2303.18223, 2023.
[4] 人工智能医疗器械创新合作平台. 数字疗法产业发展白皮书, 2023.
[5] Steel P, Ferrari J. Sex, education and procrastination. European Journal of Personality, 2013.
[6] 中华医学会. 认知数字疗法中国专家共识(2023), 2023.
[7] Rozental A, Carlbring P. Understanding and Treating Procrastination. Psychology, 2014.
[8] O'Brien W K. Applying the transtheoretical model to academic procrastination. University of Houston, 2002.
