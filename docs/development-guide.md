# MindFlow 开发环境搭建指南

## 环境要求

- Windows 10/11 64位
- Anaconda/Miniconda（已安装）
- Node.js 18+（前端）
- Git

## 一、Conda 环境

```bash
# 创建并激活环境
conda create -n mindflow python=3.11 -y
conda activate mindflow

# 安装后端依赖
cd mindflow-app/backend
pip install -r requirements.txt
```

## 二、后端启动

```bash
conda activate mindflow
cd mindflow-app/backend

# 开发模式启动（热重载）
uvicorn mindflow.main:app --reload --host 127.0.0.1 --port 8765

# 访问:
# API文档: http://localhost:8765/docs
# 健康检查: http://localhost:8765/api/v1/status
```

## 三、前端启动

```bash
cd mindflow-app/frontend

# 安装依赖
npm install

# 开发模式启动
npm run dev

# 访问: http://localhost:5173
```

## 四、项目结构

```
mindflow-app/
├── backend/
│   ├── mindflow/
│   │   ├── __init__.py
│   │   ├── config.py            # 全局配置
│   │   ├── main.py              # FastAPI 入口
│   │   ├── collector/           # 数据采集模块
│   │   │   ├── __init__.py
│   │   │   ├── tracker.py       # 窗口跟踪器
│   │   │   └── scheduler.py     # 采集调度器
│   │   ├── analyzer/            # 行为分析模块
│   │   │   ├── __init__.py
│   │   │   ├── features.py      # 特征提取
│   │   │   └── patterns.py      # 模式识别
│   │   ├── llm/                 # LLM集成（Phase 2）
│   │   │   ├── __init__.py
│   │   │   └── attribution.py
│   │   ├── intervention/        # 干预策略（Phase 2）
│   │   │   ├── __init__.py
│   │   │   └── strategies.py
│   │   ├── models/              # 数据模型
│   │   │   ├── __init__.py
│   │   │   └── database.py
│   │   └── api/                 # API层
│   │       ├── __init__.py
│   │       ├── routes.py
│   │       └── websocket.py
│   ├── tests/
│   │   └── __init__.py
│   └── requirements.txt
├── frontend/                    # React 前端
│   ├── src/
│   │   ├── App.tsx
│   │   ├── main.tsx
│   │   ├── components/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── StatusCard.tsx
│   │   │   ├── AppUsageChart.tsx
│   │   │   ├── FocusTrendChart.tsx
│   │   │   └── SettingsPanel.tsx
│   │   ├── hooks/
│   │   │   ├── useApi.ts
│   │   │   └── useWebSocket.ts
│   │   └── types/
│   │       └── index.ts
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── data/                        # SQLite 数据库文件（gitignore）
└── docs/                        # 项目文档
    ├── design-spec.md
    ├── implementation-plan.md
    └── development-guide.md
```

## 五、数据库初始化

数据库文件会在首次启动后端时自动创建在 `backend/data/mindflow.db`。
无需手动初始化。

## 六、验证安装

```bash
# 后端
conda activate mindflow
cd mindflow-app/backend
python -c "from mindflow.models.database import init_db; init_db(); print('DB OK')"

# 前端
cd mindflow-app/frontend
npm run build  # 确认构建成功
```

## 七、常用命令

| 操作 | 命令 |
|------|------|
| 激活环境 | `conda activate mindflow` |
| 安装新依赖 | `pip install <package>` 然后 `pip freeze > requirements.txt` |
| 运行测试 | `pytest backend/tests/ -v` |
| 代码格式化 | `ruff check . && ruff format .` |
| 前端类型检查 | `cd frontend && npx tsc --noEmit` |
