# MindFlow 项目实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零搭建MindFlow智能专注助手完整系统，分五阶段交付。

**Architecture:** Python FastAPI后端 + React TypeScript前端，SQLite本地存储，模块化设计支持独立开发测试。

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.0, SQLite, React 19, TypeScript, Vite, Ant Design, recharts

---

## Phase 0: 基础设施与Demo（2026年5月 - 当前）

### Task 0.1: 项目脚手架搭建

**Files:**
- Create: `mindflow-app/backend/requirements.txt`
- Create: `mindflow-app/backend/mindflow/__init__.py`
- Create: `mindflow-app/backend/mindflow/config.py`
- Create: `mindflow-app/backend/mindflow/main.py`
- Create: `mindflow-app/frontend/` (Vite project)

- [x] 创建conda环境 `mindflow` (Python 3.11)
- [x] 创建项目目录结构
- [ ] 安装后端依赖
- [ ] 初始化前端项目

### Task 0.2: 数据模型层

**Files:**
- Create: `backend/mindflow/models/__init__.py`
- Create: `backend/mindflow/models/database.py`
- Create: `backend/mindflow/models/schemas.py`

- [ ] 实现SQLAlchemy引擎与会话管理
- [ ] 定义User, ActivityLog, FocusSession, DailyReport模型
- [ ] 实现init_db()自动建表
- [ ] 验证: `python -c "from mindflow.models.database import init_db; init_db()"`

### Task 0.3: 数据采集模块

**Files:**
- Create: `backend/mindflow/collector/__init__.py`
- Create: `backend/mindflow/collector/tracker.py`
- Create: `backend/mindflow/collector/scheduler.py`

- [ ] 实现Windows活动窗口监控 (win32gui + psutil)
- [ ] 实现用户空闲检测
- [ ] 实现APScheduler定时采集调度
- [ ] 验证: 启动采集器，确认ActivityLog表有数据写入

### Task 0.4: 行为分析模块

**Files:**
- Create: `backend/mindflow/analyzer/__init__.py`
- Create: `backend/mindflow/analyzer/features.py`
- Create: `backend/mindflow/analyzer/patterns.py`

- [ ] 实现专注分数计算 (0-100)
- [ ] 实现应用使用排名统计
- [ ] 实现窗口切换频率计算
- [ ] 实现专注会话识别
- [ ] 实现日报生成

### Task 0.5: API层

**Files:**
- Create: `backend/mindflow/api/__init__.py`
- Create: `backend/mindflow/api/routes.py`
- Create: `backend/mindflow/api/websocket.py`

- [ ] 实现所有P0/P1 REST端点
- [ ] 实现WebSocket实时推送
- [ ] 实现标准JSON响应格式
- [ ] 实现CORS中间件
- [ ] 验证: 访问 http://localhost:8765/docs 确认Swagger可用

### Task 0.6: 前端Dashboard

**Files:**
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/hooks/useApi.ts`
- Create: `frontend/src/hooks/useWebSocket.ts`
- Create: `frontend/src/components/Dashboard.tsx`
- Create: `frontend/src/components/StatusCard.tsx`
- Create: `frontend/src/components/AppUsageChart.tsx`
- Create: `frontend/src/components/FocusTrendChart.tsx`
- Create: `frontend/src/components/SettingsPanel.tsx`

- [ ] 实现Dashboard主页（专注得分 + 应用饼图 + 趋势图）
- [ ] 实现实时状态卡片（WebSocket更新）
- [ ] 实现设置页面（采集开关 + 偏好设置）
- [ ] 验证: `npm run build` 构建成功

### Task 0.7: 端到端验证

- [ ] 启动后端 (uvicorn)
- [ ] 启动前端 (npm run dev)
- [ ] 启动数据采集
- [ ] 确认Dashboard实时显示当前活动窗口
- [ ] 确认应用使用统计正确
- [ ] 确认专注趋势图有数据

---

## Phase 1: 数据采集增强（2026年6-7月）

### Task 1.1: 采集精度优化

- [ ] 实现窗口切换事件驱动采集（替代纯定时轮询）
- [ ] 实现浏览器URL级别追踪（Chrome/Firefox扩展或COM接口）
- [ ] 实现应用分类标签（生产力/娱乐/社交/其他）

### Task 1.2: 隐私与存储优化

- [ ] 实现数据脱敏（敏感窗口标题过滤）
- [ ] 实现数据保留策略（自动清理N天前数据）
- [ ] 实现数据导出功能（CSV/JSON）

### Task 1.3: 系统托盘

- [ ] 实现Windows系统托盘图标
- [ ] 右键菜单：启动/停止采集、打开Dashboard、退出
- [ ] 托盘图标状态指示（运行中/暂停/错误）

---

## Phase 2: 行为建模（2026年8-10月）

### Task 2.1: 特征工程深化

- [ ] 时序特征提取（滑动窗口统计）
- [ ] 行为序列模式挖掘（频繁模式/序列模式）
- [ ] 用户画像构建（工作习惯、专注高峰时段、分心触发因素）

### Task 2.2: 模式识别算法

- [ ] 实现时序聚类（识别不同行为模式类别）
- [ ] 实现隐马尔可夫模型（行为状态推断）
- [ ] 实现分心预测模型（基于历史数据预测分心概率）
- [ ] 算法效果评估（精确率/召回率）

### Task 2.3: 可视化增强

- [ ] 行为模式热力图（一天中各时段行为分布）
- [ ] 专注/分心时段时间轴
- [ ] 周度对比报告

---

## Phase 3: LLM归因分析（2026年11月-2027年1月）

### Task 3.1: LLM基础设施

- [ ] LangChain集成配置
- [ ] 国产大模型API接入（通义千问/文心一言/DeepSeek）
- [ ] Prompt工程（CBT框架映射、归因模板）
- [ ] Token用量与成本控制

### Task 3.2: 归因引擎

- [ ] 行为数据→自然语言摘要转换
- [ ] CBT框架映射：识别认知扭曲与行为模式
- [ ] 拖延类型分类：任务畏惧型、信息过载型、决策困难型、完美主义型
- [ ] 归因结果置信度评估

### Task 3.3: 对话式交互

- [ ] 实现反思对话接口（用户可追问归因原因）
- [ ] 实现每日反思提示生成
- [ ] 实现行为改善建议生成

---

## Phase 4: 智能干预（2027年2-3月）

### Task 4.1: 干预策略引擎

- [ ] 任务拆解策略（LLM将大任务分解为微任务）
- [ ] 环境优化策略（分心检测→温和提醒）
- [ ] 智能排序策略（截止日期+精力状态→任务推荐）
- [ ] 干预强度分级（温和/标准/严格）

### Task 4.2: 干预执行与反馈

- [ ] 桌面通知推送（Windows Toast Notification）
- [ ] 干预效果追踪（干预前后行为变化对比）
- [ ] 策略自适应调整（A/B测试框架）

---

## Phase 5: 实验验证与交付（2027年4-5月）

### Task 5.1: 实验设计

- [ ] 对照组实验设计
- [ ] 招募被试（目标30+人）
- [ ] 数据采集与预处理

### Task 5.2: 效果评估

- [ ] 专注时长变化分析
- [ ] 拖延行为改善评估
- [ ] 用户满意度调查
- [ ] 统计显著性检验

### Task 5.3: 交付物

- [ ] 项目研究报告
- [ ] 开源代码整理与GitHub发布
- [ ] 答辩PPT与演示准备
- [ ] 用户手册

---

## 风险管理

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| LLM API费用超预算 | 中 | 高 | 优先使用免费/低价模型，本地缓存归因结果 |
| 数据采集精度不足 | 中 | 高 | Phase 1即实现事件驱动采集，持续优化 |
| 用户隐私顾虑 | 低 | 高 | 全本地处理，数据脱敏，透明化隐私策略 |
| 干预效果不显著 | 中 | 中 | 设计对照组实验，迭代优化策略 |
| 团队时间冲突 | 中 | 中 | 模块独立开发，明确接口边界 |

## 团队分工

| 成员 | 主要职责 | 次要职责 |
|------|----------|----------|
| 胡淙煜 | 后端架构、数据采集、LLM集成 | 项目协调、文档 |
| 张皓 | 前端Dashboard、数据可视化 | 数据采集辅助 |
| 杨智杰 | 前端组件开发、数据清洗 | API对接、测试 |
