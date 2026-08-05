# MindFlow 后端技术解析报告 — 总览章

> 目标读者：**从未写过项目的人**。读完本报告 + 各专题章，应能理解 MindFlow 后端"为什么这么设计"并复刻它。
> 本总览章由团队负责人撰写，作为各专题章的导航与收束。各专题章文件名：`01-storage.md`（数据存储）、`02-collection.md`（数据采集）、`03-training-data.md`（训练数据）、`04-training-methods.md`（训练方法）、`05-langgraph.md`（LangGraph 图）、`06-retry.md`（重试与降级）、`07-prompts-theories.md`（提示词与理论）、`08-intervention.md`（实时提醒与调度）。

---

## 0.1 这个项目到底做了什么（一句话）

MindFlow 是一个**本地优先的智能专注助手**：它持续观察你电脑上"在用哪些软件、切了多少次窗口、键盘鼠标忙不忙"，用统计学 + 机器学习判断"你现在是不是在拖延/分心"，然后用多专家 LLM 会诊给出**为什么**，最后在合适的时机弹出提醒，帮你把注意力拉回正事。

它从头到尾**不把数据上传云端**（隐私第一），LLM 分析也分三档：能用云端 DeepSeek 就用，不行退回本地 Ollama，再不行退回纯规则引擎——所以**永远可用**。

## 0.2 一图看懂整体架构

```
┌──────────────┐   HTTP/WS   ┌─────────────────────────────────────────────┐
│  前端 React   │ ◄────────► │  后端 FastAPI (:8765)                        │
│  (浏览器 UI)   │             │                                             │
└──────────────┘             │  API 层 (routes + middleware + auth)        │
                             │      │                                      │
                             │      ▼                                      │
                             │  Services 层 (业务编排)                      │
                             │   分析 / 报告 / 干预 / 聊天 / 调度 / 训练      │
                             │      │                    │                 │
                             │      ▼                    ▼                 │
                             │  agents/ + graph/       ML train/          │
                             │  (LLM 专家会诊,          (聚类/分类/集成/HMM) │
                             │   LangGraph)              │                 │
                             │      │                    │                 │
                             │      ▼                    ▼                 │
                             │  Infrastructure 层: SQLite(WAL) + 采集器     │
                             └─────────────────────────────────────────────┘
                                       ▲
                                       │ 5s / 30s / 浏览器
                          ┌────────────┴────────────┐
                          │  本机采集器 (跨平台)        │
                          │  窗口活动/键鼠/浏览器       │
                          └─────────────────────────┘
```

**架构主法则**（方向不可逆，只能一层依赖下面一层）：
`domain（纯模型）→ infrastructure（数据库/采集/LLM 客户端）→ services（业务编排）→ api（对外接口） / agents+graph（LLM 编排）`

这条单向分层是复刻时最重要的纪律：domain 层**绝不 import** 框架，所有副作用（连数据库、发 HTTP）都被压到 infrastructure，services 只做"安排谁干什么"。

## 0.3 技术栈全景（复刻需要装什么）

| 领域 | 技术 | 为什么选它 |
|------|------|-----------|
| 语言/运行时 | Python 3.11+（conda env `mindflow` 或 uv venv） | 数据科学 + Web 一体，生态最全 |
| Web 框架 | FastAPI + Uvicorn（端口 8765） | 异步、自动 OpenAPI 文档、类型校验 |
| 数据库 | SQLite（WAL 模式）+ SQLAlchemy async + aiosqlite | 本地优先、零运维、单文件 |
| 迁移 | Alembic | SQLite 有限的 ALTER TABLE 能力需要谨慎迁移 |
| LLM SDK | LangChain/LangGraph + langchain-deepseek | 声明式图编排 + 三档降级 |
| ML | scikit-learn（聚类/分类/集成）+ hmmlearn（HMM） | 经典可解释模型，本地可跑，无需 GPU |
| 任务调度 | APScheduler | 每日分析、定时作业、干预窗口 |
| 观测 | 本地 OpenTelemetry（SQLite exporter） | 无外部上报，隐私合规 |
| 采集 | 各平台原生 API（win32 / X11 / macOS Quartz） | 拿前台窗口、输入事件 |
| 打包 | PyInstaller（mindflow.spec）、浏览器扩展 | 桌面分发 |

## 0.4 关键设计决策（复刻时最容易踩坑的点）

1. **所有数据本地**：SQLite WAL 单文件 + `{data_dir}` 目录（Windows 默认在用户数据目录，见 `01-storage.md`）。WAL 允许"多读一写"不互相阻塞。
2. **无全局单例**：FastAPI 用 `create_app(settings)` 工厂 + 依赖注入，共享状态挂在 `app.state`。这是为了可测试（每次测试造一个干净的 app）。
3. **LLM 输出当不可信数据**：Pydantic 严格模式 `extra="forbid"` + 禁用词校验 + 证据引用代码级校验 + 独立危机检测器。LLM 说的话一个字都不能直接信。
4. **三级降级链**：DeepSeek → Ollama → RuleEngine，见 `06-retry.md`。这是"永远可用"的保证。
5. **专家会诊用 LangGraph**：5+1 专家组成图，12 次 LLM 调用硬预算封顶，见 `05-langgraph.md`。
6. **特征窗口 v3**：原始事件先 rollup 成 5 分钟特征窗口再训练，切换计数用"驻留 10 秒 + 忽略瞬时进程"防抖动，见 `02-collection.md`、`03-training-data.md`。
7. **质量门**：训练前 7 道就绪度检查、训练后 `calibration_better_than_rule` 等评估门，防止拿不够格的模型上线，见 `03-training-data.md`、`04-training-methods.md`。

## 0.5 一张表看懂数据流（从采集到提醒）

| 阶段 | 做什么 | 在哪个专题章 |
|------|--------|------------|
| ① 采集 | 窗口活动(5s) / 键鼠(30s) / 浏览器(约10s) 写原始事件表 | 02-collection |
| ② 特征化 | 原始事件 → 5 分钟特征窗口 (schema v3) | 02-collection / 03-training-data |
| ③ 基线 | Welford 在线均值/方差维护用户"正常"行为基线 | 08-intervention |
| ④ 偏离检测 | 当前行为 vs 基线 → 偏差分数（Z 值等） | 08-intervention |
| ⑤ 干预判定 | 偏差 + 拖延类型 → 是否提醒、提醒多强 | 08-intervention |
| ⑥ 每日分析 | 聚合一天数据 → 专家会诊 → 归因报告 + 建议 | 05-langgraph / 07-prompts |
| ⑦ 反馈闭环 | 用户对提醒/报告的反馈 → 节流调节 + 训练标签 | 08-intervention / 03-training-data |
| ⑧ 训练 | 积累的反馈 + 特征窗口 → 训练/更新模型 | 03 / 04-training-methods |

## 0.6 给初学者的"复刻路线图"（10 步）

1. **搭骨架**：FastAPI `create_app` 工厂 + 配置（Pydantic BaseSettings + `MINDFLOW_` 前缀环境变量）。
2. **建库**：SQLite WAL + SQLAlchemy async + Alembic 初始化，按 `01-storage.md` 建表。
3. **采集**：先写一个"每 5 秒取前台窗口"的采集器，落表。
4. **特征化**：把原始事件 rollup 成特征窗口，写 `count_confirmed_switches`。
5. **基线+偏离**：Welford 在线统计算基线，Z 分算偏离（`08-intervention.md` 有伪代码）。
6. **干预**：触发条件 + 节流 + 弹窗。
7. **LLM 接入**：先实现三级降级链里最容易的 L3 规则引擎，再按 `07-prompts-theories.md` 接一个专家。
8. **LangGraph**：把"一个专家"扩成"多专家图"，套 12 预算 + 校验节点（`05-langgraph.md`）。
9. **训练**：跑合成数据 → 真实数据 → 质量门（`03`/`04` 两章）。
10. **反馈闭环 + 调度**：APScheduler 每日分析 + 疗效回写。

> 每章末尾的"可复刻性"小节都给了最小代码骨架，按路线图拼接即可。

## 0.7 验证：怎么确认我复刻对了

- 后端测试：`uv run python -m pytest tests/ -q`（基线 1956 passed, 12 skipped）。
- 评估：`uv run python -m mindflow.eval --mode both`（30 场景规则引擎对照，无需 API key）。
- 训练：`uv run python -m mindflow.train --source synthetic_v2`。
- 健康检查：`GET /api/v1/health/ready`。
