# MindFlow 后端架构优化评估报告

> **日期**: 2026-08-14 · **对象**: backend-next/（FastAPI + LangGraph + SQLite + ML）
> **性质**: 架构咨询评估（非改动计划）——基于当前代码实际状态，评估 LangGraph/ML/数据采集/存储等方向的优化价值与优先级
> **数据基线**: activity_events 24,454 行 · feature_windows 5,352 行 · focus_sessions 327 行 · 单用户应用

---

## 0. 总体判断

当前架构是**单用户本地应用**的合理形态：SQLite WAL + LangGraph 编排 + 弱监督 ML + 三层 LLM 降级，规模下没有明显错误。以下建议按 **价值/成本** 排序，分四档：

| 档位 | 含义 | 条目数 |
|------|------|--------|
| 🟢 **立即值得做**（低风险高收益） | 本周可落地，直接改善体验/稳定性 | 6 |
| 🟡 **规划后值得做**（中等投入） | 1-2 周内，需配套测试 | 7 |
| 🟠 **条件性值得**（看产品方向） | 依赖产品定位，不急于动手 | 6 |
| ⚫ **不建议**（当前阶段） | 过度设计，成本大于收益 | 4 |

---

## 1. LangGraph 设计

### 现状
- 三个图：**AnalysisGraph**（日常分析组合根，实现 AnalysisWorkflowPort）、**PanelGraph**（多专家审议子图：Analyst → 3×并行 Attribution → 冲突检测 → Moderator → Critic）、**ChatGraph**（对话生命周期）。
- 图状态 TypedDict 可检查点化（runtime 走 ContextVar 不入 state——v2 cutover 后的正确做法）。
- **checkpointing_enabled 默认 False**；AnalysisGraph 用 `graph.compile()` 无 checkpointer（analysis_graph.py:1261）；PanelGraph 仅在 `human_review_enabled` 时用 MemorySaver（panel_graph.py:961）。
- 幂等/预算机制完善：cache + budget reservation + workflow_runs 状态机。

### 🟢 1.1 真正启用持久化 Checkpointer（收益最高的单点优化）

**现状**：checkpointing 是配置项但默认关闭，且 AnalysisGraph 编译时根本没传 checkpointer——不是"可选"而是"未接线"。崩溃恢复目前靠 cache 幂等（已够用），但代价是：
- **LLM 调用无法跨崩溃续跑**：panel 跑到第 8 次 LLM 调用时进程崩溃，重启后从零重跑全部 12 次调用（付费 API 成本翻倍）；
- **human_review 流程（`human_review_enabled`）依赖 MemorySaver**：内存态，服务重启即丢失待审批的 panel。

**建议**：`ApplicationCheckpointer`（SQLite 版）已存在且适配器齐全，只差接线：
1. `_build_compiled_graph()` 改为 `graph.compile(checkpointer=app_checkpointer.saver)`；
2. `ainvoke` 传稳定 `thread_id`（`panel_{user_id}_{date}` 已有）；
3. 保持 `checkpointing_enabled=False` 默认（避免行为变化），但让 `human_review_enabled=True` 时自动启用 SQLite checkpointer；
4. 测试：一条 `checkpointing_enabled=True` 的冒烟测试（上一轮已修好 state 序列化，frozenset→tuple，此路已通）。

**收益**：LLM 成本减半（崩溃续跑）、人工审批可持久化、为未来"中断/恢复"交互铺路。**成本**：约 1 天。

### 🟡 1.2 图拓扑数据驱动 + 可观测性增强

**现状**：图拓扑是硬编码的 `add_edge/add_conditional_edges` 调用（analysis_graph.py:1154-1260 约 100 行）。改动节点流（如调整 fallback 顺序）需要读代码理解拓扑；`workflow_node_events` 已记录节点执行，但无图级可视化。

**建议**：
- 用 `graph.get_graph().draw_mermaid_png()` 或 langgraph 的 mermaid 导出，把图拓扑渲染到 /diagnostics 页（纯展示，零侵入）；
- 将拓扑描述抽成声明式 dict（nodes/edges/router），`_build_compiled_graph` 按数据构建——让"图结构"可读、可测试、可未来可视化。

### 🟡 1.3 Panel 并行归因的细粒度并发控制

**现状**：3 路 attribution 用 `Send` 并行（正确做法）。但预算计数 `_MAX_CALLS=12` 是模块级常量，所有 panel 共享上限——不区分"首次尝试"与"重试轮"；`rebuttal` 循环次数（`moderator_redo_count`）由状态字段追踪但无上限校验之外的分级预算。

**建议**：预算按阶段分配（analyst=1, attribution=3, moderator=1+rebuttal_n, critic=1），把 `_MAX_CALLS` 换成每阶段配额表，并让 PanelVerdict 暴露 `call_count` 分布——用户能看到"这次分析用了多少次 LLM 调用"，透明化成本。

### 🟠 1.4 Human-in-the-loop 审批流（产品方向决定）

**现状**：`human_review_enabled` 存在但默认关，PanelGraph 有 interrupt 概念但未接入实际 UI。

**评估**：如果产品愿景是"AI 建议 + 用户确认"（如干预前预览），这值得做——检查点续跑已铺路，前端只需一个审批卡片。如果定位是"全自动陪伴"，则优先级低。

---

## 2. ML 设计

### 现状
- **特征**：V2 24 维 5 分钟窗口（schema v3），隐私友好（无 PII）。
- **模型**：EnsembleClassifier（RF + XGB 软投票）+ BehaviorClustering + BehaviorHMM（Markov 链 fallback）。
- **训练**：弱监督——显式反馈优先 + 时间重叠匹配；7 项质量门（GroupKFold 按日期折叠）；shadow/ready 两态部署；TrainingJobService 用 `asyncio.to_thread` 跑离线训练（正确）。
- **推理**：FocusPredictionService 批推理（已修好移出事件循环）。
- **现状数据**：5,352 个特征窗口 / 质量门因"反馈日 < 7"无法通过（训练报告显示 minimum_days=false）→ 模型长期停留在 shadow 模式。

### 🟢 2.1 冷启动：让模型在"数据不足"时也有价值

**核心痛点**：质量门要求 ≥7 反馈日才能激活，但用户前两周基本用不上 ML（一直 shadow）。当前"数据不足→全规则引擎"是**二元跳变**，体验断层明显。

**建议**：
1. **渐进式激活**：质量门从"全过才 ready"改为"分档部署"——minimum_days>=3 且 balanced_accuracy 达标 → ready_low_confidence 档（模型可用但预测标注 low_confidence，UI 显示"早期模式"）；
2. **隐式反馈补充**：当前只依赖显式反馈（用户手动标注）。可从行为数据派生弱标签（如：单一 app 驻留 >45 分钟 → focus；5 分钟内 >8 次切换 → distract）扩充训练集，加速跨过 7 日门槛；
3. **跨用户先验（可选）**：单用户冷启动数据太少，可用"学生原型"（train/user_profiles.py 已有 8 个原型）生成预训练模型作为**先验**，再在线微调——本质是迁移学习，1 天工作量。

### 🟡 2.2 在线学习：从"批训练"到"增量更新"

**现状**：训练是纯批量（需手动/API 触发，`training_jobs`），且新反馈不会自动进入模型。用户标注了 20 条反馈后，模型不会自动变好。

**建议**：
- **Welford 基线已在线增量**（设计正确）——把它扩展到分类器：每日 scheduler 任务检查"新增反馈 ≥ N 条 → 自动触发一次轻量增量训练（用最新 14 天窗口）"，成功后静默切换 shadow；
- 配合 2.1 的分档部署，实现"模型自动进步"的体验，无需用户进模型中心点训练。

### 🟡 2.3 HMM 的实用化

**现状**：BehaviorHMM 训练了状态转移矩阵，但**没有消费端**——推理、干预、报告都不使用 HMM 输出（grep 未见 hmm.predict 或转移矩阵的运行时使用）。它是"训练了但没用"的模型。

**建议**：两个方向选一：
a) **用起来**：HMM 转移矩阵 → 预测"接下来 30 分钟分心概率"（P(分心态 | 当前态)），作为 focus_prediction 的时间维度补充（现在是纯特征窗口打分，无时序）；
b) **砍掉**：如果 3 个月内用不上，从训练管线移除 HMM 减少训练时间/复杂度。

### 🟠 2.4 特征工程：纳入"任务上下文"

**现状**：24 维特征全是行为统计（切换数、空闲比、输入率…）+ 时间编码（hour/weekday sin/cos），**没有任务类型**（用户正在 coding/writing/meeting？）。`task_type_code` 是窗口级默认值而非从应用分类推断。

**评估**：`app_classification`（用户自建规则分类应用）已存在，把"当前任务类型"作为特征注入窗口，可显著提升"写文档时的高切换"与"刷网页时的高切换"的区分度（前者是正常、后者是拖延）。这是 ML 精度提升的最大单项杠杆，但需要特征 schema 升级（v3→v4）+ 窗口回填，属于中期投入。

### 🟠 2.5 可解释性 → 用户信任

**现状**：`train/explain.py` 有 SHAP 解释器，但仅在训练时生成 feature_importance；前端 ModelCenter 显示"特征重要性"列表。用户看到的是数字，不是"为什么现在提醒我"。

**评估**：干预/报告里加一句**自然语言解释**（"你今天在 14:00-16:00 切换了 47 次，远超你基线 12 次/小时"）——数据都已存在（baseline deviation 已在 EvidenceBundle），只是没渲染到用户可见文案。**这是提升信任感性价比最高的改动**，建议并入干预文案生成（LLM 已有平台约束 prompt，加一句"引用基线对比"即可）。

---

## 3. 数据采集

### 现状
- 5 秒采集 tick（collect_interval_s=5），10 秒心跳合并（pulsetime_s=10）——窗口快照 + 相邻合并，表膨胀可控。
- 平台采集器：Win32（psutil+pywin32）/ macOS / X11 / Wayland fallback，输入遥测（按键/鼠标/滚动）独立进程。
- 浏览器扩展心跳（免认证 + token），telemetry rollup 15 分钟回滚 2 小时窗口（v3 特征）。
- 合并竞态已修复（条件 UPDATE）。

### 🟢 3.1 采集器自愈与状态可见性（体验直接相关）

**现状**：采集器失败时 `collector_recovery.py` 有恢复逻辑，但**用户无感知**——仪表盘只显示"运行中/已停止"，不知道"采集器 3 小时前崩溃过并自动重启了 2 次"。

**建议**：`collector_intervals` 表（已存在，记录 open/close/reason/failure/sleep）已能支撑——在 /health 暴露 collector_health（uptime_s、last_failure_at、recovery_count），仪表盘"采集器状态"卡片显示：正常 / 已恢复(最近失败时间) / 采集中断。用户能感知采集器是否可靠，避免"今天怎么没数据"的困惑。

### 🟡 3.2 采集频率自适应（省电 + 精度平衡）

**现状**：固定 5 秒 tick。活跃时 5 秒足够，但**空闲时也 5 秒 tick**——浪费（CPU/电量）且生成大量 idle 事件。

**建议**：双档 tick——检测到系统空闲（输入遥测无活动 >60s）时降到 30s，恢复输入后回到 5s。`input_telemetry_service` 已有输入事件流，实现简单；收益：笔记本用户电量 + 事件量减少 60%+。

### 🟡 3.3 采集数据完整性校验（防止静默缺口）

**现状**：采集 tick 失败只记日志；用户看到的特征窗口如果因为采集缺口而稀疏，会被 `coverage_ratio` 标 stale——但**没人解释"为什么缺"**。

**建议**：每 15 分钟 rollup 时，对比"期望窗口数（时间跨度/5min）vs 实际窗口数"，缺口 >20% 时写一条 `collector_intervals` 的 failure 记录（reason=coverage_gap），并在 3.1 的采集器健康卡片显示。

### 🟠 3.4 浏览器扩展的采集粒度

**现状**：浏览器只采集 domain 级心跳（`browser_segments`：domain、audible、时长）。无法区分"在 GitHub 写代码"vs"在 GitHub 刷 issue"。

**评估**：URL path 级采集能显著提升特征质量（尤其 task_type 推断），但隐私代价大（违反 ADR-003 无 PII 原则）。**不建议默认开启**；若产品需要，做成显式 opt-in 配置。

---

## 4. 存储方式

### 现状
- SQLite WAL + busy_timeout + journal_size_limit（配置正确）；10 处 JSON 列（特征/分析/偏好/上下文）。
- 时间戳全部 ISO8601 TEXT（统一 UTC，可比较排序，代价是范围查询不走索引优化——但有针对性索引补偿）。
- 保留策略：事件 30 天、workflow 30 天、分析/聊天永久。
- 单用户（user_id 大量硬编码 1）。

### 🟢 4.1 特征窗口 JSON 列 → 独立特征表（ML 查询性能）

**现状**：`behavior_feature_windows.features_json` 存整个 24 维向量 JSON，训练时 `json.loads` 全部行。5,352 行时没问题，但**一年后 ~10 万行**，每次训练/回填都要全量 JSON 解析（数秒→数十秒）。

**建议**：加一张 `feature_window_values(user_id, window_id, feature_name, value)` 或把 24 维拆成 24 个 REAL 列（SQLite 支持，`ALTER TABLE ADD COLUMN` 24 次或重建表）。前者对 ML 友好（pivot 查询），后者对简单。**建议 24 列方案**——SQL 直接 `SELECT` 向量，训练加载不再 JSON 解析，且可对单特征建索引。

### 🟡 4.2 活动事件表的分区/归档（长期膨胀控制）

**现状**：`activity_events` 30 天保留后删除（`cleanup_old_telemetry`），单表无分区。日增量约 2,500 事件（5 秒 tick × 活跃 3.5h），30 天峰值 ~7.5 万行——**当前规模完全没问题**。

**评估**：SQLite 单表到 100 万行仍可用（有 (user_id, timestamp) 索引）。**当前不需要分区**；建议在保留策略上做文章：30 天事件 + 无限期保留"汇总特征窗口"已是最优组合（原始事件删、特征保留，隐私 + 分析兼得）。这条是**未来观察项**，不投入。

### 🟡 4.3 迁移策略：schema.py 双源一致性

**现状**：`infrastructure/schema.py` 是"单一事实源"（注释声明），但 `activity_events` 表仍定义在 `repositories/activity.py`（其模块级 metadata 与 schema.py 的 metadata 分离）——schema.py 文档自己标注"activity_events 不在本模块，后续移入"。

**建议**：完成合并（低风险，纯搬移 + 更新 import），让 `alembic autogenerate` 能看到全部表，消除"表在哪个 metadata"的二义性。上一轮已让 conftest 建全表，此项收尾即可。

### 🟠 4.4 多用户支持（架构分叉点）

**现状**：user_id 字段遍布但逻辑上单用户（前端多处硬编码 user_id:1，路由直接 user_id=1）。

**评估**：如果产品定位"个人本地工具"，**不要做多用户**——SQLite 单文件 + 本机认证已是正确形态，多用户会引入账号体系、数据隔离、并发写放大，成本极高。如果未来要"家庭共享/多设备同步"，那是另一套产品（需服务端），不是当前架构的演进。**明确这条边界能避免 2 周的无效设计。**

---

## 5. 其它（横切关注点）

### 🟢 5.1 LLM 成本透明化

**现状**：`llm_cost_usd` 字段存在于分析表，但无汇总展示；workflow_runs 有 call_count。用户不知道"AI 功能每月消耗多少 API 费用"。

**建议**：/settings 加"AI 用量"卡片（本月分析次数、LLM 调用数、估算成本）——对 API key 用户是刚需（避免月底账单惊吓），对纯规则引擎用户显示"本地模式，零成本"（强化本地优先卖点）。

### 🟢 5.2 首次启动引导（体验断层最大处）

**现状**：新用户启动后 dashboard 全是空态/"--"（无数据），要等 1-2 天才有报告、7 天才有人像。没有"预期管理"。

**建议**：空态文案从"暂无数据"升级为**带时间线的引导**："已采集 2 小时 · 专注报告明日生成 · 个人画像需 7 天数据 · 模型中心可查看进度"。数据都已存在（采集时长、窗口数），只需渲染。这是**提升首日体验性价比最高**的前端改动。

### 🟡 5.3 报告 AI 化（报告中心的价值提升）

**现状**：`pattern_summary` 是规则模板拼接（report_service.py），无 AI 润色。UX 审查文档 P5 已标注为可选增强。

**建议**：日报/周报加"AI 解读"段（复用已有 LLM 降级链，prompt 注入今日统计 + 基线对比，输出 3 句话洞察）。成本低（复用 panel 的 gateway），价值高（报告从"数字表格"变"读得懂的总结"）。与 2.5 的解释性建议合并实施。

### 🟠 5.4 多设备/导出生态

**现状**：已有 `export` 端点（CSV/JSON），但无导入。

**评估**：导入（换电脑迁移）比导出重要——用户重装系统后数据丢失会流失。**建议做"一键备份/恢复"**（打包 mindflow.db + models + token，做成 .mindflowbundle）：比导入单个 JSON 实用得多。

### ⚫ 5.5 不建议：微服务化 / 消息队列 / 独立分析引擎

单用户本地应用引入 Kafka/Redis/Celery 是典型过度设计。当前 asyncio + SQLite + 后台任务已覆盖全部需求；未来唯一合理演进是"可选云同步"（见 4.4），那也是独立产品线。

### ⚫ 5.6 不建议：换 PostgreSQL / 上 ORM 高级特性

SQLite WAL 在单用户规模（<100 万行）下性能、备份（VACUUM INTO）、零运维优势不可替代。迁移 PG 的收益（并发写）在此场景不存在。

---

## 6. 建议路线图

### 本周（🟢 档，总 ~3-4 天）
1. **1.1 持久化 Checkpointer 接线**（LLM 成本减半）
2. **3.1 采集器健康可见性** + 3.3 覆盖缺口检测（采集可靠性透明）
3. **5.2 首启引导 + 5.1 AI 用量卡片**（首日体验 + 成本透明）
4. **4.3 schema.py 合并收尾**（低风险清理）

### 本月（🟡 档，总 ~2 周）
5. **2.1 冷启动渐进激活 + 隐式反馈**（ML 提前可用）
6. **2.2 自动增量训练**（模型自动进步）
7. **1.2 图拓扑可视化 + 1.3 阶段预算**（LangGraph 可观测）
8. **3.2 采集频率自适应**（省电）
9. **4.1 特征窗口拆列**（ML 查询性能，配合 v4 schema 一起做）
10. **2.5 + 5.3 解释性文案与 AI 报告**（用户信任）

### 看产品方向（🟠 档）
- 1.4 人工审批流 · 2.4 任务类型特征 · 2.3 HMM 实用化/裁剪 · 3.4 浏览器 URL 粒度 · 4.4 多用户边界 · 5.4 一键备份恢复

### 不做（⚫ 档）
- 微服务化、消息队列、PostgreSQL 迁移

---

## 7. 一句话总结

**当前架构方向正确，最大的三个杠杆是：(1) 真正启用 LangGraph Checkpointer 省 LLM 成本；(2) 让 ML 冷启动渐进生效（而不是两周全黑）；(3) 把已有的丰富数据（采集健康、基线对比、用量）翻译成用户能感知的体验（首启引导、AI 解释、健康卡片）。这三项都改动小、价值直接，且不会破坏现有架构。**
