# MindFlow 优化方案：聊天 Agent 会诊调用 + 实时分心提醒

> **日期**: 2026-07-27
> **审查流程**: Claude 提出方案 → Codex Round 1 深度审查（8 项发现）→ Claude 修订方案 → Codex Round 2 确认（全项通过，含语义修正）
> **状态**: 待用户审批后实施

---

## 目录

1. [优化需求一：聊天 Agent 调用每日专家会诊结论](#1-优化需求一)
2. [优化需求二：实时分心检测与即时提醒](#2-优化需求二)
3. [实施分期与工作量估算](#3-实施分期)
4. [Codex 审查记录摘要](#4-codex-审查记录)

---

## 1. 优化需求一：聊天 Agent 调用每日专家会诊结论

### 1.1 现状问题

| 现有工具 | 实际行为 | 问题 |
|----------|----------|------|
| `get_latest_analysis()` | 读 `procrastination_analyses` 表的 ML 分析 | 不是专家会诊结论 |
| `run_panel()` | 触发新的 6-12 次 LLM 调用 | 昂贵、每会话限 1 次 |

**核心缺口**：聊天 Agent 无法读取已存储的每日专家会诊结论（23:30 定时任务或手动触发的结果）。

### 1.2 数据层修复（Codex 发现 #1：行覆盖风险）

**问题**：Panel 和普通 ML 分析共享 `(user_id, date)` 唯一行。后续普通分析可覆盖 Panel 结果并清空 `panel_transcript_json`。

**方案**：增加两个区分维度：

```sql
ALTER TABLE procrastination_analyses ADD COLUMN analysis_kind TEXT;
ALTER TABLE procrastination_analyses ADD COLUMN source TEXT;

-- analysis_kind 枚举值
--   daily_panel      : 每日专家会诊（23:30 定时 / 手动触发）
--   daily_attribution: 每日 ML 归因分析（L2 单专家降级）
--   ml               : 纯 ML 分析
--   legacy_unknown   : 迁移前旧数据，无法识别

-- source 枚举值（与现有 PanelSource 对齐）
--   panel | single_expert | ollama | rule_engine

-- 复合唯一约束
CREATE UNIQUE INDEX idx_analysis_user_date_kind
  ON procrastination_analyses(user_id, date, analysis_kind);
```

**迁移策略**（Codex Round 2 修正）：
1. 列先设为 nullable
2. 从 `panel_transcript_json` / `source` 字段反向识别已有 panel 行，标记 `daily_panel`
3. 不可识别行标记 `legacy_unknown`
4. 重建 NOT NULL 约束（避免 SQLite 允许多个 NULL 破坏唯一性）

**降级语义**：
```
daily_panel + source=panel        → 完整专家会诊成功
daily_panel + source=single_expert → 专家团不可用，降级到单专家
daily_panel + source=rule_engine   → 全部 LLM 不可用，降级到规则引擎
```

> **关键**：Panel 降级结果也标记 `analysis_kind=daily_panel`，确保聊天 Agent 始终能读到"今日会诊"（即使降级）。

### 1.3 新增 LangChain 工具：`get_panel_verdict`

```python
@tool
async def get_panel_verdict(detail: str = "summary") -> str:
    """读取今日已存储的专家会诊结论（只读，不触发新 LLM 调用）。

    Args:
        detail: "summary" 返回裁决+理由+来源信息（默认）;
                "discussion" 额外返回专家发言记录。

    Returns:
        JSON 字符串，结构见下方。
    """
```

**返回结构**：

```json
{
  "status": "fresh",
  "requested_date": "2026-07-27",
  "resolved_date": "2026-07-27",
  "age_days": 0,
  "requested_kind": "daily_panel",
  "resolved_kind": "daily_panel",
  "source": "panel",
  "fallback_used": false,
  "degraded": false,
  "verdict": {
    "types": ["impulsivity"],
    "confidence": {"impulsivity": 0.82},
    "technique": "stimulus_control",
    "rationale": "你今天下午的窗口切换频率显著高于基线..."
  },
  "dissent_count": 0,
  "escalated": false
}
```

**`detail="discussion"` 时追加**：
```json
{
  "transcript": [
    {"role": "数据分析师", "content": "下午14-16点切换频率是基线的2.3倍...", "round": 0},
    {"role": "CBT归因专家", "content": "符合冲动性拖延模式，建议刺激控制...", "round": 1}
  ]
}
```

> **Codex 修正**：transcript 在 discussion 级别返回，不在 summary 中（summary 只含裁决+理由+来源）。

**降级/缺失处理**：

| 状态 | 条件 | Agent 行为 |
|------|------|------------|
| `fresh` | 今日会诊已存在，age_days=0 | 直接返回 |
| `stale` | 会诊存在但非今日（如昨天） | 返回旧数据 + 标注 age_days |
| `missing` | 无任何会诊记录 | 返回 status=missing，Agent 可追问是否要 run_panel |

**注册为 evidence 工具**（Codex 确认）：
```python
_EVIDENCE_TOOLS = frozenset({
    "query_evidence", "get_latest_analysis", "get_panel_verdict"
})
```
但仅当返回 `status != "missing"` 时 `evidence_cited` 才为 true。

### 1.4 Panel 访问策略（Codex 发现 #5 修正）

**核心原则**：
- `get_panel_verdict` — **永远只读**，不触发 LLM 调用
- `run_panel` — 显式触发昂贵操作，保持 1 次/会话上限
- 两者共享 `PanelService.get_or_run_daily_panel()`，但只有 `run_panel` 可请求执行

**每日声明机制**：
```
force=False → 先查存储结果 → 无结果则尝试 canonical daily claim → 都失败返回 missing
force=True  → 本次不实现（日后用独立 panel_runs 表，不覆盖调度器声明）
```

使用 `scheduled_job_runs` 仅用于 canonical daily panel（23:30 定时任务），不用于手动 rerun。

### 1.5 PanelService 持久化修正

**当前问题**：降级时 `LLMService` 直接写 `procrastination_analyses`，绕过了 `PanelService`，导致 `analysis_kind` 被设为 `daily_attribution` 而非 `daily_panel`。

**修正**：`PanelService` 拥有最终持久化权——即使是降级路径，也由 `PanelService` 统一写入，标记正确的 `analysis_kind=daily_panel` + `fallback_used=true`。

---

## 2. 优化需求二：实时分心检测与即时提醒

### 2.1 架构设计（Codex 发现 #2/#5 修正后）

```
┌─ CollectorService ──────────────────────────────────────────┐
│  persist_event() → enqueue put_nowait() → bounded Queue(200) │
│  Overflow: drop oldest, set coalesced needs_resync flag       │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─ DistractionMonitorService (managed asyncio.Task) ──────────┐
│  await queue.get() → 滑动窗口分析                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ 1. 预检: 用户是否在"专注意向"模式? → 否 → skip        │    │
│  │ 2. 规则评估:                                          │    │
│  │    a. 娱乐应用驻留 ≥60s                                │    │
│  │    b. 快速切换 ≥5次/120s                               │    │
│  │    c. 空闲后分心 (状态机)                               │    │
│  │ 3. 白名单检查 → 静默跳过                               │    │
│  │ 4. 用户分类覆盖 → 调整规则灵敏度                        │    │
│  │ 5. 连续正窗口 ≥2 → 触发告警                            │    │
│  │ 6. 滞后: 需 ≥3min 生产活动才复位                       │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─ 原子节流预约 (BEGIN IMMEDIATE 事务) ───────────────────────┐
│  生命周期: reserved → dispatching → submitted|failed|expired  │
│  合并去重: 同规则类型 15min 冷却                              │
│  硬上限: 3 条/小时（所有规则类型合计）                         │
└──────────────────┬──────────────────────────────────────────┘
                   ▼
┌─ InterventionService ───────────────────────────────────────┐
│  渲染模板 → 持久化 → 桌面通知 + WebSocket 推送               │
│  通知状态: {                                                 │
│    "desktop": "submitted|failed|unsupported|log_only",       │
│    "websocket": "sent|no_clients|failed"                     │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 运输层：`asyncio.Queue`（非 WebSocket）

**Codex 明确反对**用 WebSocket 做实时处理输入：
- WebSocket 当前是 server→client 出向通道
- 入向消息除 `ping` 外均被忽略
- `broadcast_activity_update()` 无生产调用者

**正确方案**：
```python
# CollectorService 中
async def _persist_and_publish(self, event):
    await self._repo.insert(event)
    try:
        self._monitor_queue.put_nowait(event)
    except asyncio.QueueFull:
        self._needs_resync = True  # 合并不重复设
        logger.warning("Monitor queue full, resync needed")

# DistractionMonitorService 中
async def _process_loop(self):
    while self._running:
        event = await self._queue.get()
        if self._needs_resync:
            await self._rebuild_window()
            self._needs_resync = False
        await self._evaluate(event)
```

**队列参数**：
- `maxsize=200`（5 秒采样 ≈ 17 分钟缓冲；Codex 认为 1000 过大，会导致延迟过高的提醒）
- 溢出时丢弃最旧事件 + 设 `needs_resync` 标志
- 忽略超过最大延迟（30s）的排队事件
- 采集器先于监控器停止；监控器排空后停止

### 2.3 检测规则详细设计（Codex 发现 #4 修正）

#### 2.3.1 前置条件：专注意向模式

**不**复用现有 `focus_sessions`（那是事后投影，不是实时状态）。

新增轻量状态 API：
```python
# POST /api/v1/focus/intent
# Body: {"action": "start" | "stop"}
# 设置一个 time-bounded 专注意向标记，默认 2h 后自动过期

# 备选：自动推断（非首选）
# 连续 5min 生产上下文 → 自动激活（但有假阳性风险）
```

#### 2.3.2 三条检测规则

| 规则 | 触发条件 | 防误报措施 |
|------|----------|------------|
| **娱乐应用驻留** | 用户分类为"娱乐"的应用驻留 ≥60s | 60s 驻留窗口；白名单排除（如午休时段允许） |
| **快速切换** | 120s 滑动窗口内切换 ≥5 次 | 需 ≥2 个连续正窗口；排除 IDE 内多文件切换（同进程） |
| **空闲后分心** | 状态机：idle(≥2min) → active → 60s 内打开娱乐应用 | 显式状态转换；区分"休息后回复消息"vs"刷视频" |

#### 2.3.3 滞后与冷却

```
触发条件: ≥2 个连续正窗口（每个窗口 ~30s）
复位条件: ≥3min 连续生产活动
规则冷却: 同规则类型 15min
硬上限:   3 条/小时（所有规则合计，由原子预约强制执行）
```

### 2.4 节流预约原子化（Codex 发现 #3 修正）

**当前问题**：`can_intervene()` 读和 `insert` 不在同一事务中，30min 定时、手动触发、实时检测可能同时通过检查。

**修正**：新增原子预约方法：
```python
async def reserve_intervention_slot(
    self, user_id: int, kind: str, session: AsyncSession
) -> str | None:
    """在同一个 BEGIN IMMEDIATE 事务中:
    1. 检查所有限制（每日上限、类型上限、冷却时间）
    2. 插入 reservation 行（status=reserved, expires_at=now+60s）
    3. 返回 reservation_id 或 None（被限流）
    """
```

**预约生命周期**：
```
reserved(60s TTL) → dispatching → submitted | partial | failed | expired
```

- 节流查询计数 `reserved` + `submitted`，排除 `failed` 和 `expired`
- 当前 2h 全局冷却与新的 15min 规则冷却/3-per-hour 上限共存时，**以原子预约层为准**，监控器只做 debounce/hysteresis

### 2.5 白名单与分类集成（Codex 发现 #6 修正）

**当前**：`UserAppClassifier` 已实现但未接入运行时。前端后端分类名不兼容。

**修正**：

优先级链：
```
白名单（仅抑制提醒，仍记录指标）
  → 用户分类（覆盖提醒规则中的分类判断）
  → 检测规则（阈值判断）
  → 原子节流预约
  → 干预分发
```

实施：
1. 统一前后端分类名 → 用后端 `app_classification_rules.category` 为准，前端同步
2. `DistractionMonitorService` 初始化时加载分类规则到本地缓存
3. 设置变更时通过事件总线/回调失效缓存（不每次查询 `get_all()`）

### 2.6 通知可观测性（Codex 发现 #8 修正）

**当前**：通知发送后忽略结果，macOS/Linux 只记日志。

**修正**：在 `intervention_logs` 中增加 `notification_status_json`：

```json
{
  "desktop": {"status": "submitted", "channel": "winrt"},
  "websocket": {"status": "sent", "client_count": 1}
}
```

状态值：
- `submitted` — 已提交给平台 API
- `failed` — 提交失败（含错误原因）
- `unsupported` — 平台不支持（macOS/Linux 当前）
- `log_only` — 仅记录日志
- `sent` / `no_clients` — WebSocket 特有

`dismissed` 保留在 `user_response` 中，不属于 `delivery_status`。

---

## 3. 实施分期与工作量估算

### Phase A: 数据层修复（1-2 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| A1 | Alembic 迁移：增加 `analysis_kind` + `source` 列，复合唯一索引 | 新建 migration |
| A2 | 回填已有数据（panel 行 → daily_panel，不可识别 → legacy_unknown） | migration 脚本 |
| A3 | 更新 `ProcrastinationAnalysisRepository`：增/改/查 接受 `analysis_kind` 参数 | `infrastructure/repositories/analysis.py` |
| A4 | PanelService 统一持久化（降级也由 PanelService 写，不绕过） | `services/panel_service.py` |
| A5 | 单元测试：验证不同 kind 不互相覆盖 | `tests/` |

### Phase B: get_panel_verdict 工具（1 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| B1 | 实现 `make_get_panel_verdict` 工具工厂 | `agents/langchain_tools.py` |
| B2 | `PanelService.get_or_read_daily_panel()` 方法（只读路径） | `services/panel_service.py` |
| B3 | 注册到 ChatService 工具列表 + `_EVIDENCE_TOOLS` | `services/chat_service.py` |
| B4 | Chat 前端展示 panel verdict 引用卡片 | `frontend/src/pages/Chat.tsx` |
| B5 | 集成测试：聊天 Agent 调用 get_panel_verdict 各状态 | `tests/` |

### Phase C: 实时分心检测核心（2-3 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| C1 | 专注意向 API：`POST /api/v1/focus/intent {action: start\|stop}` | `api/routes/focus.py` |
| C2 | `DistractionMonitorService`：asyncio.Task + Queue + 滑动窗口 | 新建 `services/distraction_monitor.py` |
| C3 | 三条检测规则实现（含状态机、连续窗口、滞后） | 同上 |
| C4 | 集成到 CollectorService（`put_nowait` 发布） | `services/collector_service.py` |
| C5 | 白名单 + 用户分类缓存接入 | 同上 |
| C6 | 应用生命周期（启动/停止顺序：监控器先启后停） | `main.py` / `app.py` |

### Phase D: 原子节流 + 通知可观测性（1-2 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| D1 | `reserve_intervention_slot()` BEGIN IMMEDIATE 事务 | `services/intervention_service.py` |
| D2 | 预约生命周期状态机 | 同上 |
| D3 | 通知状态 JSON 记录 | `infrastructure/notification.py` |
| D4 | 干预日志写入 `notification_status_json` | `services/intervention_service.py` |
| D5 | 统一前端/后端分类名 | `frontend/` + `api/routes/app_classification.py` |

### Phase E: 前后端联动 + 测试（1-2 天）

| 任务 | 内容 | 文件 |
|------|------|------|
| E1 | 前端专注意向按钮（"开始专注"/"结束专注"） | `frontend/src/pages/Focus.tsx` |
| E2 | 前端实时提醒展示（通知弹窗 + 侧边栏 badge） | `frontend/src/` |
| E3 | WebSocket `intervention_alert` 事件推送 | `api/websocket.py` |
| E4 | 集成测试 + E2E 测试 | `tests/` |

**总估算**：6-10 个工作日

---

## 4. Codex 审查记录摘要

### Round 1（8 项发现）

| # | 严重度 | 发现 | 处置 |
|---|--------|------|------|
| 1 | 🔴 Blocker | Panel/ML 分析共享 (user_id, date) 行，可互相覆盖 | 增加 `analysis_kind` 复合唯一约束 |
| 2 | 🟠 High | WebSocket 是错误运输层 | 改用 `asyncio.Queue` 内部管道 |
| 3 | 🟠 High | 节流检查非原子，并发可绕过 | `BEGIN IMMEDIATE` + 预约生命周期 |
| 4 | 🟠 High | 分心检测规则会产生即时假阳性 | 驻留窗口+连续正窗口+滞后+冷却 |
| 5 | 🟠 High | 降级和成本控制未明确 | 结构化 provenance + 每日声明 |
| 6 | 🟠 High | 白名单/分类未接入运行时 | 优先级链 + 缓存 + 前后端统一 |
| 7 | 🟡 Medium | "完整 transcript" 当前不存在 | 区分 summary/discussion 级别 |
| 8 | 🟡 Medium | 通知不可观测 | 通知状态 JSON + 分渠道追踪 |

### Round 2（语义修正，全项通过）

| 修正点 | 内容 |
|--------|------|
| `analysis_kind` vs `source` | 分成两列，`daily_panel` + `source=ollama` 正确表达"降级的会诊" |
| `get_panel_verdict` 只读 | 移除 `force=true`，始终保持只读 |
| 每日声明机制 | 仅用于 canonical daily panel，不用于手动 rerun |
| detail 合同 | summary=裁决+理由，discussion=transcript |
| asyncio.Task（非 APScheduler）| 匹配现有 collector 架构，正确延迟和背压 |
| Queue maxsize | 1000→200（避免延迟过高的提醒） |
| 迁移默认值 | `legacy_unknown`（非 NULL 非 ml） |
