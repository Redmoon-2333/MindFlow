# MindFlow 系统架构设计文档

> **文档编号**: 04-architecture-design.md
> **版本**: v2.1-reviewed
> **日期**: 2026-07-17
> **作者**: 编排者
> **评审**: critic agent 独立评审，2 Critical + 7 Major + 5 Minor 全部修复（记录见 §0）
> **状态**: Gate 3 验收材料（就绪）
> **上游依赖**: [01-project-analysis](01-project-analysis.md) · [02-benchmark-research](02-benchmark-research.md) · [03-requirements](03-requirements.md)
> **总体原则**: 高性能 > 高可用 > 高专业度 → 事件溯源 + 全异步 + 接口隔离 + 优雅降级

## 0. 评审修复记录

| # | 等级 | 问题 | 修复 |
|---|------|------|------|
| C1 | CRITICAL | Alembic 迁移代码在 AsyncEngine 上用同步 API（3 处错误，运行必崩） | §5.3 重写：独立同步引擎 + `asyncio.to_thread` 包装 |
| C2 | CRITICAL | UUID7 是 Python 3.12 特性，与 3.11 最低版本矛盾 | 采用 `uuid6` PyPI 库（提供 uuid7()，兼容 3.11），新增 ADR-006 |
| M1 | MAJOR | 采集 tick 机制矛盾（APScheduler vs 手动循环） | 统一为 asyncio 手动循环；APScheduler 仅用于定时任务（日报/清理/备份），新增 ADR-007 |
| M2 | MAJOR | heartbeat 合并 UPDATE 与 append-only 语义矛盾 | 语义修正为 **append-mostly**，合并例外事务内原子保证，ADR-001 修订 |
| M3 | MAJOR | NotificationService 被引用但未定义 | 新增 §3.7 通知服务设计 + `infrastructure/notification.py` |
| M4 | MAJOR | C4 对话式反思 / C5 效果评估无架构映射 | 新增 §3.8 扩展点设计（对话状态模型 + EffectivenessService 最小设计，Wave 7/8） |
| M5 | MAJOR | S6 行为画像缺端点 | 新增 `GET /api/v1/analytics/profile` + `get_behavioral_profile()` |
| M6 | MAJOR | baseline 端点超出 03 范围 | 03 API 表同步补充（S1 自然延伸，已注明） |
| M7 | MAJOR | 数据留存清理机制未设计 | §2.4 新增 EventCleanup 调度任务行 |
| m1 | MINOR | WaylandFallbackCollector 与 W5 关系不清 | 澄清：W5 排除的是原生 Wayland 支持，pid 级降级正是 03 W5 规定的行为，保留并标注 |
| m2 | MINOR | NF-P1/P6 无 CI 门禁映射 | §11 补充：性能 NF 由 pytest-benchmark 在 CI 验证 |
| m4 | MINOR | Wave 2 "100% 覆盖" 不现实 | 改为 ≥80%（对齐 NF-Q1），注明 Wave 1 依赖 |
| m5 | MINOR | Wave 4 验收依赖 Wave 3 | 验收拆分：框架+中间件先验收，采集器绑定端点在 Wave 3 后补验 |

---

## 1. 架构总览

### 1.1 进程与部署拓扑

```
┌──────────────────────────────────────────────────────────┐
│                    MindFlow Desktop App                    │
│  (PyInstaller 打包的单可执行文件, watchdog 守护)            │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │  uvicorn.Server (asyncio event loop)              │    │
│  │                                                    │    │
│  │  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │  FastAPI App  │  │  WebSocket    │               │    │
│  │  │  (REST :8765) │  │  /api/v1/ws   │               │    │
│  │  └──────┬───────┘  └──────┬───────┘               │    │
│  │         │                  │                        │    │
│  │  ┌──────┴──────────────────┴───────┐               │    │
│  │  │       Dependency Injection       │               │    │
│  │  │  (FastAPI Depends / 接口协议)     │               │    │
│  │  └──────┬──────────────────┬───────┘               │    │
│  │         │                  │                        │    │
│  │  ┌──────┴──────┐  ┌───────┴────────┐              │    │
│  │  │  Services    │  │  Repositories   │              │    │
│  │  │  (业务逻辑)   │  │  (数据访问)      │              │    │
│  │  └──────┬──────┘  └───────┬────────┘              │    │
│  │         │                  │                        │    │
│  │  ┌──────┴──────────────────┴───────┐               │    │
│  │  │  SQLAlchemy AsyncEngine (aiosqlite) │            │    │
│  │  │  SQLite WAL 模式                  │               │    │
│  │  └──────────────────────────────────┘               │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │  CollectorService (同进程 asyncio task)            │    │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────┐     │    │
│  │  │ WinTracker │ │MacTracker│ │X11Tracker  │     │    │
│  │  │ (win32gui) │ │ (pyobjc) │ │(python-xlib)│    │    │
│  │  └────────────┘ └──────────┘ └────────────┘     │    │
│  │         ↓ (EventCollector protocol)               │    │
│  │  ┌──────────────────────────────────────────┐    │    │
│  │  │  asyncio task: 5s tick 循环 (while+sleep)  │    │    │
│  │  │  → EventBus (asyncio.Queue)               │    │    │
│  │  │  → EventWriter → SQLite (append-mostly)    │    │    │
│  │  └──────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────┘    │
│                                                           │
│  ┌──────────────────────────────────────────────────┐    │
│  │  LLM Pipeline (可选, 三层降级)                     │    │
│  │  DeepSeek API → Ollama(local) → RuleEngine       │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

**关键设计选择**:
- **同进程, 异步 IO 隔离**: Collector 在同一 Python 进程内作为独立 asyncio task 运行。它与 API 层通过 `EventBus`（asyncio.Queue）通信，不与 API 请求路径共享可变状态。对比 ActivityWatch 的多进程方案：复杂度降低了 ~300 行 IPC 代码，watchdog 重启覆盖了"进程崩溃"场景。
- **依赖注入**: 所有 Service 和 Repository 通过 FastAPI `Depends()` 获取实例。废弃旧代码中 `from scheduler import collector` 的全局单例模式。测试时替换为 mock，生产时注入真实实现。

### 1.2 分层架构

```
┌─────────────────────────────────────────┐
│  Presentation 层 (api/)                  │
│  routes.py · websocket.py · middleware/  │
│  → FastAPI endpoints + WS handler        │
├─────────────────────────────────────────┤
│  Application 层 (services/)              │
│  collector_service · analysis_service   │
│  llm_service · intervention_service     │
│  → 业务编排, 无框架依赖                   │
├─────────────────────────────────────────┤
│  Domain 层 (domain/)                     │
│  events.py · procrastination.py         │
│  baseline.py · deviation.py · features  │
│  → 纯 Python, 零外部依赖                  │
├─────────────────────────────────────────┤
│  Infrastructure 层 (infrastructure/)     │
│  database.py · repositories/            │
│  collectors/ · llm/ · config/           │
│  → SQLAlchemy/APScheduler/pywin32 适配   │
└─────────────────────────────────────────┘
```

依赖方向: Presentation → Application → Domain ← Infrastructure（Domain 不依赖任何层）

### 1.3 模块清单

| 模块 | 目录 | 职责 | 依赖 |
|------|------|------|------|
| `api` | `src/api/` | REST + WebSocket 端点，middleware（auth/host/cors/logging），exception handlers | services, domain |
| `services` | `src/services/` | CollectorService, AnalysisService, LLMService, InterventionService, ReportService, ExportService | domain, infrastructure |
| `domain` | `src/domain/` | ActivityEvent, FocusSession, DailyReport, BaselineModel, DeviationDetector, ProcrastinationLabels, CBTTechniques — **零框架/零 IO 依赖** | — |
| `infrastructure` | `src/infrastructure/` | AsyncDatabase, Repositories(Activity/Focus/Report/User), CollectorPlatforms(Win/Mac/X11), LLMClient, ConfigLoader | 外部库(SQLAlchemy/pywin32 等) |
| `train` | `src/train/` | ML 训练入口 CLI + pipeline，合成数据生成器 | domain, infrastructure |

**模块依赖 DAG**: `api` → `services` → `domain` ← `infrastructure` → `api`（唯一允许的逆向: infrastructure 的 middleware 被 `main.py` 注入到 api）

---

## 2. 数据模型 (Event Sourcing)

### 2.1 核心思想

从旧架构的 CRUD-ORM 模型转向 **Event Sourcing**。所有行为数据作为不可变的 `ActivityEvent` 追加，分析结果作为 **投影 (Projection)** 从事件流计算得到。

**Decision Record: Event Sourcing over CRUD**
- **为什么**: 旧代码的 `duration_seconds` 用配置值估算（P0 技术债 #2）。事件流保留原始 tick 数据，duration 从相邻事件时间戳精确计算。合并在查询时配置，不丢失分辨率。
- **对标**: ActivityWatch 的 Bucket+Event 模型，ActivityWatch 的 heartbeat 合并机制
- **权衡**: 存储量增加（每条 tick 一行 vs 聚合后一行），但根据 AW 的经验，heartbeat 合并将 90%+ 的磁盘写压缩为单行更新。30 天滑动窗口限制无限增长

### 2.2 事件存储 Schema

```sql
-- 事件流 (append-mostly: 常规仅追加; 唯一例外是 heartbeat 合并对上一行的原子 UPDATE)
CREATE TABLE activity_events (
    id TEXT PRIMARY KEY,              -- UUIDv7 via `uuid6` 库 (时间排序, 兼容 Python 3.11, 见 ADR-006)
    user_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,          -- ISO8601 UTC (带时区)
    duration_s REAL NOT NULL DEFAULT 0.0,  -- 距上一个事件的实测间隔
    data_json TEXT NOT NULL,          -- {"app_name":"...","window_title":"...","is_idle":false,...}
    event_type TEXT NOT NULL DEFAULT 'window_snapshot',  -- window_snapshot | idle_change | manual_tag
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX idx_events_user_time ON activity_events(user_id, timestamp);
CREATE INDEX idx_events_type ON activity_events(user_id, event_type, timestamp);
```

**Heartbeat 合并（append-mostly 的唯一例外）**: 当 `event_type='window_snapshot'` 且 `data_json` 中的 `app_name` 与上一个事件相同，在 `pulsetime_s` 窗口内（默认 10 秒），不插入新行，**在单事务内**更新上一行的 `duration_s += pulsetime`。这是对 append-only 语义的明确让步（换取 90%+ 写削减），事务原子性保证并发安全。窗口配置化为 `config.heartbeat_pulsetime_s`。

### 2.3 投影表 (聚合视图)

```sql
-- 专注会话 (从事件流聚合)
CREATE TABLE focus_sessions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,               -- YYYY-MM-DD
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    session_type TEXT NOT NULL,       -- focus | distraction | neutral
    dominant_app TEXT,
    focus_score REAL,
    switch_count INTEGER,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX idx_sessions_user_date ON focus_sessions(user_id, date);

-- 日报 (幂等 — 每日一次)
CREATE TABLE daily_reports (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    total_focus_min REAL DEFAULT 0,
    total_distraction_min REAL DEFAULT 0,
    focus_score REAL DEFAULT 0,
    top_apps_json TEXT,               -- [{"app":"code","minutes":120},...]
    switch_frequency REAL DEFAULT 0,  -- avg per hour
    pattern_summary TEXT,             -- 自然语言摘要
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE(user_id, date)
);

-- LLM 归因分析 (幂等 — 按 session 日期)
CREATE TABLE procrastination_analyses (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    procrastination_types_json TEXT,  -- ["impulsivity","emotional_regulation"]
    type_confidence_json TEXT,        -- {"impulsivity":0.82,"emotional_regulation":0.67}
    cognitive_distortions_json TEXT,
    cbt_technique TEXT,
    response_text TEXT,
    llm_model TEXT,                   -- 哪个模型生成的
    llm_cost_usd REAL,                -- 单次调用的美元成本
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE(user_id, date)
);

-- 干预日志
CREATE TABLE intervention_logs (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    triggered_at TEXT NOT NULL,
    intervention_type TEXT NOT NULL,  -- task_breakdown | nudge | environment_optimization | smart_prioritization
    cbt_technique TEXT,
    context_json TEXT,                -- 触发时的行为摘要
    user_response TEXT,               -- accepted | ignored | dismissed
    response_latency_s REAL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- 基线模型 (JSON 大对象 — 在线 Welford 批量更新)
CREATE TABLE baseline_models (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE,
    model_json TEXT NOT NULL,         -- Welford 统计值转 JSON
    training_events_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- 用户偏好 (Key-Value JSON)
CREATE TABLE user_preferences (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE,
    preferences_json TEXT NOT NULL DEFAULT '{}',  -- 全量偏好 JSON
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
```

**时区政策**: 所有 `timestamp` 列存储 UTC ISO8601 **带时区标记** (`+00:00`)，查询时转换为 `zoneinfo` 本地时区。废弃旧代码中的 naive datetime（P0 技术债 #3）。

### 2.4 聚合策略

| 投影 | 触发方式 | 聚合逻辑 |
|------|---------|---------|
| FocusSession | 调度任务（每天 23:59）或按需 | 扫描 `activity_events BETWEEN start AND end`，窗口会话聚合（专注块分隔阈值 = config.focus_block_gap_s） |
| DailyReport | 调度任务（每天 00:01）或按需，幂等检查 | 聚合当天的 FocusSession + ActivityEvent 统计数据 |
| BaselineModel | 每日增量更新 + 每次新事件后缓冲更新 | Welford 在线算法，避免全量重新计算 |
| ProcrastinationAnalysis | LLM 归因 API 调用后 | 写入一次，幂等性靠 UNIQUE(user_id, date) 保证 |
| **EventCleanup** | 调度任务（APScheduler，每天 03:00） | `DELETE FROM activity_events WHERE timestamp < now - config.event_retention_days`（默认 30 天，7-90 可配置），分批删除（每批 10,000 行 + 提交）避免长事务；聚合表永久保留 |

---

## 3. 核心模块设计

### 3.1 Collector Service（采集器抽象）

```python
# 平台无关接口 — 业务逻辑零平台依赖
class EventCollector(Protocol):
    async def snapshot(self) -> WindowSnapshot: ...
    async def idle_seconds(self) -> float: ...

@dataclass
class WindowSnapshot:
    app_name: str
    window_title: str
    process_name: str
    is_idle: bool
    timestamp_utc: datetime  # 带时区

# 平台实现 (每平台 <200 行)
class Win32Collector(EventCollector): ...   # win32gui + GetLastInputInfo
class MacOSCollector(EventCollector): ...    # pyobjc NSWorkspace + CGEventSourceSecondsSinceLastEventType
class X11Collector(EventCollector): ...      # python-xlib EWMH
class WaylandFallbackCollector(EventCollector): ...  # pid 级降级 (03 W5 规定的行为: 非原生支持, psutil 仅进程名, 无窗口标题)

# 工厂
def create_collector(platform: str | None = None) -> EventCollector:
    platform = platform or sys.platform
    collectors = {
        "win32": Win32Collector,
        "darwin": MacOSCollector,
        "linux": X11Collector,  # 登录 session 检测 XDG_SESSION_TYPE=wayland → WaylandFallbackCollector
    }
    return collectors[platform]()
```

**采集循环**:
```
async def _collect_loop(collector, event_bus, config):
    while running:
        snapshot = await collector.snapshot()
        await event_bus.put(snapshot)
        await asyncio.sleep(config.collect_interval_s)
```

**Heartbeat 合并在 EventWriter 层**: `EventWriter` 持有 `_last_event` 缓存，`pulsetime_s` 内相同 app 合并，减少 90%+ 的 INSERT。

### 3.2 Analysis Service（行为分析）

核心算法来自旧代码的可复用组件，融入事件流模型：

| 分析任务 | 方法 | 输入 | 输出 | 来源 |
|---------|------|------|------|------|
| 专注分数 (0-100) | 多因素加权 | 当前事件的 ActivityEvent[] | float | 旧 `features.py:86-107` |
| 应用使用排名 | 按 app_name 分组求和 duration_s | ActivityEvent[] over 时间段 | AppUsage[] | 旧 `features.py` |
| 专注会话识别 | 窗口 + 专注/分心阈值 | ActivityEvent[] over 天 | FocusSession[] | 旧 `patterns.py` |
| 日报生成 | 聚合 + 幂等检查 | FocusSession[] + ActivityEvent[] | DailyReport (幂等) | 旧 `patterns.py` |
| 基线更新 | Welford 在线 | ActivityEvent metrics | BaselineModel | 旧 `baseline.py:57-108` |
| 偏差检测 | 多维 Z-score | BaselineModel + 最新窗口 | Deviation[] + 严重度 | 旧 `deviation.py:46-99` |
| HMM 状态推断 | 回退链 (hmmlearn → Markov) | 特征向量序列 | 隐藏状态序列 | 旧 `ml_models.py:325-448` |
| 弱监督标签 | 6 信号 Consensus Labeler | 特征向量 | Label + 置信度 | 旧 `labeling.py:128-165` |

### 3.3 LLM Pipeline（归因+干预）

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ 行为摘要生成  │ ──→ │  LLM 客户端 (三层) │ ──→ │ ProcrastinationAnalysis │
│ context_packer │     │                   │     │ (Pydantic 约束解码)     │
│ (旧代码 保留)   │     │ L1: DeepSeek API  │     └──────────────────┘
└──────────────┘     │ L2: Ollama local   │              │
                     │ L3: RuleEngine     │              ▼
                     └───────────────────┘     ┌──────────────────┐
                                               │ Intervention 生成  │
                                               │ → CBT 技术选择     │
                                               │ → 具体建议文本     │
                                               │ → 节流检查(C3)    │
                                               └──────────────────┘
```

**三层降级协议**:
```python
class LLMClient:
    async def analyze(self, summary: BehaviorSummary) -> ProcrastinationAnalysis:
        # L1: DeepSeek API (主路径, 期望 95% 请求走此路径)
        try:
            result = await self._deepseek_call(summary)
            return ProcrastinationAnalysis.model_validate(result)
        except (APIError, TimeoutError, ValidationError):
            logger.warning("DeepSeek unavailable, falling back to L2")

        # L2: Ollama / 本地模型 (可选, 零成本)
        try:
            if self._ollama_available:
                result = await self._ollama_call(summary)
                return ProcrastinationAnalysis.model_validate(result)
        except Exception:
            logger.warning("Ollama unavailable, falling back to L3")

        # L3: Rule Engine (兜底, ¥0, 永远可用)
        return self._rule_engine.analyze(summary)
```

**Rule Engine 兜底逻辑** (LLM 不可用时的行为):
```python
class RuleEngine:
    def analyze(self, summary: BehaviorSummary) -> ProcrastinationAnalysis:
        # 基于 TMT 5 类可计算规则 (需求文档 §3.4)
        # 输出结构与 LLM 一致的 ProcrastinationAnalysis，标注 source="rule_engine"
```

### 3.4 Intervention Service（干预引擎）

```python
class InterventionService:
    def __init__(self, throttle: InterventionThrottle, notifier: NotificationService):
        ...

    async def maybe_intervene(self, analysis: ProcrastinationAnalysis) -> Intervention | None:
        # 1. 节流检查
        if not await self.throttle.can_intervene(analysis.user_id, analysis.types):
            return None

        # 2. 深度工作状态检测 → 零打扰
        if await self._is_deep_work(analysis.user_id):
            return None

        # 3. 干预策略生成
        intervention = self._build_intervention(analysis)

        # 4. 推送通知 + 日志
        await self.notifier.send(intervention)

        return intervention
```

**节流状态机** (JITAI 理论 + DIAMANTE RCT 验证):
```
状态: IDLE → ALLOWED (每天 3 次, 间隔≥2h) → THROTTLED (超限) → COOLDOWN (次日重置)
疲劳检测: 7 日 ignore_rate > 60% → 自动降频到 1 次/天
深度工作: focus_score > 80 → 拦截所有干预请求
手动触发: 用户主动 /intervention/trigger → 不受节流限制 (但有严格速率限制)
```

### 3.5 API 层设计

**端点结构**（最终，已对接 WebSocket 路径 + 速率限制）：

| 方法 | 路径 | 说明 | 速率限制 | MoSCoW |
|------|------|------|---------|--------|
| GET | /api/v1/health | 采集器/DB 健康 | 无 | Must |
| GET | /api/v1/collector | 采集器运行状态 | 全局 | Must |
| POST | /api/v1/collector | 启动采集器 | 全局 | Must |
| POST | /api/v1/collector/stop | 停止采集器 | 全局 | Must |
| GET | /api/v1/activities | 今日活动 (分页+过滤) | 全局 | Must |
| GET | /api/v1/activities/current | 当前活动快照 | 全局 | Must |
| WS | /api/v1/ws | 实时推送 | 推送节流 2s | Must |
| GET | /api/v1/focus | 今日专注报告 | 全局 | Must |
| GET | /api/v1/focus/trend | N 日趋势 | 全局 | Must |
| GET | /api/v1/analytics/patterns | 分心模式 | 全局 | Should |
| GET | /api/v1/analytics/baseline | 基线模型 (S1 只读延伸, 03 已同步) | 全局 | Should |
| GET | /api/v1/analytics/profile | 行为画像 (S6: 工作习惯/专注高峰/分心触发) | 全局 | Should |
| GET | /api/v1/reports/daily | 日报 | 全局 | Should |
| GET | /api/v1/reports/weekly | 周报 | 全局 | Should |
| POST | /api/v1/analytics/attribution | LLM 归因 | 1/30s **\|** 20/日 | Could |
| POST | /api/v1/analytics/train | 触发模型训练 | 1/60s **\|** 3/日 | Should |
| POST | /api/v1/intervention/trigger | 触发干预 | C3 节流 | Could |
| GET | /api/v1/export | 数据导出 | 全局 | Should |
| PUT/PATCH | /api/v1/preferences | 用户偏好 | 全局 | Must |

**Middleware 栈**（按执行顺序）:
```
Request
  → StructuredLoggingMiddleware (request_id, timing)
  → HostValidationMiddleware (localhost only)
  → AuthMiddleware (Bearer token, check)
  → RateLimitMiddleware (token bucket)
  → CORSMiddleware (localhost origins only)
  → ExceptionHandlerMiddleware (RFC 9457 problem+json)
  → Route handler
```

### 3.6 Repository 模式

```python
# 数据访问抽象 — Service 层只依赖 Repository 协议, 不直接接触 SQL
class ActivityRepository(Protocol):
    async def append_event(self, event: ActivityEvent) -> str: ...
    async def query_range(self, user_id: int, start: datetime, end: datetime) -> list[ActivityEvent]: ...
    async def last_event(self, user_id: int) -> ActivityEvent | None: ...

class FocusSessionRepository(Protocol):
    async def get_or_create_daily(self, user_id: int, date: date) -> FocusSession: ...
    async def query_range(self, ...) -> list[FocusSession]: ...

class ProcrastinationAnalysisRepository(Protocol):
    async def get_by_date(self, user_id: int, date: date) -> ProcrastinationAnalysis | None: ...
    async def save(self, analysis: ProcrastinationAnalysis) -> None: ...
```

**实现**: `SQLAlchemyActivityRepository` 包装 `AsyncSession`。测试时替换为 `InMemoryActivityRepository`。

### 3.7 通知服务（NotificationService）

InterventionService 的干预推送和 NF-R7 的备份失败告警共用一套通知抽象：

```python
class NotificationService(Protocol):
    async def send(self, title: str, body: str, urgency: Literal["low","normal","critical"] = "normal") -> bool: ...

# 平台实现 (infrastructure/notification.py, 每平台 <80 行)
class WindowsNotifier(NotificationService): ...   # winrt / plyer (Toast)
class MacOSNotifier(NotificationService): ...     # pyobjc NSUserNotificationCenter
class LinuxNotifier(NotificationService): ...     # notify-send / dbus
class LogOnlyNotifier(NotificationService): ...   # 降级兜底: 仅写日志 (无桌面环境/依赖缺失)
```

工厂 `create_notifier()` 按平台选择，依赖缺失时自动降级到 `LogOnlyNotifier`（通知失败绝不抛异常到业务层）。前端在线时优先走 WebSocket `intervention` 消息帧，桌面通知作为前端不在线时的兜底通道。

### 3.8 C4/C5 扩展点（本期最小设计，Wave 7/8 择机实现）

**C4 对话式反思** — 架构预留：
- `domain/dialog_state.py`: `ReflectionDialog`（对话轮次上限 10、上下文窗口 = 最近 N 轮 + 当日行为摘要、状态机 OPEN→ACTIVE→CLOSED）
- 端点 `POST /api/v1/analytics/reflection`（Could 级，速率限制同 attribution）
- 对话历史存 `reflection_dialogs` 表（本期建表可延迟到实现波次的 Alembic 迁移）
- LLM 调用复用 §3.3 三层降级链（规则引擎层返回预置反思问题模板）

**C5 效果评估** — 架构预留：
- `services/effectiveness_service.py`: `compare_windows(intervention_id)` — 对比干预前后 30 分钟窗口的 focus_score / switch_rate / distraction_ratio
- 数据来源：`intervention_logs.triggered_at` ± 30min 的事件流查询（无需新表）
- 输出并入周报（`reports/weekly` 响应的 `intervention_effectiveness` 字段）

**决策**: 两者均不阻塞 Wave 1-6；若 2027 Q1 工期紧张，C4 降级为"预置反思问题（无 LLM 多轮）"，C5 保持（纯统计计算，成本低价值高——结题报告的实验数据来源）。

---

## 4. 安全性设计

### 4.1 威胁模型

| 威胁 | 攻击面 | 缓解 |
|------|--------|------|
| 恶意本地进程访问 API | localhost:8765 无认证 → 任意读写用户行为数据 | Token 认证 (随机 64B hex 文件, 0600 权限) |
| 恶意网页 DNS rebinding 打 localhost | 同源策略不保护 localhost | Host header 校验 + Token 兜底 |
| 恶意软件窃取 token | 读取 ~/.mindflow/token | 文件权限 0600; Windows: 加密存储; macOS: Keychain |
| CSRF (如果后续加 Web 前端) | 浏览器跨域请求 localhost | 当前 N/A (前端同源托管), 未来 SameSite cookie |
| LLM API 滥用 | 高频调用 DeepSeek → 超预算 | 速率限制 20 次/日 + 内置 key 月度硬上限 |
| LLM 输出含不安全内容 | DeepSeek 输出注入/诊断用语 | 输出过滤 + constraint decoding Pydantic 双重验证 |
| 数据泄露 | 备份文件可被任意用户读取 | SQLite 文件 + 备份文件权限 0600 |
| 危机场景误处理 | LLM 误判自杀意念 | **危机检测独立于 LLM**, 关键词匹配在 LLM 调用前执行 |

### 4.2 认证流程

```
应用启动
  → 检查 ~/.mindflow/token 是否存在
  → 不存在 → 生成 os.urandom(64).hex() → 写入文件 (0600)
  → 存在 → 从文件读取
  → 前端/CLI 读取同一文件 → 每个请求附加 Authorization: Bearer <token>
  → AuthMiddleware 校验 → 不匹配 → 401
```

**不持久化 token 在数据库中** — 文件系统是桌面应用的标准本地秘密存储机制（对标 KeePassXC 的 Native Messaging token 设计）。

---

## 5. 可靠性设计

### 5.1 崩溃恢复

```
主进程 watchdog:
  while True:
    try:
      server.serve()
    except Exception:
      crash_count += 1
      if crash_count > 3 within 60s:
        logger.critical("Crash loop detected, exiting")
        break
      logger.error(f"Restarting in 1s... (crash #{crash_count})")
      time.sleep(1)
```

**关键**: 采集器崩溃通过 `try/except` 在 asyncio task 内捕获，不影响 API 事件循环。WebSocket 连接在采集器重启期间收到 `collector_unavailable` 状态推送。

### 5.2 数据可靠性

- **SQLite WAL**: `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;`
- **启动完整性**: `PRAGMA integrity_check` → 失败 → 自动 VACUUM 尝试恢复 → 恢复失败 → 记录错误 + 继续启动（数据已备份，见下）
- **每日自动备份**: APScheduler 每日任务 `VACUUM INTO '{data_dir}/backups/mindflow-{date}.db'`
- **优雅关闭**: `lifespan` shutdown handler: 停止采集器 → 等待 EventWriter 清空队列 → 关闭 SQLAlchemy engine → 退出。超时 5 秒强制退出。

### 5.3 数据库迁移

```python
# Alembic 是同步 API，且需要同步 Connection。
# 在 async lifespan 中：独立同步 URL + asyncio.to_thread，绝不阻塞事件循环。
def _run_migrations_sync(db_url: str) -> None:
    from alembic.config import Config
    from alembic import command
    cfg = Config(str(BASE_DIR / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")

async def run_migrations(async_db_url: str) -> bool:
    """返回迁移是否成功；失败时降级运行 (NF-R5)。"""
    sync_url = async_db_url.replace("sqlite+aiosqlite://", "sqlite://")
    try:
        await asyncio.to_thread(_run_migrations_sync, sync_url)
        return True
    except Exception as e:
        logger.critical(f"Migration failed: {e}. Running with existing schema.")
        return False  # health endpoint 暴露 migration_failed 状态
```

`env.py` 必须启用 `render_as_batch=True`（SQLite `ALTER COLUMN` 限制）。

---

## 6. 可观测性设计

### 6.1 日志

```python
# loguru 配置
logger.add(
    appdirs.user_data_dir / "logs" / "mindflow_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    compression="gz",
    format="{time:ISO} | {level: <8} | {extra[request_id]} | {name}:{function}:{line} | {message}",
    serialize=False,  # 文本格式 (开发友好)
)
# JSON 格式用于生产:
logger.add(
    appdirs.user_data_dir / "logs" / "mindflow_json_{time}.log",
    rotation="10 MB",
    retention="7 days",
    serialize=True,   # JSON 格式 (可被 Sentry/ELK 消费)
)
```

### 6.2 指标

| 指标 | 类型 | 说明 |
|------|------|------|
| `collector.tick_duration_ms` | Histogram | 采集 tick 耗时 |
| `api.request_duration_ms` | Histogram | API 端点到端点响应时间 |
| `db.write_duration_ms` | Histogram | SQLite INSERT 耗时 |
| `llm.call_duration_ms` | Histogram | LLM API 调用往返时间 |
| `llm.call_cost_usd` | Counter | LLM 累计调用费用 |
| `events.processed` | Counter | 采集器处理后的事件数 |
| `events.heartbeat_merged` | Counter | 合并的 heartbeat 事件数 |

实现方式: `prometheus_client` 输出 `/metrics` 端点（仅在 debug 模式，localhost only）。

### 6.3 崩溃上报 (opt-in)

```python
# 首次启动:
if ask_user_consent("Would you like to send crash reports to help improve MindFlow?"):
    sentry_sdk.init(dsn=SENTRY_DSN, before_send=filter_sensitive, traces_sample_rate=0)
```

`filter_sensitive`: 移除本地路径、窗口标题内容、用户名。仅保留异常类型、traceback、版本。

---

## 7. 打包与分发

### 7.1 PyInstaller 配置

```ini
# mindflow.spec
a = Analysis(['src/main.py'],
    pathex=[],
    binaries=[],
    datas=[('alembic/', 'alembic/'), ('alembic.ini', '.')],
    hiddenimports=['sklearn', 'hmmlearn', 'sqlalchemy', 'aiosqlite', 'pydantic'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'test', 'unittest'],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas,
    name='MindFlow',
    console=False,  # 无终端窗口
    icon='assets/icon.ico',
)
```

**hmmlearn 依赖解决**: sklearn 有官方 PyInstaller hook — hook-skimage (sklearn 也适用)。Joblib 的 `loky` 后端打包后需额外隐藏导入 `joblib.externals.loky.backend`。

### 7.2 自动更新

```python
# tufup 集成
from tufup.client import Client
updater = Client(app_name='MindFlow', current_version='2.0.0',
    metadata_dir=appdirs.user_cache_dir,
    target_dir=appdirs.user_data_dir,
    url='https://releases.mindflow.app/metadata/')
# 定时检查 + 用户手动触发
```

---

## 8. 目录结构（最终）

```
mindflow-app/backend-next/
├── alembic/                  # Alembic 迁移
│   ├── env.py               # render_as_batch=True
│   └── versions/
├── alembic.ini
├── pyproject.toml           # [project] + [tool.ruff] + [tool.mypy]
├── mindflow.spec            # PyInstaller spec
├── src/
│   ├── main.py              # 入口: uvicorn.Server 编程式启动 + watchdog
│   ├── config.py            # Pydantic BaseSettings (多源: env/.env/config.toml)
│   ├── logging_config.py    # loguru 配置
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py        # 路由注册 + 异常处理
│   │   ├── routes/          # 按领域拆分路由模块
│   │   │   ├── collector.py
│   │   │   ├── activities.py
│   │   │   ├── focus.py
│   │   │   ├── analytics.py
│   │   │   ├── reports.py
│   │   │   ├── export.py
│   │   │   └── preferences.py
│   │   ├── websocket.py     # WS handler + 客户端追踪
│   │   └── middleware/
│   │       ├── auth.py      # Bearer token 验证
│   │       ├── host.py      # Host header 校验
│   │       ├── ratelimit.py # 令牌桶
│   │       └── logging.py   # 结构化请求日志
│   ├── services/
│   │   ├── collector_service.py
│   │   ├── analysis_service.py
│   │   ├── llm_service.py
│   │   ├── intervention_service.py
│   │   ├── report_service.py
│   │   └── export_service.py
│   ├── domain/
│   │   ├── events.py        # ActivityEvent, WindowSnapshot
│   │   ├── sessions.py      # FocusSession
│   │   ├── features.py      # 专注分数, 应用排名 (旧 features.py 迁移)
│   │   ├── baseline.py      # Welford 在线算法 (旧 baseline.py 迁移)
│   │   ├── deviation.py     # 多维 Z-score (旧 deviation.py 迁移)
│   │   ├── procrastination.py  # TMT 标签模型 + 标签分类逻辑
│   │   ├── labeling.py      # Consensus Labeler (旧 labeling.py 迁移)
│   │   └── cbt_techniques.py  # CBT 技术枚举 + 匹配映射
│   ├── infrastructure/
│   │   ├── database.py      # AsyncEngine + SessionLocal + WAL PRAGMA
│   │   ├── repositories/
│   │   │   ├── activity.py  # SQLAlchemy + heartbeat 合并
│   │   │   ├── focus.py
│   │   │   ├── report.py
│   │   │   └── analysis.py
│   │   ├── collectors/
│   │   │   ├── base.py      # EventCollector Protocol
│   │   │   ├── win32.py     # <200 行
│   │   │   ├── darwin.py    # <200 行
│   │   │   ├── x11.py       # <200 行
│   │   │   └── wayland_fallback.py
│   │   ├── llm/
│   │   │   ├── client.py    # DeepSeek API (httpx async)
│   │   │   ├── instructor.py # Pydantic constraint decoding
│   │   │   └── rule_engine.py # 兜底规则引擎
│   │   ├── notification.py  # NotificationService 协议 + 平台实现 + LogOnly 降级
│   │   └── security/
│   │       ├── token_manager.py  # Token 生成/读取/验证
│   │       └── crisis_detector.py # 独立危机检测
│   └── train/               # ML 训练 CLI
│       ├── __init__.py
│       ├── pipeline.py      # 训练流程编排 (旧 train.py 重构)
│       └── synthetic_data.py # 合成数据生成器 (旧 data_pipeline.py 迁移)
├── tests/
│   ├── conftest.py           # async fixtures (engine, client, test db)
│   ├── unit/
│   │   ├── test_domain/      # 纯函数测试
│   │   └── test_services/    # Service+Mock 测试
│   ├── integration/
│   │   ├── test_api/         # httpx.AsyncClient 端点测试
│   │   └── test_db/          # 真实 SQLite 测试
│   ├── ml/
│   │   ├── test_baseline.py
│   │   ├── test_deviation.py
│   │   ├── test_labeling.py
│   │   └── test_hmm.py
│   └── performance/
│       ├── test_collector_perf.py
│       └── test_api_perf.py
└── assets/
    ├── icon.ico
    └── icon.png
```

### 技术栈最终表

| 层 | 技术 | 版本 |
|----|------|------|
| Python | 3.11+ | 3.11 最低 |
| Web | FastAPI (async) + uvicorn (programmatic) | ≥0.115 |
| ORM | SQLAlchemy 2.0 (asyncio + aiosqlite) | ≥2.0 |
| DB | SQLite WAL | 3.35+ (系统自带) |
| 迁移 | Alembic (render_as_batch=True) | ≥1.13 |
| 调度 | APScheduler 3.x (AsyncIOScheduler) — **仅定时任务**（日报/清理/备份），采集 tick 用 asyncio 循环 (ADR-007) | ≥3.10 |
| ID 生成 | uuid6 (PyPI) — UUIDv7 时间排序, 兼容 Python 3.11 (ADR-006) | ≥2024.x |
| 采集 Win | pywin32, psutil, win32gui | latest |
| 采集 Mac | pyobjc-framework-Cocoa, pyobjc-framework-Quartz | latest |
| 采集 Linux X11 | python-xlib | ≥0.33 |
| ML | scikit-learn, hmmlearn, pandas | ≥1.5 |
| LLM | httpx + openai SDK (DeepSeek 兼容) + instructor (constraint decoding) | latest |
| 验证 | Pydantic ≥2.0 | ≥2.0 |
| 日志 | loguru | ≥0.7 |
| 指标 | prometheus_client (可选 debug 模式) | latest |
| 备份 | SQLite VACUUM INTO (内建) | — |
| 崩溃上报 | sentry-sdk（opt-in） | latest |
| 打包 | PyInstaller + tufup | ≥6.0 |
| CI | GitHub Actions: pytest, mypy --strict, ruff, bandit | — |
| 测试 | pytest + pytest-asyncio + pytest-cov + hypothesis + httpx | latest |
| 路径 | platformdirs | ≥4.0 |

---

## 9. Architecture Decision Records (ADR)

### ADR-001: Event Sourcing 数据模型（append-mostly）
- **状态**: ACCEPTED (v2.1 修订)
- **决定**: 行为数据使用 `activity_events` 表，**append-mostly** 语义：常规仅追加；唯一例外是 heartbeat 合并对最近一行的原子 UPDATE（单事务）。分析结果作为投影表
- **理由**: 消除旧架构 duration 用配置值估算的 P0 缺陷；保留原始 tick 级数据支持不同粒度的后处理分析；对标 ActivityWatch 的成熟模型
- **替代方案**: 严格 append-only + 查询时合并 — 存储与查询成本高；DELETE+INSERT 合并 — 写放大 2 倍且 id 不稳定
- **权衡**: 放弃纯不可变语义换取 90%+ 写削减（AW 实践验证）；UPDATE 只触及最近一行，历史行事实上不可变

### ADR-002: 同进程 + asyncio Task 采集器
- **状态**: ACCEPTED
- **决定**: 采集器以独立 asyncio task 运行在 API 进程内，通过 EventBus（Queue）通信
- **理由**: 降低 IPC 复杂度 ~300 行代码；watchdog 机制已覆盖崩溃恢复，隔离 IPC 并非必要；预留 `CollectorService` 接口可未来无痛拆分
- **替代方案**: AW 式多进程（ActivityWatch 模式） — MindFlow 没有开放 watcher 生态的需求，多进程的收益不抵成本
- **风险**: 采集器异步死循环可能阻塞事件循环 — 通过 `asyncio.wait_for` + timeout 保护

### ADR-003: 三层 LLM 降级链
- **状态**: ACCEPTED
- **决定**: DeepSeek API (L1) → Ollama local (L2) → RuleEngine (L3)
- **理由**: LLM API 不可用时行为分析和基本干预必须可用（NF-R3）；规则引擎 ¥0 成本且永不失败，是 LLM 功能的可靠性基底
- **替代方案**: 仅 API（单点故障） — 违反高可用原则

### ADR-004: localhost Token 文件认证
- **状态**: ACCEPTED
- **决定**: 随机 64 字节 hex token 存储在 `platformdirs` 配置目录的文件中（0600 权限）
- **理由**: 无网络依赖、无密码记忆、文件系统权限即安全边界；对标 KeePassXC 的 Native Messaging 方案
- **替代方案**: 无认证（现状 — 不可接受）、OAuth（桌面应用 overkill）

### ADR-005: SQLite WAL + VACUUM INTO 而非 PostgreSQL
- **状态**: ACCEPTED
- **决定**: 坚持 SQLite 作为唯一生产数据库
- **理由**: 零配置（用户无需装 Docker/PostgreSQL）、单文件隐私友好、WAL 模式已解决并发读写场景；Litestream/VACUUM INTO 覆盖备份需求。ActivityWatch 的生产实践证明了这条路径
- **权衡**: 不支持远程并发客户端 — 但对于纯本地应用程序没有这个需求
- **备用**: 连接 URL 配置方式预留 PostgreSQL，供未来扩展

### ADR-006: UUIDv7 经 `uuid6` 库而非提升 Python 版本
- **状态**: ACCEPTED
- **决定**: 使用 PyPI `uuid6` 库提供 `uuid7()`，保持 Python 3.11 最低版本
- **理由**: `uuid.uuid7()` 是 Python 3.12 特性；现有 conda 环境为 3.11.15，队友环境迁移成本 > 一个 50KB 纯 Python 依赖；UUIDv7 时间排序特性对 append 写入的 B-tree 局部性有实测收益
- **替代方案**: 提升到 3.12（环境迁移摩擦）；uuid4（放弃时间排序，插入性能 -5~10%）

### ADR-007: 采集 tick 用裸 asyncio 循环；APScheduler 仅管定时任务
- **状态**: ACCEPTED
- **决定**: 采集器 5s tick 用 `while running: ...; await asyncio.sleep(interval)` 裸循环（asyncio.create_task 托管）；APScheduler AsyncIOScheduler 只负责日报生成（00:01）、事件清理（03:00）、每日备份等 cron 类任务
- **理由**: tick 不需要 cron/coalesce/misfire 语义，裸循环少一层调度抽象、误差更小、崩溃恢复路径更简单（task 重建即可）
- **替代方案**: 全部走 APScheduler — tick 场景引入不必要的 job store/executor 概念

---

## 10. 从旧代码的迁移映射

| 旧文件 | 新文件 | 操作 |
|--------|--------|------|
| `collector/tracker.py` | `infrastructure/collectors/win32.py` + `base.py` | rewrite |
| `collector/scheduler.py` | `services/collector_service.py` | rewrite |
| `models/database.py` | `infrastructure/database.py` | refactor → async |
| `models/schemas.py` | `domain/events.py` + Alembic migrations | keep → 重构为 event model |
| `analyzer/features.py` | `domain/features.py` | keep (几乎不变) |
| `analyzer/patterns.py` | `services/analysis_service.py` | keep → 事件流化 |
| `analyzer/baseline.py` | `domain/baseline.py` | keep (直接迁移) |
| `analyzer/deviation.py` | `domain/deviation.py` | keep (直接迁移) |
| `analyzer/labeling.py` | `domain/labeling.py` | keep (直接迁移) |
| `analyzer/title_analyzer.py` | `domain/features.py` (内联) | keep → 合并 |
| `analyzer/context_packer.py` | `services/llm_service.py` | keep → 集成到服务 |
| `analyzer/ml_models.py` | `train/pipeline.py` + `domain/` (HMM) | refactor → 拆分 |
| `analyzer/data_pipeline.py` | `train/synthetic_data.py` | keep（合成数据生成器） |
| `analyzer/train.py` | `train/pipeline.py` | refactor → pipeline 编排 |
| `api/routes.py` | `api/routes/` (多文件) | refactor → 按领域拆分 |
| `api/websocket.py` | `api/websocket.py` | refactor |
| `config.py` | `config.py` | refactor → 扩展 |
| `main.py` | `main.py` | rewrite → programmatic uvicorn |
| `tray.py` | **删除**（打包时另建） | drop |
| `logging_config.py` | `logging_config.py` | refactor → loguru |

---

## 11. 测试策略（架构级）

### 11.1 测试金字塔

```
        ┌──────────┐
        │  E2E     │  2-3: 启动完整服务器 + 模拟用户工作流
        │ (全栈)    │      工具: 手动 QA + 脚本化的 Subprocess
        ├──────────┤
        │集成测试   │  ~20: API 端点异步客户端, DB 回滚, WS 握手
        │(API+DB)   │      工具: httpx.AsyncClient + aiosqlite :memory:
        ├──────────┤
        │ 单元测试  │  ~80: domain 层纯函数, service+repository mock,
        │          │      规则引擎/降级链/节流逻辑
        │          │      工具: pytest + pytest-asyncio + hypothesis
        └──────────┘
```

### 11.2 关键测试场景

- `test_domain_features.py` — 专注分数 & 应用排名（真实/边界/空输入）
- `test_domain_baseline.py` — Welford 增量更新 100 次后 vs 全量重新计算
- `test_domain_deviation.py` — Z-score 计算 在已知分布下的预期值
- `test_domain_labeling.py` — 6 信号所有排列组合
- `test_domain_procrastination.py` — 5 类型 TMT 规则的正/反/边角示例
- `test_services_collector.py` — 采集循环: mock 采集器返回固定快照, 验证 EventBus 写入
- `test_services_llm.py` — L1 mock → L2 mock → L3 RuleEngine 触发, 验证输出一致
- `test_services_intervention.py` — Throttle: 3次后拒绝第4次; 每日重置; 疲劳降频逻辑
- `test_api_routes.py` — 每个端点: 200 正常 + 错误 + 边界
- `test_api_websocket.py` — WS 连接 / ping-pong / 消息帧解析 / 断开重连
- `test_infra_heartbeat.py` — 相同 app 的 pulsetime 窗口心跳合并; 不同 app 不合并
- `test_infra_migration.py` — 已知迁移的 up/downgrade 在 SQLite :memory: 中来回
- `performance/` — 验证 NF-P1..P5

**性能 NF 的 CI 门禁映射（非仅 debug 指标）**: NF-P2（API p95）、NF-P3（tick ≤50ms）、NF-P5（备份 ≤5s）由 `tests/performance/` 中的 pytest-benchmark 用例在 CI 强制验证，超标即失败；NF-P1（CPU）与 NF-P6（内存）依赖真实桌面环境，在 Wave 9 的发布检查清单中用脚本实测（psutil 采样 10 分钟），结果写入发布记录。

---

## 12. 实现波次 (Phase 5 预览)

| 波次 | 模块 | 依赖 | 验收标准 |
|------|------|------|---------|
| **Wave 1** 基础设施 | config, database, repositories, migration, logging | 无 | AsyncEngine 启动, WAL 配置, token 生成, loguru, Alembic 自动迁移, platformdirs 路径 |
| **Wave 2** 数据层 | domain/events, domain/features, domain/baseline, domain/deviation, domain/labeling | Wave 1（pyproject/logging 就绪） | 事件序列化/反序列化, 专注分数, Welford/Z-score, 6 信号 consensus, 纯函数单元测试覆盖 ≥80%（对齐 NF-Q1） |
| **Wave 3** 采集器 | infrastructure/collectors/*, services/collector_service, EventBus, heartbeat merge | Wave 2 | Win32 采集器采集真实窗口, EventBus 写入, heartbeat pulsetime 合并, 空闲检测 |
| **Wave 4** API+通知 | api/ (全部), middleware (auth/host/ratelimit/logging), websocket, infrastructure/notification | Wave 2（框架部分不依赖 Wave 3） | **两段验收**: ①框架+中间件+静态端点（不依赖采集器）②采集器绑定端点（collector/ws）在 Wave 3 完成后补验; RFC 9457, OpenAPI, WS 消息帧, 通知服务降级链 |
| **Wave 5** 报告 | services/report_service, services/analysis_service, daily/weekly 报告, **EventCleanup 调度任务** | Wave 2 | 日报生成幂等, 周报, 趋势 API, 30 天数据清理任务分批执行 |
| **Wave 6** LLM | infrastructure/llm/*, services/llm_service, domain/procrastination, domain/cbt_techniques | Wave 5 | 行为摘要生成, DeepSeek API 集成, Pydantic 约束解码, 规则引擎兜底 |
| **Wave 7** 干预 | services/intervention_service, throttle, crisis_detector, intervention_log | Wave 6 | 干预生成, 节流, 深度工作不打扰, 疲劳降频, 危机检测 |
| **Wave 8** ML 训练+导出+打包 | train/pipeline, export, packaging | Wave 2+5 | 训练 CLI, 合成数据生成, CSV/JSON 导出, PyInstaller spec, tufup 更新 |
| **Wave 9** 全栈集成 | E2E QA, 性能验证, 文档更新, 安全审计 | 全部 | Gate 5 最终验收 |
