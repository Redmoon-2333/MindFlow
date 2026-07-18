# 第1章 项目总览与架构设计

## 1.1 项目定位

**MindFlow** 是一个本地优先的智能专注力追踪与抗拖延助手。它运行在用户的个人电脑上，通过周期性采集活跃窗口信息，分析用户的行为模式，并在检测到拖延倾向时生成个性化的认知行为干预。MindFlow 的核心设计原则是 **高性能 > 高可用 > 高专业度**，始终在本地运行，不依赖云端服务——所有行为数据存储在用户的 SQLite 文件中，LLM 归因分析采用三层降级链（DeepSeek API → Ollama 本地模型 → 规则引擎），确保核心功能在任何网络条件下都可工作。

MindFlow 不是 SAAS 产品，而是一个 PyInstaller 打包的单进程桌面应用。它服务于希望了解自己注意力分布、减少无意识拖延的个人用户。

## 1.2 架构全景

### 1.2.1 三层架构

MindFlow 采用经典的**分层架构**，依赖方向严格单向：Presentation → Application → Domain ← Infrastructure。

```
┌─────────────────────────────────────────┐
│  Presentation 层 (api/)                  │
│  routes/ · websocket.py · middleware/    │
│  → FastAPI endpoints + WS handler        │
│  → 依赖: services, domain                │
├─────────────────────────────────────────┤
│  Application 层 (services/)              │
│  collector_service · analysis_service   │
│  llm_service · intervention_service     │
│  → 业务编排, 无框架依赖                   │
│  → 依赖: domain, infrastructure          │
├─────────────────────────────────────────┤
│  Domain 层 (domain/)                     │
│  events.py · procrastination.py         │
│  baseline.py · deviation.py · features  │
│  → 纯 Python, 零外部依赖                  │
│  → 依赖: 无                              │
├─────────────────────────────────────────┤
│  Infrastructure 层 (infrastructure/)     │
│  database.py · repositories/            │
│  collectors/ · llm/ · config/           │
│  → SQLAlchemy/APScheduler/pywin32 适配   │
│  → 依赖: 外部库 + domain                 │
└─────────────────────────────────────────┘
```

**依赖方向说明**：箭头 `→` 表示"依赖"。Domain 位于底层中心，不依赖任何其他层；Infrastructure 依赖 Domain（实现其接口）；Services 编排 Domain 和 Infrastructure；API 是入口，依赖于 Services。

### 1.2.2 进程与部署拓扑

```
┌──────────────────────────────────────────────────────────────┐
│                MindFlow Desktop App (单进程)                    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  uvicorn.Server (asyncio event loop)                  │    │
│  │                                                        │    │
│  │  ┌──────────────┐  ┌──────────────┐                   │    │
│  │  │  FastAPI App  │  │  WebSocket    │                   │    │
│  │  │  (REST :8765) │  │  /api/v1/ws   │                   │    │
│  │  └──────┬───────┘  └──────┬───────┘                   │    │
│  │         │                  │                            │    │
│  │  ┌──────┴──────────────────┴───────┐                   │    │
│  │  │       Dependency Injection       │                   │    │
│  │  │  (FastAPI Depends / 接口协议)     │                   │    │
│  │  └──────┬──────────────────┬───────┘                   │    │
│  │         │                  │                            │    │
│  │  ┌──────┴──────┐  ┌───────┴────────┐                  │    │
│  │  │  Services    │  │  Repositories   │                  │    │
│  │  │  (业务逻辑)   │  │  (数据访问)      │                  │    │
│  │  └──────┬──────┘  └───────┬────────┘                  │    │
│  │         │                  │                            │    │
│  │  ┌──────┴──────────────────┴───────┐                   │    │
│  │  │  SQLAlchemy AsyncEngine (aiosqlite) │                │    │
│  │  │  SQLite WAL 模式                  │                   │    │
│  │  └──────────────────────────────────┘                   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  CollectorService (同进程 asyncio task)                │    │
│  │  ┌────────────┐ ┌──────────┐ ┌────────────┐         │    │
│  │  │ WinTracker │ │MacTracker│ │X11Tracker  │         │    │
│  │  │ (win32gui) │ │ (pyobjc) │ │(python-xlib)│        │    │
│  │  └────────────┘ └──────────┘ └────────────┘         │    │
│  │         ↓ (EventCollector protocol)                   │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  asyncio task: 5s tick 循环 (while+sleep)      │    │    │
│  │  │  → EventBus (asyncio.Queue)                   │    │    │
│  │  │  → EventWriter → SQLite (append-mostly)        │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  LLM Pipeline (可选, 三层降级)                         │    │
│  │  DeepSeek API → Ollama(local) → RuleEngine           │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Watchdog (崩溃恢复循环)                               │    │
│  │  最多 3 次重启/小时, 线性 backoff                       │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

**关键设计选择**:

- **同进程异步隔离**：采集器作为独立 asyncio task 运行在 API 进程内，通过 EventBus（asyncio.Queue）通信，不与 API 请求路径共享可变状态。对比 ActivityWatch 的多进程方案，降低了约 300 行 IPC 代码。
- **依赖注入**：所有 Service 和 Repository 通过 FastAPI `Depends()` 获取实例。废弃旧代码中 `from scheduler import collector` 的全局单例模式。
- **单进程打包**：最终产物是 PyInstaller 打包的单可执行文件，watchdog 进程覆盖崩溃恢复场景。

### 1.2.3 三层 LLM 降级链

LLM 归因分析是 MindFlow 的差异化特性，但其可用性不能依赖网络或第三方 API。三层降级保证核心功能永远可用：

```
L1: DeepSeek API (主路径, 期望 95% 请求)
  → 成功: 返回 LLM 归因结果
  → 失败: 降级到 L2

L2: Ollama 本地模型 (可选, 零成本)
  → 成功: 返回本地模型归因结果
  → 失败: 降级到 L3

L3: RuleEngine (兜底, ¥0, 永远可用)
  → 基于 TMT 5 类可计算规则, 输出结构与 LLM 一致
```

## 1.3 技术栈

| 层 | 技术 | 版本要求 | 用途 |
|----|------|---------|------|
| 语言 | Python | ≥3.11 | 运行时 |
| Web 框架 | FastAPI (async) | ≥0.115 | REST + WebSocket |
| ASGI 服务器 | uvicorn (编程式启动) | ≥0.30 | 事件循环主体 |
| ORM | SQLAlchemy 2.0 (asyncio + aiosqlite) | ≥2.0 | 异步数据库访问 |
| 数据库 | SQLite WAL | 系统自带 (≥3.35) | 数据持久化 |
| 迁移 | Alembic | ≥1.13 | Schema 版本管理 |
| 任务调度 | APScheduler 3.x (AsyncIOScheduler) | ≥3.10 | 日报/清理/备份等定时任务 |
| ID 生成 | uuid6 (PyPI) | ≥2024.x | UUIDv7 时间排序 ID |
| 采集 (Windows) | pywin32, psutil, win32gui | latest | 活跃窗口轮询 |
| 采集 (macOS) | pyobjc-framework-Cocoa, Quartz | latest | 活跃窗口轮询 |
| 采集 (Linux/X11) | python-xlib | ≥0.33 | 活跃窗口轮询 |
| ML | scikit-learn, hmmlearn, pandas | ≥1.5 | 行为分析模型 |
| LLM | httpx + openai SDK (DeepSeek 兼容) | latest | AI 归因调用 |
| 约束解码 | instructor | latest | Pydantic 格式 LLM 输出 |
| 数据验证 | Pydantic | ≥2.0 | Schema 校验 |
| 日志 | loguru | ≥0.7 | 结构化日志 |
| 性能指标 | prometheus_client | latest | Debug 模式 /metrics |
| 崩溃上报 | sentry-sdk | latest | Opt-in 崩溃数据收集 |
| 打包 | PyInstaller + tufup | ≥6.0 | 桌面应用分发和更新 |
| 测试 | pytest + pytest-asyncio + pytest-cov + hypothesis | latest | 全量测试 |
| 代码质量 | ruff + mypy (strict) | latest | 静态检查 |
| 路径管理 | platformdirs | ≥4.0 | 跨平台数据目录 |

## 1.4 目录结构

以下是 `backend-next/src/` 下的真实目录树：

```
mindflow-app/backend-next/
├── pyproject.toml                  # 项目配置 (依赖 / ruff / mypy / pytest)
├── alembic.ini                     # Alembic 迁移配置
├── alembic/                        # Alembic 迁移脚本
│   ├── env.py                      # render_as_batch=True (SQLite ALTER COLUMN 限制)
│   └── versions/
├── mindflow.spec                   # PyInstaller 打包配置
└── src/
    ├── main.py                     # 入口: watchdog + uvicorn.Server 编程式启动
    ├── config.py                   # Pydantic BaseSettings (多源: env / .env / 默认值)
    ├── logging_config.py           # loguru 双通道配置 (文本 + JSON)
    ├── api/
    │   ├── router.py               # 路由注册 + 全局异常处理
    │   ├── websocket.py            # WS 连接管理 + 消息广播
    │   ├── middleware/
    │   │   ├── auth.py             # Bearer token 验证
    │   │   ├── host.py             # Host header 校验 (仅 localhost)
    │   │   ├── ratelimit.py        # 令牌桶速率限制
    │   │   └── logging.py          # 结构化请求日志 (request_id + timing)
    │   └── routes/
    │       ├── health.py           # 健康检查
    │       ├── collector.py        # 采集器控制
    │       ├── activities.py       # 活动数据查询
    │       ├── focus.py            # 专注报告
    │       ├── analytics.py        # 行为分析 API
    │       ├── reports.py          # 日报/周报
    │       └── preferences.py      # 用户偏好
    ├── services/
    │   ├── collector_service.py    # 采集器编排
    │   ├── analysis_service.py     # 行为分析 (专注分数/会话识别)
    │   ├── llm_service.py          # LLM 归因 (三层降级链)
    │   ├── intervention_service.py # 干预引擎 (节流/深度工作检测)
    │   ├── report_service.py       # 日报/周报生成
    │   ├── panel_service.py        # 专家面板 (G003)
    │   ├── chat_service.py         # 对话式助手 (G004)
    │   ├── autonomy_service.py     # 自主行为规则 (G005)
    │   ├── effectiveness_service.py# 干预效果评估
    │   ├── evidence_service.py     # 证据包构建
    │   ├── intervention_throttle.py# 干预节流状态机
    │   ├── maintenance_service.py  # 数据清理 + 备份
    │   └── scheduler.py            # APScheduler 定时任务工厂
    ├── domain/
    │   ├── events.py               # ActivityEvent, WindowSnapshot (frozen dataclass)
    │   ├── sessions.py             # FocusSession 模型
    │   ├── features.py             # 专注分数, 应用排名 (旧代码迁移)
    │   ├── baseline.py             # Welford 在线算法
    │   ├── deviation.py            # 多维 Z-score 偏差检测
    │   ├── procrastination.py      # TMT 标签模型 + 分类逻辑
    │   ├── labeling.py             # 6 信号 Consensus Labeler
    │   ├── cbt_techniques.py       # CBT 技术枚举 + 匹配映射
    │   └── ids.py                  # UUIDv7 ID 生成
    └── infrastructure/
        ├── database.py             # AsyncEngine + SessionLocal + WAL PRAGMA
        ├── migrations.py           # Alembic 异步迁移包装
        ├── notification.py         # 通知服务 + 平台实现 + LogOnly 降级
        ├── repositories/
        │   ├── base.py             # Repository 协议基类
        │   ├── activity.py         # ActivityEvent + heartbeat 合并
        │   ├── focus.py            # FocusSession 查询
        │   ├── report.py           # DailyReport 持久化
        │   ├── analysis.py         # ProcrastinationAnalysis 存储
        │   ├── preferences.py      # 用户偏好 KV
        │   └── intervention.py     # 干预日志
        ├── collectors/
        │   ├── base.py             # EventCollector Protocol
        │   ├── win32.py            # Windows 采集 (<200 行)
        │   ├── darwin.py           # macOS 采集 (<200 行)
        │   ├── x11.py              # Linux X11 采集 (<200 行)
        │   └── wayland_fallback.py # Wayland 降级 (pid 级)
        ├── llm/
        │   ├── client.py           # DeepSeek API 客户端 (httpx async)
        │   └── rule_engine.py      # 兜底规则引擎
        └── security/
            ├── token_manager.py    # Token 生成/读取/验证
            └── crisis_detector.py  # 独立危机检测 (独立于 LLM)
```

## 1.5 关键代码

以下三段代码展示了 MindFlow 最核心的运行机制。

### 1.5.1 应用启动与关闭 (lifespan)

**来源**: `src/mindflow/app.py:89-377`

`_lifespan` 协程是 MindFlow 的启动和关闭生命周期管理函数。启动时按顺序装配数据库迁移、完整性检查、认证 Token、仓储实例、采集器、各业务服务、调度器；关闭时按逆序清理：

```python
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup initialisation, shutdown cleanup."""

    # ── Extract settings ─────────────────────────────────
    settings: Settings = app.state.settings
    data_dir = Path(platformdirs.user_data_dir("mindflow", ensure_exists=True))
    token_path = data_dir / "token"

    # ── Database engine ─────────────────────────────────
    engine = create_engine(settings.db_url)
    session_factory = create_session_factory(engine)

    # ── 1. Migrations (graceful on failure) ──────────────
    migration_applied = await run_migrations(settings.db_url)

    # ── 2. Integrity check ──────────────────────────────
    db_ok = await integrity_check(engine)

    # ── 3. Auth token ───────────────────────────────────
    system_token = load_or_create_token(token_path)

    # ── 4. Repositories ─────────────────────────────────
    activity_repository = SQLAlchemyActivityRepository(
        session_factory=session_factory,
        pulsetime_s=settings.heartbeat_pulsetime_s,
    )
    preferences_repository = PreferencesRepository(session_factory=session_factory)
    focus_repository = SQLAlchemyFocusSessionRepository(session_factory=session_factory)
    report_repository = SQLAlchemyDailyReportRepository(session_factory=session_factory)
    analysis_repository = SQLAlchemyProcrastinationAnalysisRepository(
        session_factory=session_factory,
    )

    # ── 5. Collector (not started yet — caller must start) ──
    collector = create_collector()
    collector_service = CollectorService(
        collector=collector,
        repository=activity_repository,
        interval_s=float(settings.collect_interval_s),
    )

    # ── 6..7c. Notifier, Analysis, Report, LLM, Intervention, Panel, Chat, Autonomy
    #     (各 Service 的依赖注入依次展开)

    # ── 8. Scheduler (Wave 5 cron jobs) ─────────────────
    scheduler = build_scheduler(...)
    scheduler.start()

    # ── Inject into app.state ────────────────────────────
    app.state.engine = engine
    app.state.collector_service = collector_service
    app.state.system_token = system_token
    app.state.scheduler = scheduler
    # ... 所有 Service 注入 ...

    yield  # ── Application runs here ──

    # ── Graceful shutdown (REVERSE ORDER) ────────────────

    # 1. Stop scheduler
    scheduler.shutdown(wait=False)

    # 2. Close WebSocket connections
    await close_all_connections()

    # 3. Stop collector (3s timeout)
    await asyncio.wait_for(collector_service.stop(), timeout=3.0)

    # 4. Dispose engine (3s timeout)
    await asyncio.wait_for(engine.dispose(), timeout=3.0)
```

**解析**：启动阶段按优先级装配组件——数据库是最底层依赖，最先初始化；Scheduler 依赖所有 Services，最后启动。关闭时严格逆序，每个清理步骤有超时保护，确保不因个别步骤挂起而阻塞进程退出。lifespan 是 FastAPI 的 ASGI 生命周期协议，Python 3.10+ 的 `yield` 在 async context manager 中天然区分启动和关闭阶段。

### 1.5.2 Watchdog 崩溃恢复循环

**来源**: `src/mindflow/main.py:30-103`

`Watchdog` 类封装了 uvicorn 服务器编程式启动和崩溃恢复逻辑。当服务器因未捕获异常退出时，watchdog 自动重启，但不超过每小时 3 次（NF-R1 可靠性的具体实现）：

```python
class Watchdog:
    """Monitors the uvicorn server and restarts on crash (NF-R1)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_restarts: int = 3,   # 每小时最多 3 次
        window_s: float = 3600.0,
    ) -> None:
        self._host = host
        self._port = port
        self._max_restarts = max_restarts
        self._window_s = window_s
        self._crash_times: list[float] = []   # 滚动窗口记录

    async def run_forever(self) -> None:
        while True:
            app = create_app(get_settings())
            config = Config(
                app=app,
                host=self._host,
                port=self._port,
                log_level="info",
                access_log=False,  # 避免 WS token 泄露到日志
            )
            server = Server(config)

            try:
                await server.serve()
            except Exception as exc:
                logger.opt(exception=True).error("Server crashed: {}", exc)
            else:
                logger.info("Server stopped cleanly")

            if not self._should_restart():
                break

            wait = self._backoff_delay()
            logger.info("Restarting in {:.0f}s...", wait)
            await asyncio.sleep(wait)

    def _should_restart(self) -> bool:
        """Crash-loop detection: max 3 restarts per rolling hour."""
        now = time.time()
        # 剔除窗口外的记录
        self._crash_times = [t for t in self._crash_times if now - t < self._window_s]
        return not len(self._crash_times) >= self._max_restarts

    def _backoff_delay(self) -> float:
        """Linear backoff: 0.5s → 1s → 2s → 3s ... capped at 5s."""
        count = len(self._crash_times)
        return min(1.0 * count, 5.0) if count else 0.5
```

**解析**：`Watchdog` 在无限循环中反复创建 uvicorn `Server` 并 `await server.serve()`。当 `serve()` 因异常返回后，记录崩溃时间戳到滚动列表，检查 1 小时窗口内崩满 3 次则停止。`_backoff_delay()` 实现线性 backoff，避免反复快速重启耗尽系统资源。`main()` 入口使用 `asyncio.wait` 同时等待 watchdog task 和 shutdown signal，配合 `add_signal_handler` 实现 SIGINT/SIGTERM 的优雅退出。

### 1.5.3 配置模型 (Settings)

**来源**: `src/mindflow/config.py:60-113`

配置系统基于 Pydantic `BaseSettings`，多源优先级：环境变量 > `.env` 文件 > 默认值。所有配置项集中管理，类型安全：

```python
class Settings(BaseSettings):
    """Application-wide settings.
    Priority: env vars (MINDFLOW_*) > .env file > defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="MINDFLOW_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- Database ---
    db_url: str = Field(
        default="sqlite+aiosqlite:///{data_dir}/mindflow.db",
        description="SQLAlchemy async database URL",
    )

    # --- Server ---
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8765)

    # --- Collector ---
    collect_interval_s: int = Field(
        default=5, ge=1, le=60,
        description="Collector tick interval in seconds",
    )
    heartbeat_pulsetime_s: int = Field(
        default=10, ge=1, le=300,
        description="Heartbeat merge window in seconds",
    )

    # --- Data Retention ---
    event_retention_days: int = Field(
        default=30, description="Raw event retention (7-90)",
    )

    @field_validator("event_retention_days")
    @classmethod
    def _validate_retention(cls, v: int) -> int:
        if not 7 <= v <= 90:
            raise ValueError(f"event_retention_days must be 7-90, got {v}")
        return v

    # --- Sub-settings (嵌套) ---
    log: LogSettings = Field(default_factory=LogSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @model_validator(mode="after")
    def _resolve_db_url(self) -> "Settings":
        """Resolve {data_dir} placeholder in db_url."""
        if "{data_dir}" in self.db_url:
            self.db_url = self.db_url.format(data_dir=_get_data_dir())
        return self
```

**解析**：`Settings` 是应用配置的唯一入口。`env_prefix="MINDFLOW_"` 使环境变量 `MINDFLOW_HOST`、`MINDFLOW_PORT` 等自动覆盖默认值。`db_url` 使用模板字符串 `{data_dir}`，在 `model_validator` 中解析为用户的平台数据目录（Windows 上为 `%APPDATA%/mindflow`，macOS 为 `~/Library/Application Support/mindflow`）。嵌套的 `LogSettings` 和 `LLMSettings` 子模型让配置结构清晰，不混在同一个平铺命名空间。`field_validator` 保障 `event_retention_days` 被限制在 7-90 天的合理范围。

## 1.6 快速开始

以下命令在 `backend-next/` 目录下执行。

```bash
# 1. 激活 conda 环境
conda activate mindflow

# 2. 安装依赖
pip install -e .

# 3. 启动 MindFlow 后端 (watchdog + uvicorn)
python -m mindflow.main

# 4. (另一个终端) 健康检查
curl http://127.0.0.1:8765/api/v1/health

# 5. 浏览器打开 API 文档
open http://127.0.0.1:8765/docs   # macOS
start http://127.0.0.1:8765/docs  # Windows
```

启动后你将看到以下日志输出（省略部分细节）：

```
2026-07-18T10:00:00.123 | INFO     | mindflow.app:setup_logging:...
2026-07-18T10:00:00.456 | INFO     | mindflow.main:run_forever:67 | Starting MindFlow watchdog (max 3 restarts/hour)
2026-07-18T10:00:00.789 | INFO     | mindflow.config:_get_data_dir:56 | Data directory: .../mindflow
2026-07-18T10:00:01.012 | INFO     | mindflow.infrastructure.database:integrity_check:... | Database integrity check passed
2026-07-18T10:00:01.234 | INFO     | mindflow.app:_lifespan:... | CollectorService created (not started)
2026-07-18T10:00:01.567 | INFO     | mindflow.app:_lifespan:... | MindFlow v2.0.0-alpha startup complete
2026-07-18T10:00:01.890 | INFO     | uvicorn.server:serve:... | Uvicorn running on http://127.0.0.1:8765
```

此时服务器已就绪，任何端点调用或 WebSocket 连接都会触发按需初始化。

## 1.7 事件溯源数据模型（衔接第2章）

MindFlow 的核心数据模型选用 **Event Sourcing（事件溯源）** 而非传统 CRUD，原因如下：

- **精度**：旧代码的 `duration_seconds` 用配置的采集间隔估算，偏差大。事件流保留原始 tick 数据，duration 从相邻事件时间戳精确计算。
- **灵活性**：合并在查询时配置，不丢失分辨率——同一个事件流可以适配不同的聚合策略。
- **可靠性**：事件流是 **append-mostly** 的：常规仅追加，唯一例外是 heartbeat 合并（对最近一行的原子 UPDATE）。这是对 append-only 语义的明确让步，换取 90%+ 的磁盘写削减（ActivityWatch 实践验证）。

核心的 Event 模型是一个 frozen dataclass：

```python
@dataclass(frozen=True)
class ActivityEvent:
    """An immutable activity event in the append-mostly event stream."""
    id: str                       # UUIDv7 (时间排序)
    user_id: int
    timestamp_utc: datetime       # 时区感知 UTC
    duration_s: float             # 距上一个事件的实测间隔
    event_type: EventType         # window_snapshot | idle_change | manual_tag
    data: WindowSnapshot          # 窗口快照
```

Layer-by-layer 理解，Domain 层定义了纯净的数据模型（零外部依赖），Infrastructure 层通过 Repository 协议实现对 SQLite 的读写，Service 层编排业务逻辑，API 层对外暴露 REST 端点和 WebSocket。

**详见第2章「数据层：事件溯源与存储设计」**，覆盖全部 Schema 定义、Repository 接口和 heartbeat 合并算法。

## 1.8 架构决策记录（ADR）

本章涉及的主要架构决策已在 `04-architecture-design.md` 中归档为 ADR。以下是快速索引：

| ADR | 决策 | 关键权衡 |
|-----|------|---------|
| ADR-001 | Event Sourcing (append-mostly) | 放弃纯不可变语义，换取 90%+ 写削减 |
| ADR-002 | 同进程 + asyncio task 采集器 | 减少 ~300 行 IPC 代码，watchdog 兜底崩溃 |
| ADR-003 | 三层 LLM 降级链 | 规则引擎 ¥0 永远可用，作为 LLM 功能的可靠性基底 |
| ADR-004 | localhost Token 文件认证 | 对标 KeePassXC，文件权限即安全边界 |
| ADR-005 | SQLite WAL 而非 PostgreSQL | 零配置单文件，WAL 解决并发读写 |
| ADR-006 | uuid6 库提供 UUIDv7 | 保持 Python 3.11 兼容，降低环境迁移成本 |
| ADR-007 | 采集 tick 用裸 asyncio 循环 | tick 不需要 cron/coalesce 语义，少一层调度抽象 |

---

> **下一章**: [第2章 数据层：事件溯源与存储设计](ch2-data-layer.md)
