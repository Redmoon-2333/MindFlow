# MindFlow 后端需求规格文档

> **文档编号**: 03-requirements.md
> **版本**: v2.1-reviewed
> **日期**: 2026-07-17
> **评审**: critic agent（独立），审阅结论 ACCEPT-WITH-MINOR；7 项修复已全部应用
> **作者**: 编排者，基于 Gate 1 全部研究成果 + 立项书申报书 + design-spec v0.1.0
> **状态**: Gate 2 验收材料（就绪）
> **设计原则**: 高性能 > 高可用 > 高专业度（不计搭建成本，敢于做大改）
> **不作要求**: 付费墙、订阅、多租户、云同步 — 均按需预留扩展点、本期不实现
> **评审修复记录**: [见 §0](#0-评审修复记录)

---

## 0. 评审修复记录

| # | 等级 | 问题 | 修复 |
|---|------|------|------|
| C1 | CRITICAL | M9 "失败降级运行" vs NF-R5 "阻塞启动" 矛盾 | NF-R5 对齐 M9：降级运行，以旧 schema 启动，health endpoint 暴露迁移失败状态 |
| M1 | MAJOR | NF-S3 "零云数据上传" 与 LLM API 矛盾 | NF-S3 重写：原始数据全本地，LLM 发送仅限聚合摘要，新增 LLM 脱敏要求 NF-S3a |
| M2 | MAJOR | 无 WebSocket 消息契约 | 新增 §4.3：消息框架 + 6 种事件类型 + payload schema + 前端重连协议 |
| M3 | MAJOR | 速率限制无参数 | 新增 §4.4：attribution 令牌桶(1/30s, 20/日硬上限) + intervention 引用 C3 节流 + 429 响应头 |
| M4 | MAJOR | 数据留存 "关闭长尾" 模糊 | 明确：原始事件保留 30 天（7-90 可配置）→ 批量删除；聚合报告永久保留 |
| M5 | MAJOR | NF-R2 "失败隔离" vs 同进程架构矛盾 | 重写为：API 请求完成不受影响，降级响应，watchdog 恢复采集 |
| M6 | MAJOR | LLM API Key 管理未定义 | 新增 §4.5：config.toml 存储、内置 key + BYOK 混合、即时生效 |
| M7 | MAJOR | WebSocket 路径不统一 | `/ws` → `/api/v1/ws` |
| m2 | MINOR | CI "过期" 笔误 | 修正为 "必须通过" |
| m3 | MINOR | 缺数据导出 | 新增 S7 + 端点 GET /api/v1/export |
| m5 | MINOR | 备份失败无告警 | 新增 NF-R7：失败日志 + 系统通知 |
| — | — | APScheduler 4.x 风险 | 技术栈表改为 `3.x (AsyncIOScheduler) 或 4.x` |

## 1. 功能需求 — MoSCoW 分级

### 1.1 Must（本期必须交付，对应 2027.5 结题下限）

M1 ⬜ **数据采集** — 跨平台(Windows+macOS+X11 Linux)无感采集，5 秒 tick + 事件驱动切换；启用/暂停/恢复；空闲检测可配置

M2 ⬜ **实时分析** — 专注分数(0-100，可配置权重)、应用使用排名、专注会话识别、窗口切换频率

M3 ⬜ **REST API + WebSocket** — /api/v1/* 完整端点，含 OpenAPI 契约完备；WebSocket 实时推送当前活动+专注状态变更；RFC 9457 Problem Details 错误格式

M4 ⬜ **多数据库后端** — SQLite 生产模式(WAL/busy-timeout/WAL-Journal 尺寸限制)、启动完整性检测、每日备份(VACUUM INTO)，连接 URL 可配置

M5 ⬜ **localhost API 三层安全** — 随机 token 认证 + Host header 校验 + 数据隔离(platformdirs)

M6 ⬜ **结构化日志与可观测性** — loguru + 每文件 10MB 轮转 + 30 天保留 + 压缩；崩溃自动重启

M7 ⬜ **桌面打包** — PyInstaller 打包为跨平台可执行文件

M8 ⬜ **打包后自动启动** — uvicorn.Server 编程式启动 + 主进程 watchdog(每小时最多 3 次)

M9 ⬜ **数据库迁移** — Alembic 启动时自动升级、失败降级运行

### 1.2 Should（行为建模核心，结题前应交付）

S1 ⬜ **用户行为基线** — Welford 在线算法累计统计 + JSON 持久化

S2 ⬜ **多维度偏差检测**(Z-score) — 可配置严重度（警告/注意/风险）

S3 ⬜ **ML 模式识别** — HMM 行为状态推断 + 时序聚类 + 标签共识模型，hmmlearn 不可用时降级 Markov 链

S4 ⬜ **分心模式识别** — 频繁模式挖掘 + 触发应用分类 + 时段热力分布

S5 ⬜ **日报/周报** — 专注趋势/应用使用/模式摘要/评分变化

S6 ⬜ **行为画像** — 个人工作习惯、专注高峰时段、分心触发因素
S7 ⬜ **数据导出** — 导出用户全部行为数据（CSV + JSON）、聚合报告（PDF 可选项）、支持按日期范围过滤

### 1.3 Could（LLM 与干预，差异化核心）

C1 ⬜ **LLM 归因分析** — 行为摘要 JSON → 5 类拖延标签(任务畏惧型/冲动分心型/决策困难型/完美主义型/情绪调节型) + 置信度 + CBT 归因文本

C2 ⬜ **个性化干预** — 任务拆解(5 分钟微任务) + 环境优化(适时机温和提醒) + 智能排序(截止日期+精力状态→任务推荐)；干预强度可调(温和/标准/严格)

C3 ⬜ **自适应节流** — 每天最多 3 次主动推送，2 小时间隔，同类干预每天最多 2 次，深度工作零打扰，7 天忽略率 >60% 自动降频

C4 ⬜ **对话式反思** — 可追问归因推理，每日反思提示，行为改善建议

C5 ⬜ **效果评估** — 干预前后操作变化对比 + 用户满意度评分

### 1.4 Won't（本期明确不做）

W1 ⬜ 云同步/多设备账号
W2 ⬜ 团队版/自习室(B2B)
W3 ⬜ 付费墙/订阅(License Key)
W4 ⬜ 浏览器扩展 Tab 追踪(可通过事件驱动架构预留)
W5 ⬜ Wayland 原生支持（降级到 pid 采集 + 文档注明）
W6 ⬜ 完整国际界面(i18n)

---

## 2. 非功能需求 — 与可测量验收标准

### 2.1 性能

NF-P1: 采集器空闲 CPU ≤ **2%**，运行态均值 ≤ **5%**（目标：2%），峰值 ≤ **10%**
NF-P2: API p95 < **50ms**(单用户本地)，p99 < **100ms**
NF-P3: 采集 tick ≤ **50ms** 执行时间（包含 DB 写入）
NF-P4: SQLite WAL：写入性能提升 ≥ **2x**（基线：sync 模式），journal_size_limit = **64MB**
NF-P5: 每日 VACUUM INTO 备份 ≤ **5 秒**
NF-P6: 全量后端进程空闲内存 ≤ **400MB**（含 sklearn/hmmlearn 库加载），采集器 ≤ **100MB**（不含 ML 库）

### 2.2 可靠性

NF-R1: 崩溃自动重启 < **5 秒**，每小时最多 3 次，历史 ≤ **20 次**（防死循环）
NF-R2: 采集器崩溃不影响正在进行的 API 请求完成；采集器重启期间 API 返回降级响应（`collector_unavailable` 状态），API 服务全程不中断，watchdog 在 NF-R1 时限内恢复采集
NF-R3: LLM API 不可用时行为分析、基本干预全部可用（三层降级链）
NF-R4: 数据零丢失：关闭时所有活跃事件落地，优雅关闭 < **5 秒**完成
NF-R5: 迁移失败**降级运行**（日志错误 + 用户通知），不阻塞启动。以旧 schema 启动，标记迁移失败状态在 health endpoint 可查。桌面应用不应因 schema 迁移问题拒绝启动（参照 engineering research §1.3 建议）
NF-R6: 每日自动 VACUUM INTO 备份 + 启动时完整性检查(`PRAGMA integrity_check`) + 试恢复
NF-R7: 备份失败（磁盘满/权限拒绝/损坏）写入结构化日志 + 触发系统通知（不阻塞主流程）

### 2.3 安全与隐私

NF-S1: 所有 API 调用需要 `Authorization: Bearer <random_token>`（否则 401）
NF-S2: Host header 校验 — 非 localhost/127.0.0.1/[::1] 的请求 403
NF-S3: 原始行为数据（含窗口标题、应用名、时间戳）**全部本地存储，不上传云端**。发送至 LLM API 的仅限于聚合行为摘要（不含窗口标题原文、文件路径、用户名、主机名），用户首次启用 LLM 时需确认数据发送范围。用户可随时关闭 LLM 功能，关闭后零 API 调用。NF-S3a: LLM 发送内容需自动脱敏：过滤文件路径、IP 地址、主机名、个人邮箱
NF-S4: 输出脱敏：API 响应不暴露文件路径/密钥/本地系统信息
NF-S5: 独立于 LLM 的危机检测（自杀/自伤关键词+即时热线+对话中断+日志记录）
NF-S6: 首次使用时必须确认免责声明后才能启用 LLM 功能
NF-S7: 禁止输出任何"诊断/治疗/心理干预/患者/处方"字样 — 自动化过滤 + LLM prompt 限定的双重保险

### 2.4 代码质量与测试

NF-Q1: 核心域（分析/干预/标签）覆盖率 ≥ **80%**；总体覆盖率 ≥ **70%**
NF-Q2: 每个端点 ≥ **1 success + 1 error + 1 edge-case 测试**
NF-Q3: 新代码为 `mypy --strict` 兼容（逐步累积至全项目）
NF-Q4: `ruff check` + `ruff format --check` + `bandit` 包含在 CI 中
NF-Q5: GitHub Actions CI — 每次 push 触发完整构建

### 2.5 架构质量

NF-A1: 所有平台相关的采集代码 `< 200` 行（薄采集层模式）
NF-A2: 核心业务逻辑**零平台依赖** — 无 `if sys.platform == 'win32'` 分支
NF-A3: 互斥依赖无循环导入 — 强制执行 DAG 拓扑排序
NF-A4: 采集器、分析器、LLM 接口、干预引擎都可**接口交换**（依赖注入协议）
NF-A5: 数据库连接 URL 可配置（无需重新打包即可切换 SQLite/PostgreSQL）

---

## 3. 领域模型

### 3.1 核心实体与边界

```
User (1) ── (N) ActivityEvent ──aggregated─→ FocusSession
                              ──aggregated─→ DailyReport
                              ──aggregated─→ WeeklyReport

ActivityEvent (1) ── (N) WindowSnapshot

BaselineModel (per-user, 在线更新)

ProcrastinationAnalysis (N:1) ←→ ActivityEvent[]
InterventionLog ──triggered─→ ActivityEvent (反向关联)
```

### 3.2 数据存储策略

- 主存储：SQLite WAL，每个用户的 `.db` 文件
- 事件存储：append-only `events` 表。原始 ActivityEvent 保留 30 天（可配置 7-90 天），到期后批量删除（调度任务每日执行）。聚合结果 DailyReport/WeeklyReport/BaselineModel 永久保留
- 分析结果：每次分析缓存至 `analyses` 表（去重/幂等键 = user_id + date + analysis_type）

### 3.3 ActivityEvent Schema（核心统一模型）

```json
{
  "id": "uuid",
  "user_id": 1,
  "timestamp": "2026-07-17T10:00:00+00:00",
  "duration_s": 5.0,
  "data": {
    "app_name": "Code.exe",
    "window_title": "main.py - VS Code",
    "is_idle": false,
    "process_name": "Code.exe",
    "url": null
  }
}
```

### 3.4 拖延类型标签模型

| 类型 | 规则条件（规则引擎兜底） | CBT 技术 |
|------|------------------------|---------|
| task_aversion | 任务难度评分高 + 自我效能充足 + 故意回避 | 暴露 + 渐进式任务分级 |
| impulsivity | 最长连续专注块 < 5 分钟 + 切换 > 12 次/小时 + 短视媒体 > 50% | 刺激控制 |
| decisional | 从启动到开始 > 30 分钟 + 启动后恢复正常 | 目标设定 + 截止日期突显 |
| perfectionism | 含"不够好/重来/失败"回避模式 + 反复重做 | 认知重构 |
| emotional_regulation | 社交媒体 > 55% + 任务切换前的回避延迟 + 频繁访问短视频/游戏 | 正念 + 即时奖赏链 |

---

## 4. API 契约摘要

### 4.1 端点（完整 OpenAPI 在实现阶段自动生成）

| 方法 | 路径 | 说明 | MoSCoW |
|------|------|------|--------|
| GET | /api/v1/health | 采集器/DB 健康信息 | Must |
| GET/POST | /api/v1/collector | 采集器状态 / 启动 | Must |
| POST | /api/v1/collector/stop | 停止采集 | Must |
| GET | /api/v1/activities | 今日活动(分页+过滤) | Must |
| GET | /api/v1/activities/current | 当前活动快照 | Must |
| WS | /api/v1/ws | 实时推送(当前活动+状态变更) | Must |
| GET | /api/v1/focus | 今日专注报告 | Must |
| GET | /api/v1/focus/trend | N 日趋势 | Must |
| GET | /api/v1/analytics/patterns | 分心模式 | Should |
| GET | /api/v1/analytics/baseline | 基线模型只读查询（S1 自然延伸，04 评审补充） | Should |
| GET | /api/v1/analytics/profile | 行为画像（S6） | Should |
| GET | /api/v1/reports/daily | 日报 | Should |
| GET | /api/v1/reports/weekly | 周报 | Should |
| POST | /api/v1/analytics/attribution | LLM 归因(限流) | Could |
| POST | /api/v1/intervention/trigger | 触发干预(限流) | Could |
| GET | /api/v1/export | 数据导出(CSV/JSON, 日期范围) | Should |
| PUT/PATCH | /api/v1/preferences | 用户偏好 | Must |

### 4.2 标准错误格式(RFC 9457)

```json
{
  "type": "https://mindflow.app/errors/collector-not-running",
  "title": "Collector Not Running",
  "status": 503,
  "detail": "The data collector is currently stopped. Start it via POST /api/v1/collector/start.",
  "instance": "/api/v1/activities"
}
```

**应用错误码表**（`[type]` URI 的 `errors/` 后缀）：

| type | status | 含义 |
|------|--------|------|
| `collector-not-running` | 503 | 采集器未启动 |
| `not-found` | 404 | 资源不存在（通用） |
| `validation-error` | 422 | 请求参数验证失败 |
| `rate-limited` | 429 | 超出速率限制 |
| `auth-required` | 401 | 缺少或无效 token |
| `forbidden-host` | 403 | Host header 不被信任 |
| `internal-error` | 500 | 内部异常（不暴露堆栈） |
| `llm-unavailable` | 503 | LLM 降级到规则引擎（请求仍完成） |

用户界面文本统一为中文（`detail` 字段）。英文 URI 仅为机器可读标识符。

### 4.3 WebSocket 消息契约

所有 WebSocket 消息使用 JSON 文本帧，统一框架：

```json
{"type": "<event_type>", "payload": <object>, "timestamp": "<ISO8601 UTC>"}
```

| type | payload | 方向 | 说明 |
|------|---------|------|------|
| `activity_update` | `{app_name, window_title, is_idle, process_name}` | S→C | 当前活动窗口变更（含采集器心跳，最多 5 秒间隔） |
| `focus_change` | `{session_type: "focus"|"distraction"|"neutral", focus_score, duration_s}` | S→C | 专注状态变更 |
| `intervention` | `{intervention_id, type, message_text, cbt_technique, dismissible}` | S→C | 推送干预建议 |
| `error` | `{code, message}` | S→C | WebSocket 层错误 |
| `ping` | `{}` | C→S | 心跳（客户端每 30 秒） |
| `pong` | `{}` | S→C | 心跳响应 |

前端实现指数退避重连（1s→2s→4s→8s→max 30s + ±20% 抖动），断连期间收到的干预消息在重连后排队发送。

### 4.4 速率限制

| API 端点 | 限制方式 | 参数 |
|----------|---------|------|
| POST /api/v1/analytics/attribution | 令牌桶 + 每日硬上限 | 1 token/30s，桶容量 5；每日硬上限 20 次 |
| POST /api/v1/intervention/trigger | 引用 C3 节流参数 | 每日 3 次 + 2h 间隔 + 同类 2 次/日 |
| WebSocket activity_update | 后端推送节流 | 最快 2 秒一次（状态未变不推送） |
| 全局 API | 令牌桶 | 100 req/min（单用户本地无实际限制意义，防异常 burst） |

响应头: `X-RateLimit-Remaining`, `X-RateLimit-Reset`，超限返回 429。

### 4.5 LLM API Key 管理

- API Key 存储在 `config.toml`（platformdirs 配置目录），用户可通过设置接口更新
- 内置项目 Key（仅开发/演示场景，每月有硬上限保护预算），用户可替换为自己的 Key（BYOK）
- Key 切换通过 `PUT /api/v1/preferences` 的 `llm.api_key` 字段；变更后即时生效（会话到期前即时重载客户端配置）
- 无有效 Key 时 LLM 功能自动降级到规则引擎（来自 NF-R3 降级链的作用域）

### 4.6 安全头

- 所有响应: `X-MindFlow-Version: x.y.z`
- 所有响应: `X-Content-Type-Options: nosniff`
- Host header 校验: 仅 `localhost:8765` / `127.0.0.1:8765` / `[::1]:8765` 有效
- 认证: 所有端点（`/health` 除外）：`Authorization: Bearer <token>`（401 否则）

---

## 5. 技术组合（最终）

| 层 | 技术 | 选择理由 |
|----|------|---------|
| 语言/运行时 | Python 3.11+ | 团队技能 + ML 生态 + 可复用资产 |
| Web 框架 | FastAPI + uvicorn （全异步） | 高性能异步 + 自动 OpenAPI + WebSocket + 稳定 |
| ORM | SQLAlchemy 2.0 (async + aiosqlite) | 类型安全查询 + 异步支持 + 迁移工具 |
| 数据库 | SQLite (WAL, busy-timeout, VACUUM 备份) | 零配置 + 单文件 + 隐私友好 + 生产就绪 |
| 迁移 | Alembic (render_as_batch=True) | 版本化管理 + 启动自动迁移 |
| 任务调度 | APScheduler 3.x (AsyncIOScheduler) 或 4.x（视发布状态） | 后台定时采集/报告，异步兼容。两者 API 兼容，3.x 已验证稳定 |
| 采集 | PyWinCtl 抽象层 + win32gui（Windows）/ pyobjc（macOS）/ python-xlib（X11） | 统一 API + 薄平台层 |
| ML | scikit-learn + hmmlearn + pandas | 聚类/HMM/时间序列，降级策略内置 |
| LLM | DeepSeek API（主要）+ Ollama（可选本地） | 低成本 + 高质量 + 开源兼容 |
| 结构化输出 | Pydantic + Instructor | 约束解码 + 生产级 JSON 保证 |
| 日志 | loguru | 极简 API + JSON 输出 + 滚动 + 压缩 |
| 崩溃跟踪 | Sentry SDK（可选用户开关） | 生产级错误追踪 |
| 打包 | PyInstaller | 社区验证，sklearn 支持 |
| 自动更新 | tufup | TUF 安全协议 + 跨平台 |
| CI | GitHub Actions: pytest + mypy --strict + ruff + bandit | 自动化质量门禁 |
| 平台路径 | platformdirs | 跨平台标准数据目录 |
| 测试 | pytest + pytest-asyncio + pytest-cov + hypothesis（属性测试） | 覆盖金字塔全层 |

---

## 6. 测试策略

### 6.1 分层覆盖

| 层 | 覆盖要求 | 工具 |
|----|---------|------|
| 单元(MOC) | 核心逻辑 ≥ 80% | pytest + hypothesis（时序属性） |
| 集成 | 每个端点 3 路径(min) | pytest-asyncio + httpx.AsyncClient |
| 数据管道 | 合成数据生成器 + 实际数据集 | pandas testing + JSON goldens |
| 性能 | 采集 tick、API p95、备份耗时 | pytest-benchmark |
| ML 路径 | 基线计算、HMM 推断、标签共识 | goldens + bootstrap CI |

### 6.2 CI 测试门禁

- 必须通过：`mypy --strict`、`ruff check`、`bandit`、全量 `pytest`
- **阻塞部署**: 安全违规 > 0%
- 合并要求：覆盖率没有回归

---

## 7. 从旧代码的迁移策略

### Phase 0: 并行启动 — 原后端保持运行（不破坏它）
- 新后端放在独立目录 `mindflow-app/backend-next/`，不触及 `/backend`
- 共享：`data/datasets/`（复制源），`data/mindflow.db`（未来重新生成）
- 旧后端在新后端完全替换后才移除

### 迁移清单（保留/重新实现/放弃）

**保留（照搬到新架构）**: Welford 在线算法、多维 Z-score 偏差检测、弱监督标签共识器、14 维特征提取、HMM 状态推断 + 降级、合成数据生成器、上下文打包器、标题分析器

**重新实现（从零或大幅重构）**: 采集器（跨平台抽象层）、调度器（消除全局单例+异步）、API 层（RFC 9457 + token + 完整测试）、数据库引擎（异步 SQLAlchemy + Alembic）、配置（分层+验证）

**放弃（旧代码）**: Tray 模块（打包时另建）、全局默认用户、Win32 假数据

---

## 8. 接受标准（Gate 2 Checklist）

- [ ] 功能需求完整覆盖立项书四大模块（F1-F4）+ Must/Should/Could 分级合理
- [ ] 非功能需求全部可测量量化且无模糊描述
- [ ] LLM 成本确认 ≤ ¥500/年（启用全部功能）
- [ ] 安全边界明确：危机检测、免责声明、文案合规（独立于 LLM）
- [ ] 跨平台范围明确：Windows 完整 + macOS/X11 Linux 核心功能
- [ ] 测试策略分层合理：目标覆盖率 + 阻塞门禁
- [ ] 技术组合有选择理由且可实现（无未发布实验库）
- [ ] 领域模型匹配 Bucket + Event 事件溯源范式
- [ ] API 契约符合 RFC 9457 + OpenAPI 标准
- [ ] 迁移策略对原后端零破坏
