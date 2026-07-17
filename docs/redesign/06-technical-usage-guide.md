# MindFlow 后端技术与使用报告

> **文档编号**: 06-technical-usage-guide.md · 2026-07-17
> **对应代码**: `mindflow-app/backend-next/`（commit `b3f5a59`）
> **读者**: 项目组全体（后端维护 / 前端对接 / 结题答辩）

---

## 1. 项目总览

| 维度 | 数字 |
|------|------|
| 源文件 / 测试 | 76 个源文件（~9000 行）/ 727 个测试 + 1 合法跳过 |
| 类型安全 | mypy --strict 全库零错误 |
| API | 19 个 REST 端点 + 1 个 WebSocket，26 条路由 |
| 定时任务 | 5 个（会话识别 23:59 / 日报 00:01 / 清理 03:00 / 备份 04:00 / 自动干预每 30min） |
| 性能实测 | API p95=16.8ms，p99=22.8ms，常驻内存 109MB |
| 质量流程 | 9 轮独立评审 + 全库安全审计（OWASP 全清）+ AI 残留扫描（零残留）+ 10 步实机 E2E |

**四大核心模块**（对应立项书）：无感采集 → 行为建模 → LLM+CBT 归因 → 个性化干预，全部落地。

## 2. 架构总图

```
┌────────────────────────────────────────────────────────────┐
│  python -m mindflow.main  (watchdog: 崩溃自动重启, ≤3次/时)  │
│  └── uvicorn.Server (asyncio, 127.0.0.1:8765)              │
│      ├── Middleware 栈: 日志→Host校验→Token认证→限流         │
│      ├── REST /api/v1/* + WS /api/v1/ws                    │
│      ├── Services: collector/analysis/report/llm/           │
│      │             intervention/effectiveness/export/maint  │
│      ├── APScheduler: 5 个定时任务                          │
│      └── SQLAlchemy async + aiosqlite (WAL) → SQLite       │
│                                                            │
│  分层: api → services → domain ← infrastructure            │
│  （domain 层纯 stdlib，零框架依赖 — 可独立测试/复用）          │
└────────────────────────────────────────────────────────────┘
```

数据流：`采集器 5s tick → EventBus → heartbeat 合并写入 events 表（append-mostly）→ 投影（focus_sessions/daily_reports）→ 分析/归因/干预`

---

## 3. 关键技术实现解析

### 3.1 Event Sourcing + Heartbeat 合并（数据层核心）

**问题**：旧后端 duration 用配置值估算（数据失真），且每 5 秒一行导致每天 1.7 万行膨胀。

**方案**（`infrastructure/repositories/activity.py:95-103`）：事件表 append-mostly——常规只追加；唯一例外是同应用连续快照在 pulsetime 窗口（10s）内做 **SQL 级原子累加**：

```python
if last is not None and self._should_merge(last, event):
    await session.execute(
        sa.update(activity_events)
        .where(activity_events.c.id == last.id)
        .values(duration_s=activity_events.c.duration_s + event.duration_s)
    )
    return
```

**技术要点**：
- `duration_s + event.duration_s` 是**数据库端表达式**而非 Python 读改写 → 并发写者各自的增量都被正确累加，无竞态（评审确认）
- 合并只动最近一行，历史行事实不可变 → 保留事件溯源的可回溯性
- 效果：磁盘写削减 90%+（对标 ActivityWatch 实践），且 duration 是 tick 实测而非估算

### 3.2 采集器：协议抽象 + 自愈 tick 循环

**跨平台**（`infrastructure/collectors/base.py`）：`EventCollector` Protocol + 工厂，平台文件各 <200 行（Win32 实现所有阻塞调用包 `asyncio.to_thread`，绝不卡事件循环）。核心业务零 `sys.platform` 分支。

**自愈循环**（`services/collector_service.py:128-165`，真实代码节选）：

```python
async def _run(self) -> None:
    while not self._stop_requested:                    # 哨兵而非 cancel()
        tick_start = datetime.now(UTC)
        try:
            await asyncio.wait_for(self._tick(), timeout=self._interval_s * 2)
            self._consecutive_failures = 0
        except TimeoutError:
            self._consecutive_failures += 1            # 挂起也计失败
            if self._consecutive_failures >= 10:
                self._status = "degraded"; break       # 熔断，API 照常服务
        ...
        elapsed = (datetime.now(UTC) - tick_start).total_seconds()
        sleep_time = max(0.0, self._interval_s - elapsed)   # 补偿 tick 耗时防漂移
```

**技术要点**：
- **哨兵停止**：`stop()` 设标志 + 等待自然退出（超时才 cancel 兜底）→ 正在写库的最后一个事件必定落地（NF-R4 数据零丢失；有慢速 mock 测试证明）
- **双保险熔断**：单次异常不杀循环，连续 10 次才降级；降级后 API 与历史数据完全可用
- tick 耗时补偿消除周期漂移

### 3.3 TMT 拖延规则引擎（LLM 的免费兜底 + 标签规范）

`domain/procrastination.py`——把 Steel 时间动机理论（Motivation = E×V / I×D）翻成可计算规则，5 类型每条阈值有文献依据：

| 类型 | 触发条件 | 置信度映射 | 依据 |
|------|---------|-----------|------|
| 冲动分心 | 最长专注块<300s 且 切换≥12次/h | 12/h→0.5 线性至 24/h→0.95 | Gonzalez & Mark 2004 |
| 决策困难 | 启动延迟>30min 且 启动后专注正常 | 30min→0.5 至 60min→0.95 | TMT |
| 完美主义 | 自我批评/反复重做信号 | 1信号→0.6，2→0.85 | Shafran 2002 |
| 情绪调节 | 娱乐占比>55% | 0.55→0.5 至 0.80→0.95 | Rozental & Carlbring |
| 任务畏惧 | 兜底：专注率<0.35 或基线偏差<-0.5 | 反比映射 | — |

**关键契约**：无显著信号时 `recommended_technique=None`（评审修复——绝不"说没问题却推荐干预"）；`assess()` 全定义域不抛异常（hypothesis 属性测试保证）→ 这就是 L3 兜底"永不失败"的数学基础。

### 3.4 LLM 三层降级链 + 危机检测前置

`services/llm_service.py:205-245`（真实代码节选）：

```python
# L1: DeepSeek API
if self._deepseek_client is not None:
    try:
        result = await self._deepseek_client.analyze(summary_json)
        return self._llm_result_to_assessment(result), "deepseek", False
    except (LLMAPIError, TimeoutError) as exc:
        logger.warning("L1 failed: {}. Falling back to L2.", exc)
# L2: Ollama 本地（可选）
if self._ollama_base_url:
    ...
# L3: RuleEngine（永不失败）
assessment = self._rule_engine_to_assessment(self._rule_engine.assess(summary))
return assessment, "rule_engine", True
```

**安全四道闸**（顺序即防线）：
1. **危机检测**在任何 LLM 调用**之前**运行（纯关键词规则，`force=True` 也无法绕过；命中→热线文案+终止，评审确认无旁路）
2. **发送脱敏**：行为摘要只含数值聚合（切换率/占比/专注块），窗口标题原文、文件路径永不出境（NF-S3a；含 manual_tag 路径特征过滤）
3. **输出禁词**：Pydantic validator 拦截"诊断/治疗/患者/处方"于**全部**输出字段（response_text/next_action/cognitive_distortions）——Woebot 监管教训的工程化
4. **降级不降智**：全链失败仍有规则引擎结果，HTTP 200 + `meta.degraded:true`（请求成功完成了，不是 503）

**成本**：幂等日缓存（同日重复调用零 API 花费）+ 中间件限流（1/30s 桶容量 5 + 20 次/日硬上限）→ 单用户全功能 <¥100/年。

### 3.5 干预节流状态机（"不惹人烦"的工程化）

`services/intervention_throttle.py`——竞品最大死因是干预惹人反感，这里是立项书关键问题 3 的答案：

```
can_intervene(user_id, type)
  ├─ 疲劳检测: 7日忽略率>60% → 每日上限降为 1
  ├─ 每日上限: 默认 3 次
  ├─ 冷却: 距上次 <2h 拒绝（查询下界 now-2×cooldown，跨天也拦得住）
  ├─ 同类上限: 每类型每日 ≤2
  └─ 全过 → 放行
另: 深度工作守卫（当前 focus>80 → 零打扰）; 用户手动触发不受节流（主动要求不拦）
```

**工程亮点**：Clock 协议注入贯穿 throttle 与 repository（单一时钟源）——测试用固定在 `2026-01-15` 的假时钟全绿，证明与系统日期彻底解耦；也为未来切换北京时区口径铺路。

### 3.6 localhost 安全三层（含一个 E2E 抓到的教科书级 bug）

1. **Token**：`secrets.token_hex(64)` 写入 0600 权限文件，`secrets.compare_digest` 恒定时间比较；WS 走 `?token=`（浏览器 WS 不能带 header），配套 `access_log=False` 防日志泄漏
2. **Host 校验**：只信 localhost/127.0.0.1/[::1]（防 DNS rebinding）。`[::1].evil.com` 走私攻击被评审抓获并修复
3. **限流**：内存令牌桶 + asyncio.Lock（`middleware/ratelimit.py:76-97`），429 带 `X-RateLimit-*` 头

**E2E 抓到的坑（值得答辩讲）**：Starlette 的 `request.url` 惰性解析 Host header——畸形 IPv6 Host 让 `urlsplit` 抛 ValueError，把本该 403 的请求变 500，**连兜底异常处理器自己都在同一行二次崩溃**。修复：中间件/错误处理器全部改用 `request.scope["path"]`（来自请求行，永不解析 Host）。单元测试测的是解析函数所以全绿——只有打真实服务器才暴露。

### 3.7 可靠性设施

- **迁移**（`infrastructure/migrations.py`）：Alembic 是同步 API → 独立同步 URL + `asyncio.to_thread` 包装（架构评审抓掉的必崩写法）；失败**降级运行**不阻塞启动，health 端点可查状态
- **SQLite 生产化**：每个连接 event listener 注入 `WAL / synchronous=NORMAL / busy_timeout=5000 / journal_size_limit=64MB`；启动 `integrity_check` 失败自动 VACUUM 重建重检
- **备份**：每日 `VACUUM INTO`（WAL 下直接拷文件不可靠）；失败→日志+桌面通知
- **watchdog**（`main.py`）：进程级崩溃自动重启，每小时≤3 次防死循环
- **优雅关闭**：停调度器→关全部 WS→停采集器（等在途事件落地）→dispose 引擎

### 3.8 Welford 在线基线（个性化的统计底座）

`domain/baseline.py`——24×7 时段桶（每小时×每星期几），单遍增量算均值/方差：

```python
prev["n"] += 1.0
delta = val_f - prev["mean"]
prev["mean"] += delta / prev["n"]
delta2 = val_f - prev["mean"]
prev["M2"] += delta * delta2          # 方差 = M2/(n-1)
```

不存原始数据、O(1) 内存、数值稳定。配合 `deviation.py` 的多维加权 Z-score（阈值 1.5/2.5/4.0 三级严重度），实现"和**你自己的**周二上午比"而非和大盘比——个性化检测的核心。移植自旧代码，评审逐行确认与原公式零偏差。

---

## 4. 使用指南

### 4.1 启动

```bash
conda activate mindflow
cd mindflow-app/backend-next
pip install -e ".[dev]"        # 首次
python -m mindflow.main        # 生产入口（含 watchdog）
# 开发热重载: uvicorn --factory "mindflow.app:create_app" ... 见 README
```

启动自动完成：建库/迁移 → 完整性检查 → 生成 token → 装配服务 → 注册定时任务。数据目录：`%LOCALAPPDATA%\mindflow\mindflow\`（db/token/logs/backups/models）。

### 4.2 API 快速上手

```bash
T=$(cat "$LOCALAPPDATA/mindflow/mindflow/token")     # 128 位 hex
H="Authorization: Bearer $T"; B=http://127.0.0.1:8765/api/v1

curl $B/health                          # 唯一免认证端点
curl -X POST -H "$H" $B/collector       # 启动采集
curl -H "$H" $B/activities/current      # 当前窗口
curl -H "$H" $B/focus                   # 今日专注报告
curl -H "$H" "$B/focus/trend?days=7"    # 趋势
curl -H "$H" $B/analytics/patterns      # 分心模式(热力图/触发应用)
curl -H "$H" $B/analytics/profile       # 行为画像
curl -X POST -H "$H" -H "Content-Type: application/json" -d '{}' \
     $B/analytics/attribution           # LLM 归因（无 key 自动走规则引擎）
curl -X POST -H "$H" $B/intervention/trigger   # 手动触发干预
curl -H "$H" "$B/export?fmt=csv"        # 导出（附件下载）
```

完整交互文档：浏览器开 `http://127.0.0.1:8765/docs`（Swagger，免认证）。
错误统一 RFC 9457 `application/problem+json`，8 个错误码见 `docs/redesign/03-requirements.md §4.2`。

### 4.3 WebSocket 实时推送

```javascript
const token = /* 读 token 文件或后端注入 */;
const ws = new WebSocket(`ws://127.0.0.1:8765/api/v1/ws?token=${token}`);
ws.onmessage = (e) => {
  const {type, payload, timestamp} = JSON.parse(e.data);
  // type ∈ activity_update | focus_change | intervention | error | pong
};
setInterval(() => ws.send(JSON.stringify({type:"ping",payload:{}})), 30_000);
// 断线重连: 指数退避 1s→2s→4s→…→30s + ±20% 抖动
```

`activity_update` 服务端已节流（2s 且状态变化才推），前端无需去抖。

### 4.4 LLM 配置

`.env`（backend-next/ 下）或环境变量：

```ini
MINDFLOW_LLM__API_KEY=sk-xxxx                       # DeepSeek key → 启用 L1
MINDFLOW_LLM__BASE_URL=https://api.deepseek.com     # 默认即此
MINDFLOW_LLM__MODEL=deepseek-chat
MINDFLOW_LLM__OLLAMA_ENABLED=true                   # 可选 L2 本地模型
```

不配 key 一切照常（规则引擎兜底）——演示零成本，配 key 后归因质量升级。限流已内建（20 次/日硬上限保护预算）。

### 4.5 ML 训练 CLI

```bash
python -m mindflow.train --source synthetic   # 合成数据端到端（种子42可复现）
python -m mindflow.train --source db          # 用真实采集数据
python -m mindflow.train --list-versions      # 模型版本列表
python -m mindflow.train --rollback 20260717  # 回滚到某日模型
```

产物：`data/models/{clustering,classifier,hmm}-YYYYMMDD.pkl + latest.json + training_report.json + baseline_user1.json`。hmmlearn 未装时 HMM 自动降级 Markov 链（`pip install -e ".[ml]"` 可补装）。

### 4.6 数据与运维

- **保留策略**：原始事件 30 天（`MINDFLOW_EVENT_RETENTION_DAYS` 7-90 可调）自动清理；聚合报告永久
- **备份**：每日 04:00 自动 `VACUUM INTO` 到 `backups/`；手动恢复=停服后用备份文件替换 `mindflow.db`
- **日志**：`logs/mindflow_YYYY-MM-DD.log`，10MB 轮转 / 30 天保留 / gz 压缩
- **验证套件**：`python -m pytest tests/ -q && python -m mypy src/ --strict && python -m ruff check src/ tests/`

### 4.7 常见问题

| 现象 | 原因/处理 |
|------|----------|
| 401 | token 没带对——重读 token 文件（每次全新安装会重新生成） |
| 403 forbidden-host | 用了非 localhost 的 Host 访问——设计如此，仅本机 |
| collector status=degraded | 连续 10 次采集失败已熔断——`POST /collector/stop` 再 `POST /collector` 重启 |
| attribution `degraded:true` | 未配 LLM key 或 API 不可用，走了规则引擎——功能正常 |
| health `migration.applied:false` | 迁移失败降级运行——看日志 CRITICAL 行，通常为 db 文件被占用 |
| 429 | 触发限流——看 `Retry-After` 头 |

## 5. 前端对接要点（张皓 / 杨智杰）

1. 所有请求带 `Authorization: Bearer <token>`；token 由桌面壳读文件注入（浏览器 fetch 拿不到本地文件）
2. 错误处理只需解析 problem+json 的 `detail`（已是中文用户文案）+ `type` 做分支
3. 实时 UI 用 WS 四种业务帧；轮询兜底可打 `GET /activities/current`
4. Dashboard 数据源映射：状态卡→WS activity_update；饼图→`/focus` 的 top_apps；趋势→`/focus/trend`；热力图→`/analytics/patterns`；干预弹窗→WS intervention 帧 + `POST /intervention/{id}/response` 回传用户反应（accepted/ignored/dismissed——这是效果评估和疲劳降频的数据源，务必回传）

## 6. 已知边界（诚实清单）

1. 导出为内存组装（90 天上限兜底），真流式列入 backlog
2. macOS/X11 采集器代码完整但未实机验证（无硬件），Wayland 仅 pid 级降级
3. PyInstaller spec 就绪未实际构建（发布清单第一项）
4. 单用户假设（user_id=1 硬编码处均有 TODO(multi-user) 标记）
5. LLM 在线效果 eval（golden set）未建——离线规则路径已全测
