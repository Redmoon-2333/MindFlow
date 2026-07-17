# 开源时间追踪/行为采集项目架构调研报告

> **文档编号**: research/oss
> **日期**: 2026-07-17
> **作者**: research-oss agent（document-specialist, MED tier）
> **状态**: 完成；每条借鉴标注证据类型（文档/源码证实 vs 推断）
> **用途**: Gate 1 验收材料之一，架构设计（Phase 4）的直接输入

---

## 一、ActivityWatch 深度分析（重点）

**项目定位**: 开源、隐私优先的全自动时间追踪器，RescueTime 的替代品。
**仓库**: https://github.com/ActivityWatch/activitywatch | **文档**: https://docs.activitywatch.net | **许可证**: MPL-2.0

### 1.1 架构分层与进程模型

模块化 C/S 架构，多进程模型：

```
aw-tauri (进程管理器)
  ├── 内嵌 aw-server-rust (REST API on localhost:5600)
  │     └── SQLite 数据库 (事件存储)
  ├── 托管 aw-webui (Web 可视化面板)
  └── 作为子进程管理多个 watcher
        ├── aw-watcher-window (活跃窗口追踪)
        ├── aw-watcher-afk (键盘/鼠标空闲检测)
        ├── aw-watcher-web (浏览器标签追踪，MV3 扩展)
        └── 第三方 watcher (VS Code, JetBrains, Android 等)
```

**进程模型决策**:
- **为何多进程**: watcher 独立进程，一个崩溃不影响其他；可独立开发/分发/升级；用户按需启用
- **为何 aw-tauri 取代 aw-qt**: 旧版有子进程无法正确关闭的 bug；aw-tauri 提供指数退避重启（2s→4s→8s，3 次失败后停止并通知）、跨平台父子进程绑定（Unix pipe / Windows Job Object）、自动发现 PATH 中 aw-* watcher
- 源: https://docs.activitywatch.net/en/latest/architecture.html

### 1.2 数据模型设计：Bucket + Event

**Bucket**（每个 watcher 每台主机一个）:
```json
{"id": "aw-watcher-window_hostname", "type": "currentwindow", "client": "aw-watcher-window", "hostname": "my-laptop"}
```

**Event**（统一三元组）:
```json
{"timestamp": "2024-01-01T10:00:00Z", "duration": 120.5, "data": {"app": "Code.exe", "title": "main.py - VS Code"}}
```
- 所有时间戳 UTC；`data` 是按 bucket type 约定 schema 的自由 JSON
- 存储 SQLite（aw-server-rust 有 bucket 层缓存 + mpsc 通道事务优化）

**Heartbeat 合并机制**（关键设计）:
- `POST /api/0/buckets/<id>/heartbeat`，pulsetime 窗口内相邻相同 data 的事件自动合并（保留最早时间戳 + 累加 duration），**减少 90%+ 磁盘写**
- 源: https://docs.activitywatch.net/en/latest/buckets-and-events.html

### 1.3 API 设计风格

- 版本化 `/api/0/` 前缀；**无认证**（强制 localhost-only，官方明确反对远程部署）
- 查询引擎 `POST /api/0/query`：非 SQL 的脚本式 DSL（`query_bucket`, `filter_keyvals`, `merge_events_by_keys`, `categorize`, `sum_durations`, `flood`），时间周期在请求层管理
- 源: https://docs.activitywatch.net/en/latest/api/rest.html

### 1.4 采集器抽象方式

- **Watcher 接口 = 能 HTTP POST 事件到 aw-server 的任何程序**；官方 watcher 用 aw-client 库封装 heartbeat/bucket 管理
- 跨平台: Linux/X11 用 python-xlib；macOS 用 Swift + Accessibility API；Windows 用 Win32 API + WMI；Wayland 单独 aw-awatcher
- 浏览器: MV3 扩展，`web.tab.current` 事件上报 url/title/audible/incognito/tabCount
- 源: https://github.com/ActivityWatch/aw-watcher-window

### 1.5 空闲检测 (aw-watcher-afk)

- 默认 3 分钟无输入 → AFK，可配置
- **设计原则：用更小的超时采集原始数据（更高分辨率），后续可合并为更大间隔——反过来不行**
- 源: https://github.com/ActivityWatch/aw-watcher-afk

### 1.6 隐私设计

- 100% 本地存储、无远程部署支持、用户可随时导出/删除/备份、开源可审计
- 同步功能 WIP，设计为**去中心化**（非中心化云端）
- 源: https://activitywatch.net/blog/activitywatch-vs-rescuetime/

---

## 二、WakaTime 插件化采集分析

**仓库**: https://github.com/wakatime/wakatime-cli | **文档**: https://wakatime.com/help/creating-plugin

### 2.1 两层插件架构

```
编辑器/IDE
  └── Editor Plugin (IDE-specific, 薄层, 50-200行)
        └── 作为子进程调用 wakatime-cli (Go, 共享逻辑)
              └── POST heartbeat → api.wakatime.com
                     └── 失败时存入本地 BoltDB (离线队列)
```

### 2.2 Heartbeat 协议

- 触发: 文件切换/修改/保存；**限频: 同一文件 2 分钟内不重复发送（除非 save 事件）**
- 字段: entity、time、write、language、project、category（coding/building/debugging/browsing）

### 2.3 离线缓存与重试

- BoltDB 队列: `PushMany`（存失败心跳）→ `PopMany`（取出重试）→ `Sync`（批量发送，默认最多 1000 条）
- 退避: `(now - backoff_at >= backoff_seconds) OR (now - backoff_at > 3600)`；退避期内直接入队不尝试发送
- 源: https://pkg.go.dev/github.com/wakatime/wakatime-cli@v1.18.7-alpha.1/pkg/offline

### 2.4 插件抽象模式

- 核心逻辑全在共享 CLI；新增编辑器只写监听层；CLI 自动从 GitHub Releases 按平台下载

---

## 三、Tockler / Selfspy / arbtt 分析

### 3.1 arbtt（Haskell）
- 单守护进程 `arbtt-capture`（每分钟采样）+ 分析工具 `arbtt-stats`
- **核心亮点：规则引擎 DSL 可追溯应用——改分类规则后所有历史数据自动重新分类**
- 源: https://github.com/nomeata/arbtt

### 3.2 Selfspy（Python + SQLAlchemy + SQLite）
- 捕获逐击键（含时间间隔）+ 鼠标坐标 + 窗口切换；可选 Blowfish 加密
- **教训：最具隐私侵入性，不适合 MindFlow 参考**
- 源: https://github.com/selfspy/selfspy

### 3.3 Tockler（TypeScript + Electron）
- 窗口标题 + idle/offline/online 三态状态机；开箱即用的精美 GUI；Electron 资源占用是权衡点
- 源: https://github.com/MayGo/tockler

---

## 四、横向对比表

| 维度 | ActivityWatch | WakaTime | Tockler | Selfspy | arbtt |
|------|:---:|:---:|:---:|:---:|:---:|
| 进程模型 | 多进程 C/S | 两层插件+CLI | 单进程 Electron | 单守护进程 | 单守护进程 |
| 存储引擎 | SQLite | BoltDB + 云端 | 日志文件 | SQLite | 二进制日志 |
| 事件粒度 | 秒级持续时间 | 2 分钟心跳 | 窗口变化 | 逐击键 | 逐分钟采样 |
| API 风格 | REST (/api/0/) | REST | 无 API | 无 API | CLI |
| 可扩展性 | 自定义 watcher (HTTP) | 自定义 plugin (CLI) | 无 | 无 | 规则配置 |
| 空闲检测 | 专用 watcher (3 分钟) | 无 | 内置状态机 | 击键间隔推断 | XScreenSaver |
| 离线支持 | 本地天然支持 | BoltDB 队列+退避 | 本地 | 本地 | 本地 |
| 浏览器追踪 | MV3 扩展 | 无 | 无 | 无 | 无 |
| 隐私模型 | 本地优先无远程 | 本地+云端 | 本地 | 本地可加密 | 本地 |

---

## 五、对 MindFlow 新架构的 10 条可执行借鉴

1. **Heartbeat 合并机制**（证实）— 实现 heartbeat 端点，pulsetime 窗口内合并相邻相同事件，O(n)→O(1) 存储，减少 90%+ 磁盘写
2. **"薄采集层 + 共享处理逻辑"模式**（证实，源自 WakaTime）— 定义 `Collector` 接口，平台代码只做平台 API 调用，共享 `EventProcessor`；新增采集目标只需几十行
3. **多进程 watcher 架构**（证实 + 推断）— 进程隔离防止采集崩溃带走整个应用；进程管理器做崩溃恢复（指数退避重启 3 次）
4. **Bucket + Event 统一数据模型**（证实）— `timestamp + duration + data` 三元组统一表示专注/分心/空闲，分析逻辑可复用
5. **离线队列 + 指数退避**（证实，源自 WakaTime）+ 本地 SQLite 主存储（源自 AW）— 网络不可用时零数据丢失
6. **空闲检测"高分辨率原始数据"原则**（证实）— ~5 秒间隔记录快照，事后聚合为专注/分心/空闲段；区分"查资料的 30 秒"和"刷视频的 5 分钟"
7. **可追溯规则引擎**（证实，源自 arbtt）— 分类规则存为配置，规则更新后历史数据按新规则重新分类（用户的分类认知会演进）
8. **查询 DSL / 链式查询构建器**（证实，源自 AW）— `query(taskId).filter(state="focus").groupBy("app").sumDuration()`，与事件流模型天然匹配
9. **避免全量击键捕获**（证实 + 推断，Selfspy 反面教材）— 只采集窗口标题 + 输入频率（每分钟击键数），不碰击键内容
10. **心跳限频策略**（证实，源自 WakaTime）— 任务未变且 <2 分钟不发送；任务切换/手动事件即时推送

---

## 来源汇总

| 项目 | 主要来源 |
|------|----------|
| ActivityWatch 架构/数据模型/API | https://docs.activitywatch.net (architecture.html, buckets-and-events.html, api/rest.html) |
| aw-server-rust / aw-watcher-* / aw-tauri | https://github.com/ActivityWatch/ 对应仓库 |
| AW 隐私哲学 | https://activitywatch.net/blog/activitywatch-vs-rescuetime/ |
| WakaTime 插件指南 / 离线队列 | https://wakatime.com/help/creating-plugin; pkg.go.dev wakatime-cli/pkg/offline |
| arbtt / Tockler / Selfspy | 各自 GitHub 仓库 |
