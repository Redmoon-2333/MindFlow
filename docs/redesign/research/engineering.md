# 商业级本地优先后端工程调研报告

> **文档编号**: research/engineering
> **日期**: 2026-07-17
> **作者**: research-engineering agent（document-specialist, MED tier）
> **状态**: 7 大主题全部完成，关键事实附来源 URL
> **用途**: Gate 1 验收材料之一，架构设计（Phase 4）与实现（Phase 5）的直接输入

---

## 主题 1：SQLite 生产化

### 1.1 WAL 模式 + busy_timeout

生产化首要 PRAGMA 组合：
```sql
PRAGMA journal_mode=WAL;           -- 写性能提升 2-3x
PRAGMA synchronous=NORMAL;          -- WAL 模式下安全
PRAGMA busy_timeout=5000;           -- 忙等待 5 秒而非立即报 SQLITE_BUSY
PRAGMA journal_size_limit=67108864; -- WAL 文件上限 64MB
```
- 现有代码已设 WAL + synchronous=NORMAL（`database.py:34-38`），但**缺 busy_timeout**，并发写仍可能 `SQLITE_BUSY`
- 建议：`connect_args={"check_same_thread": False, "timeout": 5}` + 追加 busy_timeout/journal_size_limit PRAGMA
- 源: daily.dev SQLite production guide; github.com/cashubtc/nutshell/issues/907

### 1.2 连接策略
- 桌面单用户场景保持 SQLAlchemy 默认 `QueuePool` 足够；未来锁冲突严重再切 `NullPool` + `pool_pre_ping`

### 1.3 Alembic 迁移落地（桌面应用特殊性）
- 用户机器上必须**启动时自动迁移**，失败记日志并以旧 schema 降级运行，不阻塞启动
- SQLite 不支持 `ALTER COLUMN`，`env.py` 必须 `render_as_batch=True`
- 建议分阶段：需 schema 变更时再引入 Alembic
- 源: alembic.sqlalchemy.org; github.com/sqlalchemy/alembic/issues/755

### 1.4 备份方案对比

| 方案 | WAL 感知 | 结论 |
|------|---------|------|
| `VACUUM INTO` | 是 | **桌面推荐**（原子快照） |
| `sqlite3 .backup` API | 是 | 定时备份可用 |
| 直接 cp .db 文件 | **否** | WAL 下不可靠，禁止 |
| Litestream | 是 | 云备份最佳但桌面 overkill |

- 源: litestream.io/alternatives/cron; reddit WAL 备份陷阱 PSA

### 1.5 损坏检测
- lifespan startup 执行 `PRAGMA integrity_check`；失败时保存损坏副本 + 尝试 VACUUM 重建

---

## 主题 2：进程与服务管理

### 2.1 FastAPI 嵌入式启动
- 桌面推荐 **`uvicorn.Server(Config(...)).serve()` 编程式启动**（可控 graceful shutdown，`server.should_exit = True` 停止）
- lifespan shutdown 顺序：停采集器 → 关数据库 → 退出
- 源: uvicorn.dev/#running-programmatically; github.com/Kludex/uvicorn/discussions/1103

### 2.2 崩溃自动重启
- 主进程 watchdog 循环 + 限制重启频率（3 次/分钟）防崩溃循环

### 2.3 采集器与 API 的进程边界（关键决策）

| 模式 | Pros | Cons |
|------|------|------|
| 同进程线程（现状） | 简单、零 IPC、单可执行文件 | 采集器崩溃带走应用 |
| 独立进程 + HTTP IPC（ActivityWatch） | 进程隔离、各自重启 | 子进程生命周期管理复杂 |

- **建议：保持同进程**。MindFlow 采集消耗极低；AW 分离是因其开放 watcher 生态。预留 `CollectorService` 接口，未来可无痛分离
- 注意：采集循环不得阻塞 API 事件循环（APScheduler 后台线程方式正确）

### 2.4 开机自启动
- Windows `HKCU\...\Run` 注册表 / macOS LaunchAgents / Linux autostart .desktop；UI 开关 + 首次询问

---

## 主题 3：localhost API 安全

### 3.1 认证方案
- **推荐：随机 token 文件 + Bearer header**。启动生成 64 字节随机 token → 写入 `~/.mindflow/token`（权限 600）→ 所有 API 要求 `Authorization: Bearer <token>` → 前端读文件附带
- 源: KeePassXC Native Messaging 安全设计 github.com/keepassxreboot/keepassxc/issues/287

### 3.2 DNS Rebinding 防护（三层）
1. **Host header 校验** middleware：只接受 `127.0.0.1:8765` / `localhost:8765` / `[::1]:8765`
2. Token 认证兜底
3. 前端走 Tauri/Electron 内嵌（不受 DNS rebinding 影响）
- 源: github.blog DNS rebinding attacks explained; paloaltonetworks.com

### 3.3 CORS
- 前端同源托管（FastAPI static files 或桌面壳内嵌）后 CORS 可收紧到几乎不需要

### 3.4 数据隔离
- `platformdirs` 获取标准路径（Windows `%APPDATA%/MindFlow/`）；SQLite 文件权限 600

---

## 主题 4：跨平台窗口/活动采集

### 4.1 库选型

| 库 | 平台 | 结论 |
|----|------|------|
| **PyWinCtl** | Win/macOS/Linux 统一 API | **推荐**（Win32 / AppleScript / EWMH+Xlib） |
| pyobjc + NSWorkspace | macOS only | 原生备选 |
| 现有 win32gui + psutil | Windows only | 不可移植 |

- 源: github.com/Kalmat/PyWinCtl

### 4.2 macOS 权限
- Catalina 起需 Screen Recording / Accessibility 权限；**Python 二进制需 codesign 否则权限弹窗不生效**

### 4.3 Linux 困境
- X11: python-xlib + EWMH 稳定；**Wayland: 安全模型禁止跨客户端读窗口信息，无标准 API**；GNOME 2025.11 已完全移除 X11 后端
- 建议分两档：X11 全功能；Wayland 回退 pid 级采集（无窗口标题），跟踪 ext-foreign-toplevel 协议进展

### 4.4 浏览器 tab 级采集
- Native Messaging 方案（JSON 清单注册 + stdin/stdout 通信 + token 认证回传本地后端）；**优先级低，Phase 2+**
- 源: developer.chrome.com native-messaging; MDN Native_messaging

---

## 主题 5：打包分发

### 5.1 打包工具

| 工具 | sklearn+hmmlearn 支持 | 体积 | 结论 |
|------|---------------------|------|------|
| **PyInstaller** | 官方支持 | ~300-500MB | **推荐**（社区验证最充分；注意 joblib loky 需额外配置） |
| Nuitka | C 扩展有失败风险 | ~200-350MB | 性能敏感时再评估 |
| Briefcase | 生态较小 | 类似 | 备选 |

- 源: ahmedsyntax.com 2026 对比; sparxeng.com 对比

### 5.2 自动更新
- **推荐 tufup**（TUF 安全协议：密钥签名、增量更新、回滚保护、纯 Python 跨平台）
- 源: github.com/dennisvang/tufup

### 5.3 体积优化
- UPX 压缩、排除无用后端、ML 模型懒加载

---

## 主题 6：可观测性

### 6.1 结构化日志
- **推荐 loguru**：极简 API、内置 JSON 序列化/滚动/压缩、Sentry 原生集成
```python
logger.add("logs/mindflow_{time:YYYY-MM-DD}.log", rotation="10 MB", retention="30 days", compression="gz", serialize=True)
```
- 源: dash0.com loguru guide

### 6.2 崩溃上报
- Sentry **opt-in**（首次启动询问）+ `before_send` 过滤敏感信息 + `traces_sample_rate=0`
- 源: docs.sentry.io/platforms/python/

### 6.3 性能指标
- 最小集：采集耗时、API 响应耗时、DB 大小；debug 模式下 `/metrics`（prometheus_client）

---

## 主题 7：API 工程规范

### 7.1 版本化
- 保持 `/api/v1`，补充 tags 元数据

### 7.2 错误规范 — RFC 7807/9457 Problem Details
- 现有自定义 `{code, message, data}` 非标准 → 统一为 `application/problem+json`：
```json
{"type": "https://mindflow.app/errors/user-not-found", "title": "User Not Found", "status": 404, "detail": "...", "instance": "/api/v1/..."}
```
- 实现 `ProblemDetail` 异常类 + FastAPI exception handler
- 源: RFC 7807/9457; github.com/tiangolo/fastapi/discussions/8059

### 7.3 OpenAPI 契约
- 所有路由补 description/summary/tags/response_model；WebSocket 用 openapi_extra 描述

### 7.4 WebSocket 重连协议
- 客户端职责：指数退避（1s→2s→4s→8s→max 30s）+ ±20% 抖动 + 30s 心跳 ping/pong + 断连消息队列
- 后端只做 `WebSocketDisconnect` 清理，不主动重连

---

## 优先级矩阵（按实施顺序）

| 优先级 | 主题 | 具体措施 | 工作量 |
|--------|------|---------|--------|
| **P0** | SQLite | busy_timeout + journal_size_limit + integrity_check | 低 |
| **P0** | 安全 | Host header 校验 + random token 认证 | 低 |
| **P0** | 日志 | loguru + 滚动文件 | 低 |
| **P1** | API 规范 | RFC 7807 + OpenAPI 完善 | 中 |
| **P1** | 采集器 | PyWinCtl 统一跨平台 API | 中 |
| **P1** | 进程管理 | uvicorn.Server 编程式启动 + graceful shutdown | 低 |
| **P2** | 迁移 | Alembic + startup 自动迁移 | 中 |
| **P2** | 备份 | VACUUM INTO 每日备份 | 低 |
| **P2** | WebSocket | 前端指数退避重连 | 低 |
| **P3** | 打包 | PyInstaller + UPX + tufup | 高 |
| **P3** | 崩溃上报 | Sentry opt-in | 中 |
| **P4** | Wayland / 浏览器扩展 | 持续跟踪 / Native Messaging | 高 |

---

## 来源索引（节选）

| 引用内容 | URL |
|----------|-----|
| SQLite 生产配置 | https://daily.dev/blog/sqlite-production-guide-when-how-to-use-beyond-prototyping |
| busy_timeout 生产死锁案例 | https://github.com/cashubtc/nutshell/issues/907 |
| Alembic | https://alembic.sqlalchemy.org/en/latest/autogenerate.html |
| Litestream/备份 | https://litestream.io/alternatives/cron |
| Uvicorn 编程式启动 | https://uvicorn.dev/#running-programmatically |
| ActivityWatch 架构 | https://docs.activitywatch.net/en/latest/architecture.html |
| DNS rebinding | https://github.blog/security/application-security/dns-rebinding-attacks-explained/ |
| PyWinCtl | https://github.com/Kalmat/PyWinCtl |
| GNOME 移除 X11 | https://canartuc.medium.com/gnome-completely-drops-x11-support-the-wayland-era-begins-387e961926c0 |
| Native Messaging | https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging |
| 打包对比 | https://ahmedsyntax.com/2026-comparison-pyinstaller-vs-cx-freeze-vs-nui |
| tufup | https://github.com/dennisvang/tufup |
| loguru | https://www.dash0.com/guides/python-logging-with-loguru |
| RFC 7807/9457 | https://datatracker.ietf.org/doc/html/rfc9457 |
| WebSocket 重连 | https://softwareengineering.stackexchange.com/questions/434117/ |
