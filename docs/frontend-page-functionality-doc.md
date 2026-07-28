# MindFlow 前端页面功能文档（含完整 API 参数）

> 基于 `frontend/src/` 源码与 `backend-next/src/mindflow/api/routes/` 后端接口，2026-07-26 生成。
> 补充参考：`docs/api-reference.md`（API v2.0）、`docs/frontend-api-spec.md`、`docs/handbook/ch6-api-frontend.md`。
>
> Base URL: `http://127.0.0.1:8765/api/v1` · 所有时间戳 ISO8601 UTC · 错误遵循 RFC 9457

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [登录页](#2-登录页-logintsx)
3. [仪表盘](#3-仪表盘-dashboardtsx)
4. [专注分析](#4-专注分析-focustsx)
5. [活动日志](#5-活动日志-activitiestsx)
6. [行为洞察](#6-行为洞察-analyticstsx)
7. [报告中心](#7-报告中心-reportstsx)
8. [干预中心](#8-干预中心-interventiontsx)
9. [专家面板](#9-专家面板-paneltsx)
10. [AI 对话](#10-ai-对话-chattsx)
11. [系统设置](#11-系统设置-settingstsx)
12. [公共组件与基础设施](#12-公共组件与基础设施)
13. [WebSocket 协议](#13-websocket-协议)
14. [通用约定与错误码](#14-通用约定与错误码)

---

## 1. 整体架构概览

### 前端技术栈

| 类别 | 选型 |
|------|------|
| 框架 | React 19 + TypeScript |
| 构建 | Vite (OxcPress) |
| 路由 | react-router-dom v7 (BrowserRouter) |
| HTTP 客户端 | openapi-fetch（类型安全） + 原生 fetch |
| 实时通信 | WebSocket（自封装 RealtimeClient） |
| 样式 | 纯 CSS 变量主题（`theme.css`），无 UI 框架依赖 |
| 认证 | `localStorage` marker + Bootstrap ticket 机制 |

### 页面路由一览

```
/login          → Login（未认证时显示）
/               → Dashboard（仪表盘）
/focus          → Focus（专注分析）
/activities     → Activities（活动日志）
/analytics      → Analytics（行为洞察）
/reports        → Reports（报告中心）
/intervention   → Intervention（干预中心）
/panel          → Panel（专家面板）
/chat           → Chat（AI 对话）
/settings       → Settings（系统设置）
```

### 布局结构

```
┌──────────────────────────────────────────────────┐
│  Sidebar (固定 220px)          │  Main 内容区      │
│  ┌─────────────────────┐       │                   │
│  │ MindFlow (Logo)      │       │  每个页面的        │
│  │ 仪表盘                │       │  <PageComponent>  │
│  │ 专注分析              │       │                   │
│  │ 活动日志              │       │                   │
│  │ 行为洞察              │       │                   │
│  │ 报告中心              │       │                   │
│  │ 干预中心              │       │                   │
│  │ 专家面板              │       │                   │
│  │ AI 对话               │       │                   │
│  │ 系统设置              │       │                   │
│  └─────────────────────┘       │                   │
└──────────────────────────────────────────────────┘
```

**关键组件**（`App.tsx`）：
- **`<Layout>`** — 侧边栏 + 主内容区容器。`NavLink` 高亮当前路由。
- **`<AppRoutes>`** — 9 个路由映射，每个路由指向一个页面组件。
- **认证守卫** — `authenticated === false` 时只渲染 `<Login />`，不渲染 Layout 和路由。
- **WebSocket 生命周期** — 认证后自动连接 `realtimeClient`，页面卸载时断开。

### 认证流程

1. 用户通过 MindFlow 桌面启动器打开前端页面。
2. URL hash 中携带一次性 `bootstrap` ticket（`#bootstrap=xxx`）。
3. `main.tsx` 入口调用 `bootstrapFromFragment()`，POST `/api/v1/auth/bootstrap` 换取 session cookie。
4. 成功后写入 `localStorage["mindflow_authenticated"] = "1"`，清除 URL hash。
5. 后端返回 401 时，前端清除 marker 并触发 `mindflow:auth-required` 事件，页面退回登录页。

### 速率限制

| 端点 | 速率 | 日硬上限 |
|------|------|----------|
| 全局（所有端点） | 100 请求/分钟 | — |
| `POST /chat` | 5 请求/分钟 | 60/天 |
| `POST /analytics/attribution` | 2 请求/分钟 | 20/天 |
| `POST /panel/today` | 1 请求/小时 | 3/天 |
| `GET /panel` | 10 请求/分钟 | 30/天 |

超过限制返回 429（RFC 9457），响应头含 `X-RateLimit-Remaining` 和 `X-RateLimit-Reset`。

---

## 2. 登录页 (`Login.tsx`)

**路由**：`/login`（无路由路径，认证守卫自动显示）  
**用途**：阻止未认证用户访问，引导用户通过桌面启动器打开应用。

### 组件结构

```
Login
└── card (居中卡片)
    ├── h1 "MindFlow" (Logo)
    ├── p "本地优先的智能专注助手" (副标题)
    ├── info-box (提示信息)
    │   └── "请通过 MindFlow 启动器打开界面..."
    └── btn "已通过启动器打开，重新检查"
        └── onClick → window.location.reload()
```

### 后端接口

此页面不直接调用 API，仅检查 `localStorage` 中的认证标记。

**认证端点参考**（由 `main.tsx` 调用的 bootstrap 流程）：

**`POST /api/v1/auth/bootstrap`** — 一次性认证票据兑换

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ticket` | string | ✓ | 桌面启动器生成的一次性票据 |

**响应**：成功设置 session cookie（`Set-Cookie` 头），前端写 `localStorage["mindflow_authenticated"] = "1"`。

**逻辑说明**：
- 这是一个纯前端守卫页面。没有表单、没有密码输入。
- 认证完全由桌面启动器（Electron/Tauri 壳）通过 `bootstrap` ticket 完成。
- 用户无法在浏览器地址栏直接输入凭据登录——必须通过启动器打开。

---

## 3. 仪表盘 (`Dashboard.tsx`)

**路由**：`/`  
**用途**：MindFlow 系统总览，一键查看关键指标和系统状态。是用户打开应用后的默认首页。

### 组件结构

```
Dashboard
├── header "仪表盘 · MindFlow 系统概览"
├── error-box (条件渲染)
│
├── KPI Row (4 张 stat-card)
│   ├── stat-card: 今日专注时长
│   ├── stat-card: 专注会话数
│   ├── stat-card: 平均专注评分
│   └── stat-card: 分心率
│
├── 双列布局 (grid 1fr 1fr)
│   ├── 左列
│   │   ├── card: 系统健康状态
│   │   ├── card: 采集器状态
│   │   ├── card: 数据库状态
│   │   └── card: ML 模型状态
│   └── 右列
│       ├── card: 当前活动 (含 WebSocket 实时更新)
│       └── card: 近期干预记录 (含 WebSocket 实时更新)
│
└── card: 自主控制
```

### 后端接口详解

#### `GET /api/v1/health` — 系统健康检查

免认证。所有端点中唯一免认证的。启动时第一个调用的探活接口。

| 无请求参数 |
|------|

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"ok"` 或 `"error"`，整体健康状态 |
| `version` | string | 后端版本号（如 `"2.0.0-alpha"`） |
| `timestamp` | string | 服务器当前时间 ISO8601 |
| `collector.status` | string | 采集器状态：`"running"` / `"stopped"` / `"degraded"` |
| `database.status` | string | 数据库状态：`"ok"` / `"error"` |
| `database.connected` | bool | 数据库是否连接成功 |
| `migration.applied` | bool | 数据库迁移是否已应用 |

```json
{
  "status": "ok",
  "version": "2.0.0-alpha",
  "timestamp": "2026-07-23T15:24:50.454Z",
  "collector": { "status": "running" },
  "database": { "status": "ok", "connected": true },
  "migration": { "applied": true }
}
```

#### `GET /api/v1/focus/trend` — 专注趋势（仪表盘 KPI 数据源）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `days` | int | 7 | 回溯天数（范围 1–90） |

> Dashboard 固定传 `days=7`。

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `days` | int | 实际查询天数 |
| `start_date` | string | 起始日期 `YYYY-MM-DD` |
| `end_date` | string | 结束日期 `YYYY-MM-DD` |
| `daily[]` | array | 每日聚合数据 |
| `daily[].date` | string | 日期 `YYYY-MM-DD` |
| `daily[].focus_min` | float | 当日专注分钟数 |
| `daily[].distraction_min` | float | 当日分心分钟数 |
| `daily[].session_count` | int | 当日会话数 |
| `daily[].avg_score` | float | 当日平均专注评分 |
| `total_sessions` | int | 总会话数 |

> 前端还兼容扁平格式：`today_minutes`、`total_minutes`、`trend_label`、`avg_score`、`score_change`、`distraction_rate`、`distraction_label` 等——这些来自后端版本差异，Dashboard 做了 `focusTrend?.today_minutes ?? focusTrend?.total_minutes` 的兼容处理。

```json
{
  "days": 7,
  "start_date": "2026-07-17",
  "end_date": "2026-07-23",
  "daily": [
    { "date": "2026-07-17", "focus_min": 180.5, "distraction_min": 45.2, "session_count": 12, "avg_score": 72.3 }
  ],
  "total_sessions": 84
}
```

#### `GET /api/v1/activities/current` — 当前活动窗口

| 无请求参数 |
|------|

**响应**（无活动时返回 404）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 活动记录 UUID |
| `user_id` | int | 用户 ID |
| `timestamp` | string | 时间戳 ISO8601 |
| `duration_s` | float | 活动持续秒数 |
| `event_type` | string | 事件类型：`"window_snapshot"` / `"idle_change"` / `"manual_tag"` |
| `data.app_name` | string | 应用名称 |
| `data.window_title` | string | 窗口标题 |
| `data.process_name` | string | 进程名 |
| `data.is_idle` | bool | 是否空闲 |

```json
{
  "id": "uuid",
  "user_id": 1,
  "timestamp": "2026-07-23T08:30:00Z",
  "duration_s": 300.0,
  "event_type": "window_snapshot",
  "data": {
    "app_name": "Visual Studio Code",
    "window_title": "main.py — MindFlow",
    "process_name": "code.exe",
    "is_idle": false
  }
}
```

#### `GET /api/v1/analytics/model-status` — ML 模型状态

| 无请求参数 |
|------|

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `loaded` | bool | 模型是否已加载 |
| `ready` | bool | 模型是否就绪可用 |
| `mode` | string | 运行模式：`"ml_enriched"` / `"rule_engine_only"` |
| `v2_mode` | string | v2 模式标识 |
| `message` | string | 状态描述信息 |
| `feature_schema_version` | int? | 特征 schema 版本 |
| `version` | string? | 模型版本号（如 `"20260723"`） |
| `available_versions` | string[]? | 可用版本列表 |
| `model_name` | string? | 模型名称 |

```json
{
  "loaded": true,
  "mode": "ml_enriched",
  "version": "20260723",
  "available_versions": ["20260723", "20260717"],
  "message": "ML models loaded"
}
```

未加载时：`{"loaded": false, "mode": "rule_engine_only"}`

#### `GET /api/v1/intervention/history` — 干预历史

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `days` | int | 7 | 回溯天数（范围 1–90） |

> Dashboard 固定传 `days=7`。

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `items[]` | array | 干预历史记录数组 |
| `items[].id` | string | 干预 UUID |
| `items[].user_id` | int | 用户 ID |
| `items[].triggered_at` | string | 触发时间 ISO8601 |
| `items[].intervention_type` | string | 干预类型：`"task_breakdown"` / `"nudge"` / `"environment_optimization"` / `"smart_prioritization"` |
| `items[].cbt_technique` | string? | CBT 技术名 |
| `items[].context_json` | object? | 触发上下文 |
| `items[].user_response` | string? | 用户响应：`"accepted"` / `"ignored"` / `"dismissed"` |
| `items[].response_latency_s` | float? | 响应延迟秒数 |
| `items[].feedback_rating` | string? | 评分：`"effective"` / `"neutral"` / `"ineffective"` |
| `items[].feedback_comment` | string? | 反馈文字 |
| `items[].created_at` | string | 创建时间 |
| `count` | int | 总条数 |
| `has_more` | bool | 是否还有更多 |

#### `GET /api/v1/collector` — 采集器状态

| 无请求参数 |
|------|

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"running"` / `"stopped"` / `"degraded"` |
| `running` | bool | 是否运行中 |

#### `POST /api/v1/collector` — 启动采集器

| 无请求体 |
|------|

**响应**：`{"status": "running", "message": "采集器已启动"}`

#### `POST /api/v1/collector/stop` — 停止采集器

| 无请求体 |
|------|

**响应**：`{"status": "stopped", "message": "采集器已停止"}`

#### `GET /api/v1/autonomy` — 自主模式状态

| 无请求参数 |
|------|

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `enabled` | bool | 自主模式是否启用 |
| `paused_until` | string? | 暂停截止时间 ISO8601（null=未暂停） |
| `paused` | bool | 是否已暂停 |

```json
{ "enabled": true, "paused_until": null }
```

#### `POST /api/v1/autonomy/pause` — 暂停自主模式

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `hours` | float | ✓ | 暂停时长（≥0.5 小时） |

> Dashboard 固定传 `hours=1`。

**请求体**：`{"hours": 2.0}`  
**响应**：同 GET /autonomy 格式

#### `POST /api/v1/autonomy/resume` — 恢复自主模式

| 无请求体 |
|------|

**响应**：同 GET /autonomy 格式

### 交互细节

- **首次加载**：`Promise.allSettled` 并行请求 7 个 API（health, model-status, focus/trend, activities/current, intervention/history, collector, autonomy），任一失败不阻塞其余数据展示。
- **实时更新**：
  - `activity_update` WebSocket 事件 → 更新「当前活动」卡片。
  - `intervention` WebSocket 事件 → 在「近期干预记录」列表顶部插入新干预。
- **采集器开关**：toggle 按钮在「启动采集」和「停止采集」间切换，操作期间按钮 disabled。
- **自主控制**：「暂停 1 小时」使用固定参数 `hours=1`。

---

## 4. 专注分析 (`Focus.tsx`)

**路由**：`/focus`  
**用途**：查看每日专注会话详情、7 天趋势图表，并为每个会话提交反馈标注。

### 组件结构

```
Focus
├── header "专注分析 · 查看专注会话与趋势"
├── 日期选择器 + 刷新按钮
├── error-box (条件渲染)
├── KPI Row (总专注时长/专注次数/平均评分/最长专注)
├── card: 7 天专注趋势 (CSS 柱状图)
└── card: 专注会话列表
    └── 每个会话: 日期+时间+时长+应用+评分+切换次数+反馈表单
```

### 后端接口详解

#### `GET /api/v1/focus` — 当日专注会话

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `date` | string | 今天 | 目标日期 `YYYY-MM-DD` |

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | string | 查询日期 |
| `sessions[]` | array | 专注会话列表 |
| `sessions[].id` | string | 会话 UUID |
| `sessions[].start_time` | string | 开始时间 ISO8601 |
| `sessions[].end_time` | string | 结束时间 ISO8601 |
| `sessions[].session_type` | string | 会话类型：`"focus"` / `"distraction"` / `"neutral"` |
| `sessions[].dominant_app` | string | 主要应用进程名 |
| `sessions[].focus_score` | float | 专注评分（0-100） |
| `sessions[].switch_count` | int | 窗口切换次数 |
| `session_count` | int | 会话总数 |

```json
{
  "date": "2026-07-23",
  "sessions": [
    {
      "id": "uuid",
      "start_time": "2026-07-23T09:00:00Z",
      "end_time": "2026-07-23T10:30:00Z",
      "session_type": "focus",
      "dominant_app": "code.exe",
      "focus_score": 85.5,
      "switch_count": 3
    }
  ],
  "session_count": 8
}
```

#### `GET /api/v1/focus/trend` — 专注趋势

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `days` | int | 7 | 回溯天数（范围 1–90） |

> Focus 页面固定传 `days=7`。

**响应字段**：同 Dashboard 章节所述，此外前端还兼容：`daily_data[]`（替代 `daily[]` 的别名字段）。

#### `POST /api/v1/focus/{session_id}/feedback` — 提交专注反馈

**路径参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | ✓ | 专注会话 UUID |

**请求体**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `label` | string | ✓ | 用户自评标签：`"focus"` / `"distracted"` / `"mixed"` |
| `score` | int | ✓ | 自评分数 1–5（1-2 分心，4-5 专注，3 不确定） |
| `task_type` | string | 否 | 任务类型：`"coding"` / `"writing"` / `"study"` / `"meeting"` / `"admin"` / `"creative"` / `"other"` |

```json
{ "label": "focus", "score": 4, "task_type": "coding" }
```

**响应**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 反馈 UUID |
| `user_id` | int | 用户 ID |
| `session_id` | string | 关联会话 UUID |
| `label` | string | 同请求 |
| `score` | int | 同请求 |
| `task_type` | string? | 同请求 |
| `created_at` | string | 创建时间 |

### 交互细节

- **日期联动**：切换日期自动刷新会话列表和趋势图。
- **趋势图**：纯 CSS 柱状图（无图表库依赖），当日高亮（蓝色柱），其他灰色。双柱分别展示专注/分心。
- **反馈表单**：每个会话行内展开，支持重复提交（已保存的会话按钮文字变为"已保存，可更新"）。
- **评分说明**：1–2 分用于分心标签，4–5 分用于专注标签，3 分或混合只用于不确定性评估——这是后端 `EffectivenessService` 的训练数据约束，前端在表单底部以灰色小字展示。

---

## 5. 活动日志 (`Activities.tsx`)

**路由**：`/activities`  
**用途**：查看所有窗口活动记录的明细列表，支持筛选、搜索和分页。

### 组件结构

```
Activities
├── header "活动日志"
├── error-box (条件渲染)
├── card: 当前活动
├── card: 筛选栏 (开始日期/结束日期/Debug开关/搜索框)
└── card: 数据表格 + 分页栏
```

### 后端接口详解

#### `GET /api/v1/activities` — 分页查询活动事件

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `page` | int | 1 | 页码（从 1 开始） |
| `page_size` | int | 50 | 每页条数（范围 1–200） |
| `start_date` | string | 7 天前 | 起始日期 `YYYY-MM-DD` |
| `end_date` | string | 今天 | 结束日期 `YYYY-MM-DD` |

> 前端固定 `page_size=20`，搜索为前端本地过滤（非后端搜索）。

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `items[]` | array | 活动事件列表 |
| `page` | int | 当前页码 |
| `page_size` | int | 每页条数 |
| `total` | int | 总条数 |
| `has_more` | bool | 是否还有下一页 |
| `next_cursor` | string? | 游标（预留 keyset pagination） |

> `items[]` 中每条同 `GET /activities/current` 的字段格式。

```json
{
  "items": [ { "id": "uuid", "user_id": 1, "timestamp": "...", "duration_s": 300.0, "event_type": "window_snapshot", "data": { "app_name": "VS Code", "window_title": "main.py", "process_name": "code.exe", "is_idle": false } } ],
  "page": 1,
  "page_size": 50,
  "total": 1440,
  "has_more": true
}
```

#### `GET /api/v1/activities/current` — 当前活动窗口

同 Dashboard 章节所述。无活动时返回 404。

### 交互细节

- **前端分页**：`total` 用于计算 `totalPages = Math.ceil(total / PAGE_SIZE)`。
- **前端搜索**：本地按 `app_name`、`process_name`、`window_title` 包含关键词过滤，修改搜索条件自动回第 1 页。
- **Debug 模式**：勾选「显示保留期内原始字段」后表格多出 `event_type` 和 `id` 前缀列。
- **状态 badge**：`active` 绿、`idle` 蓝、`away` 黄、`offline` 红。

---

## 6. 行为洞察 (`Analytics.tsx`)

**路由**：`/analytics`  
**用途**：深度行为分析，包含模式分析、个人画像、拖延归因、模型状态四个子 Tab。

### 组件结构

```
Analytics
├── header "行为洞察"
├── tabs: 模式分析 | 个人画像 | 拖延归因 | 模型状态
├── 时间范围选择器 (7/14/30/90 天)
│
├── [Tab 1] 模式分析
│   ├── card: 高切换时段
│   ├── card: 触发应用 Top
│   └── card: 基线对比
├── [Tab 2] 个人画像
│   ├── KPI Row (专注高峰/效率应用/平均专注块/触发应用)
│   └── card: 详细画像 table
├── [Tab 3] 拖延归因
│   └── card: 归因结果 + CBT 技术 + 证据
└── [Tab 4] 模型状态
    └── card: ML 模型状态详情
```

### 后端接口详解

#### `GET /api/v1/analytics/patterns` — 分心模式分析

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `days` | int | 14 | 分析窗口天数（范围 1–90） |

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `high_switch_periods[]` | array | 高切换时段列表 |
| `high_switch_periods[].hour` | int | 小时 (0-23) |
| `high_switch_periods[].switch_count` | int | 该小时切换次数 |
| `trigger_apps[]` | array | 触发分心的应用列表 |
| `trigger_apps[].app` | string | 应用进程名 |
| `trigger_apps[].count` | int | 触发次数 |
| `heatmap` | int[][] | 24×7 热力图矩阵 |
| `total_sessions` | int | 分析的总会话数 |
| `distraction_ratio` | float | 分心比例（0-1） |

```json
{
  "high_switch_periods": [ { "hour": 14, "switch_count": 45 } ],
  "trigger_apps": [ { "app": "bilibili.exe", "count": 12 } ],
  "heatmap": [[0,0,0,0,0,0,0], [1,2,0,0,0,0,0], ...],
  "total_sessions": 168,
  "distraction_ratio": 0.25
}
```

> 前端兼容新旧两种字段名：`period`/`label`/`intensity`/`level` → `high_switch_periods[]`，`app_name`/`name` → `trigger_apps[]`。

#### `GET /api/v1/analytics/baseline` — 个人行为基线

| 无请求参数 |
|------|

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int | 用户 ID |
| `created_at` | string | 基线创建时间 |
| `updated_at` | string | 最后更新时间 |
| `total_days` | int | 累计统计天数 |
| `total_samples` | int | 总采样数 |
| `features` | string[] | 特征字段列表 |
| `avg_focus_min` | float? | 平均每日专注分钟 |
| `avg_switches_per_day` | float? | 平均每日切换次数 |
| `productivity_score` | float? | 综合效率评分 |

```json
{
  "user_id": 1,
  "created_at": "2026-07-17T00:00:00Z",
  "updated_at": "2026-07-23T00:00:00Z",
  "total_days": 30,
  "total_samples": 1320,
  "features": ["unique_app_count", "switch_frequency"],
  "avg_focus_min": 185.2,
  "avg_switches_per_day": 42,
  "productivity_score": 72.5
}
```

#### `GET /api/v1/analytics/profile` — 行为画像

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `days` | int | 30 | 分析窗口天数（范围 1–365） |

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `peak_focus_hours[]` | array | 专注峰值时段 |
| `peak_focus_hours[].hour` | int | 小时数 (0-23) |
| `peak_focus_hours[].avg_score` | float | 该小时平均评分 |
| `top_apps[]` | array | 高频使用应用 |
| `top_apps[].app` | string | 进程名 |
| `top_apps[].total_min` | float | 总使用分钟 |
| `avg_focus_block_min` | float | 平均专注块时长（分钟） |
| `distraction_triggers[]` | array | 分心触发应用 |
| `distraction_triggers[].app` | string | 进程名 |
| `distraction_triggers[].count` | int | 触发次数 |
| `total_events_analysed` | int | 分析的事件总数 |
| `profile_date` | string | 画像生成日期 |
| `peak_focus` | string? | 专注高峰（文字描述） |
| `productivity_apps` | string[]? | 生产力应用列表 |
| `trigger_apps` | string[]? | 触发应用列表 |
| `details` | object? | 详细指标键值对 |

```json
{
  "peak_focus_hours": [ { "hour": 9, "avg_score": 82.1 }, { "hour": 15, "avg_score": 78.5 } ],
  "top_apps": [ { "app": "code.exe", "total_min": 1205.5 } ],
  "avg_focus_block_min": 45.2,
  "distraction_triggers": [ { "app": "bilibili.exe", "count": 15 } ],
  "total_events_analysed": 43200,
  "profile_date": "2026-07-23"
}
```

#### `POST /api/v1/analytics/attribution` — 触发拖延归因分析

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `date` | string | 否 | 日期 `YYYY-MM-DD`（默认今天） |
| `force` | bool | 否 | 强制重新分析（绕过缓存，默认 false） |

> 速率限制：2 次/分钟，20 次/天。

```json
{ "date": "2026-07-23", "force": false }
```

**响应字段**（兼容两种格式）：

*格式一（新）*：

| 字段 | 类型 | 说明 |
|------|------|------|
| `results[]` | array | 多个归因结果 |
| `results[].procrastination_type` | string | 拖延类型：`"task_aversion"` / `"impulsivity"` / `"decisional"` / `"perfectionism"` / `"emotional_regulation"` |
| `results[].confidence` | string\|float | 置信度 |
| `results[].cbt_technique` | string? | 推荐 CBT 技术 |
| `results[].evidence` | string? | 行为证据描述 |

*格式二（旧/单结果）*：

| 字段 | 类型 | 说明 |
|------|------|------|
| `assessment.procrastination_types` | string[] | 识别到的拖延类型列表 |
| `assessment.type_confidence` | object | 类型→置信度映射 |
| `assessment.cognitive_distortions` | string[] | 认知扭曲列表 |
| `assessment.cbt_technique` | string? | 推荐 CBT 技术 |
| `assessment.response_text` | string | 分析描述文本 |
| `assessment.next_action` | string | 建议下一步 |
| `source` | string | LLM 来源：`"deepseek"` / `"ollama"` / `"rule_engine"` |
| `cached` | bool | 是否来自缓存 |
| `meta.degraded` | bool | LLM 是否降级 |

```json
{
  "assessment": {
    "types": ["impulsivity", "task_aversion"],
    "confidence": { "impulsivity": 0.78, "task_aversion": 0.45 },
    "recommended_technique": "stimulus_control"
  },
  "source": "deepseek",
  "cached": false,
  "meta": { "degraded": false }
}
```

#### `GET /api/v1/analytics/model-status` — ML 模型状态

同 Dashboard 章节所述。

### 交互细节

- **Tab 切换**：四个 Tab 互斥，切换时仅加载对应数据。
- **时间范围**：「模式分析」和「个人画像」共用时间范围选择器（7/14/30/90 天），切换天数自动重新请求。
- **归因分析**：点击按钮触发 LLM 推理，参数 `{date, force}`。前端兼容新旧两种返回格式——优先展示 `results[]` 数组，其次展示顶层字段。

---

## 7. 报告中心 (`Reports.tsx`)

**路由**：`/reports`  
**用途**：查看日报和周报，包含时段分布图、应用使用排行、分心分析、周环比等。

### 组件结构

```
Reports
├── header "报告中心"
├── tabs: 日报 | 周报
│
├── [日报 Tab]
│   ├── 日期选择器
│   ├── KPI Row (总专注时长/专注次数/分心次数/专注评分)
│   ├── card: 时段分布 (24h CSS 柱状图)
│   ├── card: 应用使用 (table)
│   └── card: 分心分析 (table)
│
└── [周报 Tab]
    ├── 周一日期选择器
    ├── KPI Row (周总专注/总次数/总分心/日均评分)
    ├── card: 每日专注时长 (7 天 CSS 柱状图)
    ├── card: 每日详情 table
    └── card: 周环比 (4 个变化指标)
```

### 后端接口详解

#### `GET /api/v1/reports/daily` — 每日报告

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `date` | string | 今天 | 报告日期 `YYYY-MM-DD` |

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string? | 报告 UUID |
| `user_id` | int | 用户 ID |
| `date` | string | 报告日期 |
| `total_focus_min` / `total_focus_minutes` | float | 总专注分钟（兼容两种字段名） |
| `total_distraction_min` | float | 总分心分钟 |
| `focus_score` | float | 综合专注评分 |
| `top_apps[]` | array | 高频应用 `[{app, minutes}]` |
| `switch_frequency` | float | 窗口切换频率 |
| `pattern_summary` | string | 模式摘要文字 |
| `total_sessions` | int? | 专注会话总数 |
| `total_distractions` | int? | 分心事件总数 |
| `hourly_distribution` | object? | 24 小时分布 `{hour: minutes}` |
| `app_usage[]` | array? | 应用使用详情 |
| `app_usage[].app` / `name` | string | 应用名（兼容） |
| `app_usage[].duration_minutes` | float | 使用时长分钟 |
| `app_usage[].category` | string | 分类：`"productive"` / `"neutral"` / `"distracting"` |
| `distraction_analysis[]` | array? | 分心分析 |
| `distraction_analysis[].type` / `name` | string | 分心类型（兼容） |
| `distraction_analysis[].count` | int | 发生次数 |
| `distraction_analysis[].total_duration` | float | 总持续时长 |

```json
{
  "date": "2026-07-23",
  "summary": "...",
  "focus_time_min": 180.5,
  "distraction_time_min": 45.2,
  "top_productivity_apps": ["code.exe"],
  "interventions": 3,
  "focus_score_avg": 72.3
}
```

#### `GET /api/v1/reports/weekly` — 每周报告

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `week_start` | string | 本周一 | 周起始日期 `YYYY-MM-DD`（周一） |

> 前端通过 `mondayOf(new Date())` 自动计算本周一。

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `week_start` | string | 周起始日期 |
| `week_end` | string | 周结束日期 |
| `daily_reports[]` | array | 每日报告列表（同日报格式） |
| `averages.avg_focus_min` | float? | 日均专注分钟 |
| `averages.avg_distraction_min` | float? | 日均分心分钟 |
| `averages.avg_focus_score` | float? | 日均评分 |
| `averages.avg_switch_frequency` | float? | 日均切换频率 |
| `trend.focus_min_delta_pct` | float? | 专注时长变化百分比 |
| `trend.focus_score_delta` | float? | 评分变化 |
| `trend.direction` | string? | 趋势方向：`"up"` / `"down"` / `"stable"` |
| `week_number` | int | ISO 周号 |
| `intervention_effectiveness` | object? | 干预有效性数据 |
| `total_focus_minutes` | float? | 周总专注分钟 |
| `total_sessions` | int? | 周总专注次数 |
| `total_distractions` | int? | 周总分心数 |
| `avg_focus_score` | float? | 周均专注评分 |
| `daily_summary[]` | array | 每日摘要 |
| `daily_summary[].date` | string | 日期 |
| `daily_summary[].focus_minutes` | float | 专注分钟 |
| `daily_summary[].sessions` | int | 会话数 |
| `daily_summary[].distractions` | int | 分心数 |
| `daily_summary[].focus_score` | float | 评分 |
| `week_over_week` | object? | 环比数据 |
| `week_over_week.focus_change_pct` | float? | 专注变化% |
| `week_over_week.sessions_change_pct` | float? | 次数变化% |
| `week_over_week.distractions_change_pct` | float? | 分心变化% |
| `week_over_week.score_change_pct` | float? | 评分变化% |

### 交互细节

- **日报**：默认显示当天。切换日期自动加载。
- **周报**：默认显示本周一。`mondayOf()` 函数根据当前日期计算周一起始日。
- **周环比**：四个指标分别计算变化百分比，正向（专注↑）绿色、负向（分心↑）红色，趋势箭头 ↑↓。
- **柱状图**：纯 CSS 实现，无第三方图表库。

---

## 8. 干预中心 (`Intervention.tsx`)

**路由**：`/intervention`  
**用途**：管理智能干预——手动触发干预、查看干预历史、响应当前干预、提交干预反馈。

### 组件结构

```
Intervention
├── header "干预中心"
├── error-box (条件渲染)
├── card: 手动触发干预 (温和/标准/严格)
├── card: 最新干预 (条件渲染) + 响应按钮 (接受/忽略/关闭)
└── card: 干预历史 + 评价表单 (有用/一般/无效)
```

### 后端接口详解

#### `POST /api/v1/intervention/trigger` — 手动触发干预

绕过节流限制。

**查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `intensity` | string | 否 | 干预强度：`"gentle"`（温和）/ `"standard"`（标准）/ `"strict"`（严格） |

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `intervention` | object? | 干预对象（null=被跳过） |
| `intervention.id` | string | 干预 UUID |
| `intervention.intervention_type` | string | 干预类型：`"task_breakdown"` / `"nudge"` / `"environment_optimization"` / `"smart_prioritization"` |
| `intervention.title` | string | 干预标题（含 emoji，如 🔔） |
| `intervention.message` | string | 干预消息正文 |
| `intervention.dismissible` | bool | 是否可关闭 |
| `intervention.cbt_technique` | string? | 关联的 CBT 技术 |
| `intervention.created_at` | string | 创建时间 |
| `skipped` | bool | 是否被跳过 |
| `skip_reason` | string? | 跳过原因（仅 skipped=true 时） |

```json
{
  "intervention": {
    "id": "uuid",
    "intervention_type": "environment_optimization",
    "title": "🔔 环境优化建议",
    "message": "工作环境中存在较多干扰源。关闭无关标签页…",
    "dismissible": true,
    "cbt_technique": "stimulus_control",
    "created_at": "2026-07-23T14:30:00Z"
  },
  "skipped": false
}
```

跳过时：`{"intervention": null, "skipped": true, "skip_reason": "未检测到显著的拖延模式"}`

#### `POST /api/v1/intervention/{intervention_id}/response` — 记录用户响应

**路径参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `intervention_id` | string | ✓ | 干预 UUID |

**查询参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `response` | string | ✓ | 用户选择：`"accepted"`（接受）/ `"ignored"`（忽略）/ `"dismissed"`（关闭） |
| `latency_s` | float | 否 | 响应延迟秒数（从干预推送到用户点击的时间差，默认 0） |

> 前端传 `latencyS = 0`（简化实现，未记录实际延迟）。

**响应**：`{"status": "ok", "intervention_id": "uuid", "user_response": "accepted"}`

#### `POST /api/v1/intervention/{intervention_id}/feedback` — 提交干预反馈

**路径参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `intervention_id` | string | ✓ | 干预 UUID |

**请求体**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `rating` | string | ✓ | 评价：`"effective"`（有用）/ `"neutral"`（一般）/ `"ineffective"`（无效） |
| `comment` | string | 否 | 补充评论文字 |

```json
{ "rating": "effective", "comment": "很有帮助" }
```

**响应**：`{"status": "ok"}`

#### `GET /api/v1/intervention/history` — 干预历史

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `days` | int | 7 | 回溯天数（范围 1–90） |

**响应**：同 Dashboard 章节所述。返回 `items[]`、`count`、`has_more`。

### 交互细节

- **实时推送**：WebSocket `intervention` 事件到达时，自动在列表顶部插入新干预记录。
- **干预强度**：gentle=温和(蓝), standard=标准(黄), strict=严格(红)。
- **评分表单**：仅对「已响应 + 未评价」（`user_response` 非空且 `feedback_rating` 为空）的干预显示"评价"按钮。
- **历史范围**：4 个时间段 Tab（7/14/30/90 天），切换自动重新加载。

---

## 9. 专家面板 (`Panel.tsx`)

**路由**：`/panel`  
**用途**：触发多智能体协作分析（6 专家 + 1 协调者），查看拖延类型分析、CBT 技术推荐、专家讨论记录。

### 组件结构

```
Panel
├── header "专家面板 · 多智能体协作分析"
├── error-box (条件渲染)
├── 操作按钮: "运行专家面板" / "查看上次结果"
├── 降级提示 card (condition: result.degraded)
└── 结果面板
    ├── card: 拖延类型分析 (置信度进度条)
    ├── card: 推荐 CBT 技术
    ├── card: 分析依据
    └── card: 专家讨论记录 (多角色彩色边框)
```

### 后端接口详解

#### `POST /api/v1/panel/today` — 触发今日多专家会诊

运行完整的 LangGraph 面板流水线（数据分析→3 位归因专家→综合主持人→批评家）。LLM 不可用时降级并标记 `degraded: true`。

> 速率限制：1 次/小时，3 次/天。这是全系统最贵的 LLM 调用（一次触发 6+ 次 API 调用）。

| 无请求参数 |
|------|

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `types` | string[] | 识别到的拖延类型列表 |
| `confidence` | object | 类型 → 置信度映射（0-1） |
| `technique` | string? | 推荐 CBT 技术：`"behavioral_experiment"` / `"cognitive_restructuring"` / `"stimulus_control"` / `"goal_setting"` / `"graded_exposure"` / `"mindfulness"` |
| `rationale` | string | 分析依据说明文字 |
| `dissent` | string[] | 异议意见列表 |
| `transcript[]` | array | 专家讨论记录 |
| `transcript[].role` | string | 专家角色：`"analyst"` / `"psychologist"` / `"coach"` / `"strategist"` / `"facilitator"` / `"evaluator"`（或中文名） |
| `transcript[].content` | string | 发言内容 |
| `transcript[].round` | int | 讨论轮次 |
| `escalated` | bool | 是否升级（触发冲突仲裁） |
| `call_count` | int | LLM 调用总次数 |
| `degraded` | bool | 是否降级模式 |
| `meta.degraded` | bool | 冗余降级标记 |

```json
{
  "types": ["impulsivity"],
  "confidence": { "impulsivity": 0.82 },
  "technique": "stimulus_control",
  "rationale": "根据今日行为数据分析…",
  "dissent": [],
  "transcript": [
    { "role": "分析师", "content": "检测到 2.3 次/分钟的应用切换…", "round": 0 },
    { "role": "CBT归因专家", "content": "可能机制：冲动性注意力偏移…", "round": 1 },
    { "role": "综合主持人", "content": "裁决类型=impulsivity", "round": 2 }
  ],
  "escalated": false,
  "call_count": 6,
  "degraded": false,
  "meta": { "degraded": false }
}
```

#### `GET /api/v1/panel` — 读取最近一次面板结果

只读接口，不触发 LLM 调用。

> 速率限制：10 次/分钟，30 次/天。

| 无请求参数 |
|------|

**响应**：同 POST 格式。无结果时返回 404。

### 后端专家角色（6 专家 + 1 协调者）

| 角色 | 英文 | 前端颜色 | 功能 |
|------|------|---------|------|
| 数据分析师 | analyst | #4F6BF6 (蓝) | 分析行为数据，识别模式 |
| CBT 归因专家 / 心理学家 | psychologist | #8B5CF6 (紫) | CBT 专业视角，诊断拖延类型 |
| 干预策略师 / 教练 | coach | #22C55E (绿) | 提供行动建议和习惯养成策略 |
| 策略师 | strategist | #F59E0B (橙) | 制定分心应对策略 |
| 综合主持人 | facilitator | #06B6D4 (青) | 协调讨论、总结共识 |
| 批评家 / 评估师 | evaluator | #EC4899 (粉) | 评估方案有效性、质疑缺陷 |

### 交互细节

- **运行 vs 查看**：两个按钮分别触发生成（POST，消耗 LLM 配额）和读取（GET，只读缓存），互不干扰。
- **降级模式**：LLM 服务不可用时自动回退到本地规则引擎，面板顶部显示黄色警告 card。
- **置信度颜色**：≥70% 绿、≥40% 黄、<40% 红，进度条同样渐变。

---

## 10. AI 对话 (`Chat.tsx`)

**路由**：`/chat`  
**用途**：与 MindFlow 智能助手进行自由对话，支持多会话管理、工具调用展示、证据引用查看。

### 组件结构

```
Chat
├── header "AI 对话 · MindFlow 智能助手"
├── error-box (条件渲染)
└── 双栏布局
    ├── 左侧: 会话列表 (200px) + "新对话" 按钮
    └── 右侧: 聊天主区
        ├── 消息列表 (scrollable)
        │   ├── 用户气泡 (蓝色右对齐)
        │   └── AI 气泡 (灰色左对齐)
        │       └── 元信息 badges: 降级/工具/证据
        └── 输入区: textarea (auto-resize) + "发送" 按钮
```

### 后端接口详解

#### `POST /api/v1/chat` — 发送消息

> 速率限制：5 次/分钟，60 次/天。含安全门控（`CrisisDetector` 在 LLM 前硬拦截自伤/自杀内容）。

**请求体**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `message` | string | ✓ | 用户消息（非空） |
| `session_id` | string | 否 | 会话 ID（UUID）。新会话时不传，后端自动创建并返回 |

```json
{ "message": "我今天专注度怎么样？", "session_id": "uuid (可选)" }
```

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | AI 回复正文 |
| `session_id` | string | 会话 UUID（新会话时前端保存用于后续请求） |
| `tools_used` | string[] | 使用的工具列表（如 `["get_focus_sessions"]`） |
| `evidence_cited` | string \| string[] \| bool | 引用的行为证据（字符串/数组/布尔） |
| `degraded` | bool | 是否降级运行 |

```json
{
  "answer": "根据今天的数据，你上午9-11点专注度最高…",
  "session_id": "uuid",
  "tools_used": ["get_focus_sessions", "get_productivity_ratio"],
  "evidence_cited": "今日专注会话: 8个, 平均分: 72.5",
  "degraded": false
}
```

#### `GET /api/v1/chat/sessions` — 最近会话列表

| 无请求参数 |
|------|

**响应**：会话对象数组

| 字段 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话 UUID |
| `last_message_at` | string | 最后一条消息时间 |

> 旧版字段兼容：`title`（标题）、`message_count`（消息数）、`created_at`（创建时间）。

```json
[{ "session_id": "uuid", "last_message_at": "2026-07-23T15:00:00Z" }]
```

#### `GET /api/v1/chat/{session_id}/messages` — 会话历史消息

**路径参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | ✓ | 会话 UUID |

**响应**：消息对象数组

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 消息 UUID |
| `user_id` | int | 用户 ID |
| `session_id` | string | 会话 UUID |
| `role` | string | `"user"` 或 `"assistant"` |
| `content` | string | 消息正文 |
| `created_at` | string | 创建时间 |

### Chat 安全机制

后端 `ChatService` 内置两层安全防护：
1. **`CrisisDetector`** — 在 LLM 调用前硬门控检测自伤/自杀等危机内容，命中时直接返回预设安全响应（不送 LLM）。
2. **`EvidenceBundleBuilder`** — 构建行为证据上下文，确保 AI 回复有据可依。

### 交互细节

- **自动创建会话**：首次发消息时 `session_id` 为空，后端返回后自动设置 `activeSessionId` 并刷新会话列表。
- **多会话**：左侧列表支持切换历史会话，每条消息独立加载。显示为 `"会话 {session_id.slice(0, 8)}"`。
- **输入框**：`textarea` 自动撑高（最大 200px），Enter 发送、Shift+Enter 换行。
- **消息元信息**：AI 回复下方显示三个可选 badge：
  - 降级模式（黄色 `badge-warning`）
  - 工具使用（蓝色 `badge-info`，如 `工具: get_focus_sessions`）
  - 证据引用（紫色 `badge-primary`，如 `证据: xyz`）

---

## 11. 系统设置 (`Settings.tsx`)

**路由**：`/settings`  
**用途**：MindFlow 全部配置管理中心，7 大模块。

### 组件结构

```
Settings
├── header "系统设置"
├── error-box
├── card: 系统信息 (状态/版本/数据库/迁移)
├── card: 隐私行为采集 (toggle × 2 + 保留天数 + 配对码 + 数据清除)
├── card: 数据采集 (采集器开关)
├── card: 自主控制 (暂停/恢复 + 时长选择)
├── card: 应用分类 (CRUD 规则表 + 未知应用获取)
├── card: 数据导出 (格式/日期范围/下载)
└── card: 偏好设置 (JSON 编辑器 + PUT/PATCH)
```

### 后端接口详解（仅列出 Dashboard 未涉及的）

#### `GET /api/v1/telemetry/status` — 遥测状态

| 无请求参数 |
|------|

**响应字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `preferences.input_telemetry_enabled` | bool | 鼠标键盘聚合统计开关 |
| `preferences.browser_tracking_enabled` | bool | 浏览器域名统计开关 |
| `preferences.interaction_retention_days` | int | 输入桶保留天数 |
| `preferences.activity_retention_days` | int | 活动/浏览器片段保留天数 |
| `input_watcher_status` | string | 输入 watcher 状态：`"running"` / `"stopped"` |
| `database_size_bytes` | int | 数据库占用字节 |
| `interaction_bucket_count` | int | 今日输入桶数量 |
| `browser_segment_count` | int | 今日浏览器片段数量 |
| `browser_paired` | bool | 浏览器扩展是否已配对 |
| `last_interaction_at` | string? | 最后输入采集时间 |
| `last_browser_at` | string? | 最后浏览器采集时间 |

#### `PATCH /api/v1/telemetry/preferences` — 更新遥测偏好

**请求体**（部分更新）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input_telemetry_enabled` | bool | 否 | 是否采集键盘鼠标聚合 |
| `browser_tracking_enabled` | bool | 否 | 是否采集浏览器域名 |
| `interaction_retention_days` | int | 否 | 输入保留天数（1/3/7/14/30） |
| `activity_retention_days` | int | 否 | 活动保留天数（7/14/30/60/90） |

#### `POST /api/v1/telemetry/browser/pairing-code` — 生成浏览器配对码

| 无请求体 |
|------|

**响应**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | string | 6 位配对码（字母+数字） |
| `expires_at` | string | 过期时间 ISO8601（5 分钟有效） |

#### `DELETE /api/v1/telemetry/data` — 清除遥测数据

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `scope` | string | ✓ | 清除范围：`"interaction"` / `"browser"` / `"feedback"` / `"all"` |

**响应**：`{"deleted": 123}`

#### `GET /api/v1/app-classifications` — 获取分类规则列表

| 无请求参数 |
|------|

**响应**：规则对象数组（按 priority 降序）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 规则 UUID |
| `user_id` | int | 用户 ID |
| `process_name` | string | 进程名 |
| `window_title_pattern` | string? | 窗口标题 SQL LIKE 模式（`%` = 通配） |
| `category` | string | 分类：`"productive"` / `"neutral"` / `"distracting"` / `"unknown"` |
| `priority` | int | 优先级 0-100（越大越优先） |
| `created_at` | string | 创建时间 |
| `updated_at` | string | 更新时间 |

#### `POST /api/v1/app-classifications` — 添加分类规则

**请求体**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `process_name` | string | ✓ | 进程名（≤255 字符） |
| `window_title_pattern` | string? | 否 | 窗口标题模式（SQL LIKE 语法，`%` 通配） |
| `category` | string | ✓ | 分类（同上） |
| `priority` | int | 否 | 优先级 0-100（默认 0） |

```json
{ "process_name": "bilibili.exe", "window_title_pattern": null, "category": "distracting", "priority": 10 }
```

**响应**：创建的规则对象（201），含生成的 `id`。

#### `DELETE /api/v1/app-classifications/{rule_id}` — 删除分类规则

| 路径参数 | 类型 | 必填 | 说明 |
|---------|------|------|------|
| `rule_id` | string | ✓ | 规则 UUID |

> 幂等操作，规则不存在也返回 204。

#### `GET /api/v1/app-classifications/unknown-apps` — 未分类应用列表

| 无请求参数 |
|------|

**响应**：进程名字符串数组（如 `["obsidian.exe", "notion.exe"]`）。旧版返回 `[{process_name, count, last_seen}]` 对象数组。

#### `GET /api/v1/preferences` — 获取偏好设置

| 无请求参数 |
|------|

**响应**：任意 JSON 对象（自由 schema，前端定义键名，≤64KB，≤8 层嵌套）。

```json
{ "autonomy": { "enabled": true }, "theme": "dark" }
```

#### `PUT /api/v1/preferences` — 全量替换偏好

**请求体**：完整偏好 JSON 对象（任意键值，≤64KB，≤8 层嵌套）。

**响应**：同请求体。

#### `PATCH /api/v1/preferences` — 合并更新偏好

**请求体**：部分偏好字段（与现有偏好深度合并）。

**响应**：合并后的完整偏好对象。

#### `GET /api/v1/export` — 导出活动数据

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `fmt` | string | `"csv"` | 导出格式：`"csv"` 或 `"json"` |
| `start` | string | 30 天前 | 起始日期 ISO8601 |
| `end` | string | 现在 | 结束日期 ISO8601 |

> 最大导出范围 90 天。返回文件下载流（`Content-Disposition: attachment`）。前端通过 `Blob` + `URL.createObjectURL` 触发浏览器下载。

### 交互细节

- **配对码**：生成后显示大字号 6 位码（`fontSize: 30, letterSpacing: 8`），5 分钟有效期倒计时。
- **数据清除**：每个清除按钮有 `window.confirm` 二次确认（`"确定清除{scope}数据吗？此操作无法撤销。"`），防止误操作。
- **偏好设置**：先 `JSON.parse` 校验格式，通过后才发送。无效 JSON 时 `setError("JSON 格式无效")`，无效类型时 `setError("JSON 必须是对象")`。PUT 按钮全量覆盖，PATCH 按钮增量合并。
- **应用分类**：添加规则后自动刷新列表。删除按钮直接调用 `deleteClassification(id)` 并从本地 state 移除该项（乐观更新）。

---

## 12. 公共组件与基础设施

### 12.1 实时通信 (`realtime.ts`)

**`RealtimeClient` 类**：封装 WebSocket 连接管理。

| 特性 | 实现 |
|------|------|
| 连接地址 | `ws://host/api/v1/ws`（查询参数 token 认证） |
| 心跳 | 30s 间隔发送 `{"type":"ping"}`，服务端回复 `{"type":"pong"}` |
| 断线重连 | 指数退避 + 随机抖动：`min(1000 * 2^attempt + random(0~1000), 30000)` ms |
| 状态管理 | idle → connecting → connected → reconnecting → disconnected |
| 事件类型 | `activity_update`、`intervention` |
| 订阅模式 | 基于 `Map<type, Set<listener>>` 的事件总线 |

**WebSocket 消息格式**：
```json
{
  "type": "activity_update | intervention",
  "payload": { ... },
  "timestamp": "2026-07-26T..."
}
```

### 12.2 主题系统 (`theme.css`)

| 设计 Token | 用途 |
|------------|------|
| `--color-primary: #4F6BF6` | 主色调（按钮、链接、高亮） |
| `--color-bg: #F8FAFC` | 页面背景 |
| `--color-bg-elevated: #FFFFFF` | 卡片背景 |
| `--color-bg-inset: #F1F5F9` | 内嵌区域背景 |
| `--color-border: #E2E8F0` | 边框色 |
| `--color-text-primary/secondary/tertiary` | 三级文字色 |
| `--color-success/warning/danger/info` | 语义色 |
| `--sidebar-w: 220px` | 侧边栏宽度 |

**复用的 CSS 工具类**：
- `.card` — 白色圆角卡片 (`border-radius: 12px`)
- `.stat-card` / `.kpi-row` — 指标卡片和 4 列响应式网格
- `.badge-*` — 5 色状态标签 (primary/success/warning/danger/info)
- `.btn` / `.btn-ghost` / `.btn-danger` / `.btn-sm` — 按钮系统
- `.tabs` / `.tab` / `.tab.active` — Tab 切换
- `.chat-bubble` / `.chat-user` / `.chat-ai` — 聊天气泡
- `.flex` / `.flex-between` / `.gap8` / `.gap16` / `.mb16` / `.mb24` / `.mt8` / `.mt16` — 布局工具类
- `.spinner` — 加载动画 (`0.6s linear infinite`)
- `.error-box` — 红色错误提示框

### 12.3 API 客户端 (`api.ts`)

| 特性 | 实现 |
|------|------|
| 类型安全 | `openapi-fetch` + `generated/api-schema.ts` |
| 认证 | `credentials: "include"` Cookie 自动携带 |
| 错误处理 | 401 → 清除 `localStorage` marker → 触发 `mindflow:auth-required` 自定义事件 → 页面退回 Login |
| 响应校验 | 运行时类型守卫（`isActivityItem`、`isRecord` 等） |
| 降级 | `toApiError` 统一包装 `ApiError`，支持 RFC 9457 ProblemDetail 解析 |

### 12.4 认证流程 (`main.tsx`)

```
用户打开 → bootstrapFromFragment() 检查 #bootstrap ticket
              ├── 有 ticket → POST /auth/bootstrap → 成功 → 设置 localStorage → 渲染 App
              └── 无 ticket → 直接渲染 App
                                  └── App 检查 localStorage → 有 marker → Layout + 路由
                                                           └── 无 marker → Login 页面
```

---

## 13. WebSocket 协议

### 连接信息

- **端点**：`ws://127.0.0.1:8765/api/v1/ws?token=<token>`
- **认证**：查询参数 token（浏览器 `WebSocket(url)` 不支持自定义头部）
- **心跳**：客户端 30s 发 `{"type":"ping"}`，服务端回复 `{"type":"pong"}`
- **节流**：`activity_update` 最多 2 秒推送一次，状态不变时跳过

### 消息帧类型

**服务端 → 客户端**：

| type | payload 内容 | 触发时机 |
|------|-------------|----------|
| `activity_update` | `{app_name, window_title, process_name, is_idle}` | 活动窗口变化（2s 节流） |
| `intervention` | `{id, intervention_type, title, message, dismissible, cbt_technique}` | 触发干预时 |
| `pong` | `{}` | 回应客户端 ping |

**客户端 → 服务端**：

| type | payload 内容 | 用途 |
|------|-------------|------|
| `ping` | `{}` | 心跳保活（30s 间隔） |

---

## 14. 通用约定与错误码

### 通用约定

- 所有 REST 端点前缀 `/api/v1`
- 所有时间戳 ISO8601 UTC 格式
- 认证：Bearer Token（通过文件系统 `platformdirs` 读取），除 `/health` 外均需
- 安全：Host 验证中间件仅接受 `localhost`/`127.0.0.1`/`[::1]`

### 错误码速查 (RFC 9457)

| type_slug | HTTP | 说明 |
|-----------|------|------|
| `collector-not-running` | 503 | 采集器未运行 |
| `not-found` | 404 | 资源不存在（活动/报告/干预） |
| `validation-error` | 422 | 参数校验失败（日期格式/分页/偏好大小） |
| `rate-limited` | 429 | 请求频率超限（响应含 `retry_after_seconds`） |
| `auth-required` | 401 | 令牌缺失或无效 |
| `forbidden-host` | 403 | Host 头不是 localhost（防 DNS 重绑定） |
| `internal-error` | 500 | 服务器内部错误（不泄露堆栈） |
| `llm-unavailable` | 503 | LLM 服务不可用（降级到规则引擎） |

**错误响应格式**：
```json
{
  "type": "https://mindflow.app/errors/not-found",
  "title": "Not Found",
  "status": 404,
  "detail": "未找到日期 2026-07-18 的报告",
  "instance": "/api/v1/reports/daily"
}
```

> `type` URI 不可解析——仅作为机器可读的错误标识符。`detail` 使用中文，面向用户。

---

### 页面 → API 映射速查总表

| 页面 | 调用的后端 API（完整） |
|------|----------------------|
| **Login** | 不直接调用 API |
| **Dashboard** | `GET /health`, `GET /focus/trend?days=7`, `GET /activities/current`, `GET /analytics/model-status`, `GET /intervention/history?days=7`, `GET /collector`, `POST /collector`, `POST /collector/stop`, `GET /autonomy`, `POST /autonomy/pause {hours:1}`, `POST /autonomy/resume`, `WS /api/v1/ws` |
| **Focus** | `GET /focus?date=`, `GET /focus/trend?days=7`, `POST /focus/{id}/feedback {label,score,task_type}` |
| **Activities** | `GET /activities?page=&page_size=20&start_date=&end_date=`, `GET /activities/current` |
| **Analytics** | `GET /analytics/patterns?days=`, `GET /analytics/baseline`, `GET /analytics/profile?days=`, `POST /analytics/attribution {date,force}`, `GET /analytics/model-status` |
| **Reports** | `GET /reports/daily?date=`, `GET /reports/weekly?week_start=` |
| **Intervention** | `POST /intervention/trigger?intensity=`, `GET /intervention/history?days=`, `POST /intervention/{id}/response?response=&latency_s=`, `POST /intervention/{id}/feedback {rating,comment}`, `WS /api/v1/ws` |
| **Panel** | `POST /panel/today`, `GET /panel` |
| **Chat** | `POST /chat {message,session_id}`, `GET /chat/sessions`, `GET /chat/{id}/messages` |
| **Settings** | `GET /health`, `GET /telemetry/status`, `PATCH /telemetry/preferences`, `POST /telemetry/browser/pairing-code`, `DELETE /telemetry/data?scope=`, `GET /collector`, `POST /collector`, `POST /collector/stop`, `GET /autonomy`, `POST /autonomy/pause {hours}`, `POST /autonomy/resume`, `GET /app-classifications`, `POST /app-classifications`, `DELETE /app-classifications/{id}`, `GET /app-classifications/unknown-apps`, `GET /export?fmt=&start=&end=`, `GET /preferences`, `PUT /preferences`, `PATCH /preferences` |

---

> **文档维护说明**：本文档基于 2026-07-26 代码快照生成。前端代码位于 `frontend/src/`，后端 API 路由位于 `backend-next/src/mindflow/api/routes/`。页面或接口有变更时请同步更新本文档。API 参数以实际运行时 OpenAPI 文档（`http://localhost:8765/docs`）为准。
