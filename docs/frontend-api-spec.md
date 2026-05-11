# 前端 API 对接文档

> 供张皓、杨智杰参考。后端地址：`http://127.0.0.1:8765`

## 后端启动方式

```bash
cd backend
uvicorn mindflow.main:app --host 127.0.0.1 --port 8765
# API 文档: http://127.0.0.1:8765/docs
```

## 通用约定

- 所有 REST 端点前缀 `/api/v1`
- 统一响应格式：`{ code: 0, message: "success", data: {...}, timestamp: 1709251200 }`
- `code === 0` 即成功，非 0 即业务错误
- WebSocket 地址：`ws://127.0.0.1:8765/ws/activities`

## 重要架构变更（2026-05-12）

### 1. 时区
所有时间戳现为**北京时间**，前端直接显示即可，无需转换。

### 2. 应用分类
`/api/v1/activities/today` 返回的 `top_apps` 中，`process_name` 是原始进程名，前端可做展示层映射（如 `WindowsTerminal.exe` → "终端"），但核心分类逻辑在后端完成，无需前端判断。

未来改版（Phase 2）：三层分类器改造完成后，每个应用会带 `category` 字段（`code/document/browser_work/entertainment/social/other`）。

### 3. WebSocket 新增 snapshot 字段

```json
{
  "type": "activity_update",
  "data": {
    "window": {
      "process_name": "claude.exe",
      "window_title": "✳ Apply changes and plan data collection",
      "window_class": "Chrome_WidgetWin_1",
      "timestamp": "2026-05-12T10:30:00"
    },
    "collector_running": true,
    "timestamp": 1747038600,
    "snapshot": {
      "focus_score": 72.5,
      "dominant_app": "claude.exe"
    }
  }
}
```

`snapshot.focus_score` 可能为 `null`（当天无日报），做好空值处理。

### 4. 新增端点

| 端点 | 用途 | 前端场景 |
|------|------|----------|
| `GET /api/v1/health` | 健康检查 | 启动时探活，显示后端是否就绪 |
| `GET /api/v1/data/summary` | 数据概况 | 设置页/隐私面板，展示记录数和 DB 大小 |
| `GET /api/v1/analytics/deviation` | 偏差告警 | Dashboard 告警卡片，显示异常时段和严重度 |
| `GET /api/v1/analytics/clusters` | 行为聚类 | 专注报告页，显示今日行为模式分布 |
| `GET /api/v1/analytics/risk` | 分心风险 | 实时风险指示器（低/中/高），提醒用户 |

首次使用时 ML 模型可能未训练，这些端点返回 `model_available: false`，前端应展示友好占位而非报错。

## 前端建议的 Dashboard 页面结构

1. **首页 Dashboard**
   - 专注得分大数字 + 趋势迷你图
   - 今日应用使用饼图（`/activities/today` 的 `top_apps`）
   - WebSocket 实时当前窗口 + 分析快照
   - 异常告警卡片（`/analytics/deviation`）

2. **专注趋势页**
   - 7/30 天趋势折线图（`/focus/trend?days=30`）
   - 每周对比柱状图（`/reports/weekly`）
   - 行为模式聚类分布（`/analytics/clusters`）

3. **设置页**
   - 采集开关（`POST /collector/start|stop`）
   - 偏好设置表单（`GET|PUT /preferences`）
   - 数据隐私面板（`/data/summary`）
   - 模型训练状态（`/health`）

## 错误处理建议

- 网络不可达 → 显示"后端未启动"
- `code !== 0` → 显示 `message` 字段内容
- `model_available: false` → 显示"模型尚未训练，请采集数据后训练"
- WebSocket 断线 → 自动重连（指数退避，初始 2s，最大 30s）

## 非功能性需求

- 所有时间显示为北京时间，格式建议 `HH:mm` 或 `MM-DD HH:mm`
- 应用名显示建议做一层映射：`WindowsTerminal.exe` → "终端"，`claude.exe` → "Claude Code"
- 专注分数建议用环形进度条，颜色按分数渐变（<50 红，50-70 黄，>70 绿）
- 首次使用无数据时展示引导页，而非空图表
