# V2 模型训练 API

> 本文档覆盖模型中心（Model Center）相关端点的请求/响应合约。
> 基础路径：`/api/v1/analytics`

---

## GET /api/v1/analytics/training-readiness

训练就绪评估。将 V2 特征窗口与显式反馈按时间重叠匹配，报告数据充分性和 7 项质量门禁。

### 请求

浏览器通过启动引导自动获取 `HttpOnly` `SameSite=Strict` 的 `mindflow_session` Cookie，所有 API 请求自动携带。
直接调用时需在本地 root token 生成的 bootstrap ticket 交换后获得会话 Cookie，或使用 `Authorization: Bearer <token>`。

### 响应 `200 OK`

```json
{
  "raw_events": {
    "total_events": 48230,
    "coverage_days": 14,
    "oldest_timestamp": "2026-07-15T08:00:00",
    "newest_timestamp": "2026-07-29T22:00:00"
  },
  "v2_windows": {
    "total": 268,
    "schema_version": 2,
    "date_range_days": 14,
    "eligible_count": 42,
    "matched_focus_count": 28,
    "matched_distract_count": 14,
    "newest_window_start": "2026-07-29T21:00:00"
  },
  "feedback_labels": {
    "focus": 35,
    "distract": 18,
    "mixed": 3,
    "total": 56
  },
  "trainable": true,
  "trainable_window_count": 42,
  "trainable_class_count": 2,
  "evaluable": true,
  "evaluable_explicit_count": 42,
  "evaluable_date_count": 5,
  "baseline_ready": true,
  "current_mode": "rule_engine_only",
  "gates": [
    {
      "key": "minimum_days",
      "label": "最少反馈天数",
      "passed": true,
      "status": "passed",
      "actual": "5",
      "threshold": ">= 1",
      "message": "反馈天数满足最低要求",
      "blocker_code": ""
    },
    {
      "key": "minimum_explicit_feedback",
      "label": "最少显式反馈数",
      "passed": true,
      "status": "passed",
      "actual": "42",
      "threshold": ">= 20",
      "message": "显式反馈数量满足最低要求",
      "blocker_code": ""
    },
    {
      "key": "minimum_class_feedback",
      "label": "最少类别反馈数",
      "passed": true,
      "status": "passed",
      "actual": "专注=28, 分心=14",
      "threshold": "专注 >= 5 且 分心 >= 5",
      "message": "类别反馈数量满足最低要求",
      "blocker_code": ""
    },
    {
      "key": "balanced_accuracy",
      "label": "平衡准确率",
      "passed": false,
      "status": "not_evaluated",
      "actual": "-",
      "threshold": ">= 0.50",
      "message": "尚未运行训练评估，无法确定平衡准确率",
      "blocker_code": "metric_not_evaluated"
    },
    {
      "key": "minority_f1",
      "label": "少数类 F1",
      "passed": false,
      "status": "not_evaluated",
      "actual": "-",
      "threshold": ">= 0.30",
      "message": "尚未运行训练评估，无法确定少数类 F1",
      "blocker_code": "metric_not_evaluated"
    },
    {
      "key": "calibration_better_than_rule",
      "label": "校准优于规则引擎",
      "passed": false,
      "status": "not_implemented",
      "actual": "-",
      "threshold": "训练报告提供证据",
      "message": "校准比较需训练报告提供真实证据，当前硬编码为通过，不可作为绿色通行",
      "blocker_code": "not_implemented"
    },
    {
      "key": "stable_date_folds",
      "label": "日期折叠稳定性",
      "passed": false,
      "status": "not_implemented",
      "actual": "-",
      "threshold": "训练报告提供证据",
      "message": "日期折叠稳定性需训练报告提供真实证据，当前硬编码为通过，不可作为绿色通行",
      "blocker_code": "not_implemented"
    }
  ],
  "blockers": [
    {
      "code": "metric_not_evaluated",
      "message": "尚未运行训练评估，无法确定平衡准确率"
    }
  ],
  "current_training_job": null
}
```

### 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 评估成功（即使数据不足也返回 200，由 `trainable`/`blockers` 表示） |
| 404 | repository 未初始化 |

### 关键计算规则

| 字段 | 来源 | 阈值 |
|------|------|------|
| `trainable` | `prepare_v2_training_data()` 的 `explicit_mask` 求和 | >= 10 个合格窗口 且 >= 2 个唯一标签 |
| `evaluable` | `V2TrainingData.explicit_feedback_count` + `distinct_feedback_days` | >= 10 个显式样本 且 >= 3 个不同日期 |
| `baseline_ready` | `BaselineRepository.get_latest().has_sufficient_data()` | >= 30 个总样本 |

---

## POST /api/v1/analytics/training-jobs

启动 V2 模型训练任务。先执行就绪检查，再异步启动训练。

### 请求

```
POST /api/v1/analytics/training-jobs
Cookie: mindflow_session=<session_token>
Content-Type: application/json
```

请求体为空。

### 响应 `202 Accepted`

```json
{
  "job_id": "train-a1b2c3d4e5f6",
  "status": "pending"
}
```

### 错误响应

`412 Precondition Failed` — 训练数据不足（trainable=False）。额外字段合并至 ProblemDetail 顶层：

```json
{
  "type": "https://mindflow.app/errors/training-not-ready",
  "title": "Training Not Ready",
  "status": 412,
  "detail": "训练数据不足，无法启动训练任务",
  "instance": "/api/v1/analytics/training-jobs",
  "trainable": false,
  "blockers": [
    {"code": "insufficient_eligible_windows", "message": "符合条件的窗口不足（当前 3，需要 10）"}
  ]
}
```

`409 Conflict` — 已有活跃训练任务：

```json
{
  "type": "https://mindflow.app/errors/training-job-active",
  "title": "Training Job Already Active",
  "status": 409,
  "detail": "Training job train-xxx is already active (status=training)",
  "instance": "/api/v1/analytics/training-jobs"
}
```

### 状态码

| 状态码 | 说明 |
|--------|------|
| 202 | 任务已创建，后台执行中 |
| 409 | 已有活跃训练任务（每进程最多一个） |
| 412 | 训练数据不足（trainable=False） |
| 404 | 服务未初始化 |

---

## GET /api/v1/analytics/training-jobs/{job_id}

查询训练任务生命周期状态与报告。

### 请求

```
GET /api/v1/analytics/training-jobs/train-a1b2c3d4e5f6
Cookie: mindflow_session=<session_token>
```

### 响应 `200 OK`

成功状态（shadow 模式，未激活）：

```json
{
  "job_id": "train-a1b2c3d4e5f6",
  "status": "succeeded",
  "source": "db",
  "model_mode": "shadow",
  "started_at": "2026-07-30T10:00:00",
  "completed_at": "2026-07-30T10:01:23",
  "activated": false,
  "version_tag": "20260730_100123",
  "feature_schema_version": 2,
  "quality_gate": {
    "passed": false,
    "mode": "shadow",
    "checks": {
      "minimum_days": true,
      "minimum_explicit_feedback": true,
      "minimum_class_feedback": true,
      "balanced_accuracy": false,
      "minority_f1": false,
      "calibration_better_than_rule": true,
      "stable_date_folds": true
    },
    "explicit_feedback_count": 25,
    "explicit_focus_count": 15,
    "explicit_distract_count": 10,
    "distinct_feedback_days": 3
  },
  "evaluation": {
    "status": "evaluated",
    "candidate": {
      "balanced_accuracy": 0.45,
      "minority_f1": 0.25,
      "brier_score": 0.21
    }
  },
  "error": null
}
```

成功状态（ready 模式，已激活）：

```json
{
  "job_id": "train-fedcba987654",
  "status": "succeeded",
  "source": "db",
  "model_mode": "ready",
  "started_at": "2026-07-30T12:00:00",
  "completed_at": "2026-07-30T12:02:15",
  "activated": true,
  "version_tag": "20260730_120215",
  "feature_schema_version": 2,
  "quality_gate": {
    "passed": true,
    "mode": "ready",
    "checks": {
      "minimum_days": true,
      "minimum_explicit_feedback": true,
      "minimum_class_feedback": true,
      "balanced_accuracy": true,
      "minority_f1": true,
      "calibration_better_than_rule": true,
      "stable_date_folds": true
    },
    "explicit_feedback_count": 42,
    "explicit_focus_count": 28,
    "explicit_distract_count": 14,
    "distinct_feedback_days": 7
  },
  "evaluation": {
    "status": "evaluated",
    "candidate": {
      "balanced_accuracy": 0.72,
      "minority_f1": 0.45,
      "brier_score": 0.12
    },
    "logistic_baseline": {
      "balanced_accuracy": 0.68,
      "minority_f1": 0.40,
      "brier_score": 0.15
    },
    "rule_baseline": {
      "balanced_accuracy": 0.55,
      "minority_f1": 0.30,
      "brier_score": 0.22
    },
    "folds": [
      {"fold": 1, "balanced_accuracy": 0.70, "train_dates": ["2026-07-24", "2026-07-25"], "test_dates": ["2026-07-26"]},
      {"fold": 2, "balanced_accuracy": 0.75, "train_dates": ["2026-07-24", "2026-07-26"], "test_dates": ["2026-07-27"]}
    ]
  },
  "error": null
}
```

待定状态：

```json
{
  "job_id": "train-a1b2c3d4e5f6",
  "status": "preparing_data",
  "source": "db",
  "model_mode": "rule_engine_only",
  "started_at": "2026-07-30T10:00:00",
  "completed_at": null,
  "activated": false,
  "version_tag": null,
  "feature_schema_version": null,
  "quality_gate": null,
  "evaluation": null,
  "error": null
}
```

失败状态（含发布失败）：

```json
{
  "job_id": "train-a1b2c3d4e5f6",
  "status": "failed",
  "error": "PublicationError: Ready-model publication failed: load_latest() failed for ready models"
}
```

### 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 返回任务详情（含终端或进行中状态） |
| 404 | 任务 ID 不存在 |

### 状态机

```
pending ──> preparing_data ──> training ──> succeeded
                                        ──> failed
                  任何阶段 ──> cancelled（仅 terminal 前可取消）
```

### 关键字段说明

| 字段 | 说明 |
|------|------|
| `status` | `pending` / `preparing_data` / `training` / `succeeded` / `failed` / `cancelled` |
| `model_mode` | `rule_engine_only`（训练前/影子模式训练后）/ `shadow` / `ready` |
| `activated` | 训练报告是否激活模型（仅 `ready` 模式且通过质量门禁时为 true） |
| `quality_gate` | 训练质量指标（balanced_accuracy, minority_f1, calibration, stability） |
| `error` | 失败时的错误信息；仅 `failed` 状态非空 |

---

## POST /api/v1/analytics/training-jobs/{job_id}/cancel

取消待定或准备数据中的训练任务。

### 请求

```
POST /api/v1/analytics/training-jobs/train-a1b2c3d4e5f6/cancel
Cookie: mindflow_session=<session_token>
```

### 响应 `200 OK`

```json
{
  "job_id": "train-a1b2c3d4e5f6",
  "status": "cancelled",
  "source": "db",
  "model_mode": "rule_engine_only",
  "started_at": "2026-07-30T10:00:00",
  "completed_at": "2026-07-30T10:00:05",
  "activated": false,
  "version_tag": null,
  "feature_schema_version": null,
  "quality_gate": null,
  "evaluation": null,
  "error": null
}
```

### 错误响应 `409 Conflict`

```json
{
  "type": "https://mindflow.app/errors/training-cancel-rejected",
  "title": "Cancel Rejected",
  "status": 409,
  "detail": "Cannot cancel training job train-xxx: training thread is already running",
  "instance": "/api/v1/analytics/training-jobs/train-xxx/cancel"
}
```

### 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 取消成功（或任务已处于终端状态） |
| 404 | 任务 ID 不存在 |
| 409 | 训练已经开始，无法安全取消 |

### 取消规则

- 仅在 `pending` 或 `preparing_data` 状态时可取消
- 进入 `training` 后取消被拒绝（409），因后台线程可能已调用 `save_all(activate=True)` 写入激活的模型制品
- 已处于终端状态的任务返回当前状态（200，不报错）

---

## GET /api/v1/analytics/model-status

查询 V2 模型加载状态与版本信息。

### 请求

```
GET /api/v1/analytics/model-status
Cookie: mindflow_session=<session_token>
```

### 响应 `200 OK`

模型已加载：

```json
{
  "loaded": true,
  "ready": true,
  "mode": "ready",
  "feature_schema_version": 2,
  "v2_mode": "ready",
  "version": "20260730_100123",
  "available_versions": ["20260730_100123", "20260729_153000"],
  "reasons": ["v2_models_loaded"],
  "message": "Feature schema v2 model loaded and ready for inference"
}
```

仅规则引擎：

```json
{
  "loaded": false,
  "ready": false,
  "mode": "rule_engine_only",
  "feature_schema_version": 2,
  "v2_mode": "rule_engine_only",
  "reasons": ["v2_models_not_loaded"],
  "message": "V2 ML models not available, running with rule engine only"
}
```

---

## GET /api/v1/analytics/baseline

查询个人行为基线（Welford 在线统计）。

### 请求

```
GET /api/v1/analytics/baseline
Cookie: mindflow_session=<session_token>
```

### 响应 `200 OK`

```json
{
  "user_id": 1,
  "created_at": "2026-07-20T08:00:00",
  "updated_at": "2026-07-30T10:00:00",
  "total_days": 10,
  "total_samples": 4520,
  "features": [
    "focus_score",
    "app_switch_freq",
    "active_ratio_30m",
    "entertainment_ratio_30m",
    "session_count_30m"
  ]
}
```

### 响应 `404 Not Found`

```json
{
  "detail": "基线模型（暂无训练数据）"
}
```

---

## 数据流水线与就绪判定

```
activity_events（原始活动事件）
    ↓ telemetry rollup（5s 采集 + 心跳合并）
V2 feature windows（schema_version=2）
    ↓ list_feature_windows(uid, feature_schema_version=2)
特征窗口列表
    ↓ + list_focus_feedback(uid) + list_all(uid) sessions
带时间戳的反馈列表
    ↓ prepare_v2_training_data(windows, feedback_with_times)
V2TrainingData（时间重叠匹配）
    ↓
    ├─ explicit_mask.sum() >= 10 AND unique_classes >= 2 → trainable
    ├─ explicit_feedback_count >= 10 AND distinct_feedback_days >= 3 → evaluable
    └─ gates[]（7 项检查）→ activatable
```

### 常见误解

- **数据存在 ≠ 可训练**：原始事件数多不代表可训练。telemetry 必须完成 rollup 生成 V2 特征窗口，且用户显式反馈的时间戳必须与窗口范围重叠
- **基线 ≠ ML 模型**：`baseline` 端点是 Welford 在线增量统计；ML 训练是批量离线过程。两者共享 `/model-center` 页面但生命周期独立
- **`not_implemented` 不代表通过**：`calibration_better_than_rule` 和 `stable_date_folds` 两个门禁硬编码为 `not_implemented` 状态，readiness 响应中 `passed: false`，不可视为绿色
