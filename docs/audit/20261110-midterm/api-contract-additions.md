# 中期升级 API 契约增量（contract-first，前后端共同遵守）

所有新增字段**可选/可空**，旧客户端不受影响。后端以 Pydantic 模型落地后运行 `npm run generate:api` 更新 `frontend/src/generated/api-schema.ts`；前端在生成前可按本文件手写临时类型，生成后改用生成类型。

## 1. `GET /api/v1/telemetry/focus-prediction` → `FocusPredictionResponse` 新增

```ts
explanation?: {
  method: "tree_shap" | "linear" | "unavailable";
  base_value: number | null;          // 模型期望输出（见 output_space）
  output_space: "probability" | "logit";
  contributions: Array<{
    feature: string;                   // V2 特征名
    label: string;                     // 中文名，如 "应用切换频率"
    value: number;                     // 当前窗口均值（原始尺度）
    baseline_value: number | null;     // 个人基线期望值（同时段），无则 null
    contribution: number;              // 带符号：>0 推向“专注”，<0 推向“分心”
  }>;                                  // 按 |contribution| 降序，最多 8 条
  note: string;                        // 人类可读一句话总结
} | null;
```

## 2. 个人基线偏离 `DeviationSummary`（出现在归因响应与面板响应，字段名 `deviation`）

```ts
deviation?: {
  overall_score: number;               // 加权稳健 z 绝对值
  severity: "normal" | "mild" | "moderate" | "severe";
  method: "eb_shrinkage_mad";          // 经验贝叶斯收缩 + MAD 稳健 z
  bucket: { hour_of_day: number; day_of_week: number; n_samples: number; shrinkage: number }; // shrinkage = k/(n+k)
  features: Array<{ feature: string; label: string; value: number; expected: number; robust_z: number; weight: number }>; // 按 |robust_z|*weight 降序，最多 6 条
} | null;
```

## 3. TMT 动机分解 `TmtBreakdown`（`POST /analytics/attribution` 与 `GET /panel`、`POST /panel/today` 响应新增 `tmt`）

```ts
tmt?: {
  motivation: number;                  // E·V/(1+I·D)，归一化到 [0,1]
  motivation_level: "low" | "medium" | "high";
  dominant_barrier: "expectancy" | "value" | "impulsiveness" | "delay";
  components: {
    expectancy:    TmtComponent;       // E 期望：成功预期
    value:         TmtComponent;       // V 价值：任务价值/投入
    impulsiveness: TmtComponent;       // I 冲动性：易分心程度
    delay:         TmtComponent;       // D 延迟：距截止时间
  };
  note: string;                        // “行为代理变量，不是心理测量” 等边界说明
} | null;
type TmtComponent = { score: number /*0..1*/; raw: number | null; label: string; source: string; evidence_ids: string[] };
```

## 4. 干预理由 `InterventionRationale`
出现在 `POST /intervention/trigger` 响应、`GET /intervention/history` 每项、WS `intervention` 事件 payload 中，字段名 `rationale`：

```ts
rationale?: {
  procrastination_type: string | null;
  intervention_type: string;
  selection: "thompson_sampling" | "rule";
  context_bucket: string;              // 如 "afternoon|distracted"
  candidates: Array<{ intervention_type: string; label: string; alpha: number; beta: number; posterior_mean: number; sample: number; chosen: boolean }>;
  evidence: string[];                  // 人类可读依据，最多 4 条
  note: string;
} | null;
```

## 5. 主动学习反馈
- WS 新事件：`{ "type": "feedback_request", "data": { request_id: string, window_start: string /*ISO-8601*/, window_end: string, probability: number | null, reasons: string[] /*中文*/, prompt: string } }`
- 新端点：`POST /api/v1/telemetry/feedback-requests/{request_id}/answer`
  body `{ label: "focus" | "distracted" | "skip" }` → `{ request_id, labeled_windows: number, status: "recorded" | "skipped" | "expired" }`
  语义：把请求覆盖的 5 分钟特征窗口 `behavior_feature_windows.label` 写为对应标签（训练权重 0.8 的窗口标签）。
- 新端点：`GET /api/v1/telemetry/feedback-requests/pending` → `{ items: FeedbackRequest[] }`（WS 断线时前端轮询兜底，最多 1 条）。
