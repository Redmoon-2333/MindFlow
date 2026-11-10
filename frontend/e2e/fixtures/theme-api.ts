import { type Page, type Route } from "@playwright/test";

const task = { id: "task-1", title: "完成研究报告", description: "整理实验结果与图表", priority: 4, status: "pending", deadline_utc: "2026-10-02T04:00:00Z", estimated_minutes: 60, created_at: "2026-10-01T04:00:00Z", updated_at: "2026-10-01T04:00:00Z" };
const panel = { types: ["task_aversion"], confidence: {task_aversion: .8}, technique: "把任务拆成一个十分钟的小步骤", rationale: "近期任务较大，先从容易开始的步骤入手。".repeat(5), dissent: ["目前的证据还不能排除正常休息。"], transcript: [{role:"analyst", round:1, content:"根据今天的记录，我们需要结合任务难度分析。".repeat(8)}], escalated:false, call_count:6, degraded:false, cached:false, insufficient_data:false, source:"panel", meta:{source:"panel", cached:false, degradation_path:[]} };

export async function setup(page: Page) {
  await page.addInitScript(() => localStorage.setItem("mindflow_authenticated", "1"));
  await page.routeWebSocket("**/api/v1/ws", () => {});
  await page.route("**/api/v1/**", async route => {
    const url = new URL(route.request().url()); const path = url.pathname.replace("/api/v1", "");
    let json: unknown = {};
    if (path === "/settings/ai") json = {saved:false,active_configured:false,environment_override:false,restart_required:false,provider:"DeepSeek",model:"deepseek-flash"};
    else if (path === "/activities/current" || path === "/activities") {
      const activity = {id:"activity-1",timestamp:task.created_at,duration_s:120,event_type:"app_switch",data:{app_name:"msedge.exe",process_name:"msedge.exe",window_title:"研究报告与实验图表整理 · 很长的窗口标题需要完整换行显示 ".repeat(8),is_idle:false}};
      json = path === "/activities/current" ? activity : {items:[activity],page:1,page_size:10,total:1,has_more:false,next_cursor:null};
    }
    else if (path === "/focus") json = {date:"2026-10-08",sessions:[],session_count:0};
    else if (path === "/focus/trend") json = {days:7,start_date:"2026-10-02",end_date:"2026-10-08",daily:[],total_sessions:0};
    else if (path === "/analytics/patterns") json = {high_switch_periods:[{period:"15:00 - 16:00",switch_count:48}],trigger_apps:[{app:"very-long-browser-application.exe",count:2}]};
    else if (path === "/analytics/profile") json = {peak_hours:[],productive_apps:[],trigger_apps:[],details:{}};
    else if (path === "/analytics/training-readiness") json = {raw_events:{total_events:311,coverage_days:8},v2_windows:{total:56,schema_version:4,date_range_days:8,eligible_count:0,matched_focus_count:0,matched_distract_count:0},feedback_labels:{total:0,focus:0,distract:0,mixed:0},trainable:false,trainable_window_count:0,trainable_class_count:0,evaluable:false,evaluable_explicit_count:0,evaluable_date_count:0,baseline_ready:true,current_mode:"rule_engine_only",gates:[],blockers:[],current_training_job:null};
    else if (path === "/tasks") json = {items:[task], count:1};
    else if (path.startsWith("/tasks/")) json = {...task,...route.request().postDataJSON()};
    else if (path === "/interventions/blocklist") json = {items:[{domain:"bilibili.com", enabled:true, reason:"专心写报告", created_at:task.created_at}], count:1};
    else if (path === "/panel" || path === "/panel/today") json = panel;
    else if (path === "/chat/sessions") json = [];
    else if (path === "/intervention/history") json = {items:[{id:"intervention-1", intervention_type:"task_breakdown", title:"从一个小步骤开始", message:"先整理一个图表，再继续下一步。", created_at:task.created_at, triggered_at:task.created_at, user_response:null}], count:1};
    else if (path === "/health") json = {status:"ok", version:"2.0", database:{status:"ok"}, migration:{applied:true}};
    else if (path === "/health/live") json = {status:"alive", version:"2.0", timestamp:"2026-10-08T09:00:00Z"};
    else if (path === "/health/ready") json = {status:"ready", version:"2.0", timestamp:"2026-10-08T09:00:00Z", database:{status:"ok", connected:true, integrity_ok:true}, migration:{applied:true}, checkpoint_store:"available", run_store:"available"};
    else if (path === "/collector") json = {status:"stopped", running:false};
    else if (path === "/autonomy") json = {enabled:true, paused:false, paused_until:null};
    else if (path === "/app-classifications") json = [{id:"rule-1", process_name:"code.exe", window_title_pattern:"", category:"code", priority:10}];
    else if (path === "/app-classifications/unknown-apps") json = [{process_name:"editor.exe", count:12, last_seen:task.created_at}];
    else if (path === "/telemetry/status") json = {preferences:{input_telemetry_enabled:true, browser_tracking_enabled:false, interaction_retention_days:7, activity_retention_days:30}, input_watcher_status:"running", database_size_bytes:1024, interaction_bucket_count:12, browser_segment_count:0, browser_paired:false, last_interaction_at:task.created_at, last_browser_at:null};
    else if (path === "/telemetry/focus-prediction") json = {focus_probability:null, status:"no_model", mode:"rule_engine_only", reason:"未加载模型"};
    else if (path === "/analytics/usage") json = {mode:"llm", llm_calls_30d:5, llm_cost_usd_30d:.1234, panel_count_30d:1};
    else if (path === "/analytics/model-status") json = {loaded:false, ready:false, mode:"rule_engine_only", v2_mode:"rule_engine_only", reasons:[]};
    else if (path === "/analytics/baseline" || path === "/analytics/training-readiness") return route.fulfill({status:404, json:{detail:"暂无数据"}});
    else if (path === "/ai/graph") json = {available:true, nodes:["start","panel"], edges:[{from:"start",to:"panel"}]};
    else if (path === "/ai/runs") { const offset=Number(url.searchParams.get("offset") ?? 0); json = {items:Array.from({length:Math.min(20,45-offset)},(_,i) => ({run_id:`run-${offset+i}`,status:"completed",source:"panel",started_at:task.created_at,completed_at:task.created_at,duration_ms:1000})),count:45,has_more:offset+Math.min(20,45-offset)<45,next_offset:offset+Math.min(20,45-offset)<45?offset+Math.min(20,45-offset):null}; }
    else if (path.startsWith("/ai/runs/")) json = {run_id:path.split("/").at(-1), status:"completed",source:"panel",started_at:task.created_at,completed_at:task.created_at,duration_ms:1000,node_events:[],error:null};
    await route.fulfill({status:200,json});
  });
}

// ── Explainability fixtures ───────────────────────────────────────────────
// Shapes mirror docs/audit/20261110-midterm/api-contract-additions.md §1–§5.
// Kept separate from `setup()` so the pre-existing specs keep the "feature
// absent" behaviour they were written against; `setupExplainable()` layers
// these routes on top (Playwright matches the most recent handler first).

export const explanation = {
  method: "tree_shap",
  base_value: 0.41,
  output_space: "probability",
  note: "贡献按概率空间计算，正值把预测推向专注。",
  contributions: [
    { feature: "app_switch_rate", label: "应用切换频率", value: 18.4, baseline_value: 9.1, contribution: 0.21 },
    { feature: "productive_app_ratio", label: "生产类应用占比", value: 0.63, baseline_value: 0.44, contribution: 0.17 },
    { feature: "idle_ratio", label: "空闲占比", value: 0.31, baseline_value: 0.12, contribution: -0.14 },
    { feature: "entertainment_ratio", label: "娱乐类应用占比", value: 0.22, baseline_value: 0.08, contribution: -0.09 },
    { feature: "interaction_bursts", label: "输入突发次数", value: 42, baseline_value: 38, contribution: 0.03 },
  ],
};

export const tmt = {
  motivation: 0.34,
  motivation_level: "low",
  dominant_barrier: "delay",
  components: {
    expectancy: { score: 0.52, raw: 0.52, label: "同类任务历史专注成功率", source: "focus_history", evidence_ids: ["ev-1"] },
    value: { score: 0.61, raw: 0.61, label: "生产类应用占比 × 任务优先级", source: "task_context", evidence_ids: ["ev-2"] },
    impulsiveness: { score: 0.44, raw: 1.9, label: "应用切换频率（MAD-z）", source: "telemetry", evidence_ids: ["ev-3"] },
    delay: { score: 0.86, raw: 3.1, label: "距离截止时间的剩余天数（MAD-z）", source: "tasks", evidence_ids: ["ev-4"] },
  },
  note: "由行为代理变量估算，不是心理测量；仅用于组织解释与建议。",
};

export const deviation = {
  overall_score: 2.64,
  severity: "moderate",
  method: "eb_shrinkage_mad",
  bucket: { hour_of_day: 15, day_of_week: 3, n_samples: 4, shrinkage: 0.44 },
  features: [
    { feature: "app_switch_rate", label: "应用切换频率", value: 18.4, expected: 9.1, robust_z: 2.71, weight: 0.45 },
    { feature: "idle_ratio", label: "空闲占比", value: 0.31, expected: 0.12, robust_z: 1.94, weight: 0.3 },
    { feature: "entertainment_ratio", label: "娱乐类应用占比", value: 0.22, expected: 0.08, robust_z: 1.12, weight: 0.25 },
  ],
};

export const rationale = {
  procrastination_type: "task_aversion",
  intervention_type: "task_breakdown",
  selection: "thompson_sampling",
  context_bucket: "afternoon|distracted",
  candidates: [
    { intervention_type: "task_breakdown", label: "任务拆解", alpha: 7, beta: 3, posterior_mean: 0.7, sample: 9, chosen: true },
    // `intervention_type` values must stay inside the backend enum
    // (domain/intervention.py InterventionType); labels match its Chinese map.
    { intervention_type: "environment_optimization", label: "环境优化", alpha: 4, beta: 5, posterior_mean: 0.44, sample: 8, chosen: false },
    { intervention_type: "smart_prioritization", label: "优先级建议", alpha: 3, beta: 6, posterior_mean: 0.33, sample: 8, chosen: false },
  ],
  // §8.26.2: `selection_evidence()` now builds these four lines — Chinese type
  // label, 情境 (not 上下文), and a plain-language probability. No α/β, no
  // "后验". Kept in the real order and wording so the fixture cannot drift.
  evidence: [
    "拖延类型：任务厌恶",
    "情境：下午 · 分心",
    "选择任务拆解：根据你过去的反馈，估计有用的概率 70%",
    "备选环境优化：估计有用的概率 44%",
  ],
  note: "行为代理变量，不是心理测量。",
};

export const feedbackRequest = {
  request_id: "fb-1",
  window_start: "2026-10-01T06:05:00Z",
  window_end: "2026-10-01T06:10:00Z",
  probability: 0.52,
  reasons: ["模型不确定（概率贴近决策边界）", "规则与模型判断冲突"],
  prompt: "刚才你在专注吗？",
};

/**
 * `setup()` plus every explainability field the mid-term build adds.
 * The extra routes are registered last, so they win over the defaults.
 */
export async function setupExplainable(page: Page) {
  await setup(page);
  // Frontend calls differ in whether they append a query string, and a
  // glob with "?" only matches the querystring form — so register both.
  const both = async (path: string, handler: (route: Route) => Promise<unknown>) => {
    for (const pattern of [`**/api/v1/${path}`, `**/api/v1/${path}?**`]) await page.route(pattern, handler);
  };
  await both("telemetry/focus-prediction", route =>
    // `reason` is a required string in the schema (only `focus_probability`
    // and `explanation` are nullable).
    route.fulfill({ json: { focus_probability: 0.62, status: "ready", mode: "personal_model", reason: "", explanation } }));
  const withTmt = (route: Route) => route.fulfill({ json: { ...panel, tmt } });
  await both("panel", withTmt);
  await both("panel/today", withTmt);
  await both("analytics/attribution", route =>
    route.fulfill({ json: {
      assessment: {
        procrastination_types: ["task_aversion", "decisional"],
        type_confidence: { task_aversion: 0.72, decisional: 0.31, impulsivity: 0.18, perfectionism: 0.07, emotional_regulation: 0.04 },
        cognitive_distortions: ["灾难化预期", "全或无思维"],
        cbt_technique: "graded_exposure",
        response_text: "任务难度与截止时间的组合让启动成本变高。",
        next_action: "先把「整理图表」拆成一个十分钟的小步骤，只做第一步。",
      },
      source: "panel", cached: false, meta: { degraded: false }, deviation, tmt,
    } }));
  await both("intervention/history", route =>
    route.fulfill({ json: { items: [
      { id: "intervention-2", intervention_type: "task_breakdown", title: "从一个小步骤开始", message: "先整理一个图表，再继续下一步。", created_at: task.created_at, triggered_at: task.created_at, user_response: "accepted", rationale },
      { id: "intervention-1", intervention_type: "environment_optimization", title: "暂时屏蔽娱乐站点", message: "接下来 25 分钟屏蔽视频站点。", created_at: task.created_at, triggered_at: task.created_at, user_response: null, rationale: { ...rationale, intervention_type: "environment_optimization", selection: "rule", candidates: [] } },
    ], count: 2 } }));
}

/** Contract §5 delivery: the pending prompt plus a scripted answer outcome.
 *
 *  The wire shapes mirror `FeedbackRequest.to_payload()` and
 *  `FeedbackAnswerResponse`. `answer` models the three real outcomes:
 *    - `{ status: "recorded", labeled_windows: n }` — the label was written;
 *    - `{ status: "expired" }` — answered after the TTL;
 *    - `{ httpStatus: 404 }` — unknown / already answered id, which the client
 *      must report as expired rather than as a write.
 */
export async function setupFeedbackRequest(
  page: Page,
  answer: { status?: "recorded" | "skipped" | "expired"; labeled_windows?: number; httpStatus?: number } = {},
) {
  await page.route("**/api/v1/telemetry/feedback-requests/pending", route =>
    route.fulfill({ json: { items: [feedbackRequest] } }));
  await page.route(`**/api/v1/telemetry/feedback-requests/${feedbackRequest.request_id}/answer`, route =>
    route.fulfill(
      answer.httpStatus
        ? { status: answer.httpStatus, json: { detail: "反馈请求不存在、已作答或已过期" } }
        : {
            json: {
              request_id: feedbackRequest.request_id,
              labeled_windows: answer.labeled_windows ?? (answer.status === "recorded" ? 3 : 0),
              status: answer.status ?? "recorded",
            },
          },
    ));
}

/** Absent every optional field — the "backend has not shipped §1–§5 yet" case. */
export async function setupExplainabilityAbsent(page: Page) {
  await setup(page);
  await page.route("**/api/v1/analytics/attribution", route =>
    route.fulfill({ json: {
      assessment: { procrastination_types: ["task_aversion"], type_confidence: {}, cbt_technique: "graded_exposure", response_text: "证据不足，暂不下结论。" },
      source: "rule_engine", cached: false, meta: { degraded: false },
    } }));
}

/**
 * Review-only realistic data (Phase 1.1 F7).
 *
 * Used exclusively by `midterm-review-shots.spec.ts` so the screenshots a human
 * reviews look like a real Tuesday instead of the zero-value stress fixture.
 * Everything is computed from `Date.now()` in local time, so the shots stay
 * correct whenever they are regenerated.
 *
 * Deliberately NOT merged into `setup()`: the layout and density specs assert
 * geometry against the sparse/zero fixture, and giving them seven days of data
 * would change what those tests measure.
 */
export async function setupReviewShotsRealistic(page: Page) {
  await setup(page);

  const now = Date.now();
  const twoMinAgo = new Date(now - 2 * 60_000).toISOString();
  const localDay = (offsetDays: number) => {
    const d = new Date(now);
    d.setDate(d.getDate() + offsetDays);
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${mm}-${dd}`;
  };
  const todayAt = (hour: number, minute: number) => {
    const d = new Date(now);
    d.setHours(hour, minute, 0, 0);
    return d.toISOString();
  };

  const both = async (path: string, handler: (route: Route) => Promise<unknown>) => {
    for (const pattern of [`**/api/v1/${path}`, `**/api/v1/${path}?**`]) await page.route(pattern, handler);
  };

  await both("activities/current", route =>
    route.fulfill({ json: {
      id: "activity-review-1",
      timestamp: twoMinAgo,
      duration_s: 120,
      event_type: "app_switch",
      data: {
        app_name: "POWERPNT.EXE",
        process_name: "POWERPNT.EXE",
        window_title: "MindFlow 中期答辩.pptx - PowerPoint",
        is_idle: false,
      },
    } }));

  // Seven days ending today, oldest → newest (F7's fixed series).
  const focusMin = [142, 168, 95, 210, 176, 188, 183];
  const distractionMin = [38, 30, 52, 22, 35, 29, 31];
  const sessionCount = [4, 5, 3, 6, 5, 5, 4];
  const avgScore = [76.2, 79.5, 68.4, 84.1, 80.3, 82.7, 83.0];
  await both("focus/trend", route =>
    route.fulfill({ json: {
      days: 7,
      start_date: localDay(-6),
      end_date: localDay(0),
      daily: focusMin.map((minutes, index) => ({
        date: localDay(index - 6),
        focus_min: minutes,
        distraction_min: distractionMin[index],
        session_count: sessionCount[index],
        avg_score: avgScore[index],
      })),
      total_sessions: sessionCount.reduce((sum, n) => sum + n, 0),
    } }));

  await both("telemetry/focus-prediction", route =>
    route.fulfill({ json: {
      focus_probability: 0.75,
      status: "ready",
      mode: "personal_model",
      reason: "",
      explanation,
    } }));

  await both("analytics/model-status", route =>
    route.fulfill({ json: {
      loaded: true,
      ready: true,
      mode: "personal_model",
      v2_mode: "personal_model",
      reasons: [],
      demo_only: true,
    } }));

  await both("collector", route => route.fulfill({ json: { status: "running", running: true } }));
  await both("health", route =>
    route.fulfill({ json: { status: "ok", version: "2.0", database: { status: "ok" }, migration: { applied: true } } }));

  await both("intervention/history", route =>
    route.fulfill({ json: { items: [
      { id: "intervention-review-2", intervention_type: "environment_optimization", title: "暂时屏蔽娱乐站点", message: "接下来 25 分钟屏蔽视频站点，先把答辩稿的结构定下来。", created_at: todayAt(14, 5), triggered_at: todayAt(14, 5), user_response: null },
      { id: "intervention-review-1", intervention_type: "task_breakdown", title: "从一个小步骤开始", message: "先整理一个图表，再继续下一步。", created_at: todayAt(10, 20), triggered_at: todayAt(10, 20), user_response: "accepted" },
    ], count: 2 } }));

  await both("autonomy", route => route.fulfill({ json: { enabled: true, paused: false, paused_until: null } }));
}
