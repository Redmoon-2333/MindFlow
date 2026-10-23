import { type Page } from "@playwright/test";

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
    else if (path === "/health/live") json = {status:"ok"};
    else if (path === "/health/ready") json = {status:"ok", checks:{database:true}};
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
    else if (path === "/ai/runs") { const offset=Number(url.searchParams.get("offset") ?? 0); json = {items:Array.from({length:Math.min(20,45-offset)},(_,i) => ({run_id:`run-${offset+i}`,status:"completed",source:"panel",started_at:task.created_at,completed_at:task.created_at,duration_ms:1000})),total:45,limit:20,offset}; }
    else if (path.startsWith("/ai/runs/")) json = {run_id:path.split("/").at(-1), status:"completed",source:"panel",started_at:task.created_at,completed_at:task.created_at,duration_ms:1000,node_events:[],error:null};
    await route.fulfill({status:200,json});
  });
}

