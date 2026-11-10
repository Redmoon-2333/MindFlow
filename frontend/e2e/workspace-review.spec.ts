import { test, expect, type Page } from "@playwright/test";

const task = { id: "task-1", title: "完成研究报告", description: "整理实验结果与图表", priority: 4, status: "pending", deadline_utc: "2026-10-02T04:00:00Z", estimated_minutes: 60, created_at: "2026-10-01T04:00:00Z", updated_at: "2026-10-01T04:00:00Z" };
const panel = { types: ["task_aversion"], confidence: {task_aversion: .8}, technique: "把任务拆成一个十分钟的小步骤", rationale: "近期任务较大，先从容易开始的步骤入手。".repeat(5), dissent: ["目前的证据还不能排除正常休息。"], transcript: [{role:"analyst", round:1, content:"根据今天的记录，我们需要结合任务难度分析。".repeat(8)}], escalated:false, call_count:6, degraded:false, cached:false, insufficient_data:false, source:"panel", meta:{source:"panel", cached:false, degradation_path:[]} };

async function setup(page: Page) {
  await page.addInitScript(() => localStorage.setItem("mindflow_authenticated", "1"));
  await page.routeWebSocket("**/api/v1/ws", () => {});
  await page.route("**/api/v1/**", async route => {
    const url = new URL(route.request().url()); const path = url.pathname.replace("/api/v1", "");
    let json: unknown = {};
    if (path === "/tasks") json = {items:[task], count:1};
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

for (const width of [1440, 1024, 390, 320]) {
  for (const [path, title] of [["/intervention","干预中心"],["/panel","专家面板"],["/chat","AI 对话"],["/settings","系统设置"],["/execution","干预执行"],["/diagnostics","AI 诊断"],["/model-center","模型中心"]]) {
    test(`${width}px ${title}: readable layout`, async ({page}) => {
      await page.setViewportSize({width,height:1000}); await setup(page);
      const errors: string[]=[]; page.on("pageerror", e => errors.push(e.message));
      await page.goto(path); await expect(page.locator(".mf-header-name")).toBeVisible();
      if(path === "/panel") await page.getByRole("button",{name:"查看上次结果"}).click();
      if(path === "/settings") {
        await expect(page.getByText("数据采集",{exact:true})).toBeVisible();
        await expect(page.getByText("正在加载设置，部分操作暂不可用")).toHaveCount(0);
      }
      if(path === "/diagnostics") await expect(page.getByText("共 45 条",{exact:false})).toBeVisible();
      await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
      expect(errors).toEqual([]);
      if(width===1440 || width===390) await page.screenshot({path:`screenshots/ui-review/${title}-${width}.png`,fullPage:true});
    });
  }
}

test("task editing submits description and explicit clears",async ({page}) => {
  await setup(page); await page.goto("/execution");
  await page.getByRole("button",{name:"编辑",exact:true}).click();
  await expect(page.getByLabel("任务描述",{exact:true})).toHaveValue(task.description);
  await page.getByLabel("任务标题",{exact:true}).fill("完成报告初稿");
  await page.getByLabel("任务截止时间",{exact:true}).fill("");
  await page.getByLabel("预计分钟",{exact:true}).fill("");
  const req=page.waitForRequest(r => r.method()==="PATCH" && r.url().endsWith("/tasks/task-1"));
  await page.getByRole("button",{name:"保存任务",exact:true}).click();
  expect((await req).postDataJSON()).toMatchObject({title:"完成报告初稿",description:task.description,deadline_utc:null,estimated_minutes:null});
  await expect(page.getByRole("button",{name:"添加任务",exact:true})).toBeVisible();
});

test("settings groups expose real usage and classification",async ({page}) => {
  await setup(page); await page.goto("/settings");
  await page.getByRole("button",{name:"系统状态",exact:true}).click();
  // The readout is a label—value row now (spec §8.6.5), not one colon string.
  await expect(page.locator(".kv-row").filter({hasText:"估算费用"})).toContainText("0.1234 USD");
  await page.getByRole("button",{name:"应用分类",exact:true}).click();
  await page.getByRole("button",{name:"获取未知应用",exact:true}).click();
  await page.getByRole("button",{name:/editor.exe/}).click();
  await expect(page.getByLabel("进程名",{exact:true})).toHaveValue("editor.exe");
  await page.getByRole("button",{name:"数据导出",exact:true}).click();
  await expect(page.getByRole("button",{name:"导出",exact:true})).toBeVisible();
  // §8.16.1: end users never hand-edit JSON, so the editor must be gone.
  await expect(page.getByText("高级偏好设置")).toHaveCount(0);
  await expect(page.getByLabel("偏好设置 JSON")).toHaveCount(0);
});

test("diagnostics next page uses backend offset",async ({page}) => {
  await setup(page); await page.goto("/diagnostics");
  await expect(page.getByText("共 45 条",{exact:false})).toBeVisible();
  const req=page.waitForRequest(r => r.url().includes("/ai/runs?") && new URL(r.url()).searchParams.get("offset")==="20");
  await page.getByRole("button",{name:"下一页",exact:true}).click(); await req;
  await expect(page.getByText("第 2 页",{exact:false})).toBeVisible();
  await page.getByRole("button",{name:"run-20…",exact:true}).click();
  await expect(page.getByText("暂无运行记录",{exact:true})).toHaveCount(0);
  await expect(page.getByText("无节点事件记录")).toBeVisible();
});

test("panel preserves insufficient evidence and dissent", async ({page}) => {
  await setup(page);
  await page.route("**/api/v1/panel", route => route.fulfill({json:{...panel, types:[], technique:"", rationale:"", transcript:[], insufficient_data:true, evidence_gaps:["今天的采集时间不足"]}}));
  await page.goto("/panel");
  await page.getByRole("button",{name:"查看上次结果"}).click();
  await expect(page.getByText("今天的数据还不足以形成可靠结论")).toBeVisible();
  await expect(page.getByText(/今天的采集时间不足/)).toBeVisible();
  await expect(page.getByText(/今天的分析尚未显示/)).toHaveCount(0);
  await page.unroute("**/api/v1/panel");
  await page.getByRole("button",{name:"查看上次结果"}).click();
  await page.getByText("不同意见 · 1 条",{exact:true}).click();
  await expect(page.getByText(panel.dissent[0],{exact:true})).toBeVisible();
});


/** §8.17.4: the jargon "?" opens on focus, closes on Escape, and never leaves
 *  the viewport — including at 375px, where a popover anchored to a trigger
 *  near the right edge would otherwise be clipped. */
for (const width of [1440, 375]) {
  test(`${width}px help tip opens on focus, closes on Escape, stays in the viewport`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await setup(page);
    await page.goto("/");
    const trigger = page.locator(".help-tip-trigger").first();
    await expect(trigger).toBeVisible({ timeout: 10000 });
    await trigger.focus();

    const tip = page.getByRole("tooltip");
    await expect(tip).toBeVisible();
    expect((await tip.textContent())?.trim().length ?? 0).toBeGreaterThan(0);

    const box = await tip.boundingBox();
    expect(box, "the tooltip must be laid out before it can be measured").not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.y).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width);
    expect(box!.y + box!.height).toBeLessThanOrEqual(900);

    await page.keyboard.press("Escape");
    await expect(tip).toHaveCount(0);
  });
}

/** §8.22.1: the page gutter belongs to `.mf-main` alone. Four page wrappers
 *  (`.f-page`, `.activities`, `.analytics`, `.reports`) once re-added a padding
 *  of their own, which pushed their content 24px further in than the dashboard
 *  for no reason the reader could name. The acceptance criterion is positional:
 *  on every page shape, the lede starts exactly at the `.mf-main` content edge. */
test("1440px: every page shape starts at the .mf-main content edge", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await setup(page);
  // `/focus` needs a well-formed envelope; the shared fixture answers with `{}`.
  await page.route("**/api/v1/focus?**", route =>
    route.fulfill({ json: { date: "2026-10-08", session_count: 0, sessions: [] } }));
  for (const path of ["/focus", "/activities", "/analytics", "/reports", "/settings"]) {
    await page.goto(path);
    await expect(page.locator(".page-lede")).toBeVisible({ timeout: 10000 });
    const edges = await page.evaluate(() => {
      const main = document.querySelector(".mf-main")!;
      const mainRect = main.getBoundingClientRect();
      const lede = document.querySelector(".page-lede")!;
      return {
        content: mainRect.left + parseFloat(getComputedStyle(main).paddingLeft),
        lede: lede.getBoundingClientRect().left,
      };
    });
    expect(Math.abs(edges.lede - edges.content), `${path} must not add a gutter of its own`).toBeLessThan(1);
  }
});

/** §8.15: no page may scroll horizontally at 375px — the narrowest phone the
 *  shell supports. Checked on every routed page in one sweep so a single-page
 *  regression cannot hide behind the pages that happen to be tested already. */
test("375px: no page scrolls horizontally", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  await setup(page);
  for (const path of ["/", "/focus", "/activities", "/analytics", "/reports", "/intervention", "/panel", "/chat", "/settings", "/model-center", "/execution", "/diagnostics"]) {
    await page.goto(path);
    await expect(page.locator(".mf-header-name")).toBeVisible();
    await expect.poll(
      () => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
      { message: `${path} must not overflow at 375px` },
    ).toBe(true);
    await expect.poll(
      () => page.locator(".mf-main").evaluate(el => el.scrollWidth <= el.clientWidth + 1),
      { message: `${path} .mf-main must not scroll horizontally` },
    ).toBe(true);
  }
});
