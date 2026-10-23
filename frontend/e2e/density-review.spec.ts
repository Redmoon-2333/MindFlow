import { test, expect } from "@playwright/test";
import { setup } from "./fixtures/theme-api";

for (const theme of ["white", "black", "navy", "burgundy"]) for (const width of [1440, 390]) {
  test(`${theme} ${width}: content alignment, dropdown and compact controls`, async ({page}) => {
    await setup(page); await page.setViewportSize({width,height:1000});
    await page.addInitScript(value => localStorage.setItem("mindflow_theme",value),theme);
    await page.route("**/api/v1/focus?**",route => route.fulfill({json:{date:"2026-10-08",session_count:1,sessions:[{id:"d",session_type:"distraction",duration_minutes:30,focus_score:28,start_time:"2026-10-08T09:00:00",end_time:"2026-10-08T09:30:00",dominant_app:"ChatGPT.exe",switch_count:12}]}}));
    await page.route("**/api/v1/intervention/history?**",route => route.fulfill({json:{items:[{id:"i",intervention_type:"task_breakdown",title:"小提示：先完成一步",message:"先整理一个图表，再继续下一步。",created_at:"2026-10-08T09:00:00Z",user_response:"dismissed",response_latency_s:3.438}],count:1}}));
    await page.route("**/api/v1/analytics/attribution",route => route.fulfill({json:{procrastination_type:"task_aversion",cbt_technique:"graded_exposure",evidence:"任务较大，可以从小步骤开始。"}}));
    await page.goto("/"); await expect(page.locator(".d-extra-card").first()).toBeVisible();
    const rows=await page.evaluate(() => [".d-state",".d-activity-chart",".d-statistic",".d-control-inference",".d-extra"].map(s=>{const r=document.querySelector(s)!.getBoundingClientRect();return {x:r.x,width:r.width};}));
    for(const row of rows){expect(Math.abs(row.x-rows[0].x)).toBeLessThan(1);expect(Math.abs(row.width-rows[0].width)).toBeLessThan(1);}
    if(width===1440) { const heights=await page.locator(".d-control, .d-inference").evaluateAll(els=>els.map(el=>el.getBoundingClientRect().height)); expect(Math.abs(heights[0]-heights[1])).toBeLessThan(1); }
    await page.goto("/focus"); await expect(page.locator(".conversation-item")).toHaveCount(1);
    expect(await page.locator(".state-item").first().evaluate(el=>el.getBoundingClientRect().height)).toBeGreaterThanOrEqual(145);
    const selects=page.locator(".conversation-item select");
    expect(await selects.first().evaluate(el=>{const select=getComputedStyle(el), option=getComputedStyle(el.querySelector("option")!);return select.backgroundPosition.includes("14px")&&option.color!==option.backgroundColor&&select.backgroundImage.includes("svg");})).toBe(true);
    await expect.poll(()=>page.locator(".mf-main").evaluate(el=>el.scrollWidth<=el.clientWidth+1)).toBe(true);
    await page.locator(".conversation-item").scrollIntoViewIfNeeded();
    await page.screenshot({path:`screenshots/20261008/density-${theme}-${width}-focus.png`});
    await page.goto("/intervention"); await expect(page.locator(".intervention-history-evaluate")).toBeVisible();
    await expect(page.locator(".workspace-heading")).toHaveCount(0);
    if(width===1440){
      const title=await page.locator(".intervention-history-title").boundingBox(),button=await page.locator(".intervention-history-evaluate").boundingBox();
      expect(Math.abs(title!.y+title!.height/2-button!.y-button!.height/2)).toBeLessThan(6);
      const footer=page.locator(".intervention-response-row");expect(await footer.evaluate(el=>el.getBoundingClientRect().height)).toBeLessThan(50);
    }
    await page.screenshot({path:`screenshots/20261008/density-${theme}-${width}-intervention.png`});
    await page.goto("/settings"); await expect(page.locator(".settings-control-card")).toHaveCount(2);
    if(width===1440){const cards=await page.locator(".settings-control-card").evaluateAll(els=>els.map(el=>el.getBoundingClientRect().top));expect(Math.abs(cards[0]-cards[1])).toBeLessThan(1);}
    if(width===1440) {
      const stats=await page.locator(".telemetry-stats > div").evaluateAll(els=>els.map(el=>({x:el.getBoundingClientRect().x,y:el.getBoundingClientRect().y,width:el.getBoundingClientRect().width})));
      const buttons=await page.locator(".telemetry-actions > button").evaluateAll(els=>els.map(el=>({x:el.getBoundingClientRect().x,y:el.getBoundingClientRect().y,width:el.getBoundingClientRect().width})));
      expect(stats).toHaveLength(5); expect(buttons).toHaveLength(5);
      stats.forEach((stat,i)=>{ expect(Math.abs(stat.y-stats[0].y)).toBeLessThan(1); expect(Math.abs(stat.x-buttons[i].x)).toBeLessThan(1); expect(Math.abs(stat.width-buttons[i].width)).toBeLessThan(1); });
    }
    await page.locator(".settings-control-grid").scrollIntoViewIfNeeded();
    await page.screenshot({path:`screenshots/20261008/density-${theme}-${width}-settings.png`});
    await page.route("**/api/v1/analytics/profile?**",route=>route.fulfill({json:{peak_focus_hours:[{hour:8,avg_score:52}],peak_focus:"8:00 (avg 52)",productivity_apps:["code.exe"],trigger_apps:[],avg_focus_block_min:30,details:{avg_focus_block_min:30,total_events:524,distraction_ratio:0.006}}}));
    await page.goto("/analytics"); await page.getByRole("button",{name:"运行归因分析"}).click();
    await expect(page.getByText("任务厌恶（task_aversion）",{exact:false})).toBeVisible();
    await expect(page.getByText("分级面对任务（graded_exposure）",{exact:true})).toBeVisible();
    await expect(page.getByText("活动事件总数（total_events）",{exact:true})).toBeVisible();
    expect(await page.locator("select.time").evaluate(el=>el.getBoundingClientRect().width)).toBeGreaterThanOrEqual(140);
    await page.locator(".fifth-box").scrollIntoViewIfNeeded();
    await page.screenshot({path:`screenshots/20261008/density-${theme}-${width}-attribution.png`});
  });
}

 test("AI configuration explains missing route and can retry after backend update", async ({page}) => {
  await setup(page); let available = false;
  await page.route("**/api/v1/settings/ai", route => route.fulfill(available ? {json:{saved:false,active_configured:false,environment_override:false,restart_required:false,provider:"DeepSeek",model:"deepseek-flash"}} : {status:404,json:{detail:"Not Found"}}));
  await page.goto("/settings"); await page.getByRole("button",{name:"外观与 AI",exact:true}).click();
  await expect(page.getByRole("alert")).toContainText("当前后端没有 AI 配置接口");
  await expect(page.getByRole("button",{name:"保存 AI Key"})).toBeDisabled();
  available = true; await page.getByRole("button",{name:"重新加载配置"}).click();
  await expect(page.getByText("本机未保存 Key",{exact:true})).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
});
test("ChatECNU selection saves its own key and model without browser persistence", async ({page}) => {
  await setup(page);
  await page.route("**/api/v1/settings/ai", route => {
    const body = route.request().method() === "PUT" ? route.request().postDataJSON() : null;
    return route.fulfill({json:{saved:!!body,active_configured:false,environment_override:false,restart_required:!!body,provider:body?.service === "ecnu" ? "ChatECNU" : "DeepSeek",model:body?.ecnu_model ?? "deepseek-flash",service:body?.service ?? "deepseek",active_provider:"DeepSeek"}});
  });
  await page.goto("/settings"); await page.getByRole("button",{name:"外观与 AI",exact:true}).click();
  await page.getByLabel("DeepSeek API Key",{exact:true}).fill("sk-do-not-reuse-test");
  await page.getByRole("combobox",{name:"AI 服务",exact:true}).selectOption("ecnu");
  await expect(page.getByLabel("ChatECNU API Key",{exact:true})).toHaveValue("");
  await page.getByRole("combobox",{name:"ChatECNU 模型",exact:true}).selectOption("ecnu-plus");
  const secret = "sk-campus-browser-test-123";
  await page.getByLabel("ChatECNU API Key",{exact:true}).fill(secret);
  const request = page.waitForRequest(r=>r.method()==="PUT"&&r.url().endsWith("/settings/ai"));
  await page.getByRole("button",{name:"保存 AI Key"}).click();
  expect((await request).postDataJSON()).toEqual({api_key:secret,service:"ecnu",ecnu_model:"ecnu-plus"});
  await expect(page.getByLabel("ChatECNU API Key",{exact:true})).toHaveValue("");
  await expect(page.getByText("保存的服务：ChatECNU · ecnu-plus",{exact:true})).toBeVisible();
  await page.screenshot({path:"screenshots/20261009-chatecnu-settings-desktop.png"});
  await page.setViewportSize({width:390,height:1000});
  await expect.poll(()=>page.locator(".mf-content").evaluate(el=>Math.abs(el.getBoundingClientRect().x))).toBeLessThan(1);
  await expect.poll(()=>page.locator(".mf-main").evaluate(el=>el.scrollWidth<=el.clientWidth+1)).toBe(true);
  await page.screenshot({path:"screenshots/20261009-chatecnu-settings-mobile.png"});
  await expect(page.getByText("等待重启生效",{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>JSON.stringify(localStorage)+JSON.stringify(sessionStorage))).not.toContain(secret);
});
