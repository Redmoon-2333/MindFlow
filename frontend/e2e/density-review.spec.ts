import { test, expect } from "@playwright/test";
import { setup } from "./fixtures/theme-api";

for (const theme of ["white", "black", "navy", "burgundy"]) for (const width of [1440, 390]) {
  test(`${theme} ${width}: content alignment, dropdown and compact controls`, async ({page}) => {
    await setup(page); await page.setViewportSize({width,height:1000});
    await page.addInitScript(value => localStorage.setItem("mindflow_theme",value),theme);
    await page.route("**/api/v1/focus?**",route => route.fulfill({json:{date:"2026-10-08",session_count:1,sessions:[{id:"d",session_type:"distraction",duration_minutes:30,focus_score:28,start_time:"2026-10-08T09:00:00",end_time:"2026-10-08T09:30:00",dominant_app:"ChatGPT.exe",switch_count:12}]}}));
    await page.route("**/api/v1/intervention/history?**",route => route.fulfill({json:{items:[{id:"i",intervention_type:"task_breakdown",title:"小提示：先完成一步",message:"先整理一个图表，再继续下一步。",created_at:"2026-10-08T09:00:00Z",user_response:"dismissed",response_latency_s:3.438}],count:1}}));
    await page.route("**/api/v1/analytics/attribution",route => route.fulfill({json:{procrastination_type:"task_aversion",cbt_technique:"graded_exposure",evidence:"任务较大，可以从小步骤开始。"}}));
    await page.goto("/"); await expect(page.locator(".dash-hero")).toBeVisible();
    // Same "no ragged rows" property the old five-row check measured, now on
    // the single grid: the full-width blocks share one left edge and one
    // width, and the 2fr/1fr pair fills exactly that width.
    const box=async (s:string)=>page.locator(s).evaluate(el=>{const r=el.getBoundingClientRect();return {x:r.x,width:r.width,height:r.height};});
    const hero=await box(".dash-hero"), next=await box(".dash-next"), strip=await box(".sig-metrics"), trend=await box(".dash-trend");
    // 今天 and 最近 7 天 span the grid: same left edge, same width.
    expect(Math.abs(strip.x-hero.x)).toBeLessThan(1);
    expect(Math.abs(trend.x-strip.x)).toBeLessThan(1);
    expect(Math.abs(trend.width-strip.width)).toBeLessThan(1);
    if(width===1440) {
      expect(Math.abs(hero.x+hero.width+16-next.x), "hero and 下一步 are adjacent columns").toBeLessThan(1);
      expect(Math.abs(next.x+next.width-strip.x-strip.width), "the pair ends where the strip ends").toBeLessThan(1);
      expect(Math.abs(hero.height-next.height), "hero and 下一步 are equal-height peers").toBeLessThan(1);
    } else {
      expect(Math.abs(next.x-hero.x)).toBeLessThan(1);
      expect(Math.abs(next.width-hero.width)).toBeLessThan(1);
    }
    await page.goto("/focus"); await expect(page.locator(".conversation-item")).toHaveCount(1);
    // The 当日 strip is one surface with four cells (spec §3.3); the old
    // "state card is at least 145px tall" check measured the removed cards.
    await expect(page.locator(".f-state .sig-metric")).toHaveCount(4);
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
      // The telemetry readout is now five label—value rows (spec §8.6), so the
      // cells no longer have to line up column-for-column with the buttons;
      // what must still hold is one row per action and one shared button height.
      await expect(page.locator(".telemetry-stats > .kv-row")).toHaveCount(5);
      await expect(page.locator(".telemetry-actions > button")).toHaveCount(5);
      const buttons=await page.locator(".telemetry-actions > button").evaluateAll(els=>els.map(el=>({y:el.getBoundingClientRect().y,height:el.getBoundingClientRect().height})));
      const first=buttons[0];
      buttons.forEach(button=>{ expect(Math.abs(button.y-first.y)).toBeLessThan(1); expect(Math.abs(button.height-first.height)).toBeLessThan(1); });
    }
    await page.locator(".settings-control-grid").scrollIntoViewIfNeeded();
    await page.screenshot({path:`screenshots/20261008/density-${theme}-${width}-settings.png`});
    await page.route("**/api/v1/analytics/profile?**",route=>route.fulfill({json:{peak_focus_hours:[{hour:8,avg_score:52}],peak_focus:"8:00 (avg 52)",productivity_apps:["code.exe"],trigger_apps:[],avg_focus_block_min:30,details:{avg_focus_block_min:30,total_events:524,distraction_ratio:0.006}}}));
    await page.goto("/analytics"); await page.getByRole("button",{name:"运行归因分析"}).click();
    await expect(page.getByText("任务厌恶",{exact:true})).toBeVisible();
    await expect(page.getByText("分级面对任务",{exact:true})).toBeVisible();
    await expect(page.getByText("活动事件总数",{exact:true})).toBeVisible();
    // The range control is a Segmented group now (spec §8.4.2); the old
    // native <select> it replaced is gone, so assert the group is present and
    // that its options are reachable buttons.
    const rangeGroup = page.getByRole("group", { name: "时间范围" });
    await expect(rangeGroup).toBeVisible();
    await expect(rangeGroup.locator(".sig-seg-item")).toHaveCount(4);
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
