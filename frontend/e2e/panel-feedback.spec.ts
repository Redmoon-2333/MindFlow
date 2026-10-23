import {test, expect, type Page} from "@playwright/test";

async function setup(page: Page) {
  await page.addInitScript(() => localStorage.setItem("mindflow_authenticated", "1"));
  await page.routeWebSocket("**/api/v1/ws", () => {});
  await page.route("**/api/v1/**", route => route.fulfill({json:{}}));
}
const old = {types:["impulsivity"],confidence:{impulsivity:.9},rationale:"此前保存的分析",transcript:[],degraded:true,source:"single_expert",cached:true,meta:{source:"single_expert",degradation_path:["panel_unavailable"]}};

for (const outcome of ["degraded", "failed", "succeeded", "http-error"] as const) {
  test(`panel reports pending and ${outcome} instead of silently restoring cache`, async ({page}) => {
    await setup(page);
    await page.route("**/api/v1/panel", route => route.fulfill({json:old}));
    let release!: () => void;
    const held = new Promise<void>(resolve => {release=resolve;});
    await page.route("**/api/v1/panel/today", async route => {
      await held;
      if (outcome === "http-error") return route.fulfill({status:500,json:{detail:"请求失败"}});
      return route.fulfill({json:{...old, cached:outcome !== "succeeded", degraded:outcome !== "succeeded", source:outcome === "succeeded" ? "panel" : "single_expert", meta:{...old.meta,attempt_status:outcome,attempt_degradation_path:[outcome === "failed" ? "retry_error" : "panel_rejected"]}}});
    });
    await page.goto("/panel"); await page.getByRole("button",{name:"查看上次结果",exact:true}).click();
    await page.getByRole("button",{name:"重试 AI 分析",exact:true}).click();
    await expect(page.getByText(/正在运行 AI 专家分析/)).toBeVisible();
    await expect(page.getByRole("button",{name:"重试中…"})).toBeDisabled();
    await expect(page.getByText("此前保存的分析",{exact:true})).toBeVisible();
    await page.screenshot({path:`screenshots/20261009/panel-pending-${outcome}.png`});
    release();
    await expect(page.getByText(/正在运行 AI 专家分析/)).toHaveCount(0);
    if (outcome === "degraded") await expect(page.getByRole("status")).toContainText("本次重试仍未完成专家讨论，当前保留此前结果。专家裁决未通过复核");
    else if (outcome === "failed") await expect(page.getByRole("alert")).toContainText("本次重试失败");
    else if (outcome === "http-error") await expect(page.getByRole("alert")).toContainText("请求失败");
    else await expect(page.getByRole("status")).toContainText("专家分析已完成");
  });
}

for (const theme of ["white","black","navy","burgundy"]) for (const width of [1440,390]) {
  test(`delete dialog fits content ${theme} ${width}`, async ({page}) => {
    await page.setViewportSize({width,height:900}); await setup(page);
    await page.addInitScript(value => localStorage.setItem("mindflow_theme",value),theme);
    await page.route("**/api/v1/chat/sessions",route => route.fulfill({json:[{session_id:"s1",title:"非常长的会话标题".repeat(9)}]}));
    await page.goto("/chat"); await page.getByRole("button",{name:/^删除 /}).click();
    const dialog=page.getByRole("dialog"); await expect(dialog).toBeVisible();
    const geometry=await dialog.evaluate(el => {
      const box=el.getBoundingClientRect(), description=el.querySelector("p")!.getBoundingClientRect(), buttons=el.querySelector(".chat-dialog-actions")!.getBoundingClientRect();
      return {width:box.width,x:box.x,padding:parseFloat(getComputedStyle(el).paddingLeft),descriptionBottom:description.bottom,buttonsTop:buttons.top,buttonsBottom:buttons.bottom,bottom:box.bottom};
    });
    expect(geometry.padding).toBeGreaterThanOrEqual(24);
    expect(geometry.x).toBeGreaterThanOrEqual(0);
    expect(geometry.width).toBeLessThanOrEqual(width-32);
    expect(geometry.buttonsTop-geometry.descriptionBottom).toBeGreaterThanOrEqual(20);
    expect(geometry.bottom-geometry.buttonsBottom).toBeGreaterThanOrEqual(24);
    await page.screenshot({path:`screenshots/20261009/delete-dialog-${theme}-${width}.png`});
    await page.getByRole("button",{name:"取消",exact:true}).click(); await expect(dialog).toHaveCount(0);
  });
}
