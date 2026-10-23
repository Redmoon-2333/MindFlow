import { test, expect } from "@playwright/test";
import { setup } from "./fixtures/theme-api";

const paths = ["/", "/focus", "/activities", "/analytics", "/reports", "/intervention", "/panel", "/chat", "/settings", "/model-center", "/execution", "/diagnostics"];
for (const theme of ["white", "black", "navy", "burgundy"]) for (const width of [1440, 390]) for (const path of paths) {
  test(`${theme} ${width} ${path}: global theme and layout`, async ({page}) => {
    await page.setViewportSize({width,height:1000}); await setup(page);
    await page.addInitScript(value => localStorage.setItem("mindflow_theme", value), theme);
    const errors: string[] = []; page.on("pageerror", e => errors.push(e.message));
    await page.goto(path); await expect(page.locator("html")).toHaveAttribute("data-theme",theme);
    await expect(page.locator(".mf-header-name")).toBeVisible();
    await expect(page.locator(".mf-main .spinner")).toHaveCount(0);
    if (path === "/") await expect(page.locator(".d-extra-card").first()).toBeVisible();
    if (path === "/settings") { await page.getByRole("button",{name:"外观与 AI",exact:true}).click(); await expect(page.getByRole("button",{name:"保存 AI Key"})).toBeVisible(); }
    if (path === "/activities") await expect(page.locator(".present-acti")).toContainText("研究报告");
    if (path === "/analytics") await expect(page.getByText("very-long-browser-application.exe")).toBeVisible();
    if (path === "/model-center") await expect(page.getByText("原始事件",{exact:true})).toBeVisible();
    if (path === "/execution") await expect(page.getByLabel("任务标题")).toBeVisible();
    await expect.poll(() => page.locator(".mf-main").evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    expect(errors).toEqual([]);
    if (path === "/activities") {
      const rects=await page.evaluate(() => [".present-acti", ".choose-time"].map(selector => {const r=document.querySelector(selector)!.getBoundingClientRect();return {top:r.top,bottom:r.bottom};}));
      expect(rects[1].top).toBeGreaterThanOrEqual(rects[0].bottom);
    }
    if (width === 1440 && path === "/") {
      const cards=await page.evaluate(() => [".d-control", ".d-inference", ".d-extra-card:first-child", ".d-extra-card:last-child"].map(s => {const r=document.querySelector(s)!.getBoundingClientRect();return {left:r.left,width:r.width};}));
      expect(Math.abs(cards[0].left-cards[2].left)).toBeLessThan(2); expect(Math.abs(cards[1].width-cards[3].width)).toBeLessThan(2);
    }
    await page.screenshot({path:`screenshots/20261008/${theme}-${width}-${path.slice(1)||"dashboard"}.png`,fullPage:true});
  });
}

test("theme persists and primary/secondary text meet contrast across surfaces", async ({page}) => {
  await setup(page); await page.goto("/settings"); await page.getByRole("button",{name:"外观与 AI",exact:true}).click();
  for (const [id,name] of [["white","白色"],["black","黑色"],["navy","深蓝色"],["burgundy","深红色"]]) {
    await page.getByRole("button",{name,exact:false}).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme",id);
    const ratios = await page.evaluate(() => {
      const css=getComputedStyle(document.documentElement);
      function luminance(hex:string){const rgb=hex.trim().replace("#","").match(/../g)!.map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);return rgb[0]*.2126+rgb[1]*.7152+rgb[2]*.0722;}
      const result:number[]=[];
      for(const fg of ["--mf-text","--mf-text-dim"]) for(const bg of ["--color-bg","--mf-panel","--color-bg-inset"]) {const a=luminance(css.getPropertyValue(fg)),b=luminance(css.getPropertyValue(bg));result.push((Math.max(a,b)+.05)/(Math.min(a,b)+.05));}
      return result;
    }); expect(Math.min(...ratios)).toBeGreaterThanOrEqual(4.5);
  }
  await page.reload(); await expect(page.locator("html")).toHaveAttribute("data-theme","burgundy");
});

test("AI key saves without persisting in browser and reports restart requirement", async ({page}) => {
  await setup(page);
  await page.route("**/api/v1/settings/ai", route=>route.fulfill({json:{saved:route.request().method()==="PUT",active_configured:false,environment_override:false,restart_required:route.request().method()==="PUT",provider:"DeepSeek",model:"deepseek-flash"}}));
  await page.goto("/settings"); await page.getByRole("button",{name:"外观与 AI",exact:true}).click();
  const secret="sk-browser-test-only-12345"; await page.getByLabel("DeepSeek API Key",{exact:true}).fill(secret);
  const req=page.waitForRequest(r=>r.method()==="PUT"&&r.url().endsWith("/settings/ai"));
  await page.getByRole("button",{name:"保存 AI Key"}).click(); expect((await req).postDataJSON()).toEqual({api_key:secret});
  await expect(page.getByLabel("DeepSeek API Key",{exact:true})).toHaveValue("");
  await expect(page.getByText("等待重启生效")).toBeVisible();
  expect(await page.evaluate(() => JSON.stringify(localStorage)+JSON.stringify(sessionStorage))).not.toContain(secret);
  await page.getByRole("button",{name:"移除保存的 Key"}).click(); await expect(page.getByText("本机未保存 Key",{exact:true})).toBeVisible();
});

for (const theme of ["white", "black", "navy", "burgundy"]) {
  test(`${theme}: reported model status and task alignment`, async ({page}) => {
    await page.setViewportSize({width:1440,height:1000}); await setup(page);
    await page.addInitScript(value => localStorage.setItem("mindflow_theme",value),theme);
    await page.goto("/model-center");
    await page.getByRole("tab",{name:"模型状态",exact:true}).click();
    await expect(page.locator(".mc-model-mode")).toHaveCount(2);
    const spacing=await page.locator(".mc-model-mode").evaluateAll(rows => {
      const a=rows[0].querySelector(".badge")!.getBoundingClientRect();
      const b=rows[1].querySelector(".badge")!.getBoundingClientRect(); return b.top-a.bottom;
    }); expect(spacing).toBeGreaterThanOrEqual(24);
    await page.screenshot({path:`screenshots/20261008/${theme}-1440-model-status.png`});
    await page.goto("/execution"); await expect(page.getByLabel("预计分钟")).toBeVisible();
    const input=await page.getByLabel("预计分钟").boundingBox();
    const button=await page.getByRole("button",{name:"添加任务",exact:true}).boundingBox();
    expect(Math.abs(input!.y+input!.height-button!.y-button!.height)).toBeLessThan(2);
    await page.screenshot({path:`screenshots/20261008/${theme}-1440-task-alignment.png`});
    await page.goto("/"); await expect(page.locator(".d-extra-card").first()).toBeVisible();
    await page.locator(".mf-main").evaluate(el => {el.scrollTop=el.scrollHeight;});
    await page.screenshot({path:`screenshots/20261008/${theme}-1440-dashboard-bottom.png`});
  });
}
