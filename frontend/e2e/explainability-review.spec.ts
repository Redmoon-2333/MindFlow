import { test, expect, type Page } from "@playwright/test";
import { setupExplainable, setupExplainabilityAbsent, setupFeedbackRequest } from "./fixtures/theme-api";

/**
 * Explainability surfaces (contract §1–§5).
 *
 * Two things are asserted on every screen:
 *   1. the evidence actually renders when the backend sends it;
 *   2. the screen survives the backend *not* sending it (every field is
 *      optional), with neutral copy instead of an empty frame or a crash.
 *
 * Screenshots go to e2e/screenshots/midterm/ at desktop and phone widths.
 */

const SHOTS = "e2e/screenshots/midterm";
const WIDE = { width: 1440, height: 900 };
const NARROW = { width: 375, height: 812 };

async function open(page: Page, path: string, theme = "navy") {
  await page.addInitScript(value => localStorage.setItem("mindflow_theme", value), theme);
  await page.goto(path);
}

/** Any uncaught error or React failure banner fails the test immediately. */
function failOnPageError(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(String(error)));
  return () => expect(errors, `page errors: ${errors.join(" | ")}`).toHaveLength(0);
}

test("dashboard explains the prediction with signed contributions", async ({ page }) => {
  await setupExplainable(page);
  await page.setViewportSize(WIDE);
  await open(page, "/");
  const assertClean = failOnPageError(page);

  // The evidence now lives in the hero, behind "查看完整依据" (spec §4.1.1).
  const hero = page.locator(".dash-hero");
  await expect(hero).toBeVisible();
  await hero.locator(".sig-disclosure-trigger").click();

  const chart = hero.locator("figure.mf-chart");
  await expect(chart).toBeVisible();
  await expect(chart.locator("svg[role=img]")).toHaveAttribute("aria-label", /特征贡献/);
  // Five fixture contributions ⇒ five bars, and the pole labels prove the
  // chart encodes direction, not just magnitude.
  await expect(chart.locator("rect.mf-chart-bar")).toHaveCount(5);
  await expect(chart.getByText("推向专注 →")).toBeVisible();
  await expect(hero.getByText("TreeSHAP", { exact: false })).toBeVisible();
  await hero.scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/dashboard-1440-navy.png` });
  assertClean();
});

test("dashboard without an explanation shows neutral copy and no frame", async ({ page }) => {
  await setupExplainabilityAbsent(page);
  // A ready prediction with no attribution: the number is real, the evidence
  // is simply not shipped yet. (The absent fixture's default is "no model",
  // which is the *other* hero fallback, asserted in repair-20260930.)
  await page.route("**/api/v1/telemetry/focus-prediction", route =>
    route.fulfill({ json: { focus_probability: 0.62, status: "ready", mode: "personal_model", reason: "" } }));
  await page.setViewportSize(WIDE);
  await open(page, "/");

  const hero = page.locator(".dash-hero");
  await expect(hero.locator(".hero-num")).toHaveText("62%");
  // No explanation ⇒ one neutral line, no chart, no empty disclosure trigger.
  await expect(hero.locator(".hero-reason-empty")).toHaveText("当前模型不提供逐项依据");
  await expect(hero.locator("figure.mf-chart")).toHaveCount(0);
  await expect(hero.getByRole("button", { name: "查看完整依据" })).toHaveCount(0);
});

test("analytics renders type confidence, TMT and the shrunk baseline", async ({ page }) => {
  await setupExplainable(page);
  await page.setViewportSize(WIDE);
  await open(page, "/analytics");
  const assertClean = failOnPageError(page);

  await page.getByRole("button", { name: "运行归因分析" }).click();
  const evidence = page.locator(".attribution-evidence");
  await expect(evidence).toBeVisible();

  // §1 type confidence — every type with a number, leading one emphasised.
  const confidence = evidence.getByRole("region", { name: "拖延类型置信度" });
  await expect(confidence.locator(".type-bar-row")).toHaveCount(5);
  await expect(confidence.locator('.type-bar-row[data-leading="true"]')).toContainText("任务厌恶");
  await expect(confidence.getByText("72%")).toBeVisible();

  // §2 CBT extras that the old UI dropped on the floor.
  await expect(evidence.getByRole("region", { name: "识别到的认知扭曲" })).toContainText("灾难化预期");
  await expect(evidence.getByText("下一步可以做")).toBeVisible();

  // §3 TMT — the formula must be on screen, and the dominant term outlined.
  // §8.20.3 removed the English formula line; the card head carries the same
  // statement in Chinese, so the intent is asserted there.
  const tmt = evidence.getByRole("region", { name: "动机分解" });
  await expect(tmt.getByText("期望 × 价值 ÷ (1 + 冲动 × 延迟)")).toBeVisible();
  await expect(tmt.getByText("Motivation = E · V / (1 + I · D)")).toHaveCount(0);
  await expect(tmt.locator('.mf-tmt-component[data-dominant="true"]')).toContainText("D 延迟");
  await expect(tmt.getByText("主要瓶颈：D 延迟")).toBeVisible();

  // §4 deviation — robust z, and the shrinkage metadata the baseline relies on.
  const deviation = evidence.getByRole("region", { name: "行为偏离" });
  await expect(deviation).toContainText("中度偏离");
  await expect(deviation).toContainText("经验贝叶斯收缩 n=4，收缩系数 0.44");
  await expect(deviation.locator(".deviation-table tbody tr")).toHaveCount(3);
  await expect(deviation.locator(".deviation-table")).toContainText("2.71");

  await evidence.scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/analytics-1440-navy.png` });
  assertClean();
});

test("analytics without deviation or TMT still explains what it can", async ({ page }) => {
  await setupExplainabilityAbsent(page);
  await page.setViewportSize(WIDE);
  await open(page, "/analytics");

  await page.getByRole("button", { name: "运行归因分析" }).click();
  const evidence = page.locator(".attribution-evidence");
  await expect(evidence.getByRole("region", { name: "拖延类型置信度" })).toContainText("暂无类型置信度数据");
  await expect(evidence.getByRole("region", { name: "动机分解" })).toContainText("暂无动机分解数据");
  await expect(evidence.getByRole("region", { name: "行为偏离" })).toContainText("暂无偏离数据");
  await expect(evidence.getByRole("region", { name: "识别到的认知扭曲" })).toHaveCount(0);
  await expect(evidence.getByText("下一步可以做")).toHaveCount(0);
});

test("panel exposes how the verdict was reached", async ({ page }) => {
  await setupExplainable(page);
  await page.setViewportSize(WIDE);
  await open(page, "/panel");
  const assertClean = failOnPageError(page);

  await page.getByRole("button", { name: "查看上次结果" }).click();

  const meta = page.getByLabel("本次分析的过程信息");
  await expect(meta).toBeVisible();
  await expect(meta).toContainText("不确定度");
  await expect(meta).toContainText("升级复核");
  await expect(meta).toContainText("模型调用");
  await expect(meta).toContainText("6次");

  await expect(page.getByRole("heading", { name: "动机分解" })).toBeVisible();

  // The transcript is cards, and it says who spoke.
  await page.getByText(/专家讨论记录/).click();
  const opinions = page.locator(".opinion-card");
  await expect(opinions).toHaveCount(1);
  await expect(opinions.first().locator(".opinion-role")).toContainText("分析师");

  await page.getByRole("heading", { name: "动机分解" }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/panel-1440-navy.png` });
  assertClean();
});

test("intervention says why this reminder was chosen", async ({ page }) => {
  await setupExplainable(page);
  await page.setViewportSize(WIDE);
  await open(page, "/intervention");
  const assertClean = failOnPageError(page);

  const latest = page.getByLabel("为什么是这条提醒").first();
  await expect(latest).toBeVisible();
  // §8.17.2 renames the method to plain Chinese.
  await expect(latest.getByText("按效果挑选", { exact: true })).toBeVisible();
  await expect(latest).toContainText("情境：下午 · 分心");
  // Posterior bars: the chosen arm is marked, the rest stay muted.
  const chosen = latest.locator('.mf-posterior-row[data-chosen="true"]');
  await expect(chosen).toHaveCount(1);
  await expect(chosen).toContainText("任务拆解");
  await expect(chosen).toContainText("已选");

  // History items carry their own, collapsible rationale.
  // The second history item was chosen by the plain rule engine, so its
  // disclosure must say so instead of implying the bandit picked it.
  const historyDetails = page.locator("details.rationale-history");
  await expect(historyDetails).toHaveCount(2);
  await historyDetails.first().locator("summary").click();
  await expect(historyDetails.first()).toContainText("按效果挑选");
  await historyDetails.nth(1).locator("summary").click();
  await expect(historyDetails.nth(1)).toContainText("按规则挑选");

  await latest.scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/intervention-1440-navy.png` });
  assertClean();
});

test("proactive feedback question can be answered and disappears", async ({ page }) => {
  await setupExplainable(page);
  await setupFeedbackRequest(page);
  await page.setViewportSize(WIDE);
  let answered: { label?: string } | null = null;
  await page.route("**/api/v1/telemetry/feedback-requests/fb-1/answer", async route => {
    answered = route.request().postDataJSON() as { label?: string };
    await route.fulfill({ json: { request_id: "fb-1", labeled_windows: 3, status: "recorded" } });
  });

  await open(page, "/");
  const prompt = page.locator(".feedback-prompt");
  await expect(prompt).toBeVisible();
  await expect(prompt).toContainText("你在专注吗");
  await expect(prompt).toContainText("为什么问：");

  await prompt.getByRole("button", { name: "专注", exact: true }).click();
  await expect(prompt).toHaveCount(0);
  expect(answered).toEqual({ label: "focus" });
  await expect(page.locator(".toast")).toContainText("已记录：刚才在专注");
});

test("a request the server already dropped is reported as expired, not recorded", async ({ page }) => {
  await setupExplainable(page);
  await setupFeedbackRequest(page, { httpStatus: 404 });
  await page.setViewportSize(WIDE);
  await open(page, "/");

  const prompt = page.locator(".feedback-prompt");
  await expect(prompt).toBeVisible();
  await prompt.getByRole("button", { name: "专注", exact: true }).click();

  // 404 = nothing was written. Claiming "已记录" here would be a lie.
  await expect(page.locator(".toast")).toContainText("已过期");
  await expect(page.locator(".toast")).not.toContainText("已记录");
  await expect(prompt).toHaveCount(0);
});

test("a late answer the server rejects as expired clears the prompt", async ({ page }) => {
  await setupExplainable(page);
  await setupFeedbackRequest(page, { status: "expired" });
  await page.setViewportSize(WIDE);
  await open(page, "/");

  const prompt = page.locator(".feedback-prompt");
  await expect(prompt).toBeVisible();
  await prompt.getByRole("button", { name: "专注", exact: true }).click();

  await expect(page.locator(".toast")).toContainText("已过期");
  await expect(prompt).toHaveCount(0);
});

test("narrow screens keep the explainability cards inside the viewport", async ({ page }) => {
  await setupExplainable(page);
  await page.setViewportSize(NARROW);
  await open(page, "/");
  const hero = page.locator(".dash-hero");
  await expect(hero).toBeVisible();
  await hero.locator(".sig-disclosure-trigger").click();
  await hero.locator("figure.mf-chart").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/dashboard-375-navy.png` });
  expect(await page.locator(".mf-main").evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);

  await page.goto("/analytics");
  await page.getByRole("button", { name: "运行归因分析" }).click();
  await expect(page.locator(".attribution-evidence")).toBeVisible();
  await page.locator(".attribution-evidence").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/analytics-375-navy.png` });
  expect(await page.locator(".mf-main").evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);

  await page.goto("/panel");
  await page.getByRole("button", { name: "查看上次结果" }).click();
  await expect(page.getByLabel("本次分析的过程信息")).toBeVisible();
  await page.getByRole("heading", { name: "动机分解" }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/panel-375-navy.png` });
  expect(await page.locator(".mf-main").evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);

  await page.goto("/intervention");
  await expect(page.getByLabel("为什么是这条提醒").first()).toBeVisible();
  await page.getByLabel("为什么是这条提醒").first().scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${SHOTS}/intervention-375-navy.png` });
  expect(await page.locator(".mf-main").evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
});
