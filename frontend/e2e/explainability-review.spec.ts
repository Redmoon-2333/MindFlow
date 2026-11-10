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
  // §8.23.4: both empty states now say what is happening instead of naming a
  // missing payload — same intent (the block explains itself, not a blank).
  await expect(evidence.getByRole("region", { name: "动机分解" })).toContainText("这次分析没有算出动机分解。");
  await expect(evidence.getByRole("region", { name: "行为偏离" })).toContainText(
    "还在积累记录：用几天后，才能和你平时的水平比较。",
  );
  await expect(evidence.getByRole("region", { name: "识别到的认知扭曲" })).toHaveCount(0);
  await expect(evidence.getByText("下一步可以做")).toHaveCount(0);
});

/** §8.23.3: no recommended technique and nothing above the 0.2 threshold is
 *  the pair of conditions on which the backend declines to call it a pattern
 *  (`intervention_service._select_intervention_type`). The card used to name a
 *  type and a 0.15 score next to evidence saying no pattern was found; it now
 *  says so plainly. The bars stay — they are the support, not a verdict.
 *
 *  The second half is the contrast case: the same null technique with one type
 *  above the threshold must still be named, so the rule is not just "cbt null". */
test("analytics hides a below-threshold type but still names one above it", async ({ page }) => {
  const respond = (typeConfidence: Record<string, number>) => async (route: import("@playwright/test").Route) =>
    route.fulfill({ json: {
      // The card reads the top-level field, so the fixture must carry it — the
      // scenario under test is "backend named a type, but weakly".
      procrastination_type: "impulsivity",
      assessment: {
        procrastination_types: ["impulsivity"],
        type_confidence: typeConfidence,
        cbt_technique: null,
        response_text: "未检测到显著的拖延模式。",
      },
      source: "rule_engine", cached: false, meta: { degraded: false },
    } });

  await setupExplainabilityAbsent(page);
  await page.route("**/api/v1/analytics/attribution", respond({ impulsivity: 0.15 }));
  await page.setViewportSize(WIDE);
  await open(page, "/analytics");
  const assertClean = failOnPageError(page);

  await page.getByRole("button", { name: "运行归因分析" }).click();
  const card = page.locator(".fifth-box .left-box");
  const typeRow = card.locator(".kv-row").filter({ hasText: "拖延类型" });
  const confidenceRow = card.locator(".kv-row").filter({ hasText: "置信度" }).first();
  await expect(typeRow).toContainText("没有发现明显的拖延");
  await expect(typeRow, "the weak type must not be named as the verdict").not.toContainText("冲动分心");
  await expect(confidenceRow, "a below-threshold score is unknown, not a number").toContainText("—");
  await expect(confidenceRow).not.toContainText("0.15");

  // The evidence strip is a sibling of the card, not a child of it.
  const confidence = page.locator(".attribution-evidence").getByRole("region", { name: "拖延类型置信度" });
  await expect(confidence.getByText("都很低，仅供参考")).toBeVisible();
  // The bars are still the evidence: one row, with its real percentage.
  await expect(confidence.locator(".type-bar-row")).toHaveCount(1);
  await expect(confidence.locator(".type-bar-row")).toContainText("冲动分心");
  await expect(confidence.getByRole("img", { name: "冲动分心 置信度 15%" })).toBeVisible();
  assertClean();

  // Contrast: 0.55 clears the threshold, so the type is named again.
  await page.unroute("**/api/v1/analytics/attribution");
  await page.route("**/api/v1/analytics/attribution", respond({ impulsivity: 0.55 }));
  await page.reload();
  await page.getByRole("button", { name: "运行归因分析" }).click();
  await expect(typeRow).toContainText("冲动分心");
  await expect(typeRow).not.toContainText("没有发现明显的拖延");
  await expect(page.locator(".attribution-evidence").getByText("都很低，仅供参考")).toHaveCount(0);
});

/** §8.26.1: the four components all sat on the 0.5 prior, yet the card named
 *  E as "the main bottleneck" and outlined it — a prior presented as a
 *  finding. A component with `raw: null` is now marked 默认值, and a dominant
 *  component that is a default produces no bottleneck claim at all. */
test("tmt does not present prior values as findings", async ({ page }) => {
  const allDefault = {
    motivation: 0.5,
    motivation_level: "medium",
    // The dominant component is itself unmeasured — exactly the reported case.
    dominant_barrier: "expectancy",
    components: {
      expectancy: { score: 0.5, raw: null, label: "同类任务历史专注成功率", source: "先验（无近14天自评）", evidence_ids: [] },
      value: { score: 0.5, raw: null, label: "生产类应用占比 × 任务优先级", source: "先验", evidence_ids: [] },
      impulsiveness: { score: 0.5, raw: null, label: "应用切换频率", source: "先验", evidence_ids: [] },
      delay: { score: 0.5, raw: null, label: "距离截止时间的剩余天数", source: "先验", evidence_ids: [] },
    },
    note: "由行为代理变量估算，不是心理测量。",
  };
  const respond = (tmt: typeof allDefault) => async (route: import("@playwright/test").Route) =>
    route.fulfill({ json: {
      assessment: {
        procrastination_types: ["task_aversion"],
        type_confidence: { task_aversion: 0.6 },
        cbt_technique: "graded_exposure",
        response_text: "证据不足，暂不下结论。",
      },
      source: "rule_engine", cached: false, meta: { degraded: false }, tmt,
    } });

  await setupExplainabilityAbsent(page);
  await page.route("**/api/v1/analytics/attribution", respond(allDefault));
  await page.setViewportSize(WIDE);
  await open(page, "/analytics");
  const assertClean = failOnPageError(page);

  await page.getByRole("button", { name: "运行归因分析" }).click();
  const tmt = page.locator(".attribution-evidence").getByRole("region", { name: "动机分解" });

  await expect(tmt, "an all-prior reading must not claim a level").toContainText("数据不足");
  await expect(tmt).toContainText("数据还不够，暂不判断主要瓶颈");
  // No bottleneck badge, and therefore no outlined dominant row.
  await expect(tmt.locator(".mf-tmt-component-label .badge-warning")).toHaveCount(0);
  await expect(tmt.locator('.mf-tmt-component[data-dominant="true"]')).toHaveCount(0);
  // Each of the four components is tagged as a default, and the number is
  // still shown — the value is not hidden, only framed honestly.
  await expect(tmt.getByText("默认值", { exact: true })).toHaveCount(4);
  await expect(tmt.locator(".mf-tmt-component-value")).toHaveCount(4);
  await assertClean();

  // Contrast: with every component measured, the normal bottleneck line and
  // badge come back — the rule keys on `raw`, not on the card being empty.
  await page.unroute("**/api/v1/analytics/attribution");
  await page.route("**/api/v1/analytics/attribution", respond({
    ...allDefault,
    dominant_barrier: "delay",
    components: {
      expectancy: { ...allDefault.components.expectancy, raw: 0.52 },
      value: { ...allDefault.components.value, raw: 0.61 },
      impulsiveness: { ...allDefault.components.impulsiveness, raw: 1.9 },
      delay: { ...allDefault.components.delay, raw: 3.1 },
    },
  }));
  await page.reload();
  await page.getByRole("button", { name: "运行归因分析" }).click();
  await expect(tmt.getByText("主要瓶颈：D 延迟 · 截止时间压力是主要瓶颈")).toBeVisible();
  await expect(tmt.locator('.mf-tmt-component[data-dominant="true"]')).toContainText("D 延迟");
  await expect(tmt.getByText("默认值", { exact: true })).toHaveCount(0);
  await expect(tmt).not.toContainText("数据不足");
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
