/**
 * Regression assertions for the four defects confirmed by
 * docs/audit/20260930-review/acceptance.md (F1–F4).
 *
 * These are real assertions (unlike ../docs/audit/20260930-review/reproduce.cjs,
 * which only *displays* the defects). Every test here fails while the defect is
 * present and passes after the fix.
 *
 * The suite is fully isolated: all /api/v1/** traffic is intercepted, the
 * clock is pinned to 2026-09-30 (the date used by the original review), and no
 * backend, training job or paid LLM call is involved.
 *
 * Run: npx playwright test e2e/repair-20260930.spec.ts --workers=1 --reporter=list
 */
import { test, expect, type Page, type Route } from "@playwright/test";

const TODAY = "2026-09-30";

/** JSON bodies for the endpoints the four pages read. */
const baseResponses: Record<string, unknown> = {
  "/analytics/model-status": { loaded: false, mode: "rule_engine_only", reasons: [] },
  "/ai/provider-status": { provider: "generic", model: "none", configured: false, ollama_enabled: false },
  "/intervention/history": { items: [], count: 0, has_more: false },
  "/collector": { status: "stopped", running: false, message: null },
  "/autonomy": { enabled: true, paused_until: null, paused: false },
  "/telemetry/focus-prediction": {
    focus_probability: null,
    status: "no_model",
    mode: "rule_engine_only",
    reason: "未加载 ML 模型",
  },
  "/health": {
    status: "ok",
    version: "test",
    timestamp: "2026-09-30T04:00:00+00:00",
    collector: { status: "stopped" },
    database: { status: "ok", connected: true },
    migration: { applied: true },
  },
  "/analytics/profile": {
    peak_focus_hours: [],
    top_apps: [],
    avg_focus_block_min: 10,
    profile_date: TODAY,
    details: {},
  },
};

function trendBody(daily: Array<Record<string, unknown>>) {
  return {
    days: 7,
    start_date: "2026-09-24",
    end_date: TODAY,
    total_sessions: daily.reduce((sum, d) => sum + Number(d.session_count ?? 0), 0),
    daily,
  };
}

const DEFAULT_TREND = trendBody([
  { date: "2026-09-29", focus_min: 77, distraction_min: 0, session_count: 1, avg_score: 80 },
]);

function patternsBody(range: string) {
  return {
    high_switch_periods: [{ hour: 9, switch_count: 1, period: range }],
    trigger_apps: [],
    heatmap: [],
    total_sessions: 1,
    distraction_ratio: 0,
  };
}

interface MockOptions {
  /** Per-path override; return `{ status }` to answer with an error. */
  override?: (path: string, search: URLSearchParams) => { json?: unknown; status?: number } | undefined;
  /** Paths whose response is parked until `release()` is called. */
  hold?: (path: string, search: URLSearchParams) => boolean;
}

/**
 * Intercept every API call. Returns a `release()` that flushes held routes in
 * the order they arrived — that is what makes the F3 race deterministic.
 */
async function installApi(page: Page, options: MockOptions = {}) {
  /** Body derivation shared by live and released routes. */
  const bodyFor = (path: string, search: URLSearchParams): { json?: unknown; status?: number } | undefined => {
    const override = options.override?.(path, search);
    if (override) return override;
    if (path === "/analytics/baseline") return { status: 404 };
    if (path === "/focus/trend") return { json: DEFAULT_TREND };
    if (path === "/focus") return { json: { date: TODAY, session_count: 0, sessions: [] } };
    if (path === "/analytics/patterns") return { json: patternsBody(`RANGE_${search.get("days") ?? "7"}`) };
    if (path === "/activities/current") {
      return {
        status: 404,
      };
    }
    if (baseResponses[path] !== undefined) return { json: baseResponses[path] };
    return { status: 503 };
  };

  const held: Route[] = [];
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const search = url.searchParams;
    const path = url.pathname.replace(/^\/api\/v1/, "");

    if (options.hold?.(path, search)) {
      held.push(route);
      return;
    }

    const body = bodyFor(path, search);
    if (body?.status) {
      await route.fulfill({
        status: body.status,
        json:
          body.status === 404
            ? { type: "https://mindflow.app/errors/not-found", title: "Not Found", status: 404, detail: "暂无活动记录", instance: path }
            : { type: "https://mindflow.app/errors/server-error", title: "Internal Server Error", status: body.status, detail: "isolated repair harness", instance: path },
      });
      return;
    }
    await route.fulfill({ json: body?.json ?? {} });
  });

  return {
    heldCount: () => held.length,
    /** Fulfill every parked route with the body it would have had. */
    release: async () => {
      const pending = held.splice(0, held.length);
      for (const route of pending) {
        const url = new URL(route.request().url());
        const path = url.pathname.replace(/^\/api\/v1/, "");
        const body = bodyFor(path, url.searchParams);
        if (body?.status) {
          await route.fulfill({
            status: body.status,
            json: { type: "https://mindflow.app/errors/server-error", title: "Internal Server Error", status: body.status, detail: "isolated repair harness", instance: path },
          });
        } else {
          await route.fulfill({ json: body?.json ?? {} });
        }
      }
      return pending.length;
    },
  };
}

async function bootIsolated(page: Page) {
  await page.clock.setFixedTime(new Date("2026-09-30T12:00:00+08:00"));
  await page.addInitScript(() => window.localStorage.setItem("mindflow_authenticated", "1"));
  await page.routeWebSocket("**/*", () => {});
}

function statItem(page: Page, title: string) {
  return page.locator(".d-statistic-item").filter({ hasText: title });
}

async function focusStat(page: Page, title: string) {
  return page.locator(".f-state .state-item").filter({ hasText: title }).locator(".state-item-number");
}

async function chartSlots(page: Page) {
  return page.locator(".statics-chart-item").evaluateAll((els) =>
    els.map((el) => ({
      date: el.querySelector(".date")?.textContent?.trim() ?? "",
      week: el.querySelector(".week")?.textContent?.trim() ?? "",
    })),
  );
}

// ═════════════════════════════════════════════════════════════════════════
// F1 — the last trend day must not be presented as "today"/"yesterday"
// ═════════════════════════════════════════════════════════════════════════

test("F1 dashboard: today with no record shows 0, not the previous day's 77", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.goto("/");

  const minutes = statItem(page, "今日专注时长").locator(".d-statistic-number");
  await expect(minutes).toBeVisible({ timeout: 10000 });
  await expect(minutes).toHaveText("0");

  // No record for today also means no meaningful average / distraction ratio.
  await expect(statItem(page, "平均专注分数").locator(".d-statistic-number")).toHaveText("--");
  // And no yesterday → the comparison must not be invented.
  await expect(statItem(page, "今日专注时长").locator(".d-statistic-stat")).toHaveText("较昨日 —");
});

test("demo dashboard: real probability does not imply personal-model readiness", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: path => path === "/analytics/model-status"
      ? { json: { loaded: true, ready: false, demo_only: true, mode: "shadow", version: "demo_v4" } }
      : path === "/telemetry/focus-prediction"
        ? { json: { focus_probability: 0.898, status: "ready", mode: "ml", reason: "" } }
        : undefined,
  });
  await page.goto("/");
  const prediction = page.locator(".d-extra-card").filter({ hasText: "ML 专注预测" });
  const service = page.locator(".d-extra-card").filter({ hasText: "服务状态" });
  await expect(prediction.locator(".d-extra-value")).toContainText("89.8%");
  await expect(prediction).toContainText("合成数据演示模型 · 影子模式");
  await expect(service).toContainText("演示模型已加载 · 影子模式");
  await expect(service).toContainText("未通过个人模型质量门");
  await expect(service).not.toContainText("已就绪");
});

test("demo dashboard: unavailable prediction retains its stale-data reason", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: path => path === "/analytics/model-status"
      ? { json: { loaded: true, ready: false, demo_only: true, mode: "shadow" } }
      : path === "/telemetry/focus-prediction"
        ? { json: { focus_probability: null, status: "stale", mode: "ready", reason: "数据已过期" } }
        : undefined,
  });
  await page.goto("/");
  const prediction = page.locator(".d-extra-card").filter({ hasText: "ML 专注预测" });
  await expect(prediction.locator(".d-extra-value")).toHaveText("--");
  await expect(prediction).toContainText("合成数据演示模型 · 影子模式 · 数据已过期");
});

test("F1 dashboard: chart never labels a historical day 今天", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.goto("/");
  await expect(statItem(page, "今日专注时长")).toBeVisible({ timeout: 10000 });

  const labels = await page.locator(".d-state-item").first().evaluate(() => true).catch(() => false);
  expect(labels).toBeTruthy();

  const chartLabels = await page.locator(".d-chart-box-item-week").allTextContents();
  expect(chartLabels, "09-29 must keep a weekday label").toContain("周二");
  expect(chartLabels, "no slot may claim to be today when today has no data").not.toContain("今天");
});

test("F1 dashboard: 较昨日 must not compare against a non-consecutive day", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: (path) =>
      path === "/focus/trend"
        ? {
            json: trendBody([
              { date: "2026-09-28", focus_min: 50, distraction_min: 0, session_count: 1, avg_score: 70 },
              { date: "2026-09-30", focus_min: 100, distraction_min: 0, session_count: 1, avg_score: 90 },
            ]),
          }
        : undefined,
  });
  await page.goto("/");

  const minutes = statItem(page, "今日专注时长").locator(".d-statistic-number");
  await expect(minutes).toBeVisible({ timeout: 10000 });
  await expect(minutes, "09-30 is today and has data").toHaveText("100");
  await expect(
    statItem(page, "今日专注时长").locator(".d-statistic-stat"),
    "09-29 is missing, so a delta against 09-28 would be a fabricated comparison",
  ).toHaveText("较昨日 —");
  // The card still labels only the real today as 今天.
  const chartLabels = await page.locator(".d-chart-box-item-week").allTextContents();
  expect(chartLabels).toContain("今天");
  expect(chartLabels.filter((t) => t === "今天")).toHaveLength(1);
});

test("F1 dashboard: a failed trend request shows -- (unknown), never 0 (known-empty)", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: (path) => (path === "/focus/trend" ? { status: 500 } : undefined),
  });
  await page.goto("/");

  const minutes = statItem(page, "今日专注时长").locator(".d-statistic-number");
  await expect(minutes).toBeVisible({ timeout: 10000 });
  await expect(minutes, "a failed request is unknown, not measured-zero").toHaveText("--");
  await expect(page.locator(".error-box"), "the failure must be surfaced").toBeVisible();
});

test("F1 focus: a historical slot keeps its date label instead of 今天", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: (path) =>
      path === "/focus"
        ? {
            json: {
              date: TODAY,
              session_count: 2,
              sessions: [
                { id: "f", session_type: "focus", duration_minutes: 10, focus_score: 90, start_time: `${TODAY}T09:00:00`, end_time: `${TODAY}T09:10:00`, dominant_app: "Audit", switch_count: 0 },
                { id: "d", session_type: "distraction", duration_minutes: 120, focus_score: 10, start_time: `${TODAY}T10:00:00`, end_time: `${TODAY}T12:00:00`, dominant_app: "Audit", switch_count: 0 },
              ],
            },
          }
        : undefined,
  });
  await page.goto("/focus");
  await expect(page.locator(".statics-chart")).toBeVisible({ timeout: 10000 });

  const slots = await chartSlots(page);
  const slot29 = slots.find((s) => s.date === "09-29");
  expect(slot29, "the 09-29 slot must be rendered").toBeTruthy();
  expect(slot29?.week, "09-29 is yesterday-of-nothing, it must not read 今天").not.toBe("今天");
  expect(
    slots.filter((s) => s.week === "今天"),
    "today has no trend data, so no slot may claim 今天",
  ).toHaveLength(0);
});

test("F1 focus: a failed trend request is distinguished from an empty trend", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: (path) => (path === "/focus/trend" ? { status: 500 } : undefined),
  });
  await page.goto("/focus");
  await expect(page.locator(".f-statics-title")).toBeVisible({ timeout: 10000 });

  // Auto-waiting assertions: the request must settle into an error message,
  // and that message must not be the "no data" wording.
  await expect(page.locator(".f-statics"), "a failed request must not read as 'no data'").toContainText(
    "趋势加载失败",
    { timeout: 10000 },
  );
  await expect(page.locator(".f-statics")).not.toContainText("暂无趋势数据");
});

// ═════════════════════════════════════════════════════════════════════════
// F2 — focus metrics use one coherent session set
// ═════════════════════════════════════════════════════════════════════════

const MIXED_SESSIONS = [
  { id: "f", session_type: "focus", duration_minutes: 10, focus_score: 90, start_time: `${TODAY}T09:00:00`, end_time: `${TODAY}T09:10:00`, dominant_app: "Audit", switch_count: 0 },
  { id: "d", session_type: "distraction", duration_minutes: 120, focus_score: 10, start_time: `${TODAY}T10:00:00`, end_time: `${TODAY}T12:00:00`, dominant_app: "Audit", switch_count: 0 },
];

async function openFocusWithSessions(page: Page) {
  await bootIsolated(page);
  await installApi(page, {
    override: (path) =>
      path === "/focus"
        ? { json: { date: TODAY, session_count: MIXED_SESSIONS.length, sessions: MIXED_SESSIONS } }
        : undefined,
  });
  await page.goto("/focus");
  await expect(page.locator(".f-state .state-item").first()).toBeVisible({ timeout: 10000 });
}

test("F2 focus: 10m focus + 120m distraction reports 1 focus session / 10m longest", async ({ page }) => {
  await openFocusWithSessions(page);

  await expect(await focusStat(page, "总专注时长")).toHaveText("10m");
  await expect(await focusStat(page, "专注次数")).toHaveText("1 次");
  await expect(await focusStat(page, "最长专注")).toHaveText("10m");
});

test("F2 focus: 平均评分 keeps the documented all-sessions policy", async ({ page }) => {
  await openFocusWithSessions(page);
  // Product decision recorded with this fix: 平均评分 averages every session
  // of the day (matching backend /focus/trend avg_score = score_sum / count),
  // while duration/count/longest use the focus set only.
  await expect(await focusStat(page, "平均评分")).toHaveText("50.0");
});

test("F2 focus: distraction sessions stay listed with their feedback controls", async ({ page }) => {
  await openFocusWithSessions(page);

  const rows = page.locator(".conversation-item");
  await expect(rows).toHaveCount(2, { timeout: 10000 });
  await expect(rows.last()).toContainText("2h 0m");
  await expect(rows.last()).toContainText("分心");
  // The distraction row must still offer the feedback form.
  await expect(rows.last().locator(".self-accession-table")).toBeVisible();
});

test("F2 focus: an all-distraction day reports 0 focus minutes, not a crash", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: (path) =>
      path === "/focus"
        ? {
            json: {
              date: TODAY,
              session_count: 1,
              sessions: [
                { id: "d", session_type: "distraction", duration_minutes: 120, focus_score: 10, start_time: `${TODAY}T10:00:00`, end_time: `${TODAY}T12:00:00`, dominant_app: "Audit", switch_count: 0 },
              ],
            },
          }
        : undefined,
  });
  await page.goto("/focus");
  await expect(page.locator(".f-state .state-item").first()).toBeVisible({ timeout: 10000 });

  await expect(await focusStat(page, "总专注时长")).toHaveText("0m");
  await expect(await focusStat(page, "专注次数")).toHaveText("0 次");
  await expect(await focusStat(page, "最长专注")).toHaveText("—");
  await expect(page.locator(".conversation-item")).toHaveCount(1);
});

// ═════════════════════════════════════════════════════════════════════════
// F3 — a late response for an older range must not overwrite the current one
// ═════════════════════════════════════════════════════════════════════════

test("F3 analytics: a late 7-day response cannot replace the selected 30-day data", async ({ page }) => {
  await bootIsolated(page);
  // Park every 7-day patterns request from the moment the user picks it on.
  let hold7 = false;
  const api = await installApi(page, {
    hold: (path, search) => hold7 && path === "/analytics/patterns" && search.get("days") === "7",
  });
  await page.goto("/analytics");
  const combo = page.getByRole("combobox", { name: "时间范围" });
  await expect(combo).toBeVisible({ timeout: 10000 });
  // The default range is 14 days, so that is what is on screen before the race.
  await expect(page.getByText("RANGE_14", { exact: true })).toBeVisible();

  hold7 = true;
  await combo.selectOption("近7天");
  await page.waitForTimeout(200);
  await combo.selectOption("近30天");
  await expect(page.getByText("RANGE_30", { exact: true })).toBeVisible({ timeout: 10000 });

  // Release the parked 7-day response *after* the 30-day one landed.
  expect(await api.release()).toBeGreaterThan(0);
  await page.waitForTimeout(700);

  await expect(combo, "the selector keeps the user's choice").toHaveValue("近30天");
  await expect(
    page.getByText("RANGE_7", { exact: true }),
    "the stale response must not surface as the current range",
  ).toHaveCount(0);
  await expect(page.getByText("RANGE_30", { exact: true })).toBeVisible();
  // The refresh must not be reported as still loading, nor as an error.
  await expect(page.locator(".f-box .spinner")).toHaveCount(0);
  await expect(page.locator(".error-box")).toHaveCount(0);
});

test("F3 analytics: profile has the same latest-request guarantee", async ({ page }) => {
  await bootIsolated(page);
  let hold7 = false;
  const api = await installApi(page, {
    override: (path, search) =>
      path === "/analytics/profile"
        ? {
            json: {
              peak_focus_hours: [],
              top_apps: [],
              // Distinct value per range so a stale overwrite is visible.
              avg_focus_block_min: Number(search.get("days") ?? 14),
              profile_date: TODAY,
              details: {},
            },
          }
        : undefined,
    hold: (path, search) => hold7 && path === "/analytics/profile" && search.get("days") === "7",
  });
  await page.goto("/analytics");
  const combo = page.getByRole("combobox", { name: "时间范围" });
  const avgBlock = page
    .locator(".third-box-item")
    .filter({ hasText: "平均专注块" })
    .locator(".third-item-content");

  await expect(combo).toBeVisible({ timeout: 10000 });
  await expect(avgBlock, "initial load uses the default 14-day range").toHaveText("14m");

  hold7 = true;
  await combo.selectOption("近7天");
  await page.waitForTimeout(200);
  await combo.selectOption("近30天");
  await expect(avgBlock, "30-day profile must be displayed").toHaveText("30m", { timeout: 10000 });

  // Release the parked 7-day profile response *after* the 30-day one landed.
  expect(await api.release()).toBeGreaterThan(0);
  await page.waitForTimeout(700);

  await expect(combo).toHaveValue("近30天");
  await expect(avgBlock, "the stale 7-day profile must not overwrite").toHaveText("30m");
  await expect(page.locator(".error-box")).toHaveCount(0);
});

test("F3 analytics: a failed 30-day request does not keep showing the old range as if it succeeded", async ({ page }) => {
  await bootIsolated(page);
  let fail30 = false;
  await installApi(page, {
    override: (path, search) => {
      if (fail30 && path === "/analytics/patterns" && search.get("days") === "30") return { status: 500 };
      return undefined;
    },
  });
  await page.goto("/analytics");
  const combo = page.getByRole("combobox", { name: "时间范围" });
  await expect(combo).toBeVisible({ timeout: 10000 });
  // Whatever loaded successfully before the switch is the "old" range.
  await expect(page.getByText("RANGE_14", { exact: true })).toBeVisible();

  fail30 = true;
  await combo.selectOption("近30天");

  await expect(page.locator(".error-box"), "the failure must be reported").toBeVisible({ timeout: 10000 });
  await expect(
    page.getByText("RANGE_14", { exact: true }),
    "old-range rows must not be presented as the 30-day result",
  ).toHaveCount(0);
  await expect(combo).toHaveValue("近30天");
});

// ═════════════════════════════════════════════════════════════════════════
// F4 — the nav toggle's label and aria-expanded describe the real state
// ═════════════════════════════════════════════════════════════════════════

async function toggleState(page: Page) {
  const button = page.locator(".mf-header-toggle");
  return {
    label: (await button.getAttribute("aria-label")) ?? "",
    expanded: (await button.getAttribute("aria-expanded")) ?? "",
    navVisible: await page.evaluate(() => {
      const nav = document.querySelector(".mf-navbar");
      if (!nav) return false;
      const rect = nav.getBoundingClientRect();
      return rect.x >= 0 && rect.width > 0;
    }),
  };
}

test("F4 desktop 1440: expanded nav offers 收起导航 and aria-expanded=true", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.locator(".mf-header-name")).toBeVisible({ timeout: 10000 });

  const state = await toggleState(page);
  expect(state.navVisible).toBeTruthy();
  expect(state.expanded, "nav is currently expanded").toBe("true");
  expect(state.label, "the action offered is to collapse it").toBe("收起导航");
});

test("F4 desktop 1440: collapsed nav offers 展开导航 and aria-expanded=false", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.locator(".mf-header-name")).toBeVisible({ timeout: 10000 });

  await page.locator(".mf-header-toggle").click();
  await page.waitForTimeout(700);

  const state = await toggleState(page);
  expect(state.navVisible, "the sidebar moved off-canvas").toBeFalsy();
  expect(state.expanded).toBe("false");
  expect(state.label).toBe("展开导航");
});

test("F4 mobile 390: closed drawer reports aria-expanded=false", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator(".mf-header-name")).toBeVisible({ timeout: 10000 });

  const state = await toggleState(page);
  expect(state.navVisible, "the drawer starts closed").toBeFalsy();
  expect(state.expanded, "a closed drawer is not expanded").toBe("false");
  expect(state.label).toBe("展开导航");
  // A hidden drawer must not paint a scrim over the content.
  await expect(page.locator(".mf-scrim")).toHaveCount(0);
});

test("F4 mobile 390: opening the drawer reports aria-expanded=true and offers 收起导航", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator(".mf-header-name")).toBeVisible({ timeout: 10000 });

  await page.locator(".mf-header-toggle").click();
  await page.waitForTimeout(700);

  const state = await toggleState(page);
  expect(state.navVisible).toBeTruthy();
  expect(state.expanded).toBe("true");
  expect(state.label).toBe("收起导航");
  await expect(page.locator(".mf-scrim"), "a visible drawer gets a scrim").toBeVisible();

  // Close by clicking the scrim over the *content* area (the sidebar sits
  // above the scrim by design, so its column must not be the click target).
  await page.locator(".mf-scrim").click({ position: { x: 370, y: 500 } });
  await page.waitForTimeout(700);
  const closed = await toggleState(page);
  expect(closed.navVisible).toBeFalsy();
  expect(closed.expanded).toBe("false");
  expect(closed.label).toBe("展开导航");
});

test("F4 cross-breakpoint: collapsing at 1440 then resizing to 390 must not leave a dead scrim", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.locator(".mf-header-name")).toBeVisible({ timeout: 10000 });

  await page.locator(".mf-header-toggle").click();
  await page.waitForTimeout(700);
  expect((await toggleState(page)).navVisible).toBeFalsy();

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(700);

  const state = await toggleState(page);
  expect(state.navVisible, "nav stays hidden after the breakpoint change").toBeFalsy();
  expect(state.expanded).toBe("false");
  expect(state.label).toBe("展开导航");
  await expect(
    page.locator(".mf-scrim"),
    "a scrim over content with no visible nav would block the whole screen",
  ).toHaveCount(0);
});

test("F4 keyboard: a hidden sidebar must not be reachable with the keyboard", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.locator(".mf-header-name")).toBeVisible({ timeout: 10000 });

  const inertWhenHidden = await page.locator(".mf-navbar").evaluate((el) => (el as HTMLElement & { inert?: boolean }).inert === true);
  expect(inertWhenHidden, "the off-canvas sidebar must be inert while hidden").toBe(true);

  // And in practice: Shift+Tab from the toggle must not land inside it.
  await page.locator(".mf-header-toggle").focus();
  await page.keyboard.press("Shift+Tab");
  await page.waitForTimeout(200);
  const insideNav = await page.evaluate(() => Boolean(document.activeElement?.closest(".mf-navbar")));
  expect(insideNav, "focus must not enter the hidden sidebar").toBe(false);

  // Reopening restores keyboard access. Drop the focus the Shift+Tab left on
  // the skip link first — it overlays the header corner while focused, which
  // is its normal behaviour, not part of what this test is about.
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
  await page.locator(".mf-header-toggle").click();
  await page.waitForTimeout(700);
  const inertWhenOpen = await page.locator(".mf-navbar").evaluate((el) => (el as HTMLElement & { inert?: boolean }).inert === true);
  expect(inertWhenOpen, "an open drawer must be interactive").toBe(false);
  await page.keyboard.press("Shift+Tab");
  await page.waitForTimeout(200);
  const insideNavOpen = await page.evaluate(() => Boolean(document.activeElement?.closest(".mf-navbar")));
  expect(insideNavOpen, "an open drawer must be reachable").toBe(true);
});

// Remaining loading, retry and concurrent-save boundaries.
async function choosePreviousDate(page: Page, day: number) {
  await page.getByRole("button", { name: /Choose date|选择日期/ }).click();
  await page.getByRole("gridcell", { name: String(day), exact: true }).click();
}

test("boundary analytics: changing range hides the old profile summary while loading", async ({ page }) => {
  await bootIsolated(page);
  const api = await installApi(page, {
    override: (path, search) => path === "/analytics/profile" ? {
      json: { ...baseResponses["/analytics/profile"] as object, avg_focus_block_min: Number(search.get("days")) },
    } : undefined,
    hold: (path, search) => path === "/analytics/profile" && search.get("days") === "30",
  });
  await page.goto("/analytics");
  const avgBlock = page.locator(".third-box-item").filter({ hasText: "平均专注块" }).locator(".third-item-content");
  await expect(avgBlock).toHaveText("14m");
  await page.getByRole("combobox", { name: "时间范围" }).selectOption("近30天");
  await expect.poll(api.heldCount).toBeGreaterThan(0);
  await expect(avgBlock).not.toHaveText("14m");
  await api.release();
  await expect(avgBlock).toHaveText("30m");
});

test("boundary analytics: refreshing range does not dismiss a baseline failure", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: path => path === "/analytics/baseline" ? { status: 500 } : undefined,
  });
  await page.goto("/analytics");
  await expect(page.locator(".error-box")).toBeVisible();
  await page.getByRole("combobox", { name: "时间范围" }).selectOption("近30天");
  await expect(page.getByText("RANGE_30", { exact: true })).toBeVisible();
  await expect(page.locator(".error-box")).toBeVisible();
});

test("boundary analytics: mobile headings do not create horizontal scrolling", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/analytics");
  await expect(page.locator(".high-change")).toContainText("RANGE_14");
  const dimensions = await page.locator("#main-content").evaluate(main => ({
    width: main.clientWidth,
    scrollWidth: main.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.width);
});

for (const width of [390, 1440]) {
  test(`boundary analytics: attribution action stays inside its result panel at ${width}px`, async ({ page }) => {
    await bootIsolated(page);
    await installApi(page);
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/analytics");
    await expect(page.locator(".high-change")).toContainText("RANGE_14");
    const bounds = await page.locator(".analytics .left-box").evaluate(panel => {
      const box = panel.getBoundingClientRect();
      const action = panel.querySelector(".attribution-action")!.getBoundingClientRect();
      return { top: box.top, right: box.right, bottom: box.bottom, left: box.left,
        actionTop: action.top, actionRight: action.right, actionBottom: action.bottom, actionLeft: action.left };
    });
    expect(bounds.actionTop).toBeGreaterThanOrEqual(bounds.top);
    expect(bounds.actionRight).toBeLessThanOrEqual(bounds.right);
    expect(bounds.actionBottom).toBeLessThanOrEqual(bounds.bottom);
    expect(bounds.actionLeft).toBeGreaterThanOrEqual(bounds.left);
  });
}

for (const path of ["/analytics/patterns", "/analytics/profile"]) {
  test(`boundary analytics: stale ${path} failure cannot clear newer loading`, async ({ page }) => {
    await bootIsolated(page);
    const api = await installApi(page, {
      override: (p, search) => p === path && search.get("days") === "7" ? { status: 500 } : undefined,
      hold: (p, search) => p === path && ["7", "30"].includes(search.get("days") ?? ""),
    });
    await page.goto("/analytics");
    await expect(page.getByText("RANGE_14", { exact: true })).toBeVisible();
    const combo = page.getByRole("combobox", { name: "时间范围" });
    await combo.selectOption("近7天");
    await expect.poll(api.heldCount).toBeGreaterThan(0);
    await combo.selectOption("近30天");
    await expect.poll(api.heldCount).toBeGreaterThan(1);
    await api.release();
    await expect(combo).toHaveValue("近30天");
    await expect(page.locator(".error-box")).toHaveCount(0);
  });
}

test("boundary focus: loading and failed sessions never display stale or measured-zero KPIs", async ({ page }) => {
  await bootIsolated(page);
  const api = await installApi(page, {
    override: (path, search) => path === "/focus"
      ? search.get("date") === TODAY
        ? { json: { date: TODAY, session_count: 2, sessions: MIXED_SESSIONS } }
        : { status: 500 }
      : undefined,
    hold: (path, search) => path === "/focus" && search.get("date") === "2026-09-29",
  });
  await page.goto("/focus");
  await expect(await focusStat(page, "总专注时长")).toHaveText("10m");
  await choosePreviousDate(page, 29);
  await expect.poll(api.heldCount).toBeGreaterThan(0);
  await expect(await focusStat(page, "总专注时长")).toHaveText("—");
  await api.release();
  await expect(page.locator(".error-box")).toBeVisible();
  await expect(await focusStat(page, "总专注时长")).toHaveText("—");
  await expect(await focusStat(page, "专注次数")).toHaveText("—");
});

for (const outcome of ["success", "failure"]) {
  test(`boundary focus: stale trend ${outcome} cannot overwrite the latest refresh`, async ({ page }) => {
    await bootIsolated(page);
    await installApi(page);
    let pending: Route | undefined;
    let calls = 0;
    await page.route("**/api/v1/focus/trend?*", async route => {
      calls++;
      if (calls === 2) { pending = route; return; }
      await route.fulfill({ json: trendBody([{ date: TODAY, focus_min: calls === 1 ? 77 : 200, distraction_min: 0, session_count: 1, avg_score: 80 }]) });
    });
    await page.goto("/focus");
    await expect(page.locator(".zhuanzhu-number")).toHaveText("77");
    await choosePreviousDate(page, 29);
    await expect.poll(() => Boolean(pending)).toBe(true);
    await choosePreviousDate(page, 28);
    await expect(page.locator(".zhuanzhu-number")).toHaveText("200");
    const response = page.waitForResponse(r => r.url().includes("/focus/trend"));
    await pending!.fulfill(outcome === "success"
      ? { json: DEFAULT_TREND }
      : { status: 500, json: { detail: "stale trend failure" } });
    await (await response).finished();
    await expect(page.locator(".zhuanzhu-number")).toHaveText("200");
    await expect(page.locator(".f-statics")).not.toContainText("趋势加载失败");
  });
}

test("boundary focus: concurrent feedback saves retain each row's pending state", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: path => path === "/focus" ? { json: { date: TODAY, session_count: 2, sessions: MIXED_SESSIONS } } : undefined,
  });
  const pending: Route[] = [];
  await page.route("**/api/v1/focus/*/feedback", route => { pending.push(route); });
  await page.goto("/focus");
  const rows = page.locator(".conversation-item");
  await expect(rows).toHaveCount(2);
  await rows.first().getByRole("button", { name: "保存反馈", exact: true }).click();
  await rows.last().getByRole("button", { name: "保存反馈", exact: true }).click();
  await expect.poll(() => pending.length).toBe(2);
  await expect(rows.first().locator(".save-buttom")).toBeDisabled();
  await expect(rows.last().locator(".save-buttom")).toBeDisabled();
  await pending[0].fulfill({ json: { status: "ok" } });
  await expect(rows.first().locator(".f-saved")).toBeVisible();
  await expect(rows.last().locator(".save-buttom")).toBeDisabled();
  await pending[1].fulfill({ json: { status: "ok" } });
  await expect(rows.last().locator(".f-saved")).toBeVisible();
  await expect(rows.first().locator(".self-accession-table")).toBeVisible();
  await rows.first().getByRole("combobox", { name: "这次状态" }).selectOption("focus");
  await rows.first().getByRole("button", { name: "保存反馈", exact: true }).click();
  await expect.poll(() => pending.length).toBe(3);
  await pending[2].fulfill({ json: { status: "ok" } });
  await expect(rows.first().locator(".f-saved")).toContainText("专注");
});

test("boundary focus: saving another row does not dismiss a feedback failure", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: path => path === "/focus" ? { json: { date: TODAY, session_count: 2, sessions: MIXED_SESSIONS } } : undefined,
  });
  const pending: Route[] = [];
  await page.route("**/api/v1/focus/*/feedback", route => { pending.push(route); });
  await page.goto("/focus");
  const rows = page.locator(".conversation-item");
  await expect(rows).toHaveCount(2);
  await rows.first().getByRole("button", { name: "保存反馈", exact: true }).click();
  await expect.poll(() => pending.length).toBe(1);
  await pending[0].fulfill({ status: 500, json: { detail: "FIRST_ROW_FAILED" } });
  await expect(page.getByRole("alert")).toContainText("FIRST_ROW_FAILED");
  await rows.last().getByRole("button", { name: "保存反馈", exact: true }).click();
  await expect.poll(() => pending.length).toBe(2);
  await expect(rows.first().getByRole("alert")).toContainText("FIRST_ROW_FAILED");
  await pending[1].fulfill({ json: { status: "ok" } });
  await expect(rows.last().locator(".f-saved")).toBeVisible();
  await expect(rows.first().getByRole("alert")).toContainText("FIRST_ROW_FAILED");
});

test("boundary focus: zero focus and distraction minutes have zero-height bars", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: path => path === "/focus/trend"
      ? { json: trendBody([{ date: TODAY, focus_min: 0, distraction_min: 0, session_count: 0, avg_score: null }]) }
      : undefined,
  });
  await page.goto("/focus");
  await expect(page.locator(".statics-chart-item")).toHaveCount(1);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.waitForTimeout(100);
  const heights = await page.locator(".chartbox-fenxin, .chartbox-zhuanzhu").evaluateAll(bars =>
    bars.map(bar => bar.getBoundingClientRect().height));
  expect(heights).toEqual([0, 0]);
});

function dailyReportBody(date: string, app: string) {
  return {
    id: "audit-report", user_id: 1, switch_frequency: 2,
    date, data_state: "ready", total_focus_min: 120, total_distraction_min: 30,
    total_focus_minutes: 120, total_sessions: 5, total_distractions: 3, focus_score: 85,
    hourly_distribution: Object.fromEntries(Array.from({ length: 24 }, (_, hour) => [String(hour), hour === 9 ? 120 : 0])),
    top_apps: [{ app, minutes: 120 }], pattern_summary: app,
  };
}

test("boundary reports: a pending date cannot show the previous report's metrics or apps", async ({ page }) => {
  await bootIsolated(page);
  const api = await installApi(page, {
    override: (path, search) => path === "/reports/daily"
      ? { json: dailyReportBody(search.get("date") ?? TODAY, search.get("date") === TODAY ? "OLD_APP" : "NEW_APP") }
      : undefined,
    hold: (path, search) => path === "/reports/daily" && search.get("date") === "2026-09-29",
  });
  await page.goto("/reports");
  await expect(page.locator(".report-detail-list")).toContainText("OLD_APP");
  await choosePreviousDate(page, 29);
  await expect.poll(api.heldCount).toBeGreaterThan(0);
  await expect(page.locator(".report-detail-list")).not.toContainText("OLD_APP");
  await expect(page.locator(".chart-box-item")).toHaveCount(0);
  await expect(page.locator(".t-state")).not.toContainText("2h");
  await expect(page.locator(".t-state .state-item-number")).toHaveText(["—", "—", "—", "—"]);
  await api.release();
  await expect(page.locator(".report-detail-list")).toContainText("NEW_APP");
});

test("boundary reports: a zero-minute bucket has no positive bar height", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page, {
    override: path => path === "/reports/daily" ? { json: dailyReportBody(TODAY, "APP") } : undefined,
  });
  await page.goto("/reports");
  await expect(page.locator(".chart-box-item")).toHaveCount(8);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.waitForTimeout(100);
  const heights = await page.locator(".chart-box-item").evaluateAll(items => items
    .filter(item => item.querySelector(".chart-box-item-number")?.textContent === "0m")
    .map(item => item.querySelector(".chart-box-item-column")!.getBoundingClientRect().height));
  expect(heights).toHaveLength(7);
  expect(heights.every(height => height === 0)).toBe(true);
});

test("boundary navigation: Escape closes the mobile drawer and restores its toggle focus", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.locator(".mf-header-toggle").click();
  await page.locator(".mf-navbar a").first().focus();
  await page.keyboard.press("Escape");
  await expect(page.locator(".mf-header-toggle")).toHaveAttribute("aria-expanded", "false");
  await expect(page.locator(".mf-header-toggle")).toBeFocused();
  await expect(page.locator(".mf-scrim")).toHaveCount(0);
});

test("boundary navigation: an open mobile drawer does not reopen after leaving its breakpoint", async ({ page }) => {
  await bootIsolated(page);
  await installApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.locator(".mf-header-toggle").click();
  await expect(page.locator(".mf-header-toggle")).toHaveAttribute("aria-expanded", "true");
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(page.locator(".mf-navbar")).not.toHaveClass(/mf-navbar--open/);
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator(".mf-header-toggle")).toHaveAttribute("aria-expanded", "false");
  await expect(page.locator(".mf-scrim")).toHaveCount(0);
});

for (const route of ["/focus", "/reports"]) {
  for (const inputValue of ["2026/02/31", "2026/10/01"]) {
    test(`boundary date input: ${route} does not request invalid or future ${inputValue}`, async ({ page }) => {
      await bootIsolated(page);
      await installApi(page, {
        override: (path, search) => path === "/reports/daily"
          ? { json: dailyReportBody(search.get("date") ?? TODAY, "APP") }
          : undefined,
      });
      const requests: string[] = [];
      page.on("request", request => {
        const url = new URL(request.url());
        if (["/api/v1/focus", "/api/v1/reports/daily"].includes(url.pathname)) {
          requests.push(url.searchParams.get("date") ?? "");
        }
      });
      await page.goto(route);
      await expect.poll(() => requests.length).toBeGreaterThan(0);
      const picker = page.locator(".mf-picker input").first();
      await picker.fill(inputValue);
      await picker.press("Enter");
      await page.waitForTimeout(400);
      expect(requests.every(date => date === TODAY), requests.join(", ")).toBe(true);
    });
  }
}
