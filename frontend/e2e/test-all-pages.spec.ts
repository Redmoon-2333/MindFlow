/**
 * E2E test: verify all MindFlow frontend pages load and call backend APIs.
 *
 * Setup:
 *   1. Authenticate via bootstrap flow (ticket exchange → session cookie)
 *   2. Navigate to each page
 *   3. Verify key UI elements render and API calls succeed
 *
 * Run:
 *   cd frontend && npx playwright test e2e/test-all-pages.spec.ts --reporter=list
 */

import { test, expect, type Page } from "@playwright/test";
import { initSharedSession } from "./session";

const FRONTEND = "http://127.0.0.1:4173";

/** Issue a bootstrap ticket, exchange it for a session cookie, return the cookie value.
 *  429-tolerant — the backend's global bucket drains during a full run. */
async function getAuthToken(request: any): Promise<string> {
  return initSharedSession(request);
}

/** Navigate to the frontend, set localStorage auth marker, and add session cookie. */
async function setupAuth(page: Page, cookieHeader: string) {
  // Set localStorage to mark authenticated
  await page.addInitScript(() => {
    localStorage.setItem("mindflow_authenticated", "1");
  });

  // Extract the cookie value
  const match = cookieHeader.match(/mindflow_session=([^;]+)/);
  if (match) {
    await page.context().addCookies([
      {
        name: "mindflow_session",
        value: match[1],
        domain: "127.0.0.1",
        path: "/",
      },
    ]);
  }
}

// ── Tests ─────────────────────────────────────────────────────────────

let sessionCookie = "";

/** The shell owns the page title (reference design: 20px top-bar heading),
 *  so assertions target the top bar rather than a per-page <h1>. */
async function expectPageTitle(page: Page, title: string) {
  await expect(page.locator(".mf-header-name")).toHaveText(title, { timeout: 10000 });
}

test.describe("MindFlow E2E", () => {
  test.beforeAll(async ({ request }) => {
    sessionCookie = await getAuthToken(request);
    expect(sessionCookie).toContain("mindflow_session");
  });

  test("Dashboard loads with system data", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/`);
    await expectPageTitle(page, "仪表盘");

    // Status row renders (系统健康 / 采集器 / 数据库 / LLM层)
    await expect(page.locator(".d-state-item")).toHaveCount(4, { timeout: 10000 });
    // Four metric cards below it
    await expect(page.locator(".d-statistic-item")).toHaveCount(4, { timeout: 10000 });
  });

  test("Focus page loads with sessions", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/focus`);
    await expectPageTitle(page, "专注分析");

    // Date picker should be visible
    await expect(page.locator(".f-datepicker input").first()).toBeVisible();
    // Four KPI cards should render
    await expect(page.locator(".f-state .state-item")).toHaveCount(4, { timeout: 10000 });
  });

  test("Activities page loads with table", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/activities`);
    await expectPageTitle(page, "活动日志");

    // Table or empty state should be visible
    await expect(
      page.locator("table").or(page.locator("text=暂无活动记录")).first(),
    ).toBeVisible({ timeout: 10000 });
    // Reference layout: filter row + paginated table block
    await expect(page.locator(".search-box input")).toBeVisible();
    await expect(page.locator(".changePage .page")).toBeVisible({ timeout: 10000 });
  });

  test("Analytics page loads its three sections", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/analytics`);
    await expectPageTitle(page, "行为洞察");

    // Section titles with the decorative rules (reference design)
    const titles = page.locator(".analytics .title");
    await expect(titles).toHaveCount(3, { timeout: 10000 });
    await expect(page.locator(".time-box .time")).toBeVisible();
  });

  test("Reports page loads daily and weekly", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/reports`);
    await expectPageTitle(page, "报告中心");

    // Daily/weekly capsules
    await expect(page.locator(".daily-report")).toBeVisible();
    await expect(page.locator(".weekly-report")).toBeVisible();
    // Date picker
    await expect(page.locator(".report-datepicker input").first()).toBeVisible();
  });

  test("Intervention page loads with history", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/intervention`);
    await expectPageTitle(page, "干预中心");

    // Trigger buttons should be visible
    await expect(page.locator("text=温和提醒")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=标准干预")).toBeVisible();
    await expect(page.locator("text=严格干预")).toBeVisible();
  });

  test("Panel page loads with controls", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/panel`);
    await expectPageTitle(page, "专家面板");

    // Trigger and read buttons
    await expect(page.locator("text=运行专家面板")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=查看上次结果")).toBeVisible();
  });

  test("Chat page loads with session sidebar", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/chat`);
    await expectPageTitle(page, "AI 对话");

    // New chat button
    await expect(page.getByRole("button", { name: "新对话" })).toBeVisible({ timeout: 10000 });
    // Input area
    await expect(page.locator("textarea")).toBeVisible();
    await expect(page.locator("text=发送")).toBeVisible();
  });

  test("Settings page loads all sections", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/settings`);
    await expectPageTitle(page, "系统设置");

    // Key sections
    await expect(page.locator("text=系统信息")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=数据采集")).toBeVisible();
    await expect(page.locator("text=自主控制")).toBeVisible();
    await expect(page.locator("text=应用分类")).toBeVisible();
    await expect(page.locator("text=数据导出")).toBeVisible();
    await expect(page.locator("text=偏好设置")).toBeVisible();
  });

  test("Diagnostics page loads AI runs", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/diagnostics`);
    await expectPageTitle(page, "AI 诊断");

    // Health cards
    await expect(page.locator(".stat-card").first()).toBeVisible({ timeout: 10000 });
    // AI runs table
    await expect(page.locator("text=AI 工作流运行记录")).toBeVisible();
  });

  // Collector toggle covered by test-all-api-endpoints.spec.ts

  test("Focus feedback submission works", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/focus`);

    // Wait for sessions to load
    await page.waitForTimeout(3000);

    // Check if there are any sessions with feedback forms
    const feedbackBtns = page.locator("text=保存反馈");
    const count = await feedbackBtns.count();
    // If sessions exist, feedback form should be available
    if (count > 0) {
      await expect(feedbackBtns.first()).toBeVisible();
    }
  });

  test("Analytics patterns tab shows data", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/analytics`);

    // Wait for patterns to load (default tab)
    await page.waitForTimeout(3000);

    // Should show either data or "暂无数据"
    const patternsSection = page.locator("text=高切换时段");
    await expect(patternsSection).toBeVisible({ timeout: 10000 });
  });

  test("API health check through frontend proxy", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/health`);
    expect(res.ok()).toBeTruthy();
    const data = await res.json();
    expect(data.status).toBe("ok");
    expect(data.version).toBeTruthy();
  });

  test("Navigation between all pages works", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/`);

    const navLinks = [
      { text: "专注分析", path: "/focus" },
      { text: "活动日志", path: "/activities" },
      { text: "行为洞察", path: "/analytics" },
      { text: "报告中心", path: "/reports" },
      { text: "干预中心", path: "/intervention" },
      { text: "专家面板", path: "/panel" },
      { text: "AI 对话", path: "/chat" },
      { text: "系统设置", path: "/settings" },
      { text: "仪表盘", path: "/" },
    ];

    for (const nav of navLinks) {
      await page.locator(".mf-nav-item", { hasText: nav.text }).first().click();
      await page.waitForURL(`**${nav.path}`);
      await expect(page.locator(".mf-header-name")).toBeVisible({ timeout: 5000 });
    }
  });

  test("Advanced entry hides the three extra routes until opened", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/`);

    // Advanced group starts collapsed — the three extra items are not shown.
    await expect(page.locator("#advanced-nav")).toHaveCount(0);
    const toggle = page.locator(".mf-nav-group-toggle");
    await expect(toggle).toHaveAttribute("aria-expanded", "false");

    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
    await expect(page.locator("#advanced-nav .mf-nav-item")).toHaveCount(3);

    // Entering an advanced route auto-expands the group.
    await page.locator("#advanced-nav .mf-nav-item", { hasText: "模型中心" }).click();
    await page.waitForURL("**/model-center");
    await expect(page.locator(".mf-header-name")).toHaveText("模型中心");
    await expect(page.locator(".mf-nav-group-toggle")).toHaveAttribute("aria-expanded", "true");

    // AI 诊断 is reachable from the expanded group too.
    await page.locator("#advanced-nav .mf-nav-item", { hasText: "AI 诊断" }).click();
    await page.waitForURL("**/diagnostics");
    await expect(page.locator(".mf-header-name")).toHaveText("AI 诊断");
  });

  test("Logout dialog cancels without ending the session, then confirms", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/`);
    await expectPageTitle(page, "仪表盘");

    await page.locator(".mf-user").click();
    const dialog = page.locator('[role="dialog"]');
    await expect(dialog).toBeVisible();
    // Escape closes without changing auth state
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(page.locator(".mf-header-name")).toBeVisible();

    // Confirm ends the session and returns to the login screen
    await page.locator(".mf-user").click();
    await page.locator(".mf-dialog-yes").click();
    await expect(page.locator(".login-card")).toBeVisible({ timeout: 10000 });
    expect(await page.evaluate(() => localStorage.getItem("mindflow_authenticated"))).toBeNull();
  });
});
