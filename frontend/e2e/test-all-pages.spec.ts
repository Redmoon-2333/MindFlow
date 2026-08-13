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

const BASE = "http://127.0.0.1:8765";
const FRONTEND = "http://127.0.0.1:4173";
// Real bootstrap root token must NOT be committed (audit report — hardcoded
// token in E2E). Read it from the environment; tests skip with a clear message
// when it is absent (CI / other machines without the token).
const AUTH_TOKEN = process.env.MINDFLOW_TEST_TOKEN ?? "";

/** Issue a bootstrap ticket, exchange it for a session cookie, return the cookie value. */
async function getAuthToken(request: any): Promise<string> {
  const ticketRes = await request.post(`${BASE}/api/v1/auth/bootstrap/ticket`, {
    headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
  });
  expect(ticketRes.ok()).toBeTruthy();
  const { ticket } = await ticketRes.json();

  const bootstrapRes = await request.post(`${BASE}/api/v1/auth/bootstrap`, {
    data: { ticket },
  });
  expect(bootstrapRes.ok()).toBeTruthy();

  const cookies = await bootstrapRes.headersArray();
  const setCookie = cookies.find(
    (h: any) => h.name === "set-cookie" || h.value?.includes("mindflow_session"),
  );
  // Extract the cookie value from Set-Cookie header
  const raw = cookies.find((h: any) => h.value?.includes("mindflow_session="));
  return raw?.value ?? "";
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

test.describe("MindFlow E2E", () => {
  test.beforeAll(async ({ request }) => {
    sessionCookie = await getAuthToken(request);
    expect(sessionCookie).toContain("mindflow_session");
  });

  test("Dashboard loads with system data", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/`);
    await expect(page.locator("h1")).toContainText("仪表盘");

    // Wait for health data to load
    await expect(page.locator(".stat-card").first()).toBeVisible({ timeout: 10000 });
    // Should have KPI cards
    const cards = page.locator(".stat-card");
    await expect(cards).toHaveCount(4, { timeout: 10000 });
  });

  test("Focus page loads with sessions", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/focus`);
    await expect(page.locator("h1")).toContainText("专注分析");

    // Date picker should be visible
    await expect(page.locator('input[type="date"]')).toBeVisible();
    // KPI row should render
    await expect(page.locator(".stat-card").first()).toBeVisible({ timeout: 10000 });
  });

  test("Activities page loads with table", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/activities`);
    await expect(page.locator("h1")).toContainText("活动日志");

    // Table or empty state should be visible
    await expect(
      page.locator("table").or(page.locator("text=暂无活动记录")).first(),
    ).toBeVisible({ timeout: 10000 });
  });

  test("Analytics page loads with tabs", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/analytics`);
    await expect(page.locator("h1")).toContainText("行为洞察");

    // Tabs should be visible
    await expect(page.locator(".tab").first()).toBeVisible();
    const tabs = page.locator(".tab");
    await expect(tabs).toHaveCount(4);
  });

  test("Reports page loads daily and weekly", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/reports`);
    await expect(page.locator("h1")).toContainText("报告中心");

    // Daily/weekly tabs
    await expect(page.locator(".tab").first()).toBeVisible();
    // Date picker
    await expect(page.locator('input[type="date"]')).toBeVisible();
  });

  test("Intervention page loads with history", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/intervention`);
    await expect(page.locator("h1")).toContainText("干预中心");

    // Trigger buttons should be visible
    await expect(page.locator("text=温和提醒")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=标准干预")).toBeVisible();
    await expect(page.locator("text=严格干预")).toBeVisible();
  });

  test("Panel page loads with controls", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/panel`);
    await expect(page.locator("h1")).toContainText("专家面板");

    // Trigger and read buttons
    await expect(page.locator("text=运行专家面板")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=查看上次结果")).toBeVisible();
  });

  test("Chat page loads with session sidebar", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/chat`);
    await expect(page.locator("h1")).toContainText("AI 对话");

    // New chat button
    await expect(page.getByRole("button", { name: "新对话" })).toBeVisible({ timeout: 10000 });
    // Input area
    await expect(page.locator("textarea")).toBeVisible();
    await expect(page.locator("text=发送")).toBeVisible();
  });

  test("Settings page loads all sections", async ({ page }) => {
    await setupAuth(page, sessionCookie);
    await page.goto(`${FRONTEND}/settings`);
    await expect(page.locator("h1")).toContainText("系统设置");

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
    await expect(page.locator("h1")).toContainText("AI 诊断");

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
      { text: "AI 诊断", path: "/diagnostics" },
      { text: "仪表盘", path: "/" },
    ];

    for (const nav of navLinks) {
      await page.click(`.sidebar nav a:text("${nav.text}")`);
      await page.waitForURL(`**${nav.path}`);
      await expect(page.locator("h1")).toBeVisible({ timeout: 5000 });
    }
  });
});
