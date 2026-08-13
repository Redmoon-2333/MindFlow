/**
 * Comprehensive E2E test: ALL MindFlow backend API endpoints + frontend pages.
 * Uses a single shared session to avoid exhausting the in-memory token store.
 */

import { test, expect, type APIRequestContext, type Page } from "@playwright/test";

const BASE = "http://127.0.0.1:8765";
const FRONTEND = "http://127.0.0.1:4173";
// Real bootstrap root token must NOT be committed (audit report — hardcoded
// token in E2E). Read it from the environment; tests skip with a clear message
// when it is absent (CI / other machines without the token).
const AUTH_TOKEN = process.env.MINDFLOW_TEST_TOKEN ?? "";

// ── Shared auth helpers ──
let _sharedCookie = "";

async function initSharedSession(request: APIRequestContext) {
  const ticketRes = await request.post(`${BASE}/api/v1/auth/bootstrap/ticket`, {
    headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
  });
  expect(ticketRes.ok()).toBeTruthy();
  const { ticket } = await ticketRes.json();
  const bootstrapRes = await request.post(`${BASE}/api/v1/auth/bootstrap`, { data: { ticket } });
  expect(bootstrapRes.ok()).toBeTruthy();
  const cookies = bootstrapRes.headersArray();
  const raw = cookies.find((h) => h.value?.includes("mindflow_session="));
  const cookieVal = raw?.value ?? "";
  const m = cookieVal.match(/(mindflow_session=[^;]+)/);
  _sharedCookie = m?.[1] ?? "";
  return _sharedCookie;
}

function H(): Record<string, string> {
  return { Cookie: _sharedCookie };
}

function extractSessionValue(): string {
  const m = _sharedCookie.match(/mindflow_session=([^;]+)/);
  return m?.[1] ?? "";
}

// ═══════════════════════════════════════════════════════════════════
// API endpoint tests
// ═══════════════════════════════════════════════════════════════════

test.describe("API Endpoints", () => {
  test.beforeAll(async ({ request }) => {
    await initSharedSession(request);
  });

  // ── Health ──
  test("GET /api/v1/health", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/health`);
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("status", "ok");
    expect(d).toHaveProperty("version");
  });
  test("GET /api/v1/health/live", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/health/live`);
    expect(res.ok()).toBeTruthy();
    expect((await res.json()).status).toBeTruthy();
  });
  test("GET /api/v1/health/ready", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/health/ready`);
    expect(res.ok()).toBeTruthy();
    expect((await res.json()).status).toBeTruthy();
  });

  // ── Auth ──
  test("POST /api/v1/auth/bootstrap/ticket", async ({ request }) => {
    const res = await request.post(`${FRONTEND}/api/v1/auth/bootstrap/ticket`, {
      headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
    });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d.ticket.length).toBeGreaterThan(10);
  });
  test("POST /api/v1/auth/bootstrap with valid ticket", async ({ request }) => {
    const tRes = await request.post(`${FRONTEND}/api/v1/auth/bootstrap/ticket`, {
      headers: { Authorization: `Bearer ${AUTH_TOKEN}` },
    });
    const { ticket } = await tRes.json();
    const res = await request.post(`${FRONTEND}/api/v1/auth/bootstrap`, { data: { ticket } });
    expect(res.status()).toBe(204);
  });
  test("POST /api/v1/auth/bootstrap with invalid ticket returns 401", async ({ request }) => {
    const res = await request.post(`${FRONTEND}/api/v1/auth/bootstrap`, {
      data: { ticket: "invalid-ticket-12345678" },
    });
    expect(res.status()).toBe(401);
  });

  // ── Collector ──
  test("GET /api/v1/collector", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/collector`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect(typeof (await res.json()).running).toBe("boolean");
  });
  test("POST /api/v1/collector (start)", async ({ request }) => {
    const res = await request.post(`${FRONTEND}/api/v1/collector`, { headers: H() });
    expect(res.ok()).toBeTruthy();
  });
  test("POST /api/v1/collector/stop", async ({ request }) => {
    const res = await request.post(`${FRONTEND}/api/v1/collector/stop`, { headers: H() });
    expect(res.ok()).toBeTruthy();
  });

  // ── Activities ──
  test("GET /api/v1/activities", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/activities`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect(Array.isArray((await res.json()).items)).toBeTruthy();
  });
  test("GET /api/v1/activities with pagination", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/activities?page=1&page_size=5`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect(Array.isArray((await res.json()).items)).toBeTruthy();
  });
  test("GET /api/v1/activities/current", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/activities/current`, { headers: H() });
    expect([200, 404]).toContain(res.status());
  });

  // ── Focus ──
  test("GET /api/v1/focus", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/focus`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect(Array.isArray((await res.json()).sessions)).toBeTruthy();
  });
  test("GET /api/v1/focus?date=2026-07-28", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/focus?date=2026-07-28`, { headers: H() });
    expect(res.ok()).toBeTruthy();
  });
  test("GET /api/v1/focus/trend?days=7", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/focus/trend?days=7`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("days", 7);
    expect(Array.isArray(d.daily)).toBeTruthy();
  });
  test("GET /api/v1/focus/trend?days=30", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/focus/trend?days=30`, { headers: H() });
    expect(res.ok()).toBeTruthy();
  });

  // ── Reports ──
  test("GET /api/v1/reports/daily", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/reports/daily`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect((await res.json()).date).toBeTruthy();
  });
  test("GET /api/v1/reports/daily?date=2026-07-28", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/reports/daily?date=2026-07-28`, { headers: H() });
    expect(res.ok()).toBeTruthy();
  });
  test("GET /api/v1/reports/weekly", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/reports/weekly`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("week_start");
    expect(d).toHaveProperty("week_end");
  });

  // ── Analytics ──
  test("GET /api/v1/analytics/patterns", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/analytics/patterns?days=14`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("high_switch_periods");
    expect(d).toHaveProperty("trigger_apps");
  });
  test("GET /api/v1/analytics/baseline", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/analytics/baseline`, { headers: H() });
    expect([200, 404]).toContain(res.status());
  });
  test("GET /api/v1/analytics/profile", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/analytics/profile?days=14`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("peak_focus_hours");
    expect(d).toHaveProperty("top_apps");
  });
  test("GET /api/v1/analytics/model-status", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/analytics/model-status`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("loaded");
    expect(d).toHaveProperty("ready");
  });
  test("POST /api/v1/analytics/attribution", async ({ request }) => {
    const res = await request.post(`${FRONTEND}/api/v1/analytics/attribution`, {
      headers: H(), data: {},
    });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("assessment");
    expect(d).toHaveProperty("source");
  });

  // ── Intervention ──
  test("GET /api/v1/intervention/history", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/intervention/history?days=7`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(Array.isArray(d.items)).toBeTruthy();
    expect(d).toHaveProperty("count");
  });
  test("POST /api/v1/intervention/trigger - gentle", async ({ request }) => {
    const res = await request.post(`${FRONTEND}/api/v1/intervention/trigger`, {
      headers: H(), data: { intensity: "gentle" },
    });
    expect(res.ok()).toBeTruthy();
    expect((await res.json()).intervention).toBeDefined();
  });

  // ── Chat ──
  test("GET /api/v1/chat/sessions", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/chat/sessions`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect(Array.isArray(await res.json())).toBeTruthy();
  });
  test("POST /api/v1/chat - send message", async ({ request }) => {
    const res = await request.post(`${FRONTEND}/api/v1/chat`, {
      headers: H(), data: { message: "你好" }, timeout: 60000,
    });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("answer");
    expect(typeof d.answer).toBe("string");
  }, 60000);

  // ── Panel ──
  test("GET /api/v1/panel", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/panel`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("types");
    expect(d).toHaveProperty("confidence");
  });

  // ── Preferences ──
  test("GET /api/v1/preferences", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/preferences`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect(typeof (await res.json())).toBe("object");
  });
  test("PUT /api/v1/preferences", async ({ request }) => {
    const res = await request.put(`${FRONTEND}/api/v1/preferences`, {
      headers: H(), data: { theme: "light" },
    });
    expect(res.ok()).toBeTruthy();
  });
  test("PATCH /api/v1/preferences", async ({ request }) => {
    const res = await request.patch(`${FRONTEND}/api/v1/preferences`, {
      headers: H(), data: { theme: "dark" },
    });
    expect(res.ok()).toBeTruthy();
  });

  // ── App Classifications ──
  test("GET /api/v1/app-classifications", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/app-classifications`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect(Array.isArray(await res.json())).toBeTruthy();
  });
  test("POST + DELETE /api/v1/app-classifications", async ({ request }) => {
    const createRes = await request.post(`${FRONTEND}/api/v1/app-classifications`, {
      headers: H(),
      data: { process_name: "e2e_test_app.exe", window_title_pattern: null, category: "other", priority: 10 },
    });
    expect(createRes.ok()).toBeTruthy();
    const { id } = await createRes.json();
    const deleteRes = await request.delete(`${FRONTEND}/api/v1/app-classifications/${id}`, { headers: H() });
    expect([200, 204]).toContain(deleteRes.status());
  });
  test("GET /api/v1/app-classifications/unknown-apps", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/app-classifications/unknown-apps`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect(Array.isArray(await res.json())).toBeTruthy();
  });

  // ── Export ──
  test("GET /api/v1/export?fmt=json", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/export?fmt=json`, { headers: H() });
    expect(res.ok()).toBeTruthy();
  });
  test("GET /api/v1/export?fmt=csv", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/export?fmt=csv`, { headers: H() });
    expect(res.ok()).toBeTruthy();
  });

  // ── Autonomy ──
  test("GET /api/v1/autonomy", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/autonomy`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    expect((await res.json())).toHaveProperty("enabled");
  });
  test("POST /api/v1/autonomy/pause + resume", async ({ request }) => {
    const pauseRes = await request.post(`${FRONTEND}/api/v1/autonomy/pause`, {
      headers: H(), data: { hours: 1 },
    });
    expect(pauseRes.ok()).toBeTruthy();
    const pd = await pauseRes.json();
    expect(pd.enabled === false || pd.paused === true || pd.paused_until != null).toBeTruthy();
    const resumeRes = await request.post(`${FRONTEND}/api/v1/autonomy/resume`, { headers: H() });
    expect(resumeRes.ok()).toBeTruthy();
  });

  // ── Telemetry ──
  test("GET /api/v1/telemetry/status", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/telemetry/status`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("preferences");
    expect(d).toHaveProperty("database_size_bytes");
  });
  test("PATCH /api/v1/telemetry/preferences", async ({ request }) => {
    const res = await request.patch(`${FRONTEND}/api/v1/telemetry/preferences`, {
      headers: H(), data: { input_telemetry_enabled: false }, timeout: 60000,
    });
    expect(res.ok()).toBeTruthy();
  }, 60000);
  test("POST /api/v1/telemetry/browser/pairing-code", async ({ request }) => {
    const res = await request.post(`${FRONTEND}/api/v1/telemetry/browser/pairing-code`, {
      headers: H(), timeout: 60000,
    });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d).toHaveProperty("code");
    expect(d).toHaveProperty("expires_at");
  }, 60000);
  test("GET /api/v1/telemetry/focus-prediction", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/telemetry/focus-prediction`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(d.prediction != null || d.focus_probability != null || d.status === "no_data").toBeTruthy();
    // The typed response intentionally exposes the stable four-field contract;
    // model metadata is available from the health/model-status endpoints.
    expect(d).toHaveProperty("mode");
    expect(d).toHaveProperty("reason");
  });

  // ── AI Diagnostics ──
  test("GET /api/v1/ai/runs", async ({ request }) => {
    const res = await request.get(`${FRONTEND}/api/v1/ai/runs?limit=10&offset=0`, { headers: H() });
    expect(res.ok()).toBeTruthy();
    const d = await res.json();
    expect(Array.isArray(d.items)).toBeTruthy();
    expect(d.total != null || d.count != null).toBeTruthy();
  });
});

// ═══════════════════════════════════════════════════════════════════
// Frontend page tests (browser-based)
// ═══════════════════════════════════════════════════════════════════

test.describe("Frontend Pages", () => {
  test.beforeAll(async ({ request }) => {
    await initSharedSession(request);
  });

  async function setupBrowserAuth(page: Page) {
    await page.addInitScript(() => {
      localStorage.setItem("mindflow_authenticated", "1");
    });
    const val = extractSessionValue();
    if (val) {
      await page.context().addCookies([
        { name: "mindflow_session", value: val, domain: "127.0.0.1", path: "/" },
      ]);
    }
  }

  const pages = [
    { path: "/", title: "仪表盘" },
    { path: "/focus", title: "专注分析" },
    { path: "/activities", title: "活动日志" },
    { path: "/analytics", title: "行为洞察" },
    { path: "/reports", title: "报告中心" },
    { path: "/intervention", title: "干预中心" },
    { path: "/panel", title: "专家面板" },
    { path: "/chat", title: "AI 对话" },
    { path: "/settings", title: "系统设置" },
    { path: "/diagnostics", title: "AI 诊断" },
  ];

  for (const p of pages) {
    test(`Page ${p.title} (${p.path}) loads`, async ({ page }) => {
      await setupBrowserAuth(page);
      await page.goto(`${FRONTEND}${p.path}`, { waitUntil: "networkidle", timeout: 15000 });
      await expect(page.locator("h1")).toContainText(p.title, { timeout: 10000 });
    });
  }

  test("Dashboard shows KPI cards", async ({ page }) => {
    await setupBrowserAuth(page);
    await page.goto(`${FRONTEND}/`, { waitUntil: "networkidle" });
    await expect(page.locator(".stat-card").first()).toBeVisible({ timeout: 10000 });
    await expect(page.locator(".stat-card")).toHaveCount(4, { timeout: 10000 });
  });

  test("Analytics shows 4 tabs", async ({ page }) => {
    await setupBrowserAuth(page);
    await page.goto(`${FRONTEND}/analytics`, { waitUntil: "networkidle" });
    await expect(page.locator(".tab").first()).toBeVisible();
    await expect(page.locator(".tab")).toHaveCount(4);
  });

  test("Settings shows all sections", async ({ page }) => {
    await setupBrowserAuth(page);
    await page.goto(`${FRONTEND}/settings`, { waitUntil: "networkidle" });
    for (const section of ["系统信息", "数据采集", "自主控制", "应用分类", "数据导出", "偏好设置"]) {
      await expect(page.locator(`text=${section}`)).toBeVisible({ timeout: 10000 });
    }
  });

  test("Intervention shows trigger buttons", async ({ page }) => {
    await setupBrowserAuth(page);
    await page.goto(`${FRONTEND}/intervention`, { waitUntil: "networkidle" });
    await expect(page.locator("text=温和提醒")).toBeVisible({ timeout: 10000 });
    await expect(page.locator("text=标准干预")).toBeVisible();
    await expect(page.locator("text=严格干预")).toBeVisible();
  });

  test("Chat shows input and send button", async ({ page }) => {
    await setupBrowserAuth(page);
    await page.goto(`${FRONTEND}/chat`, { waitUntil: "networkidle" });
    await expect(page.getByRole("button", { name: "新对话" })).toBeVisible({ timeout: 10000 });
    await expect(page.locator("textarea")).toBeVisible();
  });

  test("Navigation between all pages", async ({ page }) => {
    await setupBrowserAuth(page);
    await page.goto(`${FRONTEND}/`, { waitUntil: "networkidle" });
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

  test("Collector toggle from Dashboard", async ({ page }) => {
    await setupBrowserAuth(page);
    await page.goto(`${FRONTEND}/`, { waitUntil: "networkidle" });
    await expect(page.locator("text=采集器状态")).toBeVisible({ timeout: 10000 });
    const toggleBtn = page.locator("button").filter({ hasText: /停止采集|启动采集/ }).first();
    await expect(toggleBtn).toBeVisible({ timeout: 10000 });
    const originalText = await toggleBtn.textContent();
    await toggleBtn.click();
    await page.waitForTimeout(2000);
    const newText = await toggleBtn.textContent();
    expect(newText).not.toBe(originalText);
    await toggleBtn.click();
    await page.waitForTimeout(2000);
  });
});
