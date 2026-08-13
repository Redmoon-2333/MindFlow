/**
 * Self-contained E2E tests for intervention history title/message rendering.
 *
 * No real backend dependency — /api/v1/intervention/history is intercepted.
 * Uses config baseURL for relative page.goto paths.
 * Run: npx playwright test e2e/test-intervention-history.spec.ts --reporter=list
 */

import { test, expect, type Page } from "@playwright/test";

// ── Shared page setup ──

async function setupPage(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem("mindflow_authenticated", "1");
  });
}

// ── Mock helpers ──

function historyResponse(items: Array<Record<string, unknown>>) {
  return { items, count: items.length, has_more: false, next_cursor: null };
}

/** A new-style record with concrete Chinese title/message stored in context_json. */
const NEW_STYLE_ITEM = {
  id: "hist-new-001",
  user_id: 1,
  triggered_at: "2026-07-30T10:00:00Z",
  intervention_type: "environment_optimization",
  cbt_technique: "stimulus_control",
  context_json: {
    intensity: "gentle",
    title: "减少干扰源",
    message: "关闭社交媒体通知，保持桌面整洁有助于提升专注力",
    procrastination_types: ["impulsivity"],
    confidence: { impulsivity: 0.85 },
  },
  user_response: null,
  response_latency_s: null,
  feedback_rating: null,
  feedback_comment: null,
  created_at: "2026-07-30T10:00:00Z",
  title: "减少干扰源",
  message: "关闭社交媒体通知，保持桌面整洁有助于提升专注力",
};

/** A legacy-style record without title/message in context_json (enriched by backend). */
const LEGACY_ITEM = {
  id: "hist-legacy-001",
  user_id: 1,
  triggered_at: "2026-07-29T15:00:00Z",
  intervention_type: "task_breakdown",
  cbt_technique: null,
  context_json: {
    intensity: "standard",
    procrastination_types: ["task_aversion"],
    confidence: { task_aversion: 0.72 },
  },
  user_response: null,
  response_latency_s: null,
  feedback_rating: null,
  feedback_comment: null,
  created_at: "2026-07-29T15:00:00Z",
  title: "来自 MindFlow 的提醒",
  message: "检测到面临的任务较大，可能感到难以着手。建议尝试以下方法：将任务拆解为 3-5 个小步骤，每次完成一个小目标",
};

async function mockInterventionHistory(page: Page) {
  const body = JSON.stringify(historyResponse([NEW_STYLE_ITEM, LEGACY_ITEM]));
  await page.route(/\/api\/v1\/intervention\/history/, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body,
    });
  });
  // Block WebSocket to avoid hanging
  await page.route("**/api/v1/ws**", (route) => route.abort());
  // Fallback: return empty 200 for any other API call not mocked
  await page.route(/\/api\/v1\//, async (route) => {
    if (route.request().url().includes("/intervention/history")) {
      await route.fallback();
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test.describe("Intervention History — title/message rendering", () => {
  test("renders concrete Chinese title and message for new-style record", async ({ page }) => {
    await mockInterventionHistory(page);
    await setupPage(page);
    await page.goto("/intervention", { waitUntil: "domcontentloaded" });

    // New-style record: concrete title "减少干扰源" should be visible
    await expect(page.locator("text=减少干扰源").first()).toBeVisible({ timeout: 10_000 });
    // New-style record: concrete message should be visible
    await expect(page.locator("text=关闭社交媒体通知，保持桌面整洁有助于提升专注力").first()).toBeVisible();
  });

  test("renders enriched fallback title/message for legacy record", async ({ page }) => {
    await mockInterventionHistory(page);
    await setupPage(page);
    await page.goto("/intervention", { waitUntil: "domcontentloaded" });

    // Legacy record: fallback title should be visible
    await expect(page.locator("text=来自 MindFlow 的提醒")).toBeVisible({ timeout: 10_000 });
    // Legacy record: fallback message contains Chinese detail, not a raw enum
    await expect(page.locator("text=将任务拆解为 3-5 个小步骤")).toBeVisible();
  });

  test("raw intervention_type enum is never rendered as visible user-facing content", async ({ page }) => {
    await mockInterventionHistory(page);
    await setupPage(page);
    await page.goto("/intervention", { waitUntil: "domcontentloaded" });

    // The raw enum strings must NOT appear anywhere in the visible page body
    await expect(page.locator("text=environment_optimization")).not.toBeVisible();
    await expect(page.locator("text=task_breakdown")).not.toBeVisible();
    await expect(page.locator("text=smart_prioritization")).not.toBeVisible();
  });

  test("history badge still shows Chinese type label (not raw enum)", async ({ page }) => {
    await mockInterventionHistory(page);
    await setupPage(page);
    await page.goto("/intervention", { waitUntil: "domcontentloaded" });

    // The type badge should show the Chinese label "环境优化", not the raw enum
    await expect(page.locator(".badge").filter({ hasText: "环境优化" }).first()).toBeVisible({ timeout: 10_000 });
    await expect(page.locator(".badge").filter({ hasText: "任务分解" })).toBeVisible();
  });

  test("latest-intervention section also shows concrete title/message", async ({ page }) => {
    // The "latest" is the first item in reversed history (the most recent trigger).
    // Since we reverse in the frontend, the highest triggered_at becomes latest.
    await mockInterventionHistory(page);
    await setupPage(page);
    await page.goto("/intervention", { waitUntil: "domcontentloaded" });

    // The "最新干预" section should show the new-style record's title and message
    const latestSection = page.locator(".card").filter({ hasText: "最新干预" });
    await expect(latestSection.locator("text=减少干扰源")).toBeVisible({ timeout: 10_000 });
    await expect(latestSection.locator("text=关闭社交媒体通知")).toBeVisible();
  });
});
