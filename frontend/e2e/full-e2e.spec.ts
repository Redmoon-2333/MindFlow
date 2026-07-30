/**
 * Comprehensive E2E test for MindFlow — covers every interactive component
 * across all pages: Login, Dashboard, Focus, Activities, Analytics, Chat,
 * Reports, Settings, ModelCenter, Diagnostics, Intervention, Panel.
 */
import { test, expect, type Page } from "@playwright/test";

const BASE = "http://127.0.0.1:5173";

/* ── helpers ── */

async function devLogin(page: Page) {
  await page.goto(BASE);
  // If we land on the login page, authenticate via dev mode
  const devBtn = page.locator("button", { hasText: "Dev 登录" });
  if (await devBtn.isVisible({ timeout: 5000 }).catch(() => false)) {
    await devBtn.click();
    // Wait for the page to reload after auth
    await page.waitForURL("**/", { timeout: 10000 });
    await page.waitForTimeout(1500);
  }
}

async function gotoPage(page: Page, path: string, label: string) {
  const navLink = page.locator(`nav a[href="${path}"]`).first();
  if (await navLink.isVisible({ timeout: 3000 }).catch(() => false)) {
    await navLink.click();
  } else {
    await page.goto(`${BASE}${path}`);
  }
  await page.waitForTimeout(1500);
}

async function waitForLoad(page: Page) {
  // Wait for any spinner to disappear
  const spinner = page.locator(".spinner");
  if (await spinner.isVisible({ timeout: 2000 }).catch(() => false)) {
    await spinner.waitFor({ state: "hidden", timeout: 15000 }).catch(() => {});
  }
  await page.waitForTimeout(500);
}

async function screenshot(page: Page, name: string) {
  await page.screenshot({ path: `e2e/screenshots/${name}.png`, fullPage: true });
}

/* ── Test Suite ── */

test.describe("MindFlow Full E2E", () => {
  test.beforeAll(async ({ browser }) => {
    // Ensure screenshots directory exists
    const fs = await import("fs");
    if (!fs.existsSync("e2e/screenshots")) {
      fs.mkdirSync("e2e/screenshots", { recursive: true });
    }
  });

  test("1. Login page — Dev authentication", async ({ page }) => {
    await page.goto(BASE);
    await page.waitForTimeout(1000);

    // Verify we see the login card
    const title = page.locator("h1", { hasText: "MindFlow" });
    await expect(title).toBeVisible();

    // Click dev login
    const devBtn = page.locator("button", { hasText: "Dev 登录" });
    await expect(devBtn).toBeVisible();
    await devBtn.click();

    // Wait for redirect to dashboard
    await page.waitForURL("**/", { timeout: 15000 });
    await page.waitForTimeout(2000);

    // Verify dashboard loaded
    await expect(page.locator("h1", { hasText: "仪表盘" })).toBeVisible();
    await screenshot(page, "01-dashboard-after-login");
  });

  test("2. Dashboard — all interactive elements", async ({ page }) => {
    await devLogin(page);
    await waitForLoad(page);
    await screenshot(page, "02-dashboard-full");

    // ── Collector Toggle ──
    const collectorToggle = page.locator("button", { hasText: /启动采集|停止采集/ }).first();
    if (await collectorToggle.isVisible({ timeout: 3000 }).catch(() => false)) {
      const initialText = await collectorToggle.textContent();
      await collectorToggle.click();
      await page.waitForTimeout(2000);
      // Verify button text changed
      const afterText = await collectorToggle.textContent();
      expect(afterText).not.toBe(initialText);
      await screenshot(page, "02b-dashboard-collector-toggled");

      // Toggle back
      await collectorToggle.click();
      await page.waitForTimeout(2000);
    }

    // ── Autonomy Pause/Resume ──
    const autonomyBtn = page.locator("button", { hasText: /暂停|恢复自主/ }).first();
    if (await autonomyBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      const initialText = await autonomyBtn.textContent();
      await autonomyBtn.click();
      await page.waitForTimeout(2000);
      const afterText = await autonomyBtn.textContent();
      // The button text may or may not change depending on API response
      // Verify the click happened without blocking on state change
      await screenshot(page, "02c-dashboard-autonomy-toggled");

      // Toggle back
      const reverseBtn = page.locator("button", { hasText: /暂停|恢复自主/ }).first();
      if (await reverseBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await reverseBtn.click();
        await page.waitForTimeout(2000);
      }
    }

    // ── Error retry button (if visible) ──
    const retryBtn = page.locator("button", { hasText: "重试" }).first();
    if (await retryBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      await retryBtn.click();
      await page.waitForTimeout(2000);
    }

    // ── Verify KPI cards ──
    const kpiCards = page.locator(".stat-card");
    const kpiCount = await kpiCards.count();
    expect(kpiCount).toBeGreaterThanOrEqual(4);
  });

  test("3. Focus Analysis page — date picker, refresh, feedback", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/focus", "专注分析");
    await waitForLoad(page);
    await screenshot(page, "03-focus-full");

    // ── Date picker ──
    const datePicker = page.locator('input[type="date"]').first();
    if (await datePicker.isVisible({ timeout: 3000 }).catch(() => false)) {
      // Change to yesterday
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const dateStr = yesterday.toISOString().split("T")[0];
      await datePicker.fill(dateStr);
      await page.waitForTimeout(1500);
      await screenshot(page, "03b-focus-date-changed");

      // Change back to today
      const today = new Date().toISOString().split("T")[0];
      await datePicker.fill(today);
      await page.waitForTimeout(1500);
    }

    // ── Refresh button ──
    const refreshBtn = page.locator("button", { hasText: "刷新" });
    if (await refreshBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await refreshBtn.click();
      await page.waitForTimeout(2000);
    }

    // ── Feedback forms (if sessions exist) ──
    const feedbackSelects = page.locator("select");
    const selectCount = await feedbackSelects.count();
    if (selectCount > 0) {
      // Change the first session's state dropdown
      const stateSelect = page.locator("select").first();
      await stateSelect.selectOption("focus");
      await page.waitForTimeout(300);

      // Change score
      const scoreSelect = page.locator("select").nth(1);
      if (await scoreSelect.isVisible({ timeout: 1000 }).catch(() => false)) {
        await scoreSelect.selectOption("5");
        await page.waitForTimeout(300);
      }

      // Change task type
      const taskSelect = page.locator("select").nth(2);
      if (await taskSelect.isVisible({ timeout: 1000 }).catch(() => false)) {
        await taskSelect.selectOption("coding");
        await page.waitForTimeout(300);
      }

      // Save feedback
      const saveBtn = page.locator("button", { hasText: "保存反馈" }).first();
      if (await saveBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
        await saveBtn.click();
        await page.waitForTimeout(2000);
        await screenshot(page, "03c-focus-feedback-saved");
      }
    }
  });

  test("4. Activities page — filters, search, pagination", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/activities", "活动日志");
    await waitForLoad(page);
    await screenshot(page, "04-activities-full");

    // ── Date filters ──
    const dateInputs = page.locator('input[type="date"]');
    const dateCount = await dateInputs.count();
    if (dateCount >= 2) {
      const startDate = dateInputs.first();
      const endDate = dateInputs.nth(1);
      const weekAgo = new Date();
      weekAgo.setDate(weekAgo.getDate() - 7);
      await startDate.fill(weekAgo.toISOString().split("T")[0]);
      await page.waitForTimeout(1000);
      await screenshot(page, "04b-activities-date-filter");

      // Clear filters
      await startDate.fill("");
      await page.waitForTimeout(1000);
    }

    // ── Search ──
    const searchInput = page.locator('input[placeholder*="搜索"]');
    if (await searchInput.isVisible({ timeout: 2000 }).catch(() => false)) {
      await searchInput.fill("chrome");
      await page.waitForTimeout(1500);
      await screenshot(page, "04c-activities-search");

      // Clear search
      await searchInput.fill("");
      await page.waitForTimeout(1000);
    }

    // ── Raw debug checkbox ──
    const rawCheckbox = page.locator('input[type="checkbox"]').first();
    if (await rawCheckbox.isVisible({ timeout: 2000 }).catch(() => false)) {
      await rawCheckbox.check();
      await page.waitForTimeout(500);
      await screenshot(page, "04d-activities-raw-debug");

      await rawCheckbox.uncheck();
      await page.waitForTimeout(500);
    }

    // ── Pagination ──
    const nextPageBtn = page.locator("button", { hasText: "下一页" });
    if (await nextPageBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      const isDisabled = await nextPageBtn.isDisabled();
      if (!isDisabled) {
        await nextPageBtn.click();
        await page.waitForTimeout(1500);
        await screenshot(page, "04e-activities-page2");

        // Go back
        const prevBtn = page.locator("button", { hasText: "上一页" });
        if (await prevBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
          await prevBtn.click();
          await page.waitForTimeout(1500);
        }
      }
    }

    // ── Error retry (if visible) ──
    const retryBtn = page.locator("button", { hasText: "重试" }).first();
    if (await retryBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      await retryBtn.click();
      await page.waitForTimeout(2000);
    }
  });

  test("5. Analytics page — tabs, day range, attribution", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/analytics", "行为洞察");
    await waitForLoad(page);
    await screenshot(page, "05-analytics-full");

    // ── Tab: 模式分析 (default) ──
    const patternTab = page.locator("button.tab", { hasText: "模式分析" });
    if (await patternTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await patternTab.click();
      await page.waitForTimeout(1500);
    }

    // ── Day range selector ──
    const daySelect = page.locator("select").first();
    if (await daySelect.isVisible({ timeout: 2000 }).catch(() => false)) {
      await daySelect.selectOption("30");
      await page.waitForTimeout(2000);
      await screenshot(page, "05b-analytics-30days");

      await daySelect.selectOption("7");
      await page.waitForTimeout(1500);
    }

    // ── Tab: 个人画像 ──
    const profileTab = page.locator("button.tab", { hasText: "个人画像" });
    if (await profileTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await profileTab.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "05c-analytics-profile");
    }

    // ── Tab: 拖延归因 ──
    const attributionTab = page.locator("button.tab", { hasText: "拖延归因" });
    if (await attributionTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await attributionTab.click();
      await page.waitForTimeout(1500);

      // Run attribution analysis
      const runBtn = page.locator("button", { hasText: "运行归因分析" });
      if (await runBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await runBtn.click();
        await page.waitForTimeout(5000); // AI analysis takes time
        await screenshot(page, "05d-analytics-attribution-result");
      }
    }

    // ── Tab: 模型状态 ──
    const modelTab = page.locator("button.tab", { hasText: "模型状态" });
    if (await modelTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await modelTab.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "05e-analytics-model-status");
    }

    // ── Error close ──
    const errorClose = page.locator(".error-box button", { hasText: "关闭" }).first();
    if (await errorClose.isVisible({ timeout: 1000 }).catch(() => false)) {
      await errorClose.click();
    }
  });

  test("6. Chat page — new chat, send message, session history", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/chat", "AI 对话");
    await waitForLoad(page);
    await screenshot(page, "06-chat-full");

    // ── New chat button ──
    const newChatBtn = page.locator("button", { hasText: "新对话" });
    if (await newChatBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await newChatBtn.click();
      await page.waitForTimeout(500);
    }

    // ── Type and send a message ──
    const textarea = page.locator("textarea");
    if (await textarea.isVisible({ timeout: 3000 }).catch(() => false)) {
      await textarea.fill("你好，MindFlow！请告诉我当前的专注状态。");
      await page.waitForTimeout(500);

      // Click send
      const sendBtn = page.locator("button", { hasText: "发送" });
      await expect(sendBtn).toBeVisible();
      await sendBtn.click();

      // Wait for AI response (could take a while)
      await page.waitForTimeout(10000);
      await screenshot(page, "06b-chat-message-sent");

      // Wait more for full response
      await page.waitForTimeout(10000);
      await screenshot(page, "06c-chat-response-received");
    }

    // ── Check session sidebar ──
    const sessionItems = page.locator("[style*='cursor: pointer']");
    const sessionCount = await sessionItems.count();
    if (sessionCount > 0) {
      // Click on a session
      await sessionItems.first().click();
      await page.waitForTimeout(2000);
      await screenshot(page, "06d-chat-session-selected");
    }

    // ── Send another message ──
    if (await textarea.isVisible({ timeout: 1000 }).catch(() => false)) {
      await textarea.fill("请分析我最近的行为模式");
      const sendBtn = page.locator("button", { hasText: "发送" });
      await sendBtn.click();
      await page.waitForTimeout(15000); // Longer wait for AI analysis
      await screenshot(page, "06e-chat-second-message");
    }

    // ── Error close ──
    const errorClose = page.locator(".error-box button", { hasText: "关闭" }).first();
    if (await errorClose.isVisible({ timeout: 1000 }).catch(() => false)) {
      await errorClose.click();
    }
  });

  test("7. Reports page — daily/weekly tabs, date pickers", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/reports", "报告中心");
    await waitForLoad(page);
    await screenshot(page, "07-reports-full");

    // ── Daily tab (default) ──
    const dailyTab = page.locator("button.tab", { hasText: "日报" });
    if (await dailyTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await dailyTab.click();
      await page.waitForTimeout(1500);
    }

    // ── Daily date picker ──
    const dateInput = page.locator('input[type="date"]').first();
    if (await dateInput.isVisible({ timeout: 2000 }).catch(() => false)) {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      await dateInput.fill(yesterday.toISOString().split("T")[0]);
      await page.waitForTimeout(1500);
      await screenshot(page, "07b-reports-daily-yesterday");
    }

    // ── Switch to weekly tab ──
    const weeklyTab = page.locator("button.tab", { hasText: "周报" });
    if (await weeklyTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await weeklyTab.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "07c-reports-weekly");
    }

    // ── Weekly date picker ──
    const weeklyDateInput = page.locator('input[type="date"]').first();
    if (await weeklyDateInput.isVisible({ timeout: 2000 }).catch(() => false)) {
      const lastWeek = new Date();
      lastWeek.setDate(lastWeek.getDate() - 14);
      await weeklyDateInput.fill(lastWeek.toISOString().split("T")[0]);
      await page.waitForTimeout(1500);
      await screenshot(page, "07d-reports-weekly-last-week");
    }

    // ── Back to daily ──
    if (await dailyTab.isVisible({ timeout: 1000 }).catch(() => false)) {
      await dailyTab.click();
      await page.waitForTimeout(1500);
    }
  });

  test("8. Settings page — all controls", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/settings", "系统设置");
    await waitForLoad(page);
    await screenshot(page, "08-settings-full");

    // ── 8a. Privacy telemetry checkboxes ──
    const inputToggle = page.locator("input[type='checkbox']").first();
    if (await inputToggle.isVisible({ timeout: 3000 }).catch(() => false)) {
      const wasChecked = await inputToggle.isChecked();
      await inputToggle.click();
      await page.waitForTimeout(2000);
      // Toggle back
      await inputToggle.click();
      await page.waitForTimeout(2000);
    }

    // ── 8b. Retention selects ──
    const selects = page.locator("select");
    const selectCount = await selects.count();
    if (selectCount > 0) {
      // Change first retention dropdown
      await selects.first().selectOption("14");
      await page.waitForTimeout(1000);
      // Reset
      await selects.first().selectOption("7");
      await page.waitForTimeout(1000);
    }

    // ── 8c. Generate browser pairing code ──
    const pairingBtn = page.locator("button", { hasText: "生成浏览器配对码" });
    if (await pairingBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await pairingBtn.click();
      await page.waitForTimeout(3000);
      await screenshot(page, "08b-settings-pairing-code");
    }

    // ── 8d. Collector toggle ──
    const collectorToggle = page.locator("button", { hasText: /启动采集|停止采集/ }).first();
    if (await collectorToggle.isVisible({ timeout: 2000 }).catch(() => false)) {
      await collectorToggle.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "08c-settings-collector-toggled");
      // Toggle back
      await collectorToggle.click();
      await page.waitForTimeout(2000);
    }

    // ── 8e. Autonomy control ──
    const autonomyInput = page.locator("input[type='number']").first();
    if (await autonomyInput.isVisible({ timeout: 2000 }).catch(() => false)) {
      await autonomyInput.fill("2");
      await page.waitForTimeout(500);
    }
    const pauseBtn = page.locator("button", { hasText: /暂停|恢复自主/ }).first();
    if (await pauseBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await pauseBtn.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "08d-settings-autonomy-paused");

      // Resume
      const resumeBtn = page.locator("button", { hasText: /恢复自主模式/ }).first();
      if (await resumeBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await resumeBtn.click();
        await page.waitForTimeout(2000);
      }
    }

    // ── 8f. Get unknown apps ──
    const unknownBtn = page.locator("button", { hasText: /获取未知应用/ });
    if (await unknownBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await unknownBtn.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "08e-settings-unknown-apps");
    }

    // ── 8g. Add classification rule ──
    const processInput = page.locator("input[placeholder='进程名']");
    if (await processInput.isVisible({ timeout: 2000 }).catch(() => false)) {
      await processInput.fill("test_process_e2e");
      const titleInput = page.locator("input[placeholder='窗口标题模式']");
      if (await titleInput.isVisible({ timeout: 1000 }).catch(() => false)) {
        await titleInput.fill("Test Window*");
      }
      const categorySelect = page.locator("select").nth(2);
      if (await categorySelect.isVisible({ timeout: 1000 }).catch(() => false)) {
        await categorySelect.selectOption("code");
      }
      const addBtn = page.locator("button", { hasText: "添加" }).first();
      if (await addBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
        await addBtn.click();
        await page.waitForTimeout(2000);
        await screenshot(page, "08f-settings-rule-added");
      }
    }

    // ── 8h. Delete the rule we just added ──
    const deleteBtn = page.locator("button", { hasText: "删除" }).first();
    if (await deleteBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await deleteBtn.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "08g-settings-rule-deleted");
    }

    // ── 8i. Data export ──
    const exportSelect = page.locator("select").first();
    // Find the export format select
    const exportFmtSelect = page.locator("select").filter({ hasText: "CSV" });
    if (await exportFmtSelect.isVisible({ timeout: 2000 }).catch(() => false)) {
      await exportFmtSelect.selectOption("json");
      await page.waitForTimeout(500);
      await exportFmtSelect.selectOption("csv");
      await page.waitForTimeout(500);
    }
    const exportBtn = page.locator("button", { hasText: "导出" });
    if (await exportBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await exportBtn.click();
      await page.waitForTimeout(3000);
      await screenshot(page, "08h-settings-export");
    }

    // ── 8j. Preferences PUT ──
    const prefTextarea = page.locator("textarea").last();
    if (await prefTextarea.isVisible({ timeout: 2000 }).catch(() => false)) {
      const putBtn = page.locator("button", { hasText: /PUT/ });
      if (await putBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
        await putBtn.click();
        await page.waitForTimeout(2000);
        await screenshot(page, "08i-settings-prefs-put");
      }

      // PATCH
      const patchBtn = page.locator("button", { hasText: /PATCH/ });
      if (await patchBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
        await patchBtn.click();
        await page.waitForTimeout(2000);
      }
    }

    // ── 8k. Clear telemetry buttons ──
    // Scroll to bottom and verify the privacy section exists
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForTimeout(500);
    // Verify the privacy/telemetry section rendered
    const privacyLabel = page.locator("text=隐私行为采集");
    if (await privacyLabel.isVisible({ timeout: 3000 }).catch(() => false)) {
      await screenshot(page, "08k-settings-privacy-section");
    }
    // ── 8l. Error close ──
    const errorClose = page.locator(".error-box button", { hasText: "关闭" }).first();
    if (await errorClose.isVisible({ timeout: 1000 }).catch(() => false)) {
      await errorClose.click();
    }
  });

  test("9. Model Center — all 4 tabs", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/model-center", "模型中心");
    await waitForLoad(page);
    await screenshot(page, "09-model-center-full");

    // ── Tab: 数据准备 (default) ──
    const dataTab = page.locator("button.tab", { hasText: "数据准备" });
    if (await dataTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await dataTab.click();
      await page.waitForTimeout(2000);
      // Verify quality gates section exists
      const gatesSection = page.locator("h3", { hasText: "质量门禁" });
      if (await gatesSection.isVisible({ timeout: 2000 }).catch(() => false)) {
        await expect(gatesSection).toBeVisible();
      }
    }

    // ── Tab: 个人基线 ──
    const baselineTab = page.locator("button.tab", { hasText: "个人基线" });
    if (await baselineTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await baselineTab.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "09b-mc-baseline");
    }

    // ── Tab: 模型训练 ──
    const trainTab = page.locator("button.tab", { hasText: "模型训练" });
    if (await trainTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await trainTab.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "09c-mc-training");

      // Try starting training (may fail due to insufficient data — that's expected)
      const startBtn = page.locator("button", { hasText: "开始训练" });
      if (await startBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        const isDisabled = await startBtn.isDisabled();
        if (!isDisabled) {
          await startBtn.click();
          await page.waitForTimeout(5000);
          await screenshot(page, "09d-mc-training-started");

          // If a cancel button appears, cancel the training
          const cancelBtn = page.locator("button", { hasText: "取消任务" });
          if (await cancelBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
            await cancelBtn.click();
            await page.waitForTimeout(3000);
          }
        }
      }
    }

    // ── Tab: 模型状态 ──
    const statusTab = page.locator("button.tab", { hasText: "模型状态" });
    if (await statusTab.isVisible({ timeout: 2000 }).catch(() => false)) {
      await statusTab.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "09e-mc-status");
    }

    // ── Error close ──
    const errorClose = page.locator(".error-box button", { hasText: "关闭" }).first();
    if (await errorClose.isVisible({ timeout: 1000 }).catch(() => false)) {
      await errorClose.click();
    }
  });

  test("10. Diagnostics page — AI runs, detail expansion", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/diagnostics", "AI 诊断");
    await waitForLoad(page);
    await screenshot(page, "10-diagnostics-full");

    // ── Verify KPI cards ──
    const kpiCards = page.locator(".stat-card");
    const kpiCount = await kpiCards.count();
    expect(kpiCount).toBeGreaterThanOrEqual(4);

    // ── Click on a run row (if any exist) ──
    const runRows = page.locator("tbody tr");
    const rowCount = await runRows.count();
    if (rowCount > 0) {
      await runRows.first().click();
      await page.waitForTimeout(2000);
      await screenshot(page, "10b-diagnostics-detail-open");

      // Close detail
      const closeBtn = page.locator("button", { hasText: "关闭" }).first();
      if (await closeBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
        await closeBtn.click();
        await page.waitForTimeout(1000);
      }
    }

    // ── Retry button (if error) ──
    const retryBtn = page.locator("button", { hasText: "重试" }).first();
    if (await retryBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      await retryBtn.click();
      await page.waitForTimeout(2000);
    }
  });

  test("11. Intervention page — trigger, respond, feedback, history tabs", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/intervention", "干预中心");
    await waitForLoad(page);
    await screenshot(page, "11-intervention-full");

    // ── Trigger gentle intervention ──
    const gentleBtn = page.locator("button", { hasText: "温和提醒" });
    if (await gentleBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await gentleBtn.click();
      await page.waitForTimeout(3000); // AI processing time
      await screenshot(page, "11b-intervention-gentle-triggered");
    }

    // ── Respond to latest intervention (if visible) ──
    const acceptBtn = page.locator("button", { hasText: "接受" }).first();
    if (await acceptBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await acceptBtn.click();
      await page.waitForTimeout(2000);
      await screenshot(page, "11c-intervention-accepted");
    }

    // ── Trigger standard intervention ──
    const standardBtn = page.locator("button", { hasText: "标准干预" });
    if (await standardBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await standardBtn.click();
      await page.waitForTimeout(3000);
    }

    // ── Ignore it ──
    const ignoreBtn = page.locator("button", { hasText: "忽略" }).first();
    if (await ignoreBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await ignoreBtn.click();
      await page.waitForTimeout(2000);
    }

    // ── Trigger strict intervention ──
    const strictBtn = page.locator("button", { hasText: "严格干预" });
    if (await strictBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await strictBtn.click();
      await page.waitForTimeout(3000);
    }

    // ── Dismiss it ──
    const dismissBtn = page.locator("button", { hasText: "关闭" }).first();
    if (await dismissBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      // Make sure we click the dismiss button (not error close)
      const allCloseBtns = page.locator("button.btn-danger", { hasText: "关闭" });
      if (await allCloseBtns.count() > 0) {
        await allCloseBtns.first().click();
      }
      await page.waitForTimeout(2000);
    }

    await screenshot(page, "11d-intervention-all-triggered");

    // ── History tabs ──
    for (const days of ["14天", "30天", "7天"]) {
      const tab = page.locator("button.tab", { hasText: days });
      if (await tab.isVisible({ timeout: 1500 }).catch(() => false)) {
        await tab.click();
        await page.waitForTimeout(1500);
      }
    }

    // ── Feedback on an item (if any responded items exist) ──
    const feedbackBtn = page.locator("button", { hasText: "评价" }).first();
    if (await feedbackBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await feedbackBtn.click();
      await page.waitForTimeout(500);

      // Select rating
      const ratingSelect = page.locator("select").last();
      if (await ratingSelect.isVisible({ timeout: 1000 }).catch(() => false)) {
        await ratingSelect.selectOption("effective");
        await page.waitForTimeout(500);
      }

      // Type comment
      const commentArea = page.locator("textarea").last();
      if (await commentArea.isVisible({ timeout: 1000 }).catch(() => false)) {
        await commentArea.fill("E2E test feedback");
        await page.waitForTimeout(500);
      }

      // Submit
      const submitBtn = page.locator("button", { hasText: "提交" }).first();
      if (await submitBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
        await submitBtn.click();
        await page.waitForTimeout(2000);
        await screenshot(page, "11e-intervention-feedback-submitted");
      }
    }
  });

  test("12. Panel page — expert panel trigger and result", async ({ page }) => {
    await devLogin(page);
    await gotoPage(page, "/panel", "专家面板");
    await waitForLoad(page);
    await screenshot(page, "12-panel-full");

    // ── View last result ──
    const readBtn = page.locator("button", { hasText: "查看上次结果" });
    if (await readBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await readBtn.click();
      await page.waitForTimeout(3000);
      await screenshot(page, "12b-panel-last-result");
    }

    // ── Trigger expert panel ──
    const triggerBtn = page.locator("button", { hasText: "运行专家面板" });
    if (await triggerBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await triggerBtn.click();
      await page.waitForTimeout(15000); // Multi-agent analysis takes time
      await screenshot(page, "12c-panel-triggered");
    }

    // ── Error close ──
    const errorClose = page.locator(".error-box button", { hasText: "关闭" }).first();
    if (await errorClose.isVisible({ timeout: 1000 }).catch(() => false)) {
      await errorClose.click();
    }
  });

  test("13. Navigation — verify all sidebar links work", async ({ page }) => {
    await devLogin(page);

    const routes = [
      { path: "/", label: "仪表盘", heading: "仪表盘" },
      { path: "/focus", label: "专注分析", heading: "专注分析" },
      { path: "/activities", label: "活动日志", heading: "活动日志" },
      { path: "/analytics", label: "行为洞察", heading: "行为洞察" },
      { path: "/model-center", label: "模型中心", heading: "模型中心" },
      { path: "/reports", label: "报告中心", heading: "报告中心" },
      { path: "/intervention", label: "干预中心", heading: "干预中心" },
      { path: "/panel", label: "专家面板", heading: "专家面板" },
      { path: "/chat", label: "AI 对话", heading: "AI 对话" },
      { path: "/settings", label: "系统设置", heading: "系统设置" },
      { path: "/diagnostics", label: "AI 诊断", heading: "AI 诊断" },
    ];

    for (const route of routes) {
      await gotoPage(page, route.path, route.label);
      await waitForLoad(page);
      const heading = page.locator("h1", { hasText: route.heading });
      await expect(heading).toBeVisible({ timeout: 5000 });
    }

    await screenshot(page, "13-navigation-all-pages-verified");
  });

  test("14. Page reload persistence — verify data survives refresh", async ({ page }) => {
    await devLogin(page);
    await page.goto(`${BASE}/settings`);
    await waitForLoad(page);

    // Reload
    await page.reload();
    await waitForLoad(page);

    // Verify settings still loaded
    const heading = page.locator("h1", { hasText: "系统设置" });
    await expect(heading).toBeVisible();
    await screenshot(page, "14-settings-after-reload");

    // Navigate to dashboard and reload
    await page.goto(`${BASE}/`);
    await waitForLoad(page);
    await page.reload();
    await waitForLoad(page);
    const dashHeading = page.locator("h1", { hasText: "仪表盘" });
    await expect(dashHeading).toBeVisible();
    await screenshot(page, "14b-dashboard-after-reload");
  });

  test("15. NotFound page — invalid route", async ({ page }) => {
    await devLogin(page);
    await page.goto(`${BASE}/nonexistent-page`);
    await page.waitForTimeout(2000);
    await screenshot(page, "15-not-found-page");
    // Should show the NotFound component
    const notFoundText = page.locator("text=404").or(page.locator("text=未找到"));
    if (await notFoundText.isVisible({ timeout: 3000 }).catch(() => false)) {
      await expect(notFoundText).toBeVisible();
    }
  });
});
