/**
 * Self-contained E2E tests for Model Center page and NotFound route.
 *
 * No real backend dependency — all API responses are intercepted.
 * Uses config baseURL for relative page.goto paths.
 * Run: npx playwright test e2e/test-model-center.spec.ts --reporter=list
 */

import { test, expect, type Page } from "@playwright/test";

// ── Shared page setup ──

async function setupPage(page: Page) {
  await page.addInitScript(() => {
    localStorage.setItem("mindflow_authenticated", "1");
  });
}

// ── Mock data helpers — real backend field names ──

const REAL_GATES: Array<{
  key: string; label: string; passed: boolean; status: string;
  actual: string; threshold: string; message: string; blocker_code: string;
}> = [
  {
    key: "minimum_days", label: "最少反馈天数",
    passed: true, status: "passed", actual: "28", threshold: ">= 1",
    message: "反馈天数满足最低要求", blocker_code: "",
  },
  {
    key: "minimum_explicit_feedback", label: "最少显式反馈数",
    passed: true, status: "passed", actual: "45", threshold: ">= 20",
    message: "显式反馈数量满足最低要求", blocker_code: "",
  },
  {
    key: "minimum_class_feedback", label: "最少类别反馈数",
    passed: true, status: "passed", actual: "专注=25, 分心=20",
    threshold: "专注 >= 5 且 分心 >= 5",
    message: "类别反馈数量满足最低要求", blocker_code: "",
  },
  {
    key: "balanced_accuracy", label: "平衡准确率",
    passed: false, status: "not_evaluated", actual: "-", threshold: ">= 0.50",
    message: "尚未运行训练评估，无法确定平衡准确率", blocker_code: "metric_not_evaluated",
  },
  {
    key: "minority_f1", label: "少数类 F1",
    passed: false, status: "not_evaluated", actual: "-", threshold: ">= 0.30",
    message: "尚未运行训练评估，无法确定少数类 F1", blocker_code: "metric_not_evaluated",
  },
  {
    key: "calibration_better_than_rule", label: "校准优于规则引擎",
    passed: false, status: "not_implemented", actual: "-",
    threshold: "训练报告提供证据",
    message: "校准比较需训练报告提供真实证据，当前硬编码为通过，不可作为绿色通行",
    blocker_code: "not_implemented",
  },
  {
    key: "stable_date_folds", label: "日期折叠稳定性",
    passed: false, status: "not_implemented", actual: "-",
    threshold: "训练报告提供证据",
    message: "日期折叠稳定性需训练报告提供真实证据，当前硬编码为通过，不可作为绿色通行",
    blocker_code: "not_implemented",
  },
];

function readinessResponse(overrides?: Record<string, unknown>) {
  return {
    raw_events: {
      total_events: 42_000, coverage_days: 28,
      oldest_timestamp: "2026-07-01T00:00:00",
      newest_timestamp: "2026-07-29T23:59:59",
    },
    v2_windows: {
      total: 320, schema_version: 2, date_range_days: 28, eligible_count: 15,
      matched_focus_count: 10, matched_distract_count: 5,
      newest_window_start: "2026-07-29T12:00:00",
    },
    feedback_labels: { focus: 45, distract: 22, mixed: 8, total: 75 },
    trainable: true,
    trainable_window_count: 15,
    trainable_class_count: 2,
    evaluable: true,
    evaluable_explicit_count: 22,
    evaluable_date_count: 7,
    baseline_ready: true,
    current_mode: "rule_engine_only",
    gates: REAL_GATES,
    blockers: [],
    current_training_job: null,
    ...overrides,
  };
}

function blockedReadiness() {
  return {
    ...readinessResponse(),
    trainable: false,
    trainable_window_count: 3,
    trainable_class_count: 1,
    evaluable: false,
    evaluable_explicit_count: 2,
    evaluable_date_count: 1,
    baseline_ready: false,
    gates: REAL_GATES.map((g) => {
      if (g.key === "minimum_days") {
        return {
          ...g, passed: false, status: "failed", actual: "2",
          message: "反馈天数不足，至少需要连续使用并标记 1 天",
          blocker_code: "insufficient_days",
        };
      }
      if (g.key === "minimum_explicit_feedback") {
        return {
          ...g, passed: false, status: "failed", actual: "5",
          message: "反馈数量不足", blocker_code: "insufficient_feedback",
        };
      }
      if (g.key === "minimum_class_feedback") {
        return {
          ...g, passed: false, status: "failed", actual: "专注=3, 分心=2",
          message: "类别反馈不足", blocker_code: "insufficient_class_feedback",
        };
      }
      return g;
    }),
    blockers: [
      { code: "insufficient_days", message: "反馈天数不足" },
      { code: "insufficient_feedback", message: "反馈数量不足" },
    ],
  };
}

function createJobResponse() {
  return { job_id: "job-e2e-001", status: "pending" as const };
}

function jobResponse(status: string, jobId = "job-e2e-001") {
  return {
    job_id: jobId,
    status,
    source: "db",
    model_mode: "rule_engine_only",
    started_at: "2026-07-29T10:00:00",
    completed_at: status === "succeeded" ? "2026-07-29T10:30:00" : null,
    activated: status === "succeeded",
    version_tag: status === "succeeded" ? "v2.1.0" : null,
    feature_schema_version: 2,
    quality_gate: null,
    evaluation: null,
    error: null,
  };
}

function modelStatusResponse() {
  return {
    loaded: true, ready: true, mode: "rule_engine_only",
    v2_mode: "rule_engine_only", feature_schema_version: 2,
    version: null, available_versions: [], reasons: ["v2_models_not_loaded"],
    message: "V2 ML models not available, running with rule engine only",
    model_name: null, last_updated: null,
  };
}

async function interceptAllApi(page: Page) {
  await page.route("**/api/v1/analytics/training-readiness", async (route) => {
    await route.fulfill({ json: readinessResponse(), status: 200 });
  });
  await page.route("**/api/v1/analytics/baseline", async (route) => {
    await route.fulfill({ status: 404 });
  });
  await page.route("**/api/v1/analytics/model-status", async (route) => {
    await route.fulfill({ json: modelStatusResponse(), status: 200 });
  });
  // Realtime/WS — abort to avoid hanging
  await page.route("**/ws/**", (route) => route.abort());
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test.describe("Model Center", () => {
  test("route renders with title and 4 tabs", async ({ page }) => {
    await interceptAllApi(page);
    await setupPage(page);
    await page.goto("/model-center", { waitUntil: "networkidle" });
    await expect(page.locator("h1")).toContainText("模型中心");

    const tabs = page.locator('[role="tab"]');
    await expect(tabs).toHaveCount(4);
    await expect(tabs.nth(0)).toContainText("数据准备");
    await expect(tabs.nth(1)).toContainText("个人基线");
    await expect(tabs.nth(2)).toContainText("模型训练");
    await expect(tabs.nth(3)).toContainText("模型状态");
  });

  test("all 7 quality gates visible with correct status badges", async ({ page }) => {
    await interceptAllApi(page);
    await setupPage(page);
    await page.goto("/model-center", { waitUntil: "networkidle" });

    await expect(page.locator("text=质量门禁（7 项检查）")).toBeVisible({ timeout: 10_000 });

    const gates = page.locator(".mc-gate");
    await expect(gates).toHaveCount(7);

    // First 3 are passed (green badges)
    await expect(gates.nth(0).locator(".badge-success")).toBeVisible();
    await expect(gates.nth(1).locator(".badge-success")).toBeVisible();
    await expect(gates.nth(2).locator(".badge-success")).toBeVisible();

    // Balanced accuracy + minority F1 = not_evaluated (info/blue)
    await expect(gates.nth(3).locator(".badge-info")).toContainText("未评估");
    await expect(gates.nth(4).locator(".badge-info")).toContainText("未评估");

    // Calibration + stable folds = not_implemented (warning/yellow)
    await expect(gates.nth(5).locator(".badge-warning")).toContainText("未实现");
    await expect(gates.nth(6).locator(".badge-warning")).toContainText("未实现");
  });

  test("blocker state shows blockers and disables training", async ({ page }) => {
    await page.route("**/api/v1/analytics/training-readiness", async (route) => {
      await route.fulfill({ json: blockedReadiness(), status: 200 });
    });
    await page.route("**/api/v1/analytics/baseline", async (route) => {
      await route.fulfill({ status: 404 });
    });
    await page.route("**/api/v1/analytics/model-status", async (route) => {
      await route.fulfill({ json: modelStatusResponse(), status: 200 });
    });
    await page.route("**/ws/**", (route) => route.abort());

    await setupPage(page);
    await page.goto("/model-center", { waitUntil: "networkidle" });

    await expect(page.locator("text=阻塞项（2）")).toBeVisible({ timeout: 10_000 });

    // Switch to training tab
    await page.locator('[role="tab"]').nth(2).click();
    // Wait for training tab content to render
    await expect(page.locator("button").filter({ hasText: "开始训练" })).toBeVisible({ timeout: 5000 });

    const startBtn = page.locator("button").filter({ hasText: "开始训练" });
    await expect(startBtn).toBeDisabled();
  });

  test("trainable state allows starting a job", async ({ page }) => {
    await interceptAllApi(page);
    await page.route("**/api/v1/analytics/training-jobs", async (route) => {
      await route.fulfill({ json: createJobResponse(), status: 202 });
    });
    await page.route("**/api/v1/analytics/training-jobs/job-e2e-001", async (route) => {
      await route.fulfill({ json: jobResponse("pending"), status: 200 });
    });

    await setupPage(page);
    await page.goto("/model-center", { waitUntil: "networkidle" });

    // Switch to training tab
    await page.locator('[role="tab"]').nth(2).click();
    await expect(page.locator("button").filter({ hasText: "开始训练" })).toBeVisible({ timeout: 5000 });

    const startBtn = page.locator("button").filter({ hasText: "开始训练" });
    await expect(startBtn).toBeEnabled();
    await startBtn.click();

    // Job ID should appear after starting
    await expect(page.locator("text=job-e2e-001")).toBeVisible({ timeout: 5000 });
  });

  test("self-healing: readiness includes pending job, polling reaches terminal", async ({ page }) => {
    let pollCount = 0;
    const AUTO_JOB_ID = "job-e2e-auto";

    await page.route("**/api/v1/analytics/training-readiness", async (route) => {
      await route.fulfill({
        json: readinessResponse({
          current_training_job: {
            job_id: AUTO_JOB_ID, status: "pending",
            started_at: "2026-07-29T10:00:00", completed_at: null,
          },
        }),
        status: 200,
      });
    });
    await page.route("**/api/v1/analytics/baseline", async (route) => {
      await route.fulfill({ status: 404 });
    });
    await page.route("**/api/v1/analytics/model-status", async (route) => {
      await route.fulfill({ json: modelStatusResponse(), status: 200 });
    });
    // Parameterized route: match the auto job id
    await page.route(`**/api/v1/analytics/training-jobs/${AUTO_JOB_ID}`, async (route) => {
      pollCount++;
      const status = pollCount >= 3 ? "succeeded" : "pending";
      await route.fulfill({ json: jobResponse(status, AUTO_JOB_ID), status: 200 });
    });
    await page.route("**/ws/**", (route) => route.abort());

    await setupPage(page);
    await page.goto("/model-center", { waitUntil: "networkidle" });

    // Switch to training tab
    await page.locator('[role="tab"]').nth(2).click();
    await expect(page.locator("button").filter({ hasText: "开始训练" })).toBeVisible({ timeout: 5000 });

    // The job id from readiness should appear without clicking start
    await expect(page.locator(`text=${AUTO_JOB_ID}`)).toBeVisible({ timeout: 5000 });

    // Wait for polling to reach terminal
    await expect(page.locator("text=已完成")).toBeVisible({ timeout: 15_000 });
  });

  test("cancel button visible for pending job", async ({ page }) => {
    await interceptAllApi(page);
    await page.route("**/api/v1/analytics/training-jobs", async (route) => {
      await route.fulfill({ json: createJobResponse(), status: 202 });
    });
    await page.route("**/api/v1/analytics/training-jobs/job-e2e-001", async (route) => {
      await route.fulfill({ json: jobResponse("pending"), status: 200 });
    });
    await page.route("**/api/v1/analytics/training-jobs/job-e2e-001/cancel", async (route) => {
      await route.fulfill({ json: jobResponse("cancelled"), status: 200 });
    });

    await setupPage(page);
    await page.goto("/model-center", { waitUntil: "networkidle" });

    // Switch to training tab and start job
    await page.locator('[role="tab"]').nth(2).click();
    await expect(page.locator("button").filter({ hasText: "开始训练" })).toBeVisible({ timeout: 5000 });

    await page.locator("button").filter({ hasText: "开始训练" }).click();
    await expect(page.locator("text=job-e2e-001")).toBeVisible({ timeout: 5000 });

    // Cancel button should appear
    const cancelBtn = page.locator("button").filter({ hasText: "取消任务" });
    await expect(cancelBtn).toBeVisible({ timeout: 5000 });
    await cancelBtn.click();
    // Verify the badge shows "已取消"
    await expect(page.locator(".badge-info").filter({ hasText: "已取消" })).toBeVisible({ timeout: 5000 });
  });

  test("polling shows aria-live region", async ({ page }) => {
    await interceptAllApi(page);
    await page.route("**/api/v1/analytics/training-jobs", async (route) => {
      await route.fulfill({ json: createJobResponse(), status: 202 });
    });
    await page.route("**/api/v1/analytics/training-jobs/job-e2e-001", async (route) => {
      await route.fulfill({ json: jobResponse("pending"), status: 200 });
    });

    await setupPage(page);
    await page.goto("/model-center", { waitUntil: "networkidle" });

    // Switch to training tab, start job
    await page.locator('[role="tab"]').nth(2).click();
    await expect(page.locator("button").filter({ hasText: "开始训练" })).toBeVisible({ timeout: 5000 });

    await page.locator("button").filter({ hasText: "开始训练" }).click();

    // aria-live region should appear
    await expect(page.locator('[aria-live="polite"]')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('[aria-live="polite"]')).toContainText("正在监控训练进度");
  });

  test("invalid URL renders not-found page", async ({ page }) => {
    await setupPage(page);
    await page.goto("/this-does-not-exist", { waitUntil: "networkidle" });

    await expect(page.locator(".nf-title")).toContainText("404");
    await expect(page.locator(".nf-desc")).toContainText("页面未找到");
    await expect(page.locator("a.nf-link")).toBeVisible();
  });

  test("tab bar has visible focus-visible outline on keyboard focus", async ({ page }) => {
    await interceptAllApi(page);
    await setupPage(page);
    await page.goto("/model-center", { waitUntil: "networkidle" });

    const firstTab = page.locator('[role="tab"]').first();
    await firstTab.focus();

    // After focus via script, the computed outline style should be non-none
    const outline = await firstTab.evaluate((el) => {
      const style = getComputedStyle(el);
      return { color: style.outlineColor, width: style.outlineWidth, style: style.outlineStyle };
    });
    expect(outline.style).not.toBe("none");
    expect(outline.width).not.toBe("0px");
  });

  test("no horizontal overflow at 375px", async ({ page }) => {
    await interceptAllApi(page);
    await setupPage(page);

    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto("/model-center", { waitUntil: "networkidle" });

    await expect(page.locator("text=质量门禁（7 项检查）")).toBeVisible({ timeout: 10_000 });

    const bodyWidth = await page.evaluate(() => document.body.scrollWidth);
    const viewportWidth = await page.evaluate(() => window.innerWidth);
    expect(bodyWidth).toBeLessThanOrEqual(viewportWidth + 1);
  });
});
