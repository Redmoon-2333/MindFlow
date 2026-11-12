import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { chromium, expect } from "@playwright/test";

const dataDir = path.resolve(process.argv[2] ?? "");
assert.ok(dataDir.startsWith(path.resolve(os.tmpdir()) + path.sep));
const base = "http://127.0.0.1:8876";
const output = path.join(path.dirname(dataDir), "extension-live");
const browser = await chromium.launch({ channel: "chromium", headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
try {
  const token = (await readFile(path.join(dataDir, "token"), "utf8")).trim();
  const issued = await context.request.post(`${base}/api/v1/auth/bootstrap/ticket`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  assert.ok(issued.ok());
  const { ticket } = await issued.json();
  const page = await context.newPage();
  await page.goto(`${base}/#bootstrap=${encodeURIComponent(ticket)}`);
  await expect(page.getByRole("heading", { name: "此刻", exact: true })).toBeVisible();
  await expect(page.locator(".hero-source")).toHaveText("基础模型 · 待个人验证");
  await page.screenshot({ path: path.join(output, "dashboard-desktop.png"), fullPage: true });

  const timings = [];
  for (let index = 0; index < 20; index += 1) {
    const started = performance.now();
    const response = await context.request.get(`${base}/api/v1/health/live`);
    assert.ok(response.ok());
    timings.push(performance.now() - started);
  }
  timings.sort((a, b) => a - b);
  const started = performance.now();
  const panel = await context.request.post(`${base}/api/v1/panel/today`, {
    data: {}, timeout: 15_000,
  });
  assert.equal(panel.status(), 404, "an empty isolated database has no panel evidence");
  const panelBody = await panel.json();
  assert.match(panelBody.detail, /暂无活动数据/);
  const results = {
    dashboard_model_source: "PASS",
    health_requests: timings.length,
    health_p50_ms: Math.round(timings[9]),
    health_p95_ms: Math.round(timings[18]),
    empty_panel_status: panel.status(),
    empty_panel_duration_ms: Math.round(performance.now() - started),
    empty_panel_reports_missing_data: true,
  };
  await writeFile(path.join(output, "runtime-results.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results, null, 2));
} finally {
  await browser.close();
}
