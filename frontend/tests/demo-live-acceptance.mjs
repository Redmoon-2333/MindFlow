import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium, expect } from "@playwright/test";

const args = process.argv.slice(2);
const option = name => args[args.indexOf(`--${name}`) + 1];
assert.ok(args.includes("--data-dir"), "Supply the isolated demo data directory");
const dataDir = path.resolve(option("data-dir"));
const marker = JSON.parse(await readFile(path.join(dataDir, ".mindflow-demo.json"), "utf8"));
assert.equal(marker.demo_only, true, "Never use the production data directory");
const token = (await readFile(path.join(dataDir, "token"), "utf8")).trim();
const base = args.includes("--base") ? option("base") : "http://127.0.0.1:8870";
assert.ok(["localhost", "127.0.0.1"].includes(new URL(base).hostname));
const out = path.resolve(option("out"));
await mkdir(out, { recursive: true });
const results = [];
const browser = await chromium.launch({ headless: true, channel: "chrome" });
try {
  for (const width of [1440, 390]) {
    const context = await browser.newContext({ viewport: { width, height: 900 } });
    try {
      const ticket = await context.request.post(`${base}/api/v1/auth/bootstrap/ticket`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      assert.equal(ticket.status(), 200);
      const exchange = await context.request.post(`${base}/api/v1/auth/bootstrap`, {
        data: { ticket: (await ticket.json()).ticket },
      });
      assert.equal(exchange.status(), 204);
      await context.addInitScript(() => localStorage.setItem("mindflow_authenticated", "1"));
      const page = await context.newPage();
      const errors = [];
      page.on("pageerror", error => errors.push(error.message));
      page.on("response", response => {
        if (response.url().includes("/api/") && response.status() >= 500) {
          errors.push(`HTTP ${response.status()} ${new URL(response.url()).pathname}`);
        }
      });
      const predictionResponse = page.waitForResponse(response =>
        new URL(response.url()).pathname === "/api/v1/telemetry/focus-prediction");
      await page.goto(base, { waitUntil: "networkidle" });
      const prediction = await (await predictionResponse).json();
      assert.equal(prediction.status, "ready");
      assert.ok(Number.isFinite(prediction.focus_probability));
      assert.ok(prediction.focus_probability >= 0 && prediction.focus_probability <= 1);
      const statusResponse = await context.request.get(`${base}/api/v1/analytics/model-status`);
      assert.equal(statusResponse.status(), 200);
      const status = await statusResponse.json();
      assert.equal(status.demo_only, true);
      assert.equal(status.loaded, true);
      assert.equal(status.ready, false);
      assert.equal(status.mode, "shadow");
      assert.equal(status.version, "demo_v4");
      const display = `${(prediction.focus_probability * 100).toFixed(1)}%`;
      const predictionCard = page.locator(".d-extra-card").filter({ hasText: "ML 专注预测" });
      await expect(predictionCard.locator(".d-extra-value")).toHaveText(display);
      await expect(predictionCard).toContainText("合成数据演示模型 · 影子模式");
      await expect(page.locator(".d-extra")).toContainText("实时连接：connected");
      await page.screenshot({ path: path.join(out, `${width}-dashboard.png`), fullPage: true });
      await page.goto(`${base}/model-center`, { waitUntil: "networkidle" });
      await expect(page.locator(".mc-header")).toContainText("未通过个人模型质量门");
      await page.getByRole("tab", { name: "模型状态", exact: true }).click();
      await expect(page.getByText("演示版本", { exact: true })).toBeVisible();
      await expect(page.getByText("demo_v4", { exact: true }).first()).toBeVisible();
      await expect(page.getByText("激活版本", { exact: true })).toHaveCount(0);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      assert.deepEqual(errors, []);
      await page.screenshot({ path: path.join(out, `${width}-model-center.png`), fullPage: true });
      results.push({ width, passed: true, model: status.version, probability: prediction.focus_probability,
        displayed: display, authenticated: true, realtimeConnected: true, personalReady: false });
      console.log(`PASS ${width}px: ${status.version}, API = UI ${display}, shadow-only`);
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
  await writeFile(path.join(out, "demo-live.json"), JSON.stringify({
    generatedAt: new Date().toISOString(), base, results,
  }, null, 2));
}
