import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, expect } from "@playwright/test";

// This smoke test may revoke pairing, so it only accepts a disposable data dir.
const dataDir = path.resolve(process.argv[2] ?? "");
const tempRoot = path.resolve(os.tmpdir()) + path.sep;
assert.ok(dataDir.startsWith(tempRoot), "Pass an isolated acceptance-server data directory");
const output = path.join(path.dirname(dataDir), "extension-live");
await mkdir(output, { recursive: true });
const extension = fileURLToPath(new URL("../../browser_extension", import.meta.url));
const base = "http://127.0.0.1:8765";
const checks = [];
const errors = [];
let context;
const site = createServer((_request, response) => {
  response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
  response.end("<!doctype html><title>MindFlow synthetic lesson</title><h1>Synthetic lesson</h1>");
});
await new Promise(resolve => site.listen(8897, "127.0.0.1", resolve));

async function check(name, run) {
  const started = performance.now();
  try {
    await run();
  } catch (error) {
    checks.push({ name, passed: false, error: error.message });
    throw error;
  }
  checks.push({ name, passed: true, duration_ms: Math.round(performance.now() - started) });
  console.log(`PASS ${name}`);
}

try {
  context = await chromium.launchPersistentContext(await mkdtemp(path.join(output, "profile-")), {
    channel: "chromium",
    headless: true,
    viewport: { width: 1440, height: 900 },
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`],
  });
  const worker = context.serviceWorkers()[0] ?? await context.waitForEvent("serviceworker");
  const extensionId = new URL(worker.url()).host;
  await check("unpacked_manifest_v3_loaded", async () => {
    assert.ok(worker.url().startsWith(`chrome-extension://${extensionId}/`));
    assert.equal(await worker.evaluate(() => chrome.runtime.getManifest().manifest_version), 3);
  });
  const heartbeats = [];
  context.on("request", request => {
    if (request.url() === `${base}/api/v1/telemetry/browser/heartbeat`) {
      heartbeats.push(request.postDataJSON());
    }
  });

  const app = await context.newPage();
  app.on("pageerror", error => errors.push(error.message));
  const rootToken = (await readFile(path.join(dataDir, "token"), "utf8")).trim();
  const ticketResponse = await context.request.post(`${base}/api/v1/auth/bootstrap/ticket`, {
    headers: { Authorization: `Bearer ${rootToken}` },
  });
  assert.ok(ticketResponse.ok(), "The supplied data dir must own this running backend");
  const { ticket } = await ticketResponse.json();
  await app.goto(`${base}/#bootstrap=${encodeURIComponent(ticket)}`);
  await expect(app.getByRole("link", { name: "AI 对话", exact: true })).toBeVisible();
  // Remove only this test's synthetic domain from the isolated backend.
  await context.request.delete(`${base}/api/v1/interventions/blocklist/localhost`);
  await check("desktop_dashboard_rendered", async () => {
    await expect(app.getByRole("heading", { name: "此刻", exact: true })).toBeVisible();
    await expect(app.locator(".dash--loading")).toHaveCount(0);
    await app.screenshot({ path: path.join(output, "dashboard-desktop.png"), fullPage: true });
    assert.equal(await app.evaluate(() =>
      document.documentElement.scrollWidth > window.innerWidth), false);
  });

  const options = await context.newPage();
  await options.goto(`chrome-extension://${extensionId}/options.html`);
  await check("pair_through_real_extension_options", async () => {
    const response = await context.request.post(`${base}/api/v1/telemetry/browser/pairing-code`);
    assert.ok(response.ok());
    const { code } = await response.json();
    await options.getByLabel("本地后端地址").fill(base);
    await options.getByLabel("八位配对码").fill(code);
    await options.getByRole("button", { name: "配对", exact: true }).click();
    await expect(options.locator("#status")).toContainText("配对成功");
    await options.screenshot({ path: path.join(output, "extension-paired.png") });
  });
  await check("localhost_alias_pairs_through_manifest_authorized_host", async () => {
    const response = await context.request.post(`${base}/api/v1/telemetry/browser/pairing-code`);
    const { code } = await response.json();
    await options.getByLabel("本地后端地址").fill("http://localhost:8765");
    await options.getByLabel("八位配对码").fill(code);
    await options.getByRole("button", { name: "配对", exact: true }).click();
    await expect(options.locator("#status")).toContainText("配对成功");
  });

  const lesson = await context.newPage();
  await lesson.goto("http://localhost:8897/lesson?private_path=synthetic");
  await lesson.bringToFront();
  await check("natural_alarm_writes_domain_only_heartbeat", async () => {
    await expect.poll(() => heartbeats.some(item => item.domain === "localhost"), {
      timeout: 45_000, intervals: [500, 1000],
    }).toBe(true);
    const body = heartbeats.find(item => item.domain === "localhost");
    assert.deepEqual(Object.keys(body).sort(), [
      "audible", "browser_name", "domain", "duration_s", "incognito", "timestamp_utc",
    ]);
    assert.ok(body.duration_s >= 1 && body.duration_s <= 60);
    assert.equal(body.incognito, false);
    const response = await context.request.get(`${base}/api/v1/telemetry/status`);
    assert.ok((await response.json()).browser_segment_count > 0);
  });

  await check("real_dynamic_rule_blocks_navigation", async () => {
    const response = await context.request.post(`${base}/api/v1/interventions/blocklist`, {
      data: { domain: "localhost", reason: "isolated smoke test" },
    });
    assert.ok(response.ok());
    await worker.evaluate(() => syncBlocklist());
    const rules = await worker.evaluate(() => chrome.declarativeNetRequest.getDynamicRules());
    assert.ok(rules.some(rule => rule.condition.urlFilter === "||localhost^"));
    await assert.rejects(lesson.goto("http://localhost:8897/blocked"), /ERR_BLOCKED_BY_CLIENT/);
  });

  await check("revoking_pairing_removes_extension_rules", async () => {
    const response = await context.request.delete(`${base}/api/v1/telemetry/data?scope=browser`);
    assert.ok(response.ok());
    await worker.evaluate(() => syncBlocklist());
    assert.deepEqual(await worker.evaluate(() =>
      chrome.declarativeNetRequest.getDynamicRules()), []);
    await lesson.goto("http://localhost:8897/unblocked");
    await expect(lesson.getByRole("heading", { name: "Synthetic lesson" })).toBeVisible();
  });

  await check("offline_chat_responds_and_mobile_layout_fits", async () => {
    await app.setViewportSize({ width: 390, height: 844 });
    await app.goto(`${base}/chat`);
    await app.getByRole("textbox", { name: "消息输入" }).fill("如何安排接下来的学习？");
    await app.getByRole("button", { name: "发送", exact: true }).click();
    await expect(app.locator(".chat-bubble.chat-ai")).toHaveCount(1, { timeout: 15_000 });
    assert.equal(await app.evaluate(() =>
      document.documentElement.scrollWidth > window.innerWidth), false);
    await app.screenshot({ path: path.join(output, "chat-mobile.png"), fullPage: true });
  });
  assert.deepEqual(errors, []);
} finally {
  await writeFile(path.join(output, "results.json"), JSON.stringify({ checks, errors }, null, 2));
  await context?.close();
  await new Promise(resolve => site.close(resolve));
}
