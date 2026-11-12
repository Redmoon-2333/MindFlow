import assert from "node:assert/strict";
import { mkdir, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import os from "node:os";
import path from "node:path";
import { chromium, expect } from "@playwright/test";

const dataDir = path.resolve(process.argv[2] ?? "");
assert.ok(dataDir.startsWith(path.resolve(os.tmpdir()) + path.sep));
const base = "http://127.0.0.1:8876";
const channel = process.env.MINDFLOW_TEST_BROWSER ?? "chromium";
const output = path.join(path.dirname(dataDir), "login-live", channel);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel, headless: true });

async function expectLogin(page) {
  await expect(page.getByRole("button", { name: "进入 MindFlow", exact: true })).toBeEnabled();
  await expect(page.locator('a[href^="mindflow:"]')).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Dev 登录/ })).toHaveCount(0);
}

try {
  for (const [host, width] of [["127.0.0.1", 1440], ["localhost", 390]]) {
    const localBase = `http://${host}:8876`;
    const context = await browser.newContext({ viewport: { width, height: 900 } });
    try {
      const rejected = await context.request.post(`${localBase}/api/v1/auth/bootstrap/ticket`);
      assert.equal(rejected.status(), 401, "Production ticket issuance stays authenticated");
      const page = await context.newPage();
      const anonymousTickets = [];
      page.on("request", request => {
        if (request.url().endsWith("/auth/bootstrap/ticket")) anonymousTickets.push(request);
      });
      await page.goto(localBase);
      await expectLogin(page);
      assert.equal(await page.evaluate(() =>
        document.documentElement.scrollWidth > window.innerWidth), false);
      await page.screenshot({ path: path.join(output, `login-${width}.png`), fullPage: true });
      const pagesBefore = context.pages().length;
      await page.getByRole("button", { name: "进入 MindFlow", exact: true }).click();
      await expect(page.getByRole("link", { name: "AI 对话", exact: true })).toBeVisible();
      assert.equal(new URL(page.url()).host, `${host}:8876`);
      assert.equal(context.pages().length, pagesBefore, "Login stays in the current tab");
      assert.equal(anonymousTickets.length, 0);
      await page.screenshot({ path: path.join(output, `logged-in-${width}.png`), fullPage: true });
      console.log(`PASS ${host} production login stays in its browser tab at ${width}px`);
    } finally {
      await context.close();
    }
  }

  const foreignContext = await browser.newContext();
  const foreignSite = createServer((_request, response) => {
    response.writeHead(200, { "Content-Type": "text/html" });
    response.end("<!doctype html><title>Foreign origin test</title>");
  });
  await new Promise(resolve => foreignSite.listen(0, "127.0.0.1", resolve));
  try {
    const page = await foreignContext.newPage();
    await page.goto(`http://127.0.0.1:${foreignSite.address().port}`);
    const [rejected] = await Promise.all([
      page.waitForResponse(response => response.url() === `${base}/api/v1/auth/local-session`),
      page.evaluate(async url => {
        await fetch(url, {
          method: "POST", mode: "no-cors", credentials: "include", body: "{}",
        }).catch(() => null);
      }, `${base}/api/v1/auth/local-session`),
    ]);
    assert.equal(rejected.status(), 403, "Another origin must not create a local session");
    const protectedResponse = await foreignContext.request.get(`${base}/api/v1/telemetry/status`);
    assert.equal(protectedResponse.status(), 401);
    console.log("PASS a cross-origin browser POST creates no authenticated session");
  } finally {
    await foreignContext.close();
    await new Promise(resolve => foreignSite.close(resolve));
  }

  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    await page.goto(`${base}/#bootstrap=expired-test-ticket`);
    await expect(page.getByRole("alert")).toContainText("认证失败");
    await expectLogin(page);
    console.log("PASS an expired ticket retains a usable local login entry");

    const token = (await readFile(path.join(dataDir, "token"), "utf8")).trim();
    const response = await context.request.post(`${base}/api/v1/auth/bootstrap/ticket`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    assert.equal(response.status(), 200);
    const { ticket } = await response.json();
    await page.goto(`${base}/#bootstrap=${encodeURIComponent(ticket)}`);
    await expect(page.getByRole("link", { name: "AI 对话", exact: true })).toBeVisible();
    assert.equal(new URL(page.url()).hash, "", "Successful bootstrap removes its ticket");
    const replay = await context.request.post(`${base}/api/v1/auth/bootstrap`, {
      data: { ticket },
    });
    assert.equal(replay.status(), 401, "A used ticket cannot be replayed");
    console.log("PASS a fresh launcher ticket logs in and cannot be reused");

    const logout = await context.request.post(`${base}/api/v1/auth/logout`);
    assert.equal(logout.status(), 204);
    await page.reload();
    await expectLogin(page);
    await page.getByRole("button", { name: "进入 MindFlow", exact: true }).click();
    await expect(page.getByRole("link", { name: "AI 对话", exact: true })).toBeVisible();
    console.log("PASS logout can be followed by another in-page login");
  } finally {
    await context.close();
  }
} finally {
  await browser.close();
}
