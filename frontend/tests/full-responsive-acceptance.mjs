import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { chromium, expect } from "@playwright/test";

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index < 0 ? fallback : args[index + 1];
};
const base = "http://127.0.0.1:4173";
const out = path.resolve(option("out", "../docs/audit/20260930-full"));
const widths = [1920, 1440, 1024, 768, 390];
const routes = ["/", "/focus", "/activities", "/analytics", "/reports", "/intervention",
  "/panel", "/chat", "/settings", "/model-center", "/execution", "/diagnostics"];
const results = [];
await mkdir(path.join(out, "screenshots"), { recursive: true });
const browser = await chromium.launch({ headless: true, channel: "chrome" });

async function check(name, run) {
  try {
    results.push({ name, passed: true, detail: await run() });
    console.log(`PASS ${name}`);
  } catch (error) {
    results.push({ name, passed: false, detail: error.message });
    console.log(`FAIL ${name}: ${error.message}`);
  }
}

try {
  for (const width of widths) {
    const context = await browser.newContext({ viewport: { width, height: width === 390 ? 844 : 900 } });
    try {
      const loginPage = await context.newPage();
      await check(`${width} login and animated canvas`, async () => {
        await loginPage.goto(base, { waitUntil: "networkidle" });
        await expect(loginPage.getByRole("heading", { name: "欢迎回来！" })).toBeVisible();
        await expect(loginPage.getByRole("button", { name: "进入 MindFlow", exact: true })).toBeEnabled();
        const geometry = await loginPage.locator(".login-card").boundingBox();
        assert.ok(geometry.x >= 0 && geometry.x + geometry.width <= width + 1);
        const pixels = () => loginPage.evaluate(() => {
          const canvas = document.querySelector("canvas");
          const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
          let checksum = 0;
          for (let index = 3; index < data.length; index += 4) checksum = (checksum + data[index]) >>> 0;
          return checksum;
        });
        await expect.poll(pixels).toBeGreaterThan(0);
        const before = await pixels();
        await expect.poll(pixels).not.toBe(before);
        await loginPage.screenshot({ path: path.join(out, "screenshots", `${width}-login.png`), fullPage: true });
        return { canvasNonblank: true, canvasAnimated: true };
      });
      await loginPage.close();
      const ticket = await context.request.post(`${base}/api/v1/auth/bootstrap/ticket`);
      assert.equal(ticket.status(), 200, "Bootstrap must succeed; no geometry-only fallback");
      const login = await context.request.post(`${base}/api/v1/auth/bootstrap`, {
        data: { ticket: (await ticket.json()).ticket },
      });
      assert.equal(login.status(), 204);
      await context.addInitScript(() => localStorage.setItem("mindflow_authenticated", "1"));
      const page = await context.newPage();
      let errors = [];
      page.on("pageerror", error => errors.push(error.message));
      page.on("response", response => {
        if (response.url().includes("/api/") && response.status() >= 500) {
          errors.push(`HTTP ${response.status()} ${new URL(response.url()).pathname}`);
        }
      });
      for (const route of routes) {
        errors = [];
        await check(`${width} ${route}`, async () => {
          await page.goto(`${base}${route}`, { waitUntil: "networkidle" });
          await expect(page.locator("main#main-content")).toBeVisible();
          await expect(page.locator(".mf-header-name")).not.toHaveText("页面未找到");
          await expect(page.locator("main")).not.toBeEmpty();
          await page.evaluate(() => document.fonts.ready);
          const detail = await page.evaluate(() => {
            const visible = element => {
              const style = getComputedStyle(element);
              return element.getClientRects().length > 0 && style.visibility !== "hidden"
                && !element.closest("[inert]");
            };
            const issues = [];
            if (document.documentElement.scrollWidth > innerWidth + 1) issues.push("document horizontal overflow");
            for (const element of document.querySelectorAll("main button, main input, main select, main textarea")) {
              if (!visible(element) || element.type === "hidden") continue;
              const id = element.id;
              const named = element.getAttribute("aria-label") || element.getAttribute("title")
                || element.getAttribute("aria-labelledby") || element.closest("label")
                || (id && document.querySelector(`label[for="${CSS.escape(id)}"]`))
                || (element.tagName === "BUTTON" && element.textContent.trim());
              if (!named) issues.push(`unnamed ${element.tagName.toLowerCase()}`);
              const rect = element.getBoundingClientRect();
              const scrollable = element.closest(".table-scroll, .mf-table-scroll, .activity-table-wrap");
              if (!scrollable && (rect.left < -1 || rect.right > innerWidth + 1)) {
                issues.push(`control outside viewport: ${element.tagName}`);
              }
            }
            const brokenImages = [...document.images].filter(image => !image.complete || image.naturalWidth === 0).length;
            if (brokenImages) issues.push(`broken images: ${brokenImages}`);
            if (![...document.images].every(image => image.hasAttribute("alt"))) issues.push("image without alt");
            if (!document.querySelector("nav[aria-label]")) issues.push("missing labelled navigation");
            if (!document.querySelector(".mf-skip-link")) issues.push("missing skip link");
            return { issues, scrollWidth: document.documentElement.scrollWidth, viewport: innerWidth };
          });
          const filename = `${width}-${route === "/" ? "dashboard" : route.slice(1)}.png`;
          await page.screenshot({ path: path.join(out, "screenshots", filename), fullPage: true });
          assert.deepEqual(detail.issues, []);
          assert.deepEqual(errors, []);
          return detail;
        });
      }
      await check(`${width} not-found and return navigation`, async () => {
        await page.goto(`${base}/acceptance-invalid-route`, { waitUntil: "networkidle" });
        await expect(page.getByRole("heading", { name: "404", exact: true })).toBeVisible();
        await page.screenshot({ path: path.join(out, "screenshots", `${width}-not-found.png`), fullPage: true });
        await page.getByRole("link", { name: "返回仪表盘", exact: true }).click();
        await expect(page).toHaveURL(`${base}/`);
        await expect(page.locator(".mf-header-name")).toHaveText("仪表盘");
      });
      await check(`${width} keyboard, navigation, and logout dialog`, async () => {
        await page.goto(base, { waitUntil: "networkidle" });
        const toggle = page.locator(".mf-header-toggle");
        if (width < 1024) {
          assert.equal(await page.locator(".mf-navbar").getAttribute("inert"), "");
          await page.keyboard.press("Tab");
          await expect(page.locator(".mf-skip-link").first()).toBeFocused();
          await page.keyboard.press("Tab");
          await expect(toggle).toBeFocused();
          await page.keyboard.press("Enter");
          await expect(toggle).toHaveAttribute("aria-expanded", "true");
          await expect(page.locator(".mf-scrim")).toBeVisible();
        }
        const user = page.locator(".mf-user");
        await user.focus();
        await page.keyboard.press("Shift+Tab");
        await page.keyboard.press("Tab");
        await expect(user).toBeFocused();
        assert.notEqual(await user.evaluate(element => getComputedStyle(element).outlineStyle), "none");
        await page.keyboard.press("Enter");
        const dialog = page.getByRole("dialog");
        await expect(dialog).toBeVisible();
        await expect(dialog.locator(".mf-dialog-no")).toBeFocused();
        await page.keyboard.press("Tab");
        await expect(dialog.locator(".mf-dialog-close")).toBeFocused();
        await page.keyboard.press("Shift+Tab");
        await expect(dialog.locator(".mf-dialog-no")).toBeFocused();
        await page.keyboard.press("Escape");
        await expect(dialog).toBeHidden();
        await expect(user).toBeFocused();
        if (width < 1024) {
          await page.keyboard.press("Escape");
          await expect(toggle).toBeFocused();
          await expect(toggle).toHaveAttribute("aria-expanded", "false");
        }
        return { authenticated: true, focusTrap: true, escapeRestoration: true };
      });
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
  await writeFile(path.join(out, "responsive-functional.json"), JSON.stringify({
    generatedAt: new Date().toISOString(), widths, routes,
    passed: results.filter(result => result.passed).length,
    failed: results.filter(result => !result.passed).length, results,
  }, null, 2));
}
process.exitCode = results.some(result => !result.passed) ? 1 : 0;
