import { test, expect, type Page } from "@playwright/test";
import { setupExplainable } from "./fixtures/theme-api";

/**
 * Review captures for the mid-term aesthetic sign-off.
 *
 * These are *not* assertions about layout — they exist so a human can look at
 * whole pages in both themes and at the three evidence charts up close. The
 * only checks here are the ones that make a capture trustworthy: the content
 * is actually on screen before the shutter fires, and the page threw nothing.
 *
 * The shell is a fixed-height frame: `.mf-body`, `.mf-content` and `.mf-main`
 * are each pinned to the viewport height with `overflow: hidden` and only
 * `.mf-main` scrolls inside them. `fullPage: true` alone therefore captures
 * exactly one viewport and everything below the fold is missing. Each capture
 * releases the height constraint down the whole chain (html → body →
 * .mf-body → .mf-content → .mf-main) for the duration of the shot, and
 * restores it afterwards so a later capture is not measuring a mutated page.
 */

const SHOTS = "e2e/screenshots/midterm-review";
const WIDE = { width: 1440, height: 900 };
const NARROW = { width: 375, height: 812 };

// Opt-in only: these are capture jobs, not regression tests. The main config
// still discovers this file, so it must stay skipped unless explicitly asked.
test.skip(process.env.MF_REVIEW_SHOTS !== "1", "set MF_REVIEW_SHOTS=1 to capture review shots");

async function open(page: Page, path: string, theme: "navy" | "white") {
  await page.addInitScript(value => localStorage.setItem("mindflow_theme", value), theme);
  await page.goto(path);
  await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
}

/** Every ancestor that pins a viewport height: `.mf-body` and `.mf-content`
 *  are both `height: 100vh; overflow: hidden`, so stretching `.mf-main` alone
 *  is not enough — the clip happens above it. */
const FRAME_SELECTOR = ".mf-body, .mf-content, .mf-main";

/** Expand the scroll chain so fullPage captures the whole document. */
async function stretchForFullPage(page: Page) {
  await page.evaluate(selector => {
    for (const node of [
      document.documentElement,
      document.body,
      ...Array.from(document.querySelectorAll<HTMLElement>(selector)),
    ]) {
      node.setAttribute("data-prev-style", node.getAttribute("style") ?? "");
      node.style.height = "auto";
      node.style.maxHeight = "none";
      node.style.overflow = "visible";
    }
  }, FRAME_SELECTOR);
}

async function restoreScrollContainer(page: Page) {
  await page.evaluate(selector => {
    for (const node of [
      document.documentElement,
      document.body,
      ...Array.from(document.querySelectorAll<HTMLElement>(selector)),
    ]) {
      const prev = node.getAttribute("data-prev-style") ?? "";
      if (prev) node.setAttribute("style", prev);
      else node.removeAttribute("style");
      node.removeAttribute("data-prev-style");
    }
  }, FRAME_SELECTOR);
}

async function fullPageShot(page: Page, name: string) {
  await page.evaluate(() => window.scrollTo(0, 0));
  await stretchForFullPage(page);
  await page.screenshot({ path: `${SHOTS}/${name}`, fullPage: true });
  await restoreScrollContainer(page);
}

/** Prepare each page so its evidence is rendered before we photograph it. */
async function prepareDashboard(page: Page) {
  const card = page.locator(".d-extra-card", { hasText: "ML 专注预测" });
  await expect(card).toBeVisible();
  await card.getByText("为什么这样判断", { exact: true }).click();
  await expect(card.locator("figure.mf-chart")).toBeVisible();
}

async function prepareAnalytics(page: Page) {
  await page.getByRole("button", { name: "运行归因分析" }).click();
  await expect(page.locator(".attribution-evidence")).toBeVisible();
  await expect(page.getByRole("region", { name: "动机分解" })).toBeVisible();
}

async function preparePanel(page: Page) {
  await page.getByRole("button", { name: "查看上次结果" }).click();
  await expect(page.getByLabel("本次分析的过程信息")).toBeVisible();
  await expect(page.getByRole("heading", { name: "动机分解" })).toBeVisible();
}

async function prepareIntervention(page: Page) {
  await expect(page.getByLabel("为什么是这条提醒").first()).toBeVisible();
}

const PAGES = [
  { key: "dashboard", path: "/", prepare: prepareDashboard },
  { key: "analytics", path: "/analytics", prepare: prepareAnalytics },
  { key: "panel", path: "/panel", prepare: preparePanel },
  { key: "intervention", path: "/intervention", prepare: prepareIntervention },
] as const;

for (const theme of ["navy", "white"] as const) {
  for (const target of PAGES) {
    test(`${target.key} full page at 1440 in the ${theme} theme`, async ({ page }) => {
      await setupExplainable(page);
      await page.setViewportSize(WIDE);
      await open(page, target.path, theme);
      await target.prepare(page);

      const errors: string[] = [];
      page.on("pageerror", e => errors.push(String(e)));

      await fullPageShot(page, `${target.key}-1440-${theme}-full.png`);
      expect(errors, `page errors: ${errors.join(" | ")}`).toHaveLength(0);
    });
  }
}

for (const target of PAGES.filter(p => p.key === "dashboard" || p.key === "intervention")) {
  test(`${target.key} full page at 375 in the navy theme`, async ({ page }) => {
    await setupExplainable(page);
    await page.setViewportSize(NARROW);
    await open(page, target.path, "navy");
    await target.prepare(page);

    const errors: string[] = [];
    page.on("pageerror", e => errors.push(String(e)));

    await fullPageShot(page, `${target.key}-375-navy-full.png`);
    // A full-page capture of a phone viewport is exactly where an overflow
    // becomes visible as a white strip, so assert it while we are here.
    expect(await page.locator(".mf-main").evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
    expect(errors, `page errors: ${errors.join(" | ")}`).toHaveLength(0);
  });
}

test("contribution bar close-up", async ({ page }) => {
  await setupExplainable(page);
  await page.setViewportSize(WIDE);
  await open(page, "/", "navy");
  await prepareDashboard(page);

  const block = page.locator(".d-extra-card", { hasText: "ML 专注预测" }).locator("details.d-explain");
  await block.scrollIntoViewIfNeeded();
  await block.screenshot({ path: `${SHOTS}/closeup-contribution-bar-navy.png` });
});

test("tmt gauge close-up", async ({ page }) => {
  await setupExplainable(page);
  await page.setViewportSize(WIDE);
  await open(page, "/analytics", "navy");
  await prepareAnalytics(page);

  const block = page.getByRole("region", { name: "动机分解" });
  await block.scrollIntoViewIfNeeded();
  await block.screenshot({ path: `${SHOTS}/closeup-tmt-gauge-navy.png` });
});

test("posterior bars close-up", async ({ page }) => {
  await setupExplainable(page);
  await page.setViewportSize(WIDE);
  await open(page, "/intervention", "navy");
  await prepareIntervention(page);

  const block = page.getByLabel("为什么是这条提醒").first();
  await block.scrollIntoViewIfNeeded();
  await block.screenshot({ path: `${SHOTS}/closeup-posterior-bars-navy.png` });
});
