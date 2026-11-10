import { test, expect, type Page } from "@playwright/test";
import { setupExplainable, setupReviewShotsRealistic } from "./fixtures/theme-api";

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

/** The dashboard is photographed with the evidence COLLAPSED (F8) — that is the
 *  state a user lands on — so this only waits for the hero reading. */
async function prepareDashboard(page: Page) {
  await expect(page.locator(".dash-hero")).toBeVisible();
  await expect(page.locator(".dash-hero .hero-num")).toBeVisible();
  await expect(page.locator(".dash-hero .sig-disclosure-trigger")).toHaveAttribute("aria-expanded", "false");
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
  { key: "dashboard", path: "/", prepare: prepareDashboard, fixture: "realistic" },
  { key: "analytics", path: "/analytics", prepare: prepareAnalytics, fixture: "explainable" },
  { key: "panel", path: "/panel", prepare: preparePanel, fixture: "explainable" },
  { key: "intervention", path: "/intervention", prepare: prepareIntervention, fixture: "explainable" },
] as const;

/** The dashboard gets the realistic series (F7); every other page keeps the
 *  fixture it was reviewed against (F8: regenerate, do not re-stage). */
async function installFixture(page: Page, fixture: "realistic" | "explainable") {
  if (fixture === "realistic") await setupReviewShotsRealistic(page);
  else await setupExplainable(page);
}

for (const theme of ["navy", "white"] as const) {
  for (const target of PAGES) {
    test(`${target.key} full page at 1440 in the ${theme} theme`, async ({ page }) => {
      await installFixture(page, target.fixture);
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
    await installFixture(page, target.fixture);
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

/** The signature block: reading, band word, activity line and signal track,
 *  with the evidence collapsed — the state a user sees on arrival. */
test("hero close-up", async ({ page }) => {
  await setupReviewShotsRealistic(page);
  await page.setViewportSize(WIDE);
  await open(page, "/", "navy");
  const hero = page.locator(".dash-hero");
  await expect(hero.locator(".hero-num")).toBeVisible();
  await expect(page.locator(".dash-hero .sig-disclosure-trigger")).toHaveAttribute("aria-expanded", "false");
  await hero.screenshot({ path: `${SHOTS}/dashboard-hero-navy.png` });
});

/** Row 1 with the evidence open (F8): shows how 此刻 grows and 下一步 stops
 *  stretching, cropped to the two-column grid rather than the whole viewport. */
test("dashboard row 1 with evidence open", async ({ page }) => {
  await setupReviewShotsRealistic(page);
  await page.setViewportSize(WIDE);
  await open(page, "/", "navy");
  await expect(page.locator(".dash-hero")).toBeVisible();

  const trigger = page.locator(".dash-hero .sig-disclosure-trigger");
  await trigger.click();
  await expect(trigger).toHaveAttribute("aria-expanded", "true");
  await expect(page.locator(".dash--evidence-open")).toHaveCount(1);
  await expect(page.locator(".dash-hero figure.mf-chart")).toBeVisible();

  // Crop to the first grid row: from the hero's top to the top of the strip,
  // which is the first full-width element below the two columns.
  const clip = await page.evaluate(() => {
    const hero = document.querySelector(".dash-hero")!.getBoundingClientRect();
    const strip = document.querySelector(".dash > .sig-metrics")!.getBoundingClientRect();
    const left = Math.min(hero.left, document.querySelector(".dash-next")!.getBoundingClientRect().left);
    const right = Math.max(hero.right, document.querySelector(".dash-next")!.getBoundingClientRect().right);
    return {
      x: Math.floor(left),
      y: Math.floor(hero.top),
      width: Math.ceil(right - left),
      height: Math.ceil(strip.top - hero.top),
    };
  });
  await page.screenshot({ path: `${SHOTS}/dashboard-evidence-open-1440-navy.png`, clip });
});

test("contribution bar close-up", async ({ page }) => {
  await setupReviewShotsRealistic(page);
  await page.setViewportSize(WIDE);
  await open(page, "/", "navy");
  await page.locator(".dash-hero .sig-disclosure-trigger").click();
  await expect(page.locator(".dash-hero figure.mf-chart")).toBeVisible();

  const block = page.locator(".dash-hero").locator("figure.mf-chart");
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
