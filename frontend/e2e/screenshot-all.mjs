import { chromium } from "@playwright/test";
import { mkdir } from "fs/promises";

const COOKIE = "NoCD5f3fgn8oua72fp4108-73Px0Ktt6OMri4JtA_pE";
const CHROME = "C:\\Users\\lenovo\\AppData\\Local\\ms-playwright\\chromium-1228\\chrome-win64\\chrome.exe";
const OUT = "D:\\大学相关\\01_学业与课程\\07_双创\\MindFlow\\mindflow-app\\frontend\\screenshots";

await mkdir(OUT, { recursive: true });

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
await ctx.addCookies([{ name: "mindflow_session", value: COOKIE, domain: "127.0.0.1", path: "/" }]);

const pages = [
  ["/", "01-dashboard"],
  ["/focus", "02-focus"],
  ["/activities", "03-activities"],
  ["/analytics", "04-analytics"],
  ["/reports", "05-reports"],
  ["/intervention", "06-intervention"],
  ["/panel", "07-panel"],
  ["/chat", "08-chat"],
  ["/settings", "09-settings"],
  ["/diagnostics", "10-diagnostics"],
];

for (const [path, name] of pages) {
  const page = await ctx.newPage();
  await page.addInitScript(() => localStorage.setItem("mindflow_authenticated", "1"));
  await page.goto(`http://127.0.0.1:4173${path}`, { waitUntil: "networkidle", timeout: 15000 });
  await page.waitForTimeout(2500);
  await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true });
  console.log(`✓ ${name}`);
  await page.close();
}

await browser.close();
console.log("All screenshots saved!");
