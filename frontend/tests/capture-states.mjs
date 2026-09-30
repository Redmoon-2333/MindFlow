// Captures the implementation's interactive states so they can be diffed
// against docs/reference-baseline/<viewport>/states/.
import { chromium } from 'playwright-core'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'

const args = process.argv.slice(2)
const argOf = (name, fallback) => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback
}
const port = argOf('port', '4173')
const base = `http://127.0.0.1:${port}`
const outDir = path.resolve(argOf('out', '../docs/impl-baseline'))
const vpName = argOf('viewport', '1920x1080')
const [w, h] = vpName.split('x').map(Number)

const DUMP_FN = `() => {
  const doc = document.documentElement;
  const pick = (el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return { sel: el.tagName.toLowerCase() + (el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\\s+/).join('.') : ''),
      x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height),
      radius: cs.borderRadius, bg: cs.backgroundColor, color: cs.color, fontSize: cs.fontSize, padding: cs.padding };
  };
  const out = { blocks: [] };
  // Same selector set as .reference-run/capture-states.mjs so both dumps
  // cover the same blocks (plus the mf-/d- shell aliases).
  const targets = document.querySelectorAll('.mf-navbar, .mf-content, .mf-header, .mf-main, .mf-dialog, .mf-nav-item, [class*="card"], [class*="block"], [class*="title"], [class*="row"], [class*="table"], [class*="item"], h1, h2, h3, .modal, .logout, [class*="dialog"]');
  for (const el of targets) {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    out.blocks.push(pick(el));
    if (out.blocks.length >= 300) break;
  }
  return out;
}`

const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH ||
    `${process.env.LOCALAPPDATA ?? ''}\\ms-playwright\\chromium-1228\\chrome-win64\\chrome.exe`,
})
const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 1 })
const page = await ctx.newPage()
const report = {}

await page.goto(`${base}/`, { waitUntil: 'domcontentloaded' })
await page.evaluate(async () => {
  try {
    const r = await fetch('/api/v1/auth/bootstrap/ticket', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
    })
    if (!r.ok) return
    const { ticket } = await r.json()
    await fetch('/api/v1/auth/bootstrap', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ ticket }),
    })
    localStorage.setItem('mindflow_authenticated', '1')
  } catch {
    /* offline capture: geometry only */
  }
})
await page.goto(`${base}/`, { waitUntil: 'networkidle' })
await page.waitForTimeout(1200)

const shot = async (name) => {
  await mkdir(path.join(outDir, vpName, 'states'), { recursive: true })
  await page.screenshot({ path: path.join(outDir, vpName, 'states', `${name}.png`) })
  report[`${vpName}/states/${name}`] = await page.evaluate(`(${DUMP_FN})()`)
  console.log(`state ${name}`)
}

// 1. Sidebar collapsed
await page.locator('.mf-header-toggle').click()
await page.waitForTimeout(900)
await shot('sidebar-collapsed')
await page.locator('.mf-header-toggle').click()
await page.waitForTimeout(700)

// 2. Logout dialog (blurred background)
await page.locator('.mf-user').click()
await page.waitForTimeout(600)
await shot('logout-dialog')
await page.locator('.mf-dialog-no').click()
await page.waitForTimeout(400)

// 3. Nav hover on a non-active item
await page.locator('.mf-nav-item').nth(2).hover()
await page.waitForTimeout(500)
await shot('nav-hover')
await page.mouse.move(5, 5)
await page.waitForTimeout(300)

// 4. Dashboard mid / bottom scroll
for (const [label, ratio] of [['mid', 0.5], ['bottom', 1.0]]) {
  await page.evaluate(`(() => {
    const sc = document.querySelector('.mf-main');
    if (sc) sc.scrollTop = (sc.scrollHeight - sc.clientHeight) * ${ratio};
  })()`)
  await page.waitForTimeout(700)
  await shot(`dashboard-${label}`)
  await dumpScroll(`dashboard-${label}`)
}

async function dumpScroll(name) {
  await page.evaluate(`(() => {
    const sc = document.querySelector('.mf-main');
    return sc ? sc.scrollTop : 0;
  })()`)
  report[`${vpName}/states/${name}`] = await page.evaluate(`(${DUMP_FN})()`)
}

// 5. Bottom of each replicable page
for (const route of ['/focus', '/analytics', '/reports', '/activities']) {
  await page.goto(`${base}${route}`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(900)
  await page.evaluate(`(() => {
    const sc = document.querySelector('.mf-main');
    if (sc) sc.scrollTop = sc.scrollHeight;
  })()`)
  await page.waitForTimeout(700)
  const name = route.replace(/\//g, '') || 'root'
  await shot(`${name}-bottom`)
  await dumpScroll(`${name}-bottom`)
}

await mkdir(outDir, { recursive: true })
await writeFile(path.join(outDir, `metrics-states-${vpName}.json`), JSON.stringify(report, null, 2), 'utf8')
await browser.close()
console.log('impl states done')
