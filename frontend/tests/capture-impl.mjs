// Captures the local implementation at the same viewports/conditions as
// capture-reference.mjs so the two sets can be diffed directly.
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

const ROUTES = [
  ['dashboard', '/'],
  ['focus', '/focus'],
  ['activities', '/activities'],
  ['analytics', '/analytics'],
  ['reports', '/reports'],
  ['intervention', '/intervention'],
  ['panel', '/panel'],
  ['chat', '/chat'],
  ['settings', '/settings'],
]

const VIEWPORTS = [
  ['1920x1080', { width: 1920, height: 1080 }],
  ['1440x900', { width: 1440, height: 900 }],
]

const DUMP_FN = `() => {
  const doc = document.documentElement;
  const out = { scroll: { scrollWidth: doc.scrollWidth, clientWidth: doc.clientWidth, scrollHeight: doc.scrollHeight, clientHeight: doc.clientHeight }, blocks: [] };
  const pick = (el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return { sel: el.tagName.toLowerCase() + (el.className && typeof el.className === 'string' ? '.' + el.className.trim().split(/\\s+/).join('.') : ''),
      x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height),
      radius: cs.borderRadius, bg: cs.backgroundColor, color: cs.color, fontSize: cs.fontSize, fontWeight: cs.fontWeight, padding: cs.padding, overflowY: cs.overflowY };
  };
  // Same selector set as capture-reference.mjs so both dumps cover the same
  // blocks; the mf-/d- container aliases are added because those class names
  // differ between the two codebases.
  const targets = document.querySelectorAll('.mf-body, .mf-navbar, .mf-content, .mf-header, .mf-main, .d-container, .body, .navbar, .main-content, .content, main, .logout, .modal, [class*="card"], [class*="block"], [class*="title"], [class*="statistic"], [class*="chart"], [class*="table"], h1, h2, h3');
  for (const el of targets) {
    const r = el.getBoundingClientRect();
    // Zero-height bars are real blocks (an empty chart column), so only skip
    // fully invisible boxes — otherwise the two dumps lose rows in step and
    // the positional pairing in compare-geometry.mjs drifts.
    if (r.width < 1 || r.height < 1) continue;
    out.blocks.push(pick(el));
    if (out.blocks.length >= 400) break;
  }
  return out;
}`

const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH ||
    `${process.env.LOCALAPPDATA ?? ''}\\ms-playwright\\chromium-1228\\chrome-win64\\chrome.exe`,
})

const report = {}
for (const [vpName, viewport] of VIEWPORTS) {
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: 1 })
  const page = await ctx.newPage()
  // The login screen is only reachable before the session exists, so capture
  // it first (matching the reference capture, which also shoots /login).
  await page.goto(`${base}/`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(1500)
  {
    const dir = path.join(outDir, vpName)
    await mkdir(dir, { recursive: true })
    await page.screenshot({ path: path.join(dir, 'login.png') })
    const m = await page.evaluate(`(${DUMP_FN})()`)
    report[`${vpName}/login`] = m
    console.log(`captured ${vpName}/login (${m.blocks.length} blocks)`)
  }
  // Authenticate against the local backend (the Vite proxy injects the
  // launcher token for the ticket endpoint). Falls back to the local marker
  // so geometry can still be captured when no backend is running.
  const authed = await page.evaluate(async () => {
    try {
      const ticketRes = await fetch('/api/v1/auth/bootstrap/ticket', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
      })
      if (!ticketRes.ok) return false
      const { ticket } = await ticketRes.json()
      const res = await fetch('/api/v1/auth/bootstrap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ ticket }),
      })
      if (!res.ok) return false
      window.localStorage.setItem('mindflow_authenticated', '1')
      return true
    } catch {
      return false
    }
  })
  console.log(`backend auth: ${authed ? 'session cookie' : 'local marker only'}`)
  page.on('pageerror', (e) => console.log(`  PAGEERR ${vpName}: ${e.message}`))
  page.on('response', (res) => {
    if (res.status() >= 400) console.log(`  HTTP ${res.status()} ${res.url()}`)
  })
  for (const [name, route] of ROUTES) {
    await page.goto(`${base}${route}`, { waitUntil: 'networkidle' })
    await page.waitForTimeout(800)
    const dir = path.join(outDir, vpName)
    await mkdir(dir, { recursive: true })
    await page.screenshot({ path: path.join(dir, `${name}.png`) })
    const metrics = await page.evaluate(`(${DUMP_FN})()`)
    report[`${vpName}/${name}`] = metrics
    console.log(`captured ${vpName}/${name} (${metrics.blocks.length} blocks)`)
  }
  await ctx.close()
}

await mkdir(outDir, { recursive: true })
await writeFile(path.join(outDir, 'metrics.json'), JSON.stringify(report, null, 2), 'utf8')
await browser.close()
console.log(`\nmetrics -> ${path.join(outDir, 'metrics.json')}`)
