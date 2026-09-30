// Lists form controls with no accessible name (label[for] / aria-label /
// wrapping label) so they can be fixed. Usage: node tests/label-audit.mjs
import { chromium } from 'playwright-core'

const base = 'http://127.0.0.1:4173'
const routes = ['/chat', '/settings', '/execution', '/focus', '/activities', '/model-center', '/diagnostics', '/intervention', '/panel', '/analytics', '/reports']

const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH ||
    `${process.env.LOCALAPPDATA ?? ''}\\ms-playwright\\chromium-1228\\chrome-win64\\chrome.exe`,
})
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } })
const page = await ctx.newPage()
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
    /* offline */
  }
})

for (const route of routes) {
  await page.goto(`${base}${route}`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(500)
  const bad = await page.evaluate(() => {
    const out = []
    for (const el of document.querySelectorAll('input, select, textarea')) {
      if (el.type === 'hidden') continue
      const id = el.getAttribute('id')
      const labelledBy = id ? document.querySelector(`label[for="${id}"]`) : null
      const named =
        labelledBy ||
        el.getAttribute('aria-label') ||
        el.getAttribute('aria-labelledby') ||
        el.closest('label')
      if (!named) {
        out.push({
          tag: el.tagName.toLowerCase(),
          type: el.type ?? '',
          cls: typeof el.className === 'string' ? el.className.slice(0, 60) : '',
          placeholder: el.getAttribute('placeholder') ?? '',
          outer: el.outerHTML.slice(0, 120),
        })
      }
    }
    return out
  })
  if (bad.length) console.log(route, JSON.stringify(bad, null, 1))
}

await browser.close()
