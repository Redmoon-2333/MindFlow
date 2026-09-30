// Keyboard / accessibility spot-checks for the reference-faithful shell.
// Usage: node tests/a11y-check.mjs [--port 4173]
import { chromium } from 'playwright-core'

const args = process.argv.slice(2)
const argOf = (name, fallback) => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback
}
const port = argOf('port', '4173')
const base = `http://127.0.0.1:${port}`
const widths = [1920, 1440, 1024, 768, 390]
const routes = ['/', '/focus', '/activities', '/analytics', '/reports', '/intervention', '/panel', '/chat', '/settings', '/model-center', '/execution', '/diagnostics']

const results = []
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH ||
    `${process.env.LOCALAPPDATA ?? ''}\\ms-playwright\\chromium-1228\\chrome-win64\\chrome.exe`,
})

for (const width of widths) {
  const ctx = await browser.newContext({ viewport: { width, height: width === 390 ? 844 : 900 } })
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
      /* geometry-only mode */
    }
  })

  for (const route of routes) {
    await page.goto(`${base}${route}`, { waitUntil: 'networkidle' })
    await page.waitForTimeout(350)
    const audit = await page.evaluate(() => {
      const out = { missingAlt: 0, unlabelledButtons: 0, unlabelledInputs: 0, h1: 0, main: false, nav: false, lang: document.documentElement.lang }
      for (const img of document.querySelectorAll('img')) {
        if (img.getAttribute('alt') === null) out.missingAlt += 1
      }
      for (const btn of document.querySelectorAll('button')) {
        const text = (btn.textContent ?? '').trim()
        const label = btn.getAttribute('aria-label') ?? btn.getAttribute('title')
        if (!text && !label) out.unlabelledButtons += 1
      }
      for (const input of document.querySelectorAll('input, select, textarea')) {
        if (input.type === 'hidden') continue
        const id = input.getAttribute('id')
        const labelled =
          (id && document.querySelector(`label[for="${id}"]`)) ||
          input.getAttribute('aria-label') ||
          input.getAttribute('aria-labelledby') ||
          input.closest('label')
        if (!labelled) out.unlabelledInputs += 1
      }
      out.h1 = document.querySelectorAll('h1').length
      out.main = Boolean(document.querySelector('main#main-content'))
      out.nav = Boolean(document.querySelector('nav[aria-label]'))
      out.skip = Boolean(document.querySelector('.mf-skip-link'))
      return out
    })
    const problems = []
    if (audit.missingAlt) problems.push(`img without alt x${audit.missingAlt}`)
    if (audit.unlabelledButtons) problems.push(`button without name x${audit.unlabelledButtons}`)
    if (audit.unlabelledInputs) problems.push(`input without label x${audit.unlabelledInputs}`)
    if (!audit.main) problems.push('missing main#main-content')
    if (!audit.nav) problems.push('missing nav[aria-label]')
    if (!audit.skip) problems.push('missing skip link')
    if (problems.length) results.push(`${width} ${route}: ${problems.join(', ')}`)
  }

  // Focus ring + focus return on the logout dialog.
  await page.goto(`${base}/`, { waitUntil: 'networkidle' })
  await page.waitForTimeout(500)
  await page.locator('.mf-user').focus()
  const ring = await page.evaluate(() => {
    const el = document.activeElement
    if (!el) return null
    const cs = getComputedStyle(el)
    return { outline: cs.outlineStyle, width: cs.outlineWidth }
  })
  if (!ring || ring.outline === 'none') results.push(`${width} focus: .mf-user has no focus ring`)

  await page.keyboard.press('Enter')
  await page.waitForTimeout(400)
  const dialogOpen = await page.locator('[role="dialog"]').count()
  if (!dialogOpen) results.push(`${width} logout: Enter did not open the dialog`)
  else {
    const focused = await page.evaluate(() => document.activeElement?.className ?? '')
    if (!focused.includes('mf-dialog-no')) results.push(`${width} logout: focus not moved into dialog (on ${focused})`)
    await page.keyboard.press('Escape')
    await page.waitForTimeout(300)
    const returned = await page.evaluate(() => document.activeElement?.className ?? '')
    if (!returned.includes('mf-user')) results.push(`${width} logout: focus not returned after Escape (on ${returned})`)
  }

  // Drawer behaviour below 1024px.
  if (width <= 1023) {
    const navVisible = await page.evaluate(() => {
      const nav = document.querySelector('.mf-navbar')
      return nav ? Math.round(nav.getBoundingClientRect().x) : null
    })
    if (navVisible !== null && navVisible >= 0) results.push(`${width} shell: sidebar should be off-canvas (x=${navVisible})`)
    await page.locator('.mf-header-toggle').click()
    await page.waitForTimeout(600)
    const opened = await page.evaluate(() => Math.round(document.querySelector('.mf-navbar').getBoundingClientRect().x))
    if (opened !== 0) results.push(`${width} drawer: did not open to x=0 (x=${opened})`)
    const scrim = await page.locator('.mf-scrim').count()
    if (!scrim) results.push(`${width} drawer: missing scrim`)
  }

  await ctx.close()
}

await browser.close()
if (results.length) {
  console.log('FINDINGS:')
  for (const r of results) console.log('  ' + r)
  process.exitCode = 1
} else {
  console.log('a11y/keyboard: PASS across ' + widths.join('/') + ' on ' + routes.length + ' routes')
}
