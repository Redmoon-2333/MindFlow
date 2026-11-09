# MindFlow Frontend Design System

> Reference for the **mid-term upgrade** UI. Every value below is read from
> `src/theme.css` + `src/themes.ts` and is what the running build actually
> applies — not a proposal. If a value here and `themes.ts` disagree, the code
> is right.

Direction: **calm instrument panel**. Low density, hierarchy from whitespace,
flat surfaces, one accent (focus teal). No gradients, no glow; the only
elevation flourish is a 1px light line on the top edge of a card.

---

## 1. Token layers

Three layers, in precedence order:

| Layer | File | Role |
|---|---|---|
| 1. Fallback | `src/theme.css` `:root` | Navy values only; exists for the first paint before JS runs |
| 2. Active theme | `src/themes.ts` → `applyTheme()` | Writes every custom property as an **inline style on `<html>`**, which outranks any stylesheet |
| 3. Consumers | page CSS, `primitives.css`, `charts.css` | Use `var(--token)` only; never re-declare a palette value |

Because layer 2 is inline style, adding a colour to a stylesheet instead of a
theme breaks it the moment the user picks another theme.

**Contrast contract** (asserted by `e2e/theme-layout-review.spec.ts`):
`--mf-text`, `--mf-text-dim`, `--color-bg`, `--mf-panel` and
`--color-bg-inset` must stay **plain hex** — the spec parses them to compute
luminance ratios. Any `color-mix()` or `rgba()` in those five breaks the probe.

Measured minimum contrast per theme — every one of `--mf-text`, `--mf-text-dim`,
`--good`, `--warn`, `--bad`, `--info` against `--color-bg`, `--mf-panel` and
`--color-bg-inset`:

| Theme | min ratio | tightest pair |
|---|---|---|
| white | 5.62 | `--good` on inset |
| black | 7.77 | `--bad` on panel |
| navy | 5.16 | `--bad` on panel |
| burgundy | 8.24 | `--bad` on panel |

All four clear WCAG AA 4.5:1. The e2e probe itself only asserts the two text
tokens; the semantic four are checked here, offline, because the probe's
luminance helper cannot parse `color-mix()`.

---

## 2. Tokens — the values in use

### Spacing (4px base)

| Token | Value |
|---|---|
| `--space-1` | 4px |
| `--space-2` | 8px |
| `--space-3` | 12px |
| `--space-4` | 16px |
| `--space-5` | 20px |
| `--space-6` | 24px |
| `--space-7` | 32px |
| `--space-8` | 48px |

The scale stops at 48px: nothing in the app needs a larger gap, and a page
gutter above that starts to look like an empty column rather than breathing room.

### Radius

| Token | Value | Use |
|---|---|---|
| `--radius-sm` | 6px | inline code, small swatches, chart legend chips |
| `--radius-md` | 10px | inputs, inner blocks, stat tiles, tooltips |
| `--radius-lg` | 16px | cards, panels, dialogs, chart figures |
| `--radius-pill` | 999px | badges, chips, buttons-as-pills, progress tracks |

Every `border-radius` in `src/` is one of these four, or a percentage for a
true circle (`50%` dots, `20%`/`25%` logo marks). A raw px radius is a bug —
`grep -rn "border-radius: [0-9]" src --include=*.css` returns only percentages.

### Type

| Token | Value |
|---|---|
| `--fs-xs` | 12px |
| `--fs-sm` | 13px |
| `--fs-base` | 14px |
| `--fs-md` | 16px |
| `--fs-lg` | 20px |
| `--fs-xl` | 24px |
| `--fs-2xl` | 32px |
| `--leading-tight` | 1.25 (headings) |
| `--leading-normal` | 1.5 (body) |
| `--leading-relaxed` | 1.7 (long prose, notes) |

Font: `--font-ui` = `"Inter", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif`.
Mono: `--font-mono` = `"JetBrains Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace`.

Inter is **not bundled** — no new dependency was allowed, so the stack falls
through to PingFang SC / Microsoft YaHei on a machine without Inter installed.
Numbers are always `font-variant-numeric: tabular-nums`; `[data-numeric]`,
`.stat-tile-value`, `.stat-card .value`, `.mf-metric` and every chart value label
opt in.

### Elevation

`--shadow-1` = `0 1px 2px rgb(0 0 0 / S×0.5)` and `--shadow-2` =
`0 8px 24px rgb(0 0 0 / S×0.85)`, where `S` is the theme's `shadowStrength`:

| Theme | `shadowStrength` | `--shadow-1` alpha | `--shadow-2` alpha |
|---|---|---|---|
| white | 0.1 | 0.05 | 0.09 |
| navy | 0.32 | 0.16 | 0.27 |
| burgundy | 0.4 | 0.20 | 0.34 |
| black | 0.5 | 0.25 | 0.43 |

Deriving rather than hard-coding means a shadow tuned for `#040b14` is not
silently reused on white, where it would read as a dirty smudge.

| Token | Value |
|---|---|
| `--shadow-focus` | `0 0 0 2px color-mix(in srgb, brand 45%, transparent)` |
| `--card-highlight` | `inset 0 1px 0 rgb(255 255 255 / 0.06)` on dark, `0.7` on white |

There is **no ambient glow and no gradient**: `--mf-bg-radial` and
`--mf-login-radial` both collapse to the flat theme background.

### Semantic state

| Token | Navy | White | Meaning |
|---|---|---|---|
| `--good` | `#2dd4bf` | `#0a6b52` | focus / accepted / positive contribution |
| `--warn` | `#f5b84b` | `#7a5200` | warnings, the dominant TMT barrier |
| `--bad` | `#f87171` | `#a8302b` | distracted / rejected / negative contribution |
| `--info` | `#7aa2ff` | `#1c4f96` | neutral information |

Each has a `-soft` sibling at 14% alpha (`--good-soft` …) for badge and callout
fills.

### Charts

| Token | Navy |
|---|---|
| `--chart-1` | `#7aa2ff` |
| `--chart-2` | `#2dd4bf` |
| `--chart-3` | `#f5b84b` |
| `--chart-4` | `#c084fc` |
| `--chart-5` | `#f87171` |
| `--chart-6` | `#94a3b8` |
| `--chart-pos` | `var(--good)` (signed: toward focus) |
| `--chart-neg` | `var(--bad)` (signed: toward distraction) |
| `--chart-grid` / `--chart-axis` / `--chart-track` | translucent stroke tokens |
| `--chart-tooltip-bg` / `--chart-tooltip-border` | tooltip surface |

### Procrastination types

Fixed slot order. Five categories, same order in every theme:

| Token | Navy |
|---|---|
| `--type-task_aversion` | `#f5b84b` |
| `--type-impulsivity` | `#f87171` |
| `--type-decisional` | `#7aa2ff` |
| `--type-perfectionism` | `#c084fc` |
| `--type-emotional_regulation` | `#2dd4bf` |

### Theme palettes (the source of truth)

| | bg | panel | inset | text | dim | brand | border |
|---|---|---|---|---|---|---|---|
| white | `#f4f6fa` | `#ffffff` | `#eaeff7` | `#101c2b` | `#4a5c72` | `#1f5fa0` | `#c2cee0` |
| black | `#111111` | `#1c1c1c` | `#181818` | `#f2f2f2` | `#bdbdbd` | `#8fb6ff` | `#4a4a4a` |
| navy | `#040b14` | `#172b46` | `#0c1a30` | `#e0e9f4` | `#a9bdd3` | `#8dc7ff` | `#38577e` |
| burgundy | `#180c12` | `#2b1a23` | `#241520` | `#f7eaf0` | `#cfb5c1` | `#ffb1c8` | `#6d4a59` |

---

## 3. Layout

| Attribute | Value |
|---|---|
| Sidebar | `250px`, fixed, collapses to a drawer at ≤1023px |
| Top bar | `65px`; `.mf-main` is the scroll container (`100vh − 65px`) |
| Content max width | none — the shell fills, pages use a token gutter |
| Page gutters | `--space-6` both sides, `--space-8` bottom (desktop); `--space-4` / `--space-7` at ≤700px |
| Card padding | `--space-5` |
| Grid gap inside a page | `--space-6` for major blocks, `--space-4` for tile rows |

`.mf-main` scrolls internally, which is why `fullPage: true` screenshots come
out blank lower down: use `scrollIntoViewIfNeeded()` plus a viewport shot.

Breakpoints in use: `1439px` (shell tightening), `1100px` (2-column grids),
`1023px` (sidebar → drawer), `767px` and `700px` (single column), `560px`
(chart compact mode), `420px` (contribution-bar pole captions dropped). A
handful of one-off `1440` / `768` / `600` / `400` queries survive from the
inherited reference CSS.

---

## 4. Primitives

| Component | File | Contract |
|---|---|---|
| `PageHeader` | `components/PageHeader.tsx` | The page's single `<h1>`. Title `--fs-xl` / 600, description `--fs-sm` muted, optional eyebrow + right-aligned actions. Hairline bottom border. |
| `WorkspacePage` | `components/WorkspacePage.tsx` | Page shell for form pages: renders `PageHeader` + children. Deliberately does **not** emit `.workspace-heading` (a spec asserts it stays absent). |
| `StatTile` | `components/StatTile.tsx` | KPI tile. `tone` ∈ `neutral / good / warn / bad / brand` drives a 3px left rule and the hint colour. `value="—"` is a legitimate unknown, never a fake zero. **No page mounts it yet** — Dashboard still builds `.d-statistic-item` by hand (see debt). |
| `EmptyState` | `components/EmptyState.tsx` | The one empty/error surface. `variant="error"` reserves the coral title. Empty copy is an invitation to act. |
| `Toaster` + `toast-store` | `components/Toaster.tsx`, `toast-store.ts` | Mounted **once** in `App.tsx`. `useSyncExternalStore`, no provider. Errors persist until dismissed; identical consecutive toasts collapse to one with `×n`; max 4 on screen. Each toast carries its own `role="status"` / `role="alert"` — the container is a plain layout div on purpose. |
| `FeedbackPrompt` | `components/FeedbackPrompt.tsx` | Proactive-learning question. Fed by the WebSocket `feedback_request` event and a 60s poll; a backend without the feature 404s and the poll goes quiet. Sits above the toast region, and on a phone it wins the corner. |

Supporting CSS in `components/primitives.css`: `.page-header*`, `.stat-tile*`,
`.stat-grid`, `.empty-state*`, `.toast*`, `.feedback-prompt*`.

### Explainability blocks (same file)

| Class | Use |
|---|---|
| `.proof` | Evidence panel: inset surface, hairline border, `--radius-lg`, `--space-4` padding |
| `.proof-head` / `.proof-title` / `.proof-muted` / `.proof-note` | Header row and secondary copy inside a proof block |
| `.proof-details` | Collapsible disclosure ("为什么这样判断"); `▸` marker rotates 90° when open |
| `.chip-row` / `.chip` | Cognitive distortions and reasons; `data-tone="warn" \| "info"` for tinted variants. `align-items: flex-start` is load-bearing — inheriting the grid's `stretch` turns a one-item row into a circle |
| `.bullet-list` | Evidence bullets with a brand-coloured dot |
| `.callout` | The "下一步可以做" strip; `data-tone="warn" \| "good"` recolours its left rule and fill |
| `.type-bars` / `.type-bar-*` | Type-confidence rows, one colour per `--type-*` slot, leading row emphasised via `data-leading` |
| `.deviation-table` / `.deviation-meta` | Robust-z table; numeric columns right-aligned and tabular |
| `.meta-strip` / `.meta-item` | Panel uncertainty / escalated / call-count readout |
| `.opinion-card` / `.opinion-list` | One card per panel speaker turn, 3px left rule in `--opinion-color` |
| `.rationale-block` / `.rationale-latest` / `.rationale-history` | "为什么是这条提醒" — candidates + evidence for the latest item, and a `<details>` per history item |

---

## 5. Charts

All in-house SVG. No chart library. Shared helpers: `scale.ts`
(`niceTicks`, `formatCompact`, `formatSigned`, `clamp`) and
`useElementWidth.ts` (responsive redraw without `ResizeObserver` leaks).

| Component | Signature | Encoding rules |
|---|---|---|
| `BarChart` | `data`, `ariaLabel`, `unit`, `color`, `highlightColor`, `colorFor`, `height`, `showValues`, `yAxisLabel`, `htmlXLabelClass` | Vertical bars, real y axis with `niceTicks` gridlines, hover **and** keyboard-focus tooltip. `role="img"` + `aria-label` + `<figcaption>` summary. `htmlXLabelClass` renders x ticks as HTML so an existing page selector (`.d-chart-box-item-week`) keeps working. |
| `StackedBar` | `data`, `series`, `ariaLabel`, `unit`, `height`, `yAxisLabel`, `emptyMessage`, `showLegend` | Two-part daily composition (专注 / 分心). Segments stack from the baseline in series order so the colour nearest the axis always means the same thing. **Currently not mounted on any page** — kept as the composition chart for Focus/Reports. |
| `ContributionBar` | `data`, `ariaLabel`, `outputSpace`, `baseValue`, `maxRows` | Signed SHAP-style rows. Zero axis sits left of centre; bars grow right toward focus, left toward distraction. Both **side and colour** carry the sign, so it survives colour-blindness and greyscale printing. Value labels live in a fixed right gutter (a long negative bar would otherwise push its label into the name column). Below 420px the pole captions are dropped. |
| `TmtGauge` | `tmt`, `showNote` | Renders `Motivation = E · V / (1 + I · D)` literally. E and V are drivers (teal), I and D are barriers (coral) — colour follows **meaning, not sign**. The dominant barrier keeps its colour and is **outlined in amber** plus a `瓶颈` badge, so "this is the bottleneck" never masks what it measures. The boundary note ("行为代理变量，不是心理测量") is on by default. |
| `PosteriorBars` | `candidates`, `ariaLabel` | Beta posterior mean (dot) + 90% credible interval (bar) per candidate. Chosen row: solid `--good`, labelled `已选` — not merely recoloured. Unchosen rows sit at 40% opacity. The interval is computed in TS (`betaInterval`), not trusted from the wire. |

Chart CSS lives in `components/charts/charts.css`; colour comes only from
`--chart-*` / `--type-*` / `--good` / `--warn` / `--bad`.

---

## 6. Explaining the system — where the evidence lives

| Screen | Surface |
|---|---|
| Dashboard | "ML 专注预测" card → `为什么这样判断` `<details>` → `ContributionBar` (`maxRows` defaults to 8; method badge reads `TreeSHAP` / `线性归因`) + `base_value` note. Missing data → `暂无解释数据` plain text, no empty frame. |
| Analytics | `归因分析` → type-confidence bars, cognitive distortions, `下一步可以做`, TMT decomposition, and the behaviour-deviation table with empirical-Bayes shrinkage metadata (`n=`, `收缩系数`). Every block degrades to `暂无…数据`. |
| Panel | `本次分析的过程信息` meta strip (不确定度 / 是否升级复核 / 模型调用 / 讨论轮次), TMT block, and the transcript as opinion cards with human role names (`分析师`, `心理学家`, …). |
| Intervention | "为什么是这条提醒" with selection method (`Thompson 采样` / `规则选择`), humanised context bucket (`情境：下午 · 分心`), evidence bullets, `PosteriorBars`. History items carry the same block in a `<details>`. |
| Global | `FeedbackPrompt` asks "刚才 {时段} 你在专注吗？" with the reason it asked, and writes the label back. |

Copy rules applied throughout: wire identifiers are translated before display
(`personal_model` → `个人模型`, `afternoon|distracted` → `下午 · 分心`,
`analyst` → `分析师`); an unknown value falls through unchanged rather than
being hidden.

---

## 7. Motion & accessibility

- Transitions are limited to hover/press feedback and bar width (0.4s ease);
  the toast and the feedback prompt animate in over 0.18s.
- `prefers-reduced-motion: reduce` disables the toast/prompt entrance, the bar
  transition and shell transitions.
- Every chart is `role="img"` with an `aria-label`; the bar chart also exposes
  its tooltip on keyboard focus and repeats the series in the figcaption.
- Focus rings: 2px `--mf-brand`, offset 2–4px, on buttons, inputs, tabs,
  disclosures and chart SVGs.
- Navigation has a skip link (`.mf-skip-link`).
- Colour is never the only signal: contribution bars encode sign by side as
  well, the chosen posterior row is labelled, and the dominant TMT barrier is
  outlined and badged.

---

## 8. Known debt

1. `.d-state-item`, `.d-chart`, `.d-control` and friends still lean on
   absolutely positioned titles from the inherited reference layout; the
   dashboard in particular uses `position: absolute` for several labels.
2. `!important` survives in 18 places, mostly page overrides that have to beat
   the inherited reference selectors: `picker.css` 7, `reports.css` 3,
   `themes.css` 2, `activities.css` 2, `analytics.css` 2, `shell.css` 1,
   `layout-refinements.css` 1. Removing them means raising specificity instead,
   which is a larger refactor than this pass.
3. `EmptyState` is mounted only on Diagnostics; the Panel / Intervention /
   Reports / Focus empty branches still use their page-local classes.
4. Several pages still use inline `style` objects for spacing (Settings,
   ModelCenter, Execution).
5. MUI, dayjs and sass remain in `package.json` but are unused; removing them
   was out of scope for this pass.
6. `StackedBar` has no live caller yet, and `StatTile` is defined but unused —
   Dashboard's KPI row predates it and would be a clean first adoption.
7. The card-title indents on Focus / Reports are now `padding-left: var(--space-7)`
   on a `width: 100%` box. They were `margin-left: 30px`, which made the box
   30px wider than its card; the shell's `overflow: hidden` hid it until the
   page gutter shrank and the overflow became visible. The `.state-item-text` /
   `.state-item-number` / `.zhuanzhudongcha-text` rules on those pages still use
   the margin form — they are inside cards with enough slack that they do not
   overflow, so they were left alone.

---

## 9. Verification

```bash
npm run build                               # tsc -b + vite build, 0 TS errors
npm run lint                                # oxlint src e2e tests
npx playwright test -c playwright.ui-review.config.ts
```

The UI-review config intercepts every API and WebSocket call, so it needs no
backend. Screenshots land in `e2e/screenshots/` (`midterm/` holds the
Dashboard / Analytics / Panel / Intervention set at 1440×900 and 375×812).
