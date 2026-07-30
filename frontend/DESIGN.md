# MindFlow Frontend Design System

> Extracted from `src/theme.css`, App shell, and adjacent pages.  
> Documents the **current** operational dashboard system — not a redesign.

---

## 1. Atmosphere & Visual Identity

| Attribute | Value |
|-----------|-------|
| Mode | Light‑only (no dark mode tokens exist) |
| Surface feel | Flat + thin borders; no heavy shadows, no glass, no noise texture |
| Character | Functional, information‑dense, dashboard‑first |
| Brand anchor | Blue accent (`#4F6BF6`) on cold white / slate backgrounds |

---

## 2. Color Palette

### Brand / Accent

| Token | Value | Usage |
|-------|-------|-------|
| `--color-primary` | `#4F6BF6` | Primary buttons, active nav indicators, logo |
| `--color-primary-hover` | `#3B55D9` | Button hover |
| `--color-primary-light` | `#EEF0FE` | Active nav background, light badges, focus rings |

### Backgrounds

| Token | Value | Usage |
|-------|-------|-------|
| `--color-bg` | `#F8FAFC` | Page background |
| `--color-bg-elevated` | `#FFFFFF` | Cards, sidebar, inputs |
| `--color-bg-inset` | `#F1F5F9` | Table row hover, ghost button hover |

### Borders

| Token | Value | Usage |
|-------|-------|-------|
| `--color-border` | `#E2E8F0` | Card borders, table dividers, sidebar divider, input borders |

### Text

| Token | Value | Usage |
|-------|-------|-------|
| `--color-text-primary` | `#0F172A` | Headings, body |
| `--color-text-secondary` | `#475569` | Subtle body, descriptions |
| `--color-text-tertiary` | `#94A3B8` | Captions, placeholders, stat labels |

### Semantic

| Token | Value | Usage |
|-------|-------|-------|
| `--color-success` | `#22C55E` | Positive indicators |
| `--color-warning` | `#F59E0B` | Warning indicators |
| `--color-danger` | `#EF4444` | Error, destructive actions |
| `--color-info` | `#06B6D4` | Info badges |

---

## 3. Typography

| Level | Size | Weight | Token |
|-------|------|--------|-------|
| Page heading (h1) | 24px | 700 | `.header h1`, `.mc-header h1` |
| Section heading (h3) | 16px | 600 | `.card h3` |
| Body / buttons | 14px | 400 / 500 | `.btn`, `input` |
| Small / table | 13px | 400 | `table td` |
| Caption / stat label | 12px | — | `.stat-card .label` |
| Badge | 11px | 500 | `.badge` |
| Gate detail | 12px | 400 | `.mc-gate .gate-detail` |

**Font stack**: `'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`

**Base size**: 16px via `:root`.

---

## 4. Spacing & Layout

### Spacing scale
| Token | Value |
|-------|-------|
| `gap8` | 8px |
| `gap16` | 16px |
| `mb16` | 16px |
| `mb24` | 24px |
| `mt8` | 8px |
| `mt16` | 16px |

### Layout
- **Fixed sidebar** (left, 220px, full height) + **scrollable main** with `margin-left: 220px`
- Main max-width: `1400px` with `padding: 24px 32px`
- **Responsive**: at `≤768px` sidebar hidden, main margin removed
- Model Center uses intrinsic grid: `repeat(auto-fill, minmax(min(180px,100%), 1fr))`

---

## 5. Components — States, Accessibility, Motion, Layout

### Sidebar (`.sidebar`)
- Fixed 220px, white, right border, `h2` brand, nav links with 3px left active border
- Hides at ≤768px

### Page Header (`.header`, `.mc-header`)
- `h1` + `p` subtitle, 24px bottom margin

### Card (`.card`)
- White, 1px border, 12px radius, 20px padding

### Stat Card (`.stat-card`)
- Same border/radius; 16px padding; `.label` (12px uppercase), `.value` (28px bold), `.sub` (12px)

### Badge (`.badge`)
- Pill shape (999px), 11px, 500 weight; variants: `-primary`, `-success`, `-warning`, `-danger`, `-info`

### Button (`.btn`)
- 8px 16px padding, 8px radius, 14px, inline-flex with gap
- Variants: primary (blue), ghost (transparent+border), danger (red), `-sm` (4px 10px, 12px)
- Disabled: `.5` opacity, `not-allowed` cursor
- **New**: `focus-visible` outline for `.tabs .tab`, `.mc-section .btn` (2px primary)

### Tabs (`.tabs` / `.tab`)
- Flex row, 8px gap; individual: 6px 14px, 13px, border + radius
- Active: primary fill, white text
- **New**: `role="tablist"`/`role="tab"`/`aria-selected` for accessibility

### Table
- Full-width, 13px, `th` 11px uppercase tertiary, row hover `.bg-inset`

### Quality Gate Row (`.mc-gate`)
- Flex row, 10px 14px padding, border, 8px radius, 13px
- `.gate-label` (500 weight), `.gate-detail` (12px tertiary)

### Model Center KPIs (`.mc-kpi-row`)
- Intrinsic grid: `repeat(auto-fill, minmax(min(180px,100%), 1fr))`

### NotFound page (`.not-found-page`)
- Centered, 80px top padding; `.nf-title` (72px, primary), `.nf-desc` (18px, secondary), `.nf-link` (btn)
- At ≤768px: 48px top padding, 48px title

### Spinner
- 24px circle, `--color-border` + `--color-primary` top, `.6s` rotation

### Error Box (`.error-box`)
- Red bg `#fef2f2`, red border `#fecaca`, red text `#991b1b` (existing debt, not tokenized)
- 12px 16px padding, 8px radius, 13px, `role="alert"`

---

## 6. Motion & Interaction

| State | Mechanism | Token |
|-------|-----------|-------|
| Nav hover | Background + color change | `--color-primary-light` |
| Nav active | Border + background | `--color-primary` |
| Button hover | Darker primary / inset | `--color-primary-hover` / `--color-bg-inset` |
| Button disabled | `.5` opacity | — |
| Input focus | Primary ring (3px) | `--color-primary-light` |
| Tab active | Fill switch | `--color-primary` |
| Focus-visible | 2px outline (new) | `--color-primary` |

**No animated transitions on existing components.**
**Spinner**: `@keyframes spin { to { transform: rotate(360deg) } }`, `animation: spin 0.6s linear infinite`.

---

## 7. Depth & Surface

- **No shadows, no z-depth** — flat design throughout
- Surfaces: page background (`--color-bg`), elevated cards (`--color-bg-elevated`), inset (`--color-bg-inset`)
- No glass, backdrop blur, or gradient surfaces

---

## 8. Accessibility Constraints & Accepted Debt

### Hard constraints (documented, not fixed)
1. **Color‑only indicators**: badge colors, stat `.sub.good/.bad` rely solely on hue
2. **Hardcoded badge semantic colors**: `badge-success`/`-warning`/`-danger`/`-info` use raw hex pairs (not tokens)
3. **Missing `prefers-reduced-motion` guard** on spinner
4. **No dark mode** — light-only

### Partially addressed (new work)
5. New tabs use `role="tablist"`/`role="tab"`/`aria-selected`
6. Error boxes use `role="alert"`
7. Job polling status uses `role="status"` with `aria-live="polite"`
8. Focus-visible outlines added to new `.tabs .tab` and `.mc-section .btn`
9. Semantic HTML (headings, tables, forms) pre-existing

### Accepted debt (not fixed in this change)
10. No `role` attributes on sidebar nav or existing tabs
11. No visible focus outlines on existing pages' tabs/buttons
12. Hardcoded inline styles in many pages instead of semantic CSS classes
13. No elevated surfaces beyond flat cards
14. No error boundary wrapping pages
15. Blockers use `.error-box` pattern (raw hex debt inherited)
