/**
 * Theme tokens — the single source of truth for every palette value.
 *
 * `applyTheme` writes custom properties as inline styles on <html>, which
 * outranks any stylesheet. theme.css therefore only carries navy fallbacks,
 * and page CSS must consume tokens rather than re-declaring colours.
 *
 * Design language "Signal" (docs/audit/20261110-midterm/ui-redesign-spec.md §2).
 * Token names follow the spec verbatim (`--surface`, `--line`, `--signal`, …);
 * the pre-redesign hooks (`--mf-*`, `--color-*`) are aliased onto them so
 * pages that were not redesigned in phase 1 keep working unchanged.
 *
 * Contract used by e2e (theme-layout-review.spec.ts): `--mf-text`,
 * `--mf-text-dim`, `--color-bg`, `--mf-panel` and `--color-bg-inset` must be
 * plain hex values so the contrast probe can compute ratios from them.
 */

export const THEMES = [
  { id: "white", label: "白色" }, { id: "black", label: "黑色" },
  { id: "navy", label: "深蓝色" }, { id: "burgundy", label: "深红色" },
] as const;
export type ThemeId = typeof THEMES[number]["id"];

interface Palette {
  /** Page backdrop. */
  bg: string;
  /** The single container layer (cards, sidebar, popovers). */
  surface: string;
  /** Hover / selected / input fill — one step off the surface. */
  surface2: string;
  /** Hairlines: container edges and separators. */
  line: string;
  /** Control borders: inputs, secondary buttons. */
  lineStrong: string;
  text: string;
  dim: string;
  /** Focus / realtime / primary action. */
  signal: string;
  /** Primary-button hover. */
  signalHover: string;
  /** Text drawn on top of a signal-filled button. */
  onSignal: string;
  /** Distraction / needs attention. */
  drift: string;
  /** Severe fault and destructive actions only. */
  alarm: string;
  /** "Everything is fine" — only the status capsule dot. Never focus. */
  ok: string;
  /** Historical and comparison series. */
  steel: string;
  /** Procrastination type colours, fixed slot order (§2.1). */
  types: [string, string, string, string, string];
  /** Inputs get a native colour scheme so autofill/scrollbars match. */
  light?: boolean;
}

const TYPES_DARK: [string, string, string, string, string] = [
  "#C9A15A", "#D9776F", "#6F9BD1", "#A68BD1", "#5FB3A3",
];
const TYPES_LIGHT: [string, string, string, string, string] = [
  "#8A6414", "#A8463E", "#2F64A3", "#6B4FA0", "#2A7A6C",
];

const palettes: Record<ThemeId, Palette> = {
  // Deep blue-grey — the design baseline. Surfaces separate by value only.
  navy: {
    bg: "#0A0E14", surface: "#121A24", surface2: "#18222E",
    line: "#1F2A36", lineStrong: "#2B3846",
    text: "#D7DEE6", dim: "#8E9BAA",
    signal: "#33B5E5", signalHover: "#5CC6EE", onSignal: "#0A0E14",
    drift: "#E0A458", alarm: "#FF5D5D", ok: "#5FBF8F", steel: "#5A7A9A",
    types: TYPES_DARK,
  },
  // Holo's original black: the surface ladder is value-only, the signal is cyan.
  black: {
    bg: "#000000", surface: "#0C0F10", surface2: "#171B1C",
    line: "#22282A", lineStrong: "#30383A",
    text: "#E8F7FA", dim: "#8A9A9D",
    signal: "#33B5E5", signalHover: "#5CC6EE", onSignal: "#0A0E14",
    drift: "#E8A951", alarm: "#FF5D5D", ok: "#5FBF8F", steel: "#5E7C86",
    types: TYPES_DARK,
  },
  // Light: every text pair clears 4.5:1 against bg / surface / surface-2
  // (the e2e contrast probe measures exactly those three).
  white: {
    bg: "#F5F7FA", surface: "#FFFFFF", surface2: "#EEF2F6",
    line: "#DDE3EA", lineStrong: "#C5CFDA",
    text: "#0F1A26", dim: "#56657A",
    signal: "#066A94", signalHover: "#055A7E", onSignal: "#FFFFFF",
    drift: "#94580A", alarm: "#C2312B", ok: "#1A7A50", steel: "#6A7F95",
    types: TYPES_LIGHT,
    light: true,
  },
  // Burgundy: same ladder as navy, warm hue. Signal stays cyan so "focus"
  // never changes meaning between themes.
  burgundy: {
    bg: "#140A0F", surface: "#1E1218", surface2: "#27171F",
    line: "#33212A", lineStrong: "#452E39",
    text: "#EEDFE5", dim: "#A9919B",
    signal: "#33B5E5", signalHover: "#5CC6EE", onSignal: "#0A0E14",
    drift: "#E0A458", alarm: "#FF6B6B", ok: "#6CC28F", steel: "#8A6A78",
    types: TYPES_DARK,
  },
};

/** `<meta name="theme-color">` follows the active backdrop so mobile chrome
 *  does not stay navy after switching to the white theme. */
function syncThemeColor(bg: string) {
  let meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (!meta) {
    meta = document.createElement("meta");
    meta.name = "theme-color";
    document.head.appendChild(meta);
  }
  meta.content = bg;
}

export function applyTheme(theme: ThemeId) {
  const p = palettes[theme]; const light = p.light === true;
  /** `color-mix` so a derived fill always tracks its base colour. */
  const mix = (color: string, percent: number) =>
    `color-mix(in srgb, ${color} ${percent}%, transparent)`;
  const variables: Record<string, string> = {
    /* ── §2.1 canonical tokens ─────────────────────────────────────────── */
    "bg": p.bg,
    "surface": p.surface,
    "surface-2": p.surface2,
    "line": p.line,
    "line-strong": p.lineStrong,
    "text": p.text,
    "text-dim": p.dim,
    "signal": p.signal,
    "signal-hover": p.signalHover,
    "on-signal": p.onSignal,
    "drift": p.drift,
    "alarm": p.alarm,
    "ok": p.ok,
    "steel": p.steel,

    /* ── Derived fills (§2.1) ──────────────────────────────────────────── */
    "signal-soft": mix(p.signal, 14),
    "drift-soft": mix(p.drift, 14),
    "alarm-soft": mix(p.alarm, 14),
    // The glow is the signature of the hero reading and the realtime dot: a
    // dark theme casts light, the light theme rings instead (a shadow would
    // read as a smudge on white).
    "signal-glow": light
      ? `0 0 0 3px color-mix(in srgb, ${p.signal} 16%, transparent)`
      : `0 0 12px color-mix(in srgb, ${p.signal} 35%, transparent)`,
    "focus-ring": `0 0 0 2px ${p.signal}`,

    /* ── Legacy hooks (§2.2): every existing page keeps working ─────────── */
    // Hex, not var(): the e2e contrast probe parses these five directly.
    "mf-text": p.text,
    "mf-text-dim": p.dim,
    "color-bg": p.bg,
    "mf-panel": p.surface,
    "color-bg-inset": p.surface2,

    "mf-bg": p.bg,
    "mf-surface": p.surface,
    "mf-surface-2": p.surface2,
    "mf-line": p.line,
    "mf-line-strong": p.lineStrong,
    "mf-signal": p.signal,
    "mf-drift": p.drift,
    "mf-alarm": p.alarm,
    "mf-steel": p.steel,
    "mf-bg-radial": p.bg,
    "mf-login-radial": p.bg,
    "mf-text-icon": p.dim,
    "mf-brand": p.signal,
    "mf-accent-line": p.line,
    "mf-panel-border": p.line,
    "mf-hairline": p.line,
    "mf-sidebar-bg": p.bg,
    "mf-topbar-bg": p.bg,

    "color-bg-elevated": p.surface,
    "color-border": p.line,
    "color-text-primary": p.text,
    "color-text-secondary": p.dim,
    "color-text-tertiary": p.dim,
    "color-primary": p.signal,
    "color-primary-hover": p.signalHover,
    "color-primary-light": p.surface2,
    "color-on-primary": p.onSignal,
    "color-success": p.ok,
    "color-warning": p.drift,
    "color-danger": p.alarm,
    "color-info": p.signal,

    // Semantic state + soft fills
    "good": p.ok, "good-soft": mix(p.ok, 14),
    "warn": p.drift, "warn-soft": mix(p.drift, 14),
    "bad": p.alarm, "bad-soft": mix(p.alarm, 14),
    "info": p.signal, "info-soft": mix(p.signal, 14),
    "neutral-soft": mix(p.dim, 12),

    // Data-viz — 专注 = signal, 历史/对照 = steel, 分心 = drift (§2.2)
    "chart-1": p.signal,
    "chart-2": p.drift,
    "chart-3": p.steel,
    "chart-4": p.types[3],
    "chart-5": p.types[1],
    "chart-6": p.dim,
    "chart-pos": p.signal, "chart-pos-soft": mix(p.signal, 24),
    "chart-neg": p.drift, "chart-neg-soft": mix(p.drift, 24),
    "chart-grid": p.line, "chart-axis": p.dim, "chart-track": p.line,
    "chart-tooltip-bg": p.surface, "chart-tooltip-border": p.lineStrong,

    // Procrastination types — fixed slot order.
    "type-task_aversion": p.types[0], "type-impulsivity": p.types[1],
    "type-decisional": p.types[2], "type-perfectionism": p.types[3],
    "type-emotional_regulation": p.types[4],

    // No shadows anywhere (§2.5): hierarchy is value + hairline.
    "shadow-1": "none",
    "shadow-2": "none",
    "shadow-focus": `0 0 0 2px ${p.signal}`,
    "card-highlight": "none",

    "mf-select-arrow": `url("data:image/svg+xml,${encodeURIComponent(`<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1 1l5 5 5-5' fill='none' stroke='${p.text}' stroke-width='1.6'/></svg>`)}")`,
  };
  Object.assign(variables, { "mf-green": p.ok, "mf-yellow": p.drift, "mf-orange": p.drift, "mf-cyan": p.signal, "mf-red": p.alarm });
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = light ? "light" : "dark";
  Object.entries(variables).forEach(([key, value]) => document.documentElement.style.setProperty(`--${key}`, value));
  syncThemeColor(p.bg);
}

export function readTheme(): ThemeId {
  try { const saved = localStorage.getItem("mindflow_theme"); if (THEMES.some(t => t.id === saved)) return saved as ThemeId; } catch { /* Storage can be disabled. */ }
  return "navy";
}
export function saveTheme(theme: ThemeId): void { applyTheme(theme); try { localStorage.setItem("mindflow_theme", theme); } catch { /* Theme still applies to this session. */ } }
