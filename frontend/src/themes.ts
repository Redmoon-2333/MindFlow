/**
 * Theme tokens — the single source of truth for every palette value.
 *
 * `applyTheme` writes custom properties as inline styles on <html>, which
 * outranks any stylesheet. theme.css therefore only carries navy fallbacks,
 * and page CSS must consume tokens rather than re-declaring colours.
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
  bg: string; panel: string; inset: string; text: string; dim: string;
  brand: string; border: string; primary: string;
  /** Chart series, ordered by hue distance so adjacent bars stay separable. */
  chart: [string, string, string, string, string, string];
  /** Signed contribution poles — one meaning per direction, per theme. */
  pos: string; neg: string;
  grid: string; axis: string; track: string; tooltipBg: string; tooltipBorder: string;
  good: string; warn: string; bad: string; info: string;
  types: [string, string, string, string, string];
  shadowStrength: number;
  /** Light surfaces need softer elevation and a darker highlight. */
  light?: boolean;
}

const palettes: Record<ThemeId, Palette> = {
  // Light theme: same hue family as navy, darkened until every pair clears
  // 4.5:1 on white. e2e computes these ratios from the computed values.
  white: {
    bg: "#f4f6fa", panel: "#ffffff", inset: "#eaeff7", text: "#101c2b", dim: "#4a5c72",
    brand: "#1f5fa0", border: "#c2cee0", primary: "#17548f",
    chart: ["#1f5fa0", "#0f7a63", "#8a5a05", "#6b34a8", "#a8323f", "#4a5a70"],
    pos: "#0b6b52", neg: "#a8302b",
    grid: "rgba(16, 28, 43, 0.12)", axis: "rgba(16, 28, 43, 0.48)",
    track: "rgba(16, 28, 43, 0.08)", tooltipBg: "#ffffff", tooltipBorder: "#c2cee0",
    good: "#0a6b52", warn: "#7a5200", bad: "#a8302b", info: "#1c4f96",
    types: ["#7a5200", "#a8302b", "#1c4f96", "#6b34a8", "#0a6b52"],
    shadowStrength: 0.1, light: true,
  },
  black: {
    bg: "#111111", panel: "#1c1c1c", inset: "#181818", text: "#f2f2f2", dim: "#bdbdbd",
    brand: "#8fb6ff", border: "#4a4a4a", primary: "#2367a8",
    chart: ["#8fb6ff", "#4bd3c0", "#f0b64e", "#c3a0ff", "#ff8f8f", "#9aa7b8"],
    pos: "#4bd3c0", neg: "#ff8f8f",
    grid: "rgba(242, 242, 242, 0.14)", axis: "rgba(242, 242, 242, 0.5)",
    track: "rgba(242, 242, 242, 0.1)", tooltipBg: "#1c1c1c", tooltipBorder: "#4a4a4a",
    good: "#4bd3c0", warn: "#f0b64e", bad: "#ff8f8f", info: "#8fb6ff",
    types: ["#f0b64e", "#ff8f8f", "#8fb6ff", "#c3a0ff", "#4bd3c0"],
    shadowStrength: 0.5,
  },
  navy: {
    bg: "#040b14", panel: "#172b46", inset: "#0c1a30", text: "#e0e9f4", dim: "#a9bdd3",
    brand: "#8dc7ff", border: "#38577e", primary: "#1761a8",
    chart: ["#7aa2ff", "#2dd4bf", "#f5b84b", "#c084fc", "#f87171", "#94a3b8"],
    pos: "#2dd4bf", neg: "#f87171",
    grid: "rgba(150, 185, 225, 0.16)", axis: "rgba(150, 185, 225, 0.55)",
    track: "rgba(150, 185, 225, 0.12)", tooltipBg: "#12253d", tooltipBorder: "rgba(150, 185, 225, 0.34)",
    good: "#2dd4bf", warn: "#f5b84b", bad: "#f87171", info: "#7aa2ff",
    types: ["#f5b84b", "#f87171", "#7aa2ff", "#c084fc", "#2dd4bf"],
    shadowStrength: 0.32,
  },
  burgundy: {
    bg: "#180c12", panel: "#2b1a23", inset: "#241520", text: "#f7eaf0", dim: "#cfb5c1",
    brand: "#ffb1c8", border: "#6d4a59", primary: "#9d2c4e",
    chart: ["#9fb6ff", "#4bd3c0", "#f5b84b", "#c9a8ff", "#ff9d94", "#b3a5ae"],
    pos: "#4bd3c0", neg: "#ff9d94",
    grid: "rgba(230, 200, 212, 0.16)", axis: "rgba(230, 200, 212, 0.52)",
    track: "rgba(230, 200, 212, 0.12)", tooltipBg: "#2b1a23", tooltipBorder: "rgba(230, 200, 212, 0.32)",
    good: "#4bd3c0", warn: "#f5b84b", bad: "#ff9d94", info: "#9fb6ff",
    types: ["#f5b84b", "#ff9d94", "#9fb6ff", "#c9a8ff", "#4bd3c0"],
    shadowStrength: 0.4,
  },
};

export function applyTheme(theme: ThemeId) {
  const p = palettes[theme]; const light = theme === "white";
  const s = p.shadowStrength;
  const variables: Record<string, string> = {
    // Flat surfaces only: the "calm instrument panel" direction bans ambient
    // gradients and glow, so both hooks collapse to the theme background.
    "mf-bg-radial": p.bg,
    "mf-login-radial": p.bg,
    "mf-panel": p.panel, "mf-panel-border": p.border, "mf-hairline": p.border,
    "mf-sidebar-bg": p.bg, "mf-topbar-bg": p.bg, "mf-text": p.text, "mf-text-dim": p.dim,
    "mf-text-icon": p.dim, "mf-brand": p.brand, "mf-accent-line": p.border,
    "color-bg": p.bg, "color-bg-elevated": p.panel, "color-bg-inset": p.inset, "color-border": p.border,
    "color-text-primary": p.text, "color-text-secondary": p.dim, "color-text-tertiary": p.dim,
    "color-primary": p.primary, "color-primary-hover": p.primary, "color-primary-light": p.inset,
    "color-on-primary": "#ffffff",
    "color-success": p.good, "color-warning": p.warn, "color-danger": p.bad, "color-info": p.info,

    // Semantic state + soft fills
    "good": p.good, "good-soft": `color-mix(in srgb, ${p.good} 14%, transparent)`,
    "warn": p.warn, "warn-soft": `color-mix(in srgb, ${p.warn} 14%, transparent)`,
    "bad": p.bad, "bad-soft": `color-mix(in srgb, ${p.bad} 14%, transparent)`,
    "info": p.info, "info-soft": `color-mix(in srgb, ${p.info} 14%, transparent)`,
    "neutral-soft": `color-mix(in srgb, ${p.dim} 12%, transparent)`,

    // Data-viz
    "chart-1": p.chart[0], "chart-2": p.chart[1], "chart-3": p.chart[2],
    "chart-4": p.chart[3], "chart-5": p.chart[4], "chart-6": p.chart[5],
    "chart-pos": p.pos, "chart-pos-soft": `color-mix(in srgb, ${p.pos} 24%, transparent)`,
    "chart-neg": p.neg, "chart-neg-soft": `color-mix(in srgb, ${p.neg} 24%, transparent)`,
    "chart-grid": p.grid, "chart-axis": p.axis, "chart-track": p.track,
    "chart-tooltip-bg": p.tooltipBg, "chart-tooltip-border": p.tooltipBorder,

    // Procrastination types — fixed slot order across every surface
    "type-task_aversion": p.types[0], "type-impulsivity": p.types[1],
    "type-decisional": p.types[2], "type-perfectionism": p.types[3],
    "type-emotional_regulation": p.types[4],

    // Elevation tuned per theme (a 0.42 black shadow is invisible on #040b14)
    "shadow-1": `0 1px 2px rgb(0 0 0 / ${(s * 0.5).toFixed(2)})`,
    "shadow-2": `0 8px 24px rgb(0 0 0 / ${(s * 0.85).toFixed(2)})`,
    "shadow-focus": `0 0 0 2px color-mix(in srgb, ${p.brand} 45%, transparent)`,
    // The one permitted flourish: a 1px top highlight, not a glow.
    "card-highlight": `inset 0 1px 0 rgb(255 255 255 / ${light ? 0.7 : 0.06})`,

    "mf-select-arrow": `url("data:image/svg+xml,${encodeURIComponent(`<svg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'><path d='M1 1l5 5 5-5' fill='none' stroke='${p.text}' stroke-width='1.6'/></svg>`)}")`,
  };
  Object.assign(variables, { "mf-green": variables["color-success"], "mf-yellow": variables["color-warning"], "mf-orange": variables["color-warning"], "mf-cyan": variables["color-info"], "mf-red": variables["color-danger"] });
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = light ? "light" : "dark";
  Object.entries(variables).forEach(([key, value]) => document.documentElement.style.setProperty(`--${key}`, value));
}

export function readTheme(): ThemeId {
  try { const saved = localStorage.getItem("mindflow_theme"); if (THEMES.some(t => t.id === saved)) return saved as ThemeId; } catch { /* Storage can be disabled. */ }
  return "navy";
}
export function saveTheme(theme: ThemeId): void { applyTheme(theme); try { localStorage.setItem("mindflow_theme", theme); } catch { /* Theme still applies to this session. */ } }
