export const THEMES = [
  { id: "white", label: "白色" }, { id: "black", label: "黑色" },
  { id: "navy", label: "深蓝色" }, { id: "burgundy", label: "深红色" },
] as const;
export type ThemeId = typeof THEMES[number]["id"];
const palettes: Record<ThemeId, { bg: string; panel: string; inset: string; text: string; dim: string; brand: string; border: string; primary: string }> = {
  white: { bg: "#f3f5f9", panel: "#ffffff", inset: "#e9eef5", text: "#172537", dim: "#485b70", brand: "#225fa8", border: "#bdcbdc", primary: "#1761a8" },
  black: { bg: "#111111", panel: "#222222", inset: "#181818", text: "#f2f2f2", dim: "#bdbdbd", brand: "#a3caff", border: "#555555", primary: "#2367a8" },
  navy: { bg: "#040b14", panel: "#172b46", inset: "#0c1a30", text: "#e0e9f4", dim: "#a9bdd3", brand: "#8dc7ff", border: "#38577e", primary: "#1761a8" },
  burgundy: { bg: "#180c12", panel: "#35212b", inset: "#271720", text: "#f7eaf0", dim: "#cfb5c1", brand: "#ffb1c8", border: "#755061", primary: "#9d2c4e" },
};
export function applyTheme(theme: ThemeId) {
  const p = palettes[theme]; const light = theme === "white";
  const variables: Record<string, string> = {
    "mf-bg-radial": `radial-gradient(ellipse at 30% 40%, ${p.inset}, ${p.bg})`,
    "mf-login-radial": `radial-gradient(ellipse at 30% 40%, ${p.inset}, ${p.bg})`,
    "mf-panel": p.panel, "mf-panel-border": p.border, "mf-hairline": p.border,
    "mf-sidebar-bg": p.bg, "mf-topbar-bg": p.bg, "mf-text": p.text, "mf-text-dim": p.dim,
    "mf-text-icon": p.dim, "mf-brand": p.brand, "mf-accent-line": p.border,
    "color-bg": p.bg, "color-bg-elevated": p.panel, "color-bg-inset": p.inset, "color-border": p.border,
    "color-text-primary": p.text, "color-text-secondary": p.dim, "color-text-tertiary": p.dim,
    "color-primary": p.primary, "color-primary-hover": p.primary, "color-primary-light": p.inset,
    "color-on-primary": "#ffffff", "color-success": light ? "#086b37" : "#85e6a4",
    "color-warning": light ? "#815200" : "#ffd176", "color-danger": light ? "#ab2438" : "#ff9da9",
    "color-info": light ? "#05647b" : "#85dcea",
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
export function saveTheme(theme: ThemeId) { applyTheme(theme); try { localStorage.setItem("mindflow_theme", theme); } catch { /* Theme still applies to this session. */ } }
