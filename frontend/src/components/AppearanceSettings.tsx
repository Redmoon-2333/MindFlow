import { useState } from "react";
import { THEMES, readTheme, saveTheme, type ThemeId } from "../themes";

export default function AppearanceSettings() {
  const [theme, setTheme] = useState<ThemeId>(readTheme);
  return <div className="card mb24"><h3>外观主题</h3><p className="workspace-note">选择整体配色。主题立即生效，并保留在当前浏览器中。</p>
    <div className="theme-options" role="group" aria-label="外观主题">{THEMES.map(item => <button type="button" className={`theme-option theme-${item.id}`} key={item.id} aria-pressed={theme === item.id} onClick={() => { saveTheme(item.id); setTheme(item.id); }}><span aria-hidden="true" className="theme-swatch" />{item.label}{theme === item.id && <span aria-hidden="true">✓</span>}</button>)}</div>
  </div>;
}
