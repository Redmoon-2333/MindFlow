import type { ReactNode } from "react";
import "./primitives.css";

export type StatTone = "neutral" | "good" | "warn" | "bad" | "brand";

interface StatTileProps {
  label: string;
  /** Pre-formatted value — the tile never formats data itself. */
  value: ReactNode;
  unit?: string;
  /** One short line of context (delta, status, source). */
  hint?: ReactNode;
  tone?: StatTone;
  /** Optional trailing icon (iconfont class). */
  icon?: string;
}

/**
 * A KPI tile. Numerals use tabular figures so a row of tiles lines up column
 * to column; `value="—"` is a legitimate state (unknown), never a fake zero.
 */
export default function StatTile({ label, value, unit, hint, tone = "neutral", icon }: StatTileProps) {
  return (
    <div className="stat-tile" data-tone={tone}>
      <div className="stat-tile-head">
        <span className="stat-tile-label">{label}</span>
        {icon && <span className={`iconfont ${icon} stat-tile-icon`} aria-hidden="true" />}
      </div>
      <div className="stat-tile-value-row">
        <span className="stat-tile-value">{value}</span>
        {unit && <span className="stat-tile-unit">{unit}</span>}
      </div>
      {hint && <div className="stat-tile-hint">{hint}</div>}
    </div>
  );
}
