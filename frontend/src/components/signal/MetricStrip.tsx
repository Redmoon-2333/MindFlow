import type { CSSProperties } from "react";
import HelpTip from "./HelpTip";
import "./signal.css";

export type MetricTone = "up" | "down" | "flat" | "none";

export interface MetricItem {
  key: string;
  label: string;
  value: string;
  unit?: string;
  /** Day-over-day comparison line; `none` renders "较昨日 —". */
  delta?: { text: string; tone: MetricTone };
  /** Jargon definition shown as a "?" after the label (§8.17.1). */
  help?: { what: string; effect?: string };
}

interface MetricStripProps {
  /** Leading label column ("今天"); omitted for a bare strip. */
  label?: string;
  items: MetricItem[];
  /** A line under the strip — e.g. a trend-request failure note. */
  note?: string;
}

/**
 * One container, vertical hairlines between cells (spec §3.3).
 *
 * Four numbers side by side used to be four boxes, each with its own border,
 * radius and padding — four competing frames for one row of facts. A single
 * surface with rules between the columns reads as a table and keeps the
 * numbers comparable.
 */
export default function MetricStrip({ label, items, note }: MetricStripProps) {
  const columns = { "--sig-metric-cols": items.length } as CSSProperties;
  return (
    <div className={`sig-surface sig-metrics${label ? "" : " sig-metrics--bare"}`} style={columns}>
      {label && <div className="sig-metrics-label">{label}</div>}
      {items.map((item) => (
        <div className="sig-metric" key={item.key}>
          <div className="sig-metric-label">
            {item.label}
            {item.help && <HelpTip term={item.label} what={item.help.what} effect={item.help.effect} />}
          </div>
          <div className="sig-metric-value">
            {item.value}
            {item.unit && <span className="sig-metric-unit">{item.unit}</span>}
          </div>
          {item.delta && (
            <div className="sig-metric-delta" data-tone={item.delta.tone}>{item.delta.text}</div>
          )}
        </div>
      ))}
      {note && <div className="sig-meter-note">{note}</div>}
    </div>
  );
}
