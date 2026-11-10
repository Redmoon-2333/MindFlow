import { useId, useState } from "react";
import "./charts.css";
import { useElementWidth } from "./useElementWidth";
import { clamp, formatCompact, niceTicks } from "./scale";

/** Approximate rendered width of a tick label at 12px: CJK glyphs are ~12px,
 *  Latin/digits ~6.6px. Used only to decide how many labels to skip. */
function textWidth(value: string): number {
  let total = 0;
  for (const ch of value) total += ch.codePointAt(0)! >= 0x2e80 ? 12 : 6.6;
  return total;
}


export interface StackedSeries {
  key: string;
  label: string;
  /** CSS colour (usually a --chart-* token). */
  color: string;
}

export interface StackedDatum {
  key: string;
  label: string;
  title?: string;
  /** One entry per series, by series key. Missing keys count as 0. */
  values: Record<string, number>;
  highlight?: boolean;
}

interface StackedBarProps {
  data: StackedDatum[];
  series: StackedSeries[];
  ariaLabel: string;
  unit?: string;
  height?: number;
  yAxisLabel?: string;
  emptyMessage?: string;
  /** Show the series legend beneath the plot. */
  showLegend?: boolean;
}

/**
 * Stacked vertical bars for two-part compositions (专注 / 分心 per day).
 *
 * Segments are drawn from the baseline up in series order, so the colour
 * nearest the axis is always the same meaning. Totals appear above the stack.
 */
export default function StackedBar({
  data,
  series,
  ariaLabel,
  unit = "",
  height = 220,
  yAxisLabel,
  emptyMessage = "暂无数据",
  showLegend = true,
}: StackedBarProps) {
  const [ref, width] = useElementWidth<HTMLDivElement>(560);
  const [active, setActive] = useState<number | null>(null);
  const titleId = useId();

  if (data.length === 0) {
    return <div className="mf-chart-empty">{emptyMessage}</div>;
  }

  const compact = width < 420;
  const padLeft = compact ? 34 : 44;
  const padRight = 8;
  const padTop = 18;
  const padBottom = compact ? 26 : 30;
  const axisBand = yAxisLabel ? 14 : 0;
  const plotLeft = padLeft + axisBand;
  const plotWidth = clamp(width - plotLeft - padRight, 40, 4000);
  const plotHeight = Math.max(60, height - padTop - padBottom);

  const totals = data.map((datum) =>
    series.reduce((sum, entry) => sum + Math.max(0, datum.values[entry.key] ?? 0), 0),
  );
  const { top, ticks } = niceTicks(Math.max(0, ...totals));
  const scaleY = (value: number) => plotHeight - (clamp(value, 0, top) / top) * plotHeight;

  const slot = plotWidth / data.length;
  const barWidth = Math.max(4, Math.min(slot * 0.618, 64));
  // Same width-aware stride as BarChart (§8.19).
  const maxLabelWidth = data.reduce((widest, datum) => Math.max(widest, textWidth(datum.label)), 0);
  const labelStride = Math.max(1, Math.ceil((maxLabelWidth + 8) / slot));

  return (
    <figure className="mf-chart" ref={ref}>
      <div style={{ position: "relative" }}>
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={ariaLabel}
          aria-describedby={titleId}
          tabIndex={0}
          onBlur={() => setActive(null)}
          onKeyDown={(event) => {
            if (event.key === "ArrowRight") {
              event.preventDefault();
              setActive((current) => (current === null ? 0 : Math.min(data.length - 1, current + 1)));
            } else if (event.key === "ArrowLeft") {
              event.preventDefault();
              setActive((current) => (current === null ? data.length - 1 : Math.max(0, current - 1)));
            } else if (event.key === "Escape") {
              setActive(null);
            }
          }}
        >
          <title id={titleId}>{ariaLabel}</title>
          <desc>
            {data
              .map((datum) => `${datum.label}：${series.map((entry) => `${entry.label} ${formatCompact(datum.values[entry.key] ?? 0)}${unit}`).join("，")}`)
              .join("；")}
          </desc>

          {ticks.map((tick) => {
            const y = padTop + scaleY(tick);
            return (
              <g key={tick}>
                <line
                  className={tick === 0 ? "mf-chart-baseline" : "mf-chart-grid"}
                  x1={plotLeft}
                  x2={plotLeft + plotWidth}
                  y1={y}
                  y2={y}
                />
                <text className="mf-chart-tick mf-chart-tick--y" x={plotLeft - 8} y={y + 4}>
                  {formatCompact(tick)}
                </text>
              </g>
            );
          })}

          {yAxisLabel && (
            <text
              className="mf-chart-axis-title"
              transform={`translate(${padLeft - 24} ${padTop + plotHeight / 2}) rotate(-90)`}
              textAnchor="middle"
            >
              {yAxisLabel}
            </text>
          )}

          {data.map((datum, index) => {
            const x = plotLeft + slot * index + (slot - barWidth) / 2;
            let cursor = 0;
            const dim = active !== null && active !== index;
            return (
              <g key={datum.key}>
                {series.map((entry, seriesIndex) => {
                  const value = Math.max(0, datum.values[entry.key] ?? 0);
                  if (value <= 0) return null;
                  const yBottom = padTop + scaleY(cursor);
                  cursor += value;
                  const yTop = padTop + scaleY(cursor);
                  const segmentHeight = Math.max(1, yBottom - yTop);
                  const isTop = !series.slice(seriesIndex + 1).some((next) => (datum.values[next.key] ?? 0) > 0);
                  return (
                    <rect
                      key={entry.key}
                      className={`mf-chart-bar${dim ? " mf-chart-bar--dim" : ""}`}
                      x={x}
                      y={yTop}
                      width={barWidth}
                      height={segmentHeight}
                      rx={isTop ? Math.min(5, barWidth / 3) : 0}
                      fill={entry.color}
                      data-series={entry.key}
                      data-value={value}
                    />
                  );
                })}
                {slot >= 26 && totals[index] > 0 && (
                  <text className="mf-chart-value-label" x={x + barWidth / 2} y={padTop + scaleY(totals[index]) - 6}>
                    {formatCompact(totals[index])}
                  </text>
                )}
                {index % labelStride === 0 && (
                  <text
                    className="mf-chart-tick mf-chart-tick--x"
                    x={plotLeft + slot * index + slot / 2}
                    y={height - padBottom + 16}
                  >
                    {datum.label}
                  </text>
                )}
                <rect
                  className="mf-chart-hit"
                  x={plotLeft + slot * index}
                  y={padTop}
                  width={slot}
                  height={plotHeight}
                  onMouseEnter={() => setActive(index)}
                  onMouseLeave={() => setActive(null)}
                />
              </g>
            );
          })}
        </svg>

        {active !== null && (
          <div
            className="mf-chart-tooltip"
            style={{
              left: clamp(plotLeft + slot * active + slot / 2, 70, Math.max(70, width - 70)),
              top: padTop + 8,
            }}
            role="status"
          >
            <div className="mf-chart-tooltip-title">{data[active].title || data[active].label}</div>
            {series.map((entry) => (
              <div className="mf-chart-tooltip-row" key={entry.key}>
                <span className="mf-chart-tooltip-key">{entry.label}</span>
                <span>
                  {formatCompact(data[active].values[entry.key] ?? 0)}
                  {unit}
                </span>
              </div>
            ))}
            <div className="mf-chart-tooltip-row mf-chart-tooltip-note">
              <span className="mf-chart-tooltip-key">合计</span>
              <span>
                {formatCompact(totals[active])}
                {unit}
              </span>
            </div>
          </div>
        )}
      </div>

      {showLegend && (
        <div className="mf-chart-legend">
          {series.map((entry) => (
            <span className="mf-chart-legend-item" key={entry.key}>
              <span className="mf-chart-legend-swatch" style={{ background: entry.color }} aria-hidden="true" />
              {entry.label}
            </span>
          ))}
        </div>
      )}
    </figure>
  );
}
