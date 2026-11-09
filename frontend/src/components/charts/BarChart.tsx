import { useId, useState } from "react";
import "./charts.css";
import { useElementWidth } from "./useElementWidth";
import { clamp, formatCompact, niceTicks } from "./scale";

export interface BarDatum {
  /** Stable key (date, hour bucket, …). */
  key: string;
  /** Axis label; keep it ≤ 5 characters for the x tick. */
  label: string;
  value: number;
  /** Tooltip title; falls back to `label`. */
  title?: string;
  /** Draw with the "current period" accent (today / this bucket). */
  highlight?: boolean;
  /** Extra tooltip rows, e.g. 专注 42m / 分心 8m. */
  detail?: Array<{ label: string; value: string }>;
}

interface BarChartProps {
  data: BarDatum[];
  /** One sentence describing what the reader is looking at; also the a11y label. */
  ariaLabel: string;
  unit?: string;
  /** Bar colour token; defaults to --chart-1. */
  color?: string;
  highlightColor?: string;
  /** Per-bar colours when a series is categorical. */
  colorFor?: (datum: BarDatum) => string;
  height?: number;
  /** Draw the numeric value above each bar. Off automatically when bars < 6px. */
  showValues?: boolean;
  /** Y-axis title (rotated). */
  yAxisLabel?: string;
  emptyMessage?: string;
  /** Rounded to a nice axis bound instead of the raw max. */
  forceZero?: boolean;
  /**
   * Render the x tick labels as HTML spans under the plot instead of SVG text.
   * HTML labels wrap and stay selectable, which matters for long weekday/date
   * captions; the class name is supplied by the caller so the page keeps its
   * own label styling (and any selector an e2e spec already relies on).
   */
  htmlXLabelClass?: string;
}

/**
 * Vertical bar chart with a real y axis, gridlines and a hover/focus tooltip.
 *
 * Accessibility: the SVG is `role="img"` with `aria-label`, plus a `<title>`
 * and a visually-hidden table-free text summary in the figcaption. Hover and
 * keyboard focus both drive the same tooltip, so nothing is mouse-only.
 */
export default function BarChart({
  data,
  ariaLabel,
  unit = "",
  color = "var(--chart-1)",
  highlightColor = "var(--chart-2)",
  colorFor,
  height = 220,
  showValues = true,
  yAxisLabel,
  emptyMessage = "暂无数据",
  forceZero = true,
  htmlXLabelClass,
}: BarChartProps) {
  const [ref, width] = useElementWidth<HTMLDivElement>(560);
  const [active, setActive] = useState<number | null>(null);
  const titleId = useId();

  if (data.length === 0) {
    return <div className="mf-chart-empty">{emptyMessage}</div>;
  }

  const compact = width < 420;
  const padLeft = compact ? 34 : 44;
  const padRight = 8;
  const padTop = 16;
  const padBottom = compact ? 26 : 30;
  const axisBand = yAxisLabel ? 14 : 0;
  const plotLeft = padLeft + axisBand;
  const plotWidth = clamp(width - plotLeft - padRight, 40, 4000);
  const plotHeight = Math.max(60, height - padTop - padBottom);

  const max = Math.max(0, ...data.map((d) => (Number.isFinite(d.value) ? d.value : 0)));
  const { top, ticks } = forceZero ? niceTicks(max) : niceTicks(max);
  const scaleY = (value: number) => plotHeight - (clamp(value, 0, top) / top) * plotHeight;

  const slot = plotWidth / data.length;
  // 61.8% of the slot reads as a bar, not a wall; the cap stops 2-bar charts
  // from becoming 200px-wide slabs.
  const barWidth = Math.max(4, Math.min(slot * 0.618, 72));
  const valuesFit = showValues && slot >= 26;

  const labelStride = slot < 26 ? Math.ceil(26 / slot) : 1;

  const tooltip = active === null ? null : data[active];
  const tooltipLeft = active === null ? 0 : plotLeft + slot * active + slot / 2;

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
          <desc>{data.map((d) => `${d.label} ${formatCompact(d.value)}${unit}`).join("；")}</desc>

          {/* Gridlines + y ticks */}
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
            const value = Number.isFinite(datum.value) ? datum.value : 0;
            const x = plotLeft + slot * index + (slot - barWidth) / 2;
            const y = padTop + scaleY(value);
            const barHeight = Math.max(value > 0 ? 2 : 0, padTop + plotHeight - y);
            const fill = colorFor ? colorFor(datum) : datum.highlight ? highlightColor : color;
            const dim = active !== null && active !== index;
            return (
              <g key={datum.key}>
                {barHeight > 0 && (
                  <rect
                    className={`mf-chart-bar${dim ? " mf-chart-bar--dim" : ""}${active === index ? " mf-chart-bar--active" : ""}`}
                    x={x}
                    y={y}
                    width={barWidth}
                    height={barHeight}
                    rx={Math.min(6, barWidth / 3)}
                    fill={fill}
                  />
                )}
                {valuesFit && value > 0 && (
                  <text className="mf-chart-value-label" x={x + barWidth / 2} y={y - 6}>
                    {formatCompact(value)}
                  </text>
                )}
                {/* X tick — HTML labels replace these when requested. */}
                {!htmlXLabelClass && index % labelStride === 0 && (
                  <text
                    className="mf-chart-tick mf-chart-tick--x"
                    x={plotLeft + slot * index + slot / 2}
                    y={height - padBottom + 16}
                  >
                    {datum.label}
                  </text>
                )}
                {/* Hit area covers the whole slot and full height */}
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

        {htmlXLabelClass && (
          <div
            className="mf-chart-html-ticks"
            aria-hidden="true"
            style={{ left: plotLeft, right: padRight, height: padBottom - 6 }}
          >
            {data.map((datum, index) => (
              <span
                key={datum.key}
                className={htmlXLabelClass}
                style={{ width: `${100 / data.length}%` }}
                data-active={active === index ? "true" : undefined}
              >
                {index % labelStride === 0 ? datum.label : ""}
              </span>
            ))}
          </div>
        )}

        {tooltip && (
          <div
            className="mf-chart-tooltip"
            style={{ left: clamp(tooltipLeft, 70, Math.max(70, width - 70)), top: padTop + 8 }}
            role="status"
          >
            <div className="mf-chart-tooltip-title">{tooltip.title || tooltip.label}</div>
            <div className="mf-chart-tooltip-row">
              <span className="mf-chart-tooltip-key">{yAxisLabel || "数值"}</span>
              <span>
                {formatCompact(tooltip.value)}
                {unit}
              </span>
            </div>
            {tooltip.detail?.map((row) => (
              <div className="mf-chart-tooltip-row" key={row.label}>
                <span className="mf-chart-tooltip-key">{row.label}</span>
                <span>{row.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}
