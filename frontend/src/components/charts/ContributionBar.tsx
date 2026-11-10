import { useId, useState } from "react";
import "./charts.css";
import { useElementWidth } from "./useElementWidth";
import { clamp, formatSigned } from "./scale";

export interface ContributionDatum {
  /** V2 feature key. */
  feature: string;
  /** Chinese label shown on the row. */
  label: string;
  /** Signed contribution: > 0 toward focus, < 0 toward distraction. */
  contribution: number;
  /** Current value on the raw scale (tooltip). */
  value?: number | null;
  /** Personal baseline (tooltip). */
  baselineValue?: number | null;
}

interface ContributionBarProps {
  data: ContributionDatum[];
  ariaLabel: string;
  /** Output space of the contributions, shown in the caption. */
  outputSpace?: "probability" | "logit";
  /** Model expected output, drawn as a caption note. */
  baseValue?: number | null;
  emptyMessage?: string;
  maxRows?: number;
}

const ROW_HEIGHT = 30;
const BAR_HEIGHT = 14;

/**
 * Signed horizontal contribution bars — the signature "evidence ruler".
 *
 * A shared zero axis sits left of centre; bars grow right (toward focus) or
 * left (toward distraction). Both the colour AND the side carry the sign, so
 * the chart survives colour-blindness and greyscale printing — the review a
 * defence demo actually gets.
 */
export default function ContributionBar({
  data,
  ariaLabel,
  outputSpace = "probability",
  baseValue = null,
  emptyMessage = "暂无归因数据",
  maxRows = 8,
}: ContributionBarProps) {
  const [ref, width] = useElementWidth<HTMLDivElement>(520);
  const [active, setActive] = useState<string | null>(null);
  const titleId = useId();

  const rows = data.slice(0, maxRows);
  if (rows.length === 0) {
    return <div className="mf-chart-empty">{emptyMessage}</div>;
  }

  const compact = width < 420;
  const labelWidth = compact ? 88 : 116;
  // Right gutter holds every value label. Drawing them at the bar's end looks
  // tighter, but a long negative bar pushes its label into the name column —
  // unreadable at 375px. A fixed column also makes the numbers comparable.
  const padRight = compact ? 44 : 56;
  const top = compact ? 4 : 18;
  const height = top + rows.length * ROW_HEIGHT + 4;
  const plotLeft = labelWidth;
  const plotRight = Math.max(plotLeft + 40, width - padRight);
  const plotWidth = plotRight - plotLeft;

  const maxPos = Math.max(0, ...rows.map((row) => row.contribution));
  const maxNeg = Math.max(0, ...rows.map((row) => -row.contribution));
  const total = maxPos + maxNeg;

  // Zero axis placement: the widest negative bar owns the left pane, so a
  // lopsided explanation still uses the full width. Clamped to [34%, 62%] so
  // the + / − boundary stays visually stable — a jumping axis reads as a
  // different chart on every refresh.
  const negShare = total > 0 ? maxNeg / total : 0.5;
  const zeroFrac = clamp(negShare, 0.34, 0.62);
  const zeroX = plotLeft + plotWidth * zeroFrac;
  const pxPerUnitNeg = total > 0 ? (plotWidth * zeroFrac) / Math.max(maxNeg, 1e-9) : 0;
  const pxPerUnitPos = total > 0 ? (plotWidth * (1 - zeroFrac)) / Math.max(maxPos, 1e-9) : 0;

  const activeRow = rows.find((row) => row.feature === active) ?? null;

  return (
    <figure className="mf-chart" ref={ref} style={{ position: "relative" }}>
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
            if (event.key === "Escape") setActive(null);
          }}
        >
          <title id={titleId}>{ariaLabel}</title>
          <desc>
            {rows
              .map((row) => `${row.label} ${formatSigned(row.contribution)}（${row.contribution >= 0 ? "推向专注" : "推向分心"}）`)
              .join("；")}
          </desc>

          {/* On a phone the two pole labels collide; the caption below still
              states the direction rule, so they are dropped there. */}
          {!compact && (
            <>
              <text className="mf-chart-pole" x={plotLeft} y={9} textAnchor="start">← 推向分心</text>
              <text className="mf-chart-pole" x={plotRight} y={9} textAnchor="end">推向专注 →</text>
            </>
          )}

          <line className="mf-chart-zero" x1={zeroX} x2={zeroX} y1={top - 6} y2={height} />

          {rows.map((row, index) => {
            const y = top + index * ROW_HEIGHT;
            const barY = y + ROW_HEIGHT / 2 - BAR_HEIGHT / 2;
            const positive = row.contribution >= 0;
            const magnitude = Math.abs(row.contribution);
            const rawWidth = magnitude * (positive ? pxPerUnitPos : pxPerUnitNeg);
            const barWidth = magnitude > 0 ? Math.max(2, rawWidth) : 0;
            const x = positive ? zeroX : zeroX - barWidth;
            const isActive = active === row.feature;
            const dim = active !== null && !isActive;
            const valueX = width - 2;

            return (
              <g key={row.feature}>
                <text className="mf-chart-row-label" x={0} y={y + ROW_HEIGHT / 2}>
                  {truncate(row.label, compact ? 7 : 10)}
                </text>
                <rect
                  className={`mf-chart-bar${dim ? " mf-chart-bar--dim" : ""}${isActive ? " mf-chart-bar--active" : ""}`}
                  x={x}
                  y={barY}
                  width={barWidth}
                  height={BAR_HEIGHT}
                  rx={4}
                  fill={positive ? "var(--signal)" : "var(--drift)"}
                />
                <text
                  className="mf-chart-row-value"
                  x={valueX}
                  y={y + ROW_HEIGHT / 2}
                  textAnchor="end"
                  fill="var(--text)"
                >
                  {formatSigned(row.contribution, 2)}
                </text>
                <rect
                  className="mf-chart-hit"
                  x={0}
                  y={y}
                  width={width}
                  height={ROW_HEIGHT}
                  onMouseEnter={() => setActive(row.feature)}
                  onMouseLeave={() => setActive(null)}
                />
              </g>
            );
          })}
        </svg>

        {activeRow && (
          <div
            className="mf-chart-tooltip"
            style={{
              left: clamp(activeRow.contribution >= 0 ? zeroX + 20 : zeroX - 20, 90, Math.max(90, width - 90)),
              top: top + rows.indexOf(activeRow) * ROW_HEIGHT - 6,
            }}
            role="status"
          >
            <div className="mf-chart-tooltip-title">{activeRow.label}</div>
            <div className="mf-chart-tooltip-row">
              <span className="mf-chart-tooltip-key">贡献</span>
              <span>{formatSigned(activeRow.contribution, 3)}</span>
            </div>
            {activeRow.value != null && (
              <div className="mf-chart-tooltip-row">
                <span className="mf-chart-tooltip-key">当前值</span>
                <span>{activeRow.value.toFixed(3)}</span>
              </div>
            )}
            <div className="mf-chart-tooltip-row">
              <span className="mf-chart-tooltip-key">个人基线</span>
              <span>{activeRow.baselineValue != null ? activeRow.baselineValue.toFixed(3) : "暂无基线"}</span>
            </div>
            <div className="mf-chart-tooltip-note">
              {activeRow.contribution >= 0 ? "推向专注" : "推向分心"}
            </div>
          </div>
        )}
      </div>

      <figcaption>
        {/* Direction is already carried by the two pole labels and by the
            backend's own `note`; repeating "正值把预测推向专注" here printed the
            same sentence twice in one card. The caption keeps only what is
            unique to the drawing: the baseline and what length encodes. */}
        {baseValue != null && `模型期望输出 ${baseValue.toFixed(3)}（${outputSpace === "logit" ? "logit" : "概率"}）· `}
        条形长度表示影响大小。
      </figcaption>
    </figure>
  );
}

function truncate(text: string, limit: number): string {
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}
