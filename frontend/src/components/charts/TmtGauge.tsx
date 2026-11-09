import type { TmtBreakdown, TmtComponentKey } from "../../types/explainability";
import { MOTIVATION_LEVEL_LABELS } from "../../types/explainability";
import "./charts.css";
import { clamp } from "./scale";

const ORDER: TmtComponentKey[] = ["expectancy", "value", "impulsiveness", "delay"];

const SHORT: Record<TmtComponentKey, string> = {
  expectancy: "E 期望",
  value: "V 价值",
  impulsiveness: "I 冲动",
  delay: "D 延迟",
};

const BARRIER_COPY: Record<TmtComponentKey, string> = {
  expectancy: "成功预期是主要瓶颈",
  value: "任务价值感是主要瓶颈",
  impulsiveness: "冲动干扰是主要瓶颈",
  delay: "截止时间压力是主要瓶颈",
};

const LEVEL_COLOR: Record<TmtBreakdown["motivation_level"], string> = {
  low: "var(--bad)",
  medium: "var(--warn)",
  high: "var(--good)",
};

interface TmtGaugeProps {
  tmt: TmtBreakdown | null | undefined;
  /** Hide the boundary note when the parent already prints it. */
  showNote?: boolean;
}

/**
 * Temporal Motivation Theory readout: Motivation = E·V / (1 + I·D).
 *
 * Four component bars (each a *barrier-raising* factor for I and D, a
 * *driver* for E and V) plus the resolved motivation value and the dominant
 * barrier. The formula is printed rather than implied — a judge should be
 * able to check the arithmetic on screen.
 */
export default function TmtGauge({ tmt, showNote = true }: TmtGaugeProps) {
  if (!tmt) {
    return <div className="mf-chart-empty">暂无动机分解数据</div>;
  }

  const motivation = clamp(tmt.motivation, 0, 1);
  const level = tmt.motivation_level;
  const dominant = tmt.dominant_barrier;

  return (
    <div className="mf-tmt">
      <div className="mf-tmt-formula" aria-hidden="true">Motivation = E · V / (1 + I · D)</div>

      <div className="mf-tmt-headline">
        <span className="mf-tmt-value" style={{ color: LEVEL_COLOR[level] }} data-numeric>
          {motivation.toFixed(2)}
        </span>
        <span className={`badge ${level === "high" ? "badge-success" : level === "low" ? "badge-danger" : "badge-warning"}`}>
          {MOTIVATION_LEVEL_LABELS[level]}
        </span>
        <span className="text-muted" style={{ fontSize: "var(--fs-xs)" }}>
          主要瓶颈：{SHORT[dominant]} · {BARRIER_COPY[dominant]}
        </span>
      </div>

      <div className="mf-tmt-components">
        {ORDER.map((key) => {
          const component = tmt.components[key];
          const score = clamp(component.score, 0, 1);
          const isDominant = key === dominant;
          // E and V are drivers (higher = better); I and D are barriers
          // (higher = worse). Colour therefore follows meaning, not sign.
          // The dominant term keeps its meaning colour and is *outlined* in
          // amber, so "this is the bottleneck" never masks what it measures.
          const driver = key === "expectancy" || key === "value";
          const fill = driver ? "var(--good)" : "var(--bad)";
          return (
            <div className="mf-tmt-component" key={key} data-dominant={isDominant}>
              <span className="mf-tmt-component-label">
                {SHORT[key]}
                {isDominant && <span className="badge badge-warning" style={{ fontSize: 10, padding: "0 6px" }}>瓶颈</span>}
              </span>
              <div
                className="mf-tmt-track"
                role="img"
                aria-label={`${component.label || SHORT[key]} ${Math.round(score * 100)}%${isDominant ? "（主要瓶颈）" : ""}`}
              >
                <div className="mf-tmt-fill" style={{ width: `${score * 100}%`, background: fill }} />
              </div>
              <span className="mf-tmt-component-value">{score.toFixed(2)}</span>
            </div>
          );
        })}
      </div>

      <div className="mf-chart-legend" aria-hidden="true">
        <span className="mf-chart-legend-item"><span className="mf-chart-legend-swatch" style={{ background: "var(--good)" }} />E/V 越高越好</span>
        <span className="mf-chart-legend-item"><span className="mf-chart-legend-swatch" style={{ background: "var(--bad)" }} />I/D 越高越阻碍</span>
        <span className="mf-chart-legend-item"><span className="mf-chart-legend-swatch" style={{ background: "transparent", boxShadow: "inset 0 0 0 2px var(--warn)" }} />描边 = 主要瓶颈</span>
      </div>

      {showNote && (tmt.note || true) && (
        <div className="mf-tmt-note">
          {tmt.note || "由行为代理变量估算，不是心理测量。"}
        </div>
      )}
    </div>
  );
}
