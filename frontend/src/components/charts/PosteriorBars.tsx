import type { RationaleCandidate } from "../../types/explainability";
import { betaInterval } from "../../types/explainability";
import "./charts.css";
import { clamp } from "./scale";

interface PosteriorBarsProps {
  candidates: RationaleCandidate[];
  ariaLabel?: string;
}

/**
 * Per-candidate Beta posterior mean with a 90% credible interval.
 *
 * The dot is the posterior mean (the number decisions actually use); the thin
 * bar behind it is the interval, which is the visual argument for *why* the
 * system is still exploring. A tight interval means the choice is settled; a
 * wide one means the bandit has not decided yet — and the UI must not pretend
 * otherwise. The chosen candidate is labelled, not just recoloured.
 *
 * Interval computed in TS (normal approximation, see `betaInterval`) rather
 * than trusting a wire field that may not exist yet.
 */
export default function PosteriorBars({ candidates, ariaLabel = "候选干预的 Thompson 采样后验" }: PosteriorBarsProps) {
  if (candidates.length === 0) {
    return <div className="mf-chart-empty">暂无候选分布数据</div>;
  }

  return (
    <div className="mf-posterior" role="img" aria-label={ariaLabel}>
      {candidates.map((candidate) => {
        const { mean, low, high } = betaInterval(candidate.alpha, candidate.beta);
        const left = clamp(low, 0, 1) * 100;
        const right = clamp(high, 0, 1) * 100;
        const widthPct = Math.max(1.2, right - left);
        return (
          <div
            className="mf-posterior-row"
            key={candidate.intervention_type || candidate.label}
            data-chosen={candidate.chosen}
            title={`Beta(${candidate.alpha.toFixed(1)}, ${candidate.beta.toFixed(1)}) · n=${candidate.sample}`}
          >
            <span className="mf-posterior-label">
              {candidate.chosen && (
                <span className="badge badge-success" style={{ fontSize: 10, padding: "0 6px" }}>已选</span>
              )}
              {candidate.label}
            </span>
            <div className="mf-posterior-track">
              <span className="mf-posterior-ci" style={{ left: `${left}%`, width: `${widthPct}%` }} />
              <span className="mf-posterior-mean" style={{ left: `${clamp(mean, 0, 1) * 100}%` }} />
            </div>
            <span className="mf-posterior-value">{Math.round(mean * 100)}%</span>
          </div>
        );
      })}
      <div className="mf-posterior-legend">
        <span>● 估计有用的概率</span>
        <span>▬ 可能的范围（90%）</span>
        <span>范围越宽 = 系统还在摸索</span>
      </div>
    </div>
  );
}
