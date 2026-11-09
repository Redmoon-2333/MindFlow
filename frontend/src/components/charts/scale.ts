/** Shared numeric helpers for the in-house SVG charts. No dependencies. */

/**
 * Round a maximum up to a "nice" axis bound and return evenly spaced ticks.
 *
 * 92 → ticks 0/25/50/75/100; 7 → 0/2/4/6/8. Judges read grids, not raw maxima,
 * so the top tick must be a round number above the data.
 */
export function niceTicks(max: number, count = 4): { top: number; ticks: number[] } {
  if (!Number.isFinite(max) || max <= 0) return { top: 1, ticks: [0, 1] };
  const raw = max / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const residual = raw / magnitude;
  const step = (residual <= 1 ? 1 : residual <= 2 ? 2 : residual <= 5 ? 5 : 10) * magnitude;
  const top = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = 0; value <= top + step / 2; value += step) {
    // Guard against float drift accumulating (0.30000000000000004).
    ticks.push(Math.round(value * 1e6) / 1e6);
  }
  return { top, ticks };
}

/** Compact axis/tooltip number: 1200 → "1.2k", 3.25 → "3.25". */
export function formatCompact(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 10_000) return `${(value / 1000).toFixed(0)}k`;
  if (abs >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(abs % 1 === 0 ? 0 : 1);
  return value.toFixed(abs % 1 === 0 ? 0 : 2);
}

/** Signed value with an explicit + so direction is never inferred from colour. */
export function formatSigned(value: number, digits = 3): string {
  const fixed = value.toFixed(digits);
  return value > 0 ? `+${fixed}` : fixed;
}

/** Clamp helper used by every bar geometry. */
export function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}
