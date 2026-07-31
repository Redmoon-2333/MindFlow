/**
 * Pure baseline-summary contract: canonical types, boundary parser, and the
 * Model Center view-state reducer (Todo 10).
 *
 * Mirrors backend `BaselineSummary` (api/schemas.py) and the V2 vocabulary
 * (domain/feature_schema.py). This module is free of fetch/DOM so ModelCenter
 * and Analytics share one parse/render decision and the contract can be
 * exercised by pure tsx scripts.
 *
 * Contract invariants (enforced at the parse boundary, never in render):
 *   - `features` is exactly the 24-name V2 vocabulary;
 *   - canonical means (`mean_app_switch_count`, `mean_active_seconds_ratio`,
 *     `mean_idle_ratio`) are finite numbers or null — a string, NaN or
 *     Infinity is never trusted;
 *   - compatibility aliases (`switch_frequency == mean_app_switch_count`,
 *     `productivity_ratio == mean_active_seconds_ratio`) are part of the
 *     exact wire contract and required like every other field;
 *   - malformed / missing payloads degrade to `ok: false` — the parser never
 *     throws and never fabricates values;
 *   - a 404 from the fetch layer maps to the `empty` outcome; the reducer
 *     clears stale data on populated→404 and the stale empty flag on
 *     404→populated, and keeps the last trusted view on a transient error.
 */

// ── Canonical types ────────────────────────────────────────────────────

export interface BaselineSummary {
  readonly ok: true;
  readonly user_id: number;
  readonly created_at: string;
  readonly updated_at: string;
  readonly total_days: number;
  readonly total_samples: number;
  readonly features: readonly string[];
  readonly mean_app_switch_count: number | null;
  readonly mean_active_seconds_ratio: number | null;
  readonly mean_idle_ratio: number | null;
  readonly switch_frequency: number | null;
  readonly productivity_ratio: number | null;
}

export interface BaselineSummaryMalformed {
  readonly ok: false;
}

export type BaselineSummaryState = BaselineSummary | BaselineSummaryMalformed;

// ── Boundary parser ────────────────────────────────────────────────────

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function asFiniteNumber(value: unknown): number | null {
  return isFiniteNumber(value) ? value : null;
}

function parseFeatures(value: unknown): readonly string[] | null {
  if (!Array.isArray(value)) return null;
  const features: string[] = [];
  for (const entry of value) {
    if (typeof entry !== "string") return null;
    features.push(entry);
  }
  return features;
}

const MALFORMED: BaselineSummaryMalformed = { ok: false };

/** Parse an untrusted wire payload into the canonical baseline summary. */
export function parseBaselineSummary(value: unknown): BaselineSummaryState {
  if (!isRecord(value)) return MALFORMED;

  const user_id = asFiniteNumber(value.user_id);
  const total_days = asFiniteNumber(value.total_days);
  const total_samples = asFiniteNumber(value.total_samples);
  const created_at = typeof value.created_at === "string" ? value.created_at : null;
  const updated_at = typeof value.updated_at === "string" ? value.updated_at : null;
  const features = parseFeatures(value.features);
  if (user_id === null || total_days === null || total_samples === null) return MALFORMED;
  if (created_at === null || updated_at === null) return MALFORMED;
  if (features === null) return MALFORMED;

  // Canonical means and their one-to-one aliases are finite-or-null; any
  // other wire type (string, NaN, Infinity) breaks the contract.
  const mean_app_switch_count = asFiniteNumber(value.mean_app_switch_count);
  const mean_active_seconds_ratio = asFiniteNumber(value.mean_active_seconds_ratio);
  const mean_idle_ratio = asFiniteNumber(value.mean_idle_ratio);
  const switch_frequency = asFiniteNumber(value.switch_frequency);
  const productivity_ratio = asFiniteNumber(value.productivity_ratio);
  if (mean_app_switch_count === null && value.mean_app_switch_count !== null) return MALFORMED;
  if (mean_active_seconds_ratio === null && value.mean_active_seconds_ratio !== null) return MALFORMED;
  if (mean_idle_ratio === null && value.mean_idle_ratio !== null) return MALFORMED;
  if (switch_frequency === null && value.switch_frequency !== null) return MALFORMED;
  if (productivity_ratio === null && value.productivity_ratio !== null) return MALFORMED;

  return {
    ok: true,
    user_id,
    created_at,
    updated_at,
    total_days,
    total_samples,
    features,
    mean_app_switch_count,
    mean_active_seconds_ratio,
    mean_idle_ratio,
    switch_frequency,
    productivity_ratio,
  };
}

// ── View-state reducer ─────────────────────────────────────────────────

/** Outcome of one baseline fetch: populated, 404-empty, or other error. */
export type BaselineLoadOutcome =
  | { readonly kind: "populated"; readonly summary: BaselineSummary }
  | { readonly kind: "empty" }
  | { readonly kind: "error" };

/** What the Model Center baseline tab renders: data or the empty card. */
export interface BaselineViewState {
  readonly summary: BaselineSummary | null;
  readonly empty: boolean;
}

export const EMPTY_BASELINE_VIEW: BaselineViewState = { summary: null, empty: false };

/** Transition the baseline view without ever keeping stale state. */
export function reduceBaselineState(prev: BaselineViewState, outcome: BaselineLoadOutcome): BaselineViewState {
  switch (outcome.kind) {
    case "populated": {
      return { summary: outcome.summary, empty: false };
    }
    case "empty": {
      return { summary: null, empty: true };
    }
    case "error": {
      // Transient failure: keep the last trusted view; the caller surfaces
      // the error banner separately.
      return prev;
    }
  }
}
