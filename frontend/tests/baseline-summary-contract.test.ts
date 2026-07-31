/**
 * Pure contract tests for the baseline-summary boundary parser and the
 * Model Center view-state reducer (Todo 10).
 *
 * Run with: npx tsx tests/baseline-summary-contract.test.ts
 * No test framework: plain node asserts, deterministic, zero clocks.
 *
 * Canonical wire contract (backend `BaselineSummary`, api/schemas.py):
 *   { user_id, created_at, updated_at, total_days, total_samples, features,
 *     mean_app_switch_count, mean_active_seconds_ratio, mean_idle_ratio,
 *     switch_frequency, productivity_ratio }
 *   - `features` is exactly the 24-name V2 vocabulary (domain/feature_schema.py);
 *   - compatibility aliases are one-to-one copies:
 *       switch_frequency == mean_app_switch_count
 *       productivity_ratio == mean_active_seconds_ratio
 *   - an empty repository is a 404 (handled at the fetch layer as `empty`).
 *
 * Rendering invariants under test:
 *   - a valid payload parses to `ok: true` with every canonical field typed;
 *   - malformed / missing / non-finite payloads degrade to `ok: false` — the
 *     parser never throws and never fabricates values (a string mean or a NaN
 *     must not become a number);
 *   - the view-state reducer clears stale data on populated→404 and clears the
 *     stale empty flag on 404→populated, and keeps the last trusted view on a
 *     transient error.
 */
import assert from "node:assert/strict";
import {
  EMPTY_BASELINE_VIEW,
  parseBaselineSummary,
  reduceBaselineState,
  type BaselineSummary,
  type BaselineViewState,
} from "../src/baseline-state";

let passed = 0;
const failures: string[] = [];

function test(name: string, fn: () => void): void {
  try {
    fn();
    passed += 1;
  } catch (error) {
    failures.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

// Exact V2 vocabulary from backend domain/feature_schema.py V2_FEATURE_NAMES.
const V2_FEATURES: readonly string[] = [
  "app_switch_count",
  "domain_switch_count",
  "longest_segment_ratio",
  "idle_ratio",
  "keypress_rate_per_min",
  "mouse_click_rate_per_min",
  "scroll_rate_per_min",
  "mouse_distance_per_min",
  "input_active_ratio",
  "interaction_bursts_per_min",
  "click_key_ratio",
  "browser_ratio",
  "audible_browser_ratio",
  "active_seconds_ratio",
  "top_app_ratio",
  "top_domain_ratio",
  "interaction_interval_mean_s",
  "interaction_interval_std_s",
  "interaction_interval_cv",
  "hour_sin",
  "hour_cos",
  "weekday_sin",
  "weekday_cos",
  "task_type_code",
];

function summaryPayload(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    user_id: 1,
    created_at: "2026-07-30T08:00:00+00:00",
    updated_at: "2026-07-30T09:00:00+00:00",
    total_days: 14,
    total_samples: 336,
    features: V2_FEATURES,
    mean_app_switch_count: 12.5,
    mean_active_seconds_ratio: 0.5,
    mean_idle_ratio: 0.2,
    switch_frequency: 12.5,
    productivity_ratio: 0.5,
    ...overrides,
  };
}

// ── Boundary parse: valid payloads ──

test("valid payload parses to ok:true with every canonical field typed", () => {
  const state = parseBaselineSummary(summaryPayload());
  assert.equal(state.ok, true);
  if (!state.ok) return;
  assert.equal(state.user_id, 1);
  assert.equal(state.total_days, 14);
  assert.equal(state.total_samples, 336);
  assert.equal(state.created_at, "2026-07-30T08:00:00+00:00");
  assert.equal(state.updated_at, "2026-07-30T09:00:00+00:00");
  assert.equal(state.mean_app_switch_count, 12.5);
  assert.equal(state.mean_active_seconds_ratio, 0.5);
  assert.equal(state.mean_idle_ratio, 0.2);
  assert.equal(state.switch_frequency, 12.5);
  assert.equal(state.productivity_ratio, 0.5);
});

test("features is exactly the 24-name V2 vocabulary", () => {
  const state = parseBaselineSummary(summaryPayload());
  assert.equal(state.ok, true);
  if (!state.ok) return;
  assert.equal(state.features.length, 24);
  assert.deepEqual([...state.features], [...V2_FEATURES]);
});

test("compat aliases equal the canonical means on the wire fixture", () => {
  const state = parseBaselineSummary(summaryPayload());
  assert.equal(state.ok, true);
  if (!state.ok) return;
  assert.equal(state.switch_frequency, state.mean_app_switch_count);
  assert.equal(state.productivity_ratio, state.mean_active_seconds_ratio);
});

test("null means are accepted and preserved (no data for that feature)", () => {
  const state = parseBaselineSummary(summaryPayload({
    mean_app_switch_count: null,
    mean_idle_ratio: null,
    switch_frequency: null,
  }));
  assert.equal(state.ok, true);
  if (!state.ok) return;
  assert.equal(state.mean_app_switch_count, null);
  assert.equal(state.mean_idle_ratio, null);
  assert.equal(state.switch_frequency, null);
  assert.equal(state.mean_active_seconds_ratio, 0.5);
});

test("zero-valued means are trusted as numbers (not falsy-dropped)", () => {
  const state = parseBaselineSummary(summaryPayload({ mean_idle_ratio: 0, productivity_ratio: 0 }));
  assert.equal(state.ok, true);
  if (!state.ok) return;
  assert.equal(state.mean_idle_ratio, 0);
  assert.equal(state.productivity_ratio, 0);
});

// ── Boundary parse: malformed payloads degrade, never throw ──

test("non-object payloads degrade to ok:false", () => {
  for (const value of [null, undefined, "ready", 42, [], true]) {
    const state = parseBaselineSummary(value);
    assert.equal(state.ok, false, `expected malformed for ${String(value)}`);
  }
});

test("missing required numeric field (user_id) degrades to ok:false", () => {
  const payload = summaryPayload();
  delete payload.user_id;
  assert.equal(parseBaselineSummary(payload).ok, false);
});

test("missing created_at / updated_at degrades to ok:false", () => {
  const noCreated = summaryPayload();
  delete noCreated.created_at;
  assert.equal(parseBaselineSummary(noCreated).ok, false);
  const noUpdated = summaryPayload();
  delete noUpdated.updated_at;
  assert.equal(parseBaselineSummary(noUpdated).ok, false);
});

test("string mean is not trusted (never coerced to a number)", () => {
  const state = parseBaselineSummary(summaryPayload({ mean_app_switch_count: "12.5" }));
  assert.equal(state.ok, false);
});

test("NaN mean degrades to ok:false (no NaN surfacing into the view)", () => {
  const state = parseBaselineSummary(summaryPayload({ mean_idle_ratio: Number.NaN }));
  assert.equal(state.ok, false);
});

test("Infinity mean degrades to ok:false", () => {
  const state = parseBaselineSummary(summaryPayload({ productivity_ratio: Number.POSITIVE_INFINITY }));
  assert.equal(state.ok, false);
});

test("features that are not a string array degrade to ok:false", () => {
  assert.equal(parseBaselineSummary(summaryPayload({ features: "app_switch_count" })).ok, false);
  assert.equal(parseBaselineSummary(summaryPayload({ features: [1, 2] })).ok, false);
  assert.equal(parseBaselineSummary(summaryPayload({ features: ["app_switch_count", 2] })).ok, false);
  const missing = summaryPayload();
  delete missing.features;
  assert.equal(parseBaselineSummary(missing).ok, false);
});

test("missing compat alias degrades to ok:false (contract requires exact fields)", () => {
  const noAlias = summaryPayload();
  delete noAlias.switch_frequency;
  assert.equal(parseBaselineSummary(noAlias).ok, false);
  const noProductivity = summaryPayload();
  delete noProductivity.productivity_ratio;
  assert.equal(parseBaselineSummary(noProductivity).ok, false);
});

test("non-finite total_samples / total_days degrade to ok:false", () => {
  assert.equal(parseBaselineSummary(summaryPayload({ total_samples: Number.NaN })).ok, false);
  assert.equal(parseBaselineSummary(summaryPayload({ total_days: "14" })).ok, false);
});

// ── View-state transitions: populated ↔ empty without stale state ──

const POPULATED: BaselineSummary = (() => {
  const state = parseBaselineSummary(summaryPayload());
  assert.equal(state.ok, true);
  return state as BaselineSummary;
})();

test("initial (never loaded) → populated renders the summary, not the empty card", () => {
  const next = reduceBaselineState(EMPTY_BASELINE_VIEW, { kind: "populated", summary: POPULATED });
  assert.equal(next.empty, false);
  assert.equal(next.summary, POPULATED);
});

test("initial (never loaded) → 404 empty renders the empty card, not stale data", () => {
  const next = reduceBaselineState(EMPTY_BASELINE_VIEW, { kind: "empty" });
  assert.equal(next.empty, true);
  assert.equal(next.summary, null);
});

test("populated → 404 empty clears the stale summary", () => {
  const before: BaselineViewState = { summary: POPULATED, empty: false };
  const next = reduceBaselineState(before, { kind: "empty" });
  assert.equal(next.empty, true);
  assert.equal(next.summary, null, "a 404 must never leave the old summary on screen");
});

test("404 empty → populated clears the stale empty flag", () => {
  const before: BaselineViewState = { summary: null, empty: true };
  const next = reduceBaselineState(before, { kind: "populated", summary: POPULATED });
  assert.equal(next.empty, false);
  assert.equal(next.summary, POPULATED, "a populated response must never stay hidden behind the empty card");
});

test("populated → transient error keeps the last trusted view", () => {
  const before: BaselineViewState = { summary: POPULATED, empty: false };
  const next = reduceBaselineState(before, { kind: "error" });
  assert.equal(next.summary, POPULATED);
  assert.equal(next.empty, false);
});

test("404 empty → transient error keeps the empty state (does not flip to populated)", () => {
  const before: BaselineViewState = { summary: null, empty: true };
  const next = reduceBaselineState(before, { kind: "error" });
  assert.equal(next.empty, true);
  assert.equal(next.summary, null);
});

// ── Report ──

if (failures.length > 0) {
  console.error(`\n${failures.length} test(s) FAILED:`);
  for (const failure of failures) console.error(`  - ${failure}`);
  console.error(`${passed} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`baseline-summary-contract: ${passed} tests passed`);
