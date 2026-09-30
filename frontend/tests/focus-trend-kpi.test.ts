/**
 * Derived-KPI tests for the Dashboard trend view (audit fix).
 *
 * Run with: npx tsx tests/focus-trend-kpi.test.ts
 * No test framework: plain node asserts, deterministic, zero clocks.
 */
import assert from "node:assert/strict";
import { deriveFocusTrendKpi, type FocusTrendResponse } from "../src/api.ts";

const trend: FocusTrendResponse = {
  days: 7,
  start_date: "2026-08-07",
  end_date: "2026-08-13",
  total_sessions: 10,
  daily: [
    { date: "2026-08-07", focus_min: 100, distraction_min: 20, session_count: 3, avg_score: 80 },
    { date: "2026-08-08", focus_min: 120, distraction_min: 30, session_count: 4, avg_score: 75 },
  ],
};

// The series below ends on 2026-08-08, so "today" for this fixture is that
// key. Passing it explicitly is what makes the assertions below meaningful:
// the old implementation answered "today" from `daily[daily.length - 1]`,
// which is the defect audit F1 (历史趋势冒充今日统计) is about. The caller now
// always passes the local date key.
const TODAY_KEY = "2026-08-08";
const kpi = deriveFocusTrendKpi(trend, TODAY_KEY);

// today's focused minutes = the entry whose date IS today (2026-08-08)
assert.equal(kpi.todayMinutes, 120, "todayMinutes should be today's focus_min");
assert.equal(kpi.totalMinutes, 220, "totalMinutes should sum focus_min across days");
assert.equal(kpi.sessionCount, 10, "sessionCount should come from total_sessions");
assert.equal(kpi.avgScore, 75, "avgScore should be today's avg_score");
// avgDuration = totalMinutes / totalFocusSessions (sum of daily session_count = 7)
assert.ok(kpi.avgDurationMinutes !== undefined && Math.abs(kpi.avgDurationMinutes - 31.4286) < 0.01,
  "avgDuration = totalMinutes / totalFocusSessions");

// score change vs yesterday (2026-08-07): (75-80)/80 = -6.25%
assert.ok(kpi.scoreChange !== undefined && Math.abs(kpi.scoreChange - (-6.25)) < 0.01,
  "scoreChange should be -6.25% vs the previous calendar day");

// distraction rate of today: 30/(120+30) = 20%
assert.ok(kpi.distractionRate !== undefined && Math.abs(kpi.distractionRate - 0.2) < 1e-9,
  "distractionRate should be 0.2 for today");
assert.equal(kpi.distractionLabel, "专注良好");
assert.equal(kpi.trendLabel, "较昨日 -6%");

// null / empty inputs
assert.deepEqual(deriveFocusTrendKpi(null), {});
const empty = deriveFocusTrendKpi({ ...trend, daily: [] }, TODAY_KEY);
assert.equal(empty.sessionCount, 10);
assert.equal(empty.todayMinutes, undefined);

// ── F1 regressions: sparse series must not promote an older day to "today" ──

// 1. Today has no record → today's numbers are absent, not the last entry's.
const noToday = deriveFocusTrendKpi({ ...trend, daily: trend.daily.slice(0, 1) }, TODAY_KEY);
assert.equal(noToday.todayMinutes, undefined, "a missing today must stay undefined");
assert.equal(noToday.avgScore, undefined, "no observation today → no average");
assert.equal(noToday.scoreChange, undefined, "yesterday missing → no fabricated delta");
assert.equal(noToday.distractionRate, undefined, "no today → no distraction ratio");
// Range-wide aggregates keep working regardless.
assert.equal(noToday.totalMinutes, 100, "totalMinutes still covers the whole range");

// 2. Sparse, non-contiguous dates: only the exact keys count.
const sparse = deriveFocusTrendKpi(
  {
    ...trend,
    daily: [
      { date: "2026-08-01", focus_min: 10, distraction_min: 0, session_count: 1, avg_score: 60 },
      { date: "2026-08-05", focus_min: 20, distraction_min: 0, session_count: 1, avg_score: 70 },
      { date: "2026-08-08", focus_min: 30, distraction_min: 0, session_count: 1, avg_score: 80 },
    ],
  },
  "2026-08-08",
);
assert.equal(sparse.todayMinutes, 30, "today is found by key, not position");
assert.equal(sparse.scoreChange, undefined, "08-07 has no record, so no delta vs 08-05");
const sparsePrev = deriveFocusTrendKpi(
  { ...trend, daily: sparseDailyWithYesterday() },
  "2026-08-08",
);
assert.ok(sparsePrev.scoreChange !== undefined, "an actual yesterday still produces a delta");

// 3. Cross-month / cross-year boundaries use the local calendar.
const crossMonth = deriveFocusTrendKpi(
  {
    ...trend,
    daily: [
      { date: "2026-08-31", focus_min: 10, distraction_min: 0, session_count: 1, avg_score: 60 },
      { date: "2026-09-01", focus_min: 30, distraction_min: 0, session_count: 1, avg_score: 80 },
    ],
  },
  "2026-09-01",
);
assert.equal(crossMonth.todayMinutes, 30, "month rollover must not break the key lookup");
assert.ok(crossMonth.scoreChange !== undefined, "2026-08-31 is yesterday across the month boundary");

const crossYear = deriveFocusTrendKpi(
  {
    ...trend,
    daily: [
      { date: "2025-12-31", focus_min: 10, distraction_min: 0, session_count: 1, avg_score: 60 },
      { date: "2026-01-01", focus_min: 30, distraction_min: 0, session_count: 1, avg_score: 80 },
    ],
  },
  "2026-01-01",
);
assert.equal(crossYear.todayMinutes, 30, "year rollover must not break the key lookup");
assert.ok(crossYear.scoreChange !== undefined, "2025-12-31 is yesterday across the year boundary");

// 4. Out-of-order daily arrays are still resolved by key.
const unsorted = deriveFocusTrendKpi(
  {
    ...trend,
    daily: [
      { date: "2026-08-08", focus_min: 30, distraction_min: 0, session_count: 1, avg_score: 80 },
      { date: "2026-08-07", focus_min: 10, distraction_min: 0, session_count: 1, avg_score: 60 },
    ],
  },
  "2026-08-08",
);
assert.equal(unsorted.todayMinutes, 30, "ordering must not decide which day is today");
assert.ok(unsorted.scoreChange !== undefined, "yesterday is found regardless of array order");

function sparseDailyWithYesterday() {
  return [
    { date: "2026-08-07", focus_min: 10, distraction_min: 0, session_count: 1, avg_score: 60 },
    { date: "2026-08-08", focus_min: 30, distraction_min: 0, session_count: 1, avg_score: 80 },
  ];
}

console.log("focus-trend-kpi: all assertions passed");