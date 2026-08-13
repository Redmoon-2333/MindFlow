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

const kpi = deriveFocusTrendKpi(trend);

// today's focused minutes = last daily entry
assert.equal(kpi.todayMinutes, 120, "todayMinutes should be the last day's focus_min");
assert.equal(kpi.totalMinutes, 220, "totalMinutes should sum focus_min across days");
assert.equal(kpi.sessionCount, 10, "sessionCount should come from total_sessions");
assert.equal(kpi.avgScore, 75, "avgScore should be the last day's avg_score");
// avgDuration = totalMinutes / totalFocusSessions (sum of daily session_count = 7)
assert.ok(kpi.avgDurationMinutes !== undefined && Math.abs(kpi.avgDurationMinutes - 31.4286) < 0.01,
  "avgDuration = totalMinutes / totalFocusSessions");

// score change vs previous day: (75-80)/80 = -6.25%
assert.ok(kpi.scoreChange !== undefined && Math.abs(kpi.scoreChange - (-6.25)) < 0.01,
  "scoreChange should be -6.25% vs previous day");

// distraction rate of the last day: 30/(120+30) = 20%
assert.ok(kpi.distractionRate !== undefined && Math.abs(kpi.distractionRate - 0.2) < 1e-9,
  "distractionRate should be 0.2 for the last day");
assert.equal(kpi.distractionLabel, "专注良好");
assert.equal(kpi.trendLabel, "较昨日 -6%");

// null / empty inputs
assert.deepEqual(deriveFocusTrendKpi(null), {});
const empty = deriveFocusTrendKpi({ ...trend, daily: [] });
assert.equal(empty.sessionCount, 10);
assert.equal(empty.todayMinutes, undefined);

console.log("focus-trend-kpi: 12 tests passed");