/**
 * Pure contract tests for the report state/view mappers (Todo 16).
 *
 * Run with: npx tsx tests/report-state-mapping.test.ts
 * No test framework: plain node asserts, deterministic, zero clocks.
 *
 * Canonical wire contracts (backend `api/schemas.py`, Todo 15):
 *   DailyReportResponse.data_state:
 *     "ready" | "no_activity" | "events_only" | "neutral_only" | "no_focus" | "future"
 *   WeeklyReportResponse.data_state: "ready" | "partial" | "no_activity" | "future"
 *
 * Rendering invariants under test:
 *   - only `data_state === "ready"` renders KPI/chart models;
 *   - every non-ready state renders an explanatory state card and NO chart model;
 *   - a ready→future (or →no_activity) transition clears the stale chart;
 *   - a ready report needs a complete 24-key finite hourly_distribution and finite
 *     KPI numbers, else it degrades to `unknown` instead of rendering a misleading
 *     chart or fabricated zeros;
 *   - malformed/missing payloads degrade to `unknown` — never throw, never fake data.
 */
import assert from "node:assert/strict";
import { parseDailyReport, parseWeeklyReport } from "../src/report-state";
import { buildHourlyChart, formatMinutes, toDailyReportView, toWeeklyReportView } from "../src/report-view";

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

// ── Fixtures (shape-matched to the backend contract) ──

function hourly24(nonzero: Record<string, number> = {}): Record<string, number> {
  const out: Record<string, number> = {};
  for (let h = 0; h < 24; h++) out[String(h)] = nonzero[String(h)] ?? 0;
  return out;
}

function dailyPayload(state: string, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: "rep-1",
    user_id: 1,
    date: "2026-07-30",
    total_focus_min: 120,
    total_distraction_min: 30,
    focus_score: 85,
    top_apps: [
      { app: "VS Code", minutes: 90 },
      { app: "浏览器", minutes: 40 },
    ],
    switch_frequency: 2.5,
    pattern_summary: "上午专注表现较好",
    created_at: "2026-07-30T12:00:00Z",
    total_focus_minutes: 120,
    total_sessions: 5,
    total_distractions: 3,
    hourly_distribution: hourly24({ "9": 30, "10": 60, "11": 30 }),
    data_state: state,
    ...overrides,
  };
}

const WEEK_SUMMARY = [
  { date: "2026-07-27", focus_minutes: 120, sessions: 5, distractions: 3, focus_score: 85 },
  { date: "2026-07-28", focus_minutes: 90, sessions: 4, distractions: 2, focus_score: 78 },
  { date: "2026-07-29", focus_minutes: 150, sessions: 6, distractions: 1, focus_score: 90 },
  { date: "2026-07-30", focus_minutes: 60, sessions: 3, distractions: 4, focus_score: 70 },
  { date: "2026-07-31", focus_minutes: 130, sessions: 5, distractions: 2, focus_score: 88 },
  { date: "2026-08-01", focus_minutes: 80, sessions: 4, distractions: 3, focus_score: 75 },
  { date: "2026-08-02", focus_minutes: 110, sessions: 5, distractions: 2, focus_score: 82 },
];

function weeklyPayload(state: string, overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    week_start: "2026-07-27",
    week_end: "2026-08-02",
    daily_reports: [dailyPayload("ready"), dailyPayload("ready")],
    averages: { avg_focus_min: 105.7, avg_distraction_min: 20, avg_focus_score: 81.1, avg_switch_frequency: 2.4 },
    trend: { focus_min_delta_pct: 12.3, focus_score_delta: 5, direction: "up" },
    week_number: 31,
    intervention_effectiveness: null,
    total_focus_minutes: 740,
    total_sessions: 32,
    total_distractions: 17,
    avg_focus_score: 81.1,
    daily_summary: WEEK_SUMMARY,
    data_state: state,
    ...overrides,
  };
}

// ── Daily: ready renders KPI + chart models ──

test("daily ready renders KPI model with formatted values", () => {
  const report = parseDailyReport(dailyPayload("ready"));
  assert.equal(report.data_state, "ready");
  const view = toDailyReportView(report);
  assert.equal(view.ready, true);
  assert.ok(view.kpis, "ready must carry a KPI model");
  assert.equal(view.kpis.totalFocusMinutes, "2h 0m");
  assert.equal(view.kpis.totalSessions, "5");
  assert.equal(view.kpis.totalDistractions, "3");
  assert.equal(view.kpis.focusScore, "85");
  assert.equal(view.stateCard, null);
});

test("daily ready renders a 24-bar hourly chart from the 24-key distribution", () => {
  const view = toDailyReportView(parseDailyReport(dailyPayload("ready")));
  assert.ok(view.hourlyChart, "ready must carry an hourly chart model");
  assert.equal(view.hourlyChart.bars.length, 24);
  assert.equal(view.hourlyChart.bars[0].hour, 0);
  assert.equal(view.hourlyChart.bars[23].hour, 23);
  // max = 60 (hour 10) → hour 9 (30m) is 50%, hour 10 is 100%.
  assert.equal(view.hourlyChart.bars[9].pct, 50);
  assert.equal(view.hourlyChart.bars[10].pct, 100);
  assert.equal(view.hourlyChart.bars[9].label, "30m");
  assert.equal(view.hourlyChart.bars[10].label, "60m");
  assert.equal(view.hourlyChart.bars[0].axisLabel, "0h");
  assert.equal(view.hourlyChart.bars[3].axisLabel, "3h");
  assert.equal(view.hourlyChart.bars[1].axisLabel, "");
});

test("buildHourlyChart is pure and 24-key driven", () => {
  const chart = buildHourlyChart(hourly24({ "9": 30, "10": 60, "11": 30 }));
  assert.equal(chart.bars.length, 24);
  assert.equal(chart.maxMinutes, 60);
  assert.equal(chart.bars[10].pct, 100);
  assert.equal(chart.bars[23].minutes, 0);
  assert.equal(chart.bars[23].pct, 0);
  assert.equal(chart.bars[23].label, "");
});

test("dense all-24-hours-nonzero keeps every bar, value label, and 3-hour anchor in the model", () => {
  const dense: Record<string, number> = {};
  for (let h = 0; h < 24; h++) dense[String(h)] = 15 + ((h * 13) % 60);
  const chart = buildHourlyChart(dense);
  assert.equal(chart.bars.length, 24);
  assert.equal(chart.maxMinutes, Math.max(...Object.values(dense)));
  for (let h = 0; h < 24; h++) {
    const bar = chart.bars[h];
    assert.ok(bar.label.length > 0, `hour ${h} must keep a value label (got "${bar.label}")`);
    assert.equal(bar.axisLabel, h % 3 === 0 ? `${h}h` : "", `hour ${h} axis anchor`);
    assert.ok(bar.minutes > 0 && bar.pct > 0, `hour ${h} bar must stay visible (pct=${bar.pct})`);
  }
  // The dense payload still parses as a ready report rendering all 24 bars.
  const view = toDailyReportView(parseDailyReport(dailyPayload("ready", { hourly_distribution: dense })));
  assert.equal(view.ready, true);
  assert.ok(view.hourlyChart, "dense ready must carry the chart model");
  assert.equal(view.hourlyChart.bars.length, 24);
});

test("daily ready maps top_apps (canonical field) into the view", () => {
  const view = toDailyReportView(parseDailyReport(dailyPayload("ready")));
  assert.equal(view.topApps.length, 2);
  assert.equal(view.topApps[0].app, "VS Code");
  assert.equal(view.topApps[0].minutes, 90);
});

test("daily ready with missing top_apps maps to empty list (schema-optional, no fake rows)", () => {
  const view = toDailyReportView(parseDailyReport(dailyPayload("ready", { top_apps: undefined })));
  assert.equal(view.ready, true);
  assert.equal(view.topApps.length, 0);
});

// ── Daily: every non-ready state renders a card and no chart ──

const DAILY_STATE_CARD_TITLES: Readonly<Record<string, string>> = {
  no_activity: "暂无日报数据",
  events_only: "仅有活动记录",
  neutral_only: "暂无专注数据",
  no_focus: "暂无专注数据",
  future: "所选日期在未来",
};

for (const state of Object.keys(DAILY_STATE_CARD_TITLES)) {
  test(`daily ${state} renders an explanatory card and NO chart model`, () => {
    const report = parseDailyReport(dailyPayload(state));
    assert.equal(report.data_state, state);
    const view = toDailyReportView(report);
    assert.equal(view.ready, false);
    assert.equal(view.hourlyChart, null, `${state} must not render a misleading chart`);
    assert.equal(view.kpis, null, `${state} must not render fabricated KPIs`);
    assert.equal(view.topApps.length, 0);
    assert.ok(view.stateCard, `${state} must render a state card`);
    assert.equal(view.stateCard.title, DAILY_STATE_CARD_TITLES[state]);
    assert.ok(view.stateCard.message.length > 0, "state card must explain the state");
  });
}

// ── Daily: ready→future transition clears stale bars ──

test("ready→future transition clears the stale hourly chart", () => {
  const readyView = toDailyReportView(parseDailyReport(dailyPayload("ready")));
  assert.ok(readyView.hourlyChart, "ready must initially render bars");
  const futureView = toDailyReportView(parseDailyReport(dailyPayload("future")));
  assert.equal(futureView.ready, false);
  assert.equal(futureView.hourlyChart, null, "stale bars must be cleared on the future state");
  assert.equal(futureView.stateCard.title, "所选日期在未来");
});

test("ready→no_activity transition clears the stale hourly chart", () => {
  const readyView = toDailyReportView(parseDailyReport(dailyPayload("ready")));
  assert.ok(readyView.hourlyChart);
  const noActivityView = toDailyReportView(parseDailyReport(dailyPayload("no_activity")));
  assert.equal(noActivityView.hourlyChart, null);
});

// ── Daily: malformed / missing input boundary ──

test("ready with 23-key hourly distribution degrades to unknown (no gap chart)", () => {
  const hourly = hourly24();
  delete hourly["5"];
  const report = parseDailyReport(dailyPayload("ready", { hourly_distribution: hourly }));
  assert.equal(report.data_state, "unknown");
  const view = toDailyReportView(report);
  assert.equal(view.ready, false);
  assert.equal(view.hourlyChart, null);
});

test("ready with non-finite hourly value degrades to unknown", () => {
  const report = parseDailyReport(dailyPayload("ready", { hourly_distribution: hourly24({ "10": Number.NaN }) }));
  assert.equal(report.data_state, "unknown");
  assert.equal(toDailyReportView(report).hourlyChart, null);
});

test("ready with missing hourly_distribution degrades to unknown", () => {
  const report = parseDailyReport(dailyPayload("ready", { hourly_distribution: undefined }));
  assert.equal(report.data_state, "unknown");
  assert.equal(toDailyReportView(report).hourlyChart, null);
});

test("ready with missing total_sessions degrades to unknown (no fabricated zero KPI)", () => {
  const report = parseDailyReport(dailyPayload("ready", { total_sessions: undefined }));
  assert.equal(report.data_state, "unknown");
  assert.equal(toDailyReportView(report).kpis, null);
});

test("unrecognized data_state degrades to unknown preserving the raw state", () => {
  const report = parseDailyReport(dailyPayload("bogus"));
  assert.equal(report.data_state, "unknown");
  assert.equal(report.raw_data_state, "bogus");
  const view = toDailyReportView(report);
  assert.equal(view.ready, false);
  assert.equal(view.hourlyChart, null);
  assert.equal(view.stateCard.title, "报告数据异常");
});

test("missing data_state degrades to unknown", () => {
  const report = parseDailyReport(dailyPayload("ready", { data_state: undefined }));
  assert.equal(report.data_state, "unknown");
});

test("non-object / missing-date payloads degrade to unknown without throwing", () => {
  for (const value of [null, undefined, "ready", 42, [], true]) {
    const report = parseDailyReport(value);
    assert.equal(report.data_state, "unknown");
    assert.equal(toDailyReportView(report).hourlyChart, null);
  }
  const missingDate = parseDailyReport(dailyPayload("ready", { date: undefined }));
  assert.equal(missingDate.data_state, "unknown");
});

// ── Weekly: ready renders KPI + chart models ──

test("weekly ready renders KPI model with formatted values", () => {
  const report = parseWeeklyReport(weeklyPayload("ready"));
  assert.equal(report.data_state, "ready");
  const view = toWeeklyReportView(report);
  assert.equal(view.ready, true);
  assert.ok(view.kpis, "ready must carry a KPI model");
  assert.equal(view.kpis.totalFocusMinutes, "12h 20m");
  assert.equal(view.kpis.totalSessions, "32");
  assert.equal(view.kpis.totalDistractions, "17");
  assert.equal(view.kpis.avgFocusScore, "81");
  assert.equal(view.stateCard, null);
});

test("weekly ready renders a 7-day chart with weekday/date labels", () => {
  const view = toWeeklyReportView(parseWeeklyReport(weeklyPayload("ready")));
  assert.ok(view.chart, "ready must carry a weekly chart model");
  assert.equal(view.chart.bars.length, 7);
  assert.equal(view.chart.bars[0].date, "2026-07-27");
  assert.equal(view.chart.bars[0].dayLabel, "周一");
  assert.equal(view.chart.bars[0].dateLabel, "7/27");
  assert.equal(view.chart.bars[0].focusMinutes, 120);
  assert.equal(view.chart.bars[0].label, "2h 0m");
  assert.equal(view.chart.maxFocusMinutes, 150);
  assert.equal(view.chart.bars[2].pct, 100);
  assert.equal(view.chart.bars[0].pct, 80);
});

test("weekly ready exposes the daily summary for the detail table", () => {
  const view = toWeeklyReportView(parseWeeklyReport(weeklyPayload("ready")));
  assert.equal(view.summary.length, 7);
  assert.equal(view.summary[0].sessions, 5);
  assert.equal(view.summary[0].focus_score, 85);
});

test("weekly ready maps trend deltas into display metrics", () => {
  const view = toWeeklyReportView(parseWeeklyReport(weeklyPayload("ready")));
  assert.ok(view.trend, "non-empty trend must map to a trend model");
  assert.equal(view.trend.metrics.length, 2);
  assert.equal(view.trend.metrics[0].label, "专注时长变化");
  assert.equal(view.trend.metrics[0].display, "+12%");
  assert.equal(view.trend.metrics[0].good, true);
  assert.equal(view.trend.metrics[0].sub, "↑ 改善");
  assert.equal(view.trend.metrics[1].label, "评分变化");
  assert.equal(view.trend.metrics[1].display, "+5");
});

test("weekly ready with empty trend renders no trend card", () => {
  const view = toWeeklyReportView(parseWeeklyReport(weeklyPayload("ready", { trend: {} })));
  assert.equal(view.trend, null);
});

// ── Weekly: every non-ready state renders a card and no chart ──

const WEEKLY_STATE_CARD_TITLES: Readonly<Record<string, string>> = {
  partial: "周报数据不完整",
  no_activity: "暂无周报数据",
  future: "所选日期在未来",
};

for (const state of Object.keys(WEEKLY_STATE_CARD_TITLES)) {
  test(`weekly ${state} renders an explanatory card and NO chart model`, () => {
    const report = parseWeeklyReport(weeklyPayload(state));
    assert.equal(report.data_state, state);
    const view = toWeeklyReportView(report);
    assert.equal(view.ready, false);
    assert.equal(view.chart, null, `${state} must not render a misleading chart`);
    assert.equal(view.kpis, null, `${state} must not render fabricated KPIs`);
    assert.equal(view.summary.length, 0);
    assert.equal(view.trend, null);
    assert.ok(view.stateCard, `${state} must render a state card`);
    assert.equal(view.stateCard.title, WEEKLY_STATE_CARD_TITLES[state]);
  });
}

// ── Weekly: malformed / missing input boundary ──

test("weekly unrecognized data_state degrades to unknown", () => {
  const report = parseWeeklyReport(weeklyPayload("bogus"));
  assert.equal(report.data_state, "unknown");
  assert.equal(report.raw_data_state, "bogus");
  const view = toWeeklyReportView(report);
  assert.equal(view.ready, false);
  assert.equal(view.chart, null);
  assert.equal(view.stateCard.title, "报告数据异常");
});

test("weekly missing week_start degrades to unknown", () => {
  const report = parseWeeklyReport(weeklyPayload("ready", { week_start: undefined }));
  assert.equal(report.data_state, "unknown");
});

test("weekly non-object payload degrades to unknown without throwing", () => {
  for (const value of [null, undefined, "ready", 42, [], true]) {
    const report = parseWeeklyReport(value);
    assert.equal(report.data_state, "unknown");
    assert.equal(toWeeklyReportView(report).chart, null);
  }
});

test("weekly ready with empty daily_summary degrades to unknown (no empty chart)", () => {
  const report = parseWeeklyReport(weeklyPayload("ready", { daily_summary: [] }));
  assert.equal(report.data_state, "unknown");
  assert.equal(toWeeklyReportView(report).chart, null);
});

test("weekly ready with malformed daily_summary entry degrades to unknown", () => {
  const bad = WEEK_SUMMARY.map((entry) => ({ ...entry }));
  bad[2] = { ...bad[2], focus_minutes: undefined };
  const report = parseWeeklyReport(weeklyPayload("ready", { daily_summary: bad }));
  assert.equal(report.data_state, "unknown");
});

// ── formatMinutes (shared formatter) ──

test("formatMinutes formats durations and degrades missing values to —", () => {
  assert.equal(formatMinutes(120), "2h 0m");
  assert.equal(formatMinutes(45), "45m");
  assert.equal(formatMinutes(90), "1h 30m");
  assert.equal(formatMinutes(null), "—");
  assert.equal(formatMinutes(undefined), "—");
});

// ── Report ──

if (failures.length > 0) {
  console.error(`\n${failures.length} test(s) FAILED:`);
  for (const failure of failures) console.error(`  - ${failure}`);
  console.error(`${passed} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`report-state-mapping: ${passed} tests passed`);
