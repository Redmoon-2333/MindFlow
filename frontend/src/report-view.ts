/**
 * Exhaustive report view mappers (Todo 16).
 *
 * One render decision shared by the Reports page: given a canonical report
 * (from `report-state.ts`), produce the display model — KPI/chart models for
 * `ready`, an explanatory state card for every other state. The mappers are
 * pure (no fetch/DOM/Date) and exhaustive: a new data-state variant must be
 * handled here or the `assertNever` default stops compiling.
 *
 * Invariant: `ready === true` iff the report carries chart/KPI models; every
 * non-ready state carries `stateCard` and `null` chart models, so a stale
 * ready→future/no_activity transition automatically drops the old bars.
 */
import type {
  DailyReport,
  DailyReportKnown,
  DailySummaryEntry,
  TopAppEntry,
  WeeklyReport,
  WeeklyReportKnown,
  WeeklyTrend,
} from "./report-contract";
import {
  DAILY_STATE_CARDS, DAILY_UNKNOWN_CARD, WEEKLY_STATE_CARDS, WEEKLY_UNKNOWN_CARD,
} from "./report-contract";
import type { DailyDataState, StateCardModel, WeeklyDataState } from "./report-contract";

// ── Shared formatters ──────────────────────────────────────────────────

export function formatMinutes(m: number | null | undefined): string {
  if (m == null || Number.isNaN(m)) return "—";
  const h = Math.floor(m / 60);
  const min = Math.round(m % 60);
  return h > 0 ? `${h}h ${min}m` : `${min}m`;
}

const WEEKDAY_NAMES = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"] as const;

/** Pure weekday index (0=Sunday..6=Saturday) for a YYYY-MM-DD string — no Date/timezone. */
function weekdayIndex(isoDate: string): number {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(isoDate);
  if (!match) return 0;
  const epochDays = Math.floor(Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])) / 86400000);
  // 1970-01-01 was a Thursday (JS getDay() === 4).
  return (epochDays + 4) % 7;
}

export function dayLabel(dateStr: string): string {
  return WEEKDAY_NAMES[weekdayIndex(dateStr)] ?? WEEKDAY_NAMES[0];
}

function fmtDate(dateStr: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateStr);
  if (!match) return "—";
  return `${Number(match[2])}/${Number(match[3])}`;
}

// ── Daily view ─────────────────────────────────────────────────────────

export interface DailyKpiModel {
  readonly totalFocusMinutes: string;
  readonly totalSessions: string;
  readonly totalDistractions: string;
  readonly focusScore: string;
}

export interface HourBarModel {
  readonly hour: number;
  readonly minutes: number;
  readonly pct: number;
  readonly label: string;
  readonly axisLabel: string;
}

export interface HourlyChartModel {
  readonly bars: readonly HourBarModel[];
  readonly maxMinutes: number;
}

export interface DailyReportView {
  readonly date: string;
  readonly ready: boolean;
  readonly state: DailyDataState | "unknown";
  readonly kpis: DailyKpiModel | null;
  readonly hourlyChart: HourlyChartModel | null;
  readonly topApps: readonly TopAppEntry[];
  readonly stateCard: StateCardModel | null;
  /** Rule-generated personal takeaway (architecture plan J/5.3). */
  readonly patternSummary: string | null;
}

/** Chart model for a ready daily report: one bar per local hour bucket. */
export function buildHourlyChart(distribution: Readonly<Record<string, number>>): HourlyChartModel {
  const values = Array.from({ length: 24 }, (_, hour) => distribution[String(hour)] ?? 0);
  const maxMinutes = Math.max(1, ...values);
  const bars = values.map((minutes, hour) => ({
    hour,
    minutes,
    pct: Math.round((minutes / maxMinutes) * 100),
    label: minutes > 0 ? `${Math.round(minutes)}m` : "",
    axisLabel: hour % 3 === 0 ? `${hour}h` : "",
  }));
  return { bars, maxMinutes };
}

function buildDailyKpis(report: DailyReportKnown): DailyKpiModel {
  return {
    totalFocusMinutes: formatMinutes(report.total_focus_minutes),
    totalSessions: String(report.total_sessions),
    totalDistractions: String(report.total_distractions),
    focusScore: String(Math.round(report.focus_score)),
  };
}

/** Exhaustive daily render decision shared by the Reports page. */
export function toDailyReportView(report: DailyReport): DailyReportView {
  switch (report.data_state) {
    case "ready": {
      return {
        date: report.date,
        ready: true,
        state: "ready",
        kpis: buildDailyKpis(report),
        hourlyChart: buildHourlyChart(report.hourly_distribution),
        topApps: report.top_apps,
        stateCard: null,
        patternSummary: report.ai_insight ?? report.pattern_summary,
      };
    }
    case "no_activity":
    case "events_only":
    case "neutral_only":
    case "no_focus":
    case "future": {
      return {
        date: report.date,
        ready: false,
        state: report.data_state,
        kpis: null,
        hourlyChart: null,
        topApps: [],
        stateCard: DAILY_STATE_CARDS[report.data_state],
        patternSummary: null,
      };
    }
    case "unknown": {
      return {
        date: report.date,
        ready: false,
        state: "unknown",
        kpis: null,
        hourlyChart: null,
        topApps: [],
        stateCard: DAILY_UNKNOWN_CARD,
        patternSummary: null,
      };
    }
    default: {
      // Exhaustiveness guard: a new daily state must be handled above.
      const exhaustive: never = report;
      return exhaustive;
    }
  }
}

// ── Weekly view ────────────────────────────────────────────────────────

export interface WeeklyKpiModel {
  readonly totalFocusMinutes: string;
  readonly totalSessions: string;
  readonly totalDistractions: string;
  readonly avgFocusScore: string;
}

export interface WeeklyDayBarModel {
  readonly date: string;
  readonly dayLabel: string;
  readonly dateLabel: string;
  readonly focusMinutes: number;
  readonly label: string;
  readonly pct: number;
}

export interface WeeklyChartModel {
  readonly bars: readonly WeeklyDayBarModel[];
  readonly maxFocusMinutes: number;
}

export interface TrendMetricModel {
  readonly label: string;
  readonly display: string;
  readonly sub: string;
  readonly good: boolean;
}

export interface WeeklyTrendModel {
  readonly metrics: readonly TrendMetricModel[];
}

export interface WeeklyReportView {
  readonly weekStart: string;
  readonly ready: boolean;
  readonly state: WeeklyDataState | "unknown";
  readonly kpis: WeeklyKpiModel | null;
  readonly chart: WeeklyChartModel | null;
  readonly summary: readonly DailySummaryEntry[];
  readonly trend: WeeklyTrendModel | null;
  readonly stateCard: StateCardModel | null;
}

/** Chart model for a ready weekly report: one bar per day of the week. */
export function buildWeeklyChart(summary: readonly DailySummaryEntry[]): WeeklyChartModel {
  const maxFocusMinutes = Math.max(1, ...summary.map((entry) => entry.focus_minutes));
  const bars = summary.map((entry) => ({
    date: entry.date,
    dayLabel: dayLabel(entry.date),
    dateLabel: fmtDate(entry.date),
    focusMinutes: entry.focus_minutes,
    label: entry.focus_minutes > 0 ? formatMinutes(entry.focus_minutes) : "",
    pct: Math.round((entry.focus_minutes / maxFocusMinutes) * 100),
  }));
  return { bars, maxFocusMinutes };
}

function buildWeeklyKpis(report: WeeklyReportKnown): WeeklyKpiModel {
  return {
    totalFocusMinutes: formatMinutes(report.total_focus_minutes),
    totalSessions: String(report.total_sessions),
    totalDistractions: String(report.total_distractions),
    avgFocusScore: String(Math.round(report.avg_focus_score)),
  };
}

/** Map the sparse trend dict to display metrics; null when nothing to compare. */
export function toWeeklyTrendModel(trend: WeeklyTrend): WeeklyTrendModel | null {
  const metrics: TrendMetricModel[] = [];
  if (trend.focus_min_delta_pct !== null) {
    const value = trend.focus_min_delta_pct;
    metrics.push({
      label: "专注时长变化",
      display: `${value > 0 ? "+" : ""}${Math.round(value)}%`,
      sub: value >= 0 ? "↑ 改善" : "↓ 下降",
      good: value >= 0,
    });
  }
  if (trend.focus_score_delta !== null) {
    const value = trend.focus_score_delta;
    metrics.push({
      label: "评分变化",
      display: `${value > 0 ? "+" : ""}${Math.round(value)}`,
      sub: value >= 0 ? "↑ 改善" : "↓ 下降",
      good: value >= 0,
    });
  }
  return metrics.length > 0 ? { metrics } : null;
}

/** Exhaustive weekly render decision shared by the Reports page. */
export function toWeeklyReportView(report: WeeklyReport): WeeklyReportView {
  switch (report.data_state) {
    case "ready": {
      return {
        weekStart: report.week_start,
        ready: true,
        state: "ready",
        kpis: buildWeeklyKpis(report),
        chart: buildWeeklyChart(report.daily_summary),
        summary: report.daily_summary,
        trend: toWeeklyTrendModel(report.trend),
        stateCard: null,
      };
    }
    case "partial":
    case "no_activity":
    case "future": {
      return {
        weekStart: report.week_start,
        ready: false,
        state: report.data_state,
        kpis: null,
        chart: null,
        summary: [],
        trend: null,
        stateCard: WEEKLY_STATE_CARDS[report.data_state],
      };
    }
    case "unknown": {
      return {
        weekStart: report.week_start,
        ready: false,
        state: "unknown",
        kpis: null,
        chart: null,
        summary: [],
        trend: null,
        stateCard: WEEKLY_UNKNOWN_CARD,
      };
    }
    default: {
      // Exhaustiveness guard: a new weekly state must be handled above.
      const exhaustive: never = report;
      return exhaustive;
    }
  }
}
