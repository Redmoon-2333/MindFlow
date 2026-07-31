/**
 * Canonical report wire contract (Todo 16).
 *
 * Mirrors backend `DailyReportResponse` / `WeeklyReportResponse` (api/schemas.py,
 * Todo 15) and the shared data-state Literals (DailyDataState / WeeklyDataState).
 * This module declares the typed contract only; boundary parsing lives in
 * `report-state.ts` and the exhaustive render decisions in `report-view.ts`.
 */

export const DAILY_DATA_STATES = [
  "ready",
  "no_activity",
  "events_only",
  "neutral_only",
  "no_focus",
  "future",
] as const;

export type DailyDataState = (typeof DAILY_DATA_STATES)[number];

export const WEEKLY_DATA_STATES = ["ready", "partial", "no_activity", "future"] as const;

export type WeeklyDataState = (typeof WEEKLY_DATA_STATES)[number];

/** Hour keys the backend always emits: "0".."23" (local business hours). */
export const HOUR_KEYS: readonly string[] = Array.from({ length: 24 }, (_, h) => String(h));

export interface TopAppEntry {
  readonly app: string;
  readonly minutes: number;
}

export interface DailyReportKnown {
  readonly data_state: DailyDataState;
  readonly id: string;
  readonly user_id: number;
  readonly date: string;
  readonly total_focus_min: number;
  readonly total_distraction_min: number;
  readonly focus_score: number;
  readonly top_apps: readonly TopAppEntry[];
  readonly switch_frequency: number;
  readonly pattern_summary: string;
  readonly created_at: string | null;
  readonly total_focus_minutes: number;
  readonly total_sessions: number;
  readonly total_distractions: number;
  readonly hourly_distribution: Readonly<Record<string, number>>;
}

export interface DailyReportUnknown {
  readonly data_state: "unknown";
  /** Untrusted original data_state, surfaced so nothing is silently dropped. */
  readonly raw_data_state: string;
  readonly date: string;
}

export type DailyReport = DailyReportKnown | DailyReportUnknown;

export interface DailySummaryEntry {
  readonly date: string;
  readonly focus_minutes: number;
  readonly sessions: number;
  readonly distractions: number;
  readonly focus_score: number;
}

export interface WeeklyTrend {
  /** Absent when the previous week had no reports to compare against. */
  readonly focus_min_delta_pct: number | null;
  readonly focus_score_delta: number | null;
  readonly direction: "up" | "down" | "stable" | null;
}

export interface WeeklyReportKnown {
  readonly data_state: WeeklyDataState;
  readonly week_start: string;
  readonly week_end: string;
  readonly daily_reports: readonly DailyReport[];
  readonly averages: Readonly<Record<string, number>>;
  readonly trend: WeeklyTrend;
  readonly week_number: number;
  readonly intervention_effectiveness: Readonly<Record<string, unknown>> | null;
  readonly total_focus_minutes: number;
  readonly total_sessions: number;
  readonly total_distractions: number;
  readonly avg_focus_score: number;
  readonly daily_summary: readonly DailySummaryEntry[];
}

export interface WeeklyReportUnknown {
  readonly data_state: "unknown";
  readonly raw_data_state: string;
  readonly week_start: string;
}

export type WeeklyReport = WeeklyReportKnown | WeeklyReportUnknown;

// ── State-card copy ────────────────────────────────────────────────────
// User-facing label for each canonical data state, keyed by the same Literals
// above so a new backend state gains its enum value and label in one place.
// The view layer renders these for every non-ready report.

export interface StateCardModel {
  readonly title: string;
  readonly message: string;
}

export const DAILY_STATE_CARDS: Readonly<Record<Exclude<DailyDataState, "ready">, StateCardModel>> = {
  no_activity: { title: "暂无日报数据", message: "这一天没有记录到任何活动或专注会话。" },
  events_only: { title: "仅有活动记录", message: "这一天有活动记录但未形成专注会话，暂不生成时段分布。" },
  neutral_only: { title: "暂无专注数据", message: "这一天存在会话但无法归类为专注或分心，暂不生成时段分布。" },
  no_focus: { title: "暂无专注数据", message: "这一天只有分心记录，没有专注会话，暂不生成时段分布。" },
  future: { title: "所选日期在未来", message: "所选日期晚于今天，暂无报告数据。" },
};

export const DAILY_UNKNOWN_CARD: StateCardModel = {
  title: "报告数据异常",
  message: "日报响应格式无法识别，请稍后重试。",
};

export const WEEKLY_STATE_CARDS: Readonly<Record<Exclude<WeeklyDataState, "ready">, StateCardModel>> = {
  partial: { title: "周报数据不完整", message: "当前周仍在进行中或部分日期缺少数据，暂不生成图表。" },
  no_activity: { title: "暂无周报数据", message: "这一周没有记录到任何活动或专注会话。" },
  future: { title: "所选日期在未来", message: "所选周开始日期晚于今天，暂无报告数据。" },
};

export const WEEKLY_UNKNOWN_CARD: StateCardModel = {
  title: "报告数据异常",
  message: "周报响应格式无法识别，请稍后重试。",
};
