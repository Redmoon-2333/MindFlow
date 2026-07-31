/**
 * Boundary parsers for the report wire contract (Todo 16).
 *
 * Parse-don't-validate: untrusted `GET /reports/*` payloads cross the boundary
 * exactly once here and become the canonical types declared in
 * `report-contract.ts`; the render path (Reports page + `report-view.ts`)
 * consumes typed values and never re-validates.
 *
 * Invariants (enforced at this boundary, never in render):
 *   - malformed / missing / unrecognized payloads degrade to the explicit
 *     `unknown` state — the parser never throws and never fabricates data;
 *   - fields consumed by the render path (KPIs, hourly distribution, weekly
 *     summary) are validated strictly — a missing/non-finite number must never
 *     become a fabricated 0, and a "ready" report missing its complete 24-hour
 *     picture degrades to `unknown` rather than rendering a misleading chart;
 *   - fields the backend schema itself marks optional (`top_apps`,
 *     `created_at`, `daily_reports`, `averages`, `trend` dict entries,
 *     `intervention_effectiveness`) are parsed leniently — absent → empty/null.
 */
import type {
  DailyDataState,
  DailyReport,
  DailyReportKnown,
  DailyReportUnknown,
  DailySummaryEntry,
  TopAppEntry,
  WeeklyDataState,
  WeeklyReport,
  WeeklyReportKnown,
  WeeklyReportUnknown,
  WeeklyTrend,
} from "./report-contract";
import { DAILY_DATA_STATES, HOUR_KEYS, WEEKLY_DATA_STATES } from "./report-contract";

export type {
  DailyDataState,
  DailyReport,
  DailyReportKnown,
  DailyReportUnknown,
  DailySummaryEntry,
  TopAppEntry,
  WeeklyDataState,
  WeeklyReport,
  WeeklyReportKnown,
  WeeklyReportUnknown,
  WeeklyTrend,
} from "./report-contract";

// ── Guards ─────────────────────────────────────────────────────────────

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function asFiniteNumber(value: unknown): number | null {
  return isFiniteNumber(value) ? value : null;
}

function isDailyDataState(value: string): value is DailyDataState {
  return (DAILY_DATA_STATES as readonly string[]).includes(value);
}

function isWeeklyDataState(value: string): value is WeeklyDataState {
  return (WEEKLY_DATA_STATES as readonly string[]).includes(value);
}

/** Extract the named numeric fields from a wire record; null if any is missing/non-finite. */
function requireFiniteFields(
  value: Readonly<Record<string, unknown>>,
  names: readonly string[],
): Readonly<Record<string, number>> | null {
  const out: Record<string, number> = {};
  for (const name of names) {
    const num = asFiniteNumber(value[name]);
    if (num === null) return null;
    out[name] = num;
  }
  return out;
}

function parseTopApp(value: unknown): TopAppEntry | null {
  if (!isRecord(value)) return null;
  const app = typeof value.app === "string" ? value.app : "";
  const minutes = asFiniteNumber(value.minutes);
  if (!app || minutes === null) return null;
  return { app, minutes };
}

function parseHourly(value: unknown): Readonly<Record<string, number>> | null {
  if (!isRecord(value)) return null;
  const out: Record<string, number> = {};
  for (const [hour, minutes] of Object.entries(value)) {
    const num = asFiniteNumber(minutes);
    if (num === null) return null;
    out[hour] = num;
  }
  return out;
}

function hasCompleteHourly(value: Readonly<Record<string, number>>): boolean {
  for (const hour of HOUR_KEYS) {
    if (!isFiniteNumber(value[hour])) return false;
  }
  return true;
}

function unknownDaily(raw_data_state: string, date: string): DailyReportUnknown {
  return { data_state: "unknown", raw_data_state, date };
}

function unknownWeekly(raw_data_state: string, week_start: string): WeeklyReportUnknown {
  return { data_state: "unknown", raw_data_state, week_start };
}

// ── Daily parser ───────────────────────────────────────────────────────

/** Parse an untrusted daily wire payload into the canonical discriminated union. */
export function parseDailyReport(value: unknown): DailyReport {
  if (!isRecord(value)) return unknownDaily("", "");
  const date = typeof value.date === "string" ? value.date : "";
  const raw_data_state = typeof value.data_state === "string" ? value.data_state : "";
  if (!date || !isDailyDataState(raw_data_state)) return unknownDaily(raw_data_state, date);

  const numbers = requireFiniteFields(value, [
    "user_id",
    "total_focus_min",
    "total_distraction_min",
    "focus_score",
    "switch_frequency",
    "total_focus_minutes",
    "total_sessions",
    "total_distractions",
  ]);
  if (numbers === null) return unknownDaily(raw_data_state, date);

  const top_apps = Array.isArray(value.top_apps)
    ? value.top_apps.map(parseTopApp).filter((entry): entry is TopAppEntry => entry !== null)
    : [];

  const hourly_distribution = parseHourly(value.hourly_distribution);
  if (hourly_distribution === null) return unknownDaily(raw_data_state, date);

  // A "ready" report without the complete 24-hour picture would render a
  // misleading chart — degrade instead.
  if (raw_data_state === "ready" && !hasCompleteHourly(hourly_distribution)) {
    return unknownDaily(raw_data_state, date);
  }

  const known: DailyReportKnown = {
    data_state: raw_data_state,
    id: typeof value.id === "string" ? value.id : "",
    user_id: numbers.user_id,
    date,
    total_focus_min: numbers.total_focus_min,
    total_distraction_min: numbers.total_distraction_min,
    focus_score: numbers.focus_score,
    top_apps,
    switch_frequency: numbers.switch_frequency,
    pattern_summary: typeof value.pattern_summary === "string" ? value.pattern_summary : "",
    created_at: typeof value.created_at === "string" ? value.created_at : null,
    total_focus_minutes: numbers.total_focus_minutes,
    total_sessions: numbers.total_sessions,
    total_distractions: numbers.total_distractions,
    hourly_distribution,
  };
  return known;
}

// ── Weekly parsers ─────────────────────────────────────────────────────

function parseNumberRecord(value: unknown): Readonly<Record<string, number>> {
  if (!isRecord(value)) return {};
  const out: Record<string, number> = {};
  for (const [key, entry] of Object.entries(value)) {
    const num = asFiniteNumber(entry);
    if (num !== null) out[key] = num;
  }
  return out;
}

function parseTrend(value: unknown): WeeklyTrend {
  if (!isRecord(value)) return { focus_min_delta_pct: null, focus_score_delta: null, direction: null };
  const direction =
    value.direction === "up" || value.direction === "down" || value.direction === "stable"
      ? value.direction
      : null;
  return {
    focus_min_delta_pct: asFiniteNumber(value.focus_min_delta_pct),
    focus_score_delta: asFiniteNumber(value.focus_score_delta),
    direction,
  };
}

function parseDailySummaryEntry(value: unknown): DailySummaryEntry | null {
  if (!isRecord(value)) return null;
  const date = typeof value.date === "string" ? value.date : "";
  const numbers = requireFiniteFields(value, ["focus_minutes", "sessions", "distractions", "focus_score"]);
  if (!date || numbers === null) return null;
  return {
    date,
    focus_minutes: numbers.focus_minutes,
    sessions: numbers.sessions,
    distractions: numbers.distractions,
    focus_score: numbers.focus_score,
  };
}

/** Parse an untrusted weekly wire payload into the canonical discriminated union. */
export function parseWeeklyReport(value: unknown): WeeklyReport {
  if (!isRecord(value)) return unknownWeekly("", "");
  const week_start = typeof value.week_start === "string" ? value.week_start : "";
  const raw_data_state = typeof value.data_state === "string" ? value.data_state : "";
  if (!week_start || !isWeeklyDataState(raw_data_state)) return unknownWeekly(raw_data_state, week_start);

  const week_end = typeof value.week_end === "string" ? value.week_end : "";
  const numbers = requireFiniteFields(value, [
    "week_number",
    "total_focus_minutes",
    "total_sessions",
    "total_distractions",
    "avg_focus_score",
  ]);
  if (!week_end || numbers === null) return unknownWeekly(raw_data_state, week_start);

  const daily_reports = Array.isArray(value.daily_reports) ? value.daily_reports.map(parseDailyReport) : [];
  const averages = parseNumberRecord(value.averages);
  const trend = parseTrend(value.trend);
  const intervention_effectiveness = isRecord(value.intervention_effectiveness)
    ? (value.intervention_effectiveness as Readonly<Record<string, unknown>>)
    : null;

  let daily_summary: readonly DailySummaryEntry[] = [];
  if (Array.isArray(value.daily_summary)) {
    const entries = value.daily_summary.map(parseDailySummaryEntry);
    if (entries.some((entry) => entry === null)) return unknownWeekly(raw_data_state, week_start);
    daily_summary = entries.filter((entry): entry is DailySummaryEntry => entry !== null);
  } else if (value.daily_summary !== undefined) {
    return unknownWeekly(raw_data_state, week_start);
  }

  // A "ready" week with no per-day rows would render an empty chart — degrade.
  if (raw_data_state === "ready" && daily_summary.length === 0) {
    return unknownWeekly(raw_data_state, week_start);
  }

  const known: WeeklyReportKnown = {
    data_state: raw_data_state,
    week_start,
    week_end,
    daily_reports,
    averages,
    trend,
    week_number: numbers.week_number,
    intervention_effectiveness,
    total_focus_minutes: numbers.total_focus_minutes,
    total_sessions: numbers.total_sessions,
    total_distractions: numbers.total_distractions,
    avg_focus_score: numbers.avg_focus_score,
    daily_summary,
  };
  return known;
}
