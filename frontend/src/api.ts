import createClient from "openapi-fetch";
import type { components, paths } from "./generated/api-schema";
import { parseFocusPrediction } from "./prediction-state";
import type { FocusPredictionResponse } from "./prediction-state";
import { parseDailyReport, parseWeeklyReport } from "./report-state";
import type { DailyReport, WeeklyReport } from "./report-state";
import { parseBaselineSummary } from "./baseline-state";
import type { BaselineSummaryState } from "./baseline-state";
import { findDayByDate, localDateStr, previousLocalDateStr } from "./date-utils";
import { parseDeviation, parseFeedbackRequest, parseRationale, parseTmt } from "./types/explainability";
import type {
  DeviationSummary, ExplainableAttribution, ExplainablePanel, FeedbackAnswerResponse, FeedbackLabel,
  FeedbackRequest, InterventionRationale, TmtBreakdown,
} from "./types/explainability";

const BASE = "/api/v1";
export const AUTH_MARKER = "mindflow_authenticated";
export const AUTH_REQUIRED_EVENT = "mindflow:auth-required";
const client = createClient<paths>();

type CollectorStatusResponse = components["schemas"]["CollectorStatusResponse"];
type ChatResponse = components["schemas"]["ChatResponse"];
type PanelResponse = components["schemas"]["PanelResponse"];
type InterventionPayload = components["schemas"]["InterventionPayload"];
/** The trigger payload, re-typed so `rationale` is the parsed shape rather
 *  than the raw wire object. Everything else comes straight from the schema. */
export type InterventionTriggerResponse = {
  intervention: (Omit<InterventionPayload, "rationale"> & { rationale?: InterventionRationale | null }) | null;
  skipped: boolean;
  skip_reason?: string | null;
};
type InterventionCommandResponse = components["schemas"]["InterventionCommandResponse"];
type InterventionIntensity = components["schemas"]["InterventionTriggerRequest"]["intensity"];
type InterventionResponse = components["schemas"]["InterventionResponseRequest"]["response"];
type InterventionRating = components["schemas"]["InterventionFeedbackRequest"]["rating"];
type ActivityQuery = NonNullable<paths["/api/v1/activities"]["get"]["parameters"]["query"]>;
export type TelemetryDeleteScope = NonNullable<paths["/api/v1/telemetry/data"]["delete"]["parameters"]["query"]>["scope"];
export type TelemetryDeleteResponse = components["schemas"]["TelemetryDeleteResponse"];

export type TelemetryDeleteNotice =
  | { kind: "success"; message: string; failures: []; retryScope: null }
  | { kind: "partial"; message: string; failures: string[]; retryScope: TelemetryDeleteScope };

export type TelemetryDeleteOutcome = {
  notice: TelemetryDeleteNotice;
  clearPairingCode: boolean;
};

export type TelemetryDeleteExecution =
  | { telemetry: TelemetryStatus; refreshError: null }
  | { telemetry: null; refreshError: unknown };

export type TelemetryDeleteDependencies = {
  clear: (scope: TelemetryDeleteScope) => Promise<TelemetryDeleteResponse>;
  refresh: () => Promise<TelemetryStatus>;
  onDeleted: (outcome: TelemetryDeleteOutcome) => void;
};

const TELEMETRY_FAILURE_LABELS: Record<string, string> = {
  browser_tokens: "浏览器配对令牌",
  model_artifacts: "本地模型文件",
};
const UNKNOWN_TELEMETRY_FAILURE = "未知清理步骤（服务未返回失败详情）";

export type CollectorStatus = CollectorStatusResponse;
export type ChatReply = ChatResponse;
export type PanelResult = PanelResponse;
export type { InterventionIntensity, InterventionPayload, InterventionResponse, InterventionRating };

export interface ProblemDetail {
  type?: string;
  title: string;
  status: number;
  detail: string;
  instance?: string;
  request_id?: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly problem?: ProblemDetail;

  constructor(message: string, status: number, problem?: ProblemDetail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.problem = problem;
  }
}

export interface ActivityData {
  app_name: string;
  window_title: string;
  process_name: string;
  is_idle: boolean;
}

export interface ActivityItem {
  id: string;
  user_id: number;
  timestamp: string;
  duration_s: number;
  event_type: string;
  data: ActivityData;
}

export interface ActivitiesResponse {
  items: ActivityItem[];
  page: number;
  page_size: number;
  total: number | null;
  has_more: boolean;
  next_cursor: string | null;
}

export interface ChatSession {
  session_id: string;
  last_message_at: string;
  title?: string;
}

export interface ChatMessageRecord {
  id: string;
  user_id: number;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface InterventionHistoryItem {
  id: string;
  user_id: number;
  triggered_at: string;
  intervention_type: string;
  cbt_technique: string | null;
  context_json: Record<string, unknown> | null;
  user_response: InterventionResponse | null;
  response_latency_s: number | null;
  feedback_rating: InterventionRating | null;
  feedback_comment: string | null;
  created_at: string;
  title?: string;
  message?: string;
  /** Contract addition: why this reminder was chosen. Optional. */
  rationale?: InterventionRationale | null;
}

export interface InterventionHistoryResponse {
  items: InterventionHistoryItem[];
  count: number;
  has_more: boolean;
  next_cursor?: string | null;
}

export interface TelemetryPreferences {
  input_telemetry_enabled: boolean;
  browser_tracking_enabled: boolean;
  interaction_retention_days: number;
  activity_retention_days: number;
}

export interface TelemetryStatus {
  preferences: TelemetryPreferences;
  input_watcher_status: string;
  database_size_bytes: number;
  interaction_bucket_count: number;
  browser_segment_count: number;
  browser_paired: boolean;
  last_interaction_at: string | null;
  last_browser_at: string | null;
}

export interface HealthData {
  status: string;
  version: string;
  timestamp: string;
  collector: {
    status: string;
    recovery_attempts?: number;
    last_error?: string | null;
    next_retry_at?: string | null;
    failure_count_7d?: number;
    last_failure_at?: string | null;
    last_failure_reason?: string | null;
    sleep_count_7d?: number;
    current_interval_s?: number;
  };
  database: { status: "ok" | "error"; connected: boolean };
  migration: { applied: boolean };
}

export interface FocusSession {
  id: string;
  start_time: string;
  end_time: string;
  session_type: string;
  dominant_app: string | null;
  focus_score: number;
  switch_count: number;
  duration_minutes?: number;
  duration?: number;
  started_at?: string;
  ended_at?: string;
  date?: string;
  main_app?: string;
  app?: string;
  app_name?: string;
  score?: number;
  switches?: number;
  /** Feedback already recorded for this session (merged in by the backend).
   *  Present so the focus page can seed its editor from the saved values
   *  instead of a default draft that would overwrite them. */
  feedback_label?: "focus" | "distracted" | "mixed";
  feedback_score?: number;
  feedback_task_type?: string | null;
}

export interface FocusSessionsResponse {
  date: string;
  sessions: FocusSession[];
  session_count: number;
}

export interface FocusTrendDay {
  date: string;
  focus_min: number;
  distraction_min: number;
  session_count: number;
  avg_score: number;
}

export interface FocusTrendResponse {
  days: number;
  start_date: string;
  end_date: string;
  daily: FocusTrendDay[];
  total_sessions: number;
}

/** Derived KPI view computed from FocusTrendResponse.daily — never
 *  asserted from the wire payload (the backend returns only the daily array).
 *  All values are optional so a missing/empty trend renders "--".
 *
 *  `todayKey`/`yesterdayKey` are **local date keys**, not array positions:
 *  the trend is a sparse aggregation of recorded days (audit F1), so the last
 *  entry may be an older day and neighbouring entries need not be adjacent.
 *  A missing today means "no record" (the caller decides between 0 and --);
 *  it is never substituted with another day's numbers. */
export interface FocusTrendKpi {
  todayMinutes?: number;
  totalMinutes?: number;
  sessionCount?: number;
  avgDurationMinutes?: number;
  avgScore?: number;
  scoreChange?: number;
  distractionRate?: number;
  distractionLabel?: string;
  trendLabel?: string;
}

export function deriveFocusTrendKpi(
  trend: FocusTrendResponse | null,
  todayKey: string = localDateStr(),
): FocusTrendKpi {
  if (!trend) return {};
  const days = trend.daily ?? [];
  if (days.length === 0) {
    return { sessionCount: trend.total_sessions || 0 };
  }
  const yesterdayKey = previousLocalDateStr(todayKey);
  // Keyed lookup: `days` is sparse, so positional indexing would promote a
  // historical day to "today" and a non-adjacent day to "yesterday".
  const today = findDayByDate(days, todayKey);
  const prev = findDayByDate(days, yesterdayKey);
  const totalMinutes = days.reduce((sum, d) => sum + (d.focus_min ?? 0), 0);
  const sessionCount = trend.total_sessions ?? days.reduce((sum, d) => sum + (d.session_count ?? 0), 0);
  const totalFocusSessions = days.reduce((sum, d) => sum + (d.session_count ?? 0), 0);
  const avgDurationMinutes = totalFocusSessions > 0 ? totalMinutes / totalFocusSessions : undefined;
  const todayMinutes = today?.focus_min;
  const avgScore = today?.avg_score;
  const scoreChange =
    prev != null && typeof prev.avg_score === "number" && typeof today?.avg_score === "number" && prev.avg_score > 0
      ? ((today.avg_score - prev.avg_score) / prev.avg_score) * 100
      : undefined;
  const totalActivity =
    today == null ? 0 : (today.focus_min ?? 0) + (today.distraction_min ?? 0);
  const distractionRate =
    today == null ? undefined : totalActivity > 0 ? (today.distraction_min ?? 0) / totalActivity : undefined;
  const distractionLabel =
    distractionRate == null ? undefined
      : distractionRate > 0.5 ? "分心偏高"
        : distractionRate > 0.3 ? "分心中等"
          : "专注良好";
  const trendLabel =
    scoreChange == null ? ""
      : scoreChange > 5 ? "较昨日 +" + scoreChange.toFixed(0) + "%"
        : scoreChange < -5 ? "较昨日 " + scoreChange.toFixed(0) + "%"
          : "与昨日持平";
  return {
    todayMinutes,
    totalMinutes,
    sessionCount,
    avgDurationMinutes,
    avgScore,
    scoreChange,
    distractionRate,
    distractionLabel,
    trendLabel,
  };
}

export interface FocusFeedbackRequest {
  label: "focus" | "distracted" | "mixed";
  score: number;
  task_type?: string;
}

export interface FocusFeedbackResponse {
  id: string;
  user_id: number;
  session_id: string;
  label: FocusFeedbackRequest["label"];
  score: number;
  task_type: string | null;
  created_at: string;
}

export interface AnalyticsPatterns {
  high_switch_periods: Array<{
    hour: number;
    switch_count: number;
    period?: string;
    label?: string;
    intensity?: string;
    level?: string;
  }>;
  trigger_apps: Array<{
    app: string;
    count: number;
    app_name?: string;
    name?: string;
    percentage?: number;
  }>;
  heatmap: number[][];
  total_sessions: number;
  distraction_ratio: number;
}

export interface BehavioralProfile {
  peak_focus_hours: Array<{ hour: number; avg_score: number }>;
  top_apps: Array<{ app: string; total_min: number }>;
  avg_focus_block_min: number;
  distraction_triggers: Array<{ app: string; count: number }>;
  total_events_analysed: number;
  profile_date: string;
  peak_focus?: string;
  productivity_apps?: string[];
  trigger_apps?: string[];
  details?: Record<string, unknown>;
}

export interface ModelStatus {
  loaded: boolean;
  ready: boolean;
  demo_only?: boolean;
  inference_available?: boolean;
  model_origin?: "bundled_base" | "personal";
  synthetic_only?: boolean;
  personalized?: boolean;
  mode: string;
  v2_mode: string;
  message: string;
  feature_schema_version?: number;
  version?: string | null;
  available_versions?: string[];
  reasons?: string[];
  model_name?: string;
  last_updated?: string;
  /** Progressive deployment tier (architecture plan E): full_ready |
   *  low_confidence | shadow. */
  deployment_tier?: "full_ready" | "low_confidence" | "shadow" | "base";
}

export interface AttributionResult {
  procrastination_type?: string;
  type?: string;
  confidence?: string | number;
  cbt_technique?: string;
  evidence?: string;
}

export interface AttributionResponse {
  assessment: {
    procrastination_types: string[];
    type_confidence: Record<string, number>;
    cognitive_distortions?: string[];
    cbt_technique: string | null;
    response_text: string;
    next_action?: string;
  };
  source: string;
  cached: boolean;
  meta: { degraded: boolean };
  results?: AttributionResult[];
  confidence?: string | number | null;
  procrastination_type?: string | null;
  cbt_technique?: string | null;
  evidence?: string | null;
  /** Contract additions (docs/audit/20261110-midterm/api-contract-additions.md).
   *  Optional until `npm run generate:api` folds them into the generated schema. */
  deviation?: DeviationSummary | null;
  tmt?: TmtBreakdown | null;
}

/** Panel response plus the optional TMT block the backend is adding. */
export type ExplainablePanelResult = PanelResult & ExplainablePanel;

export type Preferences = Record<string, unknown>;

export interface ClassificationRuleInput {
  process_name: string;
  window_title_pattern: string | null;
  category: string;
  priority: number;
}

export interface ClassificationRule extends ClassificationRuleInput {
  id: string;
  user_id: number;
  created_at: string;
  updated_at: string;
}

export interface AutonomyStatus {
  enabled: boolean;
  paused_until: string | null;
  paused?: boolean;
}

export function isAutonomyPaused(status: AutonomyStatus | null, now = Date.now()): boolean {
  if (!status) return false;
  if (status.paused === true) return true;
  if (!status.paused_until) return false;
  const rawPausedUntil = status.paused_until.trim();
  const hasTimezone = /(?:z|[+-]\d{2}:?\d{2})$/i.test(rawPausedUntil);
  const pausedUntil = Date.parse(hasTimezone ? rawPausedUntil : `${rawPausedUntil}Z`);
  return Number.isNaN(pausedUntil) || pausedUntil > now;
}

export function hasAuthenticatedSession(): boolean {
  return localStorage.getItem(AUTH_MARKER) === "1";
}

/** Own the typed client's abort signal until fetching and parsing settle. */
async function withClientTimeout<T>(
  send: (options: { credentials: "include"; signal: AbortSignal }) => Promise<T>,
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);
  try {
    return await send({ credentials: "include", signal: controller.signal });
  } catch (error: unknown) {
    if (timedOut) throw new ApiError(REQUEST_TIMEOUT_MESSAGE, 408);
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asProblemDetail(value: unknown): ProblemDetail | undefined {
  if (!isRecord(value)) return undefined;
  if (typeof value.title !== "string" || typeof value.status !== "number" || typeof value.detail !== "string") return undefined;
  return value as unknown as ProblemDetail;
}

async function toApiError(response: Response, error?: unknown): Promise<ApiError> {
  if (response.status === 401) {
    localStorage.removeItem(AUTH_MARKER);
    window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
  }
  let body = error;
  if (body === undefined) {
    try { body = await response.clone().json(); } catch { body = undefined; }
  }
  const problem = asProblemDetail(body);
  if (problem) return new ApiError(problem.detail || problem.title, response.status, problem);
  if (isRecord(body) && typeof body.detail === "string") return new ApiError(body.detail, response.status);
  return new ApiError(`请求失败 (${response.status})`, response.status);
}

async function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): Promise<T> {
  if (result.error !== undefined || !result.response.ok) throw await toApiError(result.response, result.error);
  if (result.data === undefined) throw new ApiError("响应缺少数据", result.response.status);
  return result.data;
}

export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.status === 408) {
    return `${fallback}：${error.message}`;
  }
  return error instanceof Error && error.message ? error.message : fallback;
}

export function mapTelemetryDeleteResult(
  scope: TelemetryDeleteScope,
  result: TelemetryDeleteResponse,
): TelemetryDeleteNotice {
  const reportedFailures = result.failures ?? [];
  if (!result.partial && reportedFailures.length === 0) {
    return { kind: "success", message: `已删除 ${result.deleted} 条记录`, failures: [], retryScope: null };
  }
  const hasUnknownFailure = result.partial && reportedFailures.length === 0;
  const failures = hasUnknownFailure
    ? [UNKNOWN_TELEMETRY_FAILURE]
    : reportedFailures.map((failure) => {
        const label = TELEMETRY_FAILURE_LABELS[failure];
        return label ? `${label}（${failure}）` : failure;
      });
  return {
    kind: "partial",
    message: hasUnknownFailure
      ? `已删除 ${result.deleted} 条记录，但仍有未明确报告的项目未完成`
      : `已删除 ${result.deleted} 条记录，但有 ${failures.length} 项未完成`,
    failures,
    retryScope: scope,
  };
}

export function shouldClearTelemetryPairingCode(
  scope: TelemetryDeleteScope,
  result: TelemetryDeleteResponse,
): boolean {
  if (scope !== "browser" && scope !== "all") return false;
  const failures = result.failures;
  if (result.partial && (!failures || failures.length === 0)) return false;
  return !failures?.includes("browser_tokens");
}

export async function runTelemetryDelete(
  scope: TelemetryDeleteScope,
  dependencies: TelemetryDeleteDependencies,
): Promise<TelemetryDeleteExecution> {
  const result = await dependencies.clear(scope);
  dependencies.onDeleted({
    notice: mapTelemetryDeleteResult(scope, result),
    clearPairingCode: shouldClearTelemetryPairingCode(scope, result),
  });
  try {
    return { telemetry: await dependencies.refresh(), refreshError: null };
  } catch (refreshError: unknown) {
    return { telemetry: null, refreshError };
  }
}

const REQUEST_TIMEOUT_MS = 30_000;
/** AI-backed routes need room for a real thinking-mode generation. The
 *  backend allows a single ECNU call 180s and a whole interactive workflow
 *  600s, so the frontend deadline sits above the workflow budget instead of
 *  aborting a request the server would have completed. */
const AI_REQUEST_TIMEOUT_MS = 660_000;
/** Chat's 600s turn deadline includes tools and repeated model calls. */
const CHAT_REQUEST_TIMEOUT_MS = 660_000;
const REQUEST_TIMEOUT_MESSAGE = "请求超时，请稍后重试";

async function fetchWithTimeout<T>(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  parse: (response: Response) => Promise<T>,
  timeoutMs = REQUEST_TIMEOUT_MS,
): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const abortFromCaller = () => controller.abort(init?.signal?.reason);
  init?.signal?.addEventListener("abort", abortFromCaller, { once: true });
  if (init?.signal?.aborted) abortFromCaller();
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    const response = await fetch(input, { ...init, signal: controller.signal });
    return await parse(response);
  } catch (error: unknown) {
    if (timedOut) throw new ApiError(REQUEST_TIMEOUT_MESSAGE, 408);
    throw error;
  } finally {
    clearTimeout(timeoutId);
    init?.signal?.removeEventListener("abort", abortFromCaller);
  }
}

/** Parse a JSON response body, tolerating an empty body.
 *
 *  Endpoints that return `204 No Content` — notably the intervention delete
 *  route — have no body at all. Calling `res.json()` on those throws a
 *  SyntaxError *after* the request already succeeded, which surfaces to the
 *  user as "delete failed" even though the record is gone.
 */
async function parseJsonBody<T>(res: Response): Promise<T> {
  if (res.status === 204 || res.status === 205) return undefined as T;
  const text = await res.text();
  if (!text) return undefined as T;
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new ApiError("响应格式无法解析", res.status);
  }
}

async function request<T>(url: string, init?: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const hasBody = ["POST", "PUT", "PATCH"].includes(method);
  const headers: Record<string, string> = hasBody ? { "Content-Type": "application/json" } : {};
  return fetchWithTimeout(
    `${BASE}${url}`,
    { credentials: "include", ...init, headers: { ...headers, ...((init?.headers as Record<string, string>) || {}) } },
    async (res) => {
      if (!res.ok) throw await toApiError(res);
      return parseJsonBody<T>(res);
    },
    timeoutMs,
  );
}

async function requestText(url: string): Promise<string> {
  return fetchWithTimeout(`${BASE}${url}`, { credentials: "include" }, async (res) => {
    if (!res.ok) throw await toApiError(res);
    return res.text();
  });
}

export async function bootstrapFromFragment(): Promise<boolean> {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const ticket = params.get("bootstrap");
  if (!ticket) return false;
  const result = await withClientTimeout((options) => client.POST("/api/v1/auth/bootstrap", { ...options, body: { ticket } }));
  if (result.error !== undefined || !result.response.ok) throw await toApiError(result.response, result.error);
  // Only clear the one-time ticket after the exchange succeeded, so a failed
  // attempt keeps the URL retryable (audit report 🟡-bootstrap).
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  localStorage.setItem(AUTH_MARKER, "1");
  return true;
}

export const getHealth = () => request<HealthData>("/health");

export interface AICredentialStatus { saved: boolean; active_configured: boolean; environment_override: boolean; restart_required: boolean; provider: string; model: string; service?: "deepseek" | "ecnu"; active_provider?: string; }
export const getAISettings = () => request<AICredentialStatus>("/settings/ai");
export const saveAISettings = (apiKey: string, service: "deepseek" | "ecnu" = "deepseek", ecnuModel: "ecnu-max" | "ecnu-plus" = "ecnu-max") => request<AICredentialStatus>("/settings/ai", { method: "PUT", body: JSON.stringify({ api_key: apiKey, ...(service === "ecnu" ? {service, ecnu_model: ecnuModel} : {}) }) });
export const deleteAISettings = () => request<AICredentialStatus>("/settings/ai", { method: "DELETE" });

/** Sign out of the local session backing this browser.
 *
 *  The cookie is HttpOnly, so revocation has to happen server-side: POST the
 *  session away, then drop the local marker. A 401 means the session already
 *  expired — treat it as a successful logout instead of an error, otherwise a
 *  stale tab can never leave the app. Anything else (network failure, 5xx) is
 *  rethrown so the dialog can offer a retry rather than silently pretending
 *  the user is signed out.
 */
export async function logout(): Promise<void> {
  try {
    await request<void>("/auth/logout", { method: "POST" });
  } catch (error) {
    if (!(error instanceof ApiError && error.status === 401)) throw error;
  }
  localStorage.removeItem(AUTH_MARKER);
  window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
}

function isActivityItem(value: unknown): value is ActivityItem {
  return isRecord(value) && typeof value.id === "string" && typeof value.timestamp === "string" && typeof value.duration_s === "number" && typeof value.event_type === "string" && isRecord(value.data) && typeof value.data.app_name === "string" && typeof value.data.window_title === "string" && typeof value.data.process_name === "string" && typeof value.data.is_idle === "boolean";
}

export async function getActivities(params: ActivityQuery = {}): Promise<ActivitiesResponse> {
  const raw = await unwrap<unknown>(await withClientTimeout((options) => client.GET("/api/v1/activities", { ...options, params: { query: params } })));
  if (!isRecord(raw) || !Array.isArray(raw.items) || !raw.items.every(isActivityItem)) throw new ApiError("活动列表响应格式无效", 500);
  return {
    items: raw.items,
    page: typeof raw.page === "number" ? raw.page : 1,
    page_size: typeof raw.page_size === "number" ? raw.page_size : raw.items.length,
    total: typeof raw.total === "number" ? raw.total : null,
    has_more: raw.has_more === true,
    next_cursor: typeof raw.next_cursor === "string" ? raw.next_cursor : null,
  };
}

export async function getCurrentActivity(): Promise<ActivityItem | null> {
  try {
    const raw = await unwrap<unknown>(await withClientTimeout((options) => client.GET("/api/v1/activities/current", options)));
    const candidate = isRecord(raw) && "current_activity" in raw ? raw.current_activity : raw;
    if (candidate === null) return null;
    if (!isActivityItem(candidate)) throw new ApiError("当前活动响应格式无效", 500);
    return candidate;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) return null;
    throw error;
  }
}

export const getFocusSessions = (date?: string) => request<FocusSessionsResponse>(`/focus${date ? `?date=${date}` : ""}`);
export const getFocusTrend = (days?: number) => request<FocusTrendResponse>(`/focus/trend${days ? `?days=${days}` : ""}`);
export const submitFocusFeedback = (sessionId: string, data: FocusFeedbackRequest) => request<FocusFeedbackResponse>(`/focus/${encodeURIComponent(sessionId)}/feedback`, { method: "POST", body: JSON.stringify(data) });
export const getDailyReport = async (date?: string): Promise<DailyReport> =>
  parseDailyReport(await request<unknown>(`/reports/daily${date ? `?date=${date}` : ""}`));
export const getWeeklyReport = async (weekStart?: string): Promise<WeeklyReport> =>
  parseWeeklyReport(await request<unknown>(`/reports/weekly${weekStart ? `?week_start=${weekStart}` : ""}`));
export const getAnalyticsPatterns = (days?: number) => request<AnalyticsPatterns>(`/analytics/patterns${days ? `?days=${days}` : ""}`);
export const getBaseline = async (): Promise<BaselineSummaryState> =>
  parseBaselineSummary(await request<unknown>("/analytics/baseline"));
export const getProfile = (days?: number) => request<BehavioralProfile>(`/analytics/profile${days ? `?days=${days}` : ""}`);
export const getModelStatus = () => request<ModelStatus>("/analytics/model-status");
export const getAiUsage = () => request<AiUsage>("/analytics/usage");

export interface AiUsage {
  mode: "llm" | "rule_engine";
  llm_calls_30d: number;
  llm_cost_usd_30d: number;
  panel_count_30d: number;
}

/** Allowlisted LLM configuration summary.
 *
 *  `configured` means "a credential exists locally" — it is not a connectivity
 *  probe, and the UI must not present it as one. */
export interface AiProviderStatus {
  provider: string;
  model: string;
  configured: boolean;
  ollama_enabled: boolean;
}
export const getAiProviderStatus = () =>
  request<AiProviderStatus>("/ai/provider-status");
export const runAttribution = async (body?: { date?: string; force?: boolean }): Promise<AttributionResponse> =>
  withExplainability(await request<AttributionResponse>("/analytics/attribution", { method: "POST", body: JSON.stringify(body || {}) }, AI_REQUEST_TIMEOUT_MS));

export async function sendChat(message: string, sessionId?: string, signal?: AbortSignal): Promise<ChatReply> {
  return request<ChatReply>("/chat", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  }, CHAT_REQUEST_TIMEOUT_MS);
}
export async function getChatSessions(): Promise<ChatSession[]> {
  return unwrap(await withClientTimeout((options) => client.GET("/api/v1/chat/sessions", options))) as unknown as ChatSession[];
}
export const renameChatSession = (sessionId: string, title: string) =>
  request<{session_id: string; title: string}>(`/chat/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH", body: JSON.stringify({title}),
  });
export const deleteChatSession = (sessionId: string) =>
  request<void>(`/chat/sessions/${encodeURIComponent(sessionId)}`, {method: "DELETE"});
export async function getChatMessages(sessionId: string): Promise<ChatMessageRecord[]> {
  return unwrap(await withClientTimeout((options) => client.GET("/api/v1/chat/{session_id}/messages", { ...options, params: { path: { session_id: sessionId } } }))) as unknown as ChatMessageRecord[];
}

export const triggerPanel = async (body?: { force?: boolean; retryIfDegraded?: boolean }): Promise<ExplainablePanelResult> =>
  withExplainability(await request<PanelResult>("/panel/today", {
    method: "POST",
    body: JSON.stringify({
      ...(body?.force !== undefined ? { force: body.force } : {}),
      ...(body?.retryIfDegraded !== undefined
        ? { retry_if_degraded: body.retryIfDegraded }
        : {}),
    }),
  }, AI_REQUEST_TIMEOUT_MS));
export const getPanelResult = async (): Promise<ExplainablePanelResult> =>
  withExplainability(await request<PanelResult>("/panel"));

/**
 * Contract-addition bridge (docs/audit/20261110-midterm/api-contract-additions.md).
 *
 * Until `npm run generate:api` includes `deviation` / `tmt` / `rationale`, the
 * wire payload is narrowed here — once, at the boundary — so components never
 * see an unchecked shape. A malformed or absent field becomes `undefined`,
 * which every consumer renders as "暂无解释数据".
 */
export function withExplainability<T extends object>(payload: T): T & ExplainablePanel & ExplainableAttribution {
  const record = payload as unknown as Record<string, unknown>;
  return {
    ...payload,
    ...(record.tmt !== undefined ? { tmt: parseTmt(record.tmt) ?? null } : {}),
    ...(record.deviation !== undefined ? { deviation: parseDeviation(record.deviation) ?? null } : {}),
  };
}

export async function triggerIntervention(intensity: InterventionIntensity, signal?: AbortSignal): Promise<InterventionTriggerResponse> {
  const response = await request<InterventionTriggerResponse>("/intervention/trigger", {
    method: "POST",
    body: JSON.stringify({ intensity }),
    signal,
  }, AI_REQUEST_TIMEOUT_MS);
  // The trigger payload carries the same contract §4 rationale as history.
  // Parsing it here keeps `InterventionRationale` meaning one thing: the
  // normalised shape, never a raw wire object with optional members.
  const rationale = response.intervention?.rationale;
  if (!rationale) return response;
  return {
    ...response,
    intervention: response.intervention
      ? { ...response.intervention, rationale: parseRationale(rationale) ?? null }
      : null,
  };
}
export async function respondIntervention(
  id: string,
  response: InterventionResponse,
  latencyS = 0,
  source: "human" | "auto" = "human",
): Promise<InterventionCommandResponse> {
  return unwrap(await withClientTimeout((options) => client.POST("/api/v1/intervention/{intervention_id}/response", { ...options, params: { path: { intervention_id: id } }, body: { response, latency_s: latencyS, source } })));
}
export async function feedbackIntervention(id: string, rating: InterventionRating, comment?: string): Promise<InterventionCommandResponse> {
  return unwrap(await withClientTimeout((options) => client.POST("/api/v1/intervention/{intervention_id}/feedback", { ...options, params: { path: { intervention_id: id } }, body: { rating, comment } })));
}
export async function getInterventionHistory(days = 7): Promise<InterventionHistoryResponse> {
  const raw = await unwrap(await withClientTimeout((options) => client.GET("/api/v1/intervention/history", { ...options, params: { query: { days } } }))) as unknown as InterventionHistoryResponse;
  return withRationale(raw);
}

/** Narrow each history item's optional `rationale` once, at the boundary. */
function withRationale(response: InterventionHistoryResponse): InterventionHistoryResponse {
  if (!Array.isArray(response?.items)) return response;
  return {
    ...response,
    items: response.items.map((item) => {
      const record = item as unknown as Record<string, unknown>;
      return record.rationale === undefined
        ? item
        : { ...item, rationale: parseRationale(record.rationale) ?? null };
    }),
  };
}

// ── Proactive-learning feedback requests (contract §5) ──
//
// 404 means "this backend build predates the feature". That is a normal,
// expected state during the parallel roll-out, so it is logged at debug level
// and surfaced as `undefined` instead of raising a user-visible error.

function isFeatureUnavailable(error: unknown): boolean {
  return error instanceof ApiError && (error.status === 404 || error.status === 501);
}

export async function getPendingFeedbackRequests(): Promise<FeedbackRequest[] | undefined> {
  try {
    const payload = await request<{ items?: unknown }>("/telemetry/feedback-requests/pending");
    if (!Array.isArray(payload?.items)) return [];
    return payload.items
      .map(parseFeedbackRequest)
      .filter((entry): entry is FeedbackRequest => entry !== undefined);
  } catch (error) {
    if (isFeatureUnavailable(error)) {
      console.debug("[feedback] 主动反馈接口不可用（404），跳过轮询兜底");
      return undefined;
    }
    throw error;
  }
}

export async function answerFeedbackRequest(requestId: string, label: FeedbackLabel): Promise<FeedbackAnswerResponse | undefined> {
  try {
    return await request<FeedbackAnswerResponse>(
      `/telemetry/feedback-requests/${encodeURIComponent(requestId)}/answer`,
      { method: "POST", body: JSON.stringify({ label }) },
    );
  } catch (error) {
    // A 404 on *answer* does not mean "feature absent" — the user is looking at
    // a prompt, so /pending must have answered. The backend reserves 404 for
    // "unknown, already answered, or evicted id" (routes/telemetry.py). No
    // label was written, so reporting success would claim a write that never
    // happened. Report it as `expired`: the request can no longer be answered,
    // which is exactly what the prompt already knows how to say.
    if (error instanceof ApiError && error.status === 404) {
      console.debug("[feedback] 反馈请求已失效（404），本次标注未写入", requestId);
      return { request_id: requestId, labeled_windows: 0, status: "expired" };
    }
    // 501 is the genuine "this build does not implement it" signal.
    if (error instanceof ApiError && error.status === 501) {
      console.debug("[feedback] 主动反馈接口不可用（501），已忽略本次作答");
      return undefined;
    }
    throw error;
  }
}

// ── Intervention execution: tasks (smart_prioritization) ──

export interface TaskItem {
  id: string;
  title: string;
  description: string;
  priority: number;
  status: "pending" | "in_progress" | "done";
  deadline_utc: string | null;
  estimated_minutes: number | null;
  created_at: string;
  updated_at: string;
}

export interface TaskListResponse {
  items: TaskItem[];
  count: number;
}

export interface TaskCreateInput {
  title: string;
  description?: string;
  priority?: number;
  status?: TaskItem["status"];
  deadline_utc?: string | null;
  estimated_minutes?: number | null;
}

export const getTasks = (status?: TaskItem["status"]) =>
  request<TaskListResponse>(`/tasks${status ? `?status=${encodeURIComponent(status)}` : ""}`);

export const createTask = (data: TaskCreateInput) =>
  request<TaskItem>("/tasks", { method: "POST", body: JSON.stringify(data) });

export const updateTask = (id: string, data: Partial<TaskCreateInput>) =>
  request<TaskItem>(`/tasks/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });

export const deleteTask = (id: string) =>
  request<{ status: string; task_id: string }>(`/tasks/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });

// ── Intervention execution: site blocking (environment_optimization) ──

export interface BlockedSiteItem {
  domain: string;
  enabled: boolean;
  reason: string | null;
  created_at: string;
}

export interface BlocklistResponse {
  items: BlockedSiteItem[];
  count: number;
}

export const getBlocklist = () =>
  request<BlocklistResponse>("/interventions/blocklist");

export const addBlockedSite = (domain: string, reason?: string) =>
  request<{ status: string; domain: string }>("/interventions/blocklist", {
    method: "POST",
    body: JSON.stringify({ domain, reason: reason || undefined }),
  });

export const toggleBlockedSite = (domain: string, enabled: boolean) =>
  request<{ status: string; domain: string }>(
    `/interventions/blocklist/${encodeURIComponent(domain)}`,
    { method: "PATCH", body: JSON.stringify({ enabled }) },
  );

export const removeBlockedSite = (domain: string) =>
  request<{ status: string; domain: string }>(
    `/interventions/blocklist/${encodeURIComponent(domain)}`,
    { method: "DELETE" },
  );

export const getCollectorStatus = () => request<CollectorStatus>("/collector");
export const startCollector = () => request<CollectorStatus>("/collector", { method: "POST" });
export const stopCollector = () => request<CollectorStatus>("/collector/stop", { method: "POST" });

export const getTelemetryStatus = () => request<TelemetryStatus>("/telemetry/status");
export const patchTelemetryPreferences = (data: Partial<TelemetryPreferences>) => request<TelemetryPreferences>("/telemetry/preferences", { method: "PATCH", body: JSON.stringify(data) });
export const createBrowserPairingCode = () => request<{ code: string; expires_at: string }>("/telemetry/browser/pairing-code", { method: "POST" });
export const clearTelemetryData = (scope: TelemetryDeleteScope) => request<TelemetryDeleteResponse>(`/telemetry/data?scope=${encodeURIComponent(scope)}`, { method: "DELETE" });
export const getPreferences = () => request<Preferences>("/preferences");
export const putPreferences = (data: Preferences) => request<Preferences>("/preferences", { method: "PUT", body: JSON.stringify(data) });
export const patchPreferences = (data: Preferences) => request<Preferences>("/preferences", { method: "PATCH", body: JSON.stringify(data) });
export const getClassifications = () => request<ClassificationRule[]>("/app-classifications");
export const addClassification = (data: ClassificationRuleInput) => request<ClassificationRule>("/app-classifications", { method: "POST", body: JSON.stringify(data) });
export const putClassifications = (data: ClassificationRuleInput[]) => request<ClassificationRule[]>("/app-classifications", { method: "PUT", body: JSON.stringify(data) });
export const deleteClassification = (id: string) =>
  request<unknown>(`/app-classifications/${encodeURIComponent(id)}`, { method: "DELETE" });
export const getUnknownApps = () => request<{ process_name: string; count: number; last_seen: string | null }[]>("/app-classifications/unknown-apps");
export const exportData = (fmt: "csv" | "json", start?: string, end?: string) => {
  const params = new URLSearchParams({ fmt });
  if (start) params.set("start", start);
  if (end) params.set("end", end);
  return requestText(`/export?${params.toString()}`);
};
export const getAutonomy = () => request<AutonomyStatus>("/autonomy");
export const pauseAutonomy = (hours: number) => request<AutonomyStatus>("/autonomy/pause", { method: "POST", body: JSON.stringify({ hours }) });
export const resumeAutonomy = () => request<AutonomyStatus>("/autonomy/resume", { method: "POST" });

// ── AI / Diagnostics ──

export interface AIRunItem {
  run_id: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  source: string;
  duration_ms: number | null;
}

/** `GET /ai/runs` — the backend's real envelope. There is no `total`: the
 *  route fills `count` with the total row count and `has_more` / `next_offset`
 *  carry the rest of the paging state. Kept hand-written because
 *  `components["schemas"]["DiagnosticsListResponse"]["items"]` (RunSummary)
 *  is not assignable to the `AIRunItem` this table renders. */
export interface AIRunsResponse {
  items: AIRunItem[];
  count: number;
  has_more: boolean;
  next_offset?: number | null;
}

export interface NodeEvent {
  name?: string;
  type?: string;
  status?: string;
  started_at?: string;
  completed_at?: string;
  duration_ms?: number;
  message?: string;
  [key: string]: unknown;
}

export interface AIRunDetail extends AIRunItem {
  node_events: NodeEvent[];
  error: string | null;
}

export type { FocusPredictionResponse, FocusPredictionStatus } from "./prediction-state";
export type { DailyReport, WeeklyReport } from "./report-state";
export type { BaselineSummary, BaselineSummaryState } from "./baseline-state";

/** `GET /health/live` (backend `api/routes/health.py`) — liveness only. */
export interface HealthLiveResponse {
  status: "alive";
  version: string;
  timestamp: string;
}

/** `GET /health/ready` — returns 503 when not ready, so a non-ready backend
 *  surfaces as a rejected request rather than as a 200 with a sad payload. */
export interface HealthReadyResponse {
  status: "ready" | "not_ready";
  version: string;
  timestamp: string;
  database: { status: "ok" | "error"; connected: boolean; integrity_ok: boolean };
  migration: { applied: boolean };
  checkpoint_store: "available" | "unavailable";
  run_store: "available" | "unavailable";
}

export const getAIRuns = (limit?: number, offset?: number) =>
  request<AIRunsResponse>(`/ai/runs?limit=${limit ?? 20}&offset=${offset ?? 0}`);

export const getAIRunDetail = (runId: string) =>
  request<AIRunDetail>(`/ai/runs/${encodeURIComponent(runId)}`);

export const getFocusPrediction = async (): Promise<FocusPredictionResponse> =>
  parseFocusPrediction(await request<unknown>("/telemetry/focus-prediction"));

export const getHealthLive = () =>
  request<HealthLiveResponse>("/health/live");

export const getAnalysisGraph = () =>
  request<AnalysisGraphTopology>("/ai/graph");

export interface AnalysisGraphTopology {
  nodes: string[];
  edges: { from: string; to: string }[];
  available: boolean;
}

export const getHealthReady = () =>
  request<HealthReadyResponse>("/health/ready");

// ── Model Center / Training APIs (types from generated schema) ──

type TrainingSchemas = components["schemas"];

export type JobStatus = TrainingSchemas["CreateTrainingJobResponse"]["status"];
export type GateStatus = TrainingSchemas["V2GateCheck"]["status"];
export type TrainingReadinessResponse = TrainingSchemas["TrainingReadinessResponse"];
export type CreateTrainingJobResponse = TrainingSchemas["CreateTrainingJobResponse"];
export type TrainingJobResponse = TrainingSchemas["TrainingJobResponse"];

export const getTrainingReadiness = () =>
  request<TrainingReadinessResponse>("/analytics/training-readiness");

export const createTrainingJob = () =>
  request<CreateTrainingJobResponse>("/analytics/training-jobs", { method: "POST" });

export const getTrainingJob = (jobId: string) =>
  request<TrainingJobResponse>(`/analytics/training-jobs/${encodeURIComponent(jobId)}`);

export const cancelTrainingJob = (jobId: string) =>
  request<TrainingJobResponse>(`/analytics/training-jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
