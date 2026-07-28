import createClient from "openapi-fetch";
import type { components, paths } from "./generated/api-schema";

const BASE = "/api/v1";
const AUTH_MARKER = "mindflow_authenticated";
export const AUTH_REQUIRED_EVENT = "mindflow:auth-required";
const client = createClient<paths>();

type CollectorStatusResponse = components["schemas"]["CollectorStatusResponse"];
type ChatResponse = components["schemas"]["ChatResponse"];
type PanelResponse = components["schemas"]["PanelResponse"];
type InterventionTriggerResponse = components["schemas"]["InterventionTriggerResponse"];
type InterventionPayload = components["schemas"]["InterventionPayload"];
type InterventionCommandResponse = components["schemas"]["InterventionCommandResponse"];
type InterventionIntensity = components["schemas"]["InterventionTriggerRequest"]["intensity"];
type InterventionResponse = components["schemas"]["InterventionResponseRequest"]["response"];
type InterventionRating = components["schemas"]["InterventionFeedbackRequest"]["rating"];
type ActivityQuery = NonNullable<paths["/api/v1/activities"]["get"]["parameters"]["query"]>;

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
  collector: { status: string };
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
  day?: string;
  focus_minutes?: number;
  focus?: number;
  distraction_minutes?: number;
  distraction?: number;
}

export interface FocusTrendResponse {
  days: number;
  start_date: string;
  end_date: string;
  daily: FocusTrendDay[];
  daily_data?: FocusTrendDay[];
  total_sessions: number;
  today_minutes?: number;
  total_minutes?: number;
  trend_label?: string;
  session_count?: number;
  avg_duration_minutes?: number;
  avg_score?: number;
  score_change?: number;
  distraction_rate?: number;
  distraction_label?: string;
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

export interface DailyReport {
  id?: string;
  user_id: number;
  date: string;
  total_focus_min: number;
  total_distraction_min: number;
  focus_score: number;
  top_apps: Array<{ app: string; minutes: number }>;
  switch_frequency: number;
  pattern_summary: string;
  created_at?: string;
  updated_at?: string;
  total_focus_minutes?: number;
  total_sessions?: number;
  total_distractions?: number;
  hourly_distribution?: Record<string, number>;
  app_usage?: Array<{
    app?: string;
    name?: string;
    duration_minutes?: number;
    category?: string;
  }>;
  distraction_analysis?: Array<{
    type?: string;
    name?: string;
    count?: number;
    total_duration?: number;
  }>;
}

export interface WeeklyReport {
  week_start: string;
  week_end: string;
  daily_reports: DailyReport[];
  averages: {
    avg_focus_min?: number;
    avg_distraction_min?: number;
    avg_focus_score?: number;
    avg_switch_frequency?: number;
  };
  trend: {
    focus_min_delta_pct?: number;
    focus_score_delta?: number;
    direction?: "up" | "down" | "stable";
  };
  week_number: number;
  intervention_effectiveness?: Record<string, unknown> | null;
  total_focus_minutes?: number;
  total_sessions?: number;
  total_distractions?: number;
  avg_focus_score?: number;
  daily_summary?: Array<{
    date: string;
    focus_minutes?: number;
    sessions?: number;
    distractions?: number;
    focus_score?: number;
  }>;
  week_over_week?: Record<string, number | null | undefined>;
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

export interface BaselineSummary {
  user_id: number;
  created_at: string;
  updated_at: string;
  total_days: number;
  total_samples: number;
  features: string[];
  avg_focus_min?: number;
  avg_switches_per_day?: number;
  productivity_score?: number;
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
  mode: string;
  v2_mode: string;
  message: string;
  feature_schema_version?: number;
  version?: string | null;
  available_versions?: string[];
  reasons?: string[];
  model_name?: string;
  last_updated?: string;
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
    cognitive_distortions: string[];
    cbt_technique: string | null;
    response_text: string;
    next_action: string;
  };
  source: "deepseek" | "ollama" | "rule_engine";
  cached: boolean;
  meta: { degraded: boolean };
  results?: AttributionResult[];
  confidence?: string | number;
  procrastination_type?: string;
  cbt_technique?: string;
  evidence?: string;
}

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

export function hasAuthenticatedSession(): boolean {
  return localStorage.getItem(AUTH_MARKER) === "1";
}

function requestOptions() {
  return { credentials: "include" as const };
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
  return error instanceof Error && error.message ? error.message : fallback;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const res = await fetch(`${BASE}${url}`, { credentials: "include", ...init, headers: { ...headers, ...((init?.headers as Record<string, string>) || {}) } });
  if (!res.ok) throw await toApiError(res);
  return res.json() as Promise<T>;
}

async function requestText(url: string): Promise<string> {
  const res = await fetch(`${BASE}${url}`, { credentials: "include" });
  if (!res.ok) throw await toApiError(res);
  return res.text();
}

export async function bootstrapFromFragment(): Promise<boolean> {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const ticket = params.get("bootstrap");
  if (!ticket) return false;
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  const result = await client.POST("/api/v1/auth/bootstrap", { ...requestOptions(), body: { ticket } });
  if (result.error !== undefined || !result.response.ok) throw await toApiError(result.response, result.error);
  localStorage.setItem(AUTH_MARKER, "1");
  return true;
}

export const getHealth = () => request<HealthData>("/health");

function isActivityItem(value: unknown): value is ActivityItem {
  return isRecord(value) && typeof value.id === "string" && typeof value.timestamp === "string" && typeof value.duration_s === "number" && typeof value.event_type === "string" && isRecord(value.data) && typeof value.data.app_name === "string" && typeof value.data.window_title === "string" && typeof value.data.process_name === "string" && typeof value.data.is_idle === "boolean";
}

export async function getActivities(params: ActivityQuery = {}): Promise<ActivitiesResponse> {
  const raw = await unwrap<unknown>(await client.GET("/api/v1/activities", { ...requestOptions(), params: { query: params } }));
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
    const raw = await unwrap<unknown>(await client.GET("/api/v1/activities/current", requestOptions()));
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
export const getDailyReport = (date?: string) => request<DailyReport>(`/reports/daily${date ? `?date=${date}` : ""}`);
export const getWeeklyReport = (weekStart?: string) => request<WeeklyReport>(`/reports/weekly${weekStart ? `?week_start=${weekStart}` : ""}`);
export const getAnalyticsPatterns = (days?: number) => request<AnalyticsPatterns>(`/analytics/patterns${days ? `?days=${days}` : ""}`);
export const getBaseline = () => request<BaselineSummary>("/analytics/baseline");
export const getProfile = (days?: number) => request<BehavioralProfile>(`/analytics/profile${days ? `?days=${days}` : ""}`);
export const getModelStatus = () => request<ModelStatus>("/analytics/model-status");
export const runAttribution = (body?: { date?: string; force?: boolean }) => request<AttributionResponse>("/analytics/attribution", { method: "POST", body: JSON.stringify(body || {}) });

export async function sendChat(message: string, sessionId?: string): Promise<ChatReply> {
  return unwrap(await client.POST("/api/v1/chat", { ...requestOptions(), body: { message, session_id: sessionId } }));
}
export async function getChatSessions(): Promise<ChatSession[]> {
  return unwrap(await client.GET("/api/v1/chat/sessions", requestOptions())) as unknown as ChatSession[];
}
export async function getChatMessages(sessionId: string): Promise<ChatMessageRecord[]> {
  return unwrap(await client.GET("/api/v1/chat/{session_id}/messages", { ...requestOptions(), params: { path: { session_id: sessionId } } })) as unknown as ChatMessageRecord[];
}

export const triggerPanel = async (): Promise<PanelResult> => unwrap(await client.POST("/api/v1/panel/today", requestOptions()));
export const getPanelResult = async (): Promise<PanelResult> => unwrap(await client.GET("/api/v1/panel", requestOptions()));

export async function triggerIntervention(intensity: InterventionIntensity): Promise<InterventionTriggerResponse> {
  return unwrap(await client.POST("/api/v1/intervention/trigger", { ...requestOptions(), body: { intensity } }));
}
export async function respondIntervention(id: string, response: InterventionResponse, latencyS = 0): Promise<InterventionCommandResponse> {
  return unwrap(await client.POST("/api/v1/intervention/{intervention_id}/response", { ...requestOptions(), params: { path: { intervention_id: id } }, body: { response, latency_s: latencyS } }));
}
export async function feedbackIntervention(id: string, rating: InterventionRating, comment?: string): Promise<InterventionCommandResponse> {
  return unwrap(await client.POST("/api/v1/intervention/{intervention_id}/feedback", { ...requestOptions(), params: { path: { intervention_id: id } }, body: { rating, comment } }));
}
export async function getInterventionHistory(days = 7): Promise<InterventionHistoryResponse> {
  return unwrap(await client.GET("/api/v1/intervention/history", { ...requestOptions(), params: { query: { days } } })) as unknown as InterventionHistoryResponse;
}

export const getCollectorStatus = async (): Promise<CollectorStatus> => unwrap(await client.GET("/api/v1/collector", requestOptions()));
export const startCollector = async (): Promise<CollectorStatus> => unwrap(await client.POST("/api/v1/collector", requestOptions()));
export const stopCollector = async (): Promise<CollectorStatus> => unwrap(await client.POST("/api/v1/collector/stop", requestOptions()));

export const getTelemetryStatus = () => request<TelemetryStatus>("/telemetry/status");
export const patchTelemetryPreferences = (data: Partial<TelemetryPreferences>) => request<TelemetryPreferences>("/telemetry/preferences", { method: "PATCH", body: JSON.stringify(data) });
export const createBrowserPairingCode = () => request<{ code: string; expires_at: string }>("/telemetry/browser/pairing-code", { method: "POST" });
export const clearTelemetryData = (scope: "interaction" | "browser" | "feedback" | "all") => request<{ deleted: number }>(`/telemetry/data?scope=${scope}`, { method: "DELETE" });
export const getPreferences = () => request<Preferences>("/preferences");
export const putPreferences = (data: Preferences) => request<Preferences>("/preferences", { method: "PUT", body: JSON.stringify(data) });
export const patchPreferences = (data: Preferences) => request<Preferences>("/preferences", { method: "PATCH", body: JSON.stringify(data) });
export const getClassifications = () => request<ClassificationRule[]>("/app-classifications");
export const addClassification = (data: ClassificationRuleInput) => request<ClassificationRule>("/app-classifications", { method: "POST", body: JSON.stringify(data) });
export const putClassifications = (data: ClassificationRuleInput[]) => request<ClassificationRule[]>("/app-classifications", { method: "PUT", body: JSON.stringify(data) });
export const deleteClassification = (id: string) => fetch(`${BASE}/app-classifications/${id}`, { method: "DELETE", credentials: "include" });
export const getUnknownApps = () => request<string[]>("/app-classifications/unknown-apps");
export const exportData = (fmt: "csv" | "json", start?: string, end?: string) => requestText(`/export?fmt=${fmt}&start=${start || ""}&end=${end || ""}`);
export const getAutonomy = () => request<AutonomyStatus>("/autonomy");
export const pauseAutonomy = (hours: number) => request<AutonomyStatus>("/autonomy/pause", { method: "POST", body: JSON.stringify({ hours }) });
export const resumeAutonomy = () => request<AutonomyStatus>("/autonomy/resume", { method: "POST" });
