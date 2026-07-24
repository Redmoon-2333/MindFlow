const BASE = "/api/v1";

export interface HealthData {
  status: string;
  version: string;
  timestamp: string;
  collector: {
    status: string;
  };
  database: {
    status: "ok" | "error";
    connected: boolean;
  };
  migration: {
    applied: boolean;
  };
}

function getToken(): string | null {
  return localStorage.getItem("mindflow_token");
}

export function setToken(token: string) {
  localStorage.setItem("mindflow_token", token);
}

export function getTokenValue(): string | null {
  return getToken();
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${BASE}${url}`, {
    headers: { ...headers, ...((init?.headers as Record<string, string>) || {}) },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

async function requestText(url: string): Promise<string> {
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${BASE}${url}`, { headers });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.text();
}

// ── Auth (login) ──
export const getHealth = () => request<HealthData>("/health");

export const login = () =>
  fetch(`${BASE}/auth/login`, { method: "POST" }).then((r) => r.json());

// ── Activities ──
export const getActivities = (params?: Record<string, string>) =>
  request<{ items: any[]; total: number }>(`/activities?${new URLSearchParams(params)}`);
export const getCurrentActivity = () => request<any>("/activities/current");

// ── Focus ──
export const getFocusSessions = (date?: string) =>
  request<any>(`/focus${date ? `?date=${date}` : ""}`);
export const getFocusTrend = (days?: number) =>
  request<any>(`/focus/trend${days ? `?days=${days}` : ""}`);

// ── Reports ──
export const getDailyReport = (date?: string) =>
  request<any>(`/reports/daily${date ? `?date=${date}` : ""}`);
export const getWeeklyReport = (weekStart?: string) =>
  request<any>(`/reports/weekly${weekStart ? `?week_start=${weekStart}` : ""}`);

// ── Analytics ──
export const getAnalyticsPatterns = (days?: number) =>
  request<any>(`/analytics/patterns${days ? `?days=${days}` : ""}`);
export const getBaseline = () => request<any>("/analytics/baseline");
export const getProfile = (days?: number) =>
  request<any>(`/analytics/profile${days ? `?days=${days}` : ""}`);
export const getModelStatus = () => request<any>("/analytics/model-status");
export const runAttribution = (body?: { date?: string; force?: boolean }) =>
  request<any>("/analytics/attribution", { method: "POST", body: JSON.stringify(body || {}) });

// ── Chat ──
export const sendChat = (message: string, sessionId?: string) =>
  request<any>("/chat", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
  });
export const getChatSessions = () => request<any[]>("/chat/sessions");
export const getChatMessages = (sessionId: string) =>
  request<any[]>(`/chat/${sessionId}/messages`);

// ── Panel ──
export const triggerPanel = () =>
  request<any>("/panel/today", { method: "POST" });
export const getPanelResult = () => request<any>("/panel");

// ── Intervention ──
export const triggerIntervention = (intensity: string) =>
  request<any>("/intervention/trigger", {
    method: "POST",
    body: JSON.stringify({ intensity }),
  });
export const respondIntervention = (id: number, response: string, latencyS?: number) =>
  request<any>(`/intervention/${id}/response`, {
    method: "POST",
    body: JSON.stringify({ response, latency_s: latencyS }),
  });
export const feedbackIntervention = (id: number, rating: string, comment?: string) =>
  request<any>(`/intervention/${id}/feedback`, {
    method: "POST",
    body: JSON.stringify({ rating, comment }),
  });
export const getInterventionHistory = (days?: number) =>
  request<any>(`/intervention/history${days ? `?days=${days}` : ""}`);

// ── Collector ──
export const getCollectorStatus = () => request<any>("/collector");
export const startCollector = () => request<any>("/collector", { method: "POST" });
export const stopCollector = () => request<any>("/collector/stop", { method: "POST" });

// ── Preferences ──
export const getPreferences = () => request<Record<string, any>>("/preferences");
export const putPreferences = (data: any) =>
  request<any>("/preferences", { method: "PUT", body: JSON.stringify(data) });
export const patchPreferences = (data: any) =>
  request<any>("/preferences", { method: "PATCH", body: JSON.stringify(data) });

// ── App Classifications ──
export const getClassifications = () => request<any[]>("/app-classifications");
export const addClassification = (data: any) =>
  request<any>("/app-classifications", { method: "POST", body: JSON.stringify(data) });
export const putClassifications = (data: any[]) =>
  request<any>("/app-classifications", { method: "PUT", body: JSON.stringify(data) });
export const deleteClassification = (id: number) =>
  fetch(`${BASE}/app-classifications/${id}`, { method: "DELETE" });
export const getUnknownApps = () => request<string[]>("/app-classifications/unknown-apps");

// ── Export ──
export const exportData = (fmt: "csv" | "json", start?: string, end?: string) =>
  requestText(`/export?fmt=${fmt}&start=${start || ""}&end=${end || ""}`);

// ── Autonomy ──
export const getAutonomy = () => request<any>("/autonomy");
export const pauseAutonomy = (hours: number) =>
  request<any>("/autonomy/pause", { method: "POST", body: JSON.stringify({ hours }) });
export const resumeAutonomy = () => request<any>("/autonomy/resume", { method: "POST" });
