export interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
  timestamp: number;
}

export interface ActivityInfo {
  process_name: string;
  window_title: string;
  timestamp: number;
}

export interface CollectorStatus {
  running: boolean;
  current_activity: ActivityInfo | null;
  uptime_seconds: number;
}

export interface AppUsageItem {
  app: string;
  minutes: number;
}

export interface FocusReport {
  date: string;
  total_focus_minutes: number;
  total_distraction_minutes: number;
  focus_score: number;
  top_apps: AppUsageItem[];
  switch_frequency: number;
}

export interface TrendDataPoint {
  date: string;
  focus_score: number;
  focus_minutes: number;
}

export interface TodayReport {
  focus_report: FocusReport;
  collector_status: CollectorStatus;
}

export interface Preferences {
  collector_enabled: boolean;
  focus_apps: string[];
  distraction_apps: string[];
  session_duration_minutes: number;
  break_duration_minutes: number;
}
