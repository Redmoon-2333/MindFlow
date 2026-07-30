import { useState, useEffect, useCallback } from "react";
import { getErrorMessage, getFocusSessions, getFocusTrend, submitFocusFeedback } from "../api";
import type { FocusSession, FocusTrendDay, FocusTrendResponse } from "../api";

type CompatibleTrendDay = Partial<FocusTrendDay>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isCompatibleTrendDay(value: unknown): value is CompatibleTrendDay {
  if (!isRecord(value)) return false;
  const numberFields = ["focus_min", "focus_minutes", "focus", "distraction_min", "distraction_minutes", "distraction", "session_count", "avg_score"];
  return (value.date == null || typeof value.date === "string")
    && (value.day == null || typeof value.day === "string")
    && numberFields.every((field) => value[field] == null || typeof value[field] === "number");
}

function getTrendDays(value: unknown): CompatibleTrendDay[] {
  if (!isRecord(value)) return [];
  for (const key of ["daily", "days", "daily_data"]) {
    const candidate = value[key];
    if (Array.isArray(candidate)) return candidate.filter(isCompatibleTrendDay);
  }
  return [];
}

interface FeedbackDraft {
  label: "focus" | "distracted" | "mixed";
  score: number;
  taskType: string;
}

function formatMinutes(m: number | null | undefined): string {
  if (m == null || isNaN(m)) return "—";
  const h = Math.floor(m / 60);
  const min = Math.round(m % 60);
  return h > 0 ? `${h}h ${min}m` : `${min}m`;
}

function sessionDurationMinutes(session: FocusSession): number {
  if (session.duration_minutes != null) return Number(session.duration_minutes);
  if (session.duration != null) return Number(session.duration);
  const start = new Date(session.start_time ?? session.started_at ?? "");
  const end = new Date(session.end_time ?? session.ended_at ?? "");
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return 0;
  return Math.max(0, (end.getTime() - start.getTime()) / 60000);
}

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

function dayLabel(dateStr: string, short?: boolean): string {
  const names = short
    ? ["日", "一", "二", "三", "四", "五", "六"]
    : ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
  const d = new Date(dateStr);
  return names[d.getDay()];
}

export default function Focus() {
  const [date, setDate] = useState(todayStr());
  const [sessions, setSessions] = useState<FocusSession[]>([]);
  const [trend, setTrend] = useState<FocusTrendResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [trendLoading, setTrendLoading] = useState(false);
  const [error, setError] = useState("");
  const [feedbackDrafts, setFeedbackDrafts] = useState<Record<string, FeedbackDraft>>({});
  const [feedbackSaving, setFeedbackSaving] = useState<string | null>(null);
  const [feedbackSaved, setFeedbackSaved] = useState<Set<string>>(new Set());

  const loadSessions = useCallback(async (d: string) => {
    setLoading(true);
    setError("");
    try {
      const data = await getFocusSessions(d);
      setSessions(data.sessions);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "加载失败"));
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadTrend = useCallback(async () => {
    setTrendLoading(true);
    try {
      const data = await getFocusTrend(7);
      setTrend(data);
    } catch {
      // trend is optional, don't block on failure
    } finally {
      setTrendLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions(date);
    loadTrend();
  }, [date, loadSessions, loadTrend]);

  const totalFocus = sessions.reduce((sum, session) => sum + sessionDurationMinutes(session), 0);
  const sessionCount = sessions.length;
  const avgScore =
    sessionCount > 0
      ? sessions.reduce((sum, session) => sum + Number(session.focus_score ?? session.score ?? 0), 0) / sessionCount
      : 0;
  const longestBlock = sessions.reduce(
    (maximum, session) => Math.max(maximum, sessionDurationMinutes(session)),
    0,
  );

  const trendDays = getTrendDays(trend);
  const maxFocus = Math.max(1, ...trendDays.map((day) => day.focus_min ?? day.focus_minutes ?? day.focus ?? 0));
  const maxDistraction = Math.max(1, ...trendDays.map((day) => day.distraction_min ?? day.distraction_minutes ?? day.distraction ?? 0));
  const chartMax = Math.max(maxFocus, maxDistraction);

  const saveFeedback = async (sessionId: string) => {
    const draft = feedbackDrafts[sessionId] ?? { label: "mixed", score: 3, taskType: "" };
    setFeedbackSaving(sessionId);
    setError("");
    try {
      await submitFocusFeedback(sessionId, {
        label: draft.label,
        score: draft.score,
        task_type: draft.taskType || undefined,
      });
      setFeedbackSaved((current) => new Set(current).add(sessionId));
    } catch (e: unknown) {
      setError(getErrorMessage(e, "反馈保存失败"));
    } finally {
      setFeedbackSaving(null);
    }
  };

  return (
    <div>
      <div className="header">
        <h1>专注分析</h1>
        <p>查看专注会话与趋势，了解你的注意力模式</p>
      </div>

      <div className="flex flex-between mb24">
        <div className="flex gap8">
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            style={{ width: 180 }}
          />
          <button className="btn btn-ghost" onClick={() => { loadSessions(date); loadTrend(); }}>
            刷新
          </button>
        </div>
      </div>

      {error && <div className="error-box mb16">{error}</div>}

      <div className="kpi-row mb24">
        <div className="stat-card">
          <div className="label">总专注时长</div>
          <div className="value">{formatMinutes(totalFocus)}</div>
        </div>
        <div className="stat-card">
          <div className="label">专注次数</div>
          <div className="value">{sessionCount || "—"}</div>
        </div>
        <div className="stat-card">
          <div className="label">平均评分</div>
          <div className="value">{sessionCount > 0 ? avgScore.toFixed(1) : "—"}</div>
        </div>
        <div className="stat-card">
          <div className="label">最长专注</div>
          <div className="value">{longestBlock > 0 ? formatMinutes(longestBlock) : "—"}</div>
        </div>
      </div>

      <div className="card mb24" style={{ borderLeft: "3px solid var(--color-primary)", background: "var(--color-primary-light)" }}>
        <div style={{ fontSize: 13, color: "var(--color-text-secondary)", lineHeight: 1.6 }}>
          <strong>自动分析说明：</strong>专注评分由后端 ML 模型自动计算，无需手动反馈。
          评分基于应用切换频率、专注时长、应用类型等特征。
          下方的"反馈"功能仅用于收集训练数据以改进模型精度，非必须操作。
        </div>
      </div>

      <div className="card mb24">
        <h3>7 天专注趋势</h3>
        {trendLoading && <div className="spinner" />}
        {!trendLoading && trendDays.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--color-text-tertiary)", padding: 40 }}>
            暂无趋势数据
          </div>
        )}
        {!trendLoading && trendDays.length > 0 && (
          <div style={{ display: "flex", alignItems: "flex-end", gap: 16, height: 200, paddingTop: 8 }}>
            {trendDays.map((d, i) => {
              const focusVal = d.focus_min ?? d.focus_minutes ?? d.focus ?? 0;
              const distVal = d.distraction_min ?? d.distraction_minutes ?? d.distraction ?? 0;
              const focusPct = Math.round((focusVal / chartMax) * 100);
              const distPct = Math.round((distVal / chartMax) * 100);
              const dateStr = d.date ?? d.day ?? "";
              return (
                <div
                  key={i}
                  style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", height: "100%" }}
                >
                  <span style={{ fontSize: 10, minHeight: 16 }}>
                    {focusVal > 0 ? formatMinutes(focusVal) : ""}
                  </span>
                  <div style={{ display: "flex", gap: 3, width: "100%", maxWidth: 56, justifyContent: "center", flex: 1 }}>
                    <div
                      style={{
                        width: 18,
                        height: `${Math.max(focusPct, focusVal > 0 ? 3 : 0)}%`,
                        background: dateStr === date ? "var(--color-primary)" : "var(--color-border)",
                        borderRadius: "4px 4px 0 0",
                        transition: "height 0.3s",
                        alignSelf: "flex-end",
                      }}
                      title={`专注 ${formatMinutes(focusVal)}`}
                    />
                    <div
                      style={{
                        width: 18,
                        height: `${Math.max(distPct, distVal > 0 ? 3 : 0)}%`,
                        background: dateStr === date ? "#fbbf24" : "#e2e8f0",
                        borderRadius: "4px 4px 0 0",
                        transition: "height 0.3s",
                        alignSelf: "flex-end",
                      }}
                      title={`分心 ${formatMinutes(distVal)}`}
                    />
                  </div>
                  <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 6 }}>
                    {dayLabel(dateStr, true)}
                  </span>
                  <span style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>
                    {dateStr ? dateStr.slice(5) : ""}
                  </span>
                </div>
              );
            })}
          </div>
        )}
        {!trendLoading && trendDays.length > 0 && (
          <div className="flex gap16" style={{ justifyContent: "center", marginTop: 8 }}>
            <div className="flex gap8" style={{ alignItems: "center", fontSize: 12, color: "var(--color-text-secondary)" }}>
              <span style={{ width: 12, height: 12, borderRadius: 3, background: "var(--color-primary)", display: "inline-block" }} />
              专注
            </div>
            <div className="flex gap8" style={{ alignItems: "center", fontSize: 12, color: "var(--color-text-secondary)" }}>
              <span style={{ width: 12, height: 12, borderRadius: 3, background: "#fbbf24", display: "inline-block" }} />
              分心
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h3>专注会话</h3>
        {loading && <div className="spinner" />}
        {!loading && sessions.length === 0 && (
          <div style={{ textAlign: "center", color: "var(--color-text-tertiary)", padding: 40 }}>
            暂无专注会话数据
          </div>
        )}
        {!loading && sessions.length > 0 && (
          <div className="flex gap16" style={{ flexDirection: "column" }}>
            {sessions.map((session, index) => {
              const sessionId = String(session.id ?? index);
              const startTime = session.start_time ?? session.started_at ?? "";
              const sessionDate = session.date ?? String(startTime).slice(0, 10);
              const duration = sessionDurationMinutes(session);
              const app = session.dominant_app ?? session.main_app ?? session.app ?? session.app_name ?? "—";
                            const score = session.focus_score ?? session.score;
              const sessionType = score != null ? (score >= 60 ? "focus" : score >= 35 ? "neutral" : "distraction") : null;
              const sessionTypeLabel = sessionType === "focus" ? "专注" : sessionType === "neutral" ? "中性" : sessionType === "distraction" ? "分心" : null;
              const sessionTypeClass = sessionType === "focus" ? "badge-success" : sessionType === "neutral" ? "badge-info" : sessionType === "distraction" ? "badge-danger" : "badge-warning";
              const switches = session.switch_count ?? session.switches ?? 0;
              const draft = feedbackDrafts[sessionId] ?? { label: "mixed", score: 3, taskType: "" };
              return (
                <div
                  key={sessionId}
                  style={{
                    padding: "14px 0",
                    borderBottom: index < sessions.length - 1 ? "1px solid var(--color-border)" : "none",
                  }}
                >
                  <div className="flex flex-between" style={{ gap: 18, flexWrap: "wrap" }}>
                    <div className="flex gap16" style={{ alignItems: "center", flexWrap: "wrap" }}>
                      <div style={{ minWidth: 100 }}>
                        <div style={{ fontSize: 13, fontWeight: 500 }}>{sessionDate ? sessionDate.slice(5) : "—"}</div>
                        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
                          {sessionDate ? dayLabel(sessionDate) : ""} {startTime ? new Date(startTime).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) : ""}
                        </div>
                      </div>
                      <div style={{ minWidth: 80 }}><div style={{ fontSize: 14, fontWeight: 600 }}>{formatMinutes(duration)}</div><div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>会话时长</div></div>
                      <span className="badge badge-primary">{app}</span>
                    </div>
                    <div className="flex gap16" style={{ alignItems: "center" }}>
                                            {score != null && <span className={`badge ${score >= 60 ? "badge-success" : score >= 35 ? "badge-info" : "badge-danger"}`}>{Math.round(score)}分</span>}
                      {sessionTypeLabel && <span className={`badge ${sessionTypeClass}`}>{sessionTypeLabel}</span>}
                      {(session as any).feedback_label && (
                        <span className="badge badge-primary" title={`已标记: ${(session as any).feedback_label} (${(session as any).feedback_score}/5)`}>
                          已标记: {(session as any).feedback_label}
                        </span>
                      )}
                      <div style={{ textAlign: "center", minWidth: 60 }}><div style={{ fontSize: 13, fontWeight: 500 }}>{switches}</div><div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>切换次数</div></div>
                    </div>
                  </div>
                  <div style={{ marginTop: 12, padding: 12, borderRadius: 10, background: "var(--color-bg-secondary)" }}>
                    <div className="flex flex-between" style={{ gap: 12, flexWrap: "wrap", alignItems: "end" }}>
                      <div className="form-group" style={{ margin: 0, minWidth: 150 }}><label>这次状态</label><select value={draft.label} onChange={(event) => setFeedbackDrafts((current) => ({ ...current, [sessionId]: { ...draft, label: event.target.value as FeedbackDraft["label"] } }))}><option value="focus">专注</option><option value="distracted">分心</option><option value="mixed">混合</option></select></div>
                      <div className="form-group" style={{ margin: 0, minWidth: 150 }}><label>自评分数</label><select value={draft.score} onChange={(event) => setFeedbackDrafts((current) => ({ ...current, [sessionId]: { ...draft, score: Number(event.target.value) } }))}>{[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value} 分</option>)}</select></div>
                      <div className="form-group" style={{ margin: 0, minWidth: 180 }}><label>任务类型（可选）</label><select value={draft.taskType} onChange={(event) => setFeedbackDrafts((current) => ({ ...current, [sessionId]: { ...draft, taskType: event.target.value } }))}><option value="">未选择</option><option value="coding">编程</option><option value="writing">写作</option><option value="study">学习</option><option value="meeting">会议</option><option value="admin">事务</option><option value="creative">创作</option><option value="other">其他</option></select></div>
                      <button className="btn btn-primary btn-sm" disabled={feedbackSaving === sessionId} onClick={() => saveFeedback(sessionId)}>{feedbackSaving === sessionId ? "保存中..." : feedbackSaved.has(sessionId) ? "已保存，可更新" : "保存反馈"}</button>
                    </div>
                    <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 8 }}>1–2 分用于分心标签，4–5 分用于专注标签，3 分或混合只用于不确定性评估。</div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
