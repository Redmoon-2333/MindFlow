import { useState, useEffect, useCallback } from "react";
import { getFocusSessions, getFocusTrend } from "../api";

function formatMinutes(m: number): string {
  if (m == null || isNaN(m)) return "—";
  const h = Math.floor(m / 60);
  const min = Math.round(m % 60);
  return h > 0 ? `${h}h ${min}m` : `${min}m`;
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
  const [sessions, setSessions] = useState<any[]>([]);
  const [trend, setTrend] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [trendLoading, setTrendLoading] = useState(false);
  const [error, setError] = useState("");

  const loadSessions = useCallback(async (d: string) => {
    setLoading(true);
    setError("");
    try {
      const data = await getFocusSessions(d);
      const list = Array.isArray(data) ? data : data?.sessions ?? data?.items ?? [];
      setSessions(list);
    } catch (e: any) {
      setError(e.message || "加载失败");
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

  const totalFocus = sessions.reduce((sum: number, s: any) => sum + (s.duration_minutes || s.duration || 0), 0);
  const sessionCount = sessions.length;
  const avgScore =
    sessionCount > 0
      ? sessions.reduce((sum: number, s: any) => sum + (s.score ?? 0), 0) / sessionCount
      : 0;
  const longestBlock = sessions.reduce(
    (max: number, s: any) => Math.max(max, s.duration_minutes || s.duration || 0),
    0,
  );

  const trendDays = trend?.days ?? trend?.daily_data ?? [];
  const maxFocus = Math.max(1, ...trendDays.map((d: any) => d.focus_minutes ?? d.focus ?? 0));
  const maxDistraction = Math.max(1, ...trendDays.map((d: any) => d.distraction_minutes ?? d.distraction ?? 0));
  const chartMax = Math.max(maxFocus, maxDistraction);

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
            {trendDays.map((d: any, i: number) => {
              const focusVal = d.focus_minutes ?? d.focus ?? 0;
              const distVal = d.distraction_minutes ?? d.distraction ?? 0;
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
            {sessions.map((s: any, i: number) => {
              const sessionDate = s.date ?? s.started_at?.slice(0, 10) ?? "";
              const duration = s.duration_minutes ?? s.duration ?? 0;
              const app = s.main_app ?? s.app ?? s.app_name ?? "—";
              const score = s.score;
              const switches = s.switch_count ?? s.switches ?? 0;
              return (
                <div
                  key={s.id ?? i}
                  className="flex flex-between"
                  style={{
                    padding: "12px 0",
                    borderBottom: i < sessions.length - 1 ? "1px solid var(--color-border)" : "none",
                  }}
                >
                  <div className="flex gap16" style={{ alignItems: "center" }}>
                    <div style={{ minWidth: 80 }}>
                      <div style={{ fontSize: 13, fontWeight: 500 }}>
                        {sessionDate ? sessionDate.slice(5) : "—"}
                      </div>
                      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
                        {sessionDate ? dayLabel(sessionDate) : ""}
                      </div>
                    </div>
                    <div style={{ minWidth: 80 }}>
                      <div style={{ fontSize: 14, fontWeight: 600 }}>{formatMinutes(duration)}</div>
                      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>专注时长</div>
                    </div>
                    <div>
                      <span className="badge badge-primary">{app}</span>
                    </div>
                  </div>
                  <div className="flex gap16" style={{ alignItems: "center" }}>
                    {score != null && (
                      <div style={{ textAlign: "center" }}>
                        <div
                          className={`badge ${
                            score >= 80 ? "badge-success" : score >= 50 ? "badge-warning" : "badge-danger"
                          }`}
                        >
                          {Math.round(score)}分
                        </div>
                      </div>
                    )}
                    <div style={{ textAlign: "center", minWidth: 60 }}>
                      <div style={{ fontSize: 13, fontWeight: 500 }}>{switches}</div>
                      <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>切换次数</div>
                    </div>
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
