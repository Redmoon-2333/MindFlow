import { useState, useEffect, useCallback } from "react";
import { getDailyReport, getWeeklyReport } from "../api";

type Tab = "daily" | "weekly";

function formatMinutes(m: number): string {
  if (m == null || isNaN(m)) return "—";
  const h = Math.floor(m / 60);
  const min = Math.round(m % 60);
  return h > 0 ? `${h}h ${min}m` : `${min}m`;
}

function fmtDate(d: string): string {
  if (!d) return "—";
  const dt = new Date(d);
  return `${dt.getMonth() + 1}/${dt.getDate()}`;
}

function mondayOf(date: Date): string {
  const d = new Date(date);
  d.setDate(d.getDate() - ((d.getDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

function dayLabel(dateStr: string): string {
  const names = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
  const d = new Date(dateStr);
  return names[d.getDay()];
}

export default function Reports() {
  const [tab, setTab] = useState<Tab>("daily");

  // Daily state
  const [dailyDate, setDailyDate] = useState(todayStr());
  const [daily, setDaily] = useState<any>(null);
  const [dailyLoading, setDailyLoading] = useState(false);
  const [dailyErr, setDailyErr] = useState("");

  // Weekly state
  const [weekStart, setWeekStart] = useState(mondayOf(new Date()));
  const [weekly, setWeekly] = useState<any>(null);
  const [weeklyLoading, setWeeklyLoading] = useState(false);
  const [weeklyErr, setWeeklyErr] = useState("");

  const loadDaily = useCallback(async (date: string) => {
    setDailyLoading(true);
    setDailyErr("");
    try {
      const data = await getDailyReport(date);
      setDaily(data);
    } catch (e: any) {
      setDailyErr(e.message || "加载失败");
      setDaily(null);
    } finally {
      setDailyLoading(false);
    }
  }, []);

  const loadWeekly = useCallback(async (ws: string) => {
    setWeeklyLoading(true);
    setWeeklyErr("");
    try {
      const data = await getWeeklyReport(ws);
      setWeekly(data);
    } catch (e: any) {
      setWeeklyErr(e.message || "加载失败");
      setWeekly(null);
    } finally {
      setWeeklyLoading(false);
    }
  }, []);

  useEffect(() => {
    if (tab === "daily") loadDaily(dailyDate);
  }, [tab, dailyDate, loadDaily]);

  useEffect(() => {
    if (tab === "weekly") loadWeekly(weekStart);
  }, [tab, weekStart, loadWeekly]);

  const hourlyDist = daily?.hourly_distribution || {};
  const maxHourVal = Math.max(1, ...Object.values(hourlyDist).map(Number));

  const dailySummary = weekly?.daily_summary || [];
  const maxDayFocus = Math.max(1, ...dailySummary.map((d: any) => d.focus_minutes || 0));

  return (
    <div>
      <div className="header">
        <h1>报告中心</h1>
        <p>查看你的专注度日报与周报，跟踪习惯趋势</p>
      </div>

      <div className="tabs">
        <button
          className={`tab ${tab === "daily" ? "active" : ""}`}
          onClick={() => setTab("daily")}
        >
          日报
        </button>
        <button
          className={`tab ${tab === "weekly" ? "active" : ""}`}
          onClick={() => setTab("weekly")}
        >
          周报
        </button>
      </div>

      {/* ── Daily ── */}
      {tab === "daily" && (
        <>
          <div className="flex flex-between mb16">
            <input
              type="date"
              value={dailyDate}
              onChange={(e) => setDailyDate(e.target.value)}
              style={{ width: 180 }}
            />
          </div>

          {dailyErr && <div className="error-box">{dailyErr}</div>}

          {dailyLoading && <div className="spinner" />}

          {!dailyLoading && !dailyErr && daily && (
            <>
              {/* KPI Row */}
              <div className="kpi-row mb24">
                <div className="stat-card">
                  <div className="label">总专注时长</div>
                  <div className="value">{formatMinutes(daily.total_focus_minutes)}</div>
                </div>
                <div className="stat-card">
                  <div className="label">专注次数</div>
                  <div className="value">{daily.total_sessions ?? "—"}</div>
                </div>
                <div className="stat-card">
                  <div className="label">分心次数</div>
                  <div className="value">{daily.total_distractions ?? "—"}</div>
                </div>
                {daily.focus_score != null && (
                  <div className="stat-card">
                    <div className="label">专注评分</div>
                    <div className="value">{Math.round(daily.focus_score)}</div>
                  </div>
                )}
              </div>

              {/* Hourly Distribution */}
              <div className="card mb24">
                <h3>时段分布</h3>
                <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 160, paddingTop: 8 }}>
                  {Array.from({ length: 24 }, (_, h) => {
                    const v = hourlyDist[h] || 0;
                    const pct = Math.round((v / maxHourVal) * 100);
                    return (
                      <div key={h} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", height: "100%" }}>
                        <span style={{ fontSize: 10, minHeight: 16 }}>{v > 0 ? `${Math.round(v)}m` : ""}</span>
                        <div
                          style={{
                            width: "100%",
                            height: `${Math.max(pct, v > 0 ? 4 : 0)}%`,
                            background: pct >= 60 ? "var(--color-primary)" : "var(--color-border)",
                            borderRadius: "4px 4px 0 0",
                            transition: "height 0.3s",
                          }}
                        />
                        <span style={{ fontSize: 10, color: "var(--color-text-tertiary)", marginTop: 4 }}>
                          {h % 3 === 0 ? `${h}h` : ""}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* App Usage Table */}
              {daily.app_usage?.length > 0 && (
                <div className="card mb24">
                  <h3>应用使用</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>应用</th>
                        <th>时长</th>
                        <th>类别</th>
                      </tr>
                    </thead>
                    <tbody>
                      {daily.app_usage.map((a: any, i: number) => (
                        <tr key={i}>
                          <td>{a.app || a.name || "—"}</td>
                          <td>{formatMinutes(a.duration_minutes)}</td>
                          <td>
                            <span className={`badge ${a.category === "productive" ? "badge-success" : a.category === "neutral" ? "badge-info" : "badge-warning"}`}>
                              {a.category || "—"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Distraction Analysis */}
              {daily.distraction_analysis?.length > 0 && (
                <div className="card mb24">
                  <h3>分心分析</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>类型</th>
                        <th>次数</th>
                        <th>总时长</th>
                      </tr>
                    </thead>
                    <tbody>
                      {daily.distraction_analysis.map((d: any, i: number) => (
                        <tr key={i}>
                          <td>{d.type || d.name || "—"}</td>
                          <td>{d.count ?? "—"}</td>
                          <td>{formatMinutes(d.total_duration)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}

          {!dailyLoading && !dailyErr && !daily && (
            <div className="card" style={{ textAlign: "center", color: "var(--color-text-tertiary)", padding: 40 }}>
              暂无日报数据
            </div>
          )}
        </>
      )}

      {/* ── Weekly ── */}
      {tab === "weekly" && (
        <>
          <div className="flex flex-between mb16">
            <input
              type="date"
              value={weekStart}
              onChange={(e) => setWeekStart(e.target.value)}
              style={{ width: 180 }}
            />
          </div>

          {weeklyErr && <div className="error-box">{weeklyErr}</div>}

          {weeklyLoading && <div className="spinner" />}

          {!weeklyLoading && !weeklyErr && weekly && (
            <>
              {/* KPI Row */}
              <div className="kpi-row mb24">
                <div className="stat-card">
                  <div className="label">周总专注时长</div>
                  <div className="value">{formatMinutes(weekly.total_focus_minutes)}</div>
                </div>
                <div className="stat-card">
                  <div className="label">总专注次数</div>
                  <div className="value">{weekly.total_sessions ?? "—"}</div>
                </div>
                <div className="stat-card">
                  <div className="label">总分心次数</div>
                  <div className="value">{weekly.total_distractions ?? "—"}</div>
                </div>
                {weekly.avg_focus_score != null && (
                  <div className="stat-card">
                    <div className="label">日均评分</div>
                    <div className="value">{Math.round(weekly.avg_focus_score)}</div>
                  </div>
                )}
              </div>

              {/* 7-Day Chart */}
              <div className="card mb24">
                <h3>每日专注时长</h3>
                <div style={{ display: "flex", alignItems: "flex-end", gap: 12, height: 180, paddingTop: 8 }}>
                  {dailySummary.map((d: any, i: number) => {
                    const v = d.focus_minutes || 0;
                    const pct = Math.round((v / maxDayFocus) * 100);
                    return (
                      <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", height: "100%" }}>
                        <span style={{ fontSize: 11, minHeight: 18 }}>{v > 0 ? formatMinutes(v) : ""}</span>
                        <div
                          style={{
                            width: "100%",
                            maxWidth: 60,
                            height: `${Math.max(pct, v > 0 ? 4 : 0)}%`,
                            background: i === new Date().getDay() ? "var(--color-primary)" : "var(--color-border)",
                            borderRadius: "6px 6px 0 0",
                            transition: "height 0.3s",
                          }}
                        />
                        <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 6 }}>
                          {dayLabel(d.date)}
                        </span>
                        <span style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>
                          {fmtDate(d.date)}
                        </span>
                      </div>
                    );
                  })}
                  {dailySummary.length === 0 && (
                    <div style={{ width: "100%", textAlign: "center", color: "var(--color-text-tertiary)", paddingTop: 60 }}>
                      暂无数据
                    </div>
                  )}
                </div>
              </div>

              {/* Day-by-Day Detail Table */}
              {dailySummary.length > 0 && (
                <div className="card mb24">
                  <h3>每日详情</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>日期</th>
                        <th>周几</th>
                        <th>专注时长</th>
                        <th>专注次数</th>
                        <th>分心次数</th>
                        <th>评分</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dailySummary.map((d: any, i: number) => (
                        <tr key={i}>
                          <td>{d.date ?? "—"}</td>
                          <td>{dayLabel(d.date)}</td>
                          <td>{formatMinutes(d.focus_minutes)}</td>
                          <td>{d.sessions ?? "—"}</td>
                          <td>{d.distractions ?? "—"}</td>
                          <td>{d.focus_score != null ? Math.round(d.focus_score) : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Week-over-Week Comparison */}
              {weekly.week_over_week && (
                <div className="card mb24">
                  <h3>周环比</h3>
                  <div className="kpi-row" style={{ marginBottom: 0 }}>
                    {[
                      { label: "专注时长变化", key: "focus_change_pct", suffix: "%", good: true },
                      { label: "专注次数变化", key: "sessions_change_pct", suffix: "%", good: true },
                      { label: "分心次数变化", key: "distractions_change_pct", suffix: "%", good: false },
                      { label: "评分变化", key: "score_change_pct", suffix: "%", good: true },
                    ].map((m) => {
                      const v = weekly.week_over_week[m.key];
                      if (v == null) return null;
                      const positive = m.good ? v >= 0 : v <= 0;
                      return (
                        <div className="stat-card" key={m.key}>
                          <div className="label">{m.label}</div>
                          <div className={`value ${positive ? "good" : "bad"}`} style={{ fontSize: 22 }}>
                            {v > 0 ? "+" : ""}{Math.round(v)}{m.suffix}
                          </div>
                          <div className={`sub ${positive ? "good" : "bad"}`}>
                            {positive ? "↑ 改善" : "↓ 下降"}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}

          {!weeklyLoading && !weeklyErr && !weekly && (
            <div className="card" style={{ textAlign: "center", color: "var(--color-text-tertiary)", padding: 40 }}>
              暂无周报数据
            </div>
          )}
        </>
      )}
    </div>
  );
}
