import type { DailyReport, WeeklyReport } from "../report-state";
import type { StateCardModel } from "../report-contract";
import { dayLabel, formatMinutes, toDailyReportView, toWeeklyReportView } from "../report-view";

/** Existing-style explanatory card shown for every non-ready report state. */
export function StateCard({ card }: { card: StateCardModel }) {
  return (
    <div className="card" style={{ textAlign: "center", padding: 40 }}>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>{card.title}</div>
      <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>{card.message}</div>
    </div>
  );
}

export function DailyReportBody({ report }: { report: DailyReport }) {
  const view = toDailyReportView(report);
  if (!view.ready || !view.kpis || !view.hourlyChart) {
    return view.stateCard ? <StateCard card={view.stateCard} /> : null;
  }
  const { kpis, hourlyChart, topApps } = view;

  return (
    <>
      {/* KPI Row */}
      <div className="kpi-row mb24">
        <div className="stat-card">
          <div className="label">总专注时长</div>
          <div className="value">{kpis.totalFocusMinutes}</div>
        </div>
        <div className="stat-card">
          <div className="label">专注次数</div>
          <div className="value">{kpis.totalSessions}</div>
        </div>
        <div className="stat-card">
          <div className="label">分心次数</div>
          <div className="value">{kpis.totalDistractions}</div>
        </div>
        <div className="stat-card">
          <div className="label">专注评分</div>
          <div className="value">{kpis.focusScore}</div>
        </div>
      </div>

      {/* Hourly Distribution */}
      <div className="card mb24">
        <h3>时段分布</h3>
        {/* minWidth:0 lets columns shrink below label min-content so 24 dense
            bars fit a 375px viewport; value labels ellipsize (title/aria keep
            them accessible); hour anchors stay unclipped and centered, and the
            row clip + symmetric padding keep the last anchor inside the box. */}
        <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 160, paddingTop: 8, padding: "0 4px", overflow: "hidden" }}>
          {hourlyChart.bars.map((bar) => (
            <div key={bar.hour} style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", alignItems: "center", height: "100%" }}>
              <span
                role="img"
                title={bar.label || undefined}
                aria-label={bar.label ? `${bar.hour}时专注 ${bar.label}` : undefined}
                style={{
                  display: "block",
                  width: "100%",
                  minHeight: 16,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  fontSize: 10,
                }}
              >
                {bar.label}
              </span>
              <div
                style={{
                  width: "100%",
                  height: `${Math.max(bar.pct, bar.minutes > 0 ? 4 : 0)}%`,
                  background: bar.pct >= 60 ? "var(--color-primary)" : "var(--color-border)",
                  borderRadius: "4px 4px 0 0",
                  transition: "height 0.3s",
                }}
              />
              <span
                style={{
                  display: "block",
                  width: "100%",
                  overflow: "visible",
                  whiteSpace: "nowrap",
                  textAlign: "center",
                  fontSize: 10,
                  color: "var(--color-text-tertiary)",
                  marginTop: 4,
                }}
              >
                {bar.axisLabel}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* App Usage Table */}
      {topApps.length > 0 && (
        <div className="card mb24">
          <h3>应用使用</h3>
          <table>
            <thead>
              <tr>
                <th>应用</th>
                <th>时长</th>
              </tr>
            </thead>
            <tbody>
              {topApps.map((app) => (
                <tr key={app.app}>
                  <td>{app.app}</td>
                  <td>{formatMinutes(app.minutes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

export function WeeklyReportBody({ report }: { report: WeeklyReport }) {
  const view = toWeeklyReportView(report);
  if (!view.ready || !view.kpis || !view.chart) {
    return view.stateCard ? <StateCard card={view.stateCard} /> : null;
  }
  const { kpis, chart, summary, trend } = view;

  return (
    <>
      {/* KPI Row */}
      <div className="kpi-row mb24">
        <div className="stat-card">
          <div className="label">周总专注时长</div>
          <div className="value">{kpis.totalFocusMinutes}</div>
        </div>
        <div className="stat-card">
          <div className="label">总专注次数</div>
          <div className="value">{kpis.totalSessions}</div>
        </div>
        <div className="stat-card">
          <div className="label">总分心次数</div>
          <div className="value">{kpis.totalDistractions}</div>
        </div>
        <div className="stat-card">
          <div className="label">日均评分</div>
          <div className="value">{kpis.avgFocusScore}</div>
        </div>
      </div>

      {/* 7-Day Chart */}
      <div className="card mb24">
        <h3>每日专注时长</h3>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 12, height: 180, paddingTop: 8 }}>
          {chart.bars.map((bar) => (
            <div key={bar.date} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", height: "100%" }}>
              <span style={{ fontSize: 11, minHeight: 18 }}>{bar.label}</span>
              <div
                style={{
                  width: "100%",
                  maxWidth: 60,
                  height: `${Math.max(bar.pct, bar.focusMinutes > 0 ? 4 : 0)}%`,
                  background: "var(--color-border)",
                  borderRadius: "6px 6px 0 0",
                  transition: "height 0.3s",
                }}
              />
              <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 6 }}>
                {bar.dayLabel}
              </span>
              <span style={{ fontSize: 10, color: "var(--color-text-tertiary)" }}>
                {bar.dateLabel}
              </span>
            </div>
          ))}
          {chart.bars.length === 0 && (
            <div style={{ width: "100%", textAlign: "center", color: "var(--color-text-tertiary)", paddingTop: 60 }}>
              暂无数据
            </div>
          )}
        </div>
      </div>

      {/* Day-by-Day Detail Table */}
      {summary.length > 0 && (
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
              {summary.map((d) => (
                <tr key={d.date}>
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

      {/* Week-over-Week Trend */}
      {trend && trend.metrics.length > 0 && (
        <div className="card mb24">
          <h3>周环比</h3>
          <div className="kpi-row" style={{ marginBottom: 0 }}>
            {trend.metrics.map((m) => (
              <div className="stat-card" key={m.label}>
                <div className="label">{m.label}</div>
                <div className={`value ${m.good ? "good" : "bad"}`} style={{ fontSize: 22 }}>
                  {m.display}
                </div>
                <div className={`sub ${m.good ? "good" : "bad"}`}>{m.sub}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
