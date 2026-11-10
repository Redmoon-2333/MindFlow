import NativeDateField from "../components/NativeDateField";
import { useState, useEffect, useCallback, useRef } from "react";
import { getDailyReport, getErrorMessage, getWeeklyReport } from "../api";
import type { DailyReport, WeeklyReport } from "../report-state";
import { toDailyReportView, toWeeklyReportView, formatMinutes, dayLabel } from "../report-view";
import { toThreeHourBuckets } from "../report-buckets";
import { localDateStr, mondayOf } from "../date-utils";
import Surface from "../components/signal/Surface";
import SectionHeader from "../components/signal/SectionHeader";
import MetricStrip from "../components/signal/MetricStrip";
import type { MetricItem } from "../components/signal/MetricStrip";
import Segmented from "../components/signal/Segmented";
import BarChart from "../components/charts/BarChart";
import "./reports.css";

type Tab = "daily" | "weekly";

function todayStr(): string {
  return localDateStr();
}

/** Reference always draws four metric cells; when the report is not ready we
 *  keep the row (so the layout matches) but show `—` instead of inventing
 *  numbers, and the state card text below explains why. */
function placeholderStats(tab: Tab): MetricItem[] {
  const labels = tab === "daily"
    ? ["总专注时长", "会话次数", "分心次数", "专注评分"]
    : ["日均专注时长", "有记录天数", "总分心次数", "日均评分"];
  return labels.map((label, index) => ({ key: `metric-${index}`, label, value: "—" }));
}

export default function Reports() {
  const [tab, setTab] = useState<Tab>("daily");

  // Daily state
  const [dailyDate, setDailyDate] = useState(todayStr());
  const [daily, setDaily] = useState<DailyReport | null>(null);
  const [dailyLoading, setDailyLoading] = useState(false);
  const [dailyErr, setDailyErr] = useState("");

  // Weekly state
  const [weekStart, setWeekStart] = useState(mondayOf(new Date()));
  const [weekly, setWeekly] = useState<WeeklyReport | null>(null);
  const [weeklyLoading, setWeeklyLoading] = useState(false);
  const [weeklyErr, setWeeklyErr] = useState("");

  // Request-sequence guards so a slow response for an older date/week never
  // overwrites the newer selection (audit report — stale-overwrite race).
  const dailySeqRef = useRef(0);
  const weeklySeqRef = useRef(0);

  const loadDaily = useCallback(async (date: string) => {
    const seq = ++dailySeqRef.current;
    setDailyLoading(true);
    setDailyErr("");
    setDaily(null);
    try {
      const data = await getDailyReport(date);
      if (seq !== dailySeqRef.current) return;
      setDaily(data);
    } catch (e: unknown) {
      if (seq !== dailySeqRef.current) return;
      setDailyErr(getErrorMessage(e, "加载失败"));
      setDaily(null);
    } finally {
      if (seq === dailySeqRef.current) setDailyLoading(false);
    }
  }, []);

  const loadWeekly = useCallback(async (ws: string) => {
    const seq = ++weeklySeqRef.current;
    setWeeklyLoading(true);
    setWeeklyErr("");
    setWeekly(null);
    try {
      const data = await getWeeklyReport(ws);
      if (seq !== weeklySeqRef.current) return;
      setWeekly(data);
    } catch (e: unknown) {
      if (seq !== weeklySeqRef.current) return;
      setWeeklyErr(getErrorMessage(e, "加载失败"));
      setWeekly(null);
    } finally {
      if (seq === weeklySeqRef.current) setWeeklyLoading(false);
    }
  }, []);

  useEffect(() => {
    const requests = dailySeqRef;
    if (tab === "daily") loadDaily(dailyDate);
    return () => { requests.current++; };
  }, [tab, dailyDate, loadDaily]);

  useEffect(() => {
    const requests = weeklySeqRef;
    if (tab === "weekly") loadWeekly(weekStart);
    return () => { requests.current++; };
  }, [tab, weekStart, loadWeekly]);

  const dailyView = daily ? toDailyReportView(daily) : null;
  const weeklyView = weekly ? toWeeklyReportView(weekly) : null;

  const buckets = dailyView?.hourlyChart
    ? toThreeHourBuckets(Object.fromEntries(dailyView.hourlyChart.bars.map((b) => [String(b.hour), b.minutes])))
    : [];

  let stats: MetricItem[];
  if (tab === "daily") {
    stats = daily?.data_state === "ready" && dailyView?.kpis
      ? [
          { key: "focus", label: "总专注时长", value: String(Math.round(daily.total_focus_minutes)), unit: "分钟" },
          {
            key: "sessions",
            label: "会话次数",
            value: dailyView.kpis.totalSessions,
            unit: "次",
            help: { what: "一段连续使用电脑 5 分钟以上的时间叫一次会话，不管专注还是分心都算。" },
          },
          { key: "distractions", label: "分心次数", value: dailyView.kpis.totalDistractions, unit: "次" },
          {
            key: "score",
            label: "专注评分",
            value: dailyView.kpis.focusScore,
            help: { what: "根据这一天所有使用记录打的分，满分 100：主要软件占比越高、来回切换越少，分越高。" },
          },
        ]
      : placeholderStats("daily");
  } else {
    stats = weekly?.data_state === "ready" && weeklyView?.kpis
      ? [
          {
            key: "focus",
            label: "日均专注时长",
            // Backend already sends the per-active-day average here
            // (report_service.py: total_focus_minutes = avg_focus).
            value: String(Math.round(weekly.total_focus_minutes)),
            unit: "分钟",
          },
          {
            key: "sessions",
            label: "有记录天数",
            value: String(weekly.daily_summary.length),
            unit: "天",
          },
          { key: "distractions", label: "总分心次数", value: weeklyView.kpis.totalDistractions, unit: "次" },
          {
            key: "score",
            label: "日均评分",
            value: weeklyView.kpis.avgFocusScore,
            help: { what: "根据这一周所有使用记录打的分，满分 100：主要软件占比越高、来回切换越少，分越高。" },
          },
        ]
      : placeholderStats("weekly");
  }

  // Weekly reuses the daily layout: the reference has no weekly variant, so
  // the seven-day totals feed the same distribution block (plan: same layout
  // language, real weekly data).
  const weeklyBuckets = weeklyView?.chart
    ? weeklyView.chart.bars.map((bar) => ({ hour: bar.dayLabel, minutes: Math.round(bar.focusMinutes) }))
    : [];
  const shownBuckets = tab === "daily" ? buckets : weeklyBuckets;

  const loading = tab === "daily" ? dailyLoading : weeklyLoading;
  const error = tab === "daily" ? dailyErr : weeklyErr;

  const chartTitle = tab === "daily" ? "专注时段分布" : "每日专注时长";
  const detailTitle = tab === "daily" ? "应用使用" : "每日详情";
  const insightTitle = tab === "daily" ? "今日洞察" : "本周洞察";

  // Non-ready states (no_activity / future / partial …) still owe the user an
  // explanation; the contract's state card carries it into the "今日洞察"
  // block instead of being dropped with the KPIs.
  const stateCard = tab === "daily" ? dailyView?.stateCard : weeklyView?.stateCard;
  const insight =
    tab === "daily"
      ? dailyView?.patternSummary ?? (stateCard ? `${stateCard.title}：${stateCard.message}` : null)
      : weeklyView?.trend
        ? weeklyView.trend.metrics.map((m) => `${m.label} ${m.display}（${m.sub}）`).join("，")
        : stateCard
          ? `${stateCard.title}：${stateCard.message}`
          : null;

  return (
    <>
      <div className="reports">
        <p className="page-lede">按日或按周汇总你的专注表现。</p>

        <div className="page-toolbar top-buttom">
          <Segmented
            ariaLabel="报告类型"
            options={[{ value: "daily", label: "日报" }, { value: "weekly", label: "周报" }]}
            value={tab}
            onChange={(value) => setTab(value as Tab)}
          />
          <div className="report-datepicker mf-picker">
            <NativeDateField label="选择日期" value={tab === "daily" ? dailyDate : weekStart} onChange={value => { if (tab === "daily") setDailyDate(value); else setWeekStart(mondayOf(new Date(`${value}T12:00:00`))); }} />
          </div>
        </div>

        {error && (
          <div className="error-box report-error" role="alert">
            {error}
            <button
              className="btn btn-secondary btn-sm report-retry"
              onClick={() => (tab === "daily" ? loadDaily(dailyDate) : loadWeekly(weekStart))}
            >
              重试
            </button>
          </div>
        )}

        <div className="t-state">
          <MetricStrip label={tab === "daily" ? "当日" : "本周"} items={stats} />
        </div>

        <Surface as="section" className="zhuanzhudongcha">
          <SectionHeader
            title={insightTitle}
            help={{ what: "根据固定规则自动写出的小结，不调用 AI。" }}
          />
          {loading ? (
            <div className="spinner" />
          ) : (
            <div className="zhuanzhudongcha-text">
              {insight ?? "暂无洞察：等待足够的专注数据后由规则引擎生成。"}
            </div>
          )}
        </Surface>

        <Surface as="section" className="chart-box">
          <SectionHeader title={chartTitle} />
          <BarChart
            data={shownBuckets.map((bucket, index) => ({
              key: `${bucket.hour}-${index}`,
              // §8.19: `18:00~21:00` is too wide for a tick; the axis uses the
              // short form and the tooltip keeps the exact range.
              label: `${bucket.hour.replace(/:00/g, "").replace("~", "–")}点`,
              value: bucket.minutes,
              title: bucket.hour,
            }))}
            ariaLabel={chartTitle}
            unit=" 分钟"
            valueName="专注时长"
            height={220}
            colorFor={() => "var(--signal)"}
            emptyMessage="暂无数据"
          />
        </Surface>

        <Surface as="section" className="report-detail-list">
          <SectionHeader title={detailTitle} />
          {tab === "daily" ? (
            dailyView && dailyView.topApps.length > 0 ? (
              <table className="simple-table">
                <thead>
                  <tr>
                    <th>应用</th>
                    <th>时长</th>
                  </tr>
                </thead>
                <tbody>
                  {dailyView.topApps.map((app) => (
                    <tr key={app.app}>
                      <td>{app.app}</td>
                      <td>{formatMinutes(app.minutes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="report-empty">{loading ? "加载中…" : "暂无应用使用数据"}</div>
            )
          ) : weeklyView && weeklyView.summary.length > 0 ? (
            <table className="weekly-detail-table">
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
                {weeklyView.summary.map((d) => (
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
          ) : (
            <div className="report-empty">{loading ? "加载中…" : "暂无每日详情"}</div>
          )}

          {tab === "weekly" && weeklyView?.trend && weeklyView.trend.metrics.length > 0 && (
            <div className="trend-row">
              {weeklyView.trend.metrics.map((m) => (
                <div className="kv-row kv-row--text" key={m.label}>
                  <span>{m.label}</span>
                  <span className={m.good ? "trend-value--good" : "trend-value--bad"}>
                    {m.display} · {m.sub}
                  </span>
                </div>
              ))}
            </div>
          )}
        </Surface>
      </div>
    </>
  );
}
