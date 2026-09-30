import { useState, useEffect, useCallback, useRef } from "react";
import { DatePicker } from "@mui/x-date-pickers/DatePicker";
import { LocalizationProvider } from "@mui/x-date-pickers/LocalizationProvider";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import dayjs from "dayjs";
import "dayjs/locale/zh-cn";
import { getDailyReport, getErrorMessage, getWeeklyReport } from "../api";
import type { DailyReport, WeeklyReport } from "../report-state";
import { toDailyReportView, toWeeklyReportView, formatMinutes, dayLabel } from "../report-view";
import { toThreeHourBuckets } from "../report-buckets";
import { localDateStr, mondayOf } from "../date-utils";
import "./picker.css";
import "./reports.css";

type Tab = "daily" | "weekly";

const DAILY_STATE_ICONS = ["icon-shizhong", "icon-yanjing", "icon-zhuanzhu", "icon-fenshuxian"];

function todayStr(): string {
  return localDateStr();
}

interface ReportStat {
  title: string;
  content: string;
  icon: string;
}

/** Reference always draws four metric cards; when the report is not ready we
 *  keep the row (so the layout matches) but show `—` instead of inventing
 *  numbers, and the state card text below explains why. */
function placeholderStats(tab: Tab): ReportStat[] {
  const titles = tab === "daily"
    ? ["总专注时长", "专注次数", "分心次数", "专注评分"]
    : ["总专注时长", "总专注次数", "总分心次数", "日均评分"];
  return DAILY_STATE_ICONS.map((icon, index) => ({
    title: titles[index],
    content: "—",
    icon,
  }));
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
  const maxBucket = Math.max(1, ...buckets.map((b) => b.minutes));

  const dailyStats: ReportStat[] | null = dailyView?.kpis
    ? [
        { title: "总专注时长", content: dailyView.kpis.totalFocusMinutes, icon: DAILY_STATE_ICONS[0] },
        { title: "专注次数", content: `${dailyView.kpis.totalSessions} 次`, icon: DAILY_STATE_ICONS[1] },
        { title: "分心次数", content: dailyView.kpis.totalDistractions, icon: DAILY_STATE_ICONS[2] },
        { title: "专注评分", content: dailyView.kpis.focusScore, icon: DAILY_STATE_ICONS[3] },
      ]
    : null;

  const weeklyStats: ReportStat[] | null = weeklyView?.kpis
    ? [
        { title: "总专注时长", content: weeklyView.kpis.totalFocusMinutes, icon: DAILY_STATE_ICONS[0] },
        { title: "总专注次数", content: `${weeklyView.kpis.totalSessions} 次`, icon: DAILY_STATE_ICONS[1] },
        { title: "总分心次数", content: weeklyView.kpis.totalDistractions, icon: DAILY_STATE_ICONS[2] },
        { title: "日均评分", content: weeklyView.kpis.avgFocusScore, icon: DAILY_STATE_ICONS[3] },
      ]
    : null;

  const stats = tab === "daily"
    ? (dailyStats ?? placeholderStats(tab))
    : (weeklyStats ?? placeholderStats(tab));

  // Weekly reuses the daily layout: the reference has no weekly variant, so
  // the seven-day totals feed the same distribution block (plan: same layout
  // language, real weekly data).
  const weeklyBuckets = weeklyView?.chart
    ? weeklyView.chart.bars.map((bar) => ({ hour: bar.dayLabel, minutes: Math.round(bar.focusMinutes) }))
    : [];
  const shownBuckets = tab === "daily" ? buckets : weeklyBuckets;
  const shownMax = tab === "daily" ? maxBucket : Math.max(1, ...weeklyBuckets.map((b) => b.minutes));

  const loading = tab === "daily" ? dailyLoading : weeklyLoading;
  const error = tab === "daily" ? dailyErr : weeklyErr;

  const chartTitle = tab === "daily" ? "专注时段分布" : "每日专注时长";
  const detailTitle = tab === "daily" ? "应用使用" : "每日详情";

  // Non-ready states (no_activity / future / partial …) still owe the user an
  // explanation; the contract's state card carries it into the reference's
  // "今日洞察" block instead of being dropped with the KPIs.
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
    <LocalizationProvider dateAdapter={AdapterDayjs} adapterLocale="zh-cn">
      <div className="reports">
        <div className="top-buttom">
          <button
            type="button"
            className={`daily-report${tab === "daily" ? " be-chose" : ""}`}
            aria-pressed={tab === "daily"}
            onClick={() => setTab("daily")}
          >
            日报
          </button>
          <button
            type="button"
            className={`weekly-report${tab === "weekly" ? " be-chose" : ""}`}
            aria-pressed={tab === "weekly"}
            onClick={() => setTab("weekly")}
          >
            周报
          </button>
        </div>

        <div className="report-datepicker mf-picker">
          <DatePicker
            disableFuture
            value={dayjs(tab === "daily" ? dailyDate : weekStart)}
            onChange={(newValue, context) => {
              if (!newValue?.isValid() || context.validationError != null) return;
              if (tab === "daily") setDailyDate(newValue.format("YYYY-MM-DD"));
              else setWeekStart(mondayOf(newValue.toDate()));
            }}
            format="YYYY/MM/DD"
            label="选择日期"
            slotProps={{
              textField: { fullWidth: true, size: "small" },
              layout: { className: "f-calendar" },
              popper: { className: "f-calendar" },
            }}
          />
        </div>

        {error && (
          <div className="error-box report-error" role="alert">
            {error}
            <button
              className="btn btn-sm"
              style={{ marginLeft: 12 }}
              onClick={() => (tab === "daily" ? loadDaily(dailyDate) : loadWeekly(weekStart))}
            >
              重试
            </button>
          </div>
        )}

        {stats ? (
          <div className="t-state">
            {stats.map((item) => (
              <div className="state-item" key={item.title}>
                <div className="state-item-text">{item.title}</div>
                <div className="state-item-number">{item.content}</div>
                <div className="icon-box">
                  <div className={`icon iconfont ${item.icon}`} aria-hidden="true" />
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="t-state" />
        )}

        <div className="zhuanzhudongcha">
          <div className="zhuanzhudongcha-title">今日洞察</div>
          {loading ? (
            <div className="spinner" />
          ) : (
            <div className="zhuanzhudongcha-text">
              {insight ?? (loading ? "" : "暂无洞察：等待足够的专注数据后由规则引擎生成。")}
            </div>
          )}
        </div>

        <div className="chart-box">
          <div className="title">{chartTitle}</div>
          {shownBuckets.length === 0 && !loading ? (
            <div className="report-empty" style={{ margin: "auto" }}>
              暂无数据
            </div>
          ) : (
            shownBuckets.map((item, index) => (
              <div className="chart-box-item" key={`${item.hour}-${index}`}>
                <div className="chart-box-item-during">{item.hour}</div>
                <div
                  className="chart-box-item-column"
                  style={{ height: `${item.minutes > 0 ? Math.max(4, (item.minutes / shownMax) * 150) : 0}px` }}
                >
                  <div className="chart-box-item-number">{item.minutes}m</div>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="report-detail-list">
          <div className="report-detail-title">{detailTitle}</div>
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
              <div className="report-empty">{loading ? "加载中..." : "暂无应用使用数据"}</div>
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
            <div className="report-empty">{loading ? "加载中..." : "暂无每日详情"}</div>
          )}

          {tab === "weekly" && weeklyView?.trend && weeklyView.trend.metrics.length > 0 && (
            <div className="trend-row">
              {weeklyView.trend.metrics.map((m) => (
                <div className="trend-pill" key={m.label}>
                  <div className="label">{m.label}</div>
                  <div className={`value ${m.good ? "good" : "bad"}`}>{m.display}</div>
                  <div className="sub">{m.sub}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </LocalizationProvider>
  );
}
