import { useState, useEffect, useCallback } from "react";
import { getDailyReport, getErrorMessage, getWeeklyReport } from "../api";
import type { DailyReport, WeeklyReport } from "../report-state";
import { DailyReportBody, WeeklyReportBody } from "./ReportsSections";

type Tab = "daily" | "weekly";

function mondayOf(date: Date): string {
  const d = new Date(date);
  d.setDate(d.getDate() - ((d.getDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
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

  const loadDaily = useCallback(async (date: string) => {
    setDailyLoading(true);
    setDailyErr("");
    try {
      const data = await getDailyReport(date);
      setDaily(data);
    } catch (e: unknown) {
      setDailyErr(getErrorMessage(e, "加载失败"));
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
    } catch (e: unknown) {
      setWeeklyErr(getErrorMessage(e, "加载失败"));
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

  return (
    <div>
      <div className="header">
        <h1>报告中心</h1>
        <p>查看你的专注度日报与周报，跟踪习惯趋势</p>
      </div>

      <div className="tabs">
        <button
          type="button"
          className={`tab ${tab === "daily" ? "active" : ""}`}
          onClick={() => setTab("daily")}
        >
          日报
        </button>
        <button
          type="button"
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

          {!dailyLoading && !dailyErr && daily && <DailyReportBody report={daily} />}

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

          {!weeklyLoading && !weeklyErr && weekly && <WeeklyReportBody report={weekly} />}

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
