import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  getCurrentActivity,
  getFocusTrend,
  getFocusPrediction,
  getModelStatus,
  getInterventionHistory,
  getAutonomy,
  isAutonomyPaused,
  resumeAutonomy,
  pauseAutonomy,
  getErrorMessage,
  deriveFocusTrendKpi,
} from "../api";
import type {
  ActivityItem,
  AutonomyStatus,
  FocusPredictionResponse,
  FocusTrendDay,
  FocusTrendResponse,
  InterventionHistoryItem,
  ModelStatus,
} from "../api";
import { dayLabel, findDayByDate, localDateStr, previousLocalDateStr } from "../date-utils";
import { FOCUS_BAND_COLORS, FOCUS_BAND_LABELS, focusBand, toFocusPredictionView } from "../prediction-state";
import type { FocusBand } from "../prediction-state";
import { realtimeClient } from "../realtime";
import type { RealtimeStatus } from "../realtime";
import { getInterventionTypeLabel } from "../lib/intervention-labels";
import { userTerm } from "../lib/user-terms";
import { METHOD_LABELS } from "../types/explainability";
import BarChart from "../components/charts/BarChart";
import ContributionBar from "../components/charts/ContributionBar";
import Surface from "../components/signal/Surface";
import SectionHeader from "../components/signal/SectionHeader";
import MetricStrip from "../components/signal/MetricStrip";
import type { MetricItem } from "../components/signal/MetricStrip";
import Disclosure from "../components/signal/Disclosure";
import LiveDot from "../components/signal/LiveDot";
import HelpTip from "../components/signal/HelpTip";
import "./dashboard.css";
import "../components/primitives.css";

/** Response wording shared with the intervention page. */
const RESPONSE_LABELS: Record<string, string> = {
  accept: "已接受",
  accepted: "已接受",
  ignore: "已忽略",
  ignored: "已忽略",
  dismiss: "已关闭",
  dismissed: "已关闭",
};

/** The band's name inside the track's sentence ("…处于专注区间"). */
const BAND_ZONE: Record<FocusBand, string> = { focus: "专注", uncertain: "不确定", drift: "分心" };

const RELATIVE_TIME = new Intl.RelativeTimeFormat("zh-CN", { numeric: "auto" });

/** The machine's real UTC offset, e.g. "UTC+8" or "UTC+5:30". Hardcoding
 *  "UTC+8" mislabelled every timestamp for anyone outside China. */
function utcOffsetLabel(date: Date): string {
  const minutes = -date.getTimezoneOffset();
  const sign = minutes >= 0 ? "+" : "-";
  const abs = Math.abs(minutes);
  const hours = Math.floor(abs / 60);
  const rest = abs % 60;
  return `UTC${sign}${hours}${rest ? `:${String(rest).padStart(2, "0")}` : ""}`;
}

function formatClock(iso: string | undefined): string {
  if (!iso) return "--:--";
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return "--:--";
  return parsed.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

/** "刚刚更新" under a minute, then "更新于 2分钟前" / "更新于 3小时前". */
function formatUpdatedAt(iso: string, now: Date): string | null {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return null;
  const seconds = Math.round((parsed.getTime() - now.getTime()) / 1000);
  if (Math.abs(seconds) < 60) return "刚刚更新";
  const minutes = Math.round(seconds / 60);
  if (Math.abs(minutes) < 60) return `更新于 ${RELATIVE_TIME.format(minutes, "minute")}`;
  const hours = Math.round(minutes / 60);
  if (Math.abs(hours) < 24) return `更新于 ${RELATIVE_TIME.format(hours, "hour")}`;
  return `更新于 ${RELATIVE_TIME.format(Math.round(hours / 24), "day")}`;
}

/** "18:30" inside today, "11月12日 09:00" across midnight. */
function formatPausedUntil(date: Date, now: Date): string {
  const clock = `${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();
  return sameDay ? clock : `${date.getMonth() + 1}月${date.getDate()}日 ${clock}`;
}

/** Percentage change between two days, or null when yesterday is missing —
 *  a fabricated "0%" would read as a real comparison. */
function dayOverDay(today: number | undefined, previous: number | undefined): number | null {
  if (today == null || previous == null) return null;
  if (previous === 0) return today === 0 ? 0 : null;
  return ((today - previous) / previous) * 100;
}

/** Spec §4.1.3: null → none, |d| < 0.5 → flat, else up/down. */
function deltaOf(delta: number | null): MetricItem["delta"] {
  if (delta == null) return { text: "较昨日 —", tone: "none" };
  if (Math.abs(delta) < 0.5) return { text: "较昨日 持平", tone: "flat" };
  return delta > 0
    ? { text: `较昨日 ↑${Math.round(delta)}%`, tone: "up" }
    : { text: `较昨日 ↓${Math.round(Math.abs(delta))}%`, tone: "down" };
}

/** The latest reminder by `triggered_at || created_at`. */
function latestIntervention(items: InterventionHistoryItem[]): InterventionHistoryItem | null {
  let latest: InterventionHistoryItem | null = null;
  let latestAt = Number.NEGATIVE_INFINITY;
  for (const item of items) {
    const raw = item.triggered_at || item.created_at;
    const at = raw ? Date.parse(raw) : Number.NaN;
    if (Number.isNaN(at)) continue;
    if (at >= latestAt) {
      latest = item;
      latestAt = at;
    }
  }
  return latest;
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  /** Request-level failure (the partial-failure notice). */
  const [error, setError] = useState<string | null>(null);
  /** Action-level failure (pause / resume), shown in the same notice slot. */
  const [actionError, setActionError] = useState<string | null>(null);

  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [focusTrend, setFocusTrend] = useState<FocusTrendResponse | null>(null);
  const [currentActivity, setCurrentActivity] = useState<ActivityItem | null>(null);
  const [interventions, setInterventions] = useState<InterventionHistoryItem[]>([]);
  const [autonomy, setAutonomy] = useState<AutonomyStatus | null>(null);
  const [focusPrediction, setFocusPrediction] = useState<FocusPredictionResponse | null>(null);
  /** Distinguishes "no record today" (0) from "we could not ask" (—).
   *  Both used to collapse into `trend == null`, which made a failed request
   *  indistinguishable from a genuinely empty day (audit F1). */
  const [trendStatus, setTrendStatus] = useState<"loading" | "ready" | "error">("loading");

  const [autonomyLoading, setAutonomyLoading] = useState(false);
  const [customOpen, setCustomOpen] = useState(false);
  /** Evidence disclosure state; drives whether row 1 keeps equal-height peers. */
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [controlTime, setControlTime] = useState({ hour: "0", minute: "0" });
  const [now, setNow] = useState(() => new Date());
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>("idle");

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setTrendStatus("loading");
    // Health, collector and provider status now belong to the top bar's
    // StatusChip (spec §4.1.5), so they are no longer fetched here.
    const results = await Promise.allSettled([
      getModelStatus(),
      getFocusTrend(7),
      getCurrentActivity(),
      getInterventionHistory(7),
      getAutonomy(),
      getFocusPrediction(),
    ]);
    const [ms, ft, ca, ih, au, fp] = results;
    if (ms.status === "fulfilled") setModelStatus(ms.value);
    if (ft.status === "fulfilled") {
      setFocusTrend(ft.value);
      setTrendStatus("ready");
    } else {
      // A rejected trend must not keep serving the previous range's numbers.
      setFocusTrend(null);
      setTrendStatus("error");
    }
    if (ca.status === "fulfilled") setCurrentActivity(ca.value);
    if (ih.status === "fulfilled") setInterventions(ih.value.items);
    if (au.status === "fulfilled") setAutonomy(au.value);
    if (fp.status === "fulfilled") setFocusPrediction(fp.value);
    const failed = results.filter((result) => result.status === "rejected");
    if (failed.length > 0) setError(`部分数据加载失败（${failed.length} 项），其余内容已显示`);
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // The relative "更新于" stamp and the pause countdown both read `now`.
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => realtimeClient.subscribeStatus(setRealtimeStatus), []);
  // Monotonic key for realtime updates — two WS frames in the same
  // millisecond would otherwise collide on `realtime-<timestamp>`.
  const realtimeKeyRef = useRef(0);
  useEffect(() => realtimeClient.subscribe("activity_update", (payload, timestamp) => {
    realtimeKeyRef.current += 1;
    setCurrentActivity({
      id: `realtime-${realtimeKeyRef.current}`, user_id: 1, timestamp, duration_s: 0, event_type: "window_change",
      data: { app_name: payload.app_name, window_title: payload.window_title ?? "", process_name: payload.process_name ?? "", is_idle: payload.is_idle },
    });
  }), []);
  useEffect(() => realtimeClient.subscribe("intervention", (payload, timestamp) => {
    const item: InterventionHistoryItem = {
      id: payload.id, user_id: 1, triggered_at: timestamp, intervention_type: payload.intervention_type,
      cbt_technique: payload.cbt_technique ?? null, context_json: null, user_response: null,
      response_latency_s: null, feedback_rating: null, feedback_comment: null, created_at: timestamp,
      title: payload.title, message: payload.message,
    };
    setInterventions((current) => [item, ...current.filter((entry) => entry.id !== item.id)]);
  }), []);

  /** Every pause path ends in the same endpoint; the backend keeps its
   *  30-minute floor, so the client refuses to send less. */
  const pauseFor = async (minutes: number) => {
    if (minutes < 30) {
      setActionError("暂停时长至少 30 分钟");
      return;
    }
    setAutonomyLoading(true);
    setActionError(null);
    try {
      setAutonomy(await pauseAutonomy(minutes / 60));
      setCustomOpen(false);
      setControlTime({ hour: "0", minute: "0" });
    } catch (e: unknown) {
      setActionError(getErrorMessage(e, "暂停干预失败"));
    } finally {
      setAutonomyLoading(false);
    }
  };

  const handlePause = () => {
    const hours = Number.parseInt(controlTime.hour, 10) || 0;
    const minutes = Number.parseInt(controlTime.minute, 10) || 0;
    return pauseFor(hours * 60 + Math.min(Math.max(minutes, 0), 59));
  };

  const handleResume = async () => {
    setAutonomyLoading(true);
    setActionError(null);
    try {
      setAutonomy(await resumeAutonomy());
    } catch (e: unknown) {
      setActionError(getErrorMessage(e, "恢复干预失败"));
    } finally {
      setAutonomyLoading(false);
    }
  };

  const predictionView = focusPrediction ? toFocusPredictionView(focusPrediction) : null;
  const probability = focusPrediction?.status === "ready" ? focusPrediction.focus_probability : null;
  const ready = predictionView?.ready === true && probability != null;
  const band = probability != null ? focusBand(probability) : null;
  const bandColor = band ? FOCUS_BAND_COLORS[band] : "var(--text-dim)";
  const explanation = focusPrediction?.explanation;
  const contributions = explanation?.contributions ?? [];
  // §4.1.1: the three strongest features, split by sign. A group with no
  // members simply does not render its row.
  const topReasons = [...contributions]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 3);
  const positiveReasons = topReasons.filter((row) => row.contribution > 0);
  const negativeReasons = topReasons.filter((row) => row.contribution < 0);

  // Local date keys: `daily` is a sparse, recorded-days-only series, so "today"
  // and "yesterday" are looked up by key and may legitimately be absent.
  const todayKey = localDateStr();
  const yesterdayKey = previousLocalDateStr(todayKey);
  const kpi = deriveFocusTrendKpi(focusTrend, todayKey);
  const daily: FocusTrendDay[] = focusTrend?.daily ?? [];
  const today = findDayByDate(daily, todayKey);
  const yesterday = findDayByDate(daily, yesterdayKey);
  /** True only when the trend request itself succeeded. */
  const trendKnown = trendStatus === "ready";
  const autonomyPaused = isAutonomyPaused(autonomy, now.getTime());

  // Seven-day bar chart: only a slot that really is today may read 今天.
  const chartbox = daily.map((day) => ({
    key: day.date,
    week: day.date === todayKey ? "今天" : dayLabel(day.date),
    number: Math.round(day.focus_min ?? 0),
    distraction: Math.round(day.distraction_min ?? 0),
    sessions: day.session_count ?? 0,
    score: day.avg_score ?? null,
    today: day.date === todayKey,
  }));
  const hasTodaySlot = chartbox.some((entry) => entry.today);
  const meanFocus = chartbox.length > 0
    ? chartbox.reduce((sum, entry) => sum + entry.number, 0) / chartbox.length
    : 0;
  // §4.1.4 (F10): the mean number lives in the legend, not above the line, so
  // it cannot collide with a bar's own value label. No line for an all-zero
  // series or a failed request.
  const showMean = trendKnown && meanFocus > 0;
  const referenceLine = showMean ? { value: meanFocus } : undefined;

  /** `…` while loading, `—` when the request failed or the value is unknown,
   *  otherwise the measured value. A dash never carries a unit — "—min" reads
   *  as a quantity. One unknown glyph everywhere (U+2014), never "--". */
  const cell = (value: string | null, unit: string): Pick<MetricItem, "value" | "unit"> => {
    if (trendStatus === "loading") return { value: "…" };
    if (!trendKnown) return { value: "—" };
    return value == null ? { value: "—" } : { value, unit };
  };

  const metrics: MetricItem[] = [
    {
      key: "focus",
      label: "专注时长",
      ...cell(String(Math.round(today?.focus_min ?? 0)), "分钟"),
      delta: deltaOf(trendKnown ? dayOverDay(today?.focus_min, yesterday?.focus_min) : null),
      help: {
        what: "连续使用 5 分钟以上、而且很少来回切换软件的时间，算作专注。",
        effect: "系统只看切换次数，不看内容，所以长时间看视频也可能被算成专注。",
      },
    },
    {
      key: "sessions",
      label: "会话次数",
      ...cell(String(today?.session_count ?? 0), "次"),
      delta: deltaOf(trendKnown ? dayOverDay(today?.session_count, yesterday?.session_count) : null),
      help: {
        what: "一段连续使用电脑 5 分钟以上的时间叫一次会话，不管专注还是分心都算。",
      },
    },
    {
      key: "score",
      label: "平均专注分",
      // No observation today → no average to show, never a fabricated 0.
      ...cell(today?.avg_score != null ? today.avg_score.toFixed(1) : null, "/100"),
      delta: deltaOf(trendKnown ? dayOverDay(today?.avg_score, yesterday?.avg_score) : null),
      help: {
        what: "每段会话按“主要软件占的比例”和“切换多不多”打分，满分 100，这里是平均值。",
      },
    },
    {
      key: "distraction",
      label: "分心比率",
      ...cell(kpi.distractionRate != null ? (kpi.distractionRate * 100).toFixed(1) : null, "%"),
      // Deliberately no trend tone: a rising distraction share is not "good".
      delta: { text: "较昨日 —", tone: "none" },
      help: {
        what: "分心时间 ÷（专注时间 + 分心时间）。既不专注也不分心的时间不算在内。",
      },
    },
  ];

  const activity = currentActivity?.data;
  const activityLine = !activity
    ? "当前没有活动记录"
    : activity.is_idle
      ? `当前空闲${activity.app_name ? ` · 最后使用 ${activity.app_name}` : ""}`
      : activity.window_title
        ? `正在使用 ${activity.app_name || "未知应用"} — ${activity.window_title}`
        : `正在使用 ${activity.app_name || "未知应用"}`;
  const updatedLabel = currentActivity ? formatUpdatedAt(currentActivity.timestamp, now) : null;

  const latest = latestIntervention(interventions);
  const latestAnswered = Boolean(latest?.user_response);
  const latestResponse = latestAnswered
    ? RESPONSE_LABELS[(latest?.user_response || "").toLowerCase()] ?? "已回应"
    : "未回应";

  const pausedUntilDate = autonomy?.paused_until ? new Date(autonomy.paused_until) : null;
  const pausedUntilLabel = pausedUntilDate && !Number.isNaN(pausedUntilDate.getTime())
    ? formatPausedUntil(pausedUntilDate, now)
    : null;

  /** Minutes from now to the end of the local day — "今天不再提醒". */
  const minutesLeftToday = (() => {
    const end = new Date(now);
    end.setHours(23, 59, 0, 0);
    return Math.floor((end.getTime() - now.getTime()) / 60_000);
  })();

  const sourceLine = modelStatus?.demo_only
    ? "演示模型（模拟数据，仅供展示，不影响提醒）"
    : modelStatus?.model_origin === "bundled_base"
      ? "基础模型 · 待个人验证"
      : modelStatus?.loaded
        ? `个人模型 · ${userTerm(modelStatus.mode)}`
        : "规则判断";

  // §8.21.4: the backend's `reason` uses internal terms ("v2 特征窗口"), so the
  // reader gets the fixed friendly sentence for each status instead.
  const unavailableCopy = (() => {
    const status = predictionView?.status;
    switch (status) {
      case "no_data":
        return "最近 2 小时的使用记录还不够，暂时无法判断。多用一会儿电脑后会自动更新。";
      case "stale":
        return "最近的记录不足或已过期，判断暂时暂停。继续使用电脑后会自动更新。";
      case "no_model":
        return "还没有可用的预测模型，攒够反馈后可以在模型中心训练。";
      case "schema_mismatch":
      case "inference_error":
        return "预测模型暂时无法使用，系统会先用简单规则判断。";
      default:
        return "还没有可用的预测模型，攒够反馈后可以在模型中心训练。";
    }
  })();

  // Only model states can be fixed in the model centre; a data or staleness
  // problem is not the reader's to resolve there.
  const showModelLink =
    predictionView?.status === "no_model"
    || !predictionView
    || predictionView.status === "schema_mismatch"
    || predictionView.status === "inference_error";

  const heroTrackStyle = probability != null
    ? ({ "--hero-p": `${probability * 100}%` } as CSSProperties)
    : undefined;

  if (loading) {
    return (
      <div className="dash dash--loading">
        <div className="spinner" />
      </div>
    );
  }

  return (
    <div className={`dash${evidenceOpen ? " dash--evidence-open" : ""}`}>
      {error && (
        <div className="notice" role="alert">
          <span className="notice-text">{error}</span>
          <button
            type="button"
            className="notice-close"
            aria-label="关闭提示"
            onClick={() => setError(null)}
          >
            ×
          </button>
        </div>
      )}
      {actionError && (
        <div className="notice" role="alert">
          <span className="notice-text">{actionError}</span>
          <button
            type="button"
            className="notice-close"
            aria-label="关闭提示"
            onClick={() => setActionError(null)}
          >
            ×
          </button>
        </div>
      )}

      {/* ── 此刻 ─────────────────────────────────────────────────────────── */}
      <Surface as="section" pad="lg" className="dash-hero" aria-labelledby="now-title">
        <div className="hero-head">
          <h2 id="now-title" className="sh-title">此刻</h2>
          <LiveDot status={realtimeStatus} />
          {updatedLabel && <span className="hero-updated">{updatedLabel}</span>}
        </div>

        <div className="hero-read">
          <div
            className="hero-num"
            style={{ color: ready ? bandColor : "var(--text-dim)" }}
            aria-label={ready && probability != null ? `专注概率 ${(probability * 100).toFixed(1)}%` : undefined}
          >
            {ready && probability != null ? Math.round(probability * 100) : "—"}
            {ready && <span className="hero-unit">%</span>}
          </div>
          <div className="hero-meta">
            <div className="hero-state" style={{ color: ready && band ? bandColor : "var(--text)" }}>
              {ready && band ? FOCUS_BAND_LABELS[band] : "暂无判断"}
              {ready && band && (
                <HelpTip
                  term="专注概率"
                  what="大数字是系统猜你此刻在专注的可能性，越高越可能在专注。"
                  effect="它只是估计，可能猜错。猜错时，在弹出的“模型校准”里告诉它就好。"
                />
              )}
            </div>
            <p className="hero-activity" title={activityLine}>{activityLine}</p>
          </div>
        </div>

        {ready && band && probability != null ? (
          <>
            <div
              className="hero-track"
              role="img"
              aria-label={`专注概率 ${(probability * 100).toFixed(1)}%，处于${BAND_ZONE[band]}区间`}
              style={heroTrackStyle}
            >
              <span className="hero-track-band" aria-hidden="true" />
              <span
                className="hero-cursor"
                style={{
                  background: bandColor,
                  boxShadow: band === "focus" ? "var(--signal-glow)" : undefined,
                }}
                aria-hidden="true"
              />
            </div>
            <div className="hero-track-labels" aria-hidden="true">
              <span>分心</span>
              <span className="hero-track-mid">不确定区间</span>
              <span>专注</span>
            </div>

            <div className="hero-reasons">
              {contributions.length === 0 ? (
                <p className="hero-reason-empty">当前模型不提供逐项依据</p>
              ) : (
                <>
                  {positiveReasons.length > 0 && (
                    <div className="hero-reason">
                      <span className="hero-reason-dot" data-tone="up" aria-hidden="true" />
                      <span className="hero-reason-label">支持专注</span>
                      <span className="hero-reason-items">
                        {positiveReasons.map((row) => row.label).join(" · ")}
                      </span>
                    </div>
                  )}
                  {negativeReasons.length > 0 && (
                    <div className="hero-reason">
                      <span className="hero-reason-dot" data-tone="down" aria-hidden="true" />
                      <span className="hero-reason-label">拉向分心</span>
                      <span className="hero-reason-items">
                        {negativeReasons.map((row) => row.label).join(" · ")}
                      </span>
                    </div>
                  )}
                </>
              )}
            </div>

            {contributions.length > 0 && (
              <div className="hero-evidence">
                <Disclosure
                  summary="查看完整依据"
                  onOpenChange={setEvidenceOpen}
                  help={{
                    what: "展开后能看到哪些习惯让系统觉得你专注、哪些让它觉得你分心。",
                  }}
                >
                  <p className="hero-method">
                    方法：{METHOD_LABELS[explanation?.method ?? "unavailable"]} · 按影响大小排序，最多 8 项
                  </p>
                  <ContributionBar
                    data={contributions.map((row) => ({
                      feature: row.feature,
                      label: row.label,
                      contribution: row.contribution,
                      value: row.value,
                      baselineValue: row.baseline_value,
                    }))}
                    ariaLabel="专注预测的特征贡献：右侧推向专注，左侧推向分心"
                    outputSpace={explanation?.output_space}
                    baseValue={explanation?.base_value}
                  />
                  {explanation?.note && <p className="hero-note">{explanation.note}</p>}
                  {activity && currentActivity && (
                    <div className="hero-window">
                      <h3 className="sub-title">当前窗口</h3>
                      <dl className="hero-window-list">
                        <dt>应用</dt>
                        <dd>{activity.app_name || "—"}</dd>
                        <dt>窗口</dt>
                        <dd className="hero-window-title" title={activity.window_title || undefined}>
                          {activity.window_title || "—"}
                        </dd>
                        <dt>进程</dt>
                        <dd>{activity.process_name || "—"}</dd>
                        <dt>状态</dt>
                        <dd>{activity.is_idle ? "空闲" : "活跃"}</dd>
                        <dt>最后更新</dt>
                        <dd>
                          {new Date(currentActivity.timestamp).toLocaleString("zh-CN", { hour12: false })}
                          {" "}
                          {utcOffsetLabel(new Date(currentActivity.timestamp))}
                        </dd>
                      </dl>
                    </div>
                  )}
                </Disclosure>
              </div>
            )}
          </>
        ) : (
          <p className="hero-unavailable">
            <span>{unavailableCopy}</span>
            {showModelLink && (
              <Link className="btn btn-secondary btn-sm" to="/model-center">查看模型状态</Link>
            )}
          </p>
        )}

        <div className="hero-source">{sourceLine}</div>
      </Surface>

      {/* ── 下一步 ───────────────────────────────────────────────────────── */}
      <Surface as="section" className="dash-next" aria-labelledby="next-title">
        <h2 id="next-title" className="sh-title dash-next-title">下一步</h2>

        {latest ? (
          <>
            <p className="next-message" title={latest.message || latest.title || undefined}>
              {latest.message || latest.title || getInterventionTypeLabel(latest.intervention_type)}
            </p>
            <p className="next-meta">
              {getInterventionTypeLabel(latest.intervention_type)}
              {" · "}
              {formatClock(latest.triggered_at || latest.created_at)}
              {" · "}
              <span className={latestAnswered ? "next-response next-response--accepted" : "next-response"}>
                {latestResponse}
              </span>
            </p>
            <div className="next-actions">
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => navigate("/intervention")}>
                去干预中心
              </button>
            </div>
            <p className="next-count">最近 7 天共 {interventions.length} 条提醒</p>
          </>
        ) : (
          <>
            <p className="next-empty">目前不需要提醒，保持现在的节奏。</p>
            <Link className="btn btn-secondary btn-sm next-link" to="/intervention">去干预中心</Link>
          </>
        )}

        <div className="next-auto">
          <div className="next-auto-head">
            <span className="sub-title next-auto-title">
              自动提醒
              <HelpTip
                term="自动提醒"
                what="系统发现你可能分心时，主动弹出的小提示。"
                effect="暂停后，这段时间不弹提醒，每天自动生成的 AI 分析也会暂停；记录照常进行，时间到了自动恢复。"
              />
            </span>
            <span className={autonomyPaused ? "next-auto-state next-auto-state--paused" : "next-auto-state"}>
              {autonomyPaused
                ? pausedUntilLabel ? `已暂停至 ${pausedUntilLabel}` : "已暂停"
                : "开启中"}
            </span>
          </div>

          {autonomyPaused ? (
            <div className="next-auto-body">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleResume}
                disabled={autonomyLoading}
              >
                {autonomyLoading ? "处理中…" : "恢复提醒"}
              </button>
              <p className="next-auto-hint">暂停期间不会推送干预</p>
            </div>
          ) : customOpen ? (
            <div className="next-auto-body">
              <input
                className="next-custom-input"
                placeholder="0"
                value={controlTime.hour}
                type="number"
                min="0"
                aria-label="暂停小时"
                onChange={(e) => setControlTime({ ...controlTime, hour: e.target.value })}
              />
              <span className="next-custom-unit">小时</span>
              <input
                className="next-custom-input"
                placeholder="0"
                value={controlTime.minute}
                type="number"
                min="0"
                max="59"
                aria-label="暂停分钟"
                onChange={(e) => setControlTime({ ...controlTime, minute: e.target.value })}
              />
              <span className="next-custom-unit">分钟</span>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handlePause}
                disabled={autonomyLoading}
              >
                {autonomyLoading ? "处理中…" : "暂停"}
              </button>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => setCustomOpen(false)}
                disabled={autonomyLoading}
              >
                取消
              </button>
            </div>
          ) : (
            <div className="next-auto-body">
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => pauseFor(30)}
                disabled={autonomyLoading}
              >
                {autonomyLoading ? "处理中…" : "暂停 30 分钟"}
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => pauseFor(60)}
                disabled={autonomyLoading}
              >
                {autonomyLoading ? "处理中…" : "暂停 1 小时"}
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => pauseFor(minutesLeftToday)}
                disabled={autonomyLoading || minutesLeftToday < 30}
                title={minutesLeftToday < 30 ? "今天剩余不足 30 分钟" : undefined}
              >
                {autonomyLoading ? "处理中…" : "今天不再提醒"}
              </button>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setCustomOpen(true)}
                disabled={autonomyLoading}
              >
                自定义
              </button>
            </div>
          )}
        </div>
      </Surface>

      {/* ── 今天 ─────────────────────────────────────────────────────────── */}
      <MetricStrip
        label="今天"
        items={metrics}
        note={trendStatus === "error" ? "趋势数据加载失败" : undefined}
      />

      {/* ── 最近 7 天 ────────────────────────────────────────────────────── */}
      <Surface as="section" className="dash-trend">
        <SectionHeader
          title="最近 7 天"
          summary="每日专注时长（分钟）"
          actions={
            <div className="trend-legend">
              {hasTodaySlot && (
                <span className="trend-legend-item">
                  <span className="trend-legend-swatch trend-legend-swatch--today" aria-hidden="true" />
                  今天
                </span>
              )}
              <span className="trend-legend-item">
                <span className="trend-legend-swatch" aria-hidden="true" />
                往日
              </span>
              {showMean && (
                <span className="trend-legend-item">
                  <span className="trend-legend-line" aria-hidden="true" />
                  7 日均值 {Math.round(meanFocus)} 分钟
                </span>
              )}
            </div>
          }
        />
        <BarChart
          data={chartbox.map((item) => ({
            key: item.key,
            label: item.week,
            value: item.number,
            title: `${item.week} · ${item.key}`,
            highlight: item.today,
            detail: [
              { label: "分心", value: `${item.distraction} 分钟` },
              { label: "会话次数", value: `${item.sessions} 次` },
              { label: "平均分", value: item.score != null ? item.score.toFixed(1) : "—" },
            ],
          }))}
          ariaLabel={`近 7 日每日专注分钟数，共 ${chartbox.length} 天`}
          unit=" 分钟"
          height={220}
          colorFor={(datum) => (datum.highlight ? "var(--signal)" : "var(--steel)")}
          referenceLine={referenceLine}
          htmlXLabelClass="d-chart-box-item-week"
          emptyMessage={trendStatus === "error" ? "趋势加载失败（数据未更新）" : "暂无近 7 日趋势数据"}
        />
      </Surface>
    </div>
  );
}
