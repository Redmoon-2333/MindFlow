import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  getHealth,
  getCurrentActivity,
  getFocusTrend,
  getFocusPrediction,
  getModelStatus,
  getAiProviderStatus,
  getInterventionHistory,
  getCollectorStatus,
  getAutonomy,
  startCollector,
  stopCollector,
  resumeAutonomy,
  pauseAutonomy,
  getErrorMessage,
} from "../api";
import type {
  ActivityItem,
  AiProviderStatus,
  AutonomyStatus,
  CollectorStatus,
  FocusPredictionResponse,
  FocusTrendDay,
  FocusTrendResponse,
  HealthData,
  InterventionHistoryItem,
  ModelStatus,
} from "../api";
import { deriveFocusTrendKpi } from "../api";
import { dayLabel, findDayByDate, localDateStr, previousLocalDateStr } from "../date-utils";
import { toFocusPredictionView } from "../prediction-state";
import { realtimeClient } from "../realtime";
import type { RealtimeStatus } from "../realtime";
import { getInterventionTypeLabel } from "../lib/intervention-labels";
import "./dashboard.css";

const WEEKDAY_LABELS = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];

function formatFloatHeader(date: Date): string {
  return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}  ${WEEKDAY_LABELS[date.getDay()]}`;
}

function formatClock(iso: string | undefined): string {
  if (!iso) return "--:--";
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return "--:--";
  return parsed.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
}

/** Percentage change between two days, or null when yesterday is missing —
 *  a fabricated "0%" would read as a real comparison. */
function dayOverDay(today: number | undefined, previous: number | undefined): number | null {
  if (today == null || previous == null) return null;
  if (previous === 0) return today === 0 ? 0 : null;
  return ((today - previous) / previous) * 100;
}

interface StatisticView {
  title: string;
  content: string;
  unit: string;
  delta: number | null;
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthData | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [providerStatus, setProviderStatus] = useState<AiProviderStatus | null>(null);
  const [focusTrend, setFocusTrend] = useState<FocusTrendResponse | null>(null);
  const [currentActivity, setCurrentActivity] = useState<ActivityItem | null>(null);
  const [interventions, setInterventions] = useState<InterventionHistoryItem[]>([]);
  const [collector, setCollector] = useState<CollectorStatus | null>(null);
  const [autonomy, setAutonomy] = useState<AutonomyStatus | null>(null);
  const [focusPrediction, setFocusPrediction] = useState<FocusPredictionResponse | null>(null);
  /** Distinguishes "no record today" (0) from "we could not ask" (--).
   *  Both used to collapse into `trend == null`, which made a failed request
   *  indistinguishable from a genuinely empty day (audit F1). */
  const [trendStatus, setTrendStatus] = useState<"loading" | "ready" | "error">("loading");

  const [collectorLoading, setCollectorLoading] = useState(false);
  const [autonomyLoading, setAutonomyLoading] = useState(false);
  const [controlTime, setControlTime] = useState({ hour: "0", minute: "0" });
  const [now, setNow] = useState(() => new Date());
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>("idle");

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    setTrendStatus("loading");
    const results = await Promise.allSettled([
      getHealth(), getModelStatus(), getAiProviderStatus(), getFocusTrend(7),
      getCurrentActivity(), getInterventionHistory(7), getCollectorStatus(),
      getAutonomy(), getFocusPrediction(),
    ]);
    const [h, ms, ps, ft, ca, ih, cs, au, fp] = results;
    if (h.status === "fulfilled") setHealth(h.value);
    if (ms.status === "fulfilled") setModelStatus(ms.value);
    if (ps.status === "fulfilled") setProviderStatus(ps.value);
    if (ft.status === "fulfilled") {
      setFocusTrend(ft.value);
      setTrendStatus("ready");
    } else {
      // A rejected trend must not keep serving the previous range's numbers.
      setFocusTrend(null);
      setTrendStatus("error");
    }
    if (ca.status === "fulfilled") setCurrentActivity(ca.value);
    if (ih.status === "fulfilled") setInterventions([...ih.value.items].reverse());
    if (cs.status === "fulfilled") setCollector(cs.value);
    if (au.status === "fulfilled") setAutonomy(au.value);
    if (fp.status === "fulfilled") setFocusPrediction(fp.value);
    const failed = results.filter((result) => result.status === "rejected");
    if (failed.length > 0) setError(`部分数据加载失败（${failed.length} 项），其余内容已显示`);
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Clock behind the floating header (date + weekday) — the reference ticks
  // off a module-level `new Date()`, so refresh it on a slow interval.
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

  /** Collector toggle — its own endpoint, independent from autonomy. */
  const handleCollectorToggle = async () => {
    setCollectorLoading(true);
    try {
      const next = collector?.running ? await stopCollector() : await startCollector();
      setCollector(next);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "采集器开关失败"));
    } finally {
      setCollectorLoading(false);
    }
  };

  /** Pause hours + minutes collapse into one hour count; the backend keeps
   *  its 30-minute floor (`hours >= 0.5`). */
  const handlePause = async () => {
    const hours = Number.parseInt(controlTime.hour, 10) || 0;
    const minutes = Number.parseInt(controlTime.minute, 10) || 0;
    const totalMinutes = hours * 60 + Math.min(Math.max(minutes, 0), 59);
    if (totalMinutes < 30) {
      setError("暂停时长至少 30 分钟");
      return;
    }
    setAutonomyLoading(true);
    setError(null);
    try {
      const result = await pauseAutonomy(totalMinutes / 60);
      setAutonomy(result);
      setControlTime({ hour: "0", minute: "0" });
    } catch (e: unknown) {
      setError(getErrorMessage(e, "暂停干预失败"));
    } finally {
      setAutonomyLoading(false);
    }
  };

  const handleResume = async () => {
    setAutonomyLoading(true);
    setError(null);
    try {
      const result = await resumeAutonomy();
      setAutonomy(result);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "恢复干预失败"));
    } finally {
      setAutonomyLoading(false);
    }
  };

  const predictionView = focusPrediction ? toFocusPredictionView(focusPrediction) : null;
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
  const collectorRunning = collector?.running === true;
  const autonomyPaused = autonomy?.paused === true || (autonomy?.paused_until != null && new Date(autonomy.paused_until).getTime() > Date.now());

  // Seven-day bar chart: only a slot that really is today may read 今天.
  const chartbox = daily.map((day) => ({
    week: day.date === todayKey ? "今天" : dayLabel(day.date),
    number: Math.round(day.focus_min ?? 0),
    today: day.date === todayKey,
  }));
  const hasTodaySlot = chartbox.some((entry) => entry.today);
  const maxNumber = Math.max(1, ...chartbox.map((entry) => entry.number));

  const statistics: StatisticView[] = [
    {
      title: "今日专注时长",
      // ready + no record → 0 (we looked and found nothing);
      // request failed  → -- (we do not know).
      content: !trendKnown ? "--" : String(Math.round(today?.focus_min ?? 0)),
      unit: "min",
      delta: trendKnown ? dayOverDay(today?.focus_min, yesterday?.focus_min) : null,
    },
    {
      title: "专注次数",
      content: !trendKnown ? "--" : String(today?.session_count ?? 0),
      unit: "次",
      delta: trendKnown ? dayOverDay(today?.session_count, yesterday?.session_count) : null,
    },
    {
      title: "平均专注分数",
      // No observation today → no average to show, never a fabricated 0.
      content:
        trendKnown && today && today.avg_score != null ? today.avg_score.toFixed(1) : "--",
      unit: "/100",
      delta: trendKnown ? dayOverDay(today?.avg_score, yesterday?.avg_score) : null,
    },
    {
      title: "分心比率",
      content: kpi.distractionRate != null ? (kpi.distractionRate * 100).toFixed(1) : "--",
      unit: "%",
      delta: null,
    },
  ];

  const stateItems = [
    {
      title: "系统健康",
      content: health ? (health.status === "ok" || health.status === "healthy" ? "正常" : health.status) : "--",
      icon: "icon-bodong",
      on: health != null && (health.status === "ok" || health.status === "healthy"),
      action: undefined as (() => void) | undefined,
    },
    {
      title: "采集器",
      content: collector ? (collectorRunning ? "运行中" : "已停止") : "--",
      icon: "icon-shizhong",
      on: collectorRunning,
      action: handleCollectorToggle,
    },
    {
      title: "数据库",
      content: health ? (health.database.status === "ok" ? "已连接" : "异常") : "--",
      icon: "icon-shujuku",
      on: health?.database.status === "ok",
      action: undefined,
    },
    {
      title: "LLM层",
      content: providerStatus
        ? providerStatus.configured
          ? `已配置 · ${providerStatus.model}`
          : "未配置"
        : modelStatus?.loaded
          ? "已加载"
          : "--",
      icon: "icon-yuangongguanli",
      on: providerStatus?.configured === true,
      action: undefined,
    },
  ];

  const pausedUntil = autonomy?.paused_until ? new Date(autonomy.paused_until) : null;
  const tipText = !autonomyPaused
    ? "自主模式会根据行为模式自动推送干预建议"
    : pausedUntil
      ? `已暂停至${pausedUntil.getMonth() + 1}月${pausedUntil.getDate()}日${String(pausedUntil.getHours()).padStart(2, "0")}:${String(pausedUntil.getMinutes()).padStart(2, "0")}，期间不会推送干预`
      : "已暂停推送干预";

  if (loading) {
    return (
      <div className="d-container">
        <div className="d-content">
          <div className="spinner" />
        </div>
      </div>
    );
  }

  return (
    <div className="d-container">
      <div className="d-float-header">
        <div className="d-time">{formatFloatHeader(now)}</div>
        <button
          type="button"
          className="d-switch"
          title="采集器开关"
          aria-pressed={collectorRunning}
          disabled={collectorLoading}
          style={{
            color: collectorRunning ? "var(--mf-green)" : "#94A3B8",
            border: `1px solid ${collectorRunning ? "var(--mf-green)" : "#94A3B8"}`,
          }}
          onClick={handleCollectorToggle}
        >
          <span
            className="d-switch-dot"
            style={{
              backgroundColor: collectorRunning ? "var(--mf-green)" : "#94A3B8",
              animation: collectorRunning ? "d-colorChange 2s linear infinite" : "none",
            }}
          />
          {"\u00a0\u00a0\u00a0"}
          {collectorLoading ? "切换中" : collectorRunning ? "采集中" : "已停止"}
        </button>
      </div>

      {error && (
        <div className="error-box" role="alert">
          {error}
          <button className="btn btn-sm" onClick={fetchData} style={{ marginLeft: 12 }}>
            重试
          </button>
        </div>
      )}

      <div className="d-content">
        <div className="d-state">
          {stateItems.map((item) => (
            <div
              className="d-state-item"
              key={item.title}
              title={item.action ? "采集器开关" : undefined}
              style={item.action ? { cursor: "pointer" } : undefined}
              onClick={item.action}
              role={item.action ? "button" : undefined}
              tabIndex={item.action ? 0 : undefined}
              onKeyDown={
                item.action
                  ? (event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        item.action?.();
                      }
                    }
                  : undefined
              }
            >
              <div className="d-state-title">{item.title}</div>
              <div className={item.on ? "d-state-content" : "d-state-close"}>
                {item.content}
              </div>
              <div className={item.on ? "d-icon-box" : "d-icon-box d-icon-box-close"}>
                <div className={`icon iconfont ${item.icon}`} aria-hidden="true" />
              </div>
            </div>
          ))}
        </div>

        <div className="d-activity-chart">
          <div className="d-activity">
            <div className="d-activity-title">当前活动</div>
            {currentActivity ? (
              <>
                <div className="d-activity-item">
                  <div className="d-activity-item-title">应用程序</div>
                  <div className="d-activity-item-content">{currentActivity.data.app_name || "未知应用"}</div>
                </div>
                <div className="d-activity-item">
                  <div className="d-activity-item-title">窗口标题</div>
                  <div className="d-activity-item-content">
                    {currentActivity.data.window_title || "—"}
                  </div>
                </div>
                <div className="d-activity-item">
                  <div className="d-activity-item-title">进程名称</div>
                  <div className="d-activity-item-content">
                    {currentActivity.data.process_name || "—"}
                  </div>
                </div>
                <div className="d-activity-item">
                  <div className="d-activity-item-title">空闲状态</div>
                  <div className="d-activity-item-contain">
                    <div className={currentActivity.data.is_idle ? "d-activity-item-contain-content d-activity-item-contain-content--off" : "d-activity-item-contain-content"}>
                      {currentActivity.data.is_idle ? "空闲" : "活跃"}
                    </div>
                    <div className="d-activity-item-contain-content">实时</div>
                  </div>
                </div>
                <div className="d-subline" />
                <div className="d-activity-time">
                  <div className="iconfont icon-shizhong" style={{ transform: "translateY(1px)" }} aria-hidden="true" />
                  <div className="d-activity-time-content">
                    最后更新: {new Date(currentActivity.timestamp).toLocaleString("zh-CN", { hour12: false })} UTC+8
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="d-empty">当前没有活动记录{realtimeStatus === "connected" ? "（实时连接中）" : ""}</div>
                <div className="d-subline" />
                <div className="d-activity-time">
                  <div className="iconfont icon-shizhong" style={{ transform: "translateY(1px)" }} aria-hidden="true" />
                  <div className="d-activity-time-content">最后更新: —</div>
                </div>
              </>
            )}
          </div>

          <div className="d-chart">
            <div className="d-chart-title">近 7 日专注趋势</div>
            {chartbox.length > 0 ? (
              <>
                <div className="d-chart-box">
                  {chartbox.map((item, index) => (
                    <div className="d-chart-box-item" key={`${item.week}-${index}`}>
                      <div className="d-chart-box-item-week">{item.week}</div>
                      <div
                        className={`d-chart-box-item-column${item.today ? " d-chart-box-item-column--today" : ""}`}
                        style={{ height: `${Math.max(4, (item.number / maxNumber) * 92)}px` }}
                      >
                        <div className="d-chart-box-item-number">{item.number}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="d-subline" />
                <div className="d-chart-demonstration">
                  <div className="d-chart-demonstration-item">
                    <div className="d-chart-demonstration-item-dot" />
                    <div className="d-chart-demonstration-item-text">专注时长 (分钟)</div>
                  </div>
                  {hasTodaySlot && (
                    <div className="d-chart-demonstration-item">
                      <div className="d-chart-demonstration-item-dot d-chart-demonstration-item-dot--today" />
                      <div className="d-chart-demonstration-item-text">今天 (进行中)</div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="d-empty">
                {trendStatus === "error" ? "趋势加载失败（数据未更新）" : "暂无近 7 日趋势数据"}
              </div>
            )}
          </div>
        </div>

        <div className="d-statistic">
          {statistics.map((item) => (
            <div className="d-statistic-item" key={item.title}>
              <div className="d-statistic-title">{item.title}</div>
              <div className="d-statistic-content">
                <div className="d-statistic-number">{item.content}</div>
                <div className="d-statistic-unit">{item.unit}</div>
              </div>
              {/* The icon row is always present so the comparison line keeps
                  the reference's vertical rhythm even without a delta. */}
              <div
                className={`icon iconfont d-statistic-icon${item.delta == null ? " d-statistic-icon--flat" : ""} ${item.delta != null && item.delta > 0 ? "icon-tubiaoshangshengqushi" : "icon-xiajiang"}`}
                style={{ color: item.delta == null ? "#6a89ad" : item.delta > 0 ? "#22C55E" : "#EF4444" }}
                aria-hidden="true"
              />
              <div
                className={`d-statistic-stat${item.delta == null ? " d-statistic-stat--flat" : ""}`}
                style={item.delta == null ? undefined : { color: item.delta > 0 ? "#22C55E" : "#EF4444" }}
              >
                {item.delta != null
                  ? `较昨日 ${item.delta > 0 ? "+" : ""}${item.delta.toFixed(1)}%`
                  : "较昨日 —"}
              </div>
            </div>
          ))}
        </div>

        <div className="d-control-inference">
          <div className="d-control">
            <div className="d-control-title">自主控制</div>
            <div className="d-control-module">
              <div className={autonomyPaused ? "d-control-module-jingao" : "d-control-module-statement"}>
                {autonomyPaused ? "暂停中" : "已启动"}
              </div>
              <div className="d-control-module-text">
                {autonomyPaused ? "自主干预已暂停" : "自主干预模式运行中"}
              </div>
            </div>
            <div className="d-control-pause">
              <div className="d-control-pause-text">暂停</div>
              <input
                className="d-control-pause-time"
                placeholder="0"
                value={controlTime.hour}
                type="number"
                min="0"
                aria-label="暂停小时"
                disabled={autonomyPaused}
                onChange={(e) => setControlTime({ ...controlTime, hour: e.target.value })}
              />
              <div className="d-control-pause-text">小时</div>
              <input
                className="d-control-pause-time"
                placeholder="0"
                value={controlTime.minute}
                type="number"
                min="0"
                max="59"
                aria-label="暂停分钟"
                disabled={autonomyPaused}
                onChange={(e) => setControlTime({ ...controlTime, minute: e.target.value })}
              />
              <div className="d-control-pause-text">分钟</div>
              <button
                className="d-control-submit"
                onClick={handlePause}
                disabled={autonomyPaused || autonomyLoading}
              >
                {autonomyLoading ? "处理中..." : "暂停干预"}
              </button>
              <button
                className="d-control-restore"
                onClick={handleResume}
                disabled={!autonomyPaused || autonomyLoading}
              >
                {autonomyLoading ? "处理中..." : "恢复"}
              </button>
            </div>
            <div className={`d-control-tip${autonomyPaused ? " d-control-tip--warn" : ""}`}>
              <div
                className={`iconfont icon-jinggao1${autonomyPaused ? " d-control-tip--warn" : ""}`}
                aria-hidden="true"
              />
              <div className="control-tips-text">{tipText}</div>
            </div>
          </div>

          <div className="d-inference">
            <div className="d-inference-title">最近干预</div>
            <button type="button" className="d-todetail" onClick={() => navigate("/intervention")}>
              查看全部
            </button>
            {interventions.length === 0 ? (
              <div className="d-empty">最近 7 天无干预记录</div>
            ) : (
              <div className="d-inference-scroll">
                {[...interventions].reverse().slice(-10).map((item, index) => {
                  const warning = !item.user_response;
                  return (
                    <div className="d-inference-item" key={item.id ?? index}>
                      <div className={`icon-box${warning ? " icon-box-orange" : ""}`}>
                        <div className={`iconfont ${warning ? "icon-jinggao" : "icon-tixing"}`} aria-hidden="true" />
                      </div>
                      <div className="d-inference-text">
                        {item.message || item.title || getInterventionTypeLabel(item.intervention_type)}
                      </div>
                      <div className={`d-inference-tag${warning ? " d-inference-tag-light" : ""}`}>
                        {warning ? "警告" : "轻提示"}
                      </div>
                      <div className="d-inference-time">{formatClock(item.triggered_at || item.created_at)}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Local-only detail: the reference dashboard has no ML/health block,
            so it is appended after the replicable sections instead of being
            injected into them. */}
        <div className="d-extra">
          <div className="d-extra-card">
            <div className="d-extra-title">ML 专注预测</div>
            <div className="d-extra-value">
              {predictionView ? predictionView.display : "暂无预测数据"}
            </div>
            <div className="d-extra-note">
              {modelStatus?.demo_only
                ? `合成数据演示模型 · 影子模式${predictionView && !predictionView.ready
                  ? ` · ${predictionView.reason || predictionView.statusLabel}` : ""}`
                : predictionView
                ? predictionView.reason || `预测模式: ${predictionView.mode}`
                : "需积累数据后由模型中心生成"}
            </div>
          </div>
          <div className="d-extra-card">
            <div className="d-extra-title">服务状态</div>
            <div className="d-extra-value">
              {modelStatus?.demo_only
                ? "演示模型已加载 · 影子模式"
                : modelStatus?.loaded ? "模型已加载" : "规则引擎模式"}
              {!modelStatus?.demo_only && predictionView ? ` · ${predictionView.statusLabel}` : ""}
            </div>
            <div className="d-extra-note">
              版本 {health?.version ?? "--"} · 实时连接：{realtimeStatus}
              {modelStatus?.demo_only ? " · 未通过个人模型质量门" : ""}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
