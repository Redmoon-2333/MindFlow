import { useState, useEffect, useCallback, useRef } from "react";
import {
  getHealth,
  getCurrentActivity,
  getFocusTrend,
  getFocusPrediction,
  getModelStatus,
  getInterventionHistory,
  getCollectorStatus,
  getAutonomy,
  startCollector,
  stopCollector,
  resumeAutonomy,
  pauseAutonomy,
  getErrorMessage,
} from "../api";
import type { ActivityItem, AutonomyStatus, CollectorStatus, FocusPredictionResponse, FocusTrendResponse, HealthData, InterventionHistoryItem, ModelStatus } from "../api";
import { deriveFocusTrendKpi } from "../api";
import { toFocusPredictionView } from "../prediction-state";
import { realtimeClient } from "../realtime";
import type { RealtimeStatus } from "../realtime";
import { getInterventionTypeLabel } from "../lib/intervention-labels";

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthData | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [focusTrend, setFocusTrend] = useState<FocusTrendResponse | null>(null);
  const [currentActivity, setCurrentActivity] = useState<ActivityItem | null>(null);
  const [interventions, setInterventions] = useState<InterventionHistoryItem[]>([]);
  const [collector, setCollector] = useState<CollectorStatus | null>(null);
  const [autonomy, setAutonomy] = useState<AutonomyStatus | null>(null);
  const [focusPrediction, setFocusPrediction] = useState<FocusPredictionResponse | null>(null);

  const [collectorLoading, setCollectorLoading] = useState(false);
  const [autonomyLoading, setAutonomyLoading] = useState(false);
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>("idle");

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    const results = await Promise.allSettled([
      getHealth(), getModelStatus(), getFocusTrend(7), getCurrentActivity(),
      getInterventionHistory(7), getCollectorStatus(), getAutonomy(), getFocusPrediction(),
    ]);
    const [h, ms, ft, ca, ih, cs, au, fp] = results;
    if (h.status === "fulfilled") setHealth(h.value);
    if (ms.status === "fulfilled") setModelStatus(ms.value);
    if (ft.status === "fulfilled") setFocusTrend(ft.value);
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

  const handleCollectorToggle = async () => {
    setCollectorLoading(true);
    try {
      const next = collector?.running ? await stopCollector() : await startCollector();
      setCollector(next);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "Operation failed"));
    } finally {
      setCollectorLoading(false);
    }
  };

  const handlePause = async () => {
    setAutonomyLoading(true);
    try {
      const result = await pauseAutonomy(1);
      setAutonomy(result);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "Operation failed"));
    } finally {
      setAutonomyLoading(false);
    }
  };

  const handleResume = async () => {
    setAutonomyLoading(true);
    try {
      const result = await resumeAutonomy();
      setAutonomy(result);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "Operation failed"));
    } finally {
      setAutonomyLoading(false);
    }
  };

  if (loading) {
    return (
      <div>
        <div className="header">
          <h1>仪表盘</h1>
          <p>MindFlow 系统概览</p>
        </div>
        <div className="spinner" />
      </div>
    );
  }

  const predictionView = focusPrediction ? toFocusPredictionView(focusPrediction) : null;
  const kpi = deriveFocusTrendKpi(focusTrend);

  return (
    <div>
      <div className="header">
        <h1>仪表盘</h1>
        <p>MindFlow 系统概览</p>
      </div>

      {error && (
        <div className="error-box mb16">
          {error}
          <button className="btn btn-sm mt8" onClick={fetchData} style={{ marginLeft: 12 }}>
            重试
          </button>
        </div>
      )}

      {/* KPI Row — derived from /focus/trend daily array (see deriveFocusTrendKpi) */}
      <div className="kpi-row">
        <div className="stat-card">
          <div className="label">今日专注时长</div>
          <div className="value">
            {kpi.todayMinutes != null
              ? `${Math.round(kpi.todayMinutes)}m`
              : kpi.totalMinutes != null
                ? `${Math.round(kpi.totalMinutes)}m`
                : "--"}
          </div>
          <div className="sub good">
            {kpi.trendLabel ?? ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">专注会话数</div>
          <div className="value">
            {kpi.sessionCount != null ? kpi.sessionCount : "--"}
          </div>
          <div className="sub">{kpi.avgDurationMinutes != null ? `均长 ${Math.round(kpi.avgDurationMinutes)}m` : ""}</div>
        </div>
        <div className="stat-card">
          <div className="label">平均专注评分</div>
          <div className="value">
            {kpi.avgScore != null ? kpi.avgScore.toFixed(1) : "--"}
          </div>
          <div className="sub">
            {kpi.scoreChange != null
              ? `${kpi.scoreChange > 0 ? "+" : ""}${kpi.scoreChange.toFixed(1)}%`
              : ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">分心率</div>
          <div className="value">
            {kpi.distractionRate != null
              ? `${(kpi.distractionRate * 100).toFixed(1)}%`
              : "--"}
          </div>
          <div className={kpi.distractionRate != null && kpi.distractionRate > 0.3 ? "sub bad" : "sub good"}>
            {kpi.distractionLabel ?? ""}
          </div>
        </div>
      </div>

      {/* Two Columns */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Left Column: System Status */}
        <div>
          {/* Health Status */}
          <div className="card mb16">
            <div className="flex-between mb16">
              <h3 style={{ marginBottom: 0 }}>系统健康状态</h3>
              <span
                className={`badge ${health?.status === "ok" || health?.status === "healthy" ? "badge-success" : "badge-danger"}`}
              >
                {health?.status ?? "未知"}
              </span>
            </div>
            <div className="flex gap8" style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
              <span>版本: {health?.version ?? "--"}</span>
            </div>
          </div>

          {/* Collector Status */}
          <div className="card mb16">
            <div className="flex-between mb16">
              <h3 style={{ marginBottom: 0 }}>采集器状态</h3>
              <span
                className={`badge ${collector?.running ? "badge-success" : "badge-warning"}`}
              >
                {collector?.running ? "运行中" : "已停止"}
              </span>
            </div>
            {collector?.status && (
              <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
                {collector.status} · 实时连接：{realtimeStatus}
              </div>
            )}
            <button
              className={`btn btn-sm ${collector?.running ? "btn-danger" : ""}`}
              onClick={handleCollectorToggle}
              disabled={collectorLoading}
            >
              {collectorLoading ? "处理中..." : collector?.running ? "停止采集" : "启动采集"}
            </button>
          </div>

          {/* Database Status */}
          <div className="card mb16">
            <div className="flex-between mb16">
              <h3 style={{ marginBottom: 0 }}>数据库状态</h3>
              <span
                className={`badge ${health?.database.status === "ok" ? "badge-success" : "badge-danger"}`}
              >
                {health?.database.status === "ok" ? "正常" : "异常"}
              </span>
            </div>
          </div>

          {/* LLM Status */}
          <div className="card mb16">
            <div className="flex-between mb16">
              <h3 style={{ marginBottom: 0 }}>ML 模型状态</h3>
              <span
                className={`badge ${modelStatus?.loaded ? "badge-success" : "badge-info"}`}
              >
                {modelStatus?.loaded ? "已加载" : "规则引擎模式"}
              </span>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div>
          {/* Current Activity */}
          <div className="card mb16">
            <h3>当前活动</h3>
            {currentActivity ? (
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                  {currentActivity.data.app_name || "未知应用"}
                </div>
                <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                  {currentActivity.data.window_title || currentActivity.data.process_name}
                  <span style={{ marginLeft: 8 }}>
                    {new Date(currentActivity.timestamp).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
                {currentActivity.duration_s > 0 && (
                  <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginTop: 4 }}>
                    持续 {Math.round(currentActivity.duration_s / 60)} 分钟
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
                当前没有活动记录
              </div>
            )}
          </div>

          {/* Focus Prediction */}
          <div className="card mb16">
            <div className="flex-between mb8">
              <h3 style={{ marginBottom: 0 }}>ML 专注预测</h3>
              <span
                className={`badge ${predictionView ? (predictionView.ready ? "badge-success" : "badge-warning") : "badge-info"}`}
              >
                {predictionView ? predictionView.statusLabel : "未获取"}
              </span>
            </div>
            {predictionView ? (
              <div>
                <div style={{ fontSize: 24, fontWeight: 700, marginBottom: 4 }}>
                  {predictionView.display}
                </div>
                <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                  {predictionView.reason || `预测模式: ${predictionView.mode}`}
                </div>
              </div>
            ) : (
              <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
                暂无预测数据
              </div>
            )}
          </div>

          {/* Recent Interventions */}
          <div className="card">
            <h3>近期干预记录</h3>
            {interventions.length === 0 ? (
              <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
                最近 7 天无干预记录
              </div>
            ) : (
              <div style={{ maxHeight: 320, overflowY: "auto" }}>
                {interventions.slice(0, 10).map((item, idx) => (
                  <div
                    key={item.id ?? idx}
                    style={{
                      padding: "10px 0",
                      borderBottom: idx < Math.min(interventions.length, 10) - 1 ? "1px solid var(--color-border)" : "none",
                    }}
                  >
                    <div className="flex-between">
                      <span style={{ fontSize: 13, fontWeight: 500 }}>
                        {item.title || getInterventionTypeLabel(item.intervention_type)}
                      </span>
                      <span
                        className={`badge ${
                          item.user_response ? "badge-success" : "badge-warning"
                        }`}
                      >
                        {item.user_response ? "已响应" : "待响应"}
                      </span>
                    </div>
                    {item.message && (
                      <div
                        style={{
                          fontSize: 12,
                          color: "var(--color-text-secondary)",
                          marginTop: 4,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {item.message}
                      </div>
                    )}
                    <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginTop: 4 }}>
                      {new Date(item.triggered_at || item.created_at).toLocaleString("zh-CN")}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Autonomy Control */}
      <div className="card mt16">
        <div className="flex-between mb16">
          <div>
            <h3 style={{ marginBottom: 4 }}>自主控制</h3>
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
              {autonomy?.paused
                ? `已暂停${autonomy?.paused_until ? `，预计 ${new Date(autonomy.paused_until).toLocaleString("zh-CN")} 恢复` : ""}`
                : autonomy?.enabled !== false
                  ? "自主模式运行中"
                  : "自主模式已关闭"}
            </div>
          </div>
          <span
            className={`badge ${autonomy?.paused ? "badge-warning" : autonomy?.enabled !== false ? "badge-success" : "badge-info"}`}
          >
            {autonomy?.paused ? "已暂停" : autonomy?.enabled !== false ? "运行中" : "已关闭"}
          </span>
        </div>
        <div className="flex gap8">
          {autonomy?.paused ? (
            <button className="btn btn-sm" onClick={handleResume} disabled={autonomyLoading}>
              {autonomyLoading ? "处理中..." : "恢复自主模式"}
            </button>
          ) : (
            <button className="btn btn-sm btn-danger" onClick={handlePause} disabled={autonomyLoading}>
              {autonomyLoading ? "处理中..." : "暂停 1 小时"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
