import { useState, useEffect, useCallback } from "react";
import {
  getHealth,
  getCurrentActivity,
  getFocusTrend,
  getModelStatus,
  getInterventionHistory,
  getCollectorStatus,
  getAutonomy,
  startCollector,
  stopCollector,
  resumeAutonomy,
  pauseAutonomy,
} from "../api";
import type { HealthData } from "../api";

interface ModelStatusData {
  loaded: boolean;
  mode: string;
  message?: string;
}

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthData | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatusData | null>(null);
  const [focusTrend, setFocusTrend] = useState<any>(null);
  const [currentActivity, setCurrentActivity] = useState<any>(null);
  const [interventions, setInterventions] = useState<any[]>([]);
  const [collector, setCollector] = useState<any>(null);
  const [autonomy, setAutonomy] = useState<any>(null);

  const [collectorLoading, setCollectorLoading] = useState(false);
  const [autonomyLoading, setAutonomyLoading] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, ms, ft, ca, ih, cs, au] = await Promise.all([
        getHealth(),
        getModelStatus(),
        getFocusTrend(7),
        getCurrentActivity(),
        getInterventionHistory(7),
        getCollectorStatus(),
        getAutonomy(),
      ]);
      setHealth(h);
      setModelStatus(ms);
      setFocusTrend(ft);
      setCurrentActivity(ca);
      setInterventions(Array.isArray(ih) ? ih : ih?.items ?? []);
      setCollector(cs);
      setAutonomy(au);
    } catch (e: any) {
      setError(e.message ?? "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleCollectorToggle = async () => {
    setCollectorLoading(true);
    try {
      const next = collector?.running ? await stopCollector() : await startCollector();
      setCollector(next);
    } catch (e: any) {
      setError(e.message ?? "操作失败");
    } finally {
      setCollectorLoading(false);
    }
  };

  const handlePause = async () => {
    setAutonomyLoading(true);
    try {
      const result = await pauseAutonomy(1);
      setAutonomy(result);
    } catch (e: any) {
      setError(e.message ?? "操作失败");
    } finally {
      setAutonomyLoading(false);
    }
  };

  const handleResume = async () => {
    setAutonomyLoading(true);
    try {
      const result = await resumeAutonomy();
      setAutonomy(result);
    } catch (e: any) {
      setError(e.message ?? "操作失败");
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

      {/* KPI Row */}
      <div className="kpi-row">
        <div className="stat-card">
          <div className="label">今日专注时长</div>
          <div className="value">
            {focusTrend?.today_minutes != null
              ? `${Math.round(focusTrend.today_minutes)}m`
              : focusTrend?.total_minutes != null
                ? `${Math.round(focusTrend.total_minutes)}m`
                : "--"}
          </div>
          <div className="sub good">
            {focusTrend?.trend_label ?? ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">专注会话数</div>
          <div className="value">
            {focusTrend?.session_count != null ? focusTrend.session_count : "--"}
          </div>
          <div className="sub">{focusTrend?.avg_duration_minutes != null ? `均长 ${Math.round(focusTrend.avg_duration_minutes)}m` : ""}</div>
        </div>
        <div className="stat-card">
          <div className="label">平均专注评分</div>
          <div className="value">
            {focusTrend?.avg_score != null ? focusTrend.avg_score.toFixed(1) : "--"}
          </div>
          <div className="sub">
            {focusTrend?.score_change != null
              ? `${focusTrend.score_change > 0 ? "+" : ""}${focusTrend.score_change.toFixed(1)}%`
              : ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">分心率</div>
          <div className="value">
            {focusTrend?.distraction_rate != null
              ? `${(focusTrend.distraction_rate * 100).toFixed(1)}%`
              : "--"}
          </div>
          <div className={focusTrend?.distraction_rate > 0.3 ? "sub bad" : "sub good"}>
            {focusTrend?.distraction_label ?? ""}
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
                {collector.status}
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
                  {currentActivity.app_name ?? currentActivity.title ?? currentActivity.name ?? "未知应用"}
                </div>
                <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                  {currentActivity.category && (
                    <span className="badge badge-primary" style={{ marginRight: 8 }}>
                      {currentActivity.category}
                    </span>
                  )}
                  {currentActivity.started_at && (
                    <span>
                      开始于 {new Date(currentActivity.started_at).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}
                    </span>
                  )}
                </div>
                {(currentActivity.duration_seconds != null || currentActivity.elapsed != null) && (
                  <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginTop: 4 }}>
                    已持续 {Math.round(((currentActivity.duration_seconds ?? currentActivity.elapsed) / 60))} 分钟
                  </div>
                )}
              </div>
            ) : (
              <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
                当前没有活动记录
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
                {interventions.slice(0, 10).map((item: any, idx: number) => (
                  <div
                    key={item.id ?? idx}
                    style={{
                      padding: "10px 0",
                      borderBottom: idx < Math.min(interventions.length, 10) - 1 ? "1px solid var(--color-border)" : "none",
                    }}
                  >
                    <div className="flex-between">
                      <span style={{ fontSize: 13, fontWeight: 500 }}>
                        {item.type ?? item.intensity ?? "干预"}
                      </span>
                      <span
                        className={`badge ${
                          item.status === "responded" || item.responded
                            ? "badge-success"
                            : item.status === "pending"
                              ? "badge-warning"
                              : "badge-primary"
                        }`}
                      >
                        {item.status === "responded" || item.responded ? "已响应" : item.status ?? "待处理"}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginTop: 4 }}>
                      {item.created_at
                        ? new Date(item.created_at).toLocaleString("zh-CN")
                        : item.timestamp
                          ? new Date(item.timestamp).toLocaleString("zh-CN")
                          : ""}
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
