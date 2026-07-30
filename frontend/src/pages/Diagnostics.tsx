import { useState, useEffect, useCallback } from "react";
import { getAIRuns, getAIRunDetail, getFocusPrediction, getHealthLive, getHealthReady, getErrorMessage } from "../api";
import type { AIRunItem, AIRunDetail, FocusPredictionResponse, HealthLiveResponse, HealthReadyResponse } from "../api";

function statusBadgeClass(status: string): string {
  const s = status.toLowerCase();
  if (s === "completed" || s === "success") return "badge-success";
  if (s === "failed" || s === "error") return "badge-danger";
  if (s === "running") return "badge-info";
  return "badge-warning";
}

function formatDuration(ms: number | null): string {
  if (ms === null || ms === undefined) return "--";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return "--";
  return new Date(ts).toLocaleString("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

export default function Diagnostics() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [runs, setRuns] = useState<AIRunItem[]>([]);
  const [totalRuns, setTotalRuns] = useState(0);
  const [selectedRun, setSelectedRun] = useState<AIRunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<FocusPredictionResponse | null>(null);
  const [live, setLive] = useState<HealthLiveResponse | null>(null);
  const [ready, setReady] = useState<HealthReadyResponse | null>(null);
  const [openRunId, setOpenRunId] = useState<string | null>(null);

  const fetchRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getAIRuns(50, 0);
      setRuns(res.items);
      setTotalRuns(res.total);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "加载 AI 运行记录失败"));
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchDetail = useCallback(async (runId: string) => {
    setDetailLoading(true);
    setDetailError(null);
    try {
      const detail = await getAIRunDetail(runId);
      setSelectedRun(detail);
    } catch (e: unknown) {
      setDetailError(getErrorMessage(e, "加载运行详情失败"));
      setSelectedRun(null);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
    getFocusPrediction().then(setPrediction).catch(() => {});
    getHealthLive().then(setLive).catch(() => {});
    getHealthReady().then(setReady).catch(() => {});
  }, [fetchRuns]);

  const handleRowClick = (runId: string) => {
    if (openRunId === runId) {
      setOpenRunId(null);
      setSelectedRun(null);
    } else {
      setOpenRunId(runId);
      fetchDetail(runId);
    }
  };

  if (loading) {
    return (
      <div>
        <div className="header">
          <h1>AI 诊断</h1>
          <p>工作流运行记录与系统健康检查</p>
        </div>
        <div className="spinner" />
      </div>
    );
  }

  return (
    <div>
      <div className="header">
        <h1>AI 诊断</h1>
        <p>工作流运行记录与系统健康检查</p>
      </div>

      {error && (
        <div className="error-box mb16">
          {error}
          <button type="button" className="btn btn-sm mt8" onClick={fetchRuns} style={{ marginLeft: 12 }}>
            重试
          </button>
        </div>
      )}

      {/* Health & Prediction Quick Cards */}
      <div className="kpi-row">
        <div className="stat-card">
          <div className="label">健康状态</div>
          <div className="value" style={{ fontSize: 22 }}>
            {live?.status === "ok" ? "正常" : ready?.status === "ok" ? "就绪" : "--"}
          </div>
          <div className="sub good">live: {live?.status ?? "--"}</div>
        </div>
        <div className="stat-card">
          <div className="label">就绪探测</div>
          <div className="value" style={{ fontSize: 22 }}>
            {ready?.status === "ok" ? "正常" : "--"}
          </div>
          <div className="sub">
            {ready?.checks ? `${Object.keys(ready.checks).length} 项检查` : ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">专注预测分数</div>
          <div className="value">
            {prediction != null ? (prediction.prediction * 100).toFixed(1) : "--"}
          </div>
          <div className="sub">
            {prediction?.source ?? ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">模型版本</div>
          <div className="value" style={{ fontSize: 22 }}>
            {prediction?.model_version ?? "--"}
          </div>
          <div className="sub">运行记录: {totalRuns} 条</div>
        </div>
      </div>

      {/* AI Runs Table */}
      <div className="card mb24">
        <h3>AI 工作流运行记录</h3>
        {runs.length === 0 ? (
          <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
            暂无运行记录
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>状态</th>
                  <th>来源</th>
                  <th>启动时间</th>
                  <th>完成时间</th>
                  <th>耗时</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr
                    key={run.run_id}
                    onClick={() => handleRowClick(run.run_id)}
                    style={{ cursor: "pointer" }}
                  >
                    <td style={{ fontFamily: "monospace", fontSize: 12 }}>
                      {run.run_id.slice(0, 12)}…
                    </td>
                    <td>
                      <span className={`badge ${statusBadgeClass(run.status)}`}>
                        {run.status}
                      </span>
                    </td>
                    <td>{run.source}</td>
                    <td style={{ fontSize: 12 }}>{formatTimestamp(run.started_at)}</td>
                    <td style={{ fontSize: 12 }}>{formatTimestamp(run.completed_at)}</td>
                    <td style={{ fontFamily: "monospace", fontSize: 12 }}>
                      {formatDuration(run.duration_ms)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Expanded Detail */}
      {openRunId && (
        <div className="card mb24">
          <div className="flex-between mb16">
            <h3 style={{ marginBottom: 0, fontFamily: "monospace", fontSize: 14 }}>
              运行详情: {openRunId}
            </h3>
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              onClick={() => { setOpenRunId(null); setSelectedRun(null); }}
            >
              关闭
            </button>
          </div>

          {detailLoading && <div className="spinner" />}

          {detailError && (
            <div className="error-box">{detailError}</div>
          )}

          {selectedRun && !detailLoading && (
            <div>
              {selectedRun.error && (
                <div className="error-box mb16">
                  <strong>错误：</strong>
                  <pre style={{ margin: "8px 0 0", fontSize: 12, whiteSpace: "pre-wrap" }}>
                    {selectedRun.error}
                  </pre>
                </div>
              )}

              {selectedRun.node_events.length > 0 ? (
                <div style={{ overflowX: "auto" }}>
                  <table>
                    <thead>
                      <tr>
                        <th>节点名称</th>
                        <th>类型</th>
                        <th>状态</th>
                        <th>开始</th>
                        <th>完成</th>
                        <th>耗时</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedRun.node_events.map((evt, idx) => (
                        <tr key={`${evt.name ?? "node"}-${evt.type ?? "event"}-${evt.started_at ?? ""}`}>
                          <td style={{ fontFamily: "monospace", fontSize: 12 }}>
                            {evt.name ?? `node-${idx}`}
                          </td>
                          <td>{evt.type ?? "--"}</td>
                          <td>
                            <span className={`badge ${statusBadgeClass(evt.status ?? "")}`}>
                              {evt.status ?? "--"}
                            </span>
                          </td>
                          <td style={{ fontSize: 12 }}>
                            {formatTimestamp(evt.started_at ?? null)}
                          </td>
                          <td style={{ fontSize: 12 }}>
                            {formatTimestamp(evt.completed_at ?? null)}
                          </td>
                          <td style={{ fontFamily: "monospace", fontSize: 12 }}>
                            {formatDuration(evt.duration_ms ?? null)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
                  无节点事件记录
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
