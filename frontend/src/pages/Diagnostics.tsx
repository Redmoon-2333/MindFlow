import WorkspacePage from "../components/WorkspacePage";
import { useState, useEffect, useCallback, useRef } from "react";
import { getAIRuns, getAIRunDetail, getFocusPrediction, getHealthLive, getHealthReady, getErrorMessage, getAnalysisGraph } from "../api";
import type { AIRunItem, AIRunDetail, AnalysisGraphTopology, FocusPredictionResponse, HealthLiveResponse, HealthReadyResponse } from "../api";
import { toFocusPredictionView } from "../prediction-state";

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

  const [page, setPage] = useState(0);
  const requestSeq = useRef(0);
  const detailSeq = useRef(0);
  const pageSize = 20;
  const [runs, setRuns] = useState<AIRunItem[]>([]);
  const [totalRuns, setTotalRuns] = useState(0);
  const [selectedRun, setSelectedRun] = useState<AIRunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [prediction, setPrediction] = useState<FocusPredictionResponse | null>(null);
  const [live, setLive] = useState<HealthLiveResponse | null>(null);
  const [ready, setReady] = useState<HealthReadyResponse | null>(null);
  const [openRunId, setOpenRunId] = useState<string | null>(null);
  const [graphTopology, setGraphTopology] = useState<AnalysisGraphTopology | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);

  const fetchRuns = useCallback(async () => {
    const seq = ++requestSeq.current;
    setLoading(true);
    setError(null);
    try {
      const res = await getAIRuns(pageSize, page * pageSize);
      if (seq !== requestSeq.current) return;
      setRuns(res.items);
      setTotalRuns(res.total);
    } catch (e: unknown) {
      if (seq !== requestSeq.current) return;
      setRuns([]);
      setError(getErrorMessage(e, "加载 AI 运行记录失败"));
    } finally {
      if (seq === requestSeq.current) setLoading(false);
    }
  }, [page]);

  const fetchGraphTopology = useCallback(async () => {
    setGraphLoading(true);
    try {
      const data = await getAnalysisGraph();
      setGraphTopology(data);
    } catch {
      setGraphTopology(null);
    } finally {
      setGraphLoading(false);
    }
  }, []);

  const fetchDetail = useCallback(async (runId: string) => {
    const seq = ++detailSeq.current;
    setSelectedRun(null);
    setDetailLoading(true);
    setDetailError(null);
    try {
      const detail = await getAIRunDetail(runId);
      if (seq !== detailSeq.current) return;
      setSelectedRun(detail);
    } catch (e: unknown) {
      if (seq !== detailSeq.current) return;
      setDetailError(getErrorMessage(e, "加载运行详情失败"));
      setSelectedRun(null);
    } finally {
      if (seq === detailSeq.current) setDetailLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRuns();
    setOpenRunId(null);
    setSelectedRun(null);
    detailSeq.current++;
  }, [fetchRuns]);

  useEffect(() => {
    fetchGraphTopology();
    getFocusPrediction().then(setPrediction).catch(() => {});
    getHealthLive().then(setLive).catch(() => {});
    getHealthReady().then(setReady).catch(() => {});
  }, [fetchGraphTopology]);

  const predictionView = prediction ? toFocusPredictionView(prediction) : null;

  const handleRowClick = (runId: string) => {
    if (openRunId === runId) {
      detailSeq.current++;
      setOpenRunId(null);
      setSelectedRun(null);
    } else {
      setOpenRunId(runId);
      fetchDetail(runId);
    }
  };

  return (
    <WorkspacePage title="AI 诊断" description="分析没有结果时，在这里查看服务状态和运行记录。展开一条记录可查看失败原因。" className="diagnostics-page">


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
            {predictionView ? predictionView.display : "--"}
          </div>
          <div className="sub">
            {predictionView
              ? predictionView.reason
                ? `${predictionView.statusLabel} · ${predictionView.reason}`
                : `预测模式: ${predictionView.mode}`
              : ""}
          </div>
        </div>
        <div className="stat-card">
          <div className="label">预测模式</div>
          <div className="value" style={{ fontSize: 22 }}>
            {prediction ? ({ ready: "已就绪", shadow: "影子模式", rule_engine_only: "规则分析" }[prediction.mode] ?? prediction.mode) : "--"}
          </div>
          <div className="sub">运行记录: {totalRuns} 条</div>
        </div>
      </div>

      {/* AI Runs Table */}
      <div className="card mb24">
        <div className="flex-between mb16"><h3 style={{margin: 0}}>AI 工作流运行记录</h3><button type="button" className="btn btn-ghost btn-sm" disabled={loading} onClick={fetchRuns}>刷新记录</button></div>
        {loading && <div className="spinner" role="status" aria-label="加载运行记录" />}
        {!loading && (runs.length === 0 ? (
          <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
            暂无运行记录
          </div>
        ) : (
          <div className="workspace-table">
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
                  >
                    <td style={{ fontFamily: "monospace", fontSize: 12 }}>
                      <button type="button" className="btn btn-ghost btn-sm" aria-expanded={openRunId === run.run_id} onClick={() => handleRowClick(run.run_id)}>{run.run_id.slice(0, 12)}…</button>
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
        ))}
        <div className="workspace-pagination"><span>共 {totalRuns} 条 · 第 {page + 1} 页</span><div className="workspace-actions"><button type="button" className="btn btn-ghost btn-sm" disabled={loading || page === 0} onClick={() => setPage(p => p - 1)}>上一页</button><button type="button" className="btn btn-ghost btn-sm" disabled={loading || (page + 1) * pageSize >= totalRuns} onClick={() => setPage(p => p + 1)}>下一页</button></div></div>
      </div>

      {/* Analysis graph topology (architecture plan G/1.2) — shows how the
          analysis pipeline is wired. */}
      <details className="workspace-details mb24">
        <summary>分析图拓扑 · 高级信息</summary>
        {graphTopology?.available ? (
          <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
            <div style={{ marginBottom: 8 }}>
              节点（{graphTopology.nodes.length}）：{graphTopology.nodes.join(" → ") || "—"}
            </div>
            <div>
              边（{graphTopology.edges.length}）：
              {graphTopology.edges.length > 0 ? (
                <div style={{ fontFamily: "monospace", fontSize: 12, marginTop: 4, lineHeight: 1.8 }}>
                  {graphTopology.edges.map((e, i) => (
                    <div key={i}>
                      {e.from} → {e.to}
                    </div>
                  ))}
                </div>
              ) : (
                "—"
              )}
            </div>
          </div>
        ) : (
          <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
            {graphLoading ? "加载中..." : "分析图当前不可用（未启用 LLM 分析工作流）"}
          </div>
        )}
      </details>

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
                <div className="workspace-table">
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
    </WorkspacePage>
  );
}
