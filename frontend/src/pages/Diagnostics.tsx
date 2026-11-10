import WorkspacePage from "../components/WorkspacePage";
import EmptyState from "../components/EmptyState";
import { useState, useEffect, useCallback, useRef } from "react";
import { getAIRuns, getAIRunDetail, getFocusPrediction, getHealthLive, getHealthReady, getErrorMessage, getAnalysisGraph } from "../api";
import type { AIRunItem, AIRunDetail, AnalysisGraphTopology, FocusPredictionResponse, HealthLiveResponse, HealthReadyResponse } from "../api";
import { toFocusPredictionView } from "../prediction-state";
import MetricStrip from "../components/signal/MetricStrip";
import type { MetricItem } from "../components/signal/MetricStrip";
import { userTerm } from "../lib/user-terms";

function statusBadgeClass(status: string): string {
  const s = status.toLowerCase();
  if (s === "completed" || s === "success") return "badge-success";
  if (s === "failed" || s === "error") return "badge-danger";
  if (s === "running") return "badge-info";
  return "badge-warning";
}

function formatDuration(ms: number | null): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${ms} 毫秒`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)} 秒`;
  return `${(ms / 60000).toFixed(1)} 分钟`;
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return "—";
  return new Date(ts).toLocaleString("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

/** The four readiness probes, in the order the popover and this strip list them. */
const READY_CHECKS = [
  { key: "database", label: "数据库" },
  { key: "migration", label: "迁移" },
  { key: "checkpoint", label: "检查点" },
  { key: "run_store", label: "运行记录" },
] as const;

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
  // A probe that fails must say so. These three used to disappear into an
  // empty `.catch(() => {})`, which rendered "—" — indistinguishable from a
  // backend that was never asked.
  const [liveFailed, setLiveFailed] = useState(false);
  const [readyFailed, setReadyFailed] = useState(false);
  const [predictionFailed, setPredictionFailed] = useState(false);
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
      // The backend returns `count`; `total` never existed on this response,
      // which made the pager think there were always 0 rows.
      setTotalRuns(res.count);
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
      // The topology is optional detail inside a collapsed <details>; an
      // unavailable graph is a normal state and renders its own empty line.
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
    getFocusPrediction().then(setPrediction).catch(() => setPredictionFailed(true));
    getHealthLive().then(setLive).catch(() => setLiveFailed(true));
    // Ready answers 503 when not ready, so the request rejects on a degraded
    // backend — that is `readyFailed`, not a transport failure.
    getHealthReady().then(setReady).catch(() => setReadyFailed(true));
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

  /** Count of the four readiness probes that actually passed.
   *
   *  `/health/ready` is only trustworthy when it answered with the contract's
   *  full body; an older or partial payload must not crash the page (it once
   *  read `database.status` off `undefined`), so each probe is read defensively.
   */
  const passedChecks = ready
    ? [
        Boolean(ready.database && ready.database.status === "ok" && ready.database.integrity_ok),
        Boolean(ready.migration && ready.migration.applied),
        ready.checkpoint_store === "available",
        ready.run_store === "available",
      ]
    : [];
  const passedCount = passedChecks.filter(Boolean).length;
  const failedCheckLabels = READY_CHECKS.filter((_, index) => ready && !passedChecks[index]).map((check) => check.label);

  const metrics: MetricItem[] = [
    {
      key: "live",
      label: "服务存活",
      value: live ? "正常" : liveFailed ? "无响应" : "—",
      delta: live ? { text: `版本 ${live.version}`, tone: "none" } : undefined,
    },
    {
      key: "ready",
      label: "就绪检查",
      ...(ready
        ? { value: String(passedCount), unit: "/4 项" }
        : readyFailed
          ? { value: "未就绪" }
          : { value: "—" }),
      delta: ready
        ? passedCount === READY_CHECKS.length
          ? { text: "数据库、迁移、检查点、运行记录", tone: "none" }
          : { text: `未通过：${failedCheckLabels.join("、")}`, tone: "down" }
        : undefined,
    },
    {
      key: "prediction",
      label: "专注预测分数",
      // §8.22.6: `display` still carries the view's legacy "--" placeholder;
      // the site uses the single em-dash for an unknown number. Normalised
      // here so `prediction-state.ts` (and its 19 assertions) stays untouched.
      value: predictionView
        ? (predictionView.display === "--" ? "—" : predictionView.display)
        : (predictionFailed ? "不可用" : "—"),
      delta: predictionView
        ? {
            text: predictionView.reason
              ? `${predictionView.statusLabel} · ${predictionView.reason}`
              : `预测模式：${predictionView.mode}`,
            tone: "none",
          }
        : undefined,
    },
    {
      key: "mode",
      label: "预测模式",
      value: prediction
        ? ({ ready: "已就绪", shadow: "影子模式", rule_engine_only: "规则分析" }[prediction.mode] ?? prediction.mode)
        : "—",
      delta: { text: `运行记录 ${totalRuns} 条`, tone: "none" },
    },
  ];

  return (
    <WorkspacePage title="AI 诊断" description="分析没有结果时，在这里查看服务状态和运行记录。展开一条记录可查看失败原因。" className="diagnostics-page">

      {error && (
        <div className="error-box mb16">
          {error}
          <button type="button" className="btn btn-secondary btn-sm diagnostics-error-retry" onClick={fetchRuns}>
            重试
          </button>
        </div>
      )}

      <MetricStrip items={metrics} />

      {/* AI Runs Table */}
      <div className="card mt16 mb24">
        <div className="flex-between mb16">
          <h3>AI 工作流运行记录</h3>
          <button type="button" className="btn btn-secondary btn-sm" disabled={loading} onClick={fetchRuns}>刷新记录</button>
        </div>
        {loading && <div className="spinner" role="status" aria-label="加载运行记录" />}
        {!loading && (runs.length === 0 ? (
          <EmptyState
            title="暂无运行记录"
            description="分析运行后会在这里留下一条记录；触发一次专家面板分析即可生成第一条。"
          />
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
                    <td className="diagnostics-mono">
                      <button type="button" className="btn btn-ghost btn-sm" aria-expanded={openRunId === run.run_id} onClick={() => handleRowClick(run.run_id)}>{run.run_id.slice(0, 12)}…</button>
                    </td>
                    <td>
                      <span className={`badge ${statusBadgeClass(run.status)}`}>
                        {userTerm(run.status)}
                      </span>
                    </td>
                    <td>{run.source}</td>
                    <td className="diagnostics-meta">{formatTimestamp(run.started_at)}</td>
                    <td className="diagnostics-meta">{formatTimestamp(run.completed_at)}</td>
                    <td className="diagnostics-mono">
                      {formatDuration(run.duration_ms)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
        <div className="workspace-pagination"><span>共 {totalRuns} 条 · 第 {page + 1} 页</span><div className="workspace-actions"><button type="button" className="btn btn-secondary btn-sm" disabled={loading || page === 0} onClick={() => setPage(p => p - 1)}>上一页</button><button type="button" className="btn btn-secondary btn-sm" disabled={loading || (page + 1) * pageSize >= totalRuns} onClick={() => setPage(p => p + 1)}>下一页</button></div></div>
      </div>

      {/* Analysis graph topology (architecture plan G/1.2) — shows how the
          analysis pipeline is wired. */}
      <details className="workspace-details mb24">
        <summary>分析图拓扑 · 高级信息</summary>
        {graphTopology?.available ? (
          <div className="diagnostics-detail">
            <div className="diagnostics-detail-row">
              节点（{graphTopology.nodes.length}）：{graphTopology.nodes.join(" → ") || "—"}
            </div>
            <div>
              边（{graphTopology.edges.length}）：
              {graphTopology.edges.length > 0 ? (
                <div className="diagnostics-edges">
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
          <div className="diagnostics-detail diagnostics-detail--dim">
            {graphLoading ? "加载中…" : "分析图当前不可用（未启用 LLM 分析工作流）"}
          </div>
        )}
      </details>

      {/* Expanded Detail */}
      {openRunId && (
        <div className="card mb24">
          <div className="flex-between mb16">
            <h3 className="diagnostics-mono">
              运行详情: {openRunId}
            </h3>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
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
                  <pre className="diagnostics-error-pre">
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
                          <td className="diagnostics-mono">
                            {evt.name ?? `node-${idx}`}
                          </td>
                          <td>{evt.type ?? "—"}</td>
                          <td>
                            <span className={`badge ${statusBadgeClass(evt.status ?? "")}`}>
                              {evt.status ? userTerm(evt.status) : "—"}
                            </span>
                          </td>
                          <td className="diagnostics-meta">
                            {formatTimestamp(evt.started_at ?? null)}
                          </td>
                          <td className="diagnostics-meta">
                            {formatTimestamp(evt.completed_at ?? null)}
                          </td>
                          <td className="diagnostics-mono">
                            {formatDuration(evt.duration_ms ?? null)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="diagnostics-empty">无节点事件记录</p>
              )}
            </div>
          )}
        </div>
      )}
    </WorkspacePage>
  );
}
