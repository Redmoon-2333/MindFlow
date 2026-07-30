import { useState, useEffect, useCallback, useRef } from "react";
import {
  getTrainingReadiness,
  getBaseline,
  createTrainingJob,
  getTrainingJob,
  cancelTrainingJob,
  getModelStatus,
  getErrorMessage,
  ApiError,
} from "../api";
import type {
  BaselineSummary,
  ModelStatusView,
  GateStatus,
  TrainingReadinessResponse,
  TrainingJobResponse,
} from "../api";
import "./model-center.css";

const TABS = ["数据准备", "个人基线", "模型训练", "模型状态"] as const;
type Tab = (typeof TABS)[number];

const NONTERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "pending", "preparing_data", "training",
]);
const POLL_INTERVAL_MS = 3000;

function badgeForGate(status: GateStatus): string {
  switch (status) {
    case "passed": return "badge-success";
    case "failed": return "badge-danger";
    case "not_evaluated": return "badge-info";
    case "not_implemented": return "badge-warning";
  }
}

function statusLabel(status: GateStatus): string {
  switch (status) {
    case "passed": return "通过";
    case "failed": return "未通过";
    case "not_evaluated": return "未评估";
    case "not_implemented": return "未实现";
  }
}

function jobStatusLabel(status: string): string {
  const map: Record<string, string> = {
    pending: "等待中",
    preparing_data: "准备数据中",
    training: "训练中",
    succeeded: "已完成",
    failed: "已失败",
    cancelled: "已取消",
  };
  return map[status] ?? status;
}

function isNonterminal(status: string): boolean {
  return NONTERMINAL_STATUSES.has(status);
}

export default function ModelCenter() {
  const [activeTab, setActiveTab] = useState<Tab>("数据准备");

  // Data states
  const [rd, setRd] = useState<TrainingReadinessResponse | null>(null);
  const [baseline, setBaselineLocal] = useState<BaselineSummary | null>(null);
  const [baselineEmpty, setBaselineEmpty] = useState(false);
  const [modelStatus, setModelStatusState] = useState<ModelStatusView | null>(null);
  const [activeJob, setActiveJob] = useState<TrainingJobResponse | null>(null);
  // Active job id tracked as state so effects react to it
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  // UI states
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [trainingLoading, setTrainingLoading] = useState(false);
  const [cancelLoading, setCancelLoading] = useState(false);

  // Refs for polling lifecycle (never passed as deps)
  const pollingTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);
  const activeJobIdRef = useRef<string | null>(null);
  const scheduleNextRef = useRef<((jobId: string) => void) | null>(null);

  // ── Guarded fetch helpers ──

  const fetchReadiness = useCallback(async () => {
    setLoading((p) => ({ ...p, readiness: true }));
    try {
      const data = await getTrainingReadiness();
      setRd(data);
      // Sync active job id from readiness if not already tracking
      if (
        data.current_training_job &&
        isNonterminal(data.current_training_job.status) &&
        !activeJobIdRef.current
      ) {
        const jid = data.current_training_job.job_id;
        setActiveJobId(jid);
        activeJobIdRef.current = jid;
      }
    } catch (e: unknown) {
      setError(getErrorMessage(e, "训练就绪评估加载失败"));
    } finally {
      setLoading((p) => ({ ...p, readiness: false }));
    }
  }, []);

  const fetchBaseline = useCallback(async () => {
    setLoading((p) => ({ ...p, baseline: true }));
    try {
      const data = await getBaseline();
      setBaselineLocal(data);
      setBaselineEmpty(false);
    } catch (e: unknown) {
      if (e instanceof ApiError && e.status === 404) {
        setBaselineLocal(null);
        setBaselineEmpty(true);
      } else {
        setError(getErrorMessage(e, "基线数据加载失败"));
      }
    } finally {
      setLoading((p) => ({ ...p, baseline: false }));
    }
  }, []);

  const fetchModelStatus = useCallback(async () => {
    setLoading((p) => ({ ...p, modelStatus: true }));
    try {
      const data = await getModelStatus();
      setModelStatusState(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "模型状态加载失败"));
    } finally {
      setLoading((p) => ({ ...p, modelStatus: false }));
    }
  }, []);

  // ── Poll loop (chained setTimeout — never overlapping) ──

  const pollOnce = useCallback(async (jobId: string) => {
    if (unmountedRef.current) return;
    try {
      const data = await getTrainingJob(jobId);
      if (unmountedRef.current) return;
      setActiveJob(data);
      if (!isNonterminal(data.status)) {
        // Terminal — clear state, refresh readiness + model status
        setActiveJobId(null);
        activeJobIdRef.current = null;
        setActiveJob(data); // preserve final state for display
        fetchReadiness();
        fetchModelStatus();
        return; // stop loop
      }
      // Nonterminal — schedule next poll via ref
      if (scheduleNextRef.current) {
        scheduleNextRef.current(jobId);
      }
    } catch (e: unknown) {
      if (unmountedRef.current) return;
      if (e instanceof ApiError && e.status === 404) {
        setActiveJobId(null);
        activeJobIdRef.current = null;
        setActiveJob(null);
        fetchReadiness();
      } else {
        setError(getErrorMessage(e, "训练状态查询失败"));
        // retry after interval
        if (scheduleNextRef.current) {
          scheduleNextRef.current(jobId);
        }
      }
    }
  }, [fetchReadiness, fetchModelStatus]);

  // scheduleNext needs to reference pollOnce without creating a cycle;
  // use a ref assigned after pollOnce is created.
  function scheduleNext(jobId: string) {
    if (unmountedRef.current) return;
    pollingTimer.current = setTimeout(() => {
      pollOnce(jobId);
    }, POLL_INTERVAL_MS);
  }
  scheduleNextRef.current = scheduleNext;

  const startPolling = useCallback((jobId: string) => {
    // Clear any existing timer
    if (pollingTimer.current) {
      clearTimeout(pollingTimer.current);
      pollingTimer.current = null;
    }
    setActiveJobId(jobId);
    activeJobIdRef.current = jobId;
    pollOnce(jobId);
  }, [pollOnce]);

  const stopPolling = useCallback(() => {
    if (pollingTimer.current) {
      clearTimeout(pollingTimer.current);
      pollingTimer.current = null;
    }
    setActiveJobId(null);
    activeJobIdRef.current = null;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    unmountedRef.current = false;
    return () => {
      unmountedRef.current = true;
      if (pollingTimer.current) {
        clearTimeout(pollingTimer.current);
        pollingTimer.current = null;
      }
    };
  }, []);

  // Drive polling from activeJobId state changes
  useEffect(() => {
    if (activeJobId && !pollingTimer.current) {
      startPolling(activeJobId);
    }
    return () => {
      // Don't stop on re-render; only on unmount or explicit stop
    };
  }, [activeJobId, startPolling]);

  // ── Start training ──

  const handleStartTraining = async () => {
    setTrainingLoading(true);
    setError(null);
    try {
      const result = await createTrainingJob();
      const jobResponse: TrainingJobResponse = {
        job_id: result.job_id,
        status: result.status,
        source: "db",
        model_mode: "rule_engine_only",
        started_at: null,
        completed_at: null,
        activated: false,
        version_tag: null,
        feature_schema_version: null,
        quality_gate: null,
        evaluation: null,
        error: null,
      };
      setActiveJob(jobResponse);
      startPolling(result.job_id);
    } catch (e: unknown) {
      if (e instanceof ApiError) {
        if (e.status === 412) {
          setError("训练数据不足，无法启动训练任务");
        } else if (e.status === 409) {
          setError("已有活跃的训练任务，请等待完成");
        } else {
          setError(getErrorMessage(e, "启动训练失败"));
        }
      } else {
        setError(getErrorMessage(e, "启动训练失败"));
      }
    } finally {
      setTrainingLoading(false);
    }
  };

  // ── Cancel training ──

  const handleCancelTraining = async () => {
    const jobId = activeJobIdRef.current;
    if (!jobId) return;
    setCancelLoading(true);
    setError(null);
    try {
      const result = await cancelTrainingJob(jobId);
      setActiveJob(result);
      // Continue polling until terminal: the cancel response may return
      // cancelled immediately, but if backend returns nonterminal we keep polling
      if (!isNonterminal(result.status)) {
        stopPolling();
        fetchReadiness();
      } else {
        // backend returned nonterminal — polling already running
      }
    } catch (e: unknown) {
      if (e instanceof ApiError && e.status === 409) {
        setError("训练已经开始，无法安全取消");
      } else {
        setError(getErrorMessage(e, "取消失败"));
      }
    } finally {
      setCancelLoading(false);
    }
  };

  // ── Initial loads ──

  useEffect(() => { fetchReadiness(); }, [fetchReadiness]);
  useEffect(() => { fetchBaseline(); }, [fetchBaseline]);
  useEffect(() => { fetchModelStatus(); }, [fetchModelStatus]);

  // ── Derived state ──

  const canStartTraining = rd?.trainable === true && !activeJobId;
  const activeJobStatus = activeJob?.status ?? rd?.current_training_job?.status;
  const cancelAllowed = activeJobStatus === "pending" || activeJobStatus === "preparing_data";
  const currentJobId = activeJob?.job_id ?? rd?.current_training_job?.job_id ?? null;
  const polling = activeJobId !== null;

  const renderLoading = (key: string) => {
    if (loading[key]) return <div className="spinner" />;
    return null;
  };

  return (
    <div>
      <div className="mc-header">
        <h1>模型中心</h1>
        <p>V2 特征模型训练就绪评估与任务管理</p>
      </div>

      {/* Tabs */}
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={activeTab === t}
            className={`tab${activeTab === t ? " active" : ""}`}
            onClick={() => setActiveTab(t)}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Global error */}
      {error && (
        <div className="error-box mb16" role="alert">
          {error}
          <button type="button" className="btn btn-sm" style={{ marginLeft: 12 }} onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}

      {/* ──────────────── 数据准备 Tab ──────────────── */}
      {activeTab === "数据准备" && (
        <div>
          {renderLoading("readiness")}

          {rd && !loading.readiness && (
            <>
              {/* Event / window summary */}
              <div className="mc-kpi-row">
                <div className="stat-card">
                  <div className="label">原始事件</div>
                  <div className="value">{rd.raw_events.total_events.toLocaleString()}</div>
                  <div className="sub" style={{ color: "var(--color-text-tertiary)" }}>
                    {rd.raw_events.coverage_days} 天覆盖
                  </div>
                </div>
                <div className="stat-card">
                  <div className="label">V2 特征窗口</div>
                  <div className="value">{rd.v2_windows.total}</div>
                  <div className="sub" style={{ color: "var(--color-text-tertiary)" }}>
                    {rd.v2_windows.date_range_days} 天 · v{rd.v2_windows.schema_version}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="label">匹配合格窗口</div>
                  <div className="value">{rd.v2_windows.eligible_count}</div>
                  <div className="sub" style={{ color: "var(--color-text-tertiary)" }}>
                    专注 {rd.v2_windows.matched_focus_count} · 干扰 {rd.v2_windows.matched_distract_count}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="label">人工标注</div>
                  <div className="value">{rd.feedback_labels.total}</div>
                  <div className="sub" style={{ color: "var(--color-text-tertiary)" }}>
                    专注 {rd.feedback_labels.focus} · 干扰 {rd.feedback_labels.distract} · 混合 {rd.feedback_labels.mixed}
                  </div>
                </div>
              </div>

              {/* Trainability & Evaluability */}
              <div className="mc-section">
                <div className="mc-kpi-row" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))" }}>
                  <div className="stat-card">
                    <div className="label">可训练性</div>
                    <div className="value" style={{ fontSize: 22 }}>
                      {rd.trainable ? "可训练" : "数据不足"}
                    </div>
                    <div className={`sub ${rd.trainable ? "good" : "bad"}`}>
                      {rd.trainable_window_count} 合格窗口 · {rd.trainable_class_count} 种标签
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="label">可评估性</div>
                    <div className="value" style={{ fontSize: 22 }}>
                      {rd.evaluable ? "可评估" : "数据不足"}
                    </div>
                    <div className={`sub ${rd.evaluable ? "good" : "bad"}`}>
                      {rd.evaluable_explicit_count} 显式样本 · {rd.evaluable_date_count} 天（需 ≥3 天）
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="label">基线就绪</div>
                    <div className="value" style={{ fontSize: 22 }}>
                      {rd.baseline_ready ? "就绪" : "未就绪"}
                    </div>
                    <div className="sub" style={{ color: "var(--color-text-tertiary)" }}>
                      当前模式: {rd.current_mode === "ready" ? "就绪" : rd.current_mode === "shadow" ? "影子模式" : "仅规则引擎"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Quality Gates — display all 7 from backend */}
              <div className="card mc-section">
                <h3>质量门禁（7 项检查）</h3>
                <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 12 }}>
                  not_evaluated 指标在训练运行后出现。not_implemented 检查不代表通过。
                </p>
                <div className="mc-gates">
                  {rd.gates.map((gate) => (
                    <div className="mc-gate" key={gate.key}>
                      <div>
                        <div className="gate-label">{gate.label}</div>
                        <div className="gate-detail">
                          当前值: {gate.actual} · 阈值: {gate.threshold}
                        </div>
                      </div>
                      <span className={`badge ${badgeForGate(gate.status as GateStatus)}`}>
                        {statusLabel(gate.status as GateStatus)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Blockers */}
              {rd.blockers.length > 0 && (
                <div className="card mc-section">
                  <h3>阻塞项（{rd.blockers.length}）</h3>
                  <div className="mc-blockers">
                    {rd.blockers.map((b) => (
                      <div className="mc-blocker" key={b.code}>
                        <strong>[{b.code}]</strong> {b.message}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {!rd && !loading.readiness && (
            <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>
              暂无数据
            </p>
          )}
        </div>
      )}

      {/* ──────────────── 个人基线 Tab ──────────────── */}
      {activeTab === "个人基线" && (
        <div>
          {renderLoading("baseline")}

          {baseline && !loading.baseline && (
            <div className="card mc-section">
              <h3>个人行为基线</h3>
              <div className="flex gap16" style={{ fontSize: 13, flexWrap: "wrap" }}>
                <div>
                  <span style={{ color: "var(--color-text-tertiary)" }}>数据天数：</span>
                  {baseline.total_days}
                </div>
                <div>
                  <span style={{ color: "var(--color-text-tertiary)" }}>样本数：</span>
                  {baseline.total_samples}
                </div>
                <div>
                  <span style={{ color: "var(--color-text-tertiary)" }}>特征：</span>
                  {baseline.features?.length ?? 0} 维
                </div>
                <div>
                  <span style={{ color: "var(--color-text-tertiary)" }}>建立时间：</span>
                  {baseline.created_at ? new Date(baseline.created_at).toLocaleDateString("zh-CN") : "N/A"}
                </div>
              </div>
              <div className="mt16" style={{ fontSize: 12, color: "var(--color-text-tertiary)", borderTop: "1px solid var(--color-border)", paddingTop: 12 }}>
                基线是基于日常行为数据的在线统计指标，与 ML 批量训练不同。基线就绪状态反映是否有足够的历史数据建立对比基准。
              </div>
            </div>
          )}

          {baselineEmpty && !loading.baseline && (
            <div className="card mc-section">
              <h3>个人行为基线</h3>
              <div className="mc-baseline-empty">
                暂无基线数据。基线在收集足够的行为数据后自动建立，用于对比日常行为变化。
              </div>
              <div className="mt16" style={{ fontSize: 12, color: "var(--color-text-tertiary)", borderTop: "1px solid var(--color-border)", paddingTop: 12 }}>
                基线是基于日常行为数据的在线统计指标，与 ML 批量训练不同。基线就绪状态反映是否有足够的历史数据建立对比基准。
              </div>
            </div>
          )}

          {!baseline && !baselineEmpty && !loading.baseline && (
            <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>
              暂无数据
            </p>
          )}
        </div>
      )}

      {/* ──────────────── 模型训练 Tab ──────────────── */}
      {activeTab === "模型训练" && (
        <div>
          {/* Active job display */}
          {(activeJob || rd?.current_training_job) && (
            <div className="card mc-section">
              <h3>当前训练任务</h3>
              <div className="mc-job-status">
                {activeJobStatus && isNonterminal(activeJobStatus) && (
                  <div className="spinner" />
                )}
                <span className={`badge ${
                  activeJobStatus === "succeeded" ? "badge-success" :
                  activeJobStatus === "failed" ? "badge-danger" :
                  activeJobStatus === "cancelled" ? "badge-info" :
                  "badge-warning"
                }`}>
                  {activeJobStatus ? jobStatusLabel(activeJobStatus) : "未知"}
                </span>
                <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                  任务 ID: {currentJobId}
                </span>
                {activeJob?.started_at && (
                  <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                    开始: {new Date(activeJob.started_at).toLocaleString("zh-CN")}
                  </span>
                )}
                {activeJob?.completed_at && (
                  <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                    完成: {new Date(activeJob.completed_at).toLocaleString("zh-CN")}
                  </span>
                )}
              </div>

              {/* Cancel button: only when pending or preparing_data */}
              {cancelAllowed && (
                <button
                  type="button"
                  className="btn btn-danger btn-sm"
                  onClick={handleCancelTraining}
                  disabled={cancelLoading}
                  style={{ marginTop: 12 }}
                >
                  {cancelLoading ? "取消中..." : "取消任务"}
                </button>
              )}

              {/* Training started explanation */}
              {activeJobStatus === "training" && (
                <div className="mt8" style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                  训练已经开始，无法安全取消。请等待完成。
                </div>
              )}

              {/* Error display */}
              {activeJob?.error && (
                <div className="error-box mt8">
                  训练错误: {activeJob.error}
                </div>
              )}

              {/* Post-training info */}
              {activeJob?.activated && (
                <div className="mt8" style={{ fontSize: 13, color: "var(--color-success)" }}>
                  模型已激活（版本: {activeJob.version_tag ?? "N/A"}）
                </div>
              )}
              {activeJob?.status === "succeeded" && !activeJob?.activated && (
                <div className="mt8" style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>
                  训练以影子模式完成（未激活）。激活前可以进行评估。
                </div>
              )}
            </div>
          )}

          {/* Start training */}
          <div className="card mc-section">
            <h3>启动训练</h3>
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
              {rd?.trainable
                ? rd?.trainable_window_count != null
                  ? `数据充足（${rd.trainable_window_count} 合格窗口，${rd.trainable_class_count} 种标签）`
                  : "数据充足，可以启动训练"
                : "数据不足，无法启动训练"}
            </div>
            <div className="mc-actions">
              <button
                type="button"
                className="btn"
                onClick={handleStartTraining}
                disabled={!canStartTraining || trainingLoading}
              >
                {trainingLoading ? "启动中..." : "开始训练"}
              </button>
              {!rd?.trainable && (
                <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                  需要的合格窗口数 ≥10，标签种类 ≥2
                </span>
              )}
            </div>

            {/* Polling indicator */}
            {polling && (
              <div
                className="mt8"
                style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}
                role="status"
                aria-live="polite"
              >
                正在监控训练进度...
              </div>
            )}
          </div>

          {/* Training conditions — thresholds from backend */}
          {rd && (
            <div className="card mc-section">
              <h3>训练条件</h3>
              <div style={{ fontSize: 13, display: "flex", flexDirection: "column", gap: 8 }}>
                <div className="flex flex-between">
                  <span style={{ color: "var(--color-text-secondary)" }}>合格窗口数（需 ≥10）</span>
                  <span className={`badge ${rd.trainable_window_count >= 10 ? "badge-success" : "badge-danger"}`}>
                    {rd.trainable_window_count}
                  </span>
                </div>
                <div className="flex flex-between">
                  <span style={{ color: "var(--color-text-secondary)" }}>标签种类（需 ≥2）</span>
                  <span className={`badge ${rd.trainable_class_count >= 2 ? "badge-success" : "badge-danger"}`}>
                    {rd.trainable_class_count}
                  </span>
                </div>
                <div className="flex flex-between">
                  <span style={{ color: "var(--color-text-secondary)" }}>显式评估样本（需 ≥10）</span>
                  <span className={`badge ${rd.evaluable_explicit_count >= 10 ? "badge-success" : "badge-danger"}`}>
                    {rd.evaluable_explicit_count}
                  </span>
                </div>
                <div className="flex flex-between">
                  <span style={{ color: "var(--color-text-secondary)" }}>评估天数（需 ≥3）</span>
                  <span className={`badge ${rd.evaluable_date_count >= 3 ? "badge-success" : "badge-danger"}`}>
                    {rd.evaluable_date_count}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ──────────────── 模型状态 Tab ──────────────── */}
      {activeTab === "模型状态" && (
        <div>
          {renderLoading("modelStatus")}

          {modelStatus && !loading.modelStatus && (
            <>
              <div className="card mc-section">
                <h3>运行模式</h3>
                <div className="mc-model-mode">
                  <span className="mode-label">当前模式</span>
                  <span className={`badge ${
                    modelStatus.mode === "ready" ? "badge-success" :
                    modelStatus.mode === "shadow" ? "badge-warning" :
                    "badge-info"
                  }`}>
                    {modelStatus.mode === "ready" ? "就绪" :
                     modelStatus.mode === "shadow" ? "影子模式" :
                     "仅规则引擎"}
                  </span>
                </div>
                {modelStatus.message && (
                  <div className="mt8" style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
                    {modelStatus.message}
                  </div>
                )}
              </div>

              <div className="card mc-section">
                <h3>模型版本</h3>
                <div className="flex gap16" style={{ fontSize: 13, flexDirection: "column" }}>
                  <div className="flex flex-between">
                    <span style={{ color: "var(--color-text-secondary)" }}>激活版本</span>
                    <span>{modelStatus.version ?? "N/A"}</span>
                  </div>
                  {modelStatus.feature_schema_version != null && (
                    <div className="flex flex-between">
                      <span style={{ color: "var(--color-text-secondary)" }}>特征版本</span>
                      <span>v{modelStatus.feature_schema_version}</span>
                    </div>
                  )}
                  {modelStatus.model_name && (
                    <div className="flex flex-between">
                      <span style={{ color: "var(--color-text-secondary)" }}>模型名称</span>
                      <span>{modelStatus.model_name}</span>
                    </div>
                  )}
                  {modelStatus.last_updated && (
                    <div className="flex flex-between">
                      <span style={{ color: "var(--color-text-secondary)" }}>最后更新</span>
                      <span>{new Date(modelStatus.last_updated).toLocaleString("zh-CN")}</span>
                    </div>
                  )}
                </div>
              </div>

              {modelStatus.reasons && modelStatus.reasons.length > 0 && (
                <div className="card mc-section">
                  <h3>就绪原因</h3>
                  <ul style={{ fontSize: 13, paddingLeft: 20, color: "var(--color-text-secondary)" }}>
                    {modelStatus.reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {modelStatus.available_versions && modelStatus.available_versions.length > 0 && (
                <div className="card mc-section">
                  <h3>可用版本</h3>
                  <div className="flex gap8" style={{ flexWrap: "wrap" }}>
                    {modelStatus.available_versions.map((v) => (
                      <span className="badge badge-primary" key={v}>{v}</span>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {!modelStatus && !loading.modelStatus && (
            <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>
              暂无数据
            </p>
          )}
        </div>
      )}
    </div>
  );
}
