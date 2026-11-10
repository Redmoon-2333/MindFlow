import { userTerm } from "../lib/user-terms";
import WorkspacePage from "../components/WorkspacePage";
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
  ModelStatus,
  GateStatus,
  TrainingReadinessResponse,
  TrainingJobResponse,
} from "../api";
import { EMPTY_BASELINE_VIEW, reduceBaselineState } from "../baseline-state";
import type { BaselineViewState } from "../baseline-state";
import MetricStrip from "../components/signal/MetricStrip";
import type { MetricItem } from "../components/signal/MetricStrip";
import Segmented from "../components/signal/Segmented";
import HelpTip from "../components/signal/HelpTip";
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
    interrupted: "已中断",
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
  // Baseline view driven through reduceBaselineState so populated→404 empty
  // and 404→populated transitions never leave stale state on screen.
  const [baselineView, setBaselineView] = useState<BaselineViewState>(EMPTY_BASELINE_VIEW);
  const [modelStatus, setModelStatusState] = useState<ModelStatus | null>(null);
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
      const state = await getBaseline();
      if (!state.ok) {
        // Malformed wire payload — never render misleading values.
        setError(getErrorMessage(new ApiError("基线数据格式无效", 500), "基线数据加载失败"));
        return;
      }
      setBaselineView((prev) => reduceBaselineState(prev, { kind: "populated", summary: state }));
    } catch (e: unknown) {
      if (e instanceof ApiError && e.status === 404) {
        setBaselineView((prev) => reduceBaselineState(prev, { kind: "empty" }));
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
      // Optimistic view from the accepted-job response. Defaults for the
      // lifecycle fields match backend TrainingJobResponse defaults and are
      // immediately superseded by the poll loop (getTrainingJob) — the UI
      // never presents fabricated training results as real.
      setActiveJob({
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
        allow_activation: true,
      });
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
  const activeJobStatus: string | undefined = activeJob?.status ?? rd?.current_training_job?.status;
  const cancelAllowed = activeJobStatus === "pending" || activeJobStatus === "preparing_data";
  const currentJobId = activeJob?.job_id ?? rd?.current_training_job?.job_id ?? null;
  const polling = activeJobId !== null;

  // §8.12: the four data-volume facts, with the coverage detail as each
  // metric's delta line so it stays visible without a second row of boxes.
  const readinessMetrics: MetricItem[] = rd
    ? [
        {
          key: "events",
          label: "原始事件",
          value: rd.raw_events.total_events.toLocaleString(),
          delta: { text: `${rd.raw_events.coverage_days} 天覆盖`, tone: "none" },
        },
        {
          key: "windows",
          label: "行为特征窗口",
          value: String(rd.v2_windows.total),
          unit: "个",
          delta: { text: `${rd.v2_windows.date_range_days} 天 · v${rd.v2_windows.schema_version}`, tone: "none" },
        },
        {
          key: "eligible",
          label: "匹配合格窗口",
          value: String(rd.v2_windows.eligible_count),
          unit: "个",
          delta: {
            text: `专注 ${rd.v2_windows.matched_focus_count} · 干扰 ${rd.v2_windows.matched_distract_count}`,
            tone: "none",
          },
        },
        {
          key: "labels",
          label: "人工标注",
          value: String(rd.feedback_labels.total),
          unit: "条",
          delta: {
            text: `专注 ${rd.feedback_labels.focus} · 干扰 ${rd.feedback_labels.distract} · 混合 ${rd.feedback_labels.mixed}`,
            tone: "none",
          },
        },
      ]
    : [];

  const renderLoading = (key: string) => {
    if (loading[key]) return <div className="spinner" />;
    return null;
  };

  return (
    <WorkspacePage title="模型中心" description="先在专注分析中校准记录，再查看数据是否足够训练属于自己的行为模型。" className="model-page">
      {modelStatus?.demo_only && (
        <div className="notice model-demo-notice mb24" role="status">
          <span className="notice-text">合成数据演示模型 · 影子模式 · 未通过个人模型质量门</span>
        </div>
      )}

      <div className="page-toolbar">
        <Segmented
          ariaLabel="模型中心分区"
          options={TABS.map((tab) => ({ value: tab, label: tab }))}
          value={activeTab}
          onChange={(value) => setActiveTab(value as Tab)}
        />
      </div>

      {/* Global error */}
      {error && (
        <div className="error-box mb16" role="alert">
          {error}
          <button type="button" className="btn btn-secondary btn-sm mc-error-close" onClick={() => setError(null)}>
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
              <div className="mc-section">
                <MetricStrip items={readinessMetrics} />
              </div>

              {/* Trainability & Evaluability */}
              <div className="card mc-section">
                <h3>就绪情况</h3>
                <div className="mc-readiness-list">
                  <div className="kv-row kv-row--text">
                    <span>
                      可训练性
                      <span className="mc-readiness-note">
                        {rd.trainable_window_count} 合格窗口 · {rd.trainable_class_count} 种标签
                      </span>
                    </span>
                    <span className={`badge ${rd.trainable ? "badge-success" : "badge-warning"}`}>
                      {rd.trainable ? "可训练" : "数据不足"}
                    </span>
                  </div>
                  <div className="kv-row kv-row--text">
                    <span>
                      可评估性
                      <span className="mc-readiness-note">
                        {rd.evaluable_explicit_count} 显式样本 · {rd.evaluable_date_count} 天（需 ≥3 天）
                      </span>
                    </span>
                    <span className={`badge ${rd.evaluable ? "badge-success" : "badge-warning"}`}>
                      {rd.evaluable ? "可评估" : "数据不足"}
                    </span>
                  </div>
                  <div className="kv-row kv-row--text">
                    <span>
                      基线就绪
                      <span className="mc-readiness-note">
                        当前模式: {rd.current_mode === "ready" ? "就绪" : rd.current_mode === "shadow" ? "影子模式" : "仅规则引擎"}
                      </span>
                    </span>
                    <span className={`badge ${rd.baseline_ready ? "badge-success" : "badge-warning"}`}>
                      {rd.baseline_ready ? "就绪" : "未就绪"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Quality Gates — display all 7 from backend */}
              <div className="card mc-section">
                <h3>
                  质量门禁（7 项检查）
                  <HelpTip term="质量检查" what="新模型必须通过的 7 项检查，确保它比简单规则更准，才会被启用。" />
                </h3>
                <p className="card-desc">
                  未评估：训练之后才会有结果；未通过：数据或效果还不够，模型不会被启用。
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
                        <strong>{userTerm(b.code)}</strong> {b.message}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {!rd && !loading.readiness && (
            <p className="mc-empty">暂无数据</p>
          )}
        </div>
      )}

      {/* ──────────────── 个人基线 Tab ──────────────── */}
      {activeTab === "个人基线" && (
        <div>
          {renderLoading("baseline")}

          {baselineView.summary && !loading.baseline && (
            <div className="card mc-section">
              <h3>
                个人行为基线
                <HelpTip term="个人基线" what="你平时习惯的平均水平，用来判断某一天是否反常，会随着使用自动更新。" />
              </h3>
              <div>
                <div className="kv-row">
                  <span>数据天数</span>
                  <span>{baselineView.summary.total_days}</span>
                </div>
                <div className="kv-row">
                  <span>样本数</span>
                  <span>{baselineView.summary.total_samples}</span>
                </div>
                <div className="kv-row">
                  <span>特征</span>
                  <span>{baselineView.summary.features?.length ?? 0} 维</span>
                </div>
                <div className="kv-row kv-row--text">
                  <span>建立时间</span>
                  <span>{baselineView.summary.created_at ? new Date(baselineView.summary.created_at).toLocaleDateString("zh-CN") : "—"}</span>
                </div>
              </div>
              <div className="mc-baseline-note">
                基线是基于日常行为数据的在线统计指标，与 ML 批量训练不同。基线就绪状态反映是否有足够的历史数据建立对比基准。
              </div>
            </div>
          )}

          {baselineView.empty && !loading.baseline && (
            <div className="card mc-section">
              <h3>
                个人行为基线
                <HelpTip term="个人基线" what="你平时习惯的平均水平，用来判断某一天是否反常，会随着使用自动更新。" />
              </h3>
              <div className="mc-baseline-empty">
                暂无基线数据。基线在收集足够的行为数据后自动建立，用于对比日常行为变化。
              </div>
              <div className="mc-baseline-note">
                基线是基于日常行为数据的在线统计指标，与 ML 批量训练不同。基线就绪状态反映是否有足够的历史数据建立对比基准。
              </div>
            </div>
          )}

          {!baselineView.summary && !baselineView.empty && !loading.baseline && (
            <p className="mc-empty">暂无数据</p>
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
                <span className="mc-job-meta">
                  任务 ID: {currentJobId}
                </span>
                {activeJob?.started_at && (
                  <span className="mc-job-meta">
                    开始: {new Date(activeJob.started_at).toLocaleString("zh-CN")}
                  </span>
                )}
                {activeJob?.completed_at && (
                  <span className="mc-job-meta">
                    完成: {new Date(activeJob.completed_at).toLocaleString("zh-CN")}
                  </span>
                )}
              </div>

              {/* Cancel button: only when pending or preparing_data */}
              {cancelAllowed && (
                <button
                  type="button"
                  className="btn btn-danger btn-sm mc-cancel"
                  onClick={handleCancelTraining}
                  disabled={cancelLoading}
                >
                  {cancelLoading ? "取消中…" : "取消任务"}
                </button>
              )}

              {/* Training started explanation */}
              {activeJobStatus === "training" && (
                <p className="mc-hint mt8">
                  训练已经开始，无法安全取消。请等待完成。
                </p>
              )}

              {/* Error display */}
              {activeJobStatus === "interrupted" && (
                <div className="error-box mt8">
                  任务因服务中断而结束，可重新启动训练。
                </div>
              )}
              {activeJob?.error && (
                <div className="error-box mt8">
                  训练错误: {activeJob.error}
                </div>
              )}

              {/* Post-training info */}
              {activeJob?.activated && (
                <p className="mc-hint mc-hint--ok mt8">
                  模型已激活（版本: {activeJob.version_tag ?? "—"}）
                </p>
              )}
              {activeJob?.status === "succeeded" && !activeJob?.activated && (
                <p className="mc-hint mt8">
                  训练以影子模式完成（未激活）。激活前可以进行评估。
                </p>
              )}
            </div>
          )}

          {/* Start training */}
          <div className="card mc-section">
            <h3>启动训练</h3>
            <p className="mc-hint">
              {rd?.trainable
                ? rd?.trainable_window_count != null
                  ? `数据充足（${rd.trainable_window_count} 合格窗口，${rd.trainable_class_count} 种标签）`
                  : "数据充足，可以启动训练"
                : "数据不足，无法启动训练"}
            </p>
            <div className="mc-actions">
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={handleStartTraining}
                disabled={!canStartTraining || trainingLoading}
              >
                {trainingLoading ? "启动中…" : "开始训练"}
              </button>
              <HelpTip
                term="训练"
                what="用你给过的反馈，在这台电脑上训练一个更懂你的个人模型。"
                effect="训练只在本机进行，会占用一些电脑性能；如果新模型通过全部检查，会自动替换当前模型，之后的判断和提醒都会用它。"
              />
              {!rd?.trainable && (
                <span className="mc-hint">
                  需要的合格窗口数 ≥10，标签种类 ≥2
                </span>
              )}
            </div>

            {/* Polling indicator */}
            {polling && (
              <div
                className="mc-hint mt8"
                role="status"
                aria-live="polite"
              >
                正在监控训练进度…
              </div>
            )}
          </div>

          {/* Training conditions — thresholds from backend */}
          {rd && (
            <div className="card mc-section">
              <h3>训练条件</h3>
              <div className="mc-conditions">
                <div className="kv-row kv-row--text">
                  <span>合格窗口数（需 ≥10）</span>
                  <span className={`badge ${rd.trainable_window_count >= 10 ? "badge-success" : "badge-warning"}`}>
                    {rd.trainable_window_count}
                  </span>
                </div>
                <div className="kv-row kv-row--text">
                  <span>标签种类（需 ≥2）</span>
                  <span className={`badge ${rd.trainable_class_count >= 2 ? "badge-success" : "badge-warning"}`}>
                    {rd.trainable_class_count}
                  </span>
                </div>
                <div className="kv-row kv-row--text">
                  <span>显式评估样本（需 ≥10）</span>
                  <span className={`badge ${rd.evaluable_explicit_count >= 10 ? "badge-success" : "badge-warning"}`}>
                    {rd.evaluable_explicit_count}
                  </span>
                </div>
                <div className="kv-row kv-row--text">
                  <span>评估天数（需 ≥3）</span>
                  <span className={`badge ${rd.evaluable_date_count >= 3 ? "badge-success" : "badge-warning"}`}>
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
                <h3>
                  运行模式
                  <HelpTip
                    term="运行模式"
                    what="基础模型：随软件提供初始预测，等待个人数据校准。就绪：个人模型已通过验证。影子模式：候选模型供观察。仅规则引擎：使用固定规则。"
                  />
                </h3>
                <div className="mc-model-mode">
                  <span className="mode-label">当前模式</span>
                  <span className={`badge ${
                    modelStatus.mode === "ready" ? "badge-success" :
                    modelStatus.mode === "shadow" ? "badge-warning" :
                    "badge-info"
                  }`}>
                    {modelStatus.mode === "ready" ? "就绪" :
                     modelStatus.mode === "base" ? "基础模型" :
                      modelStatus.mode === "shadow" ? "影子模式" :
                     "仅规则引擎"}
                  </span>
                </div>
                <div className="mc-model-mode">
                  <span className="mode-label">部署档位</span>
                  <span className={modelStatus.deployment_tier === "full_ready" ? "badge badge-success" :
                    modelStatus.deployment_tier === "low_confidence" ? "badge badge-warning" :
                    "badge badge-info"}>
                    {modelStatus.deployment_tier === "base" ? "内置基础模型（待个人验证）" :
                      modelStatus.deployment_tier === "full_ready" ? "已就绪（全量）" :
                     modelStatus.deployment_tier === "low_confidence" ? "早期模式（低置信度）" :
                     modelStatus.deployment_tier ? "影子模式" :
                     "—"}
                  </span>
                </div>

              </div>

              <div className="card mc-section">
                <h3>模型版本</h3>
                <div>
                  <div className="kv-row kv-row--text">
                    <span>{modelStatus.demo_only ? "演示版本" : modelStatus.model_origin === "bundled_base" ? "内置版本" : "激活版本"}</span>
                    <span>{modelStatus.version ?? "—"}</span>
                  </div>
                  {modelStatus.feature_schema_version != null && (
                    <div className="kv-row">
                      <span>特征版本</span>
                      <span>v{modelStatus.feature_schema_version}</span>
                    </div>
                  )}
                  {modelStatus.model_name && (
                    <div className="kv-row kv-row--text">
                      <span>模型名称</span>
                      <span>{modelStatus.model_name}</span>
                    </div>
                  )}
                  {modelStatus.last_updated && (
                    <div className="kv-row">
                      <span>最后更新</span>
                      <span>{new Date(modelStatus.last_updated).toLocaleString("zh-CN")}</span>
                    </div>
                  )}
                </div>
              </div>

              {modelStatus.reasons && modelStatus.reasons.length > 0 && (
                <div className="card mc-section">
                  <h3>就绪原因</h3>
                  <ul className="mc-reasons">
                    {modelStatus.reasons.map((r) => (
                      <li key={r}>{userTerm(r)}</li>
                    ))}
                  </ul>
                </div>
              )}

              {modelStatus.available_versions && modelStatus.available_versions.length > 0 && (
                <div className="card mc-section">
                  <h3>可用版本</h3>
                  <div className="mc-version-list">
                    {modelStatus.available_versions.map((v) => (
                      <span className="badge badge-primary" key={v}>{v}</span>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {!modelStatus && !loading.modelStatus && (
            <p className="mc-empty">暂无数据</p>
          )}
        </div>
      )}
    </WorkspacePage>
  );
}
