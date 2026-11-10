import WorkspacePage from "../components/WorkspacePage";
import { useEffect, useState } from "react";
import { triggerPanel, getPanelResult, getErrorMessage } from "../api";
import type { ExplainablePanelResult } from "../api";
import { Link } from "react-router-dom";
import TmtGauge from "../components/charts/TmtGauge";
import MetricStrip from "../components/signal/MetricStrip";
import type { MetricItem } from "../components/signal/MetricStrip";
import HelpTip from "../components/signal/HelpTip";
import "../components/primitives.css";

const TYPE_LABELS: Record<string, string> = {
  task_aversion: "任务畏惧型", impulsivity: "冲动分心型", decisional: "决策困难型",
  perfectionism: "完美主义型", emotional_regulation: "情绪调节型",
};
const TECHNIQUE_LABELS: Record<string, string> = {
  behavioral_experiment: "行为实验", cognitive_restructuring: "认知重构", stimulus_control: "刺激控制",
  goal_setting: "目标设定", graded_exposure: "逐步接触任务", mindfulness: "正念练习",
};
const SOURCE_LABELS: Record<string, string> = { panel: "专家面板", single_expert: "单专家分析", deepseek: "DeepSeek", ollama: "本地模型", rule_engine: "规则分析" };

/** Role → chart token. Hex values were hardcoded and ignored the theme, so
 *  the same expert was unreadable in the white theme and off-brand in the
 *  burgundy one. Order matters: it assigns each role a distinct hue. */
const ROLE_COLORS: Record<string, string> = {
  analyst: "var(--chart-1)",
  分析师: "var(--chart-1)",
  psychologist: "var(--chart-4)",
  心理学家: "var(--chart-4)",
  coach: "var(--chart-3)",
  教练: "var(--chart-3)",
  strategist: "var(--chart-2)",
  策略师: "var(--chart-2)",
  facilitator: "var(--chart-6)",
  主持人: "var(--chart-6)",
  evaluator: "var(--chart-5)",
  评估师: "var(--chart-5)",
};

/** Wire role ids are internal ("analyst", "psychologist"); the card should
 *  read like a person speaking, not like a log line. */
const ROLE_LABELS: Record<string, string> = {
  analyst: "分析师", psychologist: "心理学家", coach: "教练",
  strategist: "策略师", facilitator: "主持人", evaluator: "评估师",
  critic: "批评家", moderator: "主持人", validator: "校验员",
};

function roleLabel(role: string): string {
  if (!role) return "专家";
  return ROLE_LABELS[role.toLowerCase()] ?? ROLE_LABELS[role] ?? role;
}

function roleColor(role: string): string {
  const key = (role || "").toLowerCase();
  return ROLE_COLORS[key] || ROLE_COLORS[role] || "var(--chart-1)";
}

/** Semantic confidence tone: green/amber/red carry meaning, so they come from
 *  the good/warn/bad tokens rather than a chart hue. */
function confidenceColor(v: number): string {
  if (v >= 0.7) return "var(--ok)";
  if (v >= 0.4) return "var(--drift)";
  return "var(--alarm)";
}

/** The badge variant for the same three confidence bands (§8.10). */
function confidenceBadge(v: number): string {
  if (v >= 0.7) return "badge-success";
  if (v >= 0.4) return "badge-warning";
  return "badge-danger";
}

function confidencePct(v: number): string {
  return `${Math.round((v ?? 0) * 100)}%`;
}

const FAILURE_LABELS: Record<string, string> = {
  panel_timeout: "多专家讨论超时", panel_unavailable: "专家面板未就绪或没有形成有效裁决",
  panel_error: "专家讨论发生异常", panel_schema_invalid: "专家裁决格式校验未通过",
  panel_rejected: "专家裁决未通过复核", panel_insufficient_evidence: "证据不足",
  duplicate_timeout: "等待同一分析任务超时", retry_error: "重试请求在后端执行失败",
};
function failureReason(path: string[]) {
  return path.filter(key => FAILURE_LABELS[key]).map(key => FAILURE_LABELS[key]).join("；")
    || "未返回具体失败步骤，请查看 AI 诊断中的本次运行记录。";
}

export default function Panel() {
  const [loading, setLoading] = useState<"trigger" | "read" | null>(null);
  const [result, setResult] = useState<ExplainablePanelResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [elapsed, setElapsed] = useState(0);
  const [feedback, setFeedback] = useState<{kind: "success" | "warning"; message: string} | null>(null);
  useEffect(() => {
    if (loading !== "trigger") return;
    setElapsed(0);
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [loading]);
  const handleTrigger = async (retryIfDegraded = false) => {
    setLoading("trigger");
    setError(null);
    setFeedback(null);
    try {
      const data = await triggerPanel(retryIfDegraded ? { retryIfDegraded: true } : undefined);
      setResult(data);
      const status = data.meta?.attempt_status;
      const path = status ? data.meta?.attempt_degradation_path ?? [] : data.meta?.degradation_path ?? [];
      if (status === "failed") {
        setError(`本次重试失败，当前仍展示此前保存的结果。${failureReason(path)}`);
      } else if (data.insufficient_data) {
        setFeedback({kind:"warning", message:"本次分析未形成结论：采集证据不足，请查看下方的数据缺口。"});
      } else if (data.degraded) {
        const lead = status === "degraded" ? (data.cached ? "本次重试仍未完成专家讨论，当前保留此前结果。" : "本次专家讨论未完成，已返回后备分析结果。")
          : data.cached ? "服务器返回了已保存的降级结果，未确认本次重新分析是否执行。请检查后端版本。" : "本次专家讨论未完成，已返回后备分析结果。";
        setFeedback({kind:"warning", message:lead + failureReason(path)});
      } else {
        setFeedback({kind:"success", message:data.cached ? "已读取保存的专家分析结果，本次未启动新分析。" : "专家分析已完成，以下为本次结果。"});
      }
    } catch (e: unknown) {
      setError(getErrorMessage(e, "专家面板请求失败"));
    } finally {
      setLoading(null);
    }
  };

  const handleRead = async () => {
    setLoading("read");
    setError(null);
    setFeedback(null);
    try {
      const data = await getPanelResult();
      setResult(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "读取专家面板结果失败"));
    } finally {
      setLoading(null);
    }
  };

  const types: { name: string; confidence: number }[] = result?.types.map((name) => ({
    name,
    confidence: result.confidence[name] ?? 0,
  })) ?? [];
  const discussionRounds = result?.transcript ?? [];
  // §8.10: the process facts the verdict rests on, as one strip of four.
  const metaMetrics: MetricItem[] = [
    {
      key: "uncertainty",
      label: "不确定度",
      value: result?.uncertainty != null ? result.uncertainty.toFixed(2) : "—",
      help: { what: "专家们对结论有多不确定，0 到 1，越接近 0 越有把握。" },
    },
    {
      key: "escalated",
      label: "升级复核",
      value: result?.escalated ? "是" : "否",
      help: { what: "专家意见分歧大时，会多加一轮复查。“是”表示这次复查过。" },
    },
    {
      key: "calls",
      label: "模型调用",
      value: result?.call_count != null ? String(result.call_count) : "—",
      unit: "次",
      help: { what: "这次分析一共请求了几次 AI，次数越多花费越多。" },
    },
    {
      key: "rounds",
      label: "发言条数",
      value: String(discussionRounds.length),
      unit: "条",
      help: { what: "专家们在讨论中一共说了多少段话。" },
    },
  ];
  const hasContent = Boolean(result && (types.length > 0 || result.technique || result.rationale || discussionRounds.length > 0));

  return (
    <WorkspacePage title="专家面板" description="让多个专家共同分析今天的行为，查看可能的原因和下一步建议。讨论细节按需展开。" className="panel-page">


      {error && (
        <div className="error-box mb16" role="alert">
          {error}
          <button className="btn btn-secondary btn-sm panel-error-close" onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}

      <div className="page-toolbar">
        <button className="btn btn-primary" onClick={() => handleTrigger()} disabled={loading !== null}>
          {loading === "trigger" && (
            <span className="spinner spinner--inline" />
          )}
          {loading === "trigger" ? "分析中…" : "运行专家面板"}
        </button>
        <HelpTip
          term="专家面板"
          what="几位 AI“专家”分别分析你今天的情况，互相讨论后给出结论。"
          effect="需要联网调用云端 AI 多次，可能要几分钟并产生费用；今天已经有结果时会直接显示，不会重新运行。"
        />
        <button className="btn btn-secondary" onClick={handleRead} disabled={loading !== null}>
          {loading === "read" && (
            <span className="spinner spinner--inline" />
          )}
          {loading === "read" ? "读取中…" : "查看上次结果"}
        </button>
      </div>

      {loading === "trigger" && <div className="card mb24 panel-progress" role="status" aria-live="polite">
        <span className="spinner spinner--inline panel-progress-spinner" />
        <div><strong>正在运行 AI 专家分析 · 已等待 {elapsed} 秒</strong>
          <p>专家分析需要多轮模型调用，可能等待数分钟。完成后会显示本次结果或失败原因。{result ? "下方暂时展示上次结果。" : ""}</p></div>
      </div>}
      {feedback && <div className={`${feedback.kind === "warning" ? "error-box" : "info-box"} mb24`} role="status">{feedback.message}</div>}
      {(error || result?.degraded) && <div className="mb16"><Link to="/diagnostics">查看 AI 诊断与运行记录</Link></div>}
      {loading === "read" && <div role="status" className="info-box mb24">正在读取已保存的结果…</div>}

      {result?.insufficient_data && (
        <div className="info-box mb24" role="status">
          <strong>今天的数据还不足以形成可靠结论</strong>
          <p>继续使用并采集一段时间后再试。{result.evidence_gaps?.join("；")}</p>
        </div>
      )}

      {result && <div className="workspace-actions mb24">
        <span className="badge badge-info">{result.cached ? "已保存的结果" : "本次分析结果"}</span>
        <span className="text-muted">来源：{SOURCE_LABELS[result.source ?? result.meta?.source ?? ""] ?? result.source ?? result.meta?.source ?? "未知"}</span>
      </div>}

      {/* How the verdict was reached: uncertainty, whether the panel escalated,
          and how many model calls it took. All three were in the payload and
          none of them were on screen. */}
      {result && (
        <div className="panel-meta" role="group" aria-label="本次分析的过程信息">
          <MetricStrip items={metaMetrics} />
        </div>
      )}

      {result?.degraded && (
        <div className="notice mb16">
          <div className="notice-text">
            <span className="badge badge-warning">降级模式</span>
            <HelpTip
              term="降级模式"
              what="完整的专家讨论没能完成，系统改用更简单的方法给出结果，参考价值会低一些。"
            />
            <span className="panel-degraded-text">
              来源：{SOURCE_LABELS[result.source ?? result.meta?.source ?? ""] ?? "其他分析"}；专家讨论未完成，重试将使用当前配置的 AI 服务
            </span>
          </div>
          <div className="notice-action">
            <button className="btn btn-secondary btn-sm" onClick={() => handleTrigger(true)} disabled={loading !== null}>
              {loading === "trigger" ? "重试中…" : "重试 AI 分析"}
            </button>
            <HelpTip
              term="重试"
              what="重新运行一次专家讨论。"
              effect="只有上次是降级结果时才会重新运行，会再次产生 AI 费用；如果又失败，会保留原来的结果。"
            />
          </div>
        </div>
      )}

      {hasContent && (
        <div className="panel-results">
          {types.length > 0 && (
            <div className="card">
              <h3>拖延类型分析</h3>
              <div className="panel-type-list">
                {types.map((t, i) => {
                  const v = typeof t.confidence === "number" ? t.confidence : 0;
                  return (
                    <div key={i}>
                      <div className="panel-type-head">
                        <span className="panel-type-name">{TYPE_LABELS[t.name] ? `${TYPE_LABELS[t.name]}（${t.name}）` : (t.name || `类型 ${i + 1}`)}</span>
                        <span className={`badge ${confidenceBadge(v)}`}>
                          {confidencePct(v)}
                        </span>
                      </div>
                      <div className="mf-tmt-track">
                        <div
                          className="mf-tmt-fill"
                          style={{ width: confidencePct(v), background: confidenceColor(v) }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {result?.technique && (
            <div className="card">
              <h3>推荐认知行为技术（CBT）</h3>
              <p>{TECHNIQUE_LABELS[result.technique] ? `${TECHNIQUE_LABELS[result.technique]}（${result.technique}）` : result.technique}</p>
            </div>
          )}

          {result?.rationale && (
            <div className="card workspace-full">
              <h3>分析依据</h3>
              <p className="panel-rationale">{result.rationale}</p>
            </div>
          )}

          {result?.tmt && (
            <div className="card workspace-full">
              <h3>动机分解</h3>
              <TmtGauge tmt={result.tmt} />
            </div>
          )}

          {discussionRounds.length > 0 && (
            <details className="workspace-details workspace-full">
              <summary>专家讨论记录 · {discussionRounds.length} 条</summary>
              {/* One card per speaker turn: a wall of run-on paragraphs hides
                  who said what, which is the whole point of a panel. */}
              <div className="opinion-list">
                {discussionRounds.map((round, i) => (
                  <article
                    className="opinion-card"
                    key={i}
                    style={{ ["--opinion-color" as string]: roleColor(round.role) }}
                  >
                    <div className="opinion-head">
                      <span className="opinion-role">{roleLabel(round.role)}</span>
                      {round.round != null && (
                        <span className="badge badge-neutral">第 {round.round} 轮</span>
                      )}
                    </div>
                    <p className="opinion-body">{round.content}</p>
                  </article>
                ))}
              </div>
            </details>
          )}
          {(result?.dissent?.length ?? 0) > 0 && <details className="workspace-details workspace-full"><summary>不同意见 · {result?.dissent.length} 条</summary><div>{result?.dissent.map((opinion, index) => <p key={index}>{opinion}</p>)}</div></details>}
        </div>
      )}

      {!loading && !hasContent && !error && !result?.insufficient_data && (
        <div className="empty-state">今天的分析尚未显示。<br />点击“运行专家面板”获取分析，或“查看上次结果”读取已保存的结论。</div>
      )}
    </WorkspacePage>
  );
}
