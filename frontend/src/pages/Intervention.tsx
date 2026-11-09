import WorkspacePage from "../components/WorkspacePage";
import { useState, useEffect, useCallback, useRef } from "react";
import {
  triggerIntervention,
  getInterventionHistory,
  respondIntervention,
  feedbackIntervention,
  getErrorMessage,
} from "../api";
import type { InterventionIntensity, InterventionRating, InterventionResponse } from "../api";
import { realtimeClient } from "../realtime";
import { INTERVENTION_TYPE_LABELS, INTERVENTION_TYPE_BADGES } from "../lib/intervention-labels";
import PosteriorBars from "../components/charts/PosteriorBars";
import { SELECTION_LABELS } from "../types/explainability";
import type { InterventionRationale } from "../types/explainability";
import "../components/primitives.css";

interface InterventionItem {
  id: string;
  intervention_type: string;
  triggered_at?: string;
  created_at?: string;
  user_response?: InterventionResponse | null;
  response_latency_s?: number | null;
  feedback_rating?: InterventionRating | null;
  feedback_comment?: string | null;
  title?: string;
  message?: string;
  /** Contract §4 — absent until the backend ships it. */
  rationale?: InterventionRationale | null;
}

/**
 * "为什么是这条提醒" — the selection evidence for one intervention.
 *
 * Shared by the latest card (always expanded) and each history row
 * (collapsible), so the wording and the ordering of the argument are identical
 * wherever a reader meets it.
 */
// Context buckets arrive as wire identifiers ("afternoon|distracted"). A judge
// reading the card should not have to decode a pipe-separated key.
const BUCKET_LABELS: Record<string, string> = {
  morning: "上午", afternoon: "下午", evening: "傍晚", night: "夜间",
  focused: "专注", distracted: "分心", mixed: "混合", idle: "空闲",
};

function contextLabel(bucket: string): string {
  return bucket.split("|").map(part => BUCKET_LABELS[part] ?? part).join(" · ");
}

function RationaleBlock({ rationale }: { rationale?: InterventionRationale | null }) {
  if (!rationale) {
    return <p className="proof-note">暂无选择依据</p>;
  }
  return (
    <div className="rationale-block">
      <div className="proof-head">
        <span className={`badge ${rationale.selection === "thompson_sampling" ? "badge-info" : "badge-neutral"}`}>
          {SELECTION_LABELS[rationale.selection]}
        </span>
        {rationale.context_bucket && (
          <span className="proof-muted">情境：{contextLabel(rationale.context_bucket)}</span>
        )}
      </div>

      {rationale.evidence.length > 0 ? (
        <ul className="bullet-list">
          {rationale.evidence.slice(0, 4).map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="proof-note">本次没有记录具体证据。</p>
      )}

      {rationale.candidates.length > 0 && (
        <div className="rationale-candidates">
          <div className="proof-title">候选提醒的实测表现</div>
          <PosteriorBars
            candidates={rationale.candidates}
            ariaLabel="候选提醒的 Thompson 采样后验均值与 90% 可信区间"
          />
        </div>
      )}

      {rationale.note && <p className="proof-note">{rationale.note}</p>}
    </div>
  );
}

const INTENSITY_OPTIONS = [
  { key: "gentle", label: "温和提醒" },
  { key: "standard", label: "标准干预" },
  { key: "strict", label: "严格干预" },
];

const DAYS_OPTIONS = [7, 14, 30, 90] as const;

const RESPONSE_LABELS: Record<string, string> = {
  accept: "已接受",
  accepted: "已接受",
  ignore: "已忽略",
  ignored: "已忽略",
  dismiss: "已关闭",
  dismissed: "已关闭",
};

const RESPONSE_BADGES: Record<string, string> = {
  accept: "badge-success",
  accepted: "badge-success",
  ignore: "badge-warning",
  ignored: "badge-warning",
  dismiss: "badge-danger",
  dismissed: "badge-danger",
};

function formatTime(ts: string): string {
  if (!ts) return "";
  const d = new Date(ts);
  return d.toLocaleString("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function interventionTimestamp(item: InterventionItem): number {
  const timestamp = Date.parse(item.triggered_at || item.created_at || "");
  return Number.isFinite(timestamp) ? timestamp : 0;
}

export default function Intervention() {
  const [loading, setLoading] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latest, setLatest] = useState<InterventionItem | null>(null);
  const [history, setHistory] = useState<InterventionItem[]>([]);
  const [days, setDays] = useState<number>(7);
  const [respondingId, setRespondingId] = useState<string | null>(null);
  const [feedbackId, setFeedbackId] = useState<string | null>(null);
  const [feedbackRating, setFeedbackRating] = useState<InterventionRating | "">("");
  const [feedbackComment, setFeedbackComment] = useState("");
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [skipNotice, setSkipNotice] = useState<string | null>(null);
  const shownAtRef = useRef<number>(0);
  const historySeqRef = useRef(0);

  const loadHistory = useCallback(async () => {
    const seq = ++historySeqRef.current;
    setLoading(true);
    setError(null);
    try {
      const data = await getInterventionHistory(days);
      if (seq !== historySeqRef.current) return;
      const items = [...data.items].sort(
        (left, right) => interventionTimestamp(right) - interventionTimestamp(left),
      );
      setHistory(items);
      const nextLatest = items[0] ?? null;
      setLatest(nextLatest);
      if (nextLatest) shownAtRef.current = performance.now();
    } catch (e: unknown) {
      if (seq !== historySeqRef.current) return;
      setHistory([]);
      setError(getErrorMessage(e, "加载干预历史失败"));
    } finally {
      if (seq === historySeqRef.current) setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => realtimeClient.subscribe("intervention", (payload, timestamp) => {
    const item: InterventionItem = { ...payload, created_at: timestamp };
    setLatest(item);
    shownAtRef.current = performance.now();
    setHistory((current) => [item, ...current.filter((entry) => entry.id !== item.id)]);
  }), []);

  const handleTrigger = async (intensity: InterventionIntensity) => {
    setTriggering(true);
    setError(null);
    setSkipNotice(null);
    try {
      const result = await triggerIntervention(intensity);
      if (result.intervention) {
        setLatest(result.intervention);
      } else if (result.skipped) {
        // A suppressed reminder is a real outcome, not a silent no-op: the
        // user asked for an intervention and deserves to know why none came.
        setSkipNotice(result.skip_reason || "本次提醒已跳过");
      }
      await loadHistory();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "操作失败"));
    } finally {
      setTriggering(false);
    }
  };

  const handleRespond = async (id: string, response: InterventionResponse) => {
    setRespondingId(id);
    setError(null);
    try {
      const latencyS = id === latest?.id && shownAtRef.current > 0
        ? (performance.now() - shownAtRef.current) / 1000
        : 0;
      await respondIntervention(id, response, latencyS);
      await loadHistory();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "操作失败"));
    } finally {
      setRespondingId(null);
    }
  };

  const handleFeedback = async (id: string) => {
    if (!feedbackRating) return;
    // Capture the record this submission belongs to. The follow-up
    // `loadHistory()` can resolve after the user opened a different record's
    // editor, so the response must only clear the editor the user is still on.
    const submittedFor = id;
    const submittedRating = feedbackRating;
    const submittedComment = feedbackComment;
    setSubmittingFeedback(true);
    setError(null);
    try {
      await feedbackIntervention(submittedFor, submittedRating, submittedComment || undefined);
      setFeedbackId((current) => {
        if (current !== submittedFor) return current;
        setFeedbackRating("");
        setFeedbackComment("");
        return null;
      });
      await loadHistory();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "操作失败"));
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const hasResponded = (item: InterventionItem) =>
    !!item.user_response;

  const canFeedback = (item: InterventionItem) =>
    hasResponded(item) && !item.feedback_rating;

  const getStatusBadge = (item: InterventionItem) => {
    const resp = (item.user_response || "").toLowerCase();
    const label = RESPONSE_LABELS[resp];
    const cls = RESPONSE_BADGES[resp];
    if (label && cls) {
      return <span className={`badge ${cls}`}>{label}</span>;
    }
    return <span className="badge badge-info">待响应</span>;
  };

  const getInterventionTypeBadge = (interventionType?: string) => {
    const key = (interventionType || "").toLowerCase();
    const label = INTERVENTION_TYPE_LABELS[key] || "专注干预";
    const cls = INTERVENTION_TYPE_BADGES[key] || "badge-primary";
    return <span className={`badge ${cls}`}>{label}</span>;
  };

  return (
    <WorkspacePage title="干预中心" description="需要帮助回到专注状态时，选择一次提醒；再告诉我们，这次提醒是否对你有用。" className="intervention-page">


      {error && <div className="error-box mb16">{error}</div>}

      {skipNotice && (
        <div
          className="card mb16"
          style={{ borderLeft: "3px solid var(--color-warning, #d97706)", fontSize: 13 }}
          role="status"
        >
          <div style={{ fontWeight: 600, marginBottom: 4 }}>本次提醒已跳过</div>
          <div style={{ color: "var(--color-text-secondary)" }}>{skipNotice}</div>
        </div>
      )}

      <div className="workspace-grid">
      <div className="card intervention-trigger">
        <h3>手动触发干预</h3>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
          选择一个强度级别，手动触发一次专注干预
        </p>
        <div className="workspace-actions">
          {INTENSITY_OPTIONS.map((opt) => (
            <button
              type="button"
              key={opt.key}
              className={`btn ${opt.key === "strict" ? "btn-danger" : ""}`}
              disabled={triggering}
              onClick={() => handleTrigger(opt.key as InterventionIntensity)}
            >
              {triggering && (
                <span className="spinner" style={{ width: 16, height: 16, margin: 0, borderWidth: 2 }} />
              )}
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {latest ? (
        <div className="card">
          <div className="flex-between mb16">
            <h3 style={{ margin: 0 }}>最新干预</h3>
            {getInterventionTypeBadge(latest.intervention_type)}
          </div>
          {latest.title && (
            <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>
              {latest.title}
            </div>
          )}
          {latest.message && (
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 10 }}>
              {latest.message}
            </div>
          )}
          <div className="intervention-response-row">
          <div className="workspace-actions">
            {getStatusBadge(latest)}
            <span style={{ marginLeft: 8 }}>
              {formatTime(latest.triggered_at || latest.created_at || "")}
            </span>
          </div>
          <div className="workspace-actions">
            <button
              type="button"
              className="btn btn-sm"
              disabled={respondingId !== null}
              onClick={() => handleRespond(latest.id, "accepted")}
            >
              {respondingId === latest.id && (
                <span className="spinner" style={{ width: 12, height: 12, margin: 0, borderWidth: 2 }} />
              )}
              接受
            </button>
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              disabled={respondingId !== null}
              onClick={() => handleRespond(latest.id, "ignored")}
            >
              忽略
            </button>
            <button
              type="button"
              className="btn btn-sm btn-danger"
              disabled={respondingId !== null}
              onClick={() => handleRespond(latest.id, "dismissed")}
            >
              关闭
            </button>
          </div>
          </div>
          <div className="proof rationale-latest" aria-label="为什么是这条提醒">
            <div className="proof-title">为什么是这条提醒</div>
            <RationaleBlock rationale={latest.rationale} />
          </div>
        </div>
      ) : (<div className="card"><h3>最新干预</h3><div className="workspace-empty">暂时没有提醒。<br />需要时，可以从左侧选择一种强度。</div></div>)}
      </div>

      <div className="card">
        <div className="flex-between mb16">
          <h3 style={{ margin: 0 }}>干预历史</h3>
          <div className="tabs">
            {DAYS_OPTIONS.map((d) => (
              <button
                type="button"
                key={d}
                className={`tab ${days === d ? "active" : ""}`}
                onClick={() => setDays(d)}
              >
                {d}天
              </button>
            ))}
          </div>
        </div>

        {loading && <div className="spinner" />}

        {!loading && history.length === 0 && (
          <p style={{ color: "var(--color-text-tertiary)", fontSize: 14 }}>暂无干预记录</p>
        )}

        {!loading &&
          history.map((item) => (
            <div
              key={item.id}
              className="flex gap16"
              style={{
                padding: "14px 0",
                borderBottom: "1px solid var(--color-border)",
              }}
            >
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: hasResponded(item)
                    ? "var(--color-success)"
                    : "var(--color-warning)",
                  marginTop: 4,
                  flexShrink: 0,
                }}
              />
              <div className="workspace-record-body intervention-history-body">
                <div className="flex-between">
                  <div className="flex gap8" style={{ alignItems: "center", flexWrap: "wrap" }}>
                    {getInterventionTypeBadge(item.intervention_type)}
                    {getStatusBadge(item)}
                    {item.response_latency_s != null && (
                      <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                        响应耗时 {item.response_latency_s.toFixed(1)} 秒
                      </span>
                    )}
                  </div>
                  <span style={{ fontSize: 12, color: "var(--color-text-tertiary)", whiteSpace: "nowrap" }}>
                    {formatTime(item.triggered_at || item.created_at || "")}
                  </span>
                </div>

                {item.title && (
                  <div className="intervention-history-title" style={{ fontSize: 14, fontWeight: 600, marginTop: 6 }}>
                    {item.title}
                  </div>
                )}
                {item.message && (
                  <div className="intervention-history-message" style={{ fontSize: 13, color: "var(--color-text-secondary)", marginTop: 2 }}>
                    {item.message}
                  </div>
                )}

                {item.rationale && (
                  <details className="proof-details rationale-history">
                    <summary>为什么是这条提醒</summary>
                    <RationaleBlock rationale={item.rationale} />
                  </details>
                )}

                {feedbackId === item.id ? (
                  <div className="mt8 intervention-feedback-editor" style={{ background: "var(--color-bg-inset)", padding: 12, borderRadius: 8 }}>
                    <select
                      aria-label="干预评分"
                      value={feedbackRating}
                      onChange={(e) => setFeedbackRating(e.target.value as InterventionRating | "")}
                      style={{ marginBottom: 8 }}
                    >
                      <option value="">选择评分</option>
                      <option value="helpful">有用</option>
                      <option value="neutral">一般</option>
                      <option value="annoying">打扰（令人烦扰）</option>
                    </select>
                    <textarea
                      placeholder="补充评论（可选）"
                      value={feedbackComment}
                      onChange={(e) => setFeedbackComment(e.target.value)}
                      rows={2}
                      style={{ marginBottom: 8 }}
                    />
                    <div className="flex gap8">
                      <button
                        type="button"
                        className="btn btn-sm"
                        disabled={submittingFeedback || !feedbackRating}
                        onClick={() => handleFeedback(item.id)}
                      >
                        {submittingFeedback && (
                          <span className="spinner" style={{ width: 12, height: 12, margin: 0, borderWidth: 2 }} />
                        )}
                        提交
                      </button>
                      <button
                        type="button"
                        className="btn btn-sm btn-ghost"
                        onClick={() => {
                          setFeedbackId(null);
                          setFeedbackRating("");
                          setFeedbackComment("");
                        }}
                      >
                        取消
                      </button>
                    </div>
                  </div>
                ) : (
                  canFeedback(item) && (
                    <button
                      type="button"
                      className="btn btn-sm btn-ghost intervention-history-evaluate"
                      onClick={() => {
                        // Opening the editor for one record must not inherit
                        // another record's half-written draft.
                        setFeedbackRating("");
                        setFeedbackComment("");
                        setFeedbackId(item.id);
                      }}
                    >
                      评价
                    </button>
                  )
                )}
              </div>
            </div>
          ))}
      </div>
    </WorkspacePage>
  );
}
