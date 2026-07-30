import { useState, useEffect, useCallback } from "react";
import {
  triggerIntervention,
  getInterventionHistory,
  respondIntervention,
  feedbackIntervention,
  getErrorMessage,
} from "../api";
import type { InterventionIntensity, InterventionRating, InterventionResponse } from "../api";
import { realtimeClient } from "../realtime";

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
}

const INTENSITY_OPTIONS = [
  { key: "gentle", label: "温和提醒" },
  { key: "standard", label: "标准干预" },
  { key: "strict", label: "严格干预" },
];

const DAYS_OPTIONS = [7, 14, 30, 90] as const;

const INTERVENTION_TYPE_LABELS: Record<string, string> = {
  task_breakdown: "任务分解",
  nudge: "行动提示",
  environment_optimization: "环境优化",
  smart_prioritization: "优先级建议",
};

const INTERVENTION_TYPE_BADGES: Record<string, string> = {
  task_breakdown: "badge-primary",
  nudge: "badge-info",
  environment_optimization: "badge-success",
  smart_prioritization: "badge-warning",
};

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

  const loadHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getInterventionHistory(days);
      const items = [...data.items].reverse();
      setHistory(items);
      setLatest(items[0] ?? null);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "加载干预历史失败"));
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => realtimeClient.subscribe("intervention", (payload, timestamp) => {
    const item: InterventionItem = { ...payload, created_at: timestamp };
    setLatest(item);
    setHistory((current) => [item, ...current.filter((entry) => entry.id !== item.id)]);
  }), []);

  const handleTrigger = async (intensity: InterventionIntensity) => {
    setTriggering(true);
    setError(null);
    try {
      const result = await triggerIntervention(intensity);
      if (result.intervention) setLatest(result.intervention);
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
      await respondIntervention(id, response);
      await loadHistory();
    } catch (e: unknown) {
      setError(getErrorMessage(e, "操作失败"));
    } finally {
      setRespondingId(null);
    }
  };

  const handleFeedback = async (id: string) => {
    if (!feedbackRating) return;
    setSubmittingFeedback(true);
    setError(null);
    try {
      await feedbackIntervention(id, feedbackRating, feedbackComment || undefined);
      setFeedbackId(null);
      setFeedbackRating("");
      setFeedbackComment("");
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
    <div>
      <div className="header">
        <h1>干预中心</h1>
        <p>管理智能干预策略，查看干预记录与响应情况</p>
      </div>

      {error && <div className="error-box mb16">{error}</div>}

      <div className="card mb24">
        <h3>手动触发干预</h3>
        <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
          选择一个强度级别，手动触发一次专注干预
        </p>
        <div className="flex gap8">
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

      {latest && (
        <div className="card mb24">
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
          <div style={{ fontSize: 14, color: "var(--color-text-secondary)" }}>
            {getStatusBadge(latest)}
            <span style={{ marginLeft: 8 }}>
              {formatTime(latest.created_at || "")}
            </span>
          </div>
          <div className="flex gap8" style={{ marginTop: 12 }}>
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
      )}

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
              <div style={{ flex: 1 }}>
                <div className="flex-between">
                  <div className="flex gap8" style={{ alignItems: "center", flexWrap: "wrap" }}>
                    {getInterventionTypeBadge(item.intervention_type)}
                    {getStatusBadge(item)}
                    {item.response_latency_s != null && (
                      <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                        延迟 {item.response_latency_s}s
                      </span>
                    )}
                  </div>
                  <span style={{ fontSize: 12, color: "var(--color-text-tertiary)", whiteSpace: "nowrap" }}>
                    {formatTime(item.triggered_at || item.created_at || "")}
                  </span>
                </div>

                {item.title && (
                  <div style={{ fontSize: 14, fontWeight: 600, marginTop: 6 }}>
                    {item.title}
                  </div>
                )}
                {item.message && (
                  <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginTop: 2 }}>
                    {item.message}
                  </div>
                )}

                {feedbackId === item.id ? (
                  <div className="mt8" style={{ background: "var(--color-bg-inset)", padding: 12, borderRadius: 8 }}>
                    <select
                      value={feedbackRating}
                      onChange={(e) => setFeedbackRating(e.target.value as InterventionRating | "")}
                      style={{ marginBottom: 8 }}
                    >
                      <option value="">选择评分</option>
                      <option value="effective">有用</option>
                      <option value="neutral">一般</option>
                      <option value="ineffective">无效</option>
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
                    <button type="button" className="btn btn-sm btn-ghost mt8" onClick={() => setFeedbackId(item.id)}>
                      评价
                    </button>
                  )
                )}
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}
