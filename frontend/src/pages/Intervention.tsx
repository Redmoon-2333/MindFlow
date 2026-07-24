import { useState, useEffect, useCallback } from "react";
import {
  triggerIntervention,
  getInterventionHistory,
  respondIntervention,
  feedbackIntervention,
} from "../api";

interface InterventionItem {
  id: number;
  type?: string;
  intensity?: string;
  created_at?: string;
  response?: string | null;
  responded_at?: string | null;
  latency?: number | null;
  rating?: string | null;
  comment?: string | null;
}

const INTENSITY_OPTIONS = [
  { key: "gentle", label: "温和提醒" },
  { key: "standard", label: "标准干预" },
  { key: "strict", label: "严格干预" },
];

const DAYS_OPTIONS = [7, 14, 30, 90] as const;

const INTENSITY_LABELS: Record<string, string> = {
  gentle: "温和",
  standard: "标准",
  strict: "严格",
};

const INTENSITY_BADGES: Record<string, string> = {
  gentle: "badge-info",
  standard: "badge-warning",
  strict: "badge-danger",
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
  const [respondingId, setRespondingId] = useState<number | null>(null);
  const [feedbackId, setFeedbackId] = useState<number | null>(null);
  const [feedbackRating, setFeedbackRating] = useState("");
  const [feedbackComment, setFeedbackComment] = useState("");
  const [submittingFeedback, setSubmittingFeedback] = useState(false);

  const loadHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getInterventionHistory(days);
      const items: InterventionItem[] = Array.isArray(data) ? data : data.items || [];
      setHistory(items);
      setLatest(items.length > 0 ? items[0] : null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const handleTrigger = async (intensity: string) => {
    setTriggering(true);
    setError(null);
    try {
      const intervention = await triggerIntervention(intensity);
      setLatest(intervention);
      await loadHistory();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setTriggering(false);
    }
  };

  const handleRespond = async (id: number, response: string) => {
    setRespondingId(id);
    setError(null);
    try {
      await respondIntervention(id, response);
      await loadHistory();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRespondingId(null);
    }
  };

  const handleFeedback = async (id: number) => {
    if (!feedbackRating) return;
    setSubmittingFeedback(true);
    setError(null);
    try {
      await feedbackIntervention(id, feedbackRating, feedbackComment || undefined);
      setFeedbackId(null);
      setFeedbackRating("");
      setFeedbackComment("");
      await loadHistory();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const hasResponded = (item: InterventionItem) =>
    !!item.response;

  const canFeedback = (item: InterventionItem) =>
    hasResponded(item) && !item.rating;

  const getStatusBadge = (item: InterventionItem) => {
    const resp = (item.response || "").toLowerCase();
    const label = RESPONSE_LABELS[resp];
    const cls = RESPONSE_BADGES[resp];
    if (label && cls) {
      return <span className={`badge ${cls}`}>{label}</span>;
    }
    return <span className="badge badge-info">待响应</span>;
  };

  const getIntensityBadge = (intensity?: string) => {
    const key = (intensity || "").toLowerCase();
    const label = INTENSITY_LABELS[key] || intensity;
    const cls = INTENSITY_BADGES[key] || "badge-primary";
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
              key={opt.key}
              className={`btn ${opt.key === "strict" ? "btn-danger" : ""}`}
              disabled={triggering}
              onClick={() => handleTrigger(opt.key)}
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
            {getIntensityBadge(latest.intensity)}
          </div>
          <div style={{ fontSize: 14, color: "var(--color-text-secondary)" }}>
            {getStatusBadge(latest)}
            <span style={{ marginLeft: 8 }}>
              {formatTime(latest.created_at || "")}
            </span>
          </div>
          <div className="flex gap8" style={{ marginTop: 12 }}>
            <button
              className="btn btn-sm"
              disabled={respondingId !== null}
              onClick={() => handleRespond(latest.id, "accept")}
            >
              {respondingId === latest.id && (
                <span className="spinner" style={{ width: 12, height: 12, margin: 0, borderWidth: 2 }} />
              )}
              接受
            </button>
            <button
              className="btn btn-sm btn-ghost"
              disabled={respondingId !== null}
              onClick={() => handleRespond(latest.id, "ignore")}
            >
              忽略
            </button>
            <button
              className="btn btn-sm btn-danger"
              disabled={respondingId !== null}
              onClick={() => handleRespond(latest.id, "dismiss")}
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
                    {getIntensityBadge(item.intensity)}
                    {getStatusBadge(item)}
                    {item.latency != null && (
                      <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                        延迟 {item.latency}s
                      </span>
                    )}
                  </div>
                  <span style={{ fontSize: 12, color: "var(--color-text-tertiary)", whiteSpace: "nowrap" }}>
                    {formatTime(item.created_at || "")}
                  </span>
                </div>

                {feedbackId === item.id ? (
                  <div className="mt8" style={{ background: "var(--color-bg-inset)", padding: 12, borderRadius: 8 }}>
                    <select
                      value={feedbackRating}
                      onChange={(e) => setFeedbackRating(e.target.value)}
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
                    <button className="btn btn-sm btn-ghost mt8" onClick={() => setFeedbackId(item.id)}>
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
