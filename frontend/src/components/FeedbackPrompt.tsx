import { useCallback, useEffect, useRef, useState } from "react";
import { answerFeedbackRequest, getPendingFeedbackRequests } from "../api";
import { realtimeClient } from "../realtime";
import { toast } from "./toast-store";
import type { FeedbackLabel, FeedbackRequest } from "../types/explainability";
import "./primitives.css";

const POLL_INTERVAL_MS = 60_000;

/** "14:05–14:10" — the window the question is about, in local time. */
function formatWindow(request: FeedbackRequest): string {
  const start = new Date(request.window_start);
  const end = new Date(request.window_end);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return "";
  const clock = (value: Date) =>
    `${String(value.getHours()).padStart(2, "0")}:${String(value.getMinutes()).padStart(2, "0")}`;
  return `${clock(start)}–${clock(end)}`;
}

/**
 * Proactive-learning prompt (contract §5).
 *
 * Two delivery paths feed the same queue:
 *   1. the WebSocket `feedback_request` event, when the socket is up;
 *   2. a 60s poll of `/telemetry/feedback-requests/pending`, which is what
 *      keeps the feature working after a dropped connection.
 *
 * A backend build without the feature answers 404; `getPendingFeedbackRequests`
 * already turns that into `undefined`, so the poll simply goes quiet instead of
 * filling the console (or the user's screen) with errors.
 */
export default function FeedbackPrompt() {
  const [queue, setQueue] = useState<FeedbackRequest[]>([]);
  const [submitting, setSubmitting] = useState(false);
  /** Mirrors `queue` for the poll callback, which must not re-subscribe. */
  const seenRef = useRef<Set<string>>(new Set());

  const enqueue = useCallback((request: FeedbackRequest) => {
    if (seenRef.current.has(request.request_id)) return;
    seenRef.current.add(request.request_id);
    setQueue((current) => [...current, request]);
  }, []);

  useEffect(() => {
    const unsubscribe = realtimeClient.subscribe("feedback_request", (request) => enqueue(request));

    let cancelled = false;
    const poll = async () => {
      try {
        const pending = await getPendingFeedbackRequests();
        if (cancelled || !pending) return;
        pending.forEach(enqueue);
      } catch (error) {
        // A transient network blip must not kill the fallback: the next tick
        // retries. Anything else is a real defect worth seeing.
        console.debug("[feedback] 轮询待处理请求失败，下一轮重试", error);
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
      unsubscribe();
    };
  }, [enqueue]);

  const answer = async (request: FeedbackRequest, label: FeedbackLabel) => {
    setSubmitting(true);
    try {
      const result = await answerFeedbackRequest(request.request_id, label);
      setQueue((current) => current.filter((item) => item.request_id !== request.request_id));
      if (label === "skip") {
        toast.info("已跳过这次标注", "跳过不会写入训练标签。");
      } else if (result?.status === "expired") {
        toast.warn("这次提问已过期", "该时间窗已超出可标注范围，无需再作答。");
      } else {
        toast.success(
          label === "focus" ? "已记录：刚才在专注" : "已记录：刚才分心了",
          result ? `影响 ${result.labeled_windows} 个特征窗口，用于校准模型。` : undefined,
        );
      }
    } catch (error) {
      toast.error("提交失败", error instanceof Error ? error.message : "请稍后重试");
    } finally {
      setSubmitting(false);
    }
  };

  const current = queue[0];
  if (!current) return null;

  const windowLabel = formatWindow(current);
  const remaining = queue.length - 1;

  return (
    <section className="feedback-prompt" aria-labelledby="feedback-prompt-title" role="group">
      <div className="feedback-prompt-head">
        <span className="feedback-prompt-kicker">模型校准</span>
        {remaining > 0 && <span className="feedback-prompt-queue">还有 {remaining} 条</span>}
      </div>
      <p className="feedback-prompt-title" id="feedback-prompt-title">
        {windowLabel ? `刚才 ${windowLabel} 你在专注吗？` : current.prompt}
      </p>
      {current.reasons.length > 0 && (
        <p className="feedback-prompt-reasons">
          为什么问：{current.reasons.join(" · ")}
        </p>
      )}
      <div className="feedback-prompt-actions">
        <button
          type="button"
          className="btn btn-sm"
          disabled={submitting}
          onClick={() => void answer(current, "focus")}
        >
          专注
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          disabled={submitting}
          onClick={() => void answer(current, "distracted")}
        >
          分心
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          disabled={submitting}
          onClick={() => void answer(current, "skip")}
        >
          跳过
        </button>
      </div>
    </section>
  );
}
