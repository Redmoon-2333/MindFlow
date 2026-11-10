import NativeDateField from "../components/NativeDateField";
import { useState, useEffect, useCallback, useRef } from "react";
import { getErrorMessage, getFocusSessions, getFocusTrend, submitFocusFeedback } from "../api";
import type { FocusSession, FocusTrendDay, FocusTrendResponse } from "../api";
import { dayLabel, localDateStr } from "../date-utils";
import Surface from "../components/signal/Surface";
import SectionHeader from "../components/signal/SectionHeader";
import MetricStrip from "../components/signal/MetricStrip";
import type { MetricItem } from "../components/signal/MetricStrip";
import StackedBar from "../components/charts/StackedBar";
import HelpTip from "../components/signal/HelpTip";
import "./focus.css";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isCompatibleTrendDay(value: unknown): value is FocusTrendDay {
  if (!isRecord(value)) return false;
  const numberFields = ["focus_min", "distraction_min", "session_count", "avg_score"];
  return typeof value.date === "string"
    && numberFields.every((field) => value[field] == null || typeof value[field] === "number");
}

function getTrendDays(value: unknown): FocusTrendDay[] {
  if (!isRecord(value)) return [];
  for (const key of ["daily", "daily_data"]) {
    const candidate = value[key];
    if (Array.isArray(candidate)) return candidate.filter(isCompatibleTrendDay);
  }
  return [];
}

interface FeedbackDraft {
  label: "focus" | "distracted" | "mixed";
  score: number;
  taskType: string;
}

function formatMinutes(m: number | null | undefined): string {
  if (m == null || isNaN(m)) return "—";
  return `${Math.round(m)} 分钟`;
}

function sessionDurationMinutes(session: FocusSession): number {
  if (session.duration_minutes != null) return Number(session.duration_minutes);
  if (session.duration != null) return Number(session.duration);
  const start = new Date(session.start_time ?? session.started_at ?? "");
  const end = new Date(session.end_time ?? session.ended_at ?? "");
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return 0;
  return Math.max(0, (end.getTime() - start.getTime()) / 60000);
}

function todayStr(): string {
  return localDateStr();
}

/** Weekday label for a chart slot. Only a slot whose local date *is* today may
 *  read 今天 — the trend series is sparse, so "last entry" ≠ "today" (audit F1). */
function chartWeekLabel(date: string, todayKey: string): string {
  if (date === todayKey) return "今天";
  // §8.21.5: "三" alone is unambiguous only next to its date; spell out 周三
  // so the tick matches the dashboard.
  return `周${dayLabel(date, true)}`;
}

export default function Focus() {
  const [date, setDate] = useState(todayStr());
  const [sessions, setSessions] = useState<FocusSession[]>([]);
  const [trend, setTrend] = useState<FocusTrendResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [sessionsReady, setSessionsReady] = useState(false);
  const [trendLoading, setTrendLoading] = useState(false);
  /** Separates "the trend is empty" from "the trend request failed"; both used
   *  to fall through to the same 暂无趋势数据 message (audit F1). */
  const [trendError, setTrendError] = useState(false);
  const [sessionsError, setSessionsError] = useState("");
  const [feedbackErrors, setFeedbackErrors] = useState<Record<string, string>>({});
  const [feedbackDrafts, setFeedbackDrafts] = useState<Record<string, FeedbackDraft>>({});
  const [feedbackSaving, setFeedbackSaving] = useState<Set<string>>(new Set());
  const [feedbackSaved, setFeedbackSaved] = useState<Set<string>>(new Set());
  const [savedFeedback, setSavedFeedback] = useState<Record<string, FeedbackDraft>>({});

  // Request-sequence guard: a slow response for an older date must never
  // overwrite the newer selection (see audit report — stale-overwrite race).
  const requestSeqRef = useRef(0);
  const trendSeqRef = useRef(0);
  const feedbackPendingRef = useRef(new Set<string>());

  const loadSessions = useCallback(async (d: string) => {
    const seq = ++requestSeqRef.current;
    setLoading(true);
    setSessionsReady(false);
    setSessions([]);
    setSessionsError("");
    try {
      const data = await getFocusSessions(d);
      if (seq !== requestSeqRef.current) return; // stale response
      setSessions(data.sessions);
      setSessionsReady(true);
    } catch (e: unknown) {
      if (seq !== requestSeqRef.current) return;
      setSessionsError(getErrorMessage(e, "加载失败"));
      setSessions([]);
    } finally {
      if (seq === requestSeqRef.current) setLoading(false);
    }
  }, []);

  const loadTrend = useCallback(async () => {
    const seq = ++trendSeqRef.current;
    setTrendLoading(true);
    setTrendError(false);
    try {
      const data = await getFocusTrend(7);
      if (seq !== trendSeqRef.current) return;
      setTrend(data);
    } catch {
      if (seq !== trendSeqRef.current) return;
      // trend is optional for rendering the session list, but the failure must
      // be reported rather than silently presented as "no data".
      setTrend(null);
      setTrendError(true);
    } finally {
      if (seq === trendSeqRef.current) setTrendLoading(false);
    }
  }, []);

  const refresh = useCallback(() => {
    loadSessions(date);
    loadTrend();
  }, [date, loadSessions, loadTrend]);

  useEffect(() => {
    const sessionRequests = requestSeqRef;
    const trendRequests = trendSeqRef;
    loadSessions(date);
    loadTrend();
    return () => {
      sessionRequests.current++;
      trendRequests.current++;
    };
  }, [date, loadSessions, loadTrend]);

  /** Focus metric set (audit F2).
   *
   *  The backend recognises exactly three `session_type` values — `focus`,
   *  `distraction`, `neutral` (analysis_service.identify_focus_sessions) —
   *  and `/focus/trend` adds only `focus` rows into `focus_min`. The three
   *  duration cards therefore use the same `focus`-only set, so a 120-minute
   *  distraction block can never be reported as the longest focus block or
   *  inflate 专注次数. Distraction/neutral rows are still rendered below with
   *  their feedback controls; nothing is dropped to make the numbers look
   *  better.
   *
   *  平均评分 deliberately keeps averaging **every** session of the day: the
   *  backend's own `avg_score` is `score_sum / count` over all sessions
   *  (api/routes/focus.py), so that card answers "how good was the day",
   *  not "how good were the focused parts". */
  const focusSessions = sessions.filter((session) => session.session_type === "focus");
  const totalFocus = focusSessions.reduce((sum, session) => sum + sessionDurationMinutes(session), 0);
  const sessionCount = focusSessions.length;
  const avgScore =
    sessions.length > 0
      ? sessions.reduce((sum, session) => sum + Number(session.focus_score ?? session.score ?? 0), 0) / sessions.length
      : 0;
  const longestBlock = focusSessions.reduce(
    (maximum, session) => Math.max(maximum, sessionDurationMinutes(session)),
    0,
  );
  const todayKey = todayStr();

  const trendDays = getTrendDays(trend);

  const metricItems: MetricItem[] = [
    sessionsReady
      ? { key: "focus", label: "总专注时长", value: String(Math.round(totalFocus)), unit: "分钟" }
      : { key: "focus", label: "总专注时长", value: "—" },
    sessionsReady
      ? { key: "sessions", label: "专注次数", value: String(sessionCount), unit: "次" }
      : { key: "sessions", label: "专注次数", value: "—" },
    sessionsReady && sessions.length > 0
      ? {
          key: "score",
          label: "平均评分",
          value: avgScore.toFixed(1),
          help: { what: "当天所有会话得分的平均值，满分 100。" },
        }
      : { key: "score", label: "平均评分", value: "—" },
    sessionsReady && longestBlock > 0
      ? { key: "longest", label: "最长专注", value: String(Math.round(longestBlock)), unit: "分钟" }
      : { key: "longest", label: "最长专注", value: "—" },
  ];

  /** Editable draft for a session, seeded from any feedback already recorded.
   *
   *  Seeding from the saved values matters: falling back to a literal
   *  ``mixed``/3 draft meant that opening a previously-rated session and
   *  pressing save again silently replaced the user's original label with
   *  "mixed", destroying the very signal this page collects.
   */
  const draftFor = useCallback((session: FocusSession): FeedbackDraft => {
    const existing = feedbackDrafts[session.id];
    if (existing) return existing;
    const saved = savedFeedback[session.id];
    if (saved) return saved;
    return {
      label: session.feedback_label ?? "mixed",
      score: session.feedback_score ?? 3,
      taskType: session.feedback_task_type ?? "",
    };
  }, [feedbackDrafts, savedFeedback]);

  const saveFeedback = async (session: FocusSession) => {
    const draft = feedbackDrafts[session.id] ?? draftFor(session);
    const sessionId = session.id;
    if (feedbackPendingRef.current.has(sessionId)) return;
    feedbackPendingRef.current.add(sessionId);
    setFeedbackSaving((current) => new Set(current).add(sessionId));
    setFeedbackErrors((current) => ({ ...current, [sessionId]: "" }));
    try {
      await submitFocusFeedback(sessionId, {
        label: draft.label,
        score: draft.score,
        task_type: draft.taskType || undefined,
      });
      setFeedbackSaved((current) => new Set(current).add(sessionId));
      setSavedFeedback((current) => ({ ...current, [sessionId]: draft }));
    } catch (e: unknown) {
      setFeedbackErrors((current) => ({
        ...current,
        [sessionId]: getErrorMessage(e, "反馈保存失败"),
      }));
    } finally {
      feedbackPendingRef.current.delete(sessionId);
      setFeedbackSaving((current) => {
        const next = new Set(current);
        next.delete(sessionId);
        return next;
      });
    }
  };

  return (
    <>
      <div className="f-page">
        <p className="page-lede">按天查看专注与分心的时长，并校准系统对每段时间的判断。</p>

        <div className="page-toolbar f-first-row">
          <div className="f-datepicker mf-picker">
            <NativeDateField label="选择日期" value={date} onChange={setDate} />
          </div>
          <button type="button" className="btn btn-secondary f-reset" onClick={refresh} disabled={loading || trendLoading}>
            刷新
          </button>
        </div>

        {sessionsError && <div className="error-box" role="alert">{sessionsError}</div>}

        <div className="f-state">
          <MetricStrip
            label="当日"
            items={metricItems}
            note="专注评分由系统根据主要应用占比和切换频率自动算出。下方的反馈只用于改进个人模型，可以不填。"
          />
        </div>

        <Surface as="section" className="f-statics">
          <SectionHeader title="近 7 天专注与分心" as="h2" />
          {trendLoading && <div className="spinner" />}
          {!trendLoading && (
            <StackedBar
              data={(trendError ? [] : trendDays).map((d) => ({
                key: d.date,
                label: chartWeekLabel(d.date, todayKey),
                title: `${d.date.slice(5)} ${chartWeekLabel(d.date, todayKey)}`,
                values: {
                  focus: Math.round(d.focus_min ?? 0),
                  distraction: Math.round(d.distraction_min ?? 0),
                },
                highlight: d.date === todayKey,
              }))}
              series={[
                { key: "focus", label: "专注", color: "var(--signal)" },
                { key: "distraction", label: "分心", color: "var(--drift)" },
              ]}
              ariaLabel={`近 7 天每日专注与分心分钟数，共 ${trendDays.length} 天`}
              unit=" 分钟"
              height={220}
              showLegend
              emptyMessage={trendError ? "趋势加载失败，请重试" : "暂无趋势数据"}
            />
          )}
        </Surface>

        <Surface as="section" className="f-conversation">
          <SectionHeader
            title="专注会话"
            as="h2"
            help={{
              what: "会话是一段连续 5 分钟以上的电脑使用。系统会给每段打分，并判断是专注还是分心。",
              effect: "你可以在每段下面告诉系统真实情况，帮它学得更准。",
            }}
          />
          {loading && <div className="spinner" />}
          {!loading && sessionsReady && sessions.length === 0 && (
            <div className="f-empty">暂无专注会话数据</div>
          )}
          {!loading && !sessionsReady && (
            <div className="f-empty">会话加载失败，请重试</div>
          )}
          {!loading && sessions.length > 0 && (
            <div className="conversation-table">
              {sessions.map((session, index) => {
                const sessionId = String(session.id ?? index);
                const startTime = session.start_time ?? session.started_at ?? "";
                const sessionDate = session.date ?? String(startTime).slice(0, 10);
                const duration = sessionDurationMinutes(session);
                const app = session.dominant_app ?? session.main_app ?? session.app ?? session.app_name ?? "—";
                const score = session.focus_score ?? session.score;
                const switches = session.switch_count ?? session.switches ?? 0;
                const draft = draftFor(session);
                const savedDraft = savedFeedback[sessionId];
                const feedbackLabel =
                  savedDraft?.label ??
                  (typeof session.feedback_label === "string" ? session.feedback_label : undefined);
                const feedbackScore =
                  savedDraft?.score ??
                  (typeof session.feedback_score === "number" ? session.feedback_score : undefined);
                // §8.17: the backend already classifies the session; a local
                // score >= 60 guess could disagree with what the server says
                // the session was.
                const sessionTone =
                  session.session_type === "focus"
                    ? { className: "score score--focus", label: "专注" }
                    : session.session_type === "distraction"
                      ? { className: "score score--drift", label: "分心" }
                      : { className: "score score--neutral", label: "一般" };
                return (
                  <div className="conversation-item" key={sessionId}>
                    <div className="statement-table">
                      <div className="left-table">
                        <div className="time-table">
                          <div className="metric-label">
                            {sessionDate ? `${dayLabel(sessionDate)} ${sessionDate.slice(5)}` : "—"}
                          </div>
                          <div className="metric-value">
                            {startTime
                              ? new Date(startTime).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
                              : "—"}
                          </div>
                        </div>
                        <div className="during-table">
                          <div className="metric-label">会话时长</div>
                          <div className="metric-value">{formatMinutes(duration)}</div>
                        </div>
                        <div className="change-table">
                          <div className="metric-label">切换次数</div>
                          <div className="metric-value">{switches} 次</div>
                        </div>
                      </div>
                      <div className="right-table">
                        <div className="process">{app}</div>
                        <div className={sessionTone.className}>
                          {score != null ? `${Math.round(score)} 分 · ${sessionTone.label}` : "—"}
                        </div>
                      </div>
                    </div>

                    {feedbackErrors[sessionId] && (
                      <div className="error-box" role="alert">{feedbackErrors[sessionId]}</div>
                    )}
                    {feedbackSaved.has(sessionId) && (
                      <div className="f-saved">
                        已保存反馈：{feedbackLabel === "focus" ? "专注" : feedbackLabel === "distracted" ? "分心" : "混合"}（
                        {feedbackScore}/5）
                      </div>
                    )}
                    <div className="self-accession-table">
                        <div className="accession-box">
                          <div className="accssion-text">这次状态</div>
                          <select
                            className="selfStatement"
                            aria-label="这次状态"
                            value={draft.label}
                            onChange={(event) =>
                              setFeedbackDrafts((current) => ({
                                ...current,
                                [sessionId]: { ...draft, label: event.target.value as FeedbackDraft["label"] },
                              }))
                            }
                          >
                            <option value="mixed">混合</option>
                            <option value="focus">专注</option>
                            <option value="distracted">分心</option>
                          </select>
                        </div>
                        <div className="accession-box">
                          <div className="accssion-text">自评分数</div>
                          <select
                            className="selfStatement"
                            aria-label="自评分数"
                            value={draft.score}
                            onChange={(event) =>
                              setFeedbackDrafts((current) => ({
                                ...current,
                                [sessionId]: { ...draft, score: Number(event.target.value) },
                              }))
                            }
                          >
                            {[1, 2, 3, 4, 5].map((value) => (
                              <option key={value} value={value}>
                                {value}分
                              </option>
                            ))}
                          </select>
                        </div>
                        <div className="accession-box">
                          <div className="accssion-text">任务类型（可选）</div>
                          <select
                            className="selfStatement"
                            aria-label="任务类型"
                            value={draft.taskType}
                            onChange={(event) =>
                              setFeedbackDrafts((current) => ({
                                ...current,
                                [sessionId]: { ...draft, taskType: event.target.value },
                              }))
                            }
                          >
                            <option value="">未选择</option>
                            <option value="coding">编程</option>
                            <option value="writing">写作</option>
                            <option value="study">学习</option>
                            <option value="meeting">会议</option>
                            <option value="admin">事务</option>
                            <option value="creative">创作</option>
                            <option value="other">其他</option>
                          </select>
                        </div>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm save-buttom"
                          disabled={feedbackSaving.has(sessionId)}
                          onClick={() => saveFeedback(session)}
                        >
                          {feedbackSaving.has(sessionId) ? "保存中…" : "保存反馈"}
                        </button>
                        <HelpTip
                          term="反馈"
                          what="把你对这段时间的真实感受告诉系统。"
                          effect="保存后会用来训练你的个人模型；再次保存会覆盖之前的选择。"
                        />
                    </div>

                    <div className="accession-tip">
                      <div className="accession-tip-text">
                        1–2 分用于分心标签，4–5 分用于专注标签，3 分或混合只用于不确定性评估。
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Surface>
      </div>
    </>
  );
}
