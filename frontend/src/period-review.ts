/**
 * Period review ("阶段总结") — wire types + one pure view mapper.
 *
 * The backend already sends everything the page needs in a closed whitelist of
 * numbers, categories and dates (no titles, URLs or reminder text), so the
 * mapper only has to turn numbers into display strings and decide which blocks
 * have anything to show. It is pure: no fetch, no DOM, no `Date`.
 *
 * Invariant: `hasData` is true only for a `ready` digest, and every chart
 * series is empty unless `hasData` — so an `insufficient`/`no_model` review
 * renders the explanation and no half-populated charts.
 */

import type { components } from "./generated/api-schema";
import { formatMinutes } from "./report-view";

export type PeriodReview = components["schemas"]["PeriodReviewResponse"];
export type PeriodDigest = components["schemas"]["PeriodDigest"];
export type PeriodNarrative = components["schemas"]["Narrative"];
export type PeriodDays = 3 | 7 | 14;

/** Unrecognised apps/sites the review could not place (appendix A). */
export type UnclassifiedItems = components["schemas"]["UnclassifiedItemsResponse"];
export type UnclassifiedItem = components["schemas"]["UnclassifiedItemResponse"];
export type AssignableKind = UnclassifiedItem["kind"];
export type AssignableContext = components["schemas"]["TaskContextAssignRequest"]["context"];
export type TaskContextAssignResponse = components["schemas"]["TaskContextAssignResponse"];

/** The answers offered for "这是……". ``unknown`` is not one of them: the list
 *  only contains items that are already unknown, so it would be a no-op. */
export const ASSIGNABLE_CONTEXTS: ReadonlyArray<{ value: AssignableContext; label: string }> = [
  { value: "coding", label: "编程" },
  { value: "writing", label: "写作与文档" },
  { value: "reading", label: "查资料与阅读" },
  { value: "meeting", label: "沟通与会议" },
  { value: "entertainment", label: "娱乐" },
  { value: "social", label: "社交" },
];

export interface UnclassifiedRowView {
  kind: AssignableKind;
  /** Stored key — the process name or bare host. */
  key: string;
  display: string;
  /** "应用" / "网站" — the small prefix that says what the name is. */
  kindLabel: string;
  minutesText: string;
}

/** Listing rows as the card renders them; an empty/missing list is empty. */
export function toUnclassifiedRows(
  response: UnclassifiedItems | null | undefined,
): UnclassifiedRowView[] {
  return series(response?.items).map((item) => ({
    kind: item.kind,
    key: item.key,
    display: item.display,
    kindLabel: item.kind === "site" ? "网站" : "应用",
    minutesText: `${item.minutes} 分钟`,
  }));
}

export interface PeriodChartDatum {
  key: string;
  label: string;
  value: number;
}

export interface PeriodFindingView {
  text: string;
  chips: string[];
}

export interface PeriodSuggestionView {
  text: string;
}

export interface PeriodDaypartView {
  key: string;
  label: string;
  text: string;
}

export interface PeriodSignalView {
  key: string;
  label: string;
  direction: string;
  text: string;
}

export interface PeriodMetricView {
  key: string;
  label: string;
  value: string;
  unit?: string;
  help?: { what: string };
}

export interface PeriodReviewView {
  days: PeriodDays;
  hasData: boolean;
  headline: string;
  source: PeriodNarrative["source"];
  fallbackNote: string | null;
  findings: PeriodFindingView[];
  suggestions: PeriodSuggestionView[];
  papers: string[];
  metrics: PeriodMetricView[];
  workMix: PeriodChartDatum[];
  distractionByContext: PeriodChartDatum[];
  daily: PeriodChartDatum[];
  dayparts: PeriodDaypartView[];
  signals: PeriodSignalView[];
  interventionsLine: string;
  interventionChips: string[];
  payloadNote: string;
  payloadPretty: string;
}

/**
 * Distracted vs focused mean as a unit-free comparison ("只有专注时的 31%").
 * Raw feature values (scroll counts, ratios) mean nothing to the user.
 */
export function relativeText(distracted: number, focused: number): string {
  if (!(focused > 0) || !Number.isFinite(distracted)) return "";
  const ratio = distracted / focused;
  if (ratio >= 1.5) return `是专注时的 ${ratio.toFixed(1)} 倍`;
  if (ratio > 1) return `比专注时高 ${Math.round((ratio - 1) * 100)}%`;
  return `只有专注时的 ${Math.round(ratio * 100)}%`;
}

const DIRECTION_ZH: Record<string, string> = {
  higher: "偏高",
  lower: "偏低",
  similar: "差别不大",
};

/** Minutes as the digest reports them; a missing series is simply empty. */
function series<T>(value: T[] | undefined): T[] {
  return Array.isArray(value) ? value : [];
}

/** M/D for a YYYY-MM-DD string — no Date, so no timezone shift. */
export function shortDay(iso: string): string {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!match) return iso;
  return `${Number(match[2])}/${Number(match[3])}`;
}

export function toPeriodReviewView(
  review: PeriodReview,
  days: PeriodDays,
  useAi: boolean,
): PeriodReviewView {
  const digest = review.digest;
  const narrative = review.narrative;
  const hasData = digest.data_state === "ready";

  const metrics: PeriodMetricView[] = hasData
    ? [
        { key: "active", label: "有效记录", value: formatMinutes(digest.totals.active_minutes) },
        {
          key: "episodes",
          label: "分心次数",
          value: String(digest.episodes.count),
          unit: "次",
          help: { what: "模型把每 5 分钟判断一次，连在一起的分心片段算一次。" },
        },
        {
          key: "share",
          label: "分心占比",
          value: String(Math.round(digest.totals.distracted_share * 100)),
          unit: "%",
        },
        {
          key: "longest",
          label: "最长一次分心",
          value: String(digest.episodes.longest_minutes),
          unit: "分钟",
        },
      ]
    : [];

  const workMix: PeriodChartDatum[] = hasData
    ? series(digest.work_mix).map((item) => ({
        key: item.context,
        label: item.label,
        value: item.minutes,
      }))
    : [];

  const distractionByContext: PeriodChartDatum[] = hasData
    ? series(digest.distraction_by_context).map((item) => ({
        key: item.context,
        label: item.label,
        value: item.minutes,
      }))
    : [];

  const daily: PeriodChartDatum[] = hasData
    ? series(digest.daily).map((day) => ({
        key: day.date,
        label: shortDay(day.date),
        value: day.distracted_minutes,
      }))
    : [];

  const dayparts: PeriodDaypartView[] = hasData
    ? series(digest.distraction_by_daypart).map((stat) => ({
        key: stat.daypart,
        label: stat.label,
        text: `分心 ${stat.episodes} 次 · ${stat.distracted_minutes} 分钟`,
      }))
    : [];

  const signals: PeriodSignalView[] = hasData
    ? series(digest.model_signals).map((signal) => ({
        key: signal.feature,
        label: signal.label,
        direction: DIRECTION_ZH[signal.direction] ?? "差别不大",
        text: relativeText(signal.distracted_mean, signal.focused_mean),
      }))
    : [];

  const interventions = digest.interventions;
  const interventionParts = [
    `试试看 ${interventions.accepted}`,
    `先忽略 ${interventions.ignored}`,
    `不适合我 ${interventions.dismissed}`,
  ];
  if (interventions.pending > 0) interventionParts.push(`未回应 ${interventions.pending}`);

  return {
    days,
    hasData,
    headline: narrative.headline,
    source: narrative.source,
    fallbackNote:
      narrative.fallback_reason != null
        ? `${narrative.fallback_reason}，已改用本地规则总结。`
        : null,
    findings: series(narrative.findings).map((finding) => ({
      text: finding.text,
      // The card has room for three facts per sentence; the rest stay in the
      // digest payload below.
      chips: series(finding.evidence)
        .map((ref) => ref.label)
        .slice(0, 3),
    })),
    suggestions: series(narrative.suggestions).map((suggestion) => ({
      text: suggestion.text,
    })),
    papers: series(narrative.papers).map((paper) =>
      paper.pages.length > 0
        ? `${paper.title}（第${paper.pages.join("、")}页）`
        : paper.title,
    ),
    metrics,
    workMix,
    distractionByContext,
    daily,
    dayparts,
    signals,
    interventionsLine:
      interventions.total > 0
        ? `共提醒 ${interventions.total} 次：${interventionParts.join(" · ")}`
        : "这段时间没有触发提醒。",
    interventionChips: series(interventions.by_reason).map(
      (reason) => `${reason.label} ${reason.count}`,
    ),
    payloadNote: useAi
      ? "这是本次发送的全部数据：只有统计数字和类别，没有窗口标题、网址、文件名和提醒正文。"
      : "这是可发送的全部数据（本次没有发送）：只有统计数字和类别，没有窗口标题、网址、文件名和提醒正文。",
    // The backend sends compact JSON; pretty-print it so the payload is
    // readable. A malformed string is shown as-is rather than throwing.
    payloadPretty: prettyJson(review.llm_payload),
  };
}

/** Pretty-print a JSON string, falling back to the raw text. */
export function prettyJson(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}
