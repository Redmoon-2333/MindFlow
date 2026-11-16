/**
 * Period review view-model contract.
 *
 * Pins how the backend digest/narrative is turned into what the 阶段总结 tab
 * shows: metric cells, chart series, chips, fallback note and payload text.
 *
 * Run: npx tsx tests/period-review.test.ts
 */
import assert from "node:assert/strict";
import {
  ASSIGNABLE_CONTEXTS,
  prettyJson,
  relativeText,
  shortDay,
  toPeriodReviewView,
  toUnclassifiedRows,
} from "../src/period-review";
import type { PeriodReview } from "../src/period-review";

function review(overrides: Partial<PeriodReview> = {}): PeriodReview {
  const digest = {
    period: { start_date: "2026-11-10", end_date: "2026-11-16", days: 7 },
    data_state: "ready",
    model: { version: "base_v4_20261111_r2", is_base_model: true },
    totals: {
      active_minutes: 600,
      focused_minutes: 480,
      distracted_minutes: 120,
      distracted_share: 0.2,
      days_with_data: 5,
    },
    episodes: { count: 6, avg_minutes: 20, longest_minutes: 45 },
    work_mix: [
      { context: "coding", label: "编程", minutes: 400, share: 0.667 },
      { context: "entertainment", label: "娱乐", minutes: 200, share: 0.333 },
    ],
    distraction_by_context: [
      { context: "entertainment", label: "娱乐", minutes: 120, share: 1 },
    ],
    distraction_by_daypart: [
      { daypart: "morning", label: "上午", episodes: 1, distracted_minutes: 10 },
      { daypart: "afternoon", label: "下午", episodes: 4, distracted_minutes: 90 },
      { daypart: "evening", label: "晚上", episodes: 1, distracted_minutes: 20 },
      { daypart: "night", label: "深夜", episodes: 0, distracted_minutes: 0 },
    ],
    daily: [
      {
        date: "2026-11-14",
        active_minutes: 300,
        distracted_minutes: 60,
        episodes: 3,
        top_context: "coding",
        top_context_label: "编程",
      },
    ],
    model_signals: [
      {
        feature: "app_switch_count",
        label: "应用切换次数",
        direction: "higher",
        distracted_mean: 9.2,
        focused_mean: 3.1,
      },
    ],
    interventions: {
      total: 3,
      accepted: 1,
      ignored: 1,
      dismissed: 1,
      pending: 0,
      by_reason: [{ type: "impulsivity", label: "冲动分心", count: 2 }],
    },
  };
  const narrative = {
    source: "ai",
    headline: "最近7天分心 6 次",
    findings: [
      {
        text: "分心多在下午。",
        evidence: [
          { id: "a", label: "A" },
          { id: "b", label: "B" },
          { id: "c", label: "C" },
          { id: "d", label: "D" },
        ],
      },
    ],
    suggestions: [{ text: "下午先定小目标。", paper_ids: ["p1"] }],
    papers: [{ chunk_id: "p1", title: "Self-control in cyberspace", pages: [10] }],
    fallback_reason: null,
  };
  return {
    digest,
    llm_payload: JSON.stringify(digest),
    ai_requested: true,
    narrative,
    generated_at: "2026-11-16T08:00:00+00:00",
    ...overrides,
  } as unknown as PeriodReview;
}

// ── shortDay ────────────────────────────────────────────────────────────────
assert.equal(shortDay("2026-11-04"), "11/4");
assert.equal(shortDay("not-a-date"), "not-a-date");

// ── ready digest → full view ────────────────────────────────────────────────
{
  const view = toPeriodReviewView(review(), 7, true);
  assert.equal(view.hasData, true);
  assert.deepEqual(
    view.metrics.map((m) => [m.label, m.value, m.unit ?? ""]),
    [
      ["有效记录", view.metrics[0].value, ""],
      ["分心次数", "6", "次"],
      ["分心占比", "20", "%"],
      ["最长一次分心", "45", "分钟"],
    ],
  );
  assert.deepEqual(view.workMix.map((d) => [d.label, d.value]), [["编程", 400], ["娱乐", 200]]);
  assert.deepEqual(view.distractionByContext.map((d) => d.label), ["娱乐"]);
  assert.deepEqual(view.daily, [{ key: "2026-11-14", label: "11/14", value: 60 }]);
  assert.equal(view.dayparts[1].text, "分心 4 次 · 90 分钟");
  assert.equal(view.signals[0].direction, "偏高");
  assert.equal(view.signals[0].text, "是专注时的 3.0 倍");
  // At most three evidence chips per finding.
  assert.deepEqual(view.findings[0].chips, ["A", "B", "C"]);
  assert.deepEqual(view.papers, ["Self-control in cyberspace（第10页）"]);
  assert.equal(view.interventionsLine, "共提醒 3 次：试试看 1 · 先忽略 1 · 不适合我 1");
  assert.deepEqual(view.interventionChips, ["冲动分心 2"]);
  assert.equal(view.fallbackNote, null);
  assert.match(view.payloadNote, /^这是本次发送的全部数据/);
  assert.ok(view.payloadPretty.includes("\n  \"period\""));
}

// ── template fallback + not sent ────────────────────────────────────────────
{
  const base = review();
  const view = toPeriodReviewView(
    review({
      ai_requested: false,
      narrative: { ...base.narrative, source: "template", fallback_reason: "AI 响应超时" },
    } as Partial<PeriodReview>),
    3,
    false,
  );
  assert.equal(view.source, "template");
  assert.equal(view.fallbackNote, "AI 响应超时，已改用本地规则总结。");
  assert.match(view.payloadNote, /本次没有发送/);
}

// ── insufficient data → no charts, no metrics ───────────────────────────────
{
  const base = review();
  const view = toPeriodReviewView(
    review({ digest: { ...base.digest, data_state: "insufficient" } } as Partial<PeriodReview>),
    14,
    true,
  );
  assert.equal(view.hasData, false);
  assert.equal(view.metrics.length, 0);
  assert.equal(view.workMix.length, 0);
  assert.equal(view.daily.length, 0);
}

// ── no reminders ────────────────────────────────────────────────────────────
{
  const base = review();
  const view = toPeriodReviewView(
    review({
      digest: {
        ...base.digest,
        interventions: { total: 0, accepted: 0, ignored: 0, dismissed: 0, pending: 0, by_reason: [] },
      },
    } as Partial<PeriodReview>),
    7,
    true,
  );
  assert.equal(view.interventionsLine, "这段时间没有触发提醒。");
}

// ── prettyJson ──────────────────────────────────────────────────────────────
assert.equal(prettyJson('{"a":1}'), '{\n  "a": 1\n}');
assert.equal(prettyJson("not json"), "not json");

// ── unclassified rows (appendix A) ──────────────────────────────────────────
{
  const rows = toUnclassifiedRows({
    items: [
      { kind: "app", key: "MysteryTool.exe", display: "MysteryTool", minutes: 42 },
      { kind: "site", key: "unknown.example", display: "unknown.example", minutes: 7 },
    ],
  });
  assert.deepEqual(rows, [
    {
      kind: "app",
      key: "MysteryTool.exe",
      display: "MysteryTool",
      kindLabel: "应用",
      minutesText: "42 分钟",
    },
    {
      kind: "site",
      key: "unknown.example",
      display: "unknown.example",
      kindLabel: "网站",
      minutesText: "7 分钟",
    },
  ]);
}

// An empty or missing listing renders no rows at all.
assert.deepEqual(toUnclassifiedRows({ items: [] }), []);
assert.deepEqual(toUnclassifiedRows(null), []);

// The answers offered never include "unknown": every listed item already is.
assert.equal(ASSIGNABLE_CONTEXTS.length, 6);
assert.ok(!ASSIGNABLE_CONTEXTS.some((option) => (option.value as string) === "unknown"));
assert.deepEqual(
  ASSIGNABLE_CONTEXTS.map((option) => option.label),
  ["编程", "写作与文档", "查资料与阅读", "沟通与会议", "娱乐", "社交"],
);

console.log("period-review: all assertions passed");

// ── relativeText: unit-free comparison, never raw feature values ───────────
assert.equal(relativeText(370.777, 1195.925), "只有专注时的 31%");
assert.equal(relativeText(0.6, 0.5), "比专注时高 20%");
assert.equal(relativeText(1, 0), "");
console.log("relativeText: all assertions passed");
