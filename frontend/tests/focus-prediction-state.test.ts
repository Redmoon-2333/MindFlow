/**
 * Pure contract tests for the focus-prediction state mapper (Todo 4).
 *
 * Run with: npx tsx tests/focus-prediction-state.test.ts
 * No test framework: plain node asserts, deterministic, zero clocks.
 *
 * Canonical wire contract (backend `FocusPredictionResponse`):
 *   { focus_probability: number | null, status: ready|no_model|no_data|stale|schema_mismatch|inference_error,
 *     mode: string, reason: string }
 */
import assert from "node:assert/strict";
import { parseFocusPrediction, toFocusPredictionView } from "../src/prediction-state";

let passed = 0;
const failures: string[] = [];

function test(name: string, fn: () => void): void {
  try {
    fn();
    passed += 1;
  } catch (error) {
    failures.push(`${name}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

// ── Ready state: finite probability renders a percentage ──

test("ready 0.731 renders 73.1%", () => {
  const state = parseFocusPrediction({ status: "ready", focus_probability: 0.731, mode: "ready", reason: "" });
  assert.equal(state.status, "ready");
  assert.equal(state.focus_probability, 0.731);
  const view = toFocusPredictionView(state);
  assert.equal(view.display, "73.1%");
  assert.equal(view.ready, true);
  assert.equal(view.statusLabel, "已就绪");
});

test("ready 0.75 (canonical Todo-3 fixture) renders 75.0%", () => {
  const view = toFocusPredictionView(parseFocusPrediction({ status: "ready", focus_probability: 0.75, mode: "ready", reason: "" }));
  assert.equal(view.display, "75.0%");
  assert.equal(view.ready, true);
});

test("ready with empty reason keeps mode available", () => {
  const view = toFocusPredictionView(parseFocusPrediction({ status: "ready", focus_probability: 0.5, mode: "ready", reason: "" }));
  assert.equal(view.reason, "");
  assert.equal(view.mode, "ready");
});

// ── Non-ready states: always -- plus status label and preserved reason ──

const NON_READY_FIXTURES: ReadonlyArray<{ status: string; reason: string; mode: string; label: string }> = [
  { status: "no_data", reason: "在最近 2 小时内未找到 v2 特征窗口", mode: "ready", label: "无数据" },
  { status: "no_model", reason: "未加载 ML 模型，请先训练", mode: "rule_engine_only", label: "无模型" },
  { status: "stale", reason: "数据已过期（1800 秒前最后的窗口，阈值 900 秒）", mode: "ready", label: "数据过期" },
  { status: "schema_mismatch", reason: "模型特征名称与当前 V2_FEATURE_NAMES 不匹配", mode: "rule_engine_only", label: "特征不匹配" },
  { status: "inference_error", reason: "模型推理失败：qa: simulated inference crash", mode: "rule_engine_only", label: "推理错误" },
];

for (const fixture of NON_READY_FIXTURES) {
  test(`${fixture.status} renders -- with label ${fixture.label} and preserved reason`, () => {
    const state = parseFocusPrediction({ status: fixture.status, focus_probability: null, mode: fixture.mode, reason: fixture.reason });
    assert.equal(state.focus_probability, null);
    assert.equal(state.status, fixture.status);
    const view = toFocusPredictionView(state);
    assert.equal(view.display, "--");
    assert.equal(view.ready, false);
    assert.equal(view.statusLabel, fixture.label);
    assert.equal(view.reason, fixture.reason);
    assert.equal(view.mode, fixture.mode);
  });
}

// ── Defensive fixtures: null / NaN / missing / malformed ──

test("ready with null probability degrades to -- not a percentage", () => {
  const state = parseFocusPrediction({ status: "ready", focus_probability: null, mode: "ready", reason: "" });
  assert.equal(state.focus_probability, null);
  const view = toFocusPredictionView(state);
  assert.equal(view.display, "--");
  assert.equal(view.ready, false);
});

test("ready with NaN probability degrades to -- (NaN defense)", () => {
  const state = parseFocusPrediction({ status: "ready", focus_probability: Number.NaN, mode: "ready", reason: "" });
  assert.equal(state.focus_probability, null);
  const view = toFocusPredictionView(state);
  assert.equal(view.display, "--");
  assert.notEqual(view.display, "NaN%");
});

test("ready with missing probability degrades to --", () => {
  const state = parseFocusPrediction({ status: "ready", mode: "ready", reason: "" });
  assert.equal(state.focus_probability, null);
  assert.equal(toFocusPredictionView(state).display, "--");
});

test("ready with Infinity probability degrades to --", () => {
  const state = parseFocusPrediction({ status: "ready", focus_probability: Number.POSITIVE_INFINITY, mode: "ready", reason: "" });
  assert.equal(toFocusPredictionView(state).display, "--");
});

test("ready with string probability is not trusted", () => {
  const state = parseFocusPrediction({ status: "ready", focus_probability: "0.5", mode: "ready", reason: "" });
  assert.equal(state.focus_probability, null);
  assert.equal(toFocusPredictionView(state).display, "--");
});

test("non-ready with numeric probability is normalized to null (never a percentage)", () => {
  const state = parseFocusPrediction({ status: "no_data", focus_probability: 0.9, mode: "ready", reason: "无数据" });
  assert.equal(state.focus_probability, null);
  assert.equal(toFocusPredictionView(state).display, "--");
});

// ── Unknown / untrusted statuses: explicit unknown state, never a percentage ──

test("unrecognized status renders unknown state with -- not a percentage", () => {
  const state = parseFocusPrediction({ status: "bogus", focus_probability: 0.5, mode: "ready", reason: "自定义原因" });
  assert.equal(state.status, "unknown");
  assert.equal(state.focus_probability, null);
  const view = toFocusPredictionView(state);
  assert.equal(view.display, "--");
  assert.equal(view.ready, false);
  assert.equal(view.statusLabel, "未知状态");
  assert.equal(view.reason, "自定义原因");
});

test("missing status renders unknown state", () => {
  const state = parseFocusPrediction({ focus_probability: 0.5, mode: "ready", reason: "" });
  assert.equal(state.status, "unknown");
  assert.equal(toFocusPredictionView(state).display, "--");
});

test("empty-string status renders unknown state", () => {
  const state = parseFocusPrediction({ status: "", focus_probability: null, mode: "ready", reason: "" });
  assert.equal(state.status, "unknown");
  assert.equal(toFocusPredictionView(state).display, "--");
});

test("non-object payload renders unknown state safely", () => {
  for (const value of [null, undefined, "ready", 42, [], true]) {
    const state = parseFocusPrediction(value);
    assert.equal(state.status, "unknown");
    assert.equal(toFocusPredictionView(state).display, "--");
  }
});

test("unknown status without reason surfaces raw status text", () => {
  const state = parseFocusPrediction({ status: "mystery_status", mode: "ready", reason: "" });
  const view = toFocusPredictionView(state);
  assert.equal(view.statusLabel, "未知状态");
  assert.ok(view.reason.includes("mystery_status"), `reason should carry raw status, got ${view.reason}`);
});

// ── Report ──

if (failures.length > 0) {
  console.error(`\n${failures.length} test(s) FAILED:`);
  for (const failure of failures) console.error(`  - ${failure}`);
  console.error(`${passed} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`focus-prediction-state: ${passed} tests passed`);
