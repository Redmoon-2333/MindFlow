import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../../browser_extension/service_worker.js", import.meta.url), "utf8");

async function harness() {
  const state = {
    now: Date.parse("2026-11-12T04:00:00Z"),
    focused: true,
    token: "synthetic-test-token",
    rules: [],
    heartbeats: [],
    status: 200,
  };
  const event = () => ({ addListener() {} });
  const context = vm.createContext({
    URL, Response, AbortSignal, console,
    navigator: { userAgent: "Chrome" },
    Date: class extends Date { static now() { return state.now; } },
    chrome: {
      storage: {
        local: {
          async get() { return { backendUrl: "http://127.0.0.1:8765", browserToken: state.token }; },
          async remove() { state.token = ""; },
        },
        onChanged: event(),
      },
      tabs: {
        async query() { return [{ url: "https://example.com/lesson?private=discarded" }]; },
        onActivated: event(), onUpdated: event(),
      },
      windows: {
        async getLastFocused() { return { focused: state.focused }; },
        onFocusChanged: event(),
      },
      runtime: { onInstalled: event(), onStartup: event() },
      alarms: { create() {}, onAlarm: event() },
      declarativeNetRequest: {
        async getDynamicRules() { return state.rules; },
        async updateDynamicRules({ addRules = [] }) { state.rules = addRules; },
      },
    },
    async fetch(url, options) {
      if (url.endsWith("/heartbeat")) state.heartbeats.push(JSON.parse(options.body));
      return new Response(JSON.stringify({ domains: [] }), { status: state.status });
    },
  });
  vm.runInContext(source, context);
  const run = code => vm.runInContext(code, context);
  await run("Promise.all([reconcileContext(), syncBlocklist()])");
  state.heartbeats.length = 0;
  return { state, run };
}

test("browser loses focus: flush once, then stop counting background time", async () => {
  const { state, run } = await harness();
  state.now += 30_000;
  state.focused = false;
  await run("reconcileContext()");
  state.now += 30_000;
  await run("reconcileContext({flushCurrent: true})");
  assert.equal(state.heartbeats.length, 1);
  assert.equal(state.heartbeats[0].timestamp_utc, "2026-11-12T04:00:00.000Z");
  assert.equal(state.heartbeats[0].duration_s, 30);
  state.now += 30_000;
  await run("reconcileContext({flushCurrent: true})");
  assert.equal(state.heartbeats.length, 1);
});

test("simultaneous tab and alarm events do not duplicate a dwell interval", async () => {
  const { state, run } = await harness();
  state.now += 30_000;
  await run("Promise.all([reconcileContext({flushCurrent: true}), reconcileContext({flushCurrent: true})])");
  assert.equal(state.heartbeats.length, 1);
  assert.equal(state.heartbeats[0].duration_s, 30);
});

test("a sleep gap is not fabricated as browser dwell", async () => {
  const { state, run } = await harness();
  state.now += 300_000;
  await run("reconcileContext({flushCurrent: true})");
  assert.equal(state.heartbeats.length, 0);
  state.now += 30_000;
  await run("reconcileContext({flushCurrent: true})");
  assert.equal(state.heartbeats[0].timestamp_utc, "2026-11-12T04:05:00.000Z");
  assert.equal(state.heartbeats[0].duration_s, 30);
});

for (const mode of ["missing", "revoked"]) {
  test(`${mode} pairing clears rules left by a previously paired session`, async () => {
    const { state, run } = await harness();
    state.rules = [{ id: 1 }];
    if (mode === "missing") state.token = "";
    else state.status = 401;
    await run("syncBlocklist()");
    assert.equal(state.rules.length, 0);
    assert.equal(state.token, "");
  });
}
