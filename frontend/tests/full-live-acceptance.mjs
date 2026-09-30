// Complements the ordinary matrix with real lifecycle and persistence checks.
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const args = process.argv.slice(2);
const option = name => {
  const index = args.indexOf(`--${name}`);
  return index < 0 ? undefined : args[index + 1];
};
const base = option("base");
const backend = option("backend");
const dataDir = path.resolve(option("data-dir") ?? "");
const out = path.resolve(option("out") ?? "../docs/audit/20260930-full");
assert.equal(base, "http://127.0.0.1:4173");
assert.equal(backend, "http://127.0.0.1:8866");
assert.ok(dataDir.startsWith(`${path.resolve(os.tmpdir())}${path.sep}`));
assert.ok(path.basename(dataDir).startsWith("mindflow-full-"));
const rootToken = (await readFile(path.join(dataDir, "token"), "utf8")).trim();
const results = [];
let cookie = "";
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function call(method, route, body, { anonymous = false, headers = {}, origin = base } = {}) {
  const response = await fetch(`${origin}${route}`, {
    method,
    headers: {
      ...(!anonymous && cookie ? { Cookie: cookie } : {}),
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(180_000),
  });
  const text = await response.text();
  let json;
  try { json = JSON.parse(text); } catch { json = undefined; }
  return { status: response.status, json, text, headers: response.headers };
}

async function check(name, fn) {
  try {
    const detail = await fn();
    results.push({ name, passed: true, detail });
    console.log(`PASS ${name}`);
  } catch (error) {
    results.push({ name, passed: false, detail: error.message });
    console.log(`FAIL ${name}: ${error.message}`);
  }
}

async function waitUntil(fn, timeoutMs = 30_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const result = await fn();
    if (result) return result;
    await sleep(250);
  }
  throw new Error("Timed out waiting for the required persisted state");
}

const ticket = await call("POST", "/api/v1/auth/bootstrap/ticket", undefined, {
  origin: backend, anonymous: true, headers: { Authorization: `Bearer ${rootToken}` },
});
assert.equal(ticket.status, 200);
const login = await call("POST", "/api/v1/auth/bootstrap", { ticket: ticket.json.ticket }, {
  origin: backend, anonymous: true,
});
assert.equal(login.status, 204);
cookie = login.headers.get("set-cookie").match(/mindflow_session=[^;]+/)[0];
// An in-memory session minted by the known isolated backend must work via Vite.
assert.equal((await call("GET", "/api/v1/preferences")).status, 200);
const provider = await call("GET", "/api/v1/ai/provider-status");
assert.equal(provider.json.configured, false);
assert.equal(provider.json.ollama_enabled, false);

try {
  await check("native collector start, idempotent start, persistence, and stop", async () => {
    await call("POST", "/api/v1/collector/stop");
    const before = await call("GET", "/api/v1/activities?page=1&page_size=1");
    try {
      const started = await call("POST", "/api/v1/collector");
      assert.equal(started.status, 200);
      assert.equal(started.json.running, true);
      const repeated = await call("POST", "/api/v1/collector");
      assert.equal(repeated.json.running, true);
      const after = await waitUntil(async () => {
        const response = await call("GET", "/api/v1/activities?page=1&page_size=1");
        return response.status === 200 && response.json.total > before.json.total
          ? response.json.total : false;
      });
      return { persistedNewEvents: after - before.json.total };
    } finally {
      const stopped = await call("POST", "/api/v1/collector/stop");
      assert.equal(stopped.status, 200);
      assert.equal((await call("GET", "/api/v1/collector")).json.running, false);
    }
  });

  await check("bulk classification replacement and exact restoration", async () => {
    const old = await call("GET", "/api/v1/app-classifications");
    assert.equal(old.status, 200);
    const fields = item => ({
      process_name: item.process_name, window_title_pattern: item.window_title_pattern,
      category: item.category, priority: item.priority,
    });
    const rules = old.json.map(fields);
    const added = { process_name: "release-acceptance.exe", window_title_pattern: null, category: "other", priority: 3 };
    try {
      const response = await call("PUT", "/api/v1/app-classifications", [...rules, added]);
      assert.equal(response.status, 200);
      const read = await call("GET", "/api/v1/app-classifications");
      assert.ok(read.json.some(item => item.process_name === added.process_name));
    } finally {
      assert.equal((await call("PUT", "/api/v1/app-classifications", rules)).status, 200);
      assert.deepEqual((await call("GET", "/api/v1/app-classifications")).json.map(fields), rules);
    }
  });

  await check("intervention dispatch, response, feedback, and history read-back", async () => {
    const triggered = await call("POST", "/api/v1/intervention/trigger", { intensity: "gentle" });
    assert.equal(triggered.status, 200);
    assert.equal(triggered.json.skipped, false);
    const id = triggered.json.intervention.id;
    const response = await call("POST", `/api/v1/intervention/${id}/response`, {
      response: "accepted", latency_s: 1.0, source: "human",
    });
    assert.equal(response.status, 200);
    const feedback = await call("POST", `/api/v1/intervention/${id}/feedback`, {
      rating: "helpful", comment: "synthetic acceptance feedback",
    });
    assert.equal(feedback.status, 200);
    const history = await call("GET", "/api/v1/intervention/history?days=7");
    const saved = history.json.items.find(item => item.id === id);
    assert.equal(saved.user_response, "accepted");
    assert.equal(saved.feedback_rating, "helpful");
    return { responsePersisted: true, feedbackPersisted: true };
  });

  await check("chat response and same-session history persistence", async () => {
    const sent = await call("POST", "/api/v1/chat", { message: "synthetic acceptance greeting" });
    assert.equal(sent.status, 200);
    assert.ok(sent.json.answer.trim().length > 0);
    const id = sent.json.session_id;
    assert.ok(id);
    const history = await call("GET", `/api/v1/chat/${id}/messages`);
    assert.equal(history.status, 200);
    assert.ok(history.json.some(item => item.role === "user"));
    assert.ok(history.json.some(item => item.role === "assistant"));
    const sessions = await call("GET", "/api/v1/chat/sessions");
    assert.ok(sessions.json.some(item => item.session_id === id || item.id === id));
    return { messages: history.json.length, onlineInference: false };
  });

  await check("attribution, panel, and diagnostic run details", async () => {
    const attribution = await call("POST", "/api/v1/analytics/attribution", {});
    assert.equal(attribution.status, 200);
    assert.ok(attribution.json.assessment);
    const panel = await call("POST", "/api/v1/panel/today", { force: true });
    assert.equal(panel.status, 200);
    assert.equal((await call("GET", "/api/v1/panel")).status, 200);
    const runs = await call("GET", "/api/v1/ai/runs?limit=10");
    assert.equal(runs.status, 200);
    assert.ok(runs.json.items.length > 0);
    const id = runs.json.items[0].run_id;
    assert.ok(id);
    const detail = await call("GET", `/api/v1/ai/runs/${id}`);
    assert.equal(detail.status, 200);
    return { diagnosticDetailAvailable: true, onlineInference: false };
  });

  await check("real training start, duplicate exclusion, cancel semantics, and terminal persistence", async () => {
    const readiness = await call("GET", "/api/v1/analytics/training-readiness");
    assert.equal(readiness.status, 200);
    assert.equal(readiness.json.trainable, true);
    let created = await call("POST", "/api/v1/analytics/training-jobs");
    assert.equal(created.status, 202);
    assert.equal((await call("POST", "/api/v1/analytics/training-jobs")).status, 409);
    const cancelled = await call("POST", `/api/v1/analytics/training-jobs/${created.json.job_id}/cancel`);
    assert.ok([200, 409].includes(cancelled.status));
    if (cancelled.status === 200) {
      assert.equal(cancelled.json.status, "cancelled");
      const saved = await call("GET", `/api/v1/analytics/training-jobs/${created.json.job_id}`);
      assert.equal(saved.json.status, "cancelled");
      created = await call("POST", "/api/v1/analytics/training-jobs");
      assert.equal(created.status, 202);
    } else {
      assert.equal(cancelled.json.status, 409);
    }
    const terminal = await waitUntil(async () => {
      const response = await call("GET", `/api/v1/analytics/training-jobs/${created.json.job_id}`);
      assert.equal(response.status, 200);
      return ["succeeded", "failed", "cancelled", "interrupted"].includes(response.json.status)
        ? response.json : false;
    }, 240_000);
    assert.equal(terminal.status, "succeeded");
    assert.equal(terminal.feature_schema_version, 4);
    assert.ok(terminal.version_tag);
    const report = JSON.parse(await readFile(
      path.join(dataDir, "models", "v2", `training_report-${terminal.version_tag}.json`), "utf8",
    ));
    assert.equal(report.version_tag, terminal.version_tag);
    return { terminal: terminal.status, modelMode: terminal.model_mode, cancelStatus: cancelled.status };
  });

  let browserToken;
  await check("browser pairing, authenticated heartbeat, and blocklist", async () => {
    assert.equal((await call("PATCH", "/api/v1/telemetry/preferences", { browser_tracking_enabled: true })).status, 200);
    const code = await call("POST", "/api/v1/telemetry/browser/pairing-code");
    assert.equal(code.status, 200);
    const paired = await call("POST", "/api/v1/telemetry/browser/pair", { code: code.json.code }, { anonymous: true });
    assert.equal(paired.status, 200);
    browserToken = paired.json.token;
    assert.ok(browserToken);
    const headers = { "X-Browser-Token": browserToken };
    const heartbeat = await call("POST", "/api/v1/telemetry/browser/heartbeat", {
      timestamp_utc: new Date().toISOString(), duration_s: 5, browser_name: "acceptance",
      domain: "https://acceptance.example/synthetic", audible: false, incognito: false,
    }, { anonymous: true, headers });
    assert.equal(heartbeat.status, 200);
    const blocklist = await call("GET", "/api/v1/telemetry/browser/blocklist", undefined, { anonymous: true, headers });
    assert.equal(blocklist.status, 200);
    assert.ok(Array.isArray(blocklist.json.domains));
    return { paired: true, heartbeatAccepted: true };
  });

  await check("CSV and JSON exports contain actual records", async () => {
    const json = await call("GET", "/api/v1/export?fmt=json");
    assert.equal(json.status, 200);
    assert.ok(json.text.length > 10);
    const csv = await call("GET", "/api/v1/export?fmt=csv");
    assert.equal(csv.status, 200);
    assert.ok(csv.text.includes("\n"));
    return { jsonNonempty: true, csvNonempty: true };
  });

  await check("all telemetry deletion scopes and browser-token revocation", async () => {
    assert.ok(browserToken, "Pairing must have passed before checking revocation");
    for (const scope of ["interaction", "browser", "feedback", "all"]) {
      const deleted = await call("DELETE", `/api/v1/telemetry/data?scope=${scope}`);
      assert.equal(deleted.status, 200);
      assert.equal(deleted.json.partial ?? false, false);
      assert.ok(Number.isInteger(deleted.json.deleted));
      if (scope === "browser") {
        const revoked = await call("GET", "/api/v1/telemetry/browser/blocklist", undefined, {
          anonymous: true, headers: { "X-Browser-Token": browserToken },
        });
        assert.equal(revoked.status, 401);
      }
    }
    const readiness = await call("GET", "/api/v1/analytics/training-readiness");
    assert.equal(readiness.status, 200);
    assert.equal(readiness.json.trainable, false);
    return { scopes: 4, pairedTokenRevoked: true, trainingDataCleared: true };
  });
} finally {
  await call("POST", "/api/v1/collector/stop").catch(() => {});
  await mkdir(out, { recursive: true });
  await writeFile(path.join(out, "live-functional.json"), JSON.stringify({
    generatedAt: new Date().toISOString(), base, backend,
    passed: results.filter(item => item.passed).length,
    failed: results.filter(item => !item.passed).length,
    results,
  }, null, 2));
}
process.exitCode = results.some(item => !item.passed) ? 1 : 0;
