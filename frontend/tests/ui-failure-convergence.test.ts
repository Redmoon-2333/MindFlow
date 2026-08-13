import assert from "node:assert/strict";
import {
  ApiError,
  getErrorMessage,
  isAutonomyPaused,
  runAttribution,
} from "../src/api.ts";

const now = Date.parse("2026-08-06T00:00:00.000Z");

assert.equal(isAutonomyPaused(null, now), false);
assert.equal(
  isAutonomyPaused({ enabled: false, paused_until: null, paused: true }, now),
  true,
);
assert.equal(
  isAutonomyPaused({ enabled: false, paused_until: "2026-08-06T01:00:00.000Z" }, now),
  true,
);
assert.equal(
  isAutonomyPaused({ enabled: false, paused_until: "2026-08-06T01:00:00" }, now),
  true,
);
assert.equal(
  isAutonomyPaused({ enabled: true, paused_until: "2026-08-05T23:00:00.000Z" }, now),
  false,
);
assert.equal(
  isAutonomyPaused({ enabled: false, paused_until: "not-a-timestamp" }, now),
  true,
);
assert.equal(
  isAutonomyPaused({ enabled: false, paused_until: null }, now),
  false,
);

assert.equal(
  getErrorMessage(new ApiError("请求超时，请稍后重试", 408), "归因分析失败"),
  "归因分析失败：请求超时，请稍后重试",
);

const originalFetch = globalThis.fetch;
const originalSetTimeout = globalThis.setTimeout;
let signalWasAborted = false;
let timeoutDelay = 0;

try {
  globalThis.setTimeout = ((handler: TimerHandler, delay?: number) => {
    timeoutDelay = delay ?? 0;
    if (typeof handler === "function") Reflect.apply(handler, undefined, []);
    return 0;
  }) as typeof globalThis.setTimeout;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    signalWasAborted = init?.signal?.aborted === true;
    throw new Error("request interrupted");
  }) as typeof fetch;

  await assert.rejects(
    runAttribution(),
    (error: unknown) => error instanceof ApiError && error.status === 408,
  );
  assert.equal(signalWasAborted, true);
  assert.equal(timeoutDelay, 90_000);
} finally {
  globalThis.fetch = originalFetch;
  globalThis.setTimeout = originalSetTimeout;
}

console.log("ui-failure-convergence: 10 tests passed");
