import assert from "node:assert/strict";
import { after, test } from "node:test";

const nativeFetch = globalThis.fetch;
const NativeRequest = globalThis.Request;
const nativeSetTimeout = globalThis.setTimeout;
const nativeClearTimeout = globalThis.clearTimeout;
let mode: "success" | "failure" | "stall" = "success";
let requestSignal: AbortSignal | undefined;

globalThis.Request = class extends NativeRequest {
  constructor(input: RequestInfo | URL, init?: RequestInit) {
    super(typeof input === "string" ? new URL(input, "http://127.0.0.1") : input, init);
  }
};
globalThis.fetch = async (input, init) => {
  requestSignal = init?.signal ?? (input instanceof NativeRequest ? input.signal : undefined);
  if (mode === "stall") {
    return new Promise<Response>((_resolve, reject) => {
      const abort = () => reject(new DOMException("Aborted", "AbortError"));
      requestSignal?.addEventListener("abort", abort, { once: true });
      if (requestSignal?.aborted) abort();
    });
  }
  return new Response(JSON.stringify(mode === "failure" ? { detail: "Unavailable" } : []), {
    status: mode === "failure" ? 503 : 200,
    headers: { "Content-Type": "application/json" },
  });
};

const { ApiError, getChatSessions, getCurrentActivity } = await import("../src/api.ts");

after(() => {
  globalThis.fetch = nativeFetch;
  globalThis.Request = NativeRequest;
});

for (const outcome of ["success", "failure"] as const) {
  test(`typed request clears its deadline after ${outcome}`, async () => {
    mode = outcome;
    const pending = new Set<ReturnType<typeof setTimeout>>();
    globalThis.setTimeout = ((callback: (...args: unknown[]) => void, delay?: number) => {
      const id = nativeSetTimeout(callback, delay);
      pending.add(id);
      return id;
    }) as typeof setTimeout;
    globalThis.clearTimeout = ((id: ReturnType<typeof setTimeout>) => {
      pending.delete(id);
      nativeClearTimeout(id);
    }) as typeof clearTimeout;
    try {
      if (outcome === "success") assert.deepEqual(await getChatSessions(), []);
      else await assert.rejects(getChatSessions(), (error: unknown) =>
        error instanceof ApiError && error.status === 503);
      assert.equal(pending.size, 0, "settled requests must not retain abort timers");
    } finally {
      for (const id of pending) nativeClearTimeout(id);
      globalThis.setTimeout = nativeSetTimeout;
      globalThis.clearTimeout = nativeClearTimeout;
    }
  });
}

test("a stalled typed request aborts and exposes the shared timeout error", async () => {
  mode = "stall";
  globalThis.setTimeout = ((callback: (...args: unknown[]) => void) =>
    nativeSetTimeout(callback, 10)) as typeof setTimeout;
  try {
    await assert.rejects(getCurrentActivity(), (error: unknown) =>
      error instanceof ApiError && error.status === 408);
    assert.equal(requestSignal?.aborted, true);
  } finally {
    globalThis.setTimeout = nativeSetTimeout;
  }
});
