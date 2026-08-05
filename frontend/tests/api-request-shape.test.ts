/** Regression checks for request payload names at the frontend/backend boundary. */
import assert from "node:assert/strict";
import { clearTelemetryData, triggerPanel } from "../src/api.ts";

const originalFetch = globalThis.fetch;

try {
  let capturedInput: RequestInfo | URL | undefined;
  let capturedInit: RequestInit | undefined;
  let responseBody: unknown = {};
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    capturedInput = input;
    capturedInit = init;
    return new Response(JSON.stringify(responseBody), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) as typeof fetch;

  await triggerPanel({ retryIfDegraded: true });

  assert.equal(
    capturedInit?.body,
    JSON.stringify({ retry_if_degraded: true }),
    "panel retry must use the backend snake_case field name",
  );

  responseBody = {
    deleted: 12,
    partial: true,
    failures: ["browser_tokens"],
  };
  const deleteResult = await clearTelemetryData("all");

  assert.equal(capturedInput, "/api/v1/telemetry/data?scope=all");
  assert.equal(capturedInit?.method, "DELETE");
  assert.deepEqual(deleteResult, responseBody);
  console.log("api-request-shape: 4 tests passed");
} finally {
  globalThis.fetch = originalFetch;
}
