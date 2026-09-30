/**
 * Shared session bootstrap for the live-backend E2E suites.
 *
 * Two things are handled here so the suites stay deterministic:
 *
 * 1. **Target** — sessions are minted through the Vite dev proxy (4173), not
 *    against a hardcoded backend port. The proxy injects the launcher's root
 *    token for the ticket request, so the cookie always belongs to whichever
 *    backend the pages themselves talk to, and no root token has to live in
 *    the test environment. `MINDFLOW_TEST_TOKEN` is still honoured when a run
 *    is pointed straight at a backend.
 * 2. **Rate limit** — the backend fronts every `/api` request with a global
 *    token bucket (100 req/min). A full Playwright run easily bursts past
 *    that, so the ticket handshake would randomly answer 429 and fail
 *    `beforeAll` for reasons unrelated to the feature under test. Retrying on
 *    429 (with a backoff that outlasts the bucket refill) keeps the suites
 *    deterministic without touching the production rate limit.
 */
import { expect, type APIRequestContext } from "@playwright/test";

/** Vite dev proxy — injects the launcher token for the bootstrap ticket. */
const BASE = "http://127.0.0.1:4173";
/** Optional root token for a run pointed directly at a backend; never committed. */
export const AUTH_TOKEN = process.env.MINDFLOW_TEST_TOKEN ?? "";

const MAX_ATTEMPTS = 6;
const BACKOFF_MS = 2_500;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Authorization for the ticket request: only present when a root token is
 *  supplied. Going through the proxy the header is rewritten anyway, and an
 *  empty `Bearer ` would otherwise 401 when no token file is configured. */
export function ticketHeaders(): Record<string, string> {
  return AUTH_TOKEN ? { Authorization: `Bearer ${AUTH_TOKEN}` } : {};
}

/** POST /auth/bootstrap/ticket, retrying while the global bucket is empty. */
async function issueTicket(request: APIRequestContext): Promise<string> {
  let lastStatus = 0;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const res = await request.post(`${BASE}/api/v1/auth/bootstrap/ticket`, {
      headers: ticketHeaders(),
    });
    lastStatus = res.status();
    if (res.ok()) {
      const { ticket } = await res.json();
      return ticket;
    }
    if (lastStatus !== 429) break;
    await sleep(BACKOFF_MS);
  }
  expect(lastStatus, `bootstrap ticket kept failing with HTTP ${lastStatus}`).toBe(200);
  return "";
}

/**
 * Exchange a ticket for a `mindflow_session` cookie and return the raw
 * `Cookie` header value. One session is reused across a suite so the
 * bounded in-memory session store is not exhausted.
 */
export async function initSharedSession(request: APIRequestContext): Promise<string> {
  const ticket = await issueTicket(request);
  const bootstrapRes = await request.post(`${BASE}/api/v1/auth/bootstrap`, { data: { ticket } });
  if (bootstrapRes.status() === 429) {
    await sleep(BACKOFF_MS);
    const retryTicket = await issueTicket(request);
    const retry = await request.post(`${BASE}/api/v1/auth/bootstrap`, {
      data: { ticket: retryTicket },
    });
    expect(retry.ok(), `bootstrap exchange failed with HTTP ${retry.status()}`).toBeTruthy();
    return cookieFrom(retry);
  }
  expect(bootstrapRes.ok(), `bootstrap exchange failed with HTTP ${bootstrapRes.status()}`).toBeTruthy();
  return cookieFrom(bootstrapRes);
}

function cookieFrom(response: { headersArray: () => Array<{ name?: string; value?: string }> }): string {
  const cookies = response.headersArray();
  const raw = cookies.find((h) => h.value?.includes("mindflow_session="));
  const m = (raw?.value ?? "").match(/(mindflow_session=[^;]+)/);
  return m?.[1] ?? "";
}

/** Derive the bare cookie value (without `name=`) from a Cookie header. */
export function sessionValue(cookieHeader: string): string {
  const m = cookieHeader.match(/mindflow_session=([^;]+)/);
  return m?.[1] ?? "";
}

/**
 * GET that tolerates the global rate-limit bucket.
 *
 * The API smoke lists fire dozens of requests back to back; when the 100/min
 * bucket is empty the correct product answer is 429, not a broken feature —
 * so the assertion is retried instead of failing the suite for timing.
 */
export async function getWithRateLimitRetry(
  request: APIRequestContext,
  url: string,
  options: { headers?: Record<string, string> } = {},
): Promise<{ ok: boolean; status: number }> {
  let status = 0;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const res = await request.get(url, options);
    status = res.status();
    if (status !== 429) return { ok: res.ok(), status };
    await sleep(BACKOFF_MS);
  }
  return { ok: false, status };
}
