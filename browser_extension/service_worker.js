const DEFAULT_BACKEND = "http://127.0.0.1:8765";
const HEARTBEAT_SECONDS = 30;
const BLOCKLIST_REFRESH_MINUTES = 1;

let activeContext = null;

async function getSettings() {
  return chrome.storage.local.get({ backendUrl: DEFAULT_BACKEND, browserToken: "" });
}

// ── Intervention execution: website blocking ─────────────────────────────
// The backend records blocked domains when an environment_optimization
// intervention fires (or the user manages the blocklist in the UI).  This
// worker polls the blocklist endpoint and translates enabled domains into
// declarativeNetRequest dynamic rules, so blocking is real execution rather
// than a suggestion.

function dynamicRuleFor(domain, index) {
  // ``||example.com^`` matches the domain and any of its subdomains.
  return {
    id: index + 1,
    priority: 1,
    action: { type: "block" },
    condition: { urlFilter: `||${domain}^` },
  };
}

async function syncBlocklist() {
  const settings = await getSettings();
  if (!settings.browserToken) return;
  try {
    const response = await fetch(
      `${settings.backendUrl}/api/v1/telemetry/browser/blocklist`,
      { headers: { "X-Browser-Token": settings.browserToken } },
    );
    if (!response.ok) return;
    const data = await response.json();
    const domains = Array.isArray(data.domains) ? data.domains : [];
    const rules = domains.map((domain, index) => dynamicRuleFor(domain, index));
    const oldRules = await chrome.declarativeNetRequest.getDynamicRules();
    await chrome.declarativeNetRequest.updateDynamicRules({
      removeRuleIds: oldRules.map((rule) => rule.id),
      addRules: rules,
    });
  } catch {
    // Non-fatal — the next alarm tick retries.
  }
}

function browserName() {
  return navigator.userAgent.includes("Edg/") ? "edge" : "chrome";
}

function contextFromTab(tab) {
  if (!tab || tab.incognito || !tab.url) return null;
  try {
    const url = new URL(tab.url);
    if (!url.hostname || !["http:", "https:"].includes(url.protocol)) return null;
    return {
      domain: url.hostname,
      audible: Boolean(tab.audible),
    };
  } catch {
    return null;
  }
}

async function currentContext() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return contextFromTab(tab);
}

function isSameContext(first, second) {
  return first?.domain === second?.domain && first?.audible === second?.audible;
}

async function flushContext(context, endedAt) {
  if (!context) return;
  const settings = await getSettings();
  if (!settings.browserToken) return;
  const elapsedSeconds = Math.max(1, Math.min(60, (endedAt - context.lastSentAt) / 1000));
  const response = await fetch(`${settings.backendUrl}/api/v1/telemetry/browser/heartbeat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Browser-Token": settings.browserToken,
    },
    body: JSON.stringify({
      timestamp_utc: new Date(context.lastSentAt).toISOString(),
      duration_s: elapsedSeconds,
      browser_name: browserName(),
      domain: context.domain,
      audible: context.audible,
      incognito: false,
    }),
  }).catch(() => null);
  if (response?.ok) context.lastSentAt = endedAt;
}

async function reconcileContext({ flushCurrent = false } = {}) {
  const now = Date.now();
  const nextContext = await currentContext();
  if (!isSameContext(activeContext, nextContext)) {
    await flushContext(activeContext, now);
    activeContext = nextContext ? { ...nextContext, lastSentAt: now } : null;
    return;
  }
  if (flushCurrent) await flushContext(activeContext, now);
}

function ensureAlarm() {
  chrome.alarms.create("mindflow-heartbeat", { periodInMinutes: HEARTBEAT_SECONDS / 60 });
  chrome.alarms.create("mindflow-blocklist", { periodInMinutes: BLOCKLIST_REFRESH_MINUTES });
}

chrome.runtime.onInstalled.addListener(() => {
  ensureAlarm();
  reconcileContext();
  syncBlocklist();
});

chrome.runtime.onStartup.addListener(() => {
  ensureAlarm();
  reconcileContext();
  syncBlocklist();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "mindflow-heartbeat") reconcileContext({ flushCurrent: true });
  if (alarm.name === "mindflow-blocklist") syncBlocklist();
});

chrome.tabs.onActivated.addListener(() => reconcileContext());
chrome.tabs.onUpdated.addListener((_tabId, changeInfo) => {
  if (changeInfo.status === "complete" || changeInfo.url || changeInfo.audible !== undefined) {
    reconcileContext();
  }
});
chrome.windows.onFocusChanged.addListener(() => reconcileContext());

ensureAlarm();
reconcileContext();
syncBlocklist();
