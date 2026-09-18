import { test, expect, type Page, type Route, type WebSocketRoute } from "@playwright/test";

async function setup(page: Page) {
  await page.clock.install();
  await page.addInitScript(() => {
    localStorage.setItem("mindflow_authenticated", "1");
    const notifications: string[] = [];
    Object.defineProperty(window, "__auditNotifications", { value: notifications });
    class FakeNotification {
      static permission = "granted";
      static async requestPermission() { return "granted"; }
      onclick: (() => void) | null = null;
      constructor(title: string) { notifications.push(title); }
      close() {}
    }
    Object.defineProperty(window, "Notification", { value: FakeNotification });
  });
  // Fail closed: no request can reach a real backend, including training POSTs.
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/chat/sessions") {
      await route.fulfill({ json: [] });
    } else if (url.pathname === "/api/v1/intervention/history") {
      await route.fulfill({ json: { items: [], count: 0 } });
    } else {
      await route.fulfill({ status: 503, json: { detail: "Unmocked audit endpoint" } });
    }
  });
}

test("chat survives old deadlines, aborts at total deadline and unlocks input", async ({ page }) => {
  await setup(page);
  await page.routeWebSocket("**/api/v1/ws", () => {});
  let requests = 0;
  await page.route("**/api/v1/chat", () => { requests++; });
  await page.goto("/chat");
  const input = page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行");
  await input.fill("Synthetic deadline question");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect.poll(() => requests).toBe(1);
  await page.clock.fastForward(200_001);
  await expect(input).toBeDisabled();
  await expect(page.locator(".error-box")).toHaveCount(0);
  await page.clock.fastForward(460_001);
  await expect(page.locator(".error-box")).toContainText("请求超时");
  await expect(input).toBeEnabled();
  await expect(page.locator(".chat-ai .spinner")).toHaveCount(0);
  expect(requests).toBe(1);
});

test("chat result after the old 200s budget is displayed without false evidence", async ({ page }) => {
  await setup(page);
  await page.routeWebSocket("**/api/v1/ws", () => {});
  let pending: Route | undefined;
  await page.route("**/api/v1/chat", (route) => { pending = route; });
  await page.goto("/chat");
  const input = page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行");
  await input.fill("Synthetic slow question");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect.poll(() => Boolean(pending)).toBe(true);
  await page.clock.fastForward(240_000);
  await pending!.fulfill({
    json: {
      session_id: "mock-session", answer: "Synthetic completed answer",
      degraded: false, evidence_cited: false, tools_used: ["query_evidence"],
    },
  });
  await expect(page.getByText("Synthetic completed answer")).toBeVisible();
  await expect(page.getByText("已引用行为证据", { exact: true })).toHaveCount(0);
  await expect(input).toBeEnabled();
  await page.clock.fastForward(660_000);
  await expect(page.locator(".error-box")).toHaveCount(0);
});

for (const outcome of ["success", "failure"] as const) {
  test(`late history ${outcome} cannot overwrite a new conversation`, async ({ page }) => {
    await setup(page);
    await page.routeWebSocket("**/api/v1/ws", () => {});
    await page.route("**/api/v1/chat/sessions", route => route.fulfill({
      json: [{ session_id: "old-session", last_message_at: "2026-09-19T00:00:00Z" }],
    }));
    let history: Route | undefined;
    await page.route("**/api/v1/chat/old-session/messages", route => { history = route; });
    await page.route("**/api/v1/chat", route => route.fulfill({
      json: {
        session_id: "new-session", answer: "NEW_ANSWER",
        degraded: false, evidence_cited: false, tools_used: [],
      },
    }));
    await page.goto("/chat");
    await page.getByText("会话 old-sess", { exact: true }).click();
    await expect.poll(() => Boolean(history)).toBe(true);
    await page.getByRole("button", { name: "新对话", exact: true }).click();
    const input = page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行");
    await input.fill("NEW_QUESTION");
    await page.getByRole("button", { name: "发送", exact: true }).click();
    await expect(input).toBeEnabled();
    const historyResponse = page.waitForResponse("**/api/v1/chat/old-session/messages");
    await history!.fulfill(outcome === "success"
      ? { json: [{ role: "assistant", content: "OLD_PRIVATE_HISTORY" }] }
      : { status: 500, json: { detail: "OLD_HISTORY_ERROR" } });
    await (await historyResponse).finished();
    await expect(page.getByText("NEW_QUESTION", { exact: true })).toBeVisible();
    await expect(page.getByText("NEW_ANSWER", { exact: true })).toBeVisible();
    await expect(page.getByText("OLD_PRIVATE_HISTORY", { exact: true })).toHaveCount(0);
    await expect(page.locator(".error-box")).toHaveCount(0);
    await expect(page.locator(".spinner")).toHaveCount(0);
  });
}

test("new conversation clears history loading immediately", async ({ page }) => {
  await setup(page);
  await page.routeWebSocket("**/api/v1/ws", () => {});
  await page.route("**/api/v1/chat/sessions", route => route.fulfill({
    json: [{ session_id: "old-session", last_message_at: "2026-09-19T00:00:00Z" }],
  }));
  let requested = false;
  await page.route("**/api/v1/chat/old-session/messages", () => { requested = true; });
  await page.goto("/chat");
  await page.getByText("会话 old-sess", { exact: true }).click();
  await expect.poll(() => requested).toBe(true);
  await page.getByRole("button", { name: "新对话", exact: true }).click();
  await expect(page.getByText("开始新对话，向 MindFlow 智能助手提问", { exact: true })).toBeVisible();
  await expect(page.locator(".spinner")).toHaveCount(0);
});

test("interrupted training is terminal and does not keep polling or offer cancel", async ({ page }) => {
  await setup(page);
  await page.routeWebSocket("**/api/v1/ws", () => {});
  let polls = 0;
  const jobId = "synthetic-job";
  await page.route("**/api/v1/analytics/training-readiness", (route) => route.fulfill({
    json: {
      raw_events: { total_events: 0, coverage_days: 0 },
      v2_windows: { total: 0, date_range_days: 0, schema_version: 3, eligible_count: 0 },
      feedback_labels: { focus: 0, distract: 0, mixed: 0, total: 0 },
      gates: [], blockers: [], trainable: false,
      current_training_job: { job_id: jobId, status: polls >= 2 ? "interrupted" : "training" },
    },
  }));
  await page.route("**/api/v1/analytics/baseline", route => route.fulfill({ status: 404 }));
  await page.route("**/api/v1/analytics/model-status", route => route.fulfill({
    json: { ready: false, loaded: false, mode: "rule_engine_only", reasons: [] },
  }));
  await page.route(`**/api/v1/analytics/training-jobs/${jobId}`, route => {
    polls++;
    return route.fulfill({
      json: {
        job_id: jobId, status: polls >= 2 ? "interrupted" : "training",
        activated: false, error: polls >= 2 ? "Synthetic restart" : null,
      },
    });
  });
  await page.goto("/model-center");
  await page.getByRole("tab", { name: "模型训练", exact: true }).click();
  await expect(page.getByText("训练中", { exact: true })).toBeVisible();
  await page.clock.fastForward(3_001);
  await expect(page.getByText("已中断", { exact: true })).toBeVisible();
  await expect(page.getByText("任务因服务中断而结束，可重新启动训练。")).toBeVisible();
  await expect(page.getByRole("button", { name: "取消任务" })).toHaveCount(0);
  await expect(page.locator(".mc-job-status .spinner")).toHaveCount(0);
  const terminalPolls = polls;
  await page.clock.fastForward(60_000);
  expect(polls).toBe(terminalPolls);
});

test("late socket close preserves new heartbeat and native delivery suppresses browser notice", async ({ page }) => {
  await setup(page);
  const sockets: WebSocketRoute[] = [];
  const pings: number[] = [];
  await page.routeWebSocket("**/api/v1/ws", socket => {
    const index = sockets.length;
    sockets.push(socket);
    pings[index] = 0;
    socket.onMessage(message => {
      if (JSON.parse(String(message)).type === "ping") {
        pings[index]++;
        socket.send(JSON.stringify({ type: "pong" }));
      }
    });
  });
  await page.goto("/intervention");
  await expect(page.getByText("暂无干预记录", { exact: true })).toBeVisible();
  await expect.poll(() => sockets.length).toBe(1);
  // Closing the native WebSocket is asynchronous: reconnect before old close arrives.
  await page.evaluate(async () => {
    const modulePath = "/src/realtime.ts";
    const { realtimeClient } = await import(modulePath);
    realtimeClient.disconnect();
    realtimeClient.connect();
    await new Promise<void>(resolve => {
      realtimeClient.subscribeStatus((status: string) => {
        if (status === "connected") resolve();
      });
    });
  });
  await expect.poll(() => sockets.length).toBe(2);
  await page.clock.fastForward(1_001);
  expect(sockets).toHaveLength(2);
  const frame = (id: string, native: boolean) => JSON.stringify({
    type: "intervention", timestamp: "2026-09-19T10:00:00Z",
    payload: {
      id, title: id, message: "Synthetic reminder", intervention_type: "nudge",
      dismissible: true, native_delivered: native,
    },
  });
  sockets[1].send(frame("native-only", true));
  await expect(page.getByText("native-only").first()).toBeVisible();
  const notices = () => page.evaluate(() =>
    (window as unknown as { __auditNotifications: string[] }).__auditNotifications);
  expect(await notices()).toEqual([]);
  sockets[1].send(frame("browser-fallback", false));
  await expect.poll(notices).toEqual(["browser-fallback"]);
  await page.clock.fastForward(30_001);
  await expect.poll(() => pings[1]).toBe(1);
  expect(pings[0]).toBe(0);
  expect(sockets).toHaveLength(2);
});
