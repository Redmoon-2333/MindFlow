import { test, expect, type Page } from "@playwright/test";

async function installChat(page: Page) {
  await page.addInitScript(() => localStorage.setItem("mindflow_authenticated", "1"));
  await page.routeWebSocket("**/api/v1/ws", socket => socket.close());
  await page.route("**/api/v1/**", route => {
    const pathname = new URL(route.request().url()).pathname;
    let json: unknown = {};
    if (pathname.endsWith("/chat/sessions")) json = [{ session_id: "keyboard-session" }];
    else if (pathname.endsWith("/messages")) json = [{ role: "assistant", content: "Saved conversation" }];
    else if (pathname.endsWith("/chat")) json = { session_id: "new-session", answer: "Synthetic reply", degraded: true };
    return route.fulfill({ json });
  });
}

test("mobile chat keeps input usable and can send a reply", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installChat(page);
  await page.goto("/chat");
  const input = page.getByRole("textbox", { name: "消息输入" });
  await expect(input).toBeVisible();
  const box = await input.boundingBox();
  expect(box!.width).toBeGreaterThan(200);
  await input.fill("Synthetic mobile message");
  await input.press("Enter");
  await expect(page.getByText("Synthetic reply", { exact: true })).toBeVisible();
});

test("conversation history is keyboard-selectable", async ({ page }) => {
  await installChat(page);
  await page.goto("/chat");
  const session = page.getByRole("button", { name: "会话 keyboard", exact: true });
  await session.focus();
  await session.press("Enter");
  await expect(session).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Saved conversation", { exact: true })).toBeVisible();
});

test("an error from an old new-chat request cannot replace the new conversation", async ({ page }) => {
  await installChat(page);
  let release: () => void = () => {};
  const held = new Promise<void>(resolve => { release = resolve; });
  await page.route("**/api/v1/chat", async route => {
    await held;
    await route.fulfill({ status: 500, json: { detail: "Old request failed" } });
  });
  await page.goto("/chat");
  await page.getByRole("textbox", { name: "消息输入" }).fill("Old message");
  const pending = page.waitForRequest(request => request.url().endsWith("/api/v1/chat") && request.method() === "POST");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await pending;
  await page.getByRole("button", { name: "新对话", exact: true }).click();
  const finished = page.waitForResponse(response => response.url().endsWith("/api/v1/chat") && response.status() === 500);
  release();
  await finished;
  await expect(page.getByRole("textbox", { name: "消息输入" })).toBeEnabled();
  await expect(page.locator(".error-box")).toHaveCount(0);
  await expect(page.getByText("Old message", { exact: true })).toHaveCount(0);
});


test("new chat adds a selectable draft immediately", async ({ page }) => {
  await installChat(page);
  await page.goto("/chat");
  await expect(page.getByRole("button", {name: "会话 keyboard", exact: true})).toBeVisible();
  await page.getByRole("button", {name: "新对话", exact: true}).click();
  const draft = page.locator(".chat-session-select", {hasText: "新对话"});
  await expect(draft).toHaveCount(1);
  await expect(draft).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("textbox", {name: "消息输入"}).fill("Immediate session title");
  await page.getByRole("button", {name: "发送", exact: true}).click();
  await expect(page.locator(".chat-session-select", {hasText: "Immediate session title"})).toHaveCount(1);
  await expect(page.getByText("Synthetic reply", {exact: true})).toBeVisible();
});

test("pending reply survives leaving chat and returning before completion", async ({ page }) => {
  await installChat(page);
  let release: () => void = () => {};
  const held = new Promise<void>(resolve => {release = resolve;});
  await page.route("**/api/v1/chat", async route => {
    await held;
    await route.fulfill({json: {session_id: "pending-session", answer: "Reply after navigating", degraded: false}});
  });
  await page.goto("/chat");
  await page.getByRole("textbox", {name: "消息输入"}).fill("Keep my pending message");
  await page.getByRole("button", {name: "发送", exact: true}).click();
  await expect(page.getByText("正在整理回复，请稍候…")).toBeVisible();
  await page.getByRole("link", {name: "专家面板", exact: true}).click();
  await expect(page.getByText("正在整理回复，请稍候…")).not.toBeVisible();
  await page.getByRole("link", {name: "AI 对话", exact: true}).click();
  await expect(page.getByText("Keep my pending message", {exact: true})).toBeVisible();
  await expect(page.getByText("正在整理回复，请稍候…")).toBeVisible();
  release();
  await expect(page.getByText("Reply after navigating", {exact: true})).toBeVisible();
});


test("partially displayed reply remains visible after navigating", async ({page}) => {
  await installChat(page);
  const answer = "这是用于检查页面切换时逐字回复是否保留的内容。".repeat(40);
  await page.route("**/api/v1/chat", route => route.fulfill({json: {session_id: "typing-session", answer, degraded: false}}));
  await page.goto("/chat");
  await page.getByRole("textbox", {name: "消息输入"}).fill("Continue displaying my reply");
  await page.getByRole("button", {name: "发送", exact: true}).click();
  await expect(page.getByText("正在显示回复…", {exact: true})).toBeVisible();
  await expect(page.locator(".chat-ai")).not.toBeEmpty();
  await page.getByRole("link", {name: "专家面板", exact: true}).click();
  await page.getByRole("link", {name: "AI 对话", exact: true}).click();
  await expect(page.locator(".chat-ai")).toBeVisible();
  await expect(page.locator(".chat-ai")).not.toBeEmpty();
  await page.getByRole("button", {name: "立即显示全文", exact: true}).click();
  await expect(page.locator(".chat-ai")).toHaveText(answer);
});


test("returning to a saved pending conversation and selecting it does not erase its reply", async ({page}) => {
  await installChat(page);
  let release: () => void = () => {};
  const held = new Promise<void>(resolve => {release = resolve;});
  await page.route("**/api/v1/chat", async route => {
    await held;
    await route.fulfill({json: {session_id: "keyboard-session", answer: "Saved pending reply", degraded: false}});
  });
  await page.goto("/chat");
  await page.getByRole("button", {name: "会话 keyboard", exact: true}).click();
  await expect(page.getByText("Saved conversation", {exact: true})).toBeVisible();
  await page.getByRole("textbox", {name: "消息输入"}).fill("Pending in saved conversation");
  await page.getByRole("button", {name: "发送", exact: true}).click();
  await expect(page.getByText("正在整理回复，请稍候…")).toBeVisible();
  await page.getByRole("link", {name: "专注分析", exact: true}).click();
  await page.getByRole("link", {name: "AI 对话", exact: true}).click();
  await page.getByRole("button", {name: "会话 keyboard", exact: true}).click();
  await expect(page.getByText("Pending in saved conversation", {exact: true})).toBeVisible();
  await expect(page.getByText("正在整理回复，请稍候…")).toBeVisible();
  release();
  await expect(page.getByText("Saved pending reply", {exact: true})).toBeVisible();
});

test("reply completed while away is available immediately on return", async ({page}) => {
  await installChat(page);
  let release: () => void = () => {};
  const held = new Promise<void>(resolve => {release = resolve;});
  await page.route("**/api/v1/chat", async route => {
    await held;
    await route.fulfill({json: {session_id: "away-session", answer: "Completed while away", degraded: false}});
  });
  await page.goto("/chat");
  await page.getByRole("textbox", {name: "消息输入"}).fill("Reply while away");
  await page.getByRole("button", {name: "发送", exact: true}).click();
  await expect(page.getByText("正在整理回复，请稍候…")).toBeVisible();
  await page.getByRole("link", {name: "专家面板", exact: true}).click();
  const finished = page.waitForResponse(response => response.url().endsWith("/api/v1/chat") && response.status() === 200);
  release(); await finished;
  await page.getByRole("link", {name: "AI 对话", exact: true}).click();
  await expect(page.locator(".chat-ai")).toBeVisible();
  await expect(page.getByText("Completed while away", {exact: true})).toBeVisible();
});
