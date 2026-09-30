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
