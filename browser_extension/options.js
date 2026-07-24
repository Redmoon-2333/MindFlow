const DEFAULT_BACKEND = "http://127.0.0.1:8765";
const backendInput = document.querySelector("#backend");
const codeInput = document.querySelector("#code");
const statusElement = document.querySelector("#status");

chrome.storage.local.get({ backendUrl: DEFAULT_BACKEND }).then((settings) => {
  backendInput.value = settings.backendUrl;
});

function normalizeBackend(value) {
  const parsed = new URL(value.trim().replace(/\/$/, ""));
  const isLoopback = ["127.0.0.1", "localhost", "[::1]"].includes(parsed.hostname);
  if (parsed.protocol !== "http:" || !isLoopback || parsed.port !== "8765") {
    throw new Error("后端地址必须是 http://127.0.0.1:8765 或等价本机地址");
  }
  return parsed.origin;
}

document.querySelector("#pair").addEventListener("click", async () => {
  const code = codeInput.value.trim();
  statusElement.textContent = "正在配对…";
  try {
    if (!/^\d{6}$/.test(code)) throw new Error("请输入六位数字配对码");
    const backendUrl = normalizeBackend(backendInput.value);
    const response = await fetch(`${backendUrl}/api/v1/telemetry/browser/pair`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    if (!response.ok) throw new Error("配对码无效或已过期");
    const body = await response.json();
    await chrome.storage.local.set({ backendUrl, browserToken: body.token });
    statusElement.textContent = "配对成功，域名停留统计已启用。";
  } catch (error) {
    statusElement.textContent = error instanceof Error ? error.message : "配对失败";
  }
});
