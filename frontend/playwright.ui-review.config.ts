import { defineConfig } from "@playwright/test";

// Isolated UI checks: fixtures intercept all API and WebSocket traffic.
export default defineConfig({
  testDir: "./e2e",
  testMatch: ["workspace-review.spec.ts", "test-model-center.spec.ts", "test-intervention-history.spec.ts", "repair-20260930.spec.ts"],
  timeout: 30000,
  retries: 0,
  workers: 2,
  outputDir: "./test-results/ui-review",
  use: { baseURL: "http://127.0.0.1:4184", channel: "chrome", headless: true, screenshot: "only-on-failure" },
  webServer: { command: "npm run dev -- --host 127.0.0.1 --port 4184", url: "http://127.0.0.1:4184", reuseExistingServer: false, timeout: 30000 },
});
