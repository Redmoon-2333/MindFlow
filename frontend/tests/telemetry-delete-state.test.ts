/** Contract tests for privacy-delete UI state mapping. */
import assert from "node:assert/strict";
import {
  mapTelemetryDeleteResult,
  runTelemetryDelete,
  shouldClearTelemetryPairingCode,
} from "../src/api.ts";

const complete = mapTelemetryDeleteResult("interaction", {
  deleted: 8,
  partial: false,
  failures: [],
});
assert.deepEqual(complete, {
  kind: "success",
  message: "已删除 8 条记录",
  failures: [],
  retryScope: null,
});

const partial = mapTelemetryDeleteResult("all", {
  deleted: 21,
  partial: true,
  failures: ["browser_tokens", "model_artifacts"],
});
assert.deepEqual(partial, {
  kind: "partial",
  message: "已删除 21 条记录，但有 2 项未完成",
  failures: [
    "浏览器配对令牌（browser_tokens）",
    "本地模型文件（model_artifacts）",
  ],
  retryScope: "all",
});

const failuresWithoutPartialFlag = mapTelemetryDeleteResult("browser", {
  deleted: 3,
  partial: false,
  failures: ["future_cleanup_step"],
});
assert.deepEqual(failuresWithoutPartialFlag, {
  kind: "partial",
  message: "已删除 3 条记录，但有 1 项未完成",
  failures: ["future_cleanup_step"],
  retryScope: "browser",
});

for (const result of [
  { deleted: 5, partial: true },
  { deleted: 5, partial: true, failures: [] },
]) {
  const unknownPartial = mapTelemetryDeleteResult("all", result);
  assert.equal(unknownPartial.kind, "partial");
  assert.doesNotMatch(unknownPartial.message, /0 项/);
  assert.deepEqual(unknownPartial.failures, ["未知清理步骤（服务未返回失败详情）"]);
  assert.equal(shouldClearTelemetryPairingCode("all", result), false);
}

assert.equal(
  shouldClearTelemetryPairingCode("all", {
    deleted: 5,
    partial: true,
    failures: ["model_artifacts"],
  }),
  true,
);
assert.equal(
  shouldClearTelemetryPairingCode("browser", {
    deleted: 5,
    partial: true,
    failures: ["browser_tokens"],
  }),
  false,
);

const committed: unknown[] = [];
const refreshFailure = new Error("status unavailable");
const execution = await runTelemetryDelete("all", {
  clear: async () => ({ deleted: 9, partial: false, failures: [] }),
  refresh: async () => { throw refreshFailure; },
  onDeleted: (outcome) => committed.push(outcome),
});
assert.deepEqual(committed, [{
  notice: {
    kind: "success",
    message: "已删除 9 条记录",
    failures: [],
    retryScope: null,
  },
  clearPairingCode: true,
}]);
assert.equal(execution.telemetry, null);
assert.equal(execution.refreshError, refreshFailure);

console.log("telemetry-delete-state: 8 tests passed");
