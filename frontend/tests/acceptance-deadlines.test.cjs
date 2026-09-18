// Source-driven regressions with fake HTTP/timers. No server or training.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function load(relative, imports, globals = {}, suffix = "") {
  const filename = path.resolve(__dirname, relative);
  const code = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
    },
  }).outputText;
  const exports = {};
  vm.runInNewContext(code + suffix, {
    exports,
    require(name) {
      assert.ok(name in imports, `Unexpected import: ${name}`);
      return imports[name];
    },
    AbortController, Response, console, ...globals,
  }, { filename });
  return exports;
}

async function main() {
  let passed = 0;
  for (const method of ["sendChat", "triggerIntervention", "triggerPanel"]) {
    for (const mode of ["timeout", "success", "cancel"]) {
      if (method === "triggerPanel" && mode === "cancel") continue;
      const timers = new Map();
      let nextTimer = 0;
      let signal;
      let complete;
      const api = load("../src/api.ts", {
        "openapi-fetch": { default: () => ({}) },
        "./prediction-state": {}, "./report-state": {}, "./baseline-state": {},
      }, {
        setTimeout(fn, delay) {
          timers.set(++nextTimer, { fn, delay });
          return nextTimer;
        },
        clearTimeout(id) { timers.delete(id); },
        fetch(_url, init) {
          signal = init.signal;
          return new Promise((resolve, reject) => {
            complete = () => resolve(new Response("{}", { status: 200 }));
            signal.addEventListener("abort", () => reject(signal.reason), { once: true });
          });
        },
      });
      const caller = new AbortController();
      const pending = method === "sendChat"
        ? api.sendChat("synthetic", "test-session", caller.signal)
        : method === "triggerIntervention"
          ? api.triggerIntervention("gentle", caller.signal)
          : api.triggerPanel();
      assert.equal(timers.size, 1, "one actual HTTP deadline, no shadow timeout");
      assert.equal([...timers.values()][0].delay, 660_000);
      assert.equal(signal.aborted, false, "no inner 30s timer");
      if (mode === "success") {
        complete();
        await pending;
      } else if (mode === "timeout") {
        const rejected = assert.rejects(pending, e => e instanceof api.ApiError && e.status === 408);
        [...timers.values()][0].fn();
        await rejected;
        assert.equal(signal.aborted, true);
      } else {
        const rejected = assert.rejects(pending, e => e.name === "AbortError" && !(e instanceof api.ApiError));
        caller.abort();
        await rejected;
        assert.equal(signal.aborted, true);
      }
      assert.equal(timers.size, 0, "timer cleaned after every outcome");
      passed++;
    }
  }
  const React = require("react");
  const { renderToStaticMarkup } = require("react-dom/server");
  let stateIndex = 0;
  const readiness = { current_training_job: { job_id: "synthetic", status: "interrupted" } };
  const page = load("../src/pages/ModelCenter.tsx", {
    react: {
      useState(initial) {
        const index = stateIndex++;
        return [index === 0 ? "模型训练" : index === 1 ? readiness : initial, () => {}];
      },
      useEffect() {},
      useCallback(fn) { return fn; },
      useRef(current) { return { current }; },
    },
    "react/jsx-runtime": require("react/jsx-runtime"),
    "../api": {},
    "../baseline-state": { EMPTY_BASELINE_VIEW: {} },
    "./model-center.css": {},
  }, {}, "\nexports.testState = {isNonterminal, jobStatusLabel};");
  assert.equal(page.testState.isNonterminal("interrupted"), false);
  assert.equal(page.testState.jobStatusLabel("interrupted"), "已中断");
  const html = renderToStaticMarkup(React.createElement(page.default));
  assert.ok(html.includes("已中断"));
  assert.ok(html.includes("任务因服务中断而结束"));
  assert.ok(!html.includes("取消任务"));
  assert.ok(!html.includes('class="spinner"'));
  passed++;
  console.log(`acceptance-deadlines: ${passed} tests passed`);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
