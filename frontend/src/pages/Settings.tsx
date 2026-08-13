import { useState, useEffect, useCallback } from "react";
import {
  getHealth,
  getCollectorStatus,
  startCollector,
  stopCollector,
  getAutonomy,
  pauseAutonomy,
  resumeAutonomy,
  getClassifications,
  addClassification,
  deleteClassification,
  getUnknownApps,
  exportData,
  getPreferences,
  putPreferences,
  patchPreferences,
  getTelemetryStatus,
  patchTelemetryPreferences,
  createBrowserPairingCode,
  clearTelemetryData,
  getErrorMessage,
  isAutonomyPaused,
  runTelemetryDelete,
} from "../api";
import type { AutonomyStatus, ClassificationRule, ClassificationRuleInput, CollectorStatus, HealthData, Preferences, TelemetryDeleteNotice, TelemetryDeleteScope, TelemetryPreferences, TelemetryStatus } from "../api";

const CATEGORY_OPTIONS = ["code", "browser_work", "communication", "document", "entertainment", "social", "other"];

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(2)} MB`;
}

function formatTelemetryTime(value: string | null): string {
  if (!value) return "暂无";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN");
}

function isPreferences(value: unknown): value is Preferences {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export default function Settings() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthData | null>(null);
  const [collector, setCollector] = useState<CollectorStatus | null>(null);
  const [autonomy, setAutonomy] = useState<AutonomyStatus | null>(null);
  const [classifications, setClassifications] = useState<ClassificationRule[]>([]);
  const [preferences, setPreferences] = useState("");

  const [collectorLoading, setCollectorLoading] = useState(false);
  const [autonomyLoading, setAutonomyLoading] = useState(false);
  const [pauseHours, setPauseHours] = useState(1);

  const [newRule, setNewRule] = useState<ClassificationRuleInput>({
    process_name: "",
    window_title_pattern: "",
    category: "neutral",
    priority: 0,
  });
  const [unknownApps, setUnknownApps] = useState<string[]>([]);
  const [fetchingUnknown, setFetchingUnknown] = useState(false);
  const [addingRule, setAddingRule] = useState(false);

  const [exportFmt, setExportFmt] = useState<"csv" | "json">("csv");
  const [exportStart, setExportStart] = useState("");
  const [exportEnd, setExportEnd] = useState("");
  const [exporting, setExporting] = useState(false);

  const [prefLoading, setPrefLoading] = useState(false);
  const isCollectorRunning = collector?.running === true || collector?.status === "running";
  const autonomyPaused = isAutonomyPaused(autonomy);
  const [telemetry, setTelemetry] = useState<TelemetryStatus | null>(null);
  const [telemetryLoading, setTelemetryLoading] = useState(false);
  const [pairingCode, setPairingCode] = useState<{ code: string; expires_at: string } | null>(null);
  const [telemetryMessage, setTelemetryMessage] = useState<string | null>(null);
  const [telemetryDeleteNotice, setTelemetryDeleteNotice] = useState<TelemetryDeleteNotice | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    const [healthResult, collectorResult, autonomyResult, classificationsResult, preferencesResult, telemetryResult] = await Promise.allSettled([
      getHealth(),
      getCollectorStatus(),
      getAutonomy(),
      getClassifications(),
      getPreferences(),
      getTelemetryStatus(),
    ]);
    const failures: string[] = [];

    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else failures.push(getErrorMessage(healthResult.reason, "系统状态加载失败"));
    if (collectorResult.status === "fulfilled") setCollector(collectorResult.value);
    else failures.push(getErrorMessage(collectorResult.reason, "采集器状态加载失败"));
    if (autonomyResult.status === "fulfilled") setAutonomy(autonomyResult.value);
    else failures.push(getErrorMessage(autonomyResult.reason, "自主模式状态加载失败"));
    if (classificationsResult.status === "fulfilled") setClassifications(Array.isArray(classificationsResult.value) ? classificationsResult.value : []);
    else failures.push(getErrorMessage(classificationsResult.reason, "应用分类加载失败"));
    if (preferencesResult.status === "fulfilled") setPreferences(JSON.stringify(preferencesResult.value, null, 2));
    else failures.push(getErrorMessage(preferencesResult.reason, "偏好设置加载失败"));
    if (telemetryResult.status === "fulfilled") setTelemetry(telemetryResult.value);
    else failures.push(getErrorMessage(telemetryResult.reason, "遥测状态加载失败"));

    if (failures.length > 0) setError(`部分设置加载失败：${failures.join("；")}`);
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleCollectorToggle = async () => {
    setCollectorLoading(true);
    try {
      const next = isCollectorRunning ? await stopCollector() : await startCollector();
      setCollector(next);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "操作失败"));
    } finally {
      setCollectorLoading(false);
    }
  };

  const handlePause = async () => {
    setAutonomyLoading(true);
    try {
      const result = await pauseAutonomy(pauseHours);
      setAutonomy(result);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "操作失败"));
    } finally {
      setAutonomyLoading(false);
    }
  };

  const handleResume = async () => {
    setAutonomyLoading(true);
    try {
      const result = await resumeAutonomy();
      setAutonomy(result);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "操作失败"));
    } finally {
      setAutonomyLoading(false);
    }
  };

  const handleFetchUnknown = async () => {
    setFetchingUnknown(true);
    try {
      const apps = await getUnknownApps();
      setUnknownApps(Array.isArray(apps) ? apps : []);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "获取失败"));
    } finally {
      setFetchingUnknown(false);
    }
  };

  const handleAddRule = async () => {
    if (!newRule.process_name) return;
    setAddingRule(true);
    try {
      await addClassification(newRule);
      setNewRule({ process_name: "", window_title_pattern: "", category: "neutral", priority: 0 });
      const cls = await getClassifications();
      setClassifications(Array.isArray(cls) ? cls : []);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "添加失败"));
    } finally {
      setAddingRule(false);
    }
  };

  const handleDeleteRule = async (id: string) => {
    try {
      await deleteClassification(id);
      setClassifications((prev) => prev.filter((r) => r.id !== id));
    } catch (e: unknown) {
      setError(getErrorMessage(e, "删除失败"));
    }
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      const text = await exportData(exportFmt, exportStart || undefined, exportEnd || undefined);
      const blob = new Blob([text], { type: exportFmt === "csv" ? "text/csv" : "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `mindflow_export.${exportFmt}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "导出失败"));
    } finally {
      setExporting(false);
    }
  };

  /** Shared JSON parse+validate for the preferences editor. Returns null and
   *  surfaces the error when the text is invalid (caller must then return). */
  const parsePreferencesText = (): Parameters<typeof putPreferences>[0] | null => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(preferences);
    } catch {
      setError("JSON 格式无效");
      setPrefLoading(false);
      return null;
    }
    if (!isPreferences(parsed)) {
      setError("JSON 必须是对象");
      setPrefLoading(false);
      return null;
    }
    return parsed;
  };

  const handlePutPrefs = async () => {
    setPrefLoading(true);
    try {
      const parsed = parsePreferencesText();
      if (parsed === null) return; // error already surfaced
      const result = await putPreferences(parsed);
      setPreferences(JSON.stringify(result, null, 2));
    } catch (e: unknown) {
      setError(getErrorMessage(e, "保存失败"));
    } finally {
      setPrefLoading(false);
    }
  };

  const updateTelemetryPreferences = async (updates: Partial<TelemetryPreferences>) => {
    setTelemetryLoading(true);
    setTelemetryMessage(null);
    setTelemetryDeleteNotice(null);
    try {
      await patchTelemetryPreferences(updates);
      setTelemetry(await getTelemetryStatus());
      setTelemetryMessage("隐私采集设置已更新");
    } catch (e: unknown) {
      setError(getErrorMessage(e, "遥测设置更新失败"));
    } finally {
      setTelemetryLoading(false);
    }
  };

  const handleCreatePairingCode = async () => {
    setTelemetryLoading(true);
    setTelemetryMessage(null);
    setTelemetryDeleteNotice(null);
    try {
      const result = await createBrowserPairingCode();
      setPairingCode(result);
      setTelemetry(await getTelemetryStatus());
    } catch (e: unknown) {
      setError(getErrorMessage(e, "生成配对码失败"));
    } finally {
      setTelemetryLoading(false);
    }
  };

  const handleClearTelemetry = async (scope: TelemetryDeleteScope) => {
    const labels = { interaction: "输入行为", browser: "浏览器", feedback: "反馈", all: "全部行为" };
    if (!window.confirm(`确定清除${labels[scope]}数据吗？此操作无法撤销。`)) return;
    setTelemetryLoading(true);
    setTelemetryMessage(null);
    setTelemetryDeleteNotice(null);
    try {
      const execution = await runTelemetryDelete(scope, {
        clear: clearTelemetryData,
        refresh: getTelemetryStatus,
        onDeleted: ({ notice, clearPairingCode }) => {
          setTelemetryDeleteNotice(notice);
          if (clearPairingCode) setPairingCode(null);
        },
      });
      if (execution.refreshError === null) {
        setTelemetry(execution.telemetry);
      } else {
        const detail = getErrorMessage(execution.refreshError, "未知错误");
        setError(`数据已清除，但状态刷新失败：${detail}`);
      }
    } catch (e: unknown) {
      setError(getErrorMessage(e, "清除失败"));
    } finally {
      setTelemetryLoading(false);
    }
  };

  const handlePatchPrefs = async () => {
    setPrefLoading(true);
    try {
      const parsed = parsePreferencesText();
      if (parsed === null) return; // error already surfaced
      const result = await patchPreferences(parsed);
      setPreferences(JSON.stringify(result, null, 2));
    } catch (e: unknown) {
      setError(getErrorMessage(e, "更新失败"));
    } finally {
      setPrefLoading(false);
    }
  };

  return (
    <div>
      <div className="header">
        <h1>系统设置</h1>
        <p>管理 MindFlow 配置与数据</p>
      </div>

      {error && (
        <div className="error-box mb16">
          {error}
          <button className="btn btn-sm mt8" onClick={() => setError(null)} style={{ marginLeft: 12 }}>
            关闭
          </button>
        </div>
      )}

      {loading && (
        <div className="card mb24" role="status">
          <div className="flex gap8" style={{ alignItems: "center" }}>
            <span className="spinner" style={{ width: 16, height: 16, margin: 0, borderWidth: 2 }} />
            正在加载设置，部分操作暂不可用
          </div>
        </div>
      )}

      {/* 1. System Info */}
      <div className="card mb24">
        <h3>系统信息</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12 }}>
          <div>
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>状态</div>
            <span className={`badge mt8 ${health?.status === "ok" || health?.status === "healthy" ? "badge-success" : "badge-danger"}`}>
              {health?.status ?? "未知"}
            </span>
          </div>
          <div>
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>版本</div>
            <div style={{ fontSize: 14, fontWeight: 500, marginTop: 4 }}>{health?.version ?? "--"}</div>
          </div>
          {health?.database != null && (
            <div>
              <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>数据库</div>
              <div style={{ fontSize: 14, fontWeight: 500, marginTop: 4 }}>{health.database.status}</div>
            </div>
          )}
          {health?.migration != null && (
            <div>
              <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>迁移</div>
              <div style={{ fontSize: 14, fontWeight: 500, marginTop: 4 }}>{health.migration.applied ? "applied" : "pending"}</div>
            </div>
          )}
        </div>
      </div>

      <div className="card mb24">
        <div className="flex flex-between" style={{ alignItems: "flex-start" }}>
          <div>
            <h3 style={{ marginBottom: 6 }}>隐私行为采集</h3>
            <p style={{ color: "var(--color-text-secondary)", fontSize: 13, margin: 0 }}>
              只保存 30 秒聚合计数和浏览器域名，不记录按键内容、鼠标坐标或完整网址。
            </p>
          </div>
          <span className={`badge ${telemetry?.input_watcher_status === "running" ? "badge-success" : "badge-warning"}`}>
            输入 watcher：{telemetry?.input_watcher_status ?? "未知"}
          </span>
        </div>
        {telemetryMessage && <div className="success-box mt16">{telemetryMessage}</div>}
        {telemetryDeleteNotice?.kind === "success" && (
          <div className="success-box mt16">{telemetryDeleteNotice.message}</div>
        )}
        {telemetryDeleteNotice?.kind === "partial" && (
          <div className="error-box mt16" role="status">
            <strong>{telemetryDeleteNotice.message}</strong>
            <div style={{ marginTop: 8 }}>未完成项目：</div>
            <ul style={{ margin: "6px 0 10px", paddingLeft: 20 }}>
              {telemetryDeleteNotice.failures.map((failure) => <li key={failure}>{failure}</li>)}
            </ul>
            <button
              className="btn btn-sm btn-danger"
              disabled={telemetryLoading}
              onClick={() => handleClearTelemetry(telemetryDeleteNotice.retryScope)}
            >
              再次尝试清除
            </button>
          </div>
        )}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16, marginTop: 20 }}>
          <label className="flex flex-between" style={{ padding: 14, border: "1px solid var(--color-border)", borderRadius: 10 }}>
            <span><strong style={{ display: "block", fontSize: 14 }}>鼠标与键盘聚合统计</strong><span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>点击、滚动、移动距离和输入活跃秒数</span></span>
            <input type="checkbox" checked={telemetry?.preferences.input_telemetry_enabled ?? false} disabled={loading || telemetryLoading} onChange={(event) => updateTelemetryPreferences({ input_telemetry_enabled: event.target.checked })} />
          </label>
          <label className="flex flex-between" style={{ padding: 14, border: "1px solid var(--color-border)", borderRadius: 10 }}>
            <span><strong style={{ display: "block", fontSize: 14 }}>浏览器域名统计</strong><span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>仅 Edge / Chrome 活动标签页域名</span></span>
            <input type="checkbox" checked={telemetry?.preferences.browser_tracking_enabled ?? false} disabled={loading || telemetryLoading} onChange={(event) => updateTelemetryPreferences({ browser_tracking_enabled: event.target.checked })} />
          </label>
        </div>
        <div className="form-row mt16" style={{ alignItems: "end" }}>
          <div className="form-group"><label>输入桶保留天数</label><select value={telemetry?.preferences.interaction_retention_days ?? 7} disabled={loading || telemetryLoading} onChange={(event) => updateTelemetryPreferences({ interaction_retention_days: Number(event.target.value) })}>{[1, 3, 7, 14, 30].map((days) => <option key={days} value={days}>{days} 天</option>)}</select></div>
          <div className="form-group"><label>活动与浏览器片段保留天数</label><select value={telemetry?.preferences.activity_retention_days ?? 30} disabled={loading || telemetryLoading} onChange={(event) => updateTelemetryPreferences({ activity_retention_days: Number(event.target.value) })}>{[7, 14, 30, 60, 90].map((days) => <option key={days} value={days}>{days} 天</option>)}</select></div>
          <button className="btn btn-primary" disabled={loading || telemetryLoading} onClick={handleCreatePairingCode}>生成浏览器配对码</button>
        </div>
        {pairingCode && <div style={{ marginTop: 16, padding: 16, borderRadius: 10, background: "var(--color-bg-secondary)" }}><div style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>在扩展设置页输入，5 分钟内有效</div><div style={{ fontSize: 30, fontWeight: 700, letterSpacing: 8, marginTop: 6 }}>{pairingCode.code}</div><div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>到期：{formatTelemetryTime(pairingCode.expires_at)}</div></div>}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginTop: 18 }}>
          <div><span className="text-muted">数据库占用</span><strong style={{ display: "block" }}>{formatBytes(telemetry?.database_size_bytes ?? 0)}</strong></div>
          <div><span className="text-muted">今日输入桶</span><strong style={{ display: "block" }}>{telemetry?.interaction_bucket_count ?? 0}</strong></div>
          <div><span className="text-muted">今日浏览器片段</span><strong style={{ display: "block" }}>{telemetry?.browser_segment_count ?? 0}</strong></div>
          <div><span className="text-muted">最后输入采集</span><strong style={{ display: "block", fontSize: 12 }}>{formatTelemetryTime(telemetry?.last_interaction_at ?? null)}</strong></div>
          <div><span className="text-muted">最后浏览器采集</span><strong style={{ display: "block", fontSize: 12 }}>{formatTelemetryTime(telemetry?.last_browser_at ?? null)}</strong></div>
        </div>
        <div className="flex gap8 mt16" style={{ flexWrap: "wrap" }}>
          <button className="btn btn-ghost btn-sm" disabled={loading || telemetryLoading} onClick={() => handleClearTelemetry("interaction")}>清除输入数据</button>
          <button className="btn btn-ghost btn-sm" disabled={loading || telemetryLoading} onClick={() => handleClearTelemetry("browser")}>清除浏览器数据</button>
          <button className="btn btn-ghost btn-sm" disabled={loading || telemetryLoading} onClick={() => handleClearTelemetry("feedback")}>清除反馈数据</button>
          <button className="btn btn-danger btn-sm" disabled={loading || telemetryLoading} onClick={() => handleClearTelemetry("all")}>清除全部行为数据</button>
        </div>
      </div>

      {/* 2. Data Collection */}
      <div className="card mb24">
        <div className="flex-between mb16">
          <h3 style={{ marginBottom: 0 }}>数据采集</h3>
          <span className={`badge ${isCollectorRunning ? "badge-success" : "badge-warning"}`}>
            {isCollectorRunning ? "运行中" : "已停止"}
          </span>
        </div>
        {collector?.status && (
          <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
            {collector.status}
          </div>
        )}
        <button
          className={`btn btn-sm ${isCollectorRunning ? "btn-danger" : ""}`}
          onClick={handleCollectorToggle}
          disabled={loading || collectorLoading}
        >
          {collectorLoading ? "处理中..." : isCollectorRunning ? "停止采集" : "启动采集"}
        </button>
      </div>

      {/* 3. Autonomy Control */}
      <div className="card mb24">
        <div className="flex-between mb16">
          <div>
            <h3 style={{ marginBottom: 4 }}>自主控制</h3>
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
              {autonomyPaused
                ? `已暂停${autonomy?.paused_until ? `，预计 ${new Date(autonomy.paused_until).toLocaleString("zh-CN")} 恢复` : ""}`
                : autonomy?.enabled !== false
                  ? "自主模式运行中"
                  : "自主模式已关闭"}
            </div>
          </div>
          <span className={`badge ${autonomyPaused ? "badge-warning" : autonomy?.enabled !== false ? "badge-success" : "badge-info"}`}>
            {autonomyPaused ? "已暂停" : autonomy?.enabled !== false ? "运行中" : "已关闭"}
          </span>
        </div>
        <div className="flex gap8" style={{ alignItems: "center" }}>
          {autonomyPaused ? (
            <button className="btn btn-sm" onClick={handleResume} disabled={loading || autonomyLoading}>
              {autonomyLoading ? "处理中..." : "恢复自主模式"}
            </button>
          ) : (
            <>
              <input
                type="number"
                min={1}
                max={72}
                value={pauseHours}
                onChange={(e) => setPauseHours(Number(e.target.value))}
                disabled={loading || autonomyLoading}
                style={{ width: 80 }}
              />
              <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>小时</span>
              <button className="btn btn-sm btn-danger" onClick={handlePause} disabled={loading || autonomyLoading}>
                {autonomyLoading ? "处理中..." : "暂停"}
              </button>
            </>
          )}
        </div>
      </div>

      {/* 4. App Classifications */}
      <div className="card mb24">
        <div className="flex-between mb16">
          <h3 style={{ marginBottom: 0 }}>应用分类</h3>
          <button className="btn btn-sm btn-ghost" onClick={handleFetchUnknown} disabled={fetchingUnknown}>
            {fetchingUnknown ? "获取中..." : "获取未知应用"}
          </button>
        </div>

        {unknownApps.length > 0 && (
          <div className="mb16" style={{ background: "var(--color-bg-inset)", padding: 12, borderRadius: 8 }}>
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 8 }}>未分类应用</div>
            <div className="flex gap8" style={{ flexWrap: "wrap" }}>
              {unknownApps.map((app, idx) => (
                <span key={idx} className="badge badge-warning">{app}</span>
              ))}
            </div>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 140px 80px auto", gap: 8, marginBottom: 12 }}>
          <input
            placeholder="进程名"
            value={newRule.process_name}
            onChange={(e) => setNewRule((r) => ({ ...r, process_name: e.target.value }))}
          />
          <input
            placeholder="窗口标题模式"
            value={newRule.window_title_pattern ?? ""}
            onChange={(e) => setNewRule((r) => ({ ...r, window_title_pattern: e.target.value }))}
          />
          <select
            value={newRule.category}
            onChange={(e) => setNewRule((r) => ({ ...r, category: e.target.value }))}
          >
            {CATEGORY_OPTIONS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <input
            type="number"
            placeholder="优先级"
            value={newRule.priority}
            onChange={(e) => setNewRule((r) => ({ ...r, priority: Number(e.target.value) }))}
          />
          <button className="btn btn-sm" onClick={handleAddRule} disabled={addingRule || !newRule.process_name}>
            {addingRule ? "添加中..." : "添加"}
          </button>
        </div>

        {classifications.length === 0 ? (
          <div style={{ fontSize: 13, color: "var(--color-text-tertiary)" }}>暂无分类规则</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>进程名</th>
                <th>窗口标题模式</th>
                <th>分类</th>
                <th>优先级</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {classifications.map((rule) => (
                <tr key={rule.id}>
                  <td>{rule.process_name}</td>
                  <td>{rule.window_title_pattern || "--"}</td>
                  <td>
                    <span className={`badge ${
                      rule.category === "productive" ? "badge-success"
                      : rule.category === "distracting" ? "badge-danger"
                      : rule.category === "neutral" ? "badge-info"
                      : "badge-warning"
                    }`}>
                      {rule.category}
                    </span>
                  </td>
                  <td>{rule.priority}</td>
                  <td>
                    <button
                      className="btn btn-sm btn-danger"
                      onClick={() => rule.id != null && handleDeleteRule(rule.id)}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* 5. Data Export */}
      <div className="card mb24">
        <h3>数据导出</h3>
        <div className="flex gap16" style={{ alignItems: "flex-end", flexWrap: "wrap" }}>
          <div style={{ minWidth: 100 }}>
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 4 }}>格式</div>
            <select value={exportFmt} onChange={(e) => setExportFmt(e.target.value as "csv" | "json")}>
              <option value="csv">CSV</option>
              <option value="json">JSON</option>
            </select>
          </div>
          <div>
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 4 }}>开始日期</div>
            <input type="date" value={exportStart} onChange={(e) => setExportStart(e.target.value)} />
          </div>
          <div>
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 4 }}>结束日期</div>
            <input type="date" value={exportEnd} onChange={(e) => setExportEnd(e.target.value)} />
          </div>
          <button className="btn" onClick={handleExport} disabled={exporting}>
            {exporting ? "导出中..." : "导出"}
          </button>
        </div>
      </div>

      {/* 6. Preferences */}
      <div className="card mb24">
        <h3>偏好设置</h3>
        <textarea
          value={preferences}
          onChange={(e) => setPreferences(e.target.value)}
          rows={12}
          style={{ fontFamily: "monospace", fontSize: 13, marginBottom: 12 }}
          placeholder="{}"
        />
        <div className="flex gap8">
          <button className="btn btn-sm" onClick={handlePutPrefs} disabled={prefLoading}>
            {prefLoading ? "保存中..." : "PUT 全量更新"}
          </button>
          <button className="btn btn-sm btn-ghost" onClick={handlePatchPrefs} disabled={prefLoading}>
            {prefLoading ? "更新中..." : "PATCH 增量更新"}
          </button>
        </div>
      </div>
    </div>
  );
}
