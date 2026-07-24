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
} from "../api";
import type { HealthData } from "../api";

interface ClassRule {
  id?: number;
  process_name: string;
  window_title_pattern: string;
  category: string;
  priority: number;
}

const CATEGORY_OPTIONS = ["productive", "neutral", "distracting", "unknown"];

export default function Settings() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthData | null>(null);
  const [collector, setCollector] = useState<any>(null);
  const [autonomy, setAutonomy] = useState<any>(null);
  const [classifications, setClassifications] = useState<ClassRule[]>([]);
  const [preferences, setPreferences] = useState("");

  const [collectorLoading, setCollectorLoading] = useState(false);
  const [autonomyLoading, setAutonomyLoading] = useState(false);
  const [pauseHours, setPauseHours] = useState(1);

  const [newRule, setNewRule] = useState<ClassRule>({
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

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, cs, au, cls, prefs] = await Promise.all([
        getHealth(),
        getCollectorStatus(),
        getAutonomy(),
        getClassifications(),
        getPreferences(),
      ]);
      setHealth(h);
      setCollector(cs);
      setAutonomy(au);
      setClassifications(Array.isArray(cls) ? cls : []);
      setPreferences(JSON.stringify(prefs, null, 2));
    } catch (e: any) {
      setError(e.message ?? "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleCollectorToggle = async () => {
    setCollectorLoading(true);
    try {
      const next = collector?.running ? await stopCollector() : await startCollector();
      setCollector(next);
    } catch (e: any) {
      setError(e.message ?? "操作失败");
    } finally {
      setCollectorLoading(false);
    }
  };

  const handlePause = async () => {
    setAutonomyLoading(true);
    try {
      const result = await pauseAutonomy(pauseHours);
      setAutonomy(result);
    } catch (e: any) {
      setError(e.message ?? "操作失败");
    } finally {
      setAutonomyLoading(false);
    }
  };

  const handleResume = async () => {
    setAutonomyLoading(true);
    try {
      const result = await resumeAutonomy();
      setAutonomy(result);
    } catch (e: any) {
      setError(e.message ?? "操作失败");
    } finally {
      setAutonomyLoading(false);
    }
  };

  const handleFetchUnknown = async () => {
    setFetchingUnknown(true);
    try {
      const apps = await getUnknownApps();
      setUnknownApps(Array.isArray(apps) ? apps : []);
    } catch (e: any) {
      setError(e.message ?? "获取失败");
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
    } catch (e: any) {
      setError(e.message ?? "添加失败");
    } finally {
      setAddingRule(false);
    }
  };

  const handleDeleteRule = async (id: number) => {
    try {
      await deleteClassification(id);
      setClassifications((prev) => prev.filter((r) => r.id !== id));
    } catch (e: any) {
      setError(e.message ?? "删除失败");
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
    } catch (e: any) {
      setError(e.message ?? "导出失败");
    } finally {
      setExporting(false);
    }
  };

  const handlePutPrefs = async () => {
    setPrefLoading(true);
    try {
      let parsed: any;
      try {
        parsed = JSON.parse(preferences);
      } catch {
        setError("JSON 格式无效");
        setPrefLoading(false);
        return;
      }
      const result = await putPreferences(parsed);
      setPreferences(JSON.stringify(result, null, 2));
    } catch (e: any) {
      setError(e.message ?? "保存失败");
    } finally {
      setPrefLoading(false);
    }
  };

  const handlePatchPrefs = async () => {
    setPrefLoading(true);
    try {
      let parsed: any;
      try {
        parsed = JSON.parse(preferences);
      } catch {
        setError("JSON 格式无效");
        setPrefLoading(false);
        return;
      }
      const result = await patchPreferences(parsed);
      setPreferences(JSON.stringify(result, null, 2));
    } catch (e: any) {
      setError(e.message ?? "更新失败");
    } finally {
      setPrefLoading(false);
    }
  };

  if (loading) {
    return (
      <div>
        <div className="header">
          <h1>系统设置</h1>
          <p>管理 MindFlow 配置与数据</p>
        </div>
        <div className="spinner" />
      </div>
    );
  }

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

      {/* 2. Data Collection */}
      <div className="card mb24">
        <div className="flex-between mb16">
          <h3 style={{ marginBottom: 0 }}>数据采集</h3>
          <span className={`badge ${collector?.running ? "badge-success" : "badge-warning"}`}>
            {collector?.running ? "运行中" : "已停止"}
          </span>
        </div>
        {collector?.status && (
          <div style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 12 }}>
            {collector.status}
          </div>
        )}
        <button
          className={`btn btn-sm ${collector?.running ? "btn-danger" : ""}`}
          onClick={handleCollectorToggle}
          disabled={collectorLoading}
        >
          {collectorLoading ? "处理中..." : collector?.running ? "停止采集" : "启动采集"}
        </button>
      </div>

      {/* 3. Autonomy Control */}
      <div className="card mb24">
        <div className="flex-between mb16">
          <div>
            <h3 style={{ marginBottom: 4 }}>自主控制</h3>
            <div style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>
              {autonomy?.paused
                ? `已暂停${autonomy?.paused_until ? `，预计 ${new Date(autonomy.paused_until).toLocaleString("zh-CN")} 恢复` : ""}`
                : autonomy?.enabled !== false
                  ? "自主模式运行中"
                  : "自主模式已关闭"}
            </div>
          </div>
          <span className={`badge ${autonomy?.paused ? "badge-warning" : autonomy?.enabled !== false ? "badge-success" : "badge-info"}`}>
            {autonomy?.paused ? "已暂停" : autonomy?.enabled !== false ? "运行中" : "已关闭"}
          </span>
        </div>
        <div className="flex gap8" style={{ alignItems: "center" }}>
          {autonomy?.paused ? (
            <button className="btn btn-sm" onClick={handleResume} disabled={autonomyLoading}>
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
                style={{ width: 80 }}
              />
              <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>小时</span>
              <button className="btn btn-sm btn-danger" onClick={handlePause} disabled={autonomyLoading}>
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
            value={newRule.window_title_pattern}
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
