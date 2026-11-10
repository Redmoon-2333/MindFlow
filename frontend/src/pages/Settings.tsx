import { userTerm } from "../lib/user-terms";
import WorkspacePage from "../components/WorkspacePage";
import AppearanceSettings from "../components/AppearanceSettings";
import AISettings from "../components/AISettings";
import Segmented from "../components/signal/Segmented";
import HelpTip from "../components/signal/HelpTip";
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
  getTelemetryStatus,
  patchTelemetryPreferences,
  createBrowserPairingCode,
  clearTelemetryData,
  getAiUsage,
  getErrorMessage,
  isAutonomyPaused,
  runTelemetryDelete,
} from "../api";
import type { AiUsage, AutonomyStatus, ClassificationRule, ClassificationRuleInput, CollectorStatus, HealthData, TelemetryDeleteNotice, TelemetryDeleteScope, TelemetryPreferences, TelemetryStatus } from "../api";

/** §8.16.2: the export section no longer shares a tab with a JSON editor. */
const SETTINGS_SECTIONS = ["采集与隐私", "应用分类", "数据导出", "外观与 AI", "系统状态"] as const;

/** §8.17.2: every rule row shows a Chinese class name, including the advanced
 *  categories a historical rule may still carry. */
const CATEGORY_LABELS: Record<string, string> = {
  code: "工作软件",
  document: "文档",
  browser_work: "浏览器工作",
  communication: "沟通",
  entertainment: "娱乐软件",
  social: "社交",
  other: "不一定",
};

const USER_SOFT_CLASSIFICATION_OPTIONS: ReadonlyArray<{
  label: string;
  value: ClassificationRuleInput["category"];
  hint: string;
}> = [
  { label: "工作软件", value: "code", hint: "训练与提醒中按工作软件处理" },
  { label: "娱乐软件", value: "entertainment", hint: "参与训练与统计" },
  { label: "不一定", value: "other", hint: "不建新类别,回退自动与弱标签" },
];

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

/** One unclassified application, as returned by GET /app-classifications/unknown-apps. */
interface UnknownApp {
  process_name: string;
  count: number;
  last_seen: string | null;
}

/** Coerce the unknown-apps payload into UnknownApp objects.
 *
 *  Tolerates a bare string (older/other backends) so the UI keeps working, but
 *  the object shape is what the current API returns. */
function normaliseUnknownApps(value: unknown): UnknownApp[] {
  if (!Array.isArray(value)) return [];
  const out: UnknownApp[] = [];
  for (const entry of value) {
    if (typeof entry === "string") {
      out.push({ process_name: entry, count: 0, last_seen: null });
      continue;
    }
    if (typeof entry === "object" && entry !== null && "process_name" in entry) {
      const record = entry as Record<string, unknown>;
      const name = record.process_name;
      if (typeof name !== "string" || !name) continue;
      out.push({
        process_name: name,
        count: typeof record.count === "number" ? record.count : 0,
        last_seen: typeof record.last_seen === "string" ? record.last_seen : null,
      });
    }
  }
  return out;
}

export default function Settings() {
  const [section, setSection] = useState("采集与隐私");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthData | null>(null);
  const [collector, setCollector] = useState<CollectorStatus | null>(null);
  const [autonomy, setAutonomy] = useState<AutonomyStatus | null>(null);
  const [classifications, setClassifications] = useState<ClassificationRule[]>([]);

  const [collectorLoading, setCollectorLoading] = useState(false);
  const [autonomyLoading, setAutonomyLoading] = useState(false);
  const [pauseHours, setPauseHours] = useState(1);

  const [newRule, setNewRule] = useState<ClassificationRuleInput>({
    process_name: "",
    window_title_pattern: "",
    category: "other",
    priority: 0,
  });
  const [unknownApps, setUnknownApps] = useState<UnknownApp[]>([]);
  const [fetchingUnknown, setFetchingUnknown] = useState(false);
  const [addingRule, setAddingRule] = useState(false);

  const [exportFmt, setExportFmt] = useState<"csv" | "json">("csv");
  const [exportStart, setExportStart] = useState("");
  const [exportEnd, setExportEnd] = useState("");
  const [exporting, setExporting] = useState(false);

  const isCollectorRunning = collector?.running === true || collector?.status === "running";
  const autonomyPaused = isAutonomyPaused(autonomy);
  const [telemetry, setTelemetry] = useState<TelemetryStatus | null>(null);
  const [telemetryLoading, setTelemetryLoading] = useState(false);
  const [pairingCode, setPairingCode] = useState<{ code: string; expires_at: string } | null>(null);
  const [telemetryMessage, setTelemetryMessage] = useState<string | null>(null);
  const [telemetryDeleteNotice, setTelemetryDeleteNotice] = useState<TelemetryDeleteNotice | null>(null);
  const [aiUsage, setAiUsage] = useState<AiUsage | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    const [healthResult, collectorResult, autonomyResult, classificationsResult, telemetryResult] = await Promise.allSettled([
      getHealth(),
      getCollectorStatus(),
      getAutonomy(),
      getClassifications(),
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
    if (telemetryResult.status === "fulfilled") setTelemetry(telemetryResult.value);
    else failures.push(getErrorMessage(telemetryResult.reason, "遥测状态加载失败"));
    try { setAiUsage(await getAiUsage()); } catch { /* non-critical */ }

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
      // The backend returns objects ({process_name, count, last_seen}), not
      // bare strings. Rendering them directly as strings showed "[object
      // Object]" and put that into the new-rule form's process_name.
      const apps = await getUnknownApps();
      setUnknownApps(normaliseUnknownApps(apps));
    } catch (e: unknown) {
      setError(getErrorMessage(e, "获取失败"));
    } finally {
      setFetchingUnknown(false);
    }
  };

  const fillUnknownApp = (app: UnknownApp) => {
    // Strip a Windows executable suffix only for display in the rule form;
    // the stored value stays exactly as it appeared in the events.
    setNewRule((prev) => ({
      ...prev,
      process_name: app.process_name,
      category: prev.category ?? "other",
    }));
  };

  const handleAddRule = async () => {
    if (!newRule.process_name) return;
    setAddingRule(true);
    try {
      // §8.16.3: no priority field for the user. The backend matches by
      // priority DESC, so a rule carrying a title condition is raised above
      // the general rule for the same app automatically.
      const priority = (newRule.window_title_pattern ?? "").trim() ? 10 : 0;
      await addClassification({ ...newRule, priority });
      setNewRule({ process_name: "", window_title_pattern: "", category: "other", priority: 0 });
      const cls = await getClassifications();
      setClassifications(Array.isArray(cls) ? cls : []);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "添加失败"));
    } finally {
      setAddingRule(false);
    }
  };

  const handleDeleteRule = async (id: string) => {
    if (!window.confirm("确定删除这条分类规则吗？")) return;
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
    // §8.17.2: the confirmation repeats the button's own words and what the
    // user loses, instead of a generic "medicine" warning.
    const labels: Record<TelemetryDeleteScope, { name: string; effect: string }> = {
      interaction: {
        name: "清除输入数据",
        effect: "同时会删除由它算出的行为特征，以及你在“模型校准”里的回答。删除后无法恢复。",
      },
      browser: {
        name: "清除浏览器数据",
        effect: "同时会解除浏览器扩展的配对，并删除行为特征和“模型校准”里的回答。删除后无法恢复。",
      },
      feedback: {
        name: "清除反馈数据",
        effect: "“模型校准”里的回答不会被删除。删除后无法恢复。",
      },
      all: {
        name: "清除全部行为数据",
        effect: "聊天记录、设置和 AI Key 会保留；每天的自动备份里可能仍有旧数据。删除后无法恢复。",
      },
    };
    if (!window.confirm(`确定${labels[scope].name}吗？
${labels[scope].effect}`)) return;
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

  return (
    <WorkspacePage title="系统设置" description="决定采集哪些数据、哪些应用属于工作，以及数据如何保留。设置只作用于本机。" className="settings-page">


      {error && (
        <div className="error-box mb16">
          {error}
          <button className="btn btn-secondary btn-sm settings-error-close" onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}

      {loading && (
        <div className="card mb24" role="status">
          <div className="settings-loading">
            <span className="spinner spinner--inline" />
            正在加载设置，部分操作暂不可用
          </div>
        </div>
      )}

      <div className="page-toolbar">
        <Segmented
          ariaLabel="设置分区"
          options={SETTINGS_SECTIONS.map((name) => ({ value: name, label: name }))}
          value={section}
          onChange={setSection}
        />
      </div>
      {section === "外观与 AI" && <><AppearanceSettings /><AISettings /></>}
      {section === "系统状态" && <>
      {/* 1. System Info */}
      <div className="card mb24">
        <h3>系统信息</h3>
        <div className="settings-kv-list">
          <div className="kv-row">
            <span>状态</span>
            <span className={`badge ${health?.status === "ok" || health?.status === "healthy" ? "badge-success" : "badge-danger"}`}>
              {!health ? "未知" : health.status === "ok" || health.status === "healthy" ? "正常" : "异常"}
            </span>
          </div>
          <div className="kv-row"><span>版本</span><span>{health?.version ?? "—"}</span></div>
          {health?.database != null && (
            <div className="kv-row kv-row--text"><span>数据库</span><span>{health.database.status}</span></div>
          )}
          {health?.migration != null && (
            <div className="kv-row kv-row--text"><span>数据库版本</span><span>{health.migration.applied ? "已是最新" : "待更新"}</span></div>
          )}
        </div>
      </div>

      <div className="card mb24">
        <h3>AI 用量</h3>
        {aiUsage ? (
          <div className="settings-kv-list">
            <div className="kv-row"><span>近 30 天 AI 分析</span><span>{aiUsage.llm_calls_30d} 次</span></div>
            <div className="kv-row"><span>专家面板</span><span>{aiUsage.panel_count_30d} 次</span></div>
            {/* §8.17.2: `mode` reflects the ML model state, not cloud-AI spend,
                so the cost row is always shown. */}
            <div className="kv-row">
              <span>
                估算费用
                <HelpTip
                  term="估算费用"
                  what="近 30 天调用云端 AI 的估算花费，实际以服务商账单为准。"
                />
              </span>
              <span>{aiUsage.llm_cost_usd_30d.toFixed(4)} USD</span>
            </div>
          </div>
        ) : (
          <p className="settings-empty">暂无用量数据</p>
        )}
      </div>
      </>}
      {section === "采集与隐私" && <>
      <div className="card mb24 privacy-settings">
        <div className="flex flex-between">
          <div>
            <h3>隐私行为采集</h3>
            <p className="card-desc">
              只保存 30 秒聚合计数和浏览器域名，不记录按键内容、鼠标坐标或完整网址。
            </p>
          </div>
          <span className={`badge ${telemetry?.input_watcher_status === "running" ? "badge-success" : "badge-warning"}`}>
            输入采集器：{userTerm(telemetry?.input_watcher_status)}
          </span>
        </div>
        {telemetryMessage && <div className="success-box mt16">{telemetryMessage}</div>}
        {telemetryDeleteNotice?.kind === "success" && (
          <div className="success-box mt16">{telemetryDeleteNotice.message}</div>
        )}
        {telemetryDeleteNotice?.kind === "partial" && (
          <div className="error-box mt16" role="status">
            <strong>{telemetryDeleteNotice.message}</strong>
            <div className="settings-error-detail">未完成项目：</div>
            <ul className="settings-error-list">
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
        <div className="settings-switches">
          <label className="settings-switch">
            <span>
              <span className="settings-switch-title">
                鼠标与键盘聚合统计
                <HelpTip
                  term="键鼠统计"
                  what="每 30 秒记一次你点了几下鼠标、按了几次键盘、滚动了多少，不记录按了哪些键。"
                  effect="开启后马上开始统计，帮助更准确地判断专注；关闭后停止统计，已经记下的数据不会自动删除。"
                />
              </span>
              <span className="settings-switch-desc">点击、滚动、移动距离和输入活跃秒数</span>
            </span>
            <input type="checkbox" checked={telemetry?.preferences.input_telemetry_enabled ?? false} disabled={loading || telemetryLoading} onChange={(event) => updateTelemetryPreferences({ input_telemetry_enabled: event.target.checked })} />
          </label>
          <label className="settings-switch">
            <span>
              <span className="settings-switch-title">
                浏览器域名统计
                <HelpTip
                  term="浏览器域名统计"
                  what="记录你在浏览器里正在看哪个网站（只记域名，比如 bilibili.com）。需要先安装并配对 MindFlow 浏览器扩展。"
                  effect="开启后能看出网页上的分心情况；无痕窗口永远不会被记录。"
                />
              </span>
              <span className="settings-switch-desc">仅 Edge / Chrome 活动标签页域名</span>
            </span>
            <input type="checkbox" checked={telemetry?.preferences.browser_tracking_enabled ?? false} disabled={loading || telemetryLoading} onChange={(event) => updateTelemetryPreferences({ browser_tracking_enabled: event.target.checked })} />
          </label>
        </div>
        <div className="settings-retention">
          <div className="form-group"><label>键盘鼠标活动统计保留天数<HelpTip term="保留天数" what="键盘鼠标统计数据保存多少天。" effect="改短后，超过天数的旧数据会在下次清理时被永久删除。" /></label><select aria-label="键盘鼠标活动统计保留天数" value={telemetry?.preferences.interaction_retention_days ?? 7} disabled={loading || telemetryLoading} onChange={(event) => updateTelemetryPreferences({ interaction_retention_days: Number(event.target.value) })}>{[1, 3, 7, 14, 30].map((days) => <option key={days} value={days}>{days} 天</option>)}</select></div>
          <div className="form-group"><label>应用与网页记录保留天数<HelpTip term="保留天数" what="软件、窗口和网页记录保存多少天。" effect="改短后，更早的记录会在每天凌晨 3 点被永久删除，报告和模型训练也会少掉这些数据。" /></label><select aria-label="应用与网页记录保留天数" value={telemetry?.preferences.activity_retention_days ?? 30} disabled={loading || telemetryLoading} onChange={(event) => updateTelemetryPreferences({ activity_retention_days: Number(event.target.value) })}>{[7, 14, 30, 60, 90].map((days) => <option key={days} value={days}>{days} 天</option>)}</select></div>
        </div>
        {pairingCode && (
          <div className="pairing-code">
            <div className="pairing-code-hint">在扩展设置页输入，5 分钟内有效</div>
            <div className="pairing-code-value">{pairingCode.code}</div>
            <div className="pairing-code-expiry">到期：{formatTelemetryTime(pairingCode.expires_at)}</div>
          </div>
        )}
        <div className="telemetry-stats">
          <div className="kv-row"><span>数据库占用</span><span>{formatBytes(telemetry?.database_size_bytes ?? 0)}</span></div>
          <div className="kv-row"><span>今日键鼠统计条数</span><span>{telemetry?.interaction_bucket_count ?? 0}</span></div>
          <div className="kv-row"><span>今日网页记录条数</span><span>{telemetry?.browser_segment_count ?? 0}</span></div>
          <div className="kv-row"><span>最后输入采集</span><span>{formatTelemetryTime(telemetry?.last_interaction_at ?? null)}</span></div>
          <div className="kv-row"><span>最后浏览器采集</span><span>{formatTelemetryTime(telemetry?.last_browser_at ?? null)}</span></div>
        </div>
        <div className="telemetry-actions">
          <button className="btn btn-primary btn-sm" disabled={loading || telemetryLoading} onClick={handleCreatePairingCode}>生成浏览器配对码</button>
          <HelpTip
            term="配对码"
            what="把 MindFlow 浏览器扩展和这台电脑上的 MindFlow 连起来的临时密码，5 分钟内有效。"
            effect="生成后会自动开启“浏览器域名统计”。配对成功后，扩展会上报你正在看的网站，并帮你拦截名单里的网站。"
          />
          <button className="btn btn-secondary btn-sm" disabled={loading || telemetryLoading} onClick={() => handleClearTelemetry("interaction")}>清除输入数据</button>
          <HelpTip
            term="清除输入数据"
            what="删除所有键盘鼠标统计。"
            effect="同时会删除由它算出的行为特征，以及你在“模型校准”里的回答。删除后无法恢复。"
          />
          <button className="btn btn-secondary btn-sm" disabled={loading || telemetryLoading} onClick={() => handleClearTelemetry("browser")}>清除浏览器数据</button>
          <HelpTip
            term="清除浏览器数据"
            what="删除所有网页记录。"
            effect="同时会解除浏览器扩展的配对，并删除行为特征和“模型校准”里的回答。删除后无法恢复。"
          />
          <button className="btn btn-secondary btn-sm" disabled={loading || telemetryLoading} onClick={() => handleClearTelemetry("feedback")}>清除反馈数据</button>
          <HelpTip
            term="清除反馈数据"
            what="删除你在“专注分析”页给每段会话的反馈。"
            effect="“模型校准”里的回答不会被删除。删除后无法恢复。"
          />
          <button className="btn btn-danger btn-sm" disabled={loading || telemetryLoading} onClick={() => handleClearTelemetry("all")}>清除全部行为数据</button>
          <HelpTip
            term="清除全部行为数据"
            what="删除所有记录、报告、分析、提醒记录和训练好的个人模型。"
            effect="聊天记录、设置和 AI Key 会保留；每天的自动备份里可能仍有旧数据。删除后无法恢复。"
          />
        </div>
      </div>

      {/* 2. Data Collection */}
      <div className="settings-control-grid">
      <div className="card settings-control-card">
        <div className="settings-control-info">
          <h3>数据采集</h3>
          <p className="card-desc">记录你正在使用的应用和窗口标题，用于专注分析；停止后不再记录。</p>
          <span className={`badge ${isCollectorRunning ? "badge-success" : "badge-warning"}`}>
            {isCollectorRunning ? "运行中" : "已停止"}
          </span>
        </div>
        <div className="workspace-actions">
          <button
            className={`btn btn-sm ${isCollectorRunning ? "btn-danger" : "btn-secondary"}`}
            onClick={handleCollectorToggle}
            disabled={loading || collectorLoading}
          >
            {collectorLoading ? "处理中…" : isCollectorRunning ? "停止采集" : "启动采集"}
          </button>
        </div>
      </div>

      {/* 3. Autonomy Control */}
      <div className="card settings-control-card">
        <div className="settings-control-info">
          <h3>
            自主控制
            <HelpTip
              term="自主模式"
              what="开启时，系统会自己判断什么时候提醒你，并每天自动生成一次 AI 分析。"
              effect="暂停后这两件事都会停下，时间到了自动恢复；记录不受影响。"
            />
          </h3>
          <p className="card-desc">
            {autonomyPaused
              ? `已暂停${autonomy?.paused_until ? `，预计 ${new Date(autonomy.paused_until).toLocaleString("zh-CN")} 恢复` : ""}`
              : autonomy?.enabled !== false
                ? "自主模式运行中"
                : "自主模式已关闭"}
          </p>
          <span className={`badge ${autonomyPaused ? "badge-warning" : autonomy?.enabled !== false ? "badge-success" : "badge-info"}`}>
            {autonomyPaused ? "已暂停" : autonomy?.enabled !== false ? "运行中" : "已关闭"}
          </span>
        </div>
        <div className="workspace-actions">
          {autonomyPaused ? (
            <button className="btn btn-secondary btn-sm" onClick={handleResume} disabled={loading || autonomyLoading}>
              {autonomyLoading ? "处理中…" : "恢复自主模式"}
            </button>
          ) : (
            <>
              {/* §8.16/§8.22.5: a fixed set of durations, not a free number —
                  the reader should not have to invent a valid value. */}
              <select
                aria-label="暂停自主干预小时数"
                value={pauseHours}
                onChange={(e) => setPauseHours(Number(e.target.value))}
                disabled={loading || autonomyLoading}
                className="settings-hours"
              >
                {[1, 2, 4, 8, 24].map((hours) => (
                  <option key={hours} value={hours}>{hours} 小时</option>
                ))}
              </select>
              <button className="btn btn-sm btn-danger" onClick={handlePause} disabled={loading || autonomyLoading}>
                {autonomyLoading ? "处理中…" : "暂停"}
              </button>
            </>
          )}
        </div>
      </div>

      </div>
      </>}
      {section === "应用分类" && <>
      {/* 4. App Classifications */}
      <div className="card mb24">
        <div className="flex-between mb16">
          <h3>
            应用分类
            <HelpTip
              term="应用分类"
              what="告诉系统某个软件是工作用还是娱乐用。"
              effect="设成“工作软件”后，你在用它时系统会尽量不打扰你；分类也会用于训练个人模型。"
            />
          </h3>
          <button className="btn btn-secondary btn-sm" onClick={handleFetchUnknown} disabled={fetchingUnknown}>
            {fetchingUnknown ? "获取中…" : "获取未知应用"}
          </button>
          <HelpTip term="未知应用" what="列出最近用过、但还没分类的软件。" effect="只是查看，不会改动任何东西。" />
        </div>

        {unknownApps.length > 0 && (
          <div className="settings-unknown-apps">
            <div className="settings-unknown-hint">点击应用名可直接填入下方表单，再选择分类即可。</div>
            <div className="settings-unknown-list">
              {unknownApps.map((app) => (
                <button
                  key={app.process_name}
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => fillUnknownApp(app)}
                  title={`为 ${app.process_name} 新建分类规则（出现 ${app.count} 次）`}
                >
                  {app.process_name}
                  {app.count > 0 && <span className="settings-unknown-count">×{app.count}</span>}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="settings-classification-form">
          <input
            placeholder="应用进程名，例如 chrome.exe"
            aria-label="进程名"
            value={newRule.process_name}
            onChange={(e) => setNewRule((r) => ({ ...r, process_name: e.target.value }))}
          />
          <select
            aria-label="分类"
            value={newRule.category}
            onChange={(e) => setNewRule((r) => ({ ...r, category: e.target.value as ClassificationRuleInput["category"] }))}
          >
            {USER_SOFT_CLASSIFICATION_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{`${option.label} · ${option.hint}`}</option>
            ))}
          </select>
          <button className="btn btn-primary btn-sm" onClick={handleAddRule} disabled={addingRule || !newRule.process_name}>
            {addingRule ? "添加中…" : "添加"}
          </button>
        </div>

        <details className="settings-rule-more">
          <summary>更多条件（可选）</summary>
          <div className="settings-rule-more-body">
            <input
              placeholder="窗口标题包含的文字，例如 GitHub"
              aria-label="窗口标题包含"
              value={newRule.window_title_pattern ?? ""}
              onChange={(e) => setNewRule((r) => ({ ...r, window_title_pattern: e.target.value }))}
            />
            <p className="settings-rule-more-hint">
              只在窗口标题包含这段文字时才按此分类，不区分大小写。
            </p>
          </div>
        </details>

        {classifications.length === 0 ? (
          <p className="settings-empty">暂无分类规则</p>
        ) : (
          <div className="workspace-table"><table>
            <thead>
              <tr>
                <th>进程名</th>
                <th>窗口标题包含</th>
                <th>分类</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {classifications.map((rule) => (
                <tr key={rule.id}>
                  <td>{rule.process_name}</td>
                  <td>{rule.window_title_pattern || "—"}</td>
                  <td>
                    <span className={`badge ${["code", "document", "browser_work"].includes(rule.category) ? "badge-success" : ["entertainment", "social"].includes(rule.category) ? "badge-danger" : "badge-info"}`}>
                      {CATEGORY_LABELS[rule.category] ?? rule.category}
                    </span>
                  </td>
                  <td>
                    <button
                      className="btn btn-danger btn-sm"
                      onClick={() => rule.id != null && handleDeleteRule(rule.id)}
                    >
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table></div>
        )}
      </div>

      </>}
      {section === "数据导出" && <>
      {/* 5. Data Export */}
      <div className="card mb24">
        <h3>数据导出</h3>
        <div className="settings-export">
          <div className="settings-export-field">
            <label className="settings-export-label" htmlFor="export-format">格式</label>
            <select id="export-format" aria-label="导出格式" value={exportFmt} onChange={(e) => setExportFmt(e.target.value as "csv" | "json")}>
              <option value="csv">CSV（可用 Excel 打开）</option>
              <option value="json">JSON</option>
            </select>
          </div>
          <div className="settings-export-field">
            <label className="settings-export-label" htmlFor="export-start">开始日期</label>
            <input id="export-start" type="date" aria-label="导出开始日期" value={exportStart} onChange={(e) => setExportStart(e.target.value)} />
          </div>
          <div className="settings-export-field">
            <label className="settings-export-label" htmlFor="export-end">结束日期</label>
            <input id="export-end" type="date" aria-label="导出结束日期" value={exportEnd} onChange={(e) => setExportEnd(e.target.value)} />
          </div>
          <button className="btn btn-secondary settings-export-submit" onClick={handleExport} disabled={exporting}>
            {exporting ? "导出中…" : "导出"}
          </button>
          <HelpTip
            term="导出"
            what="把记录下载成文件。CSV 文件可以用 Excel 打开。"
            effect="文件里包含窗口标题等隐私信息，请妥善保管。"
          />
        </div>
      </div>
      </>}
    </WorkspacePage>
  );
}
