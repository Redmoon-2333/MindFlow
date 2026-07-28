import { useState, useEffect, useCallback } from "react";
import {
  getAnalyticsPatterns,
  getBaseline,
  getProfile,
  getModelStatus,
  runAttribution,
  getErrorMessage,
} from "../api";
import type {
  AnalyticsPatterns,
  AttributionResponse,
  BaselineSummary,
  BehavioralProfile,
  ModelStatus,
} from "../api";

const DAYS_OPTIONS = [7, 14, 30, 90];
const TABS = ["模式分析", "个人画像", "拖延归因", "模型状态"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function profileDetailValue(value: unknown): string | number {
  const detailValue = isRecord(value) ? value.value : value;
  if (typeof detailValue === "string" || typeof detailValue === "number") return detailValue;
  if (typeof detailValue === "boolean") return String(detailValue);
  return "N/A";
}

function profileDetailTrend(value: unknown): string {
  if (!isRecord(value) || typeof value.trend !== "string") return "—";
  return value.trend;
}

export default function Analytics() {
  const [days, setDays] = useState(14);
  const [activeTab, setActiveTab] = useState(TABS[0]);

  const [patterns, setPatterns] = useState<AnalyticsPatterns | null>(null);
  const [baseline, setBaseline] = useState<BaselineSummary | null>(null);
  const [profile, setProfile] = useState<BehavioralProfile | null>(null);
  const [modelStatus, setModelStatusState] = useState<ModelStatus | null>(null);
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null);

  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);

  const fetchPatterns = useCallback(async () => {
    setLoading((p) => ({ ...p, patterns: true }));
    try {
      const data = await getAnalyticsPatterns(days);
      setPatterns(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "模式分析加载失败"));
    } finally {
      setLoading((p) => ({ ...p, patterns: false }));
    }
  }, [days]);

  const fetchBaseline = useCallback(async () => {
    setLoading((p) => ({ ...p, baseline: true }));
    try {
      const data = await getBaseline();
      setBaseline(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "基线数据加载失败"));
    } finally {
      setLoading((p) => ({ ...p, baseline: false }));
    }
  }, []);

  const fetchProfile = useCallback(async () => {
    setLoading((p) => ({ ...p, profile: true }));
    try {
      const data = await getProfile(days);
      setProfile(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "个人画像加载失败"));
    } finally {
      setLoading((p) => ({ ...p, profile: false }));
    }
  }, [days]);

  const fetchModelStatus = useCallback(async () => {
    setLoading((p) => ({ ...p, modelStatus: true }));
    try {
      const data = await getModelStatus();
      setModelStatusState(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "模型状态加载失败"));
    } finally {
      setLoading((p) => ({ ...p, modelStatus: false }));
    }
  }, []);

  useEffect(() => { fetchBaseline(); fetchModelStatus(); }, [fetchBaseline, fetchModelStatus]);
  useEffect(() => { fetchPatterns(); fetchProfile(); }, [fetchPatterns, fetchProfile]);

  const handleAttribution = async () => {
    setLoading((p) => ({ ...p, attribution: true }));
    setAttribution(null);
    try {
      const data = await runAttribution();
      setAttribution(data);
    } catch (e: unknown) {
      setError(getErrorMessage(e, "归因分析失败"));
    } finally {
      setLoading((p) => ({ ...p, attribution: false }));
    }
  };

  const renderLoading = (key: string) => {
    if (loading[key]) return <div className="spinner" />;
    return null;
  };

  const badgeClass = (value: unknown) => {
    const map: Record<string, string> = { high: "badge-danger", medium: "badge-warning", low: "badge-success" };
    return typeof value === "string" ? map[value.toLowerCase()] || "badge-info" : "badge-info";
  };

  return (
    <div>
      <div className="header">
        <h1>行为洞察</h1>
        <p>深度分析你的行为模式，识别拖延根源，获取个性化洞察</p>
      </div>

      <div className="flex flex-between mb24">
        <div className="tabs" style={{ marginBottom: 0 }}>
          {TABS.map((t) => (
            <button
              key={t}
              className={`tab${activeTab === t ? " active" : ""}`}
              onClick={() => setActiveTab(t)}
            >
              {t}
            </button>
          ))}
        </div>

        {(activeTab === "模式分析" || activeTab === "个人画像") && (
          <div className="flex gap8" style={{ alignItems: "center" }}>
            <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>时间范围</span>
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              style={{ width: "auto" }}
            >
              {DAYS_OPTIONS.map((d) => (
                <option key={d} value={d}>近 {d} 天</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {error && (
        <div className="error-box">
          {error}
          <button className="btn btn-sm" style={{ marginLeft: 12 }} onClick={() => setError(null)}>
            关闭
          </button>
        </div>
      )}

      {/* ── 模式分析 Tab ── */}
      {activeTab === "模式分析" && (
        <div className="flex gap16" style={{ flexDirection: "column" }}>
          <div className="flex gap16">
            <div className="card" style={{ flex: 1 }}>
              <h3>高切换时段</h3>
              {renderLoading("patterns")}
              {patterns?.high_switch_periods?.length > 0 ? (
                <ul style={{ listStyle: "none", padding: 0 }}>
                  {patterns.high_switch_periods.map((p, i) => (
                    <li
                      key={i}
                      className="flex flex-between"
                      style={{
                        padding: "8px 0",
                        borderBottom: "1px solid var(--color-border)",
                        fontSize: 13,
                      }}
                    >
                      <span>{p.period || p.label || `时段 ${i + 1}`}</span>
                      <span className={`badge ${badgeClass(p.intensity || p.level)}`}>
                        {p.switch_count != null ? `${p.switch_count} 次切换` : p.intensity || p.level}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                !loading.patterns && (
                  <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>暂无数据</p>
                )
              )}
            </div>

            <div className="card" style={{ flex: 1 }}>
              <h3>触发应用 Top</h3>
              {renderLoading("patterns")}
              {patterns?.trigger_apps?.length > 0 ? (
                <ul style={{ listStyle: "none", padding: 0 }}>
                  {patterns.trigger_apps.map((a, i) => (
                    <li
                      key={i}
                      className="flex flex-between"
                      style={{
                        padding: "8px 0",
                        borderBottom: "1px solid var(--color-border)",
                        fontSize: 13,
                      }}
                    >
                      <span>{a.app_name || a.name || `应用 ${i + 1}`}</span>
                      <span className="badge badge-warning">
                        {a.count != null ? `${a.count} 次` : a.percentage}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                !loading.patterns && (
                  <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>暂无数据</p>
                )
              )}
            </div>
          </div>

          {baseline && (
            <div className="card">
              <h3>基线对比</h3>
              <div className="flex gap16" style={{ fontSize: 13 }}>
                <div>
                  <span style={{ color: "var(--color-text-tertiary)" }}>平均专注时长：</span>
                  {baseline.avg_focus_min != null ? `${baseline.avg_focus_min} 分钟` : "N/A"}
                </div>
                <div>
                  <span style={{ color: "var(--color-text-tertiary)" }}>日均切换次数：</span>
                  {baseline.avg_switches_per_day != null ? baseline.avg_switches_per_day : "N/A"}
                </div>
                <div>
                  <span style={{ color: "var(--color-text-tertiary)" }}>效率评分：</span>
                  {baseline.productivity_score != null ? baseline.productivity_score : "N/A"}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── 个人画像 Tab ── */}
      {activeTab === "个人画像" && (
        <div className="flex gap16" style={{ flexDirection: "column" }}>
          {renderLoading("profile")}
          {profile && !loading.profile && (
            <>
              <div className="kpi-row" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
                <div className="stat-card">
                  <div className="label">专注高峰</div>
                  <div className="value" style={{ fontSize: 22 }}>
                    {profile.peak_focus || "N/A"}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="label">效率应用</div>
                  <div className="value" style={{ fontSize: 22 }}>
                    {profile.productivity_apps?.length ?? 0}
                  </div>
                  <div className="sub" style={{ color: "var(--color-text-tertiary)" }}>
                    {Array.isArray(profile.productivity_apps)
                      ? profile.productivity_apps.slice(0, 3).join(", ")
                      : "N/A"}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="label">平均专注块</div>
                  <div className="value" style={{ fontSize: 22 }}>
                    {profile.avg_focus_block_min != null
                      ? `${profile.avg_focus_block_min}m`
                      : "N/A"}
                  </div>
                </div>
                <div className="stat-card">
                  <div className="label">触发应用</div>
                  <div className="value" style={{ fontSize: 22 }}>
                    {profile.trigger_apps?.length ?? 0}
                  </div>
                  <div className="sub" style={{ color: "var(--color-text-tertiary)" }}>
                    {Array.isArray(profile.trigger_apps)
                      ? profile.trigger_apps.slice(0, 3).join(", ")
                      : "N/A"}
                  </div>
                </div>
              </div>

              <div className="card">
                <h3>详细画像</h3>
                <table>
                  <thead>
                    <tr>
                      <th>指标</th>
                      <th>数值</th>
                      <th>趋势</th>
                    </tr>
                  </thead>
                  <tbody>
                    {profile.details &&
                      Object.entries(profile.details).map(([key, value]) => {
                        const trend = profileDetailTrend(value);
                        return (
                          <tr key={key}>
                            <td>{key}</td>
                            <td>{profileDetailValue(value)}</td>
                            <td>
                              <span className={`badge ${trend === "up" ? "badge-success" : trend === "down" ? "badge-danger" : "badge-info"}`}>
                                {trend}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {!profile && !loading.profile && (
            <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>暂无数据</p>
          )}
        </div>
      )}

      {/* ── 拖延归因 Tab ── */}
      {activeTab === "拖延归因" && (
        <div>
          <div className="mb16">
            <button className="btn" onClick={handleAttribution} disabled={loading.attribution}>
              {loading.attribution ? "分析中..." : "运行归因分析"}
            </button>
          </div>

          {renderLoading("attribution")}

          {attribution && !loading.attribution && (
            <div className="flex gap16" style={{ flexDirection: "column" }}>
              {attribution.results && attribution.results.length > 0 ? (
                attribution.results.map((r, i) => (
                  <div className="card" key={i}>
                    <div className="flex flex-between mb16">
                      <h3 style={{ margin: 0 }}>
                        {r.procrastination_type || r.type || `归因结果 ${i + 1}`}
                      </h3>
                      <span className={`badge ${badgeClass(r.confidence)}`}>
                        置信度: {r.confidence != null ? r.confidence : "N/A"}
                      </span>
                    </div>
                    {r.cbt_technique && (
                      <div className="mb16">
                        <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                          CBT 技术
                        </span>
                        <p style={{ fontSize: 13, marginTop: 4 }}>{r.cbt_technique}</p>
                      </div>
                    )}
                    {r.evidence && (
                      <div>
                        <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
                          证据
                        </span>
                        <p style={{ fontSize: 13, marginTop: 4 }}>{r.evidence}</p>
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="card">
                  <h3>归因结果</h3>
                  <div className="mb16">
                    <span className={`badge ${badgeClass(attribution.confidence)}`}>
                      置信度: {attribution.confidence != null ? attribution.confidence : "N/A"}
                    </span>
                  </div>
                  {attribution.procrastination_type && (
                    <p style={{ fontSize: 13, marginBottom: 12 }}>
                      <strong>拖延类型：</strong>
                      {attribution.procrastination_type}
                    </p>
                  )}
                  {attribution.cbt_technique && (
                    <div className="mb16">
                      <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>CBT 技术</span>
                      <p style={{ fontSize: 13, marginTop: 4 }}>{attribution.cbt_technique}</p>
                    </div>
                  )}
                  {attribution.evidence && (
                    <div>
                      <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>证据</span>
                      <p style={{ fontSize: 13, marginTop: 4 }}>{attribution.evidence}</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {!attribution && !loading.attribution && (
            <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>
              点击上方按钮运行归因分析，识别拖延模式
            </p>
          )}
        </div>
      )}

      {/* ── 模型状态 Tab ── */}
      {activeTab === "模型状态" && (
        <div>
          {renderLoading("modelStatus")}
          {modelStatus && !loading.modelStatus && (
            <div className="card">
              <h3>ML 模型状态</h3>
              <div className="flex gap16" style={{ flexDirection: "column", fontSize: 13 }}>
                <div className="flex flex-between">
                  <span style={{ color: "var(--color-text-secondary)" }}>模型加载状态</span>
                  <span className={`badge ${modelStatus.loaded ? "badge-success" : "badge-danger"}`}>
                    {modelStatus.loaded ? "已加载" : "未加载"}
                  </span>
                </div>
                {modelStatus.version && (
                  <div className="flex flex-between">
                    <span style={{ color: "var(--color-text-secondary)" }}>版本</span>
                    <span>{modelStatus.version}</span>
                  </div>
                )}
                {modelStatus.model_name && (
                  <div className="flex flex-between">
                    <span style={{ color: "var(--color-text-secondary)" }}>模型名称</span>
                    <span>{modelStatus.model_name}</span>
                  </div>
                )}
                {modelStatus.last_updated && (
                  <div className="flex flex-between">
                    <span style={{ color: "var(--color-text-secondary)" }}>最后更新</span>
                    <span>{modelStatus.last_updated}</span>
                  </div>
                )}
              </div>
            </div>
          )}
          {!modelStatus && !loading.modelStatus && (
            <p style={{ color: "var(--color-text-tertiary)", fontSize: 13 }}>暂无数据</p>
          )}
        </div>
      )}
    </div>
  );
}
