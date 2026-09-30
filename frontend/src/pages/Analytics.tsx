import { useState, useEffect, useCallback, useRef } from "react";
import Timeline from "@mui/lab/Timeline";
import TimelineItem from "@mui/lab/TimelineItem";
import TimelineSeparator from "@mui/lab/TimelineSeparator";
import TimelineConnector from "@mui/lab/TimelineConnector";
import TimelineContent from "@mui/lab/TimelineContent";
import TimelineDot from "@mui/lab/TimelineDot";
import AdjustIcon from "@mui/icons-material/Adjust";
import {
  getAnalyticsPatterns,
  getBaseline,
  getProfile,
  getModelStatus,
  getAiProviderStatus,
  runAttribution,
  getErrorMessage,
  ApiError,
} from "../api";
import type {
  AnalyticsPatterns,
  AttributionResponse,
  BaselineSummary,
  BehavioralProfile,
  ModelStatus,
} from "../api";
import "./analytics.css";

const DAYS_OPTIONS = [7, 14, 30, 90];
const SECTION_TITLES = ["模式分析", "个人画像", "拖延归因"] as const;

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

/** One row of the reference's dual tables. */
interface TableRow {
  left: string;
  right: string;
}

function toHighSwitchRows(patterns: AnalyticsPatterns | null): TableRow[] {
  return (patterns?.high_switch_periods ?? []).map((p, i) => ({
    left:
      p.period ||
      p.label ||
      (p.hour != null
        ? `${String(p.hour).padStart(2, "0")}:00 - ${String(p.hour + 1).padStart(2, "0")}:00`
        : `时段 ${i + 1}`),
    right: p.switch_count != null ? `${p.switch_count}次切换` : p.intensity || p.level || "—",
  }));
}

function toTriggerAppRows(patterns: AnalyticsPatterns | null): TableRow[] {
  return (patterns?.trigger_apps ?? []).map((a, i) => ({
    left: a.app || a.app_name || a.name || `应用 ${i + 1}`,
    right: a.count != null ? `${a.count}次` : String(a.percentage ?? "—"),
  }));
}

export default function Analytics() {
  const [days, setDays] = useState(14);

  const [patterns, setPatterns] = useState<AnalyticsPatterns | null>(null);
  const [baseline, setBaseline] = useState<BaselineSummary | null>(null);
  const [profile, setProfile] = useState<BehavioralProfile | null>(null);
  const [modelStatus, setModelStatusState] = useState<ModelStatus | null>(null);
  const [providerStatus, setProviderStatus] = useState<{ configured: boolean; model: string } | null>(null);
  const [attribution, setAttribution] = useState<AttributionResponse | null>(null);

  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string | null>>({});
  const error = Object.values(errors).filter(Boolean).join("；");
  const setSectionError = useCallback((section: string, message: string | null) => {
    setErrors((current) => ({ ...current, [section]: message }));
  }, []);

  // Scroll-linked progress rail — the reference highlights the section whose
  // title has crossed the container's 40% line, and pins "拖延归因" once the
  // bottom block enters the viewport.
  const pageRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const titleRefs = useRef<Array<HTMLDivElement | null>>([]);
  const [activeIndex, setActiveIndex] = useState(0);

  /** Latest-request guards (audit F3): each series owns its own sequence so a
   *  slow 7-day response can neither commit data, clear the spinner, nor
   *  overwrite the error of a newer 30-day request. */
  const patternsSeqRef = useRef(0);
  const profileSeqRef = useRef(0);

  const fetchPatterns = useCallback(async () => {
    const seq = ++patternsSeqRef.current;
    setLoading((p) => ({ ...p, patterns: true }));
    setPatterns(null);
    setSectionError("patterns", null);
    try {
      const data = await getAnalyticsPatterns(days);
      if (seq !== patternsSeqRef.current) return; // stale response
      setPatterns(data);
    } catch (e: unknown) {
      if (seq !== patternsSeqRef.current) return;
      // Never keep the previous range's rows on screen under the new label.
      setPatterns(null);
      setSectionError("patterns", getErrorMessage(e, "模式分析加载失败"));
    } finally {
      if (seq === patternsSeqRef.current) setLoading((p) => ({ ...p, patterns: false }));
    }
  }, [days, setSectionError]);

  const fetchBaseline = useCallback(async () => {
    setLoading((p) => ({ ...p, baseline: true }));
    setSectionError("baseline", null);
    try {
      const state = await getBaseline();
      if (!state.ok) {
        // Malformed wire payload — hide the comparison card, surface the error.
        setBaseline(null);
        setSectionError("baseline", getErrorMessage(new ApiError("基线数据格式无效", 500), "基线数据加载失败"));
        return;
      }
      setBaseline(state);
    } catch (e: unknown) {
      if (e instanceof ApiError && e.status === 404) {
        // No baseline yet is a normal business state — show the empty card
        // (same as ModelCenter) instead of a red error banner.
        setBaseline(null);
      } else {
        setSectionError("baseline", getErrorMessage(e, "基线数据加载失败"));
      }
    } finally {
      setLoading((p) => ({ ...p, baseline: false }));
    }
  }, [setSectionError]);

  const fetchProfile = useCallback(async () => {
    const seq = ++profileSeqRef.current;
    setLoading((p) => ({ ...p, profile: true }));
    setProfile(null);
    setSectionError("profile", null);
    try {
      const data = await getProfile(days);
      if (seq !== profileSeqRef.current) return; // stale response
      setProfile(data);
    } catch (e: unknown) {
      if (seq !== profileSeqRef.current) return;
      setProfile(null);
      setSectionError("profile", getErrorMessage(e, "个人画像加载失败"));
    } finally {
      if (seq === profileSeqRef.current) setLoading((p) => ({ ...p, profile: false }));
    }
  }, [days, setSectionError]);

  const fetchModelStatus = useCallback(async () => {
    setLoading((p) => ({ ...p, modelStatus: true }));
    setSectionError("modelStatus", null);
    try {
      const data = await getModelStatus();
      setModelStatusState(data);
    } catch (e: unknown) {
      setSectionError("modelStatus", getErrorMessage(e, "模型状态加载失败"));
    } finally {
      setLoading((p) => ({ ...p, modelStatus: false }));
    }
  }, [setSectionError]);

  const fetchProviderStatus = useCallback(async () => {
    try {
      const data = await getAiProviderStatus();
      setProviderStatus({ configured: data.configured, model: data.model });
    } catch {
      // Provider status is optional decoration on this page; a failure here
      // must not blank the section (the other fetches report their own).
    }
  }, []);

  useEffect(() => {
    fetchBaseline();
    fetchModelStatus();
    fetchProviderStatus();
  }, [fetchBaseline, fetchModelStatus, fetchProviderStatus]);
  useEffect(() => {
    const patternRequests = patternsSeqRef;
    const profileRequests = profileSeqRef;
    fetchPatterns();
    fetchProfile();
    return () => {
      patternRequests.current++;
      profileRequests.current++;
    };
  }, [fetchPatterns, fetchProfile]);

  useEffect(() => {
    const targets = titleRefs.current.filter(Boolean) as HTMLDivElement[];
    const bottomEl = bottomRef.current;
    const page = pageRef.current;
    if (!bottomEl || !page) return;

    let lastBottom = false;

    const pickTitleActive = () => {
      if (lastBottom) {
        setActiveIndex(2);
        return;
      }
      const rect = page.getBoundingClientRect();
      const triggerBottom = rect.top + rect.height * 0.4;
      let bestIdx = 0;
      let bestBottom = -Infinity;
      targets.forEach((t, i) => {
        const tRect = t.getBoundingClientRect();
        if (tRect.top < triggerBottom) {
          const bottom = tRect.bottom - rect.top;
          if (bottom > bestBottom) {
            bestBottom = bottom;
            bestIdx = i;
          }
        }
      });
      setActiveIndex(bestIdx);
    };

    const titleObserver = new IntersectionObserver(() => pickTitleActive(), {
      root: page,
      rootMargin: "0px 0px -60% 0px",
      threshold: [0, 0.25, 0.5, 1],
    });
    targets.forEach((t) => titleObserver.observe(t));

    const bottomObserver = new IntersectionObserver(
      (entries) => {
        const atBottom = entries.some((entry) => entry.isIntersecting);
        lastBottom = atBottom;
        if (atBottom) setActiveIndex(2);
        else pickTitleActive();
      },
      { root: page, rootMargin: "0px 0px -40px 0px", threshold: 0 },
    );
    bottomObserver.observe(bottomEl);

    pickTitleActive();

    return () => {
      titleObserver.disconnect();
      bottomObserver.disconnect();
    };
  }, [loading.patterns, loading.profile]);

  const handleAttribution = async () => {
    setLoading((p) => ({ ...p, attribution: true }));
    setSectionError("attribution", null);
    setAttribution(null);
    try {
      const data = await runAttribution();
      setAttribution(data);
    } catch (e: unknown) {
      setSectionError("attribution", getErrorMessage(e, "归因分析失败"));
    } finally {
      setLoading((p) => ({ ...p, attribution: false }));
    }
  };

  const highSwitchRows = toHighSwitchRows(patterns);
  const triggerAppRows = toTriggerAppRows(patterns);
  const profileDetails = profile?.details ? Object.entries(profile.details) : [];

  const renderTitle = (text: string, index: number) => (
    <div
      className="title"
      ref={(el) => {
        titleRefs.current[index] = el;
      }}
    >
      {text}
    </div>
  );

  return (
    <div className="analytics" ref={pageRef}>
      <Timeline className="position">
        {SECTION_TITLES.map((label, index) => (
          <TimelineItem key={label}>
            <TimelineContent className={activeIndex === index ? "is-active" : undefined}>{label}</TimelineContent>
            <TimelineSeparator>
              <TimelineDot sx={{ bgcolor: "transparent", boxShadow: "none", p: 0 }}>
                <AdjustIcon sx={activeIndex === index ? { color: "#1890ff" } : { color: "#6a89ad" }} />
              </TimelineDot>
              {index < SECTION_TITLES.length - 1 && <TimelineConnector className="line" />}
            </TimelineSeparator>
          </TimelineItem>
        ))}
      </Timeline>

      <div className="time-box">
        <div className="time-text">时间范围</div>
        <select
          className="time"
          value={`近${days}天`}
          aria-label="时间范围"
          onChange={(e) => {
            const parsed = Number(String(e.target.value).replace(/\D/g, ""));
            if (Number.isFinite(parsed) && parsed > 0) setDays(parsed);
          }}
        >
          {DAYS_OPTIONS.map((d) => (
            <option key={d} value={`近${d}天`}>
              近{d}天
            </option>
          ))}
        </select>
      </div>

      {error && (
        <div className="error-box error-slot" role="alert">
          {error}
          <button className="btn btn-sm" style={{ marginLeft: 12 }} onClick={() => setErrors({})}>
            关闭
          </button>
        </div>
      )}

      {renderTitle("模式分析", 0)}

      <div className="top-box">
        <div className="high-change">
          <div className="high-change-title">高切换时段</div>
          <div className="f-box">
            {loading.patterns ? (
              <div className="spinner" />
            ) : highSwitchRows.length > 0 ? (
              <table className="simple-table">
                <tbody>
                  {highSwitchRows.map((row, i) => (
                    <tr key={i}>
                      {/* The reference pills BOTH cells (table.scss sty=0) */}
                      <td>
                        <div className="full-box1">{row.left}</div>
                      </td>
                      <td>
                        <div className="full-box1">{row.right}</div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="f-empty">暂无数据</div>
            )}
          </div>
        </div>

        <div className="toggle-app">
          <div className="toggle-app-title">触发应用 Top</div>
          <div className="f-box">
            {loading.patterns ? (
              <div className="spinner" />
            ) : triggerAppRows.length > 0 ? (
              <table className="simple-table">
                <tbody>
                  {triggerAppRows.map((row, i) => (
                    <tr key={i}>
                      {/* The reference pills BOTH cells (table.scss sty=1) */}
                      <td>
                        <div className="full-box2">{row.left}</div>
                      </td>
                      <td>
                        <div className="full-box2">{row.right}</div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="f-empty">暂无数据</div>
            )}
          </div>
        </div>
      </div>

      <div className="second-box">
        <div className="box-title">基线对比</div>
        <div className="second-content">
          <div className="second-content-item">
            平均切换次数：
            <b>{baseline?.mean_app_switch_count ?? "N/A"}</b>
          </div>
          <div className="second-content-item">
            活跃时间占比：
            <b>
              {baseline?.mean_active_seconds_ratio != null
                ? `${(baseline.mean_active_seconds_ratio * 100).toFixed(1)}%`
                : "N/A"}
            </b>
          </div>
          <div className="second-content-item">
            空闲时间占比：
            <b>
              {baseline?.mean_idle_ratio != null ? `${(baseline.mean_idle_ratio * 100).toFixed(1)}%` : "N/A"}
            </b>
          </div>
        </div>
      </div>

      {renderTitle("个人画像", 1)}

      <div className="third-box">
        <div className="third-box-item">
          <div className="third-item-title">专注高峰</div>
          <div className="third-item-content">{profile?.peak_focus || "N/A"}</div>
        </div>
        <div className="third-box-item">
          <div className="third-item-title">效率应用</div>
          <div className="third-item-content">{profile ? profile.productivity_apps?.length ?? 0 : "—"}</div>
          <div className="detail">
            {Array.isArray(profile?.productivity_apps) ? profile.productivity_apps.slice(0, 3).join(", ") : "N/A"}
          </div>
        </div>
        <div className="third-box-item">
          <div className="third-item-title">平均专注块</div>
          <div className="third-item-content">
            {profile?.avg_focus_block_min != null ? `${profile.avg_focus_block_min}m` : "N/A"}
          </div>
        </div>
        <div className="third-box-item">
          <div className="third-item-title">触发应用</div>
          <div className="third-item-content">{profile ? profile.trigger_apps?.length ?? 0 : "—"}</div>
          <div className="detail">
            {Array.isArray(profile?.trigger_apps) ? profile.trigger_apps.slice(0, 3).join(", ") : "N/A"}
          </div>
        </div>
      </div>

      <div className="fourth-box">
        <div className="fourth-box-title">详细画像</div>
        <div className="table-box">
          {loading.profile ? (
            <div className="spinner" />
          ) : profileDetails.length === 0 ? (
            <div className="f-empty">暂无数据</div>
          ) : (
            <table className="simple-table">
              <thead>
                <tr>
                  <th>指标</th>
                  <th>数值</th>
                  <th>趋势</th>
                </tr>
              </thead>
              <tbody>
                {profileDetails.map(([key, value]) => {
                  const trend = profileDetailTrend(value);
                  return (
                    <tr key={key}>
                      <td>{key}</td>
                      <td>{profileDetailValue(value)}</td>
                      <td>
                        <span
                          className={`badge ${trend === "up" ? "badge-success" : trend === "down" ? "badge-danger" : "badge-info"}`}
                        >
                          {trend}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {renderTitle("拖延归因", 2)}

      <div className="fifth-box" ref={bottomRef}>
        <div className="left-box">
          <div className="left-box-title1">归因结果</div>
          <div className="left-box-content1">
            置信度：{attribution?.confidence != null ? attribution.confidence : "—"}
          </div>
          <div className="left-box-content2">
            <b>拖延类型：</b>
            {attribution?.procrastination_type ?? attribution?.results?.[0]?.procrastination_type ?? "—"}
          </div>
          <div className="left-box-title2">CBT 技术</div>
          <div className="left-box-content2">
            {attribution?.cbt_technique ?? attribution?.results?.[0]?.cbt_technique ?? "—"}
          </div>
          <div className="left-box-title2">证据</div>
          <div className="left-box-content3">
            {attribution?.evidence ?? attribution?.results?.[0]?.evidence ?? "尚未运行归因分析"}
          </div>
          <button
            type="button"
            className="btn attribution-action"
            onClick={handleAttribution}
            disabled={loading.attribution}
          >
            {loading.attribution ? "分析中..." : "运行归因分析"}
          </button>
        </div>

        <div className="right-box">
          <div className="right-box-title1">ML 模型状态</div>
          <div className="content1">
            模型加载状态:
            <span
              className={`model-pill ${modelStatus?.loaded ? "model-pill--on" : "model-pill--off"}`}
            >
              {modelStatus?.loaded ? "已加载" : "未加载"}
            </span>
          </div>
          <div className="model-meta">
            {providerStatus
              ? providerStatus.configured
                ? `LLM 已配置 · ${providerStatus.model}`
                : "LLM 未配置"
              : "LLM 状态未知"}
          </div>
          <div className="model-meta">
            {modelStatus?.version ? `模型版本 ${modelStatus.version}` : "规则引擎模式"}
          </div>
          <div className="model-meta">
            {modelStatus?.loaded ? `模式 ${modelStatus.mode ?? "ready"}` : "暂无已加载模型"}
          </div>
        </div>
      </div>
    </div>
  );
}
