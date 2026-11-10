import { userTerm } from "../lib/user-terms";
import { useState, useEffect, useCallback, useRef } from "react";
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
import TmtGauge from "../components/charts/TmtGauge";
import { SEVERITY_LABELS } from "../types/explainability";
import type { DeviationSummary, TmtBreakdown } from "../types/explainability";
import Surface from "../components/signal/Surface";
import SectionHeader from "../components/signal/SectionHeader";
import MetricStrip from "../components/signal/MetricStrip";
import type { MetricItem } from "../components/signal/MetricStrip";
import Segmented from "../components/signal/Segmented";
import HelpTip from "../components/signal/HelpTip";
import "./analytics.css";
import "../components/primitives.css";

const DAYS_OPTIONS = [7, 14, 30, 90];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function profileDetailValue(value: unknown): string | number {
  const detailValue = isRecord(value) ? value.value : value;
  if (typeof detailValue === "string" || typeof detailValue === "number") return detailValue;
  if (typeof detailValue === "boolean") return String(detailValue);
  return "—";
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
    right: p.switch_count != null ? `${p.switch_count} 次切换` : p.intensity || p.level || "—",
  }));
}

function toTriggerAppRows(patterns: AnalyticsPatterns | null): TableRow[] {
  return (patterns?.trigger_apps ?? []).map((a, i) => ({
    left: a.app || a.app_name || a.name || `应用 ${i + 1}`,
    right: a.count != null ? `${a.count} 次` : String(a.percentage ?? "—"),
  }));
}

/** `"8:00 (avg 52)"` → period `8:00`, average `52`. A trailing parenthetical
 *  without an `avg N` inside is still dropped from the period (§8.4.7). */
function splitPeakFocus(peakFocus: string | undefined): { period: string; average: string | null } {
  if (!peakFocus) return { period: "—", average: null };
  const split = peakFocus.search(/\s*\(|（/);
  const period = (split === -1 ? peakFocus : peakFocus.slice(0, split)).trim();
  const average = /avg\s+([\d.]+)/i.exec(peakFocus)?.[1] ?? null;
  return { period: period || peakFocus.trim(), average };
}

const TYPE_KEYS = [
  "task_aversion",
  "impulsivity",
  "decisional",
  "perfectionism",
  "emotional_regulation",
] as const;

const TYPE_LABELS: Record<string, string> = {
  task_aversion: "任务厌恶",
  impulsivity: "冲动分心",
  decisional: "决策拖延",
  perfectionism: "完美主义",
  emotional_regulation: "情绪调节",
};

const SEVERITY_TONE: Record<string, string> = {
  normal: "badge-success",
  mild: "badge-info",
  moderate: "badge-warning",
  severe: "badge-danger",
};

/**
 * 拖延归因的可解释部分.
 *
 * Every block here comes from an optional contract addition, so each one
 * renders independently: a response with `tmt` but no `deviation` shows the
 * TMT gauge and a neutral line where the deviation table would be.
 */
function AttributionEvidence({
  typeConfidence,
  distortions,
  nextAction,
  tmt,
  deviation,
}: {
  typeConfidence: Record<string, number>;
  distortions: string[];
  nextAction?: string;
  tmt?: TmtBreakdown | null;
  deviation?: DeviationSummary | null;
}) {
  const typeRows = TYPE_KEYS.flatMap((key) => {
    const value = typeConfidence[key];
    if (typeof value !== "number" || !Number.isFinite(value)) return [];
    return [{ key, label: TYPE_LABELS[key], value }];
  }).sort((a, b) => b.value - a.value);
  const leading = typeRows[0]?.key;

  return (
    <div className="attribution-evidence sig-surface">
      <section className="proof" aria-label="拖延类型置信度">
        <div className="proof-head">
          <span className="proof-title">
            类型置信度
            <HelpTip term="类型置信度" what="每种拖延类型的可能性，条越长越可能。" />
          </span>
          <span className="proof-muted">模型对各类型的判断强度</span>
        </div>
        {typeRows.length === 0 ? (
          <p className="proof-note">暂无类型置信度数据</p>
        ) : (
          <div className="type-bars">
            {typeRows.map((row) => {
              const percent = Math.max(0, Math.min(1, row.value)) * 100;
              return (
                <div className="type-bar-row" key={row.key} data-leading={row.key === leading}>
                  <span className="type-bar-name">
                    <span
                      className="mf-chart-legend-swatch"
                      style={{ background: `var(--type-${row.key})` }}
                      aria-hidden="true"
                    />
                    {row.label}
                  </span>
                  <div
                    className="type-bar-track"
                    role="img"
                    aria-label={`${row.label} 置信度 ${percent.toFixed(0)}%`}
                  >
                    <div
                      className="type-bar-fill"
                      style={{ width: `${percent}%`, background: `var(--type-${row.key})` }}
                    />
                  </div>
                  <span className="type-bar-value">{(row.value * 100).toFixed(0)}%</span>
                </div>
              );
            })}
          </div>
        )}
      </section>

      {distortions.length > 0 && (
        <section className="proof" aria-label="识别到的认知扭曲">
          <div className="proof-head">
            <span className="proof-title">
              认知扭曲
              <HelpTip term="认知扭曲" what="一些让人更想拖延的想法，比如“必须做到完美才开始”。" />
            </span>
            <span className="proof-muted">CBT 视角下的思维模式</span>
          </div>
          <p className="distortion-line">{distortions.join("、")}</p>
        </section>
      )}

      {nextAction && (
        <div className="callout" data-tone="good">
          <span className="callout-title">下一步可以做</span>
          <span>{nextAction}</span>
        </div>
      )}

      <section className="proof" aria-label="动机分解">
        <div className="proof-head">
          <span className="proof-title">
            动机分解（TMT）
            <HelpTip
              term="动机分解"
              what="用一个心理学公式估计你现在有多想做这件事：觉得能做成（期望）、觉得值得（价值）会提高动力；容易被别的东西吸引（冲动）、截止日期还远（延迟）会降低动力。"
              effect="这是根据使用行为估算的，不是心理测试。"
            />
          </span>
          <span className="proof-muted">期望 × 价值 ÷ (1 + 冲动 × 延迟)</span>
        </div>
        <TmtGauge tmt={tmt ?? null} />
      </section>

      <section className="proof" aria-label="行为偏离">
        <div className="proof-head">
          <span className="proof-title">
            行为偏离
            <HelpTip term="行为偏离" what="把今天这个时段和你平时同一时段相比，看看有没有明显不一样。" />
          </span>
          {deviation && (
            <span className={`badge ${SEVERITY_TONE[deviation.severity]}`}>
              {SEVERITY_LABELS[deviation.severity]}
            </span>
          )}
        </div>
        {!deviation ? (
          <p className="proof-note">暂无偏离数据</p>
        ) : (
          <>
            <div className="deviation-meta">
              <span>
                综合偏离分 {deviation.overall_score.toFixed(2)}
                <HelpTip
                  term="综合偏离分"
                  what="数字越大，说明和平时越不一样。数据少的时候，系统会更多参考一般人的水平，避免误判。"
                />
              </span>
              <span>
                经验贝叶斯收缩 n={deviation.bucket.n_samples}，收缩系数{" "}
                {deviation.bucket.shrinkage.toFixed(2)}
              </span>
              <span>
                对比时段 {String(deviation.bucket.hour_of_day).padStart(2, "0")}:00 ·
                周{["日", "一", "二", "三", "四", "五", "六"][deviation.bucket.day_of_week] ?? "—"}
              </span>
            </div>
            {deviation.features.length === 0 ? (
              <p className="proof-note">当前时段没有显著偏离的特征</p>
            ) : (
              <table className="deviation-table">
                <thead>
                  <tr>
                    <th scope="col">特征</th>
                    <th scope="col" className="num">当前</th>
                    <th scope="col" className="num">个人期望</th>
                    <th scope="col" className="num">
                      稳健 z
                      <HelpTip
                        term="偏离程度"
                        what="这一项和你平时差了多少：正数是比平时高，负数是比平时低，超过 2 就算明显不同。"
                      />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {deviation.features.map((feature) => (
                    <tr key={feature.feature}>
                      <td>{feature.label}</td>
                      <td className="num">{feature.value.toFixed(2)}</td>
                      <td className="num">{feature.expected.toFixed(2)}</td>
                      <td className="num">{feature.robust_z.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}
      </section>
    </div>
  );
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
  // Four sections can fail with the same reason (e.g. a 429 storm); the banner
  // should say it once (§8.21.2).
  const error = Array.from(new Set(Object.values(errors).filter(Boolean))).join("；");
  const setSectionError = useCallback((section: string, message: string | null) => {
    setErrors((current) => ({ ...current, [section]: message }));
  }, []);

  // Scroll-linked progress rail — the reference highlights the section whose
  // title has crossed the container's 40% line, and pins "拖延归因" once the
  // bottom block enters the viewport.
  const pageRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const titleRefs = useRef<Array<HTMLHeadingElement | null>>([]);


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
    <h2
      className="page-section-title"
      ref={(el) => {
        titleRefs.current[index] = el;
      }}
    >
      {text}
    </h2>
  );

  const percentItem = (key: string, label: string, value: number | null | undefined): MetricItem =>
    value != null
      ? { key, label, value: (value * 100).toFixed(1), unit: "%" }
      : { key, label, value: "—" };

  const baselineItems: MetricItem[] = [
    baseline?.mean_app_switch_count != null
      ? {
          key: "switch",
          label: "平均切换次数",
          value: String(baseline.mean_app_switch_count),
          unit: "次",
        }
      : { key: "switch", label: "平均切换次数", value: "—" },
    percentItem("active", "活跃时间占比", baseline?.mean_active_seconds_ratio),
    percentItem("idle", "空闲时间占比", baseline?.mean_idle_ratio),
  ];

  const peak = splitPeakFocus(profile?.peak_focus);
  const profileItems: MetricItem[] = profile
    ? [
        {
          key: "peak",
          label: "专注高峰",
          value: peak.period,
          delta: {
            text: peak.average ? `平均分 ${peak.average}` : "平均评分最高的时段",
            tone: "none",
          },
        },
        {
          key: "productivity",
          label: "常用应用",
          value: String(profile.productivity_apps?.length ?? 0),
          unit: "个",
          delta: {
            text: profile.productivity_apps?.length
              ? profile.productivity_apps.slice(0, 3).join("、")
              : "—",
            tone: "none",
          },
        },
        profile.avg_focus_block_min != null
          ? {
              key: "block",
              label: "平均专注块",
              value: String(profile.avg_focus_block_min),
              unit: "分钟",
              delta: { text: "每次连续专注的平均时长", tone: "none" },
              help: { what: "每次连续专注平均能坚持多久。" },
            }
          : {
              key: "block",
              label: "平均专注块",
              value: "—",
              delta: { text: "每次连续专注的平均时长", tone: "none" },
              help: { what: "每次连续专注平均能坚持多久。" },
            },
        {
          key: "trigger",
          label: "触发应用",
          value: String(profile.trigger_apps?.length ?? 0),
          unit: "个",
          delta: {
            text: profile.trigger_apps?.length ? profile.trigger_apps.slice(0, 3).join("、") : "—",
            tone: "none",
          },
        },
      ]
    : [
        { key: "peak", label: "专注高峰", value: "—" },
        { key: "productivity", label: "常用应用", value: "—" },
        { key: "block", label: "平均专注块", value: "—" },
        { key: "trigger", label: "触发应用", value: "—" },
      ];

  const modelBadge = !modelStatus
    ? { className: "badge-neutral", text: "状态未知" }
    : modelStatus.ready
      ? { className: "badge-success", text: "已就绪" }
      : modelStatus.loaded
        ? { className: "badge-warning", text: "已加载，未就绪" }
        : { className: "badge-neutral", text: "未加载" };
  const providerBadge = !providerStatus
    ? { className: "badge-neutral", text: "状态未知" }
    : providerStatus.configured
      ? { className: "badge-success", text: "已配置" }
      : { className: "badge-neutral", text: "未配置" };

  const attributionType = userTerm(attribution?.procrastination_type ?? attribution?.results?.[0]?.procrastination_type);
  const attributionConfidence = attribution?.confidence ?? attribution?.results?.[0]?.confidence ?? null;
  const attributionCbt = userTerm(attribution?.cbt_technique ?? attribution?.results?.[0]?.cbt_technique);
  const attributionEvidence = attribution?.evidence ?? attribution?.results?.[0]?.evidence;

  return (
    <div className="analytics" ref={pageRef}>
      <p className="page-lede">从最近一段时间的切换和应用使用中，找出你容易分心的时段与原因。</p>

      <div className="page-toolbar time-box">
        <Segmented
          ariaLabel="时间范围"
          options={DAYS_OPTIONS.map((d) => ({ value: String(d), label: `近 ${d} 天` }))}
          value={String(days)}
          onChange={(value) => setDays(Number(value))}
        />
      </div>

      {error && (
        <div className="error-box error-slot" role="alert">
          {error}
          <button className="btn btn-secondary btn-sm error-dismiss" onClick={() => setErrors({})}>
            关闭
          </button>
        </div>
      )}

      {renderTitle("模式分析", 0)}

      <div className="page-grid-2 top-box">
        <Surface as="section" className="high-change">
          <SectionHeader
            title="高切换时段"
            help={{
              what: "一天中你最常在不同软件之间来回切换的时段。切换多通常说明注意力容易被打断。",
            }}
          />
          <div className="f-box">
            {loading.patterns ? (
              <div className="spinner" />
            ) : highSwitchRows.length > 0 ? (
              highSwitchRows.map((row, i) => (
                <div className="kv-row kv-row--strong" key={i}>
                  <span>{row.left}</span>
                  <span>{row.right}</span>
                </div>
              ))
            ) : (
              <div className="f-empty">暂无数据</div>
            )}
          </div>
        </Surface>

        <Surface as="section" className="toggle-app">
          <SectionHeader
            title="触发应用 Top"
            help={{
              what: "在被判为分心的时间里，你主要在用的软件。它们不一定是分心的原因，只是经常一起出现。",
            }}
          />
          <div className="f-box">
            {loading.patterns ? (
              <div className="spinner" />
            ) : triggerAppRows.length > 0 ? (
              triggerAppRows.map((row, i) => (
                <div className="kv-row kv-row--strong" key={i}>
                  <span>{row.left}</span>
                  <span>{row.right}</span>
                </div>
              ))
            ) : (
              <div className="f-empty">暂无数据</div>
            )}
          </div>
        </Surface>
      </div>

      <div className="second-box">
        <MetricStrip
          label="我的日常习惯"
          items={baselineItems.map((item, index) =>
            index === 0
              ? { ...item, help: { what: "你过去所有记录的平均水平，可以用来对比某一天是否反常。" } }
              : item,
          )}
          // No baseline yet is a normal early state, not an error: say what is
          // happening instead of leaving three em-dashes unexplained (§8.22.7).
          note={
            baseline === null && !loading.baseline && !errors.baseline
              ? "还在积累记录：使用几天后，这里会显示你平时的平均水平。"
              : undefined
          }
        />
      </div>

      {renderTitle("个人画像", 1)}

      <div className="third-box">
        <MetricStrip label="画像概览" items={profileItems} />
      </div>

      <Surface as="section" className="fourth-box">
        <SectionHeader title="详细画像" />
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
                      <td>
                        {userTerm(key)}
                        {key === "distraction_ratio" && (
                          <HelpTip
                            term="分心占比"
                            what="分心触发的次数 ÷ 全部活动次数，是次数比例，不是时间比例。"
                          />
                        )}
                      </td>
                      <td>{profileDetailValue(value)}</td>
                      <td>
                        <span
                          className={`badge ${trend === "up" ? "badge-success" : trend === "down" ? "badge-warning" : "badge-neutral"}`}
                        >
                          {userTerm(trend)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </Surface>

      {renderTitle("拖延归因", 2)}

      <div className="page-grid-2 fifth-box" ref={bottomRef}>
        <Surface as="section" className="left-box">
          <SectionHeader
            title="归因结果"
            actions={
              <>
                <button
                  type="button"
                  className="btn btn-primary btn-sm attribution-action"
                  onClick={handleAttribution}
                  disabled={loading.attribution}
                >
                  {loading.attribution ? "分析中…" : "运行归因分析"}
                </button>
                <HelpTip
                  term="归因分析"
                  what="让 AI 根据今天的使用记录，分析你可能为什么拖延，并给出一个具体的小方法。"
                  effect="会把今天的行为汇总（不含窗口标题）发给云端 AI，可能需要几分钟、产生少量费用；同一天再点会直接显示已有结果。"
                />
              </>
            }
          />
          <div className="kv-row kv-row--text">
            <span>
              拖延类型
              <HelpTip
                term="拖延类型"
                what="AI 判断你今天的拖延最可能属于哪一种，比如“任务厌恶”就是因为不喜欢这件事而一直拖着。"
              />
            </span>
            <span>{attributionType}</span>
          </div>
          <div className="kv-row kv-row--text">
            <span>
              置信度
              <HelpTip
                term="置信度"
                what="AI 对这个判断有多大把握，0 到 1，越接近 1 越有把握。"
              />
            </span>
            <span>{attributionConfidence != null ? String(attributionConfidence) : "—"}</span>
          </div>
          <div className="kv-row kv-row--text">
            <span>
              认知行为技术（CBT）
              <HelpTip
                term="认知行为技术"
                what="心理学里常用的改变习惯的方法。这里推荐一个最适合你当前情况的小技巧。"
              />
            </span>
            <span>{attributionCbt}</span>
          </div>
          <div className="kv-row kv-row--text evidence-row">
            <span>证据</span>
            <span>{attributionEvidence ?? "尚未运行归因分析"}</span>
          </div>
        </Surface>

        <Surface as="section" className="right-box analysis-capabilities">
          <SectionHeader
            title="个人模型与云端 AI"
            help={{
              what: "个人模型在你的电脑上学习你的习惯；云端 AI 负责写分析文字。这里显示它们现在能不能用。",
            }}
          />
          <div className="capability-group">
            <div className="capability-head">
              <h3 className="sub-title">个人专注模型（ML）</h3>
              <span className={`badge ${modelBadge.className}`}>{modelBadge.text}</span>
            </div>
            <p className="capability-text">根据你的行为与反馈预测专注或分心。未就绪时，系统仍可用规则引擎分析。</p>
            <div className="model-meta">
              {modelStatus?.version ? `版本 ${modelStatus.version} · ${userTerm(modelStatus.mode)}` : "暂无个人模型版本"}
            </div>
          </div>
          <div className="capability-group capability-group--divided">
            <div className="capability-head">
              <h3 className="sub-title">云端 AI（LLM）</h3>
              <span className={`badge ${providerBadge.className}`}>{providerBadge.text}</span>
              {providerStatus?.configured && <span className="model-name">{providerStatus.model}</span>}
            </div>
            <p className="capability-text">用于拖延归因、专家讨论与建议生成；独立于个人模型。已配置仅表示有凭证，调用成功还取决于服务可用性。</p>
          </div>
        </Surface>
      </div>

      {attribution && (
        <AttributionEvidence
          typeConfidence={attribution.assessment?.type_confidence ?? {}}
          distortions={attribution.assessment?.cognitive_distortions ?? []}
          nextAction={attribution.assessment?.next_action}
          tmt={attribution.tmt}
          deviation={attribution.deviation}
        />
      )}
    </div>
  );
}
