import { useCallback, useRef, useState } from "react";
import { assignTaskContext, getErrorMessage, getPeriodReview, getUnclassified } from "../api";
import type {
  AssignableContext,
  PeriodDays,
  PeriodReview as PeriodReviewData,
  UnclassifiedItems,
  UnclassifiedRowView,
} from "../period-review";
import { ASSIGNABLE_CONTEXTS, toPeriodReviewView, toUnclassifiedRows } from "../period-review";
import { toast } from "./toast-store";
import BarChart from "./charts/BarChart";
import Disclosure from "./signal/Disclosure";
import HelpTip from "./signal/HelpTip";
import MetricStrip from "./signal/MetricStrip";
import SectionHeader from "./signal/SectionHeader";
import Segmented from "./signal/Segmented";
import Surface from "./signal/Surface";
import "./period-review.css";

/**
 * Period review ("阶段总结") — "why did I get distracted these days?".
 *
 * Two layers, visible in the UI by design: the summary card (words) and the
 * blocks under it (numbers). The "查看发给 AI 的内容" disclosure shows the exact
 * payload, so the privacy claim is inspectable rather than asserted.
 */

const DAY_OPTIONS = [
  { value: "3", label: "最近3天" },
  { value: "7", label: "最近7天" },
  { value: "14", label: "最近14天" },
];

/** One "还没认出来的应用和网站" row: name, time, a select and a remember button. */
function UnclassifiedRow({
  row,
  savedLabel,
  onSaved,
}: {
  row: UnclassifiedRowView;
  savedLabel: string | null;
  onSaved: (row: UnclassifiedRowView, label: string) => void;
}) {
  const [context, setContext] = useState<AssignableContext | "">("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  if (savedLabel) {
    return (
      <div className="period-unclassified-row">
        <span className="period-unclassified-name">
          <span className="period-unclassified-kind">{row.kindLabel}</span>
          {row.display}
        </span>
        <span className="period-unclassified-saved">
          <span aria-hidden="true">✓</span>
          {`已记住：${row.display} → ${savedLabel}`}
        </span>
      </div>
    );
  }

  const save = async () => {
    if (!context) return;
    setSaving(true);
    setError("");
    try {
      await assignTaskContext(row.kind, row.key, context);
      const label = ASSIGNABLE_CONTEXTS.find((item) => item.value === context)?.label ?? context;
      onSaved(row, label);
      toast.success("已记住，之后会自动识别", "最近 14 天的记录会在后台重新计算。");
    } catch (e: unknown) {
      // Inline, not a toast: the error belongs to this row's answer.
      setError(getErrorMessage(e, "保存失败"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="period-unclassified-row">
      <span className="period-unclassified-name">
        <span className="period-unclassified-kind">{row.kindLabel}</span>
        {row.display}
      </span>
      <span className="period-dim">{row.minutesText}</span>
      <span className="period-unclassified-answer">
        <select
          aria-label={`${row.display} 是什么`}
          value={context}
          onChange={(event) => setContext(event.target.value as AssignableContext | "")}
          disabled={saving}
        >
          <option value="">这是……</option>
          {ASSIGNABLE_CONTEXTS.map((option) => (
            <option value={option.value} key={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={save}
          disabled={saving || !context}
        >
          记住
        </button>
      </span>
      {error && (
        <span className="period-unclassified-error" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}

export default function PeriodReview() {
  const [days, setDays] = useState<PeriodDays>(7);
  const [useAi, setUseAi] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [review, setReview] = useState<PeriodReviewData | null>(null);
  const [generatedWith, setGeneratedWith] = useState<{ days: PeriodDays; useAi: boolean } | null>(null);
  // Unrecognised apps/sites for the generated span. Local-only: these names are
  // never part of the digest and never sent to an AI.
  const [unclassified, setUnclassified] = useState<UnclassifiedItems | null>(null);
  const [savedRows, setSavedRows] = useState<Record<string, string>>({});

  // Same sequence guard as the daily/weekly loaders: a slow response for an
  // older range must never overwrite the newer one.
  const seqRef = useRef(0);

  const generate = useCallback(async (span: PeriodDays, withAi: boolean) => {
    const seq = ++seqRef.current;
    setLoading(true);
    setError("");
    setUnclassified(null);
    setSavedRows({});
    // The listing runs alongside the review, not after it: it is a separate,
    // cheaper query and must not add its latency to the summary.
    const listing = getUnclassified(span).catch((e: unknown) => {
      // An optional nudge: a failure hides the card but must not fail the
      // summary. It still reaches the console so a broken endpoint is
      // diagnosable rather than invisible.
      console.warn("MindFlow: unclassified listing failed", getErrorMessage(e, "加载失败"));
      return null;
    });
    try {
      const data = await getPeriodReview(span, withAi);
      if (seq !== seqRef.current) return;
      setReview(data);
      setGeneratedWith({ days: span, useAi: withAi });
    } catch (e: unknown) {
      if (seq !== seqRef.current) return;
      setError(getErrorMessage(e, "生成失败"));
      setReview(null);
      setGeneratedWith(null);
    } finally {
      if (seq === seqRef.current) setLoading(false);
    }

    const rows = await listing;
    if (seq !== seqRef.current) return;
    setUnclassified(rows);
  }, []);

  const view = review && generatedWith
    ? toPeriodReviewView(review, generatedWith.days, generatedWith.useAi)
    : null;
  const aiRequested = review?.ai_requested === true;
  const unclassifiedRows = toUnclassifiedRows(unclassified);

  const rememberRow = useCallback((row: UnclassifiedRowView, label: string) => {
    setSavedRows((previous) => ({ ...previous, [`${row.kind}:${row.key}`]: label }));
  }, []);

  // Rendered whenever a summary exists and something is unrecognised — also
  // when the digest is `insufficient`, where fixing the unknown apps is exactly
  // what would make the next review informative.
  const unclassifiedCard = unclassifiedRows.length > 0 && (
    <Surface as="section" className="period-unclassified">
      <SectionHeader
        title="还没认出来的应用和网站"
        help={{
          what: "告诉 MindFlow 这个应用或网站平时用来做什么。",
          effect:
            "设置会一直保存，以后再遇到它就会自动归类；最近 14 天的记录会在后台重新计算，稍后重新生成总结就能看到变化。",
        }}
      />
      <div className="period-unclassified-list">
        {unclassifiedRows.map((row) => {
          const rowKey = `${row.kind}:${row.key}`;
          return (
            <UnclassifiedRow
              key={rowKey}
              row={row}
              savedLabel={savedRows[rowKey] ?? null}
              onSaved={rememberRow}
            />
          );
        })}
      </div>
    </Surface>
  );

  return (
    <div className="period-review">
      <div className="page-toolbar">
        <Segmented
          ariaLabel="回顾范围"
          options={DAY_OPTIONS}
          value={String(days)}
          onChange={(value) => setDays(Number(value) as PeriodDays)}
        />
        <span className="period-ai-toggle">
          <input
            id="period-ai-toggle"
            type="checkbox"
            checked={useAi}
            onChange={(event) => setUseAi(event.target.checked)}
          />
          <label htmlFor="period-ai-toggle">让 AI 写文字总结</label>
          <HelpTip
            term="让 AI 写文字总结"
            what="开启后，会把下方“发给 AI 的内容”里的统计数字发给大模型，让它写成文字总结。"
            effect="只发送时长、次数这类数字，不发送窗口标题、网址和文件名；关闭后用本地规则写总结。"
          />
        </span>
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => generate(days, useAi)}
          disabled={loading}
        >
          {loading ? "生成中…" : "生成总结"}
        </button>
      </div>

      {error && (
        <div className="error-box" role="alert">
          {error}
          <button
            type="button"
            className="btn btn-secondary btn-sm report-retry"
            onClick={() => generate(days, useAi)}
          >
            重试
          </button>
        </div>
      )}

      {loading && (
        <div className="period-loading">
          <div className="spinner" />
          <span>正在汇总，AI 总结可能需要十几秒…</span>
        </div>
      )}

      {!loading && !view && !error && (
        <Surface>
          <div className="period-empty">选好回顾范围，点“生成总结”。</div>
        </Surface>
      )}

      {!loading && view && (
        <>
          <Surface as="section" className="period-narrative">
            <SectionHeader
              title={view.headline}
              actions={
                <span className="period-source-chip">
                  {view.source === "ai" ? "AI 总结" : "本地规则总结"}
                </span>
              }
            />
            {view.fallbackNote && (
              <p className="period-fallback-note">{view.fallbackNote}</p>
            )}
            <ul className="period-findings">
              {view.findings.map((finding, index) => (
                <li key={`finding-${index}`}>
                  {finding.text}
                  {finding.chips.map((chip) => (
                    <span className="period-chip" key={`${index}-${chip}`}>
                      {chip}
                    </span>
                  ))}
                </li>
              ))}
            </ul>
            {view.suggestions.length > 0 && (
              <div className="period-suggestions">
                <h3 className="period-subhead">下一步可以试试</h3>
                <ol>
                  {view.suggestions.map((suggestion, index) => (
                    <li key={`suggestion-${index}`}>{suggestion.text}</li>
                  ))}
                </ol>
              </div>
            )}
            {view.papers.length > 0 && (
              <p className="period-papers">
                参考文献：{view.papers.join("；")}
              </p>
            )}
          </Surface>

          {view.hasData && (
            <>
              <div className="t-state period-metrics">
                <MetricStrip label={`最近${view.days}天`} items={view.metrics} />
              </div>

              <div className="period-grid">
                <Surface as="section">
                  <SectionHeader title="主要在做什么" />
                  <BarChart
                    data={view.workMix}
                    ariaLabel="主要在做哪类工作"
                    unit=" 分钟"
                    valueName="时长"
                    height={220}
                    colorFor={() => "var(--signal)"}
                    emptyMessage="暂无数据"
                  />
                </Surface>
                <Surface as="section">
                  <SectionHeader title="在哪类事情上分心" />
                  <BarChart
                    data={view.distractionByContext}
                    ariaLabel="分心发生在哪类事情上"
                    unit=" 分钟"
                    valueName="分心时长"
                    height={220}
                    colorFor={() => "var(--warn, var(--signal))"}
                    emptyMessage="暂无数据"
                  />
                </Surface>
              </div>

              <Surface as="section" className="period-daily">
                <SectionHeader title="每天的情况" />
                <BarChart
                  data={view.daily}
                  ariaLabel="每天的分心时长"
                  unit=" 分钟"
                  valueName="分心时长"
                  height={220}
                  colorFor={() => "var(--warn, var(--signal))"}
                  emptyMessage="暂无数据"
                />
                <div className="period-dayparts">
                  {view.dayparts.map((part) => (
                    <div className="kv-row" key={part.key}>
                      <span>{part.label}</span>
                      <span className="period-dim">{part.text}</span>
                    </div>
                  ))}
                </div>
              </Surface>

              {unclassifiedCard}

              <Surface as="section">
                <SectionHeader
                  title="模型看到的分心信号"
                  help={{
                    what: "由行为分类模型对分心片段做解释得到，说明模型凭什么判断，不代表原因一定如此。",
                  }}
                />
                {view.signals.length > 0 ? (
                  <div className="period-signals">
                    {view.signals.map((signal) => (
                      <div className="kv-row" key={signal.key}>
                        <span>{`分心时「${signal.label}」${signal.direction}`}</span>
                        <span className="period-dim">{signal.text}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="period-empty">分心片段太少，暂时看不出稳定信号。</div>
                )}
              </Surface>
            </>
          )}

          {/* Without enough data there is no "每天的情况" to sit after, but the
              unknown apps are still exactly what the user can act on — and
              fixing them is what would make the next review informative. */}
          {!view.hasData && unclassifiedCard}

          <Surface as="section" className="period-interventions">
            <SectionHeader title="提醒与你的回应" />
            <p className="period-interventions-line">{view.interventionsLine}</p>
            {view.interventionChips.length > 0 && (
              <div className="period-chips">
                {view.interventionChips.map((chip) => (
                  <span className="period-chip" key={chip}>
                    {chip}
                  </span>
                ))}
              </div>
            )}
          </Surface>

          <Disclosure
            summary={
              aiRequested
                ? "查看发给 AI 的内容"
                : "查看可发给 AI 的内容（本次没有发送）"
            }
          >
            <p className="period-dim">{view.payloadNote}</p>
            <pre className="period-payload">{view.payloadPretty}</pre>
          </Disclosure>
        </>
      )}
    </div>
  );
}
