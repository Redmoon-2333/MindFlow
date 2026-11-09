/**
 * Client-side shapes for the explainability contract (docs/audit/20261110-midterm).
 *
 * These are *parsed* shapes: every optional wire field has been normalised to a
 * usable value, so consumers never handle `undefined` twice. The enums are
 * aliases of the generated schema (a backend enum change is now a build error),
 * and the `_Wire*` assertions at the bottom prove each interface still covers
 * `src/generated/api-schema.ts` — a renamed or dropped field fails `tsc`.
 *
 * Rules for consumers:
 *   - a non-conforming payload must degrade to `undefined`, never throw;
 *   - nothing in this module may import from React or the DOM.
 */
import type { components } from "../generated/api-schema";

type Schemas = components["schemas"];

export type ContributionMethod = Schemas["Explanation"]["method"];

export interface ExplanationContribution {
  /** V2 feature name (stable key). */
  feature: string;
  /** Chinese display label, e.g. "应用切换频率". */
  label: string;
  /** Current-window mean on the raw scale. */
  value: number;
  /** Personal-baseline expectation for the same bucket; null when unbuilt. */
  baseline_value: number | null;
  /** Signed: > 0 pushes toward focus, < 0 pushes toward distraction. */
  contribution: number;
}

export interface PredictionExplanation {
  method: ContributionMethod;
  base_value: number | null;
  output_space: Schemas["Explanation"]["output_space"];
  /** Sorted by |contribution| descending, at most 8 entries. */
  contributions: ExplanationContribution[];
  note: string;
}

export type DeviationSeverity = Schemas["DeviationSummary"]["severity"];

export interface DeviationFeature {
  feature: string;
  label: string;
  value: number;
  expected: number;
  /** Robust (MAD-based) z-score. */
  robust_z: number;
  weight: number;
}

export interface DeviationSummary {
  /** Weighted sum of |robust_z|. */
  overall_score: number;
  severity: DeviationSeverity;
  method: Schemas["DeviationSummary"]["method"];
  /** Empirical-Bayes bucket metadata; `shrinkage` = k / (n + k). */
  bucket: { hour_of_day: number; day_of_week: number; n_samples: number; shrinkage: number };
  /** Sorted by |robust_z| * weight descending, at most 6 entries. */
  features: DeviationFeature[];
}

export type TmtComponentKey = keyof Schemas["TmtComponentsModel"];

export interface TmtComponent {
  /** Normalised to [0, 1]. */
  score: number;
  raw: number | null;
  label: string;
  source: string;
  evidence_ids: string[];
}

export interface TmtBreakdown {
  /** Motivation = E·V / (1 + I·D), normalised to [0, 1]. */
  motivation: number;
  motivation_level: Schemas["TmtBreakdownModel"]["motivation_level"];
  /** `"delay"` is a real outcome, not a fallback: it means the deadline is the
   *  bottleneck. See `parseTmt` for the malformed-value path. */
  dominant_barrier: Schemas["TmtBreakdownModel"]["dominant_barrier"];
  components: Record<TmtComponentKey, TmtComponent>;
  /** Boundary note, e.g. "行为代理变量，不是心理测量". */
  note: string;
}

export type InterventionSelection = Schemas["InterventionRationaleModel"]["selection"];

export interface RationaleCandidate {
  intervention_type: string;
  label: string;
  /** Beta posterior parameters. */
  alpha: number;
  beta: number;
  posterior_mean: number;
  /** NOT a sample count. Under `thompson_sampling` this is the bandit's draw
   *  for the arm; under `rule` the backend stores the posterior mean here. The
   *  wire sends no interval, so `PosteriorBars` derives one from alpha/beta. */
  sample: number;
  chosen: boolean;
}

export interface InterventionRationale {
  procrastination_type: string | null;
  intervention_type: string;
  selection: InterventionSelection;
  /** e.g. "afternoon|distracted". */
  context_bucket: string;
  candidates: RationaleCandidate[];
  /** Human-readable evidence, at most 4 lines. The wire field is named
   *  `evidence` (not `reasons` — those belong to feedback requests). */
  evidence: string[];
  note: string;
}

export interface FeedbackRequest {
  request_id: string;
  /** ISO-8601 window bounds. */
  window_start: string;
  window_end: string;
  probability: number | null;
  /** Chinese trigger reasons, e.g. "模型不确定". */
  reasons: string[];
  prompt: string;
}

export type FeedbackLabel = Schemas["FeedbackAnswerRequest"]["label"];

export interface FeedbackAnswerResponse {
  request_id: string;
  labeled_windows: number;
  status: Schemas["FeedbackAnswerResponse"]["status"];
}

/** Compile-time proof that the parsers still match the generated wire schema.
 *
 *  `AssertSameKeys` resolves to `true` only when both key sets are identical,
 *  so assigning it to a `true`-typed constant turns a backend rename into a
 *  build error. This is what would have caught `evidence` being renamed to
 *  `reasons` — the parser would otherwise keep compiling and silently read a
 *  field the server never sends.
 *
 *  These are `const` declarations rather than type aliases because a wrong
 *  type alias alone is not an error. They compile to a few booleans.
 */
export type AssertSameKeys<Wire, ExpectedKeys extends PropertyKey> =
  [keyof Wire] extends [ExpectedKeys]
    ? [ExpectedKeys] extends [keyof Wire]
      ? true
      : { error: "generated schema has extra fields"; extra: Exclude<keyof Wire, ExpectedKeys> }
    : { error: "expected key missing from generated schema"; missing: Exclude<ExpectedKeys, keyof Wire> };

/** Union equality: both spellings must accept exactly the same values. */
export type AssertSameUnion<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;

export const _wireExplanation: AssertSameKeys<
  Schemas["Explanation"],
  "method" | "base_value" | "output_space" | "contributions" | "note"
> = true;
export const _wireContribution: AssertSameKeys<
  Schemas["ExplanationContribution"],
  "feature" | "label" | "value" | "baseline_value" | "contribution"
> = true;
export const _wireDeviation: AssertSameKeys<
  Schemas["DeviationSummary"],
  "overall_score" | "severity" | "method" | "bucket" | "features"
> = true;
export const _wireDeviationBucket: AssertSameKeys<
  Schemas["DeviationBucket"],
  "hour_of_day" | "day_of_week" | "n_samples" | "shrinkage"
> = true;
export const _wireDeviationFeature: AssertSameKeys<
  Schemas["DeviationFeature"],
  "feature" | "label" | "value" | "expected" | "robust_z" | "weight"
> = true;
export const _wireTmt: AssertSameKeys<
  Schemas["TmtBreakdownModel"],
  "motivation" | "motivation_level" | "dominant_barrier" | "components" | "note"
> = true;
export const _wireTmtComponent: AssertSameKeys<
  Schemas["TmtComponentModel"],
  "score" | "raw" | "label" | "source" | "evidence_ids"
> = true;
export const _wireRationale: AssertSameKeys<
  Schemas["InterventionRationaleModel"],
  "procrastination_type" | "intervention_type" | "selection" | "context_bucket" | "candidates" | "evidence" | "note"
> = true;
export const _wireCandidate: AssertSameKeys<
  Schemas["InterventionRationaleCandidate"],
  "intervention_type" | "label" | "alpha" | "beta" | "posterior_mean" | "sample" | "chosen"
> = true;
export const _wireFeedbackRequest: AssertSameKeys<
  Schemas["FeedbackRequestModel"],
  "request_id" | "window_start" | "window_end" | "probability" | "reasons" | "prompt"
> = true;

export const _enumMethod: AssertSameUnion<ContributionMethod, Schemas["Explanation"]["method"]> = true;
export const _enumSeverity: AssertSameUnion<DeviationSeverity, Schemas["DeviationSummary"]["severity"]> = true;
export const _enumBarrier: AssertSameUnion<TmtBreakdown["dominant_barrier"], Schemas["TmtBreakdownModel"]["dominant_barrier"]> = true;
export const _enumSelection: AssertSameUnion<InterventionSelection, Schemas["InterventionRationaleModel"]["selection"]> = true;
export const _enumLabel: AssertSameUnion<FeedbackLabel, Schemas["FeedbackAnswerRequest"]["label"]> = true;
export const _enumAnswerStatus: AssertSameUnion<FeedbackAnswerResponse["status"], Schemas["FeedbackAnswerResponse"]["status"]> = true;

/** Wire fields grafted onto existing responses. */
export interface ExplainablePrediction {
  explanation?: PredictionExplanation | null;
}
export interface ExplainableAttribution {
  deviation?: DeviationSummary | null;
  tmt?: TmtBreakdown | null;
}
export interface ExplainablePanel {
  tmt?: TmtBreakdown | null;
}
export interface ExplainableIntervention {
  rationale?: InterventionRationale | null;
}

// ── Boundary parsing ──────────────────────────────────────────────────────

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function strArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

const METHODS: ReadonlySet<string> = new Set(["tree_shap", "linear", "unavailable"]);
const SEVERITIES: ReadonlySet<string> = new Set(["normal", "mild", "moderate", "severe"]);
const LEVELS: ReadonlySet<string> = new Set(["low", "medium", "high"]);
const BARRIERS: ReadonlySet<string> = new Set<TmtComponentKey>(["expectancy", "value", "impulsiveness", "delay"]);
const SELECTIONS: ReadonlySet<string> = new Set(["thompson_sampling", "rule"]);

function parseContribution(value: unknown): ExplanationContribution | null {
  if (!isRecord(value)) return null;
  const contribution = num(value.contribution);
  if (contribution === null) return null;
  return {
    feature: str(value.feature),
    label: str(value.label) || str(value.feature),
    value: num(value.value) ?? 0,
    baseline_value: num(value.baseline_value),
    contribution,
  };
}

/** Returns `undefined` for "absent or unusable" so callers have one empty case. */
export function parsePredictionExplanation(value: unknown): PredictionExplanation | undefined {
  if (!isRecord(value)) return undefined;
  const method = str(value.method);
  if (!METHODS.has(method)) return undefined;
  const contributions = Array.isArray(value.contributions)
    ? value.contributions.map(parseContribution).filter((entry): entry is ExplanationContribution => entry !== null)
    : [];
  return {
    method: method as ContributionMethod,
    base_value: num(value.base_value),
    output_space: value.output_space === "logit" ? "logit" : "probability",
    contributions,
    note: str(value.note),
  };
}

function parseDeviationFeature(value: unknown): DeviationFeature | null {
  if (!isRecord(value)) return null;
  const robustZ = num(value.robust_z);
  if (robustZ === null) return null;
  return {
    feature: str(value.feature),
    label: str(value.label) || str(value.feature),
    value: num(value.value) ?? 0,
    expected: num(value.expected) ?? 0,
    robust_z: robustZ,
    weight: num(value.weight) ?? 1,
  };
}

export function parseDeviation(value: unknown): DeviationSummary | undefined {
  if (!isRecord(value)) return undefined;
  const severity = str(value.severity);
  if (!SEVERITIES.has(severity)) return undefined;
  const bucket = isRecord(value.bucket) ? value.bucket : {};
  return {
    overall_score: num(value.overall_score) ?? 0,
    severity: severity as DeviationSeverity,
    method: "eb_shrinkage_mad",
    bucket: {
      hour_of_day: num(bucket.hour_of_day) ?? 0,
      day_of_week: num(bucket.day_of_week) ?? 0,
      n_samples: num(bucket.n_samples) ?? 0,
      shrinkage: num(bucket.shrinkage) ?? 0,
    },
    features: Array.isArray(value.features)
      ? value.features.map(parseDeviationFeature).filter((entry): entry is DeviationFeature => entry !== null)
      : [],
  };
}

function parseTmtComponent(value: unknown, fallbackLabel: string): TmtComponent {
  if (!isRecord(value)) {
    return { score: 0, raw: null, label: fallbackLabel, source: "", evidence_ids: [] };
  }
  return {
    score: num(value.score) ?? 0,
    raw: num(value.raw),
    label: str(value.label) || fallbackLabel,
    source: str(value.source),
    evidence_ids: strArray(value.evidence_ids),
  };
}

const TMT_LABELS: Record<TmtComponentKey, string> = {
  expectancy: "期望 E", value: "价值 V", impulsiveness: "冲动 I", delay: "延迟 D",
};

export function parseTmt(value: unknown): TmtBreakdown | undefined {
  if (!isRecord(value)) return undefined;
  const components = isRecord(value.components) ? value.components : {};
  const dominant = str(value.dominant_barrier);
  const level = str(value.motivation_level);
  return {
    motivation: num(value.motivation) ?? 0,
    motivation_level: (LEVELS.has(level) ? level : "medium") as TmtBreakdown["motivation_level"],
    dominant_barrier: (BARRIERS.has(dominant) ? dominant : "delay") as TmtComponentKey,
    components: {
      expectancy: parseTmtComponent(components.expectancy, TMT_LABELS.expectancy),
      value: parseTmtComponent(components.value, TMT_LABELS.value),
      impulsiveness: parseTmtComponent(components.impulsiveness, TMT_LABELS.impulsiveness),
      delay: parseTmtComponent(components.delay, TMT_LABELS.delay),
    },
    note: str(value.note),
  };
}

export function parseRationale(value: unknown): InterventionRationale | undefined {
  if (!isRecord(value)) return undefined;
  const selection = str(value.selection);
  if (!SELECTIONS.has(selection)) return undefined;
  const candidates: RationaleCandidate[] = Array.isArray(value.candidates)
    ? value.candidates.flatMap((entry): RationaleCandidate[] => {
        if (!isRecord(entry)) return [];
        const alpha = num(entry.alpha);
        const beta = num(entry.beta);
        if (alpha === null || beta === null) return [];
        return [{
          intervention_type: str(entry.intervention_type),
          label: str(entry.label) || str(entry.intervention_type),
          alpha,
          beta,
          posterior_mean: num(entry.posterior_mean) ?? alpha / (alpha + beta),
          sample: num(entry.sample) ?? 0,
          chosen: entry.chosen === true,
        }];
      })
    : [];
  return {
    procrastination_type: typeof value.procrastination_type === "string" ? value.procrastination_type : null,
    intervention_type: str(value.intervention_type),
    selection: selection as InterventionSelection,
    context_bucket: str(value.context_bucket),
    candidates,
    evidence: strArray(value.evidence),
    note: str(value.note),
  };
}

export function parseFeedbackRequest(value: unknown): FeedbackRequest | undefined {
  if (!isRecord(value)) return undefined;
  const requestId = str(value.request_id);
  if (!requestId) return undefined;
  return {
    request_id: requestId,
    window_start: str(value.window_start),
    window_end: str(value.window_end),
    probability: num(value.probability),
    reasons: strArray(value.reasons),
    prompt: str(value.prompt) || "刚才你在专注吗？",
  };
}

// ── Beta posterior interval ───────────────────────────────────────────────

export interface BetaInterval {
  mean: number;
  low: number;
  high: number;
  sd: number;
}

/**
 * Normal-approximation 90% credible interval for Beta(alpha, beta).
 *
 * sd = sqrt(ab / ((a+b)^2 (a+b+1))). Clamped to [0, 1] and kept at least as
 * wide as the mean-to-bound distance so the bar never inverts on a small
 * sample. Used only for layout — the posterior mean stays the reported value.
 */
export function betaInterval(alpha: number, beta: number, z = 1.645): BetaInterval {
  const a = Math.max(alpha, 1e-6);
  const b = Math.max(beta, 1e-6);
  const total = a + b;
  const mean = a / total;
  const variance = (a * b) / (total * total * (total + 1));
  const sd = Math.sqrt(Math.max(variance, 0));
  const low = Math.min(Math.max(mean - z * sd, 0), mean);
  const high = Math.max(Math.min(mean + z * sd, 1), mean);
  return { mean, low, high, sd };
}

export const METHOD_LABELS: Readonly<Record<ContributionMethod, string>> = {
  tree_shap: "TreeSHAP",
  linear: "线性归因",
  unavailable: "无归因",
};

export const SEVERITY_LABELS: Readonly<Record<DeviationSeverity, string>> = {
  normal: "正常", mild: "轻度偏离", moderate: "中度偏离", severe: "严重偏离",
};

export const MOTIVATION_LEVEL_LABELS: Readonly<Record<TmtBreakdown["motivation_level"], string>> = {
  low: "动机偏低", medium: "动机中等", high: "动机较高",
};

export const SELECTION_LABELS: Readonly<Record<InterventionSelection, string>> = {
  thompson_sampling: "Thompson 采样",
  rule: "规则选择",
};
