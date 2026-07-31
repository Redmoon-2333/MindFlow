/**
 * Pure focus-prediction contract: canonical types, boundary parser, and view mapper.
 *
 * Mirrors backend `FocusPredictionResponse` (api/schemas.py) and the domain
 * six-value status Literal (domain/prediction.py). This module is intentionally
 * free of fetch/DOM so both Dashboard and Diagnostics share one exhaustive
 * render decision, and the contract can be exercised by pure tsx scripts.
 *
 * Contract invariants (enforced at the parse boundary, never in render):
 *   - only `status === "ready"` may carry a numeric `focus_probability`;
 *   - a finite `focus_probability` in [0, 1] renders as a percentage;
 *   - every non-ready / null / NaN / missing / unknown fixture renders `--`
 *     with a meaningful status label and preserved reason — never a percentage.
 */

export const FOCUS_PREDICTION_STATUSES = [
  "ready",
  "no_model",
  "no_data",
  "stale",
  "schema_mismatch",
  "inference_error",
] as const;

export type FocusPredictionStatus = (typeof FOCUS_PREDICTION_STATUSES)[number];

/** Status labels shown alongside `--`; ready renders the percentage itself. */
export const FOCUS_PREDICTION_STATUS_LABELS: Readonly<Record<FocusPredictionStatus, string>> = {
  ready: "已就绪",
  no_model: "无模型",
  no_data: "无数据",
  stale: "数据过期",
  schema_mismatch: "特征不匹配",
  inference_error: "推理错误",
};

export const FOCUS_PREDICTION_UNKNOWN_LABEL = "未知状态";

export interface FocusPredictionReadyState {
  readonly status: "ready";
  /** Finite probability in [0, 1] — guaranteed by the boundary parser. */
  readonly focus_probability: number;
  readonly mode: string;
  readonly reason: string;
}

export interface FocusPredictionUnavailableState {
  readonly status: Exclude<FocusPredictionStatus, "ready">;
  readonly focus_probability: null;
  readonly mode: string;
  readonly reason: string;
}

export interface FocusPredictionUnknownState {
  readonly status: "unknown";
  /** Untrusted original status value, surfaced so nothing is silently dropped. */
  readonly raw_status: string;
  readonly focus_probability: null;
  readonly mode: string;
  readonly reason: string;
}

export type FocusPredictionResponse =
  | FocusPredictionReadyState
  | FocusPredictionUnavailableState
  | FocusPredictionUnknownState;

const KNOWN_STATUS_SET: ReadonlySet<string> = new Set(FOCUS_PREDICTION_STATUSES);

function isKnownStatus(value: string): value is FocusPredictionStatus {
  return KNOWN_STATUS_SET.has(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Parse an untrusted wire payload into the canonical discriminated union. */
export function parseFocusPrediction(value: unknown): FocusPredictionResponse {
  if (!isRecord(value)) {
    return { status: "unknown", raw_status: "", focus_probability: null, mode: "", reason: "" };
  }
  const status = typeof value.status === "string" ? value.status : "";
  const mode = typeof value.mode === "string" ? value.mode : "";
  const reason = typeof value.reason === "string" ? value.reason : "";

  if (status === "ready") {
    const probability = value.focus_probability;
    if (typeof probability === "number" && Number.isFinite(probability)) {
      return { status: "ready", focus_probability: probability, mode, reason };
    }
    // ready with null/NaN/missing probability is an unavailable state — never 0%, never NaN%.
    return { status: "unknown", raw_status: status, focus_probability: null, mode, reason };
  }
  if (isKnownStatus(status)) {
    // Non-ready statuses are present-and-null even if the wire leaks a number.
    return { status, focus_probability: null, mode, reason };
  }
  return { status: "unknown", raw_status: status, focus_probability: null, mode, reason };
}

export interface FocusPredictionView {
  readonly ready: boolean;
  /** "73.1%" for finite ready probability, otherwise "--". */
  readonly display: string;
  readonly statusLabel: string;
  readonly reason: string;
  readonly mode: string;
}

function formatProbability(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}

/** Exhaustive render decision shared by Dashboard and Diagnostics. */
export function toFocusPredictionView(state: FocusPredictionResponse): FocusPredictionView {
  switch (state.status) {
    case "ready": {
      return {
        ready: true,
        display: formatProbability(state.focus_probability),
        statusLabel: FOCUS_PREDICTION_STATUS_LABELS.ready,
        reason: state.reason,
        mode: state.mode,
      };
    }
    case "no_model":
    case "no_data":
    case "stale":
    case "schema_mismatch":
    case "inference_error": {
      return {
        ready: false,
        display: "--",
        statusLabel: FOCUS_PREDICTION_STATUS_LABELS[state.status],
        reason: state.reason,
        mode: state.mode,
      };
    }
    case "unknown": {
      return {
        ready: false,
        display: "--",
        statusLabel: FOCUS_PREDICTION_UNKNOWN_LABEL,
        reason: state.reason || `未知状态: ${state.raw_status}`,
        mode: state.mode,
      };
    }
    default: {
      // Exhaustiveness guard: a new status variant must be handled above.
      const exhaustive: never = state;
      return exhaustive;
    }
  }
}
