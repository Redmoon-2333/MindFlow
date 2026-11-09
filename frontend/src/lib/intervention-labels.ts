// "任务拆解" is the term the explainability copy uses; the history badge used
// to say "任务分解", so the same technique appeared under two names on one page.
export const INTERVENTION_TYPE_LABELS: Record<string, string> = {
  task_breakdown: "任务拆解",
  nudge: "行动提示",
  environment_optimization: "环境优化",
  smart_prioritization: "优先级建议",
};

export const INTERVENTION_TYPE_BADGES: Record<string, string> = {
  task_breakdown: "badge-primary",
  nudge: "badge-info",
  environment_optimization: "badge-success",
  smart_prioritization: "badge-warning",
};

export function getInterventionTypeLabel(interventionType?: string): string {
  return INTERVENTION_TYPE_LABELS[(interventionType || "").toLowerCase()] || "专注干预";
}
