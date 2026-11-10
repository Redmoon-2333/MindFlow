const terms: Record<string, string> = {
  avg_focus_block_min: "平均连续专注时长（分钟）", total_events: "活动事件总数", distraction_ratio: "分心占比",
  up: "上升", down: "下降", stable: "稳定",
  task_aversion: "任务厌恶", impulsivity: "冲动分心", decisional: "决策拖延",
  perfectionism: "完美主义", emotional_regulation: "情绪调节困难",
  behavioral_experiment: "行为实验", cognitive_restructuring: "认知重构",
  stimulus_control: "刺激控制", goal_setting: "目标设定", graded_exposure: "分级面对任务", mindfulness: "正念练习",
  insufficient_days: "反馈天数不足", insufficient_feedback: "反馈数量不足",
  insufficient_class_feedback: "类别反馈不足", metric_not_evaluated: "指标尚未评估",
  insufficient_independent_date_groups: "独立反馈日期不足", insufficient_eligible_windows: "合格窗口不足",
  insufficient_classes: "标签类别不足", not_evaluated: "未评估", not_implemented: "未实现",
  v2_models_not_loaded: "尚未加载个人模型", rule_engine_only: "仅规则引擎",
  running: "运行中", stopped: "已停止", disabled: "已关闭", ready: "就绪", shadow: "影子模式",
  personal_model: "个人模型", demo_v4: "演示模型", base: "基础模型",
  synthetic_base_not_personally_validated: "内置基础模型已可预测，等待个人数据验证",
  calibration_better_than_rule: "校准不如规则引擎",
  unavailable: "不可用", completed: "已完成", failed: "失败", pending: "等待中",
  ok: "正常", healthy: "正常",
};

/** Translate a backend identifier to its Chinese name; keep unknown values
 *  intact. §8.17.3 drops the appended `（raw value）` — an end user reading a
 *  status should never be shown the wire token as well. */
export function userTerm(value?: string | null): string {
  if (!value) return "—";
  return terms[value.toLowerCase()] ?? value;
}
