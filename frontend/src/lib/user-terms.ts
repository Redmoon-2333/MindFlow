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
};

/** Keep unknown/free-form values intact; labels only affect presentation. */
export function userTerm(value?: string | null): string {
  if (!value) return "—";
  const label = terms[value.toLowerCase()];
  return label ? `${label}（${value}）` : value;
}
