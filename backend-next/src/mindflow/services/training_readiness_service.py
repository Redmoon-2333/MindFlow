"""Training-readiness assessment for V2 feature-schema model training.

Matches feature windows to explicit feedback via the same time-overlap
semantics as ``train/v2.py:prepare_v2_training_data``.  All gate-check
counts derive from the resulting ``V2TrainingData``, not from raw table
aggregates.

Concepts:
  - raw events       — aggregate activity_events (COUNT, MIN, MAX)
  - v2 windows       — schema-v2 feature windows + matched eligibility
  - explicit labels  — feedback distribution from focus_session_feedback
  - trainable        — >=10 eligible matched windows with >=2 unique labels
  - evaluable        — >=10 explicit matched samples with >=3 distinct days
  - baseline-ready   — BaselineModel with >=30 total samples
  - activatable      — all seven gate checks pass
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mindflow.api.schemas import (
    ActivityEventsSummary,
    Blocker,
    FeedbackLabelSummary,
    GateStatus,
    TrainingReadinessResponse,
    V2GateCheck,
    V2WindowsSummary,
)
from mindflow.domain.baseline import BaselineModel
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.baseline import BaselineRepository
from mindflow.infrastructure.repositories.focus import (
    SQLAlchemyFocusSessionRepository,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.train.v2 import V2TrainingData, prepare_v2_training_data

# ── Thresholds (from train/v2.py) ──────────────────────────────────────────

_MIN_ELIGIBLE_WINDOWS = 10
_MIN_EXPLICIT_FEEDBACK = 20
_MIN_FOCUS = 5
_MIN_DISTRACT = 5
_MIN_DAYS = 1
_MIN_BALANCED_ACCURACY = 0.50
_MIN_MINORITY_F1 = 0.30
_BASELINE_MIN_SAMPLES = 30
_MIN_EVAL_DATES = 3  # GroupKFold requires >=3 distinct dates


# ── Typed gate definitions ─────────────────────────────────────────────────


@dataclass(frozen=True)
class _GateDef:
    key: str
    label: str
    threshold: str

    def compute(
        self, data: V2TrainingData,
    ) -> tuple[GateStatus, str, str, str]:
        """Return (status, actual_str, fail_msg, blocker_code)."""
        raise NotImplementedError


@dataclass(frozen=True)
class _MinimumDaysGate(_GateDef):
    def compute(self, data: V2TrainingData) -> tuple[GateStatus, str, str, str]:
        days = data.distinct_feedback_days
        if days >= _MIN_DAYS:
            return "passed", str(days), "", ""
        return (
            "failed", str(days),
            "反馈天数不足，至少需要连续使用并标记 1 天",
            "insufficient_days",
        )


@dataclass(frozen=True)
class _MinimumExplicitGate(_GateDef):
    def compute(self, data: V2TrainingData) -> tuple[GateStatus, str, str, str]:
        count = data.explicit_feedback_count
        if count >= _MIN_EXPLICIT_FEEDBACK:
            return "passed", str(count), "", ""
        return (
            "failed", str(count),
            f"反馈数量不足（当前 {count}，需要 {_MIN_EXPLICIT_FEEDBACK}）",
            "insufficient_feedback",
        )


@dataclass(frozen=True)
class _MinimumClassGate(_GateDef):
    def compute(self, data: V2TrainingData) -> tuple[GateStatus, str, str, str]:
        fc = data.explicit_focus_count
        dc = data.explicit_distract_count
        actual = f"专注={fc}, 分心={dc}"
        if fc >= _MIN_FOCUS and dc >= _MIN_DISTRACT:
            return "passed", actual, "", ""
        return (
            "failed", actual,
            f"类别反馈不足（专注 {fc}/{_MIN_FOCUS}，分心 {dc}/{_MIN_DISTRACT}）",
            "insufficient_class_feedback",
        )


@dataclass(frozen=True)
class _NotEvaluatedGate(_GateDef):
    reason: str = "尚未运行训练评估"

    def compute(self, data: V2TrainingData) -> tuple[GateStatus, str, str, str]:
        return "not_evaluated", "-", self.reason, "metric_not_evaluated"


@dataclass(frozen=True)
class _NotImplementedGate(_GateDef):
    reason: str = "需训练报告提供真实证据"

    def compute(self, data: V2TrainingData) -> tuple[GateStatus, str, str, str]:
        return "not_implemented", "-", self.reason, "not_implemented"


_GATES: list[_GateDef] = [
    _MinimumDaysGate(
        key="minimum_days", label="最少反馈天数",
        threshold=f">= {_MIN_DAYS}",
    ),
    _MinimumExplicitGate(
        key="minimum_explicit_feedback", label="最少显式反馈数",
        threshold=f">= {_MIN_EXPLICIT_FEEDBACK}",
    ),
    _MinimumClassGate(
        key="minimum_class_feedback", label="最少类别反馈数",
        threshold=f"专注 >= {_MIN_FOCUS} 且 分心 >= {_MIN_DISTRACT}",
    ),
    _NotEvaluatedGate(
        key="balanced_accuracy", label="平衡准确率",
        threshold=f">= {_MIN_BALANCED_ACCURACY}",
        reason="尚未运行训练评估，无法确定平衡准确率",
    ),
    _NotEvaluatedGate(
        key="minority_f1", label="少数类 F1",
        threshold=f">= {_MIN_MINORITY_F1}",
        reason="尚未运行训练评估，无法确定少数类 F1",
    ),
    _NotImplementedGate(
        key="calibration_better_than_rule", label="校准优于规则引擎",
        threshold="训练报告提供证据",
        reason="校准比较需训练报告提供真实证据，当前硬编码为通过，不可作为绿色通行",
    ),
    _NotImplementedGate(
        key="stable_date_folds", label="日期折叠稳定性",
        threshold="训练报告提供证据",
        reason="日期折叠稳定性需训练报告提供真实证据，当前硬编码为通过，不可作为绿色通行",
    ),
]


# ── Service ───────────────────────────────────────────────────────────────


class TrainingReadinessService:
    """Read-only assessment of whether enough data exists to train a V2 model.

    Uses the same time-overlap matching semantics as
    ``train/v2.py:prepare_v2_training_data`` and derives all gate-check
    counts from the resulting ``V2TrainingData``.
    """

    def __init__(
        self,
        telemetry_repo: TelemetryRepository,
        focus_repo: SQLAlchemyFocusSessionRepository,
        activity_repo: SQLAlchemyActivityRepository,
        baseline_repo: BaselineRepository,
        *,
        v2_training_mode: str = "rule_engine_only",
        user_id: int = 1,
        training_report: dict[str, Any] | None = None,
    ) -> None:
        self._telemetry_repo = telemetry_repo
        self._focus_repo = focus_repo
        self._activity_repo = activity_repo
        self._baseline_repo = baseline_repo
        self._v2_training_mode = v2_training_mode
        self._user_id = user_id
        self._training_report = training_report

    def _report_gate_override(
        self,
    ) -> dict[str, tuple[GateStatus, str, str, str]]:
        """Extract real post-training gate values from the training report.

        When a training report exists (a training job already ran), the four
        post-training gates (balanced_accuracy, minority_f1,
        calibration_better_than_rule, stable_date_folds) are reported with
        their actual evaluated values instead of the hard-coded
        ``not_evaluated`` / ``not_implemented`` placeholders.

        Returns a mapping ``{gate_key: (status, actual, threshold, fail_msg)}``
        for gates that have real evidence; empty dict when no report is
        available (callers keep the placeholder behaviour).
        """
        report = self._training_report
        if not report:
            return {}

        evaluation = report.get("evaluation") or {}
        candidate = evaluation.get("candidate") or {}
        rule_baseline = evaluation.get("rule_baseline") or {}
        fold_stability = evaluation.get("fold_stability") or {}
        checks = (report.get("quality_gate") or {}).get("checks") or {}

        overrides: dict[str, tuple[GateStatus, str, str, str]] = {}

        ba = candidate.get("balanced_accuracy")
        if ba is not None:
            passed = bool(checks.get("balanced_accuracy", False))
            overrides["balanced_accuracy"] = (
                "passed" if passed else "failed",
                f"{ba:.3f}",
                f">= {_MIN_BALANCED_ACCURACY}",
                "" if passed else "训练后平衡准确率未达阈值",
            )

        mf1 = candidate.get("minority_f1")
        if mf1 is not None:
            passed = bool(checks.get("minority_f1", False))
            overrides["minority_f1"] = (
                "passed" if passed else "failed",
                f"{mf1:.3f}",
                f">= {_MIN_MINORITY_F1}",
                "" if passed else "训练后少数类 F1 未达阈值",
            )

        cand_brier = candidate.get("brier_score")
        rule_brier = rule_baseline.get("brier_score")
        if cand_brier is not None and rule_brier is not None:
            passed = bool(checks.get("calibration_better_than_rule", False))
            overrides["calibration_better_than_rule"] = (
                "passed" if passed else "failed",
                f"候选 {cand_brier:.3f} vs 规则 {rule_brier:.3f}",
                "候选 Brier <= 规则 Brier + 0.01",
                "" if passed else "候选模型校准不优于规则引擎",
            )

        if fold_stability:
            passed = bool(checks.get("stable_date_folds", False))
            min_ba = fold_stability.get("min_balanced_accuracy")
            rng = fold_stability.get("range")
            actual_parts: list[str] = []
            if min_ba is not None:
                actual_parts.append(f"最差折 {min_ba:.3f}")
            if rng is not None:
                actual_parts.append(f"波动 {rng:.3f}")
            overrides["stable_date_folds"] = (
                "passed" if passed else "failed",
                ", ".join(actual_parts) or "-",
                "最差折 >= 0.50 且 波动 <= 0.35",
                "" if passed else "日期折叠稳定性未达阈值",
            )

        return overrides

    async def compute(self) -> TrainingReadinessResponse:
        uid = self._user_id

        # ── 1. Raw activity events (aggregate from activity_events) ─────
        act_sum = await self._activity_repo.get_activity_summary(uid)
        raw_events = ActivityEventsSummary(
            total_events=act_sum["total_events"],
            coverage_days=act_sum["coverage_days"],
            oldest_timestamp=act_sum["oldest_timestamp"],
            newest_timestamp=act_sum["newest_timestamp"],
        )

        # ── 2. V2 feature windows ──────────────────────────────────────
        windows = await self._telemetry_repo.list_feature_windows(
            uid, feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
        newest_window_start: str | None = None
        window_dates: set[str] = set()
        for w in windows:
            ws = w.get("window_start_utc", "")
            if ws:
                window_dates.add(ws[:10])
                if newest_window_start is None or ws > newest_window_start:
                    newest_window_start = ws
        sorted_dates = sorted(window_dates)
        if len(sorted_dates) >= 2:
            d1 = datetime.fromisoformat(sorted_dates[-1])
            d0 = datetime.fromisoformat(sorted_dates[0])
            date_range = (d1 - d0).days + 1
        else:
            date_range = len(sorted_dates) if sorted_dates else 0

        # ── 3. Raw feedback label distribution ─────────────────────────
        feedback_raw = await self._telemetry_repo.list_focus_feedback(uid)
        focus_total = sum(1 for f in feedback_raw if f["label"] == "focus")
        distract_total = sum(1 for f in feedback_raw if f["label"] == "distracted")
        mixed_total = sum(1 for f in feedback_raw if f["label"] == "mixed")
        feedback_labels = FeedbackLabelSummary(
            focus=focus_total,
            distract=distract_total,
            mixed=mixed_total,
            total=len(feedback_raw),
        )

        # ── 4. Match: join focus_sessions + feedback → feedback-with-timestamps ─
        sessions = await self._focus_repo.list_all(uid)
        session_map: dict[str, dict[str, Any]] = {s["id"]: s for s in sessions}

        feedback_with_times: list[dict[str, Any]] = []
        for fb in feedback_raw:
            sid = fb["session_id"]
            fcs = session_map.get(sid)
            start_time = fb.get("session_start_utc") or (fcs or {}).get("start_time")
            end_time = fb.get("session_end_utc") or (fcs or {}).get("end_time")
            if not start_time or not end_time:
                continue
            feedback_with_times.append({
                "session_id": sid,
                "start_time": start_time,
                "end_time": end_time,
                "label": fb["label"],
                "score": fb["score"],
                "task_type": fb.get("task_type"),
            })

        # ── 5. Run the real training-data preparer ─────────────────────
        training_data = prepare_v2_training_data(windows, feedback_with_times)

        # ── 6. Trainability ────────────────────────────────────────────
        eligible_mask = training_data.explicit_mask
        eligible_count = int(eligible_mask.sum())
        eligible_labels = training_data.labels[eligible_mask]
        unique_classes = int(len(set(eligible_labels.tolist())))
        trainable = eligible_count >= _MIN_ELIGIBLE_WINDOWS and unique_classes >= 2

        # ── 7. Evaluability ────────────────────────────────────────────
        explicit_count = training_data.explicit_feedback_count
        eval_dates = training_data.distinct_feedback_days
        evaluable = explicit_count >= _MIN_ELIGIBLE_WINDOWS and eval_dates >= _MIN_EVAL_DATES

        # ── 8. V2 windows summary ──────────────────────────────────────
        v2_windows = V2WindowsSummary(
            total=len(windows),
            schema_version=2,
            date_range_days=date_range,
            eligible_count=eligible_count,
            matched_focus_count=training_data.explicit_focus_count,
            matched_distract_count=training_data.explicit_distract_count,
            newest_window_start=newest_window_start,
        )

        # ── 9. Baseline readiness ──────────────────────────────────────
        baseline: BaselineModel | None = await self._baseline_repo.get_latest(uid)
        baseline_ready = (
            baseline is not None
            and baseline.has_sufficient_data(_BASELINE_MIN_SAMPLES)
        )

        # ── 10. Gate checks ────────────────────────────────────────────
        gates: list[V2GateCheck] = []
        report_gates = self._report_gate_override()
        for gd in _GATES:
            override = report_gates.get(gd.key)
            if override is not None:
                status_str, actual, threshold, fail_msg = override
                gate_passed = status_str == "passed"
                msg = fail_msg if fail_msg else _pass_message(gd.key)
                gates.append(V2GateCheck(
                    key=gd.key,
                    label=gd.label,
                    passed=gate_passed,
                    status=status_str,
                    actual=actual,
                    threshold=threshold,
                    message=msg,
                    blocker_code=(
                        gd.key if not gate_passed else ""
                    ),
                ))
                continue
            status_str, actual, fail_msg, blocker_code = gd.compute(training_data)
            gate_passed = status_str == "passed"
            msg = fail_msg if fail_msg else _pass_message(gd.key)
            gates.append(V2GateCheck(
                key=gd.key,
                label=gd.label,
                passed=gate_passed,
                status=status_str,
                actual=actual,
                threshold=gd.threshold,
                message=msg,
                blocker_code=blocker_code if not gate_passed else "",
            ))

        # ── 11. Blockers ───────────────────────────────────────────────
        blockers: list[Blocker] = []
        for gate in gates:
            if not gate.passed and gate.blocker_code:
                blockers.append(Blocker(code=gate.blocker_code, message=gate.message))
        if not trainable:
            if eligible_count < _MIN_ELIGIBLE_WINDOWS:
                msg = (
                    f"符合条件的窗口不足（当前 {eligible_count}"
                    f"，需要 {_MIN_ELIGIBLE_WINDOWS}）"
                )
                blockers.append(Blocker(
                    code="insufficient_eligible_windows",
                    message=msg,
                ))
            if unique_classes < 2:
                blockers.append(Blocker(
                    code="insufficient_classes",
                    message="需要至少两类标签（专注和分心），当前类别不足",
                ))
        # Deduplicate by code
        seen: set[str] = set()
        unique_blockers: list[Blocker] = []
        for b in blockers:
            if b.code not in seen:
                seen.add(b.code)
                unique_blockers.append(b)

        return TrainingReadinessResponse(
            raw_events=raw_events,
            v2_windows=v2_windows,
            feedback_labels=feedback_labels,
            trainable=trainable,
            trainable_window_count=eligible_count,
            trainable_class_count=unique_classes,
            evaluable=evaluable,
            evaluable_explicit_count=explicit_count,
            evaluable_date_count=eval_dates,
            baseline_ready=baseline_ready,
            current_mode=self._v2_training_mode,
            gates=gates,
            blockers=unique_blockers,
            current_training_job=None,
        )


def _pass_message(key: str) -> str:
    return {
        "minimum_days": "反馈天数满足最低要求",
        "minimum_explicit_feedback": "显式反馈数量满足最低要求",
        "minimum_class_feedback": "类别反馈数量满足最低要求",
        "balanced_accuracy": "平衡准确率满足最低要求",
        "minority_f1": "少数类 F1 满足最低要求",
        "calibration_better_than_rule": "校准优于规则引擎",
        "stable_date_folds": "日期折叠稳定性良好",
    }.get(key, "通过")
