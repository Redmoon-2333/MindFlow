"""Eval scenarios for G006 — single-expert vs expert-panel comparison.

Each ``EvalScenario`` is a frozen dataclass holding a synthetic ``EvidenceBundle``,
the gold-standard ground truth labels, and metadata for attribution.

Design (per 07-agent-upgrade-design.md §8):
  - 25 type-specific scenarios: 5 per procrastination type
    (2 clear-cut, 1 borderline, 1 co-morbid, 1 edge case)
  - 5 mixed/ambiguous scenarios: borderline or multi-signal cases
    where the rule engine may disagree with gold-standard human labels
  - Total: 30 scenarios
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from mindflow.domain.evidence import EvidenceBundle, EvidenceItem
from mindflow.domain.procrastination import (
    BehaviorSummary,
    CBTTechnique,
    ProcrastinationType,
)

# ---------------------------------------------------------------------------
# EvalScenario — gold-standard ground truth
# ---------------------------------------------------------------------------

# Severity literal type for type annotations
# Use str instead of Literal to avoid mypy narrowing issues in conditionals.
_Severity = str


@dataclass(frozen=True)
class EvalScenario:
    """A single evaluation scenario with gold-standard ground truth.

    Attributes:
        scenario_id: Unique identifier (e.g. "IMP-001").
        description: Chinese description of the scenario.
        bundle: Synthetic EvidenceBundle with pre-computed features.
        expected_types: Gold-standard procrastination types, ordered by
            confidence descending (human-labelled ground truth).
        expected_technique: Gold-standard recommended CBT technique,
            or None when no significant procrastination expected.
    """

    scenario_id: str
    description: str
    bundle: EvidenceBundle
    expected_types: tuple[ProcrastinationType, ...]
    expected_technique: CBTTechnique | None

    @property
    def expected_top_type(self) -> ProcrastinationType | None:
        """Highest-confidence expected type, or None for empty gold standard."""
        return self.expected_types[0] if self.expected_types else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WINDOW = (datetime(2026, 7, 18, 9, 0, tzinfo=UTC), datetime(2026, 7, 18, 11, 0, tzinfo=UTC))
_EMPTY_TUPLE: tuple[()] = ()


def _item(
    metric: str,
    value: float,
    baseline: float | None,
    severity: str,
    confidence: float,
    source: str,
    human_readable: str,
) -> EvidenceItem:
    return EvidenceItem(
        metric=metric,
        value=value,
        baseline=baseline,
        severity=severity,  # type: ignore[arg-type]
        confidence=confidence,
        source=source,
        human_readable=human_readable,
    )


def _focus_item(focus_ratio: float, baseline: float = 0.72) -> EvidenceItem:
    return _item(
        metric="focus_score",
        value=focus_ratio,
        baseline=baseline,
        severity="severe" if focus_ratio < 0.25 else "moderate" if focus_ratio < 0.45 else "mild",
        confidence=0.85,
        source="feature_computation",
        human_readable=f"专注度 {focus_ratio:.0%}（基线 {baseline:.0%}）",
    )


def _switch_item(switches: float, baseline: float = 8.0) -> EvidenceItem:
    sev = (
        "severe" if switches >= 20
        else "moderate" if switches >= 14
        else "mild" if switches >= 10
        else "info"
    )
    return _item(
        metric="switch_rate",
        value=switches,
        baseline=baseline,
        severity=sev,
        confidence=0.80,
        source="feature_computation",
        human_readable=f"切换频率 {switches:.1f} 次/小时（基线 {baseline:.1f}）",
    )


def _block_item(block_s: float, baseline: float = 600.0) -> EvidenceItem:
    sev = (
        "severe" if block_s < 120
        else "moderate" if block_s < 300
        else "mild" if block_s < 600
        else "info"
    )
    return _item(
        metric="longest_focus_block_s",
        value=block_s,
        baseline=baseline,
        severity=sev,
        confidence=0.90,
        source="feature_computation",
        human_readable=f"最长专注块 {block_s:.0f}s（基线 {baseline:.0f}s）",
    )


def _social_item(ratio: float, baseline: float = 0.20) -> EvidenceItem:
    sev = (
        "severe" if ratio >= 0.70
        else "moderate" if ratio >= 0.55
        else "mild" if ratio >= 0.35
        else "info"
    )
    return _item(
        metric="social_media_ratio",
        value=ratio,
        baseline=baseline,
        severity=sev,
        confidence=0.75,
        source="feature_computation",
        human_readable=f"娱乐占比 {ratio:.0%}（基线 {baseline:.0%}）",
    )


def _delay_item(delay_min: float, baseline: float = 10.0) -> EvidenceItem:
    sev = (
        "severe"
        if delay_min >= 60
        else "moderate"
        if delay_min >= 30
        else "mild"
        if delay_min >= 15
        else "info"
    )
    return _item(
        metric="start_delay_min",
        value=delay_min,
        baseline=baseline,
        severity=sev,
        confidence=0.80,
        source="feature_computation",
        human_readable=f"启动延迟 {delay_min:.0f}min（基线 {baseline:.0f}min）",
    )


def _deviation_item(deviation: float) -> EvidenceItem:
    abs_d = abs(deviation)
    sev = (
        "severe"
        if abs_d >= 1.5
        else "moderate"
        if abs_d >= 1.0
        else "mild"
        if abs_d >= 0.5
        else "info"
    )
    return _item(
        metric="behavior_deviation",
        value=deviation,
        baseline=0.0,
        severity=sev,
        confidence=0.85,
        source="welford_baseline",
        human_readable=f"行为偏差 {deviation:+.1f}σ",
    )


def _make_bundle(
    *,
    user_id: int = 1,
    intended_task: str | None,
    duration_min: float,
    actual_focus_min: float,
    context_switches_per_hour: float,
    longest_focus_block_s: float,
    social_media_ratio: float,
    start_delay_min: float = 2.0,
    keyword_flags: frozenset[str] = frozenset(),
    baseline_deviation: float | None = None,
) -> EvidenceBundle:
    """Build an EvidenceBundle from BehaviorSummary parameters.

    EvidenceItems are automatically derived from the summary fields.
    """
    items: list[EvidenceItem] = []

    focus_ratio = actual_focus_min / duration_min if duration_min > 0 else 0.0
    items.append(_focus_item(focus_ratio))
    items.append(_switch_item(context_switches_per_hour))
    items.append(_block_item(longest_focus_block_s))
    items.append(_social_item(social_media_ratio))

    if start_delay_min >= 15:
        items.append(_delay_item(start_delay_min))

    if baseline_deviation is not None and abs(baseline_deviation) >= 0.5:
        items.append(_deviation_item(baseline_deviation))

    return EvidenceBundle(
        user_id=user_id,
        window=_WINDOW,
        items=tuple(items),
        behavior_summary=BehaviorSummary(
            intended_task=intended_task,
            duration_min=duration_min,
            actual_focus_min=actual_focus_min,
            context_switches_per_hour=context_switches_per_hour,
            longest_focus_block_s=longest_focus_block_s,
            social_media_ratio=social_media_ratio,
            start_delay_min=start_delay_min,
            keyword_flags=keyword_flags,
            baseline_deviation=baseline_deviation,
        ),
        intervention_history=_EMPTY_TUPLE,
        novelty_flags=_EMPTY_TUPLE,
    )


# ===================================================================
# 30 Scenarios — organised by procrastination type
# ===================================================================
# IMPULSIVITY (IMP) — 5 scenarios
# ===================================================================

_SCENARIOS: list[EvalScenario] = []

# -- IMP-001: Classic high-switch, short-block pattern -----------------
_SCENARIOS.append(EvalScenario(
    scenario_id="IMP-001",
    description="高频切换与短专注块 — 典型冲动分心模式",
    bundle=_make_bundle(
        intended_task="写课程论文",
        duration_min=60.0,
        actual_focus_min=20.0,
        context_switches_per_hour=18.0,
        longest_focus_block_s=120.0,
        social_media_ratio=0.30,
    ),
    expected_types=(ProcrastinationType.IMPULSIVITY,),
    expected_technique=CBTTechnique.STIMULUS_CONTROL,
))

# -- IMP-002: Severe — very high switches, extremely short blocks -----
_SCENARIOS.append(EvalScenario(
    scenario_id="IMP-002",
    description="严重冲动分心 — 极高频切换与极短专注块",
    bundle=_make_bundle(
        intended_task="复习期末考试",
        duration_min=60.0,
        actual_focus_min=5.0,
        context_switches_per_hour=35.0,
        longest_focus_block_s=30.0,
        social_media_ratio=0.15,
    ),
    expected_types=(ProcrastinationType.IMPULSIVITY,),
    expected_technique=CBTTechnique.STIMULUS_CONTROL,
))

# -- IMP-003: Borderline — just above threshold -----------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="IMP-003",
    description="临界冲动 — 刚好满足冲动阈值",
    bundle=_make_bundle(
        intended_task="完成小组作业",
        duration_min=60.0,
        actual_focus_min=25.0,
        context_switches_per_hour=13.0,
        longest_focus_block_s=250.0,
        social_media_ratio=0.25,
    ),
    expected_types=(ProcrastinationType.IMPULSIVITY,),
    expected_technique=CBTTechnique.STIMULUS_CONTROL,
))

# -- IMP-004: Impulsivity + emotional regulation ----------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="IMP-004",
    description="冲动伴随情绪调节 — 高切换 + 高娱乐占比",
    bundle=_make_bundle(
        intended_task="准备汇报PPT",
        duration_min=60.0,
        actual_focus_min=10.0,
        context_switches_per_hour=20.0,
        longest_focus_block_s=60.0,
        social_media_ratio=0.65,
    ),
    expected_types=(ProcrastinationType.IMPULSIVITY, ProcrastinationType.EMOTIONAL_REGULATION),
    expected_technique=CBTTechnique.STIMULUS_CONTROL,
))

# -- IMP-005: Impulsivity with deceptive total focus ------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="IMP-005",
    description="冲动但总专注时长尚可 — 短暂但频繁的专注",
    bundle=_make_bundle(
        intended_task="编程练习",
        duration_min=120.0,
        actual_focus_min=50.0,
        context_switches_per_hour=16.0,
        longest_focus_block_s=150.0,
        social_media_ratio=0.35,
    ),
    expected_types=(ProcrastinationType.IMPULSIVITY,),
    expected_technique=CBTTechnique.STIMULUS_CONTROL,
))

# ===================================================================
# DECISIONAL (DEC) — 5 scenarios
# ===================================================================

# -- DEC-001: Classic decisional --------------------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="DEC-001",
    description="典型启动困难 — 延迟后恢复正常专注",
    bundle=_make_bundle(
        intended_task="写实验报告",
        duration_min=90.0,
        actual_focus_min=45.0,
        context_switches_per_hour=6.0,
        longest_focus_block_s=600.0,
        social_media_ratio=0.20,
        start_delay_min=45.0,
    ),
    expected_types=(ProcrastinationType.DECISIONAL,),
    expected_technique=CBTTechnique.GOAL_SETTING,
))

# -- DEC-002: Severe — very long delay --------------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="DEC-002",
    description="严重启动困难 — 极长延迟后高专注",
    bundle=_make_bundle(
        intended_task="完成数学作业",
        duration_min=120.0,
        actual_focus_min=60.0,
        context_switches_per_hour=5.0,
        longest_focus_block_s=900.0,
        social_media_ratio=0.10,
        start_delay_min=90.0,
    ),
    expected_types=(ProcrastinationType.DECISIONAL,),
    expected_technique=CBTTechnique.GOAL_SETTING,
))

# -- DEC-003: Borderline -- just above both thresholds ----------------
_SCENARIOS.append(EvalScenario(
    scenario_id="DEC-003",
    description="临界启动困难 — 刚好满足延迟与恢复阈值",
    bundle=_make_bundle(
        intended_task="读文献",
        duration_min=100.0,
        actual_focus_min=42.0,
        context_switches_per_hour=8.0,
        longest_focus_block_s=500.0,
        social_media_ratio=0.25,
        start_delay_min=31.0,
    ),
    expected_types=(ProcrastinationType.DECISIONAL,),
    expected_technique=CBTTechnique.GOAL_SETTING,
))

# -- DEC-004: Decisional + perfectionism keywords ---------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="DEC-004",
    description="决策困难伴随完美主义词汇",
    bundle=_make_bundle(
        intended_task="写文献综述",
        duration_min=90.0,
        actual_focus_min=45.0,
        context_switches_per_hour=7.0,
        longest_focus_block_s=500.0,
        social_media_ratio=0.15,
        start_delay_min=45.0,
        keyword_flags=frozenset({"redo_pattern"}),
    ),
    expected_types=(ProcrastinationType.DECISIONAL, ProcrastinationType.PERFECTIONISM),
    expected_technique=CBTTechnique.GOAL_SETTING,
))

# -- DEC-005: Delay but no focus recovery → task aversion -------------
_SCENARIOS.append(EvalScenario(
    scenario_id="DEC-005",
    description="低专注无恢复 — 不满足决策困难条件",
    bundle=_make_bundle(
        intended_task="做课程设计",
        duration_min=90.0,
        actual_focus_min=30.0,
        context_switches_per_hour=10.0,
        longest_focus_block_s=300.0,
        social_media_ratio=0.30,
        start_delay_min=45.0,
    ),
    expected_types=(ProcrastinationType.TASK_AVERSION,),
    expected_technique=CBTTechnique.GRADED_EXPOSURE,
))

# ===================================================================
# PERFECTIONISM (PER) — 5 scenarios
# ===================================================================

# -- PER-001: Classic perfectionism -----------------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="PER-001",
    description="典型完美主义 — 自我批评关键词",
    bundle=_make_bundle(
        intended_task="写作文",
        duration_min=60.0,
        actual_focus_min=15.0,
        context_switches_per_hour=8.0,
        longest_focus_block_s=200.0,
        social_media_ratio=0.20,
        start_delay_min=10.0,
        keyword_flags=frozenset({"self_criticism"}),
    ),
    expected_types=(ProcrastinationType.PERFECTIONISM,),
    expected_technique=CBTTechnique.COGNITIVE_RESTRUCTURING,
))

# -- PER-002: Both keywords -------------------------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="PER-002",
    description="双重完美主义关键词 — 更强信号",
    bundle=_make_bundle(
        intended_task="完成编程大作业",
        duration_min=60.0,
        actual_focus_min=10.0,
        context_switches_per_hour=6.0,
        longest_focus_block_s=100.0,
        social_media_ratio=0.10,
        start_delay_min=5.0,
        keyword_flags=frozenset({"self_criticism", "redo_pattern"}),
    ),
    expected_types=(ProcrastinationType.PERFECTIONISM,),
    expected_technique=CBTTechnique.COGNITIVE_RESTRUCTURING,
))

# -- PER-003: Perfectionism + impulsivity -----------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="PER-003",
    description="完美主义伴随冲动分心",
    bundle=_make_bundle(
        intended_task="准备竞赛",
        duration_min=60.0,
        actual_focus_min=10.0,
        context_switches_per_hour=14.0,
        longest_focus_block_s=200.0,
        social_media_ratio=0.20,
        start_delay_min=5.0,
        keyword_flags=frozenset({"redo_pattern"}),
    ),
    expected_types=(ProcrastinationType.PERFECTIONISM, ProcrastinationType.IMPULSIVITY),
    expected_technique=CBTTechnique.COGNITIVE_RESTRUCTURING,
))

# -- PER-004: Perfectionism low intensity -----------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="PER-004",
    description="轻度完美主义 — 单关键词但专注尚可",
    bundle=_make_bundle(
        intended_task="制作课堂展示",
        duration_min=120.0,
        actual_focus_min=50.0,
        context_switches_per_hour=8.0,
        longest_focus_block_s=400.0,
        social_media_ratio=0.20,
        start_delay_min=10.0,
        keyword_flags=frozenset({"self_criticism"}),
    ),
    expected_types=(ProcrastinationType.PERFECTIONISM,),
    expected_technique=CBTTechnique.COGNITIVE_RESTRUCTURING,
))

# -- PER-005: Perfectionism + emotional regulation --------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="PER-005",
    description="完美主义结合情绪调节 — 自评焦虑 + 高娱乐",
    bundle=_make_bundle(
        intended_task="写个人陈述",
        duration_min=90.0,
        actual_focus_min=20.0,
        context_switches_per_hour=10.0,
        longest_focus_block_s=200.0,
        social_media_ratio=0.60,
        start_delay_min=15.0,
        keyword_flags=frozenset({"self_criticism"}),
    ),
    expected_types=(ProcrastinationType.PERFECTIONISM, ProcrastinationType.EMOTIONAL_REGULATION),
    expected_technique=CBTTechnique.COGNITIVE_RESTRUCTURING,
))

# ===================================================================
# EMOTIONAL REGULATION (EMO) — 5 scenarios
# ===================================================================

# -- EMO-001: Classic emotional regulation ----------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="EMO-001",
    description="典型情绪调节 — 高娱乐占比",
    bundle=_make_bundle(
        intended_task="写周报",
        duration_min=60.0,
        actual_focus_min=15.0,
        context_switches_per_hour=10.0,
        longest_focus_block_s=200.0,
        social_media_ratio=0.70,
    ),
    expected_types=(ProcrastinationType.EMOTIONAL_REGULATION,),
    expected_technique=CBTTechnique.MINDFULNESS,
))

# -- EMO-002: Severe — very high social media -------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="EMO-002",
    description="严重情绪调节 — 极高娱乐占比",
    bundle=_make_bundle(
        intended_task="完成翻译作业",
        duration_min=60.0,
        actual_focus_min=5.0,
        context_switches_per_hour=15.0,
        longest_focus_block_s=60.0,
        social_media_ratio=0.90,
    ),
    expected_types=(ProcrastinationType.EMOTIONAL_REGULATION,),
    expected_technique=CBTTechnique.MINDFULNESS,
))

# -- EMO-003: Borderline -- just above social media threshold ---------
_SCENARIOS.append(EvalScenario(
    scenario_id="EMO-003",
    description="临界情绪调节 — 刚过娱乐阈值",
    bundle=_make_bundle(
        intended_task="写读书笔记",
        duration_min=60.0,
        actual_focus_min=20.0,
        context_switches_per_hour=10.0,
        longest_focus_block_s=300.0,
        social_media_ratio=0.58,
    ),
    expected_types=(ProcrastinationType.EMOTIONAL_REGULATION,),
    expected_technique=CBTTechnique.MINDFULNESS,
))

# -- EMO-004: Emotional regulation + impulsivity ----------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="EMO-004",
    description="情绪调节伴随冲动分心",
    bundle=_make_bundle(
        intended_task="做英语听力练习",
        duration_min=60.0,
        actual_focus_min=10.0,
        context_switches_per_hour=15.0,
        longest_focus_block_s=200.0,
        social_media_ratio=0.65,
    ),
    expected_types=(ProcrastinationType.EMOTIONAL_REGULATION, ProcrastinationType.IMPULSIVITY),
    expected_technique=CBTTechnique.MINDFULNESS,
))

# -- EMO-005: Emotional regulation low severity -----------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="EMO-005",
    description="轻度情绪调节 — 娱乐略高但专注尚可",
    bundle=_make_bundle(
        intended_task="整理笔记",
        duration_min=60.0,
        actual_focus_min=30.0,
        context_switches_per_hour=8.0,
        longest_focus_block_s=400.0,
        social_media_ratio=0.56,
    ),
    expected_types=(ProcrastinationType.EMOTIONAL_REGULATION,),
    expected_technique=CBTTechnique.MINDFULNESS,
))

# ===================================================================
# TASK AVERSION (TAV) — 5 scenarios
# ===================================================================

# -- TAV-001: Classic task aversion — low focus, no specific rule -----
_SCENARIOS.append(EvalScenario(
    scenario_id="TAV-001",
    description="典型任务畏惧 — 低专注无特定模式",
    bundle=_make_bundle(
        intended_task="做化学实验预习",
        duration_min=120.0,
        actual_focus_min=15.0,
        context_switches_per_hour=8.0,
        longest_focus_block_s=500.0,
        social_media_ratio=0.20,
    ),
    expected_types=(ProcrastinationType.TASK_AVERSION,),
    expected_technique=CBTTechnique.GRADED_EXPOSURE,
))

# -- TAV-002: Deviation-based task aversion ---------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="TAV-002",
    description="偏差驱动的任务畏惧 — 基线显著下降",
    bundle=_make_bundle(
        intended_task="数据分析练习",
        duration_min=120.0,
        actual_focus_min=40.0,
        context_switches_per_hour=8.0,
        longest_focus_block_s=600.0,
        social_media_ratio=0.20,
        baseline_deviation=-0.8,
    ),
    expected_types=(ProcrastinationType.TASK_AVERSION,),
    expected_technique=CBTTechnique.GRADED_EXPOSURE,
))

# -- TAV-003: Very low focus ------------------------------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="TAV-003",
    description="极低专注 — 几乎无法投入",
    bundle=_make_bundle(
        intended_task="背书",
        duration_min=120.0,
        actual_focus_min=5.0,
        context_switches_per_hour=6.0,
        longest_focus_block_s=400.0,
        social_media_ratio=0.15,
    ),
    expected_types=(ProcrastinationType.TASK_AVERSION,),
    expected_technique=CBTTechnique.GRADED_EXPOSURE,
))

# -- TAV-004: Unrelated keywords — not perfectionism ------------------
_SCENARIOS.append(EvalScenario(
    scenario_id="TAV-004",
    description="无关关键词 — 不触发完美主义",
    bundle=_make_bundle(
        intended_task="做物理题",
        duration_min=60.0,
        actual_focus_min=10.0,
        context_switches_per_hour=6.0,
        longest_focus_block_s=500.0,
        social_media_ratio=0.15,
        keyword_flags=frozenset({"boring", "hard"}),
    ),
    expected_types=(ProcrastinationType.TASK_AVERSION,),
    expected_technique=CBTTechnique.GRADED_EXPOSURE,
))

# -- TAV-005: Moderate focus dip — rule engine says no significant ----
_SCENARIOS.append(EvalScenario(
    scenario_id="TAV-005",
    description="专注轻度下降 — 规则引擎判定为无显著模式（人工标注为轻度任务畏惧）",
    bundle=_make_bundle(
        intended_task="整理代码注释",
        duration_min=60.0,
        actual_focus_min=25.0,
        context_switches_per_hour=8.0,
        longest_focus_block_s=500.0,
        social_media_ratio=0.20,
    ),
    expected_types=(ProcrastinationType.TASK_AVERSION,),
    expected_technique=CBTTechnique.GRADED_EXPOSURE,
))

# ===================================================================
# MIXED (MIX) — 5 ambiguous scenarios
# ===================================================================

# -- MIX-001: Low focus, borderline social — RE says TAV, human says EMO
_SCENARIOS.append(EvalScenario(
    scenario_id="MIX-001",
    description="模糊边界 — 低专注 + 中等娱乐占比：规则引擎判为任务畏惧，人工标为情绪调节",
    bundle=_make_bundle(
        intended_task="做PS设计作业",
        duration_min=120.0,
        actual_focus_min=20.0,
        context_switches_per_hour=10.0,
        longest_focus_block_s=350.0,
        social_media_ratio=0.50,
    ),
    expected_types=(ProcrastinationType.EMOTIONAL_REGULATION,),
    expected_technique=CBTTechnique.MINDFULNESS,
))

# -- MIX-002: Borderline impulsivity — RE says TAV, human says IMP ---
_SCENARIOS.append(EvalScenario(
    scenario_id="MIX-002",
    description="模糊冲动 — 低于切换阈值 + 短专注块：规则引擎判为任务畏惧，人工标为冲动",
    bundle=_make_bundle(
        intended_task="看教学视频",
        duration_min=60.0,
        actual_focus_min=20.0,
        context_switches_per_hour=11.0,
        longest_focus_block_s=280.0,
        social_media_ratio=0.30,
    ),
    expected_types=(ProcrastinationType.IMPULSIVITY,),
    expected_technique=CBTTechnique.STIMULUS_CONTROL,
))

# -- MIX-003: Healthy metrics — RE says no significant, human says TAV
_SCENARIOS.append(EvalScenario(
    scenario_id="MIX-003",
    description="指标正常 — 规则引擎判为无显著模式，人工标为轻度任务畏惧",
    bundle=_make_bundle(
        intended_task="背单词",
        duration_min=60.0,
        actual_focus_min=30.0,
        context_switches_per_hour=6.0,
        longest_focus_block_s=600.0,
        social_media_ratio=0.20,
    ),
    expected_types=(ProcrastinationType.TASK_AVERSION,),
    expected_technique=CBTTechnique.GRADED_EXPOSURE,
))

# -- MIX-004: Multiple moderate signals — RE orders differ from human
_SCENARIOS.append(EvalScenario(
    scenario_id="MIX-004",
    description="多重中等信号 — 规则引擎排序与人工判断不同",
    bundle=_make_bundle(
        intended_task="写课程总结",
        duration_min=60.0,
        actual_focus_min=30.0,
        context_switches_per_hour=12.0,
        longest_focus_block_s=200.0,
        social_media_ratio=0.45,
        start_delay_min=35.0,
        keyword_flags=frozenset({"self_criticism"}),
    ),
    expected_types=(
        ProcrastinationType.DECISIONAL,
        ProcrastinationType.PERFECTIONISM,
        ProcrastinationType.IMPULSIVITY,
    ),
    expected_technique=CBTTechnique.GOAL_SETTING,
))

# -- MIX-005: Impulsivity misclassified as emotional regulation -------
_SCENARIOS.append(EvalScenario(
    scenario_id="MIX-005",
    description="模糊 — 规则引擎可能判为情绪调节，人工标为冲动分心为主",
    bundle=_make_bundle(
        intended_task="做交互设计",
        duration_min=60.0,
        actual_focus_min=25.0,
        context_switches_per_hour=12.0,
        longest_focus_block_s=250.0,
        social_media_ratio=0.56,
    ),
    expected_types=(ProcrastinationType.IMPULSIVITY,),
    expected_technique=CBTTechnique.STIMULUS_CONTROL,
))

# ===================================================================
# Public API
# ===================================================================

ALL_SCENARIOS: tuple[EvalScenario, ...] = tuple(_SCENARIOS)


def get_scenario(scenario_id: str) -> EvalScenario | None:
    """Look up a scenario by ID."""
    for s in _SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None


def validate_all_scenarios() -> list[str]:
    """Validate all scenarios and return list of issues (empty = all valid)."""
    valid_types = set(ProcrastinationType)
    issues: list[str] = []
    seen_ids: set[str] = set()

    for s in _SCENARIOS:
        # Check duplicate IDs
        if s.scenario_id in seen_ids:
            issues.append(f"Duplicate scenario_id: {s.scenario_id}")
        seen_ids.add(s.scenario_id)

        # Check bundle has behavior_summary
        if s.bundle.behavior_summary is None:
            issues.append(f"{s.scenario_id}: bundle has no behavior_summary")

        # Check expected types are valid
        for t in s.expected_types:
            if t not in valid_types:
                issues.append(f"{s.scenario_id}: invalid type {t}")

        # Check expected types non-empty → technique not None, and vice versa
        if s.expected_types and s.expected_technique is None:
            issues.append(f"{s.scenario_id}: has expected types but no technique")
        if not s.expected_types and s.expected_technique is not None:
            issues.append(f"{s.scenario_id}: no expected types but has technique")

        # Check bundle items non-empty
        if not s.bundle.items:
            issues.append(f"{s.scenario_id}: bundle has no items")

        # Check at least items consistent with at least 3 metrics
        metrics = {item.metric for item in s.bundle.items}
        if len(metrics) < 3:
            issues.append(f"{s.scenario_id}: only {len(metrics)} unique metrics")

    return issues
