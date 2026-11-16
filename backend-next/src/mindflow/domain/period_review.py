"""Period review (阶段总结) — deterministic, privacy-preserving aggregation.

The user asks "what distracted me these days, what did I actually work on, how
often did it happen". This module answers that from two *numeric* sources only:

1. every 5-minute feature window the loaded behaviour classifier has re-scored
   (:class:`ScoredWindow` — a probability and a task-context category, nothing
   else), and
2. the enumeration fields of the intervention log (:class:`InterventionFact` —
   when, which procrastination types, which button the user pressed).

Nothing else enters the digest. There is no window title, no URL, no file path,
no application name, no reminder body and no chat text, because no field exists
to carry one: :class:`PeriodDigest` is a closed whitelist (``extra="forbid"``)
whose string fields are all enumerations or dates. That is what makes the same
object safe to hand to an LLM verbatim (see
``services/period_review_service.py``) and safe to show the user.

Pure stdlib + Pydantic: no I/O, no numpy, no framework. Layer rule: ``domain``
must not import ``services`` — the procrastination-type labels are therefore a
local copy of ``services/intervention_policy.PROCRASTINATION_LABELS_ZH``, kept
honest by a test that asserts the two mappings are identical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, tzinfo
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mindflow.domain.task_context import TASK_CONTEXT_CATEGORIES

# ── Constants (binding: docs/design/period-review-spec.md §2.1) ──────────────

#: Review spans the UI offers. Anything else is rejected by the service.
PERIOD_DAYS_ALLOWED: tuple[int, ...] = (3, 7, 14)

#: A window counts as distracted at this probability — the same cut the
#: prediction service uses (``focus_probas < 0.5`` ⇔ ``p_distraction >= 0.5``).
DISTRACTION_PROBA_THRESHOLD = 0.5

#: Two distracted windows belong to one episode when the gap between them is at
#: most this many seconds (one missed 5-minute window is a break, two is not).
EPISODE_MERGE_GAP_S = 60.0

#: Fewer valid windows than this (≈ one hour) cannot support a summary.
MIN_ACTIVE_WINDOWS = 12

#: Upper bound on the "what does the model look at" signal list.
MAX_MODEL_SIGNALS = 3

CONTEXT_LABELS_ZH: Mapping[str, str] = {
    "coding": "编程",
    "writing": "写作与文档",
    "reading": "查资料与阅读",
    "meeting": "沟通与会议",
    "entertainment": "娱乐",
    "social": "社交",
    "unknown": "未识别",
}

#: (code, 中文, 起始小时含, 结束小时不含) — fixed order, covers 0–24.
DAYPARTS: tuple[tuple[str, str, int, int], ...] = (
    ("morning", "上午", 6, 12),
    ("afternoon", "下午", 12, 18),
    ("evening", "晚上", 18, 24),
    ("night", "深夜", 0, 6),
)

#: The response vocabulary of ``intervention_logs.user_response`` plus "pending"
#: for a log the user never answered (or an unrecognised legacy value).
RESPONSE_KEYS: tuple[str, ...] = ("accepted", "ignored", "dismissed", "pending")

#: Cyclic time encodings and the categorical task-type code are context, not
#: behaviour: a "signal" built from them would tell the user what hour it was.
NON_BEHAVIORAL_FEATURES: frozenset[str] = frozenset(
    {"hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "task_type_code"}
)

#: Local copy of ``services.intervention_policy.PROCRASTINATION_LABELS_ZH``
#: (``domain`` may not import ``services``). ``tests/test_period_review.py``
#: asserts the two are equal, so a label change cannot silently drift.
REASON_LABELS_ZH: Mapping[str, str] = {
    "task_aversion": "任务厌恶",
    "impulsivity": "冲动分心",
    "decisional": "决策拖延",
    "perfectionism": "完美主义",
    "emotional_regulation": "情绪调节困难",
}

FallbackReason = Literal[
    "AI 未配置",
    "AI 响应超时",
    "AI 调用失败",
    "AI 输出未通过校验",
]

# ══════════════════════════════════════════════════════════════════════════════
# Inputs
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScoredWindow:
    """One 5-minute feature window as re-scored by the loaded classifier.

    Attributes:
        start_utc: Window start (timezone-aware UTC).
        end_utc: Window end (timezone-aware UTC).
        task_context: One of ``TASK_CONTEXT_CATEGORIES``.
        p_distraction: The model's distraction probability (output column 0).
    """

    start_utc: datetime
    end_utc: datetime
    task_context: str
    p_distraction: float


@dataclass(frozen=True)
class InterventionFact:
    """The three intervention-log fields this feature is allowed to read.

    Attributes:
        triggered_at_utc: When the reminder fired.
        reasons: ``context_json.procrastination_types``, filtered to keys of
            :data:`REASON_LABELS_ZH`.
        response: One of :data:`RESPONSE_KEYS`; unknown/None reads as "pending".
    """

    triggered_at_utc: datetime
    reasons: tuple[str, ...]
    response: str


@dataclass(frozen=True)
class DistractionDriver:
    """One behavioural feature that separates distracted from focused windows.

    Attributes:
        feature: Column name from the V2 feature vocabulary.
        label: Chinese label (``domain.feature_labels.feature_label``).
        direction: Whether the feature was higher, lower or similar while
            distracted.
        distracted_mean: Mean over distracted windows (3 decimals).
        focused_mean: Mean over focused windows (3 decimals).
    """

    feature: str
    label: str
    direction: Literal["higher", "lower", "similar"]
    distracted_mean: float
    focused_mean: float


# ══════════════════════════════════════════════════════════════════════════════
# Output models — closed whitelist, frozen, no unknown keys
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class PeriodRange(BaseModel):
    """The reviewed local-date span."""

    model_config = _MODEL_CONFIG

    start_date: date
    end_date: date
    days: int


class ModelInfo(BaseModel):
    """Which classifier produced the numbers (``None`` when none is loaded)."""

    model_config = _MODEL_CONFIG

    version: str | None
    is_base_model: bool


class PeriodTotals(BaseModel):
    """Minute totals over the whole span."""

    model_config = _MODEL_CONFIG

    active_minutes: int
    focused_minutes: int
    distracted_minutes: int
    distracted_share: float
    days_with_data: int


class EpisodeSummary(BaseModel):
    """How often, how long on average, and how long at worst."""

    model_config = _MODEL_CONFIG

    count: int
    avg_minutes: float
    longest_minutes: int


class ContextMinutes(BaseModel):
    """Minutes attributed to one task-context category."""

    model_config = _MODEL_CONFIG

    context: str
    label: str
    minutes: int
    share: float


class DaypartStat(BaseModel):
    """Distraction in one part of the day."""

    model_config = _MODEL_CONFIG

    daypart: str
    label: str
    episodes: int
    distracted_minutes: int


class DailyStat(BaseModel):
    """One local calendar day; always present, zeros when nothing was recorded."""

    model_config = _MODEL_CONFIG

    date: date
    active_minutes: int
    distracted_minutes: int
    episodes: int
    top_context: str
    top_context_label: str


class ModelSignal(BaseModel):
    """One behavioural signal the model used to separate distraction."""

    model_config = _MODEL_CONFIG

    feature: str
    label: str
    direction: Literal["higher", "lower", "similar"]
    distracted_mean: float
    focused_mean: float


class ReasonCount(BaseModel):
    """How often one procrastination type was the stated reason."""

    model_config = _MODEL_CONFIG

    type: str
    label: str
    count: int


class InterventionSummary(BaseModel):
    """Reminder counts by response and by reason."""

    model_config = _MODEL_CONFIG

    total: int
    accepted: int
    ignored: int
    dismissed: int
    pending: int
    by_reason: list[ReasonCount] = Field(default_factory=list)


class PeriodDigest(BaseModel):
    """The complete, privacy-safe fact sheet for one review span."""

    model_config = _MODEL_CONFIG

    period: PeriodRange
    data_state: Literal["ready", "insufficient", "no_model"]
    model: ModelInfo
    totals: PeriodTotals
    episodes: EpisodeSummary
    work_mix: list[ContextMinutes] = Field(default_factory=list)
    distraction_by_context: list[ContextMinutes] = Field(default_factory=list)
    distraction_by_daypart: list[DaypartStat] = Field(default_factory=list)
    daily: list[DailyStat] = Field(default_factory=list)
    model_signals: list[ModelSignal] = Field(default_factory=list)
    interventions: InterventionSummary


class EvidenceRef(BaseModel):
    """A digest fact a narrative sentence is allowed to cite."""

    model_config = _MODEL_CONFIG

    id: str
    label: str


class NarrativeFinding(BaseModel):
    """One sentence of the summary plus the digest facts behind it."""

    model_config = _MODEL_CONFIG

    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class NarrativeSuggestion(BaseModel):
    """One next step, optionally grounded in retrieved literature."""

    model_config = _MODEL_CONFIG

    text: str
    paper_ids: list[str] = Field(default_factory=list)


class NarrativePaper(BaseModel):
    """A literature passage the summary actually cited."""

    model_config = _MODEL_CONFIG

    chunk_id: str
    title: str
    pages: list[int] = Field(default_factory=list)


class Narrative(BaseModel):
    """The written summary — AI-generated or from the local template."""

    model_config = _MODEL_CONFIG

    source: Literal["ai", "template"]
    headline: str
    findings: list[NarrativeFinding] = Field(default_factory=list)
    suggestions: list[NarrativeSuggestion] = Field(default_factory=list)
    papers: list[NarrativePaper] = Field(default_factory=list)
    fallback_reason: FallbackReason | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _Episode:
    start_utc: datetime
    minutes: float


def _window_minutes(window: ScoredWindow) -> float:
    """Minutes a window covers, clipped to ``[0, 5]``.

    A window longer than five minutes is a data artefact (the collector rolled
    several windows into one), and one with a zero/negative span carries no
    observable time; neither may inflate the totals.
    """
    return min(max((window.end_utc - window.start_utc).total_seconds() / 60.0, 0.0), 5.0)


def _is_distracted(window: ScoredWindow) -> bool:
    return window.p_distraction >= DISTRACTION_PROBA_THRESHOLD


def _episodes(windows: Sequence[ScoredWindow]) -> list[_Episode]:
    """Merge distracted windows into episodes, in chronological order.

    Two consecutive distracted windows belong to the same episode when the
    second starts at most :data:`EPISODE_MERGE_GAP_S` after the first one ends.
    A focused window in between leaves a ≥ 5-minute gap, so it always breaks
    the episode.
    """
    distracted = sorted(
        (window for window in windows if _is_distracted(window)),
        key=lambda window: window.start_utc,
    )
    episodes: list[_Episode] = []
    previous_end: datetime | None = None
    for window in distracted:
        merges = (
            bool(episodes)
            and previous_end is not None
            and (window.start_utc - previous_end).total_seconds() <= EPISODE_MERGE_GAP_S
        )
        if merges:
            last = episodes[-1]
            episodes[-1] = _Episode(
                start_utc=last.start_utc,
                minutes=last.minutes + _window_minutes(window),
            )
        else:
            episodes.append(
                _Episode(start_utc=window.start_utc, minutes=_window_minutes(window))
            )
        previous_end = window.end_utc
    return episodes


def _daypart_code(local: datetime) -> str:
    """Daypart code for a local timestamp (``DAYPARTS`` covers 0–24)."""
    for code, _label, start_hour, end_hour in DAYPARTS:
        if start_hour <= local.hour < end_hour:
            return code
    return DAYPARTS[0][0]  # pragma: no cover - DAYPARTS is exhaustive


def _daypart_label(code: str) -> str:
    for candidate, label, _start, _end in DAYPARTS:
        if candidate == code:
            return label
    return DAYPARTS[0][1]  # pragma: no cover - callers pass a known code


def _context_label(context: str) -> str:
    return CONTEXT_LABELS_ZH.get(context, CONTEXT_LABELS_ZH["unknown"])


def _share(part: float, whole: float) -> float:
    """Share rounded to 3 decimals; ``0.0`` when the denominator is zero."""
    if whole <= 0:
        return 0.0
    return round(part / whole, 3)


def _day_list(start_date: date, end_date: date) -> list[date]:
    span = (end_date - start_date).days
    if span < 0:
        return []
    return [start_date + timedelta(days=offset) for offset in range(span + 1)]


def build_period_digest(
    windows: Sequence[ScoredWindow],
    interventions: Sequence[InterventionFact],
    drivers: Sequence[DistractionDriver],
    *,
    start_date: date,
    end_date: date,
    tz: tzinfo,
    model_version: str | None,
    is_base_model: bool,
    model_available: bool,
) -> PeriodDigest:
    """Aggregate scored windows and intervention facts into a digest.

    Args:
        windows: Re-scored 5-minute windows (any order; sorted internally).
        interventions: Reminder facts restricted to the same span.
        drivers: Model signals already ranked and capped by the caller.
        start_date: First local calendar day of the span.
        end_date: Last local calendar day of the span.
        tz: Timezone the day/daypart buckets are resolved in.
        model_version: Version tag of the classifier, or ``None``.
        is_base_model: Whether the version is the bundled base bundle.
        model_available: ``False`` when no model is loaded; forces
            ``data_state="no_model"`` and empty aggregates.

    Returns:
        A frozen :class:`PeriodDigest`. ``daily`` always lists every day of the
        span, zeros included.
    """
    if not model_available:
        data_state: Literal["ready", "insufficient", "no_model"] = "no_model"
    elif len(windows) < MIN_ACTIVE_WINDOWS:
        data_state = "insufficient"
    else:
        data_state = "ready"

    days = _day_list(start_date, end_date)
    episodes = _episodes(windows)

    # ── Totals ──────────────────────────────────────────────────────────────
    active_minutes = math.fsum(_window_minutes(window) for window in windows)
    distracted_windows = [window for window in windows if _is_distracted(window)]
    distracted_minutes = math.fsum(_window_minutes(window) for window in distracted_windows)
    focused_minutes = max(0.0, active_minutes - distracted_minutes)

    # ── Per-day and per-context buckets ─────────────────────────────────────
    day_active: dict[date, float] = dict.fromkeys(days, 0.0)
    day_distracted: dict[date, float] = dict.fromkeys(days, 0.0)
    day_context: dict[date, dict[str, float]] = {
        day: dict.fromkeys(TASK_CONTEXT_CATEGORIES, 0.0) for day in days
    }
    context_minutes: dict[str, float] = dict.fromkeys(TASK_CONTEXT_CATEGORIES, 0.0)
    distracted_context: dict[str, float] = dict.fromkeys(TASK_CONTEXT_CATEGORIES, 0.0)
    daypart_distracted: dict[str, float] = {
        code: 0.0 for code, _label, _start, _end in DAYPARTS
    }
    day_episodes: dict[date, int] = dict.fromkeys(days, 0)
    daypart_episodes: dict[str, int] = {
        code: 0 for code, _label, _start, _end in DAYPARTS
    }

    for window in windows:
        minutes = _window_minutes(window)
        context = (
            window.task_context
            if window.task_context in TASK_CONTEXT_CATEGORIES
            else "unknown"
        )
        local = window.start_utc.astimezone(tz)
        day = local.date()
        # A window that starts on the previous local day is outside the span's
        # calendar buckets; its minutes still count in the totals above.
        in_span = day in day_active
        if in_span:
            day_active[day] += minutes
            day_context[day][context] += minutes
            daypart_distracted[_daypart_code(local)] += (
                minutes if _is_distracted(window) else 0.0
            )
        context_minutes[context] += minutes
        if _is_distracted(window):
            distracted_context[context] += minutes
            if in_span:
                day_distracted[day] += minutes

    for episode in episodes:
        local = episode.start_utc.astimezone(tz)
        day = local.date()
        if day in day_episodes:
            day_episodes[day] += 1
        daypart_episodes[_daypart_code(local)] += 1

    # ── work_mix / distraction_by_context ───────────────────────────────────
    order = {context: index for index, context in enumerate(TASK_CONTEXT_CATEGORIES)}
    work_mix = [
        ContextMinutes(
            context=context,
            label=_context_label(context),
            minutes=round(minutes),
            share=_share(minutes, active_minutes),
        )
        for context, minutes in context_minutes.items()
        if minutes > 0
    ]
    work_mix.sort(key=lambda item: (-item.minutes, order[item.context]))

    distraction_by_context = [
        ContextMinutes(
            context=context,
            label=_context_label(context),
            minutes=round(minutes),
            share=_share(minutes, distracted_minutes),
        )
        for context, minutes in distracted_context.items()
        if minutes > 0
    ]
    distraction_by_context.sort(key=lambda item: (-item.minutes, order[item.context]))

    # ── daily ───────────────────────────────────────────────────────────────
    daily: list[DailyStat] = []
    for day in days:
        contexts = day_context[day]
        observed = math.fsum(contexts.values())
        top_context = (
            max(TASK_CONTEXT_CATEGORIES, key=lambda name: contexts[name])
            if observed > 0
            else "unknown"
        )
        daily.append(
            DailyStat(
                date=day,
                active_minutes=round(day_active[day]),
                distracted_minutes=round(day_distracted[day]),
                episodes=day_episodes[day],
                top_context=top_context,
                top_context_label=_context_label(top_context),
            )
        )

    # ── dayparts (all four, fixed order) ────────────────────────────────────
    distraction_by_daypart = [
        DaypartStat(
            daypart=code,
            label=label,
            episodes=daypart_episodes[code],
            distracted_minutes=round(daypart_distracted[code]),
        )
        for code, label, _start, _end in DAYPARTS
    ]

    # ── model signals ───────────────────────────────────────────────────────
    model_signals = [
        ModelSignal(
            feature=driver.feature,
            label=driver.label,
            direction=driver.direction,
            distracted_mean=driver.distracted_mean,
            focused_mean=driver.focused_mean,
        )
        for driver in drivers[:MAX_MODEL_SIGNALS]
    ]

    # ── interventions ───────────────────────────────────────────────────────
    response_counts = dict.fromkeys(RESPONSE_KEYS, 0)
    reason_counts: dict[str, int] = {}
    for fact in interventions:
        response = fact.response if fact.response in response_counts else "pending"
        response_counts[response] += 1
        for reason in fact.reasons:
            if reason in REASON_LABELS_ZH:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    reason_order = {
        reason: index for index, reason in enumerate(REASON_LABELS_ZH)
    }
    by_reason = [
        ReasonCount(type=reason, label=REASON_LABELS_ZH[reason], count=count)
        for reason, count in reason_counts.items()
    ]
    by_reason.sort(key=lambda item: (-item.count, reason_order[item.type]))

    return PeriodDigest(
        period=PeriodRange(
            start_date=start_date,
            end_date=end_date,
            days=len(days),
        ),
        data_state=data_state,
        model=ModelInfo(version=model_version, is_base_model=is_base_model),
        totals=PeriodTotals(
            active_minutes=round(active_minutes),
            focused_minutes=round(focused_minutes),
            distracted_minutes=round(distracted_minutes),
            distracted_share=_share(distracted_minutes, active_minutes),
            days_with_data=sum(1 for minutes in day_active.values() if minutes > 0),
        ),
        episodes=EpisodeSummary(
            count=len(episodes),
            avg_minutes=(
                round(math.fsum(episode.minutes for episode in episodes) / len(episodes), 1)
                if episodes
                else 0.0
            ),
            longest_minutes=round(max((episode.minutes for episode in episodes), default=0.0)),
        ),
        work_mix=work_mix,
        distraction_by_context=distraction_by_context,
        distraction_by_daypart=distraction_by_daypart,
        daily=daily,
        model_signals=model_signals,
        interventions=InterventionSummary(
            total=len(interventions),
            accepted=response_counts["accepted"],
            ignored=response_counts["ignored"],
            dismissed=response_counts["dismissed"],
            pending=response_counts["pending"],
            by_reason=by_reason,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Evidence catalog
# ══════════════════════════════════════════════════════════════════════════════


def _fmt_number(value: float) -> str:
    """Compact decimal: ``9.2``, ``3``, ``0.125`` — never ``9.200``."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def _fmt_percent(share: float) -> str:
    """Percentage with at most one decimal: ``17.5%``, ``48%``."""
    text = f"{share * 100:.1f}".rstrip("0").rstrip(".")
    return f"{text or '0'}%"


def _fmt_day(day: date) -> str:
    return f"{day.month}月{day.day}日"


_DIRECTION_ZH: Mapping[str, str] = {
    "higher": "偏高",
    "lower": "偏低",
    "similar": "差别不大",
}


def evidence_catalog(digest: PeriodDigest) -> Mapping[str, str]:
    """Return ``ID → "中文标签：值"`` for every fact a narrative may cite.

    IDs use only ``[A-Za-z0-9_.]`` so they satisfy the citation pattern in
    ``agents/orchestrator.validate_citations``. The catalog is also the
    allowlist the AI output is checked against, so a model cannot cite a number
    that is not in the digest.
    """
    catalog: dict[str, str] = {
        "totals.active_minutes": f"有效记录时长：{digest.totals.active_minutes} 分钟",
        "totals.focused_minutes": f"专注时长：{digest.totals.focused_minutes} 分钟",
        "totals.distracted_minutes": f"分心时长：{digest.totals.distracted_minutes} 分钟",
        "totals.distracted_share": (
            f"分心时长占比：{_fmt_percent(digest.totals.distracted_share)}"
        ),
        "episodes.count": f"分心次数：{digest.episodes.count} 次",
        "episodes.avg_minutes": f"平均每次分心：{_fmt_number(digest.episodes.avg_minutes)} 分钟",
        "episodes.longest_minutes": f"最长一次分心：{digest.episodes.longest_minutes} 分钟",
        "interventions.total": f"收到提醒：{digest.interventions.total} 次",
        "interventions.accepted": f"点了“好，试试看”：{digest.interventions.accepted} 次",
        "interventions.ignored": (
            f"点了“先忽略”或未处理：{digest.interventions.ignored} 次"
        ),
        "interventions.dismissed": f"点了“不适合我”：{digest.interventions.dismissed} 次",
        "model.version": (
            "当前模型：基础模型（合成数据训练）"
            if digest.model.is_base_model
            else "当前模型：个人模型"
        ),
    }
    for item in digest.work_mix:
        catalog[f"work_mix.{item.context}"] = (
            f"{item.label}：{item.minutes} 分钟（{_fmt_percent(item.share)}）"
        )
    for item in digest.distraction_by_context:
        catalog[f"distraction_by_context.{item.context}"] = (
            f"在{item.label}上分心：{item.minutes} 分钟"
        )
    for stat in digest.distraction_by_daypart:
        if stat.episodes > 0 or stat.distracted_minutes > 0:
            catalog[f"daypart.{stat.daypart}"] = (
                f"{stat.label}：分心 {stat.episodes} 次，共 {stat.distracted_minutes} 分钟"
            )
    for day in digest.daily:
        if day.active_minutes > 0:
            catalog[f"daily.{day.date.strftime('%Y%m%d')}"] = (
                f"{_fmt_day(day.date)}：记录 {day.active_minutes} 分钟，"
                f"分心 {day.distracted_minutes} 分钟，{day.episodes} 次"
            )
    for signal in digest.model_signals:
        catalog[f"signal.{signal.feature}"] = (
            f"{signal.label}：分心时{_DIRECTION_ZH[signal.direction]}"
            f"（{_fmt_number(signal.distracted_mean)} 对 "
            f"{_fmt_number(signal.focused_mean)}）"
        )
    for reason in digest.interventions.by_reason:
        catalog[f"reason.{reason.type}"] = (
            f"提醒原因·{reason.label}：{reason.count} 次"
        )
    return catalog


# ══════════════════════════════════════════════════════════════════════════════
# Local template narrative
# ══════════════════════════════════════════════════════════════════════════════


def _hours(minutes: int) -> str:
    return _fmt_number(round(minutes / 60.0, 1))


def _peak_daypart(digest: PeriodDigest) -> DaypartStat | None:
    """The daypart with the most episodes; ties go to ``DAYPARTS`` order."""
    best: DaypartStat | None = None
    for stat in digest.distraction_by_daypart:
        if best is None or stat.episodes > best.episodes:
            best = stat
    if best is None or best.episodes <= 0:
        return None
    return best


def template_narrative(digest: PeriodDigest) -> Narrative:
    """Write the summary from fixed rules — no model, no network.

    Used when AI is off, when there is nothing to summarise, and whenever the
    AI path fails validation. The wording is fixed by the binding spec so the
    fallback reads the same every time.
    """
    if digest.data_state == "no_model":
        return Narrative(
            source="template",
            headline="还没有可用的行为分类模型",
            findings=[
                NarrativeFinding(text="模型加载后才能按 5 分钟窗口判断专注和分心。")
            ],
            suggestions=[
                NarrativeSuggestion(text="到“模型中心”确认基础模型已启用。")
            ],
        )

    if digest.data_state == "insufficient":
        return Narrative(
            source="template",
            headline="记录还不够，暂时总结不出规律",
            findings=[
                NarrativeFinding(
                    text=(
                        f"这段时间只有 {digest.totals.active_minutes} 分钟有效记录，"
                        "少于 1 小时。"
                    ),
                    evidence=[
                        EvidenceRef(
                            id="totals.active_minutes",
                            label=f"有效记录时长：{digest.totals.active_minutes} 分钟",
                        )
                    ],
                )
            ],
            suggestions=[
                NarrativeSuggestion(text="让 MindFlow 在后台多运行几天，再来看看。")
            ],
        )

    days = digest.period.days
    findings: list[NarrativeFinding] = []
    suggestions: list[NarrativeSuggestion] = []
    peak = _peak_daypart(digest)

    if digest.episodes.count > 0 and digest.distraction_by_context:
        top_label = digest.distraction_by_context[0].label
        headline = f"最近{days}天分心 {digest.episodes.count} 次，多发生在{top_label}上"
    else:
        headline = f"最近{days}天没有识别到明显的分心"

    findings.append(
        NarrativeFinding(
            text=(
                f"有效记录 {_hours(digest.totals.active_minutes)} 小时，"
                f"模型判断专注 {_hours(digest.totals.focused_minutes)} 小时、"
                f"分心 {_hours(digest.totals.distracted_minutes)} 小时，"
                f"分心约占 {round(digest.totals.distracted_share * 100)}%。"
            ),
            evidence=[
                EvidenceRef(
                    id="totals.active_minutes",
                    label=f"有效记录时长：{digest.totals.active_minutes} 分钟",
                ),
                EvidenceRef(
                    id="totals.focused_minutes",
                    label=f"专注时长：{digest.totals.focused_minutes} 分钟",
                ),
                EvidenceRef(
                    id="totals.distracted_minutes",
                    label=f"分心时长：{digest.totals.distracted_minutes} 分钟",
                ),
                EvidenceRef(
                    id="totals.distracted_share",
                    label=f"分心时长占比：{_fmt_percent(digest.totals.distracted_share)}",
                ),
            ],
        )
    )

    if digest.work_mix:
        first = digest.work_mix[0]
        text = (
            f"时间主要花在{first.label}（{_fmt_percent(first.share)}）"
        )
        evidence = [
            EvidenceRef(
                id=f"work_mix.{first.context}",
                label=(
                    f"{first.label}：{first.minutes} 分钟"
                    f"（{_fmt_percent(first.share)}）"
                ),
            )
        ]
        if len(digest.work_mix) > 1:
            second = digest.work_mix[1]
            text += f"和{second.label}（{_fmt_percent(second.share)}）"
            evidence.append(
                EvidenceRef(
                    id=f"work_mix.{second.context}",
                    label=(
                        f"{second.label}：{second.minutes} 分钟"
                        f"（{_fmt_percent(second.share)}）"
                    ),
                )
            )
        findings.append(
            NarrativeFinding(text=text + "。", evidence=evidence)
        )

    if digest.episodes.count > 0 and peak is not None:
        findings.append(
            NarrativeFinding(
                text=(
                    f"分心最常出现在{peak.label}，共 {peak.episodes} 次；"
                    f"最长一次约 {digest.episodes.longest_minutes} 分钟。"
                ),
                evidence=[
                    EvidenceRef(
                        id=f"daypart.{peak.daypart}",
                        label=(
                            f"{peak.label}：分心 {peak.episodes} 次，"
                            f"共 {peak.distracted_minutes} 分钟"
                        ),
                    ),
                    EvidenceRef(
                        id="episodes.longest_minutes",
                        label=(
                            f"最长一次分心：{digest.episodes.longest_minutes} 分钟"
                        ),
                    ),
                ],
            )
        )

    if digest.model_signals:
        parts = [
            f"{signal.label}{_DIRECTION_ZH[signal.direction]}"
            for signal in digest.model_signals[:2]
        ]
        findings.append(
            NarrativeFinding(
                text="模型判断分心时，最明显的信号是" + "、".join(parts) + "。",
                evidence=[
                    EvidenceRef(
                        id=f"signal.{signal.feature}",
                        label=(
                            f"{signal.label}：分心时{_DIRECTION_ZH[signal.direction]}"
                            f"（{_fmt_number(signal.distracted_mean)} 对 "
                            f"{_fmt_number(signal.focused_mean)}）"
                        ),
                    )
                    for signal in digest.model_signals[:2]
                ],
            )
        )

    if digest.model.is_base_model:
        findings.append(
            NarrativeFinding(
                text=(
                    "当前用的是基础模型，判断可能有偏差，"
                    "可以在提醒里点“不适合我”纠正。"
                )
            )
        )

    if digest.episodes.count > 0 and digest.distraction_by_context:
        top = digest.distraction_by_context[0]
        if top.context in {"entertainment", "social"}:
            label = peak.label if peak is not None else "分心前"
            suggestions.append(
                NarrativeSuggestion(
                    text=(
                        f"{label}开始前先写下要做的事，"
                        f"把{top.label}类应用留到任务结束后再打开。"
                    )
                )
            )
        elif peak is not None:
            suggestions.append(
                NarrativeSuggestion(
                    text=(
                        f"{peak.label}的任务可以拆小一点，"
                        "每段先定一个能完成的小目标。"
                    )
                )
            )

    if digest.interventions.dismissed > 0:
        suggestions.append(
            NarrativeSuggestion(
                text=(
                    f"有 {digest.interventions.dismissed} 次提醒你选了“不适合我”，"
                    "系统会根据这些反馈调整之后的提醒。"
                )
            )
        )

    if not suggestions:
        suggestions.append(
            NarrativeSuggestion(text="保持现在的节奏，有需要时再来回顾。")
        )

    return Narrative(
        source="template",
        headline=headline,
        findings=findings,
        suggestions=suggestions[:2],
    )


__all__ = [
    "CONTEXT_LABELS_ZH",
    "DAYPARTS",
    "DISTRACTION_PROBA_THRESHOLD",
    "EPISODE_MERGE_GAP_S",
    "FallbackReason",
    "InterventionFact",
    "InterventionSummary",
    "MAX_MODEL_SIGNALS",
    "MIN_ACTIVE_WINDOWS",
    "ModelSignal",
    "NON_BEHAVIORAL_FEATURES",
    "Narrative",
    "NarrativeFinding",
    "NarrativePaper",
    "NarrativeSuggestion",
    "PERIOD_DAYS_ALLOWED",
    "PeriodDigest",
    "REASON_LABELS_ZH",
    "RESPONSE_KEYS",
    "ScoredWindow",
    "DistractionDriver",
    "EvidenceRef",
    "build_period_digest",
    "evidence_catalog",
    "template_narrative",
]
