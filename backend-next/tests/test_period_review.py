"""Pure-function tests for the period review domain (阶段总结).

Covers the binding spec's §6 list for ``domain/period_review.py``: episode
merging, daypart attribution across midnight, work-mix ordering and shares,
the zero-filled daily series, the three ``data_state`` values, evidence-catalog
IDs and wording, and the three template headlines.

Everything here is deterministic — fixed dates, fixed timezone, no clock.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from mindflow.domain.period_review import (
    CONTEXT_LABELS_ZH,
    DAYPARTS,
    EPISODE_MERGE_GAP_S,
    MIN_ACTIVE_WINDOWS,
    NON_BEHAVIORAL_FEATURES,
    REASON_LABELS_ZH,
    InterventionFact,
    PeriodDigest,
    ScoredWindow,
    build_period_digest,
    evidence_catalog,
    template_narrative,
)
from mindflow.services.intervention_policy import PROCRASTINATION_LABELS_ZH

SHANGHAI = ZoneInfo("Asia/Shanghai")
#: Evidence IDs must satisfy the citation regex used by the panel validator.
EVIDENCE_ID = re.compile(r"^[A-Za-z0-9_.]+$")


def _window(
    start: datetime,
    minutes: float = 5.0,
    *,
    p: float = 0.1,
    context: str = "coding",
) -> ScoredWindow:
    return ScoredWindow(
        start_utc=start,
        end_utc=start + timedelta(minutes=minutes),
        task_context=context,
        p_distraction=p,
    )


def _local(day: date, hour: int, minute: int = 0) -> datetime:
    """A local (Asia/Shanghai) wall-clock time, returned as aware UTC."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=SHANGHAI).astimezone(
        UTC
    )


def _digest(
    windows: list[ScoredWindow],
    *,
    start_date: date,
    end_date: date,
    interventions: list[InterventionFact] | None = None,
    model_available: bool = True,
    model_version: str | None = "base_v4_20261111_r2",
    is_base_model: bool = True,
) -> PeriodDigest:
    return build_period_digest(
        windows,
        interventions or [],
        (),
        start_date=start_date,
        end_date=end_date,
        tz=SHANGHAI,
        model_version=model_version,
        is_base_model=is_base_model,
        model_available=model_available,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Label parity (domain copy must equal the services mapping)
# ══════════════════════════════════════════════════════════════════════════════


def test_reason_labels_match_intervention_policy() -> None:
    """``domain`` may not import ``services``, so the copy must not drift."""
    assert dict(REASON_LABELS_ZH) == dict(PROCRASTINATION_LABELS_ZH)


# ══════════════════════════════════════════════════════════════════════════════
# Episode merging
# ══════════════════════════════════════════════════════════════════════════════


def test_adjacent_distracted_windows_merge_into_one_episode() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base, p=0.9),
        _window(base + timedelta(minutes=5), p=0.8),
        _window(base + timedelta(minutes=10), p=0.7),
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert digest.episodes.count == 1
    assert digest.episodes.longest_minutes == 15
    assert digest.episodes.avg_minutes == 15.0


def test_gap_over_merge_threshold_starts_a_new_episode() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    first = _window(base, p=0.9)
    # Ends at base+5min; the next starts 5min+2s later — one focused window
    # would sit in that gap, and 2s over the 60s threshold must break it.
    second = _window(
        base + timedelta(minutes=5, seconds=EPISODE_MERGE_GAP_S + 2), p=0.9
    )
    digest = _digest([first, second], start_date=day, end_date=day)
    assert digest.episodes.count == 2


def test_gap_exactly_at_threshold_still_merges() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base, p=0.9),
        _window(base + timedelta(minutes=5, seconds=EPISODE_MERGE_GAP_S), p=0.9),
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert digest.episodes.count == 1


def test_focused_window_between_breaks_the_episode() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base, p=0.9),
        _window(base + timedelta(minutes=5), p=0.1),
        _window(base + timedelta(minutes=10), p=0.9),
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert digest.episodes.count == 2


def test_episodes_are_ordered_even_when_windows_arrive_unsorted() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base + timedelta(minutes=10), p=0.9),
        _window(base, p=0.9),
        _window(base + timedelta(minutes=5), p=0.9),
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert digest.episodes.count == 1


# ══════════════════════════════════════════════════════════════════════════════
# Daypart attribution
# ══════════════════════════════════════════════════════════════════════════════


def test_window_crossing_midnight_is_attributed_by_its_start() -> None:
    """A 23:58→00:03 window belongs to the evening of the day it started."""
    day = date(2026, 11, 10)
    start = _local(day, 23, 58)
    digest = _digest([_window(start, p=0.9)], start_date=day, end_date=day)
    evening = next(
        stat for stat in digest.distraction_by_daypart if stat.daypart == "evening"
    )
    night = next(stat for stat in digest.distraction_by_daypart if stat.daypart == "night")
    assert evening.episodes == 1
    assert evening.distracted_minutes == 5
    assert night.episodes == 0
    # Both the day bucket and the episode land on the start date.
    assert digest.daily[0].date == day
    assert digest.daily[0].episodes == 1
    assert digest.daily[0].distracted_minutes == 5


def test_all_four_dayparts_are_always_listed_in_fixed_order() -> None:
    day = date(2026, 11, 10)
    digest = _digest([_window(_local(day, 9), p=0.1)], start_date=day, end_date=day)
    assert [stat.daypart for stat in digest.distraction_by_daypart] == [
        code for code, _label, _start, _end in DAYPARTS
    ]
    assert [stat.label for stat in digest.distraction_by_daypart] == [
        label for _code, label, _start, _end in DAYPARTS
    ]


def test_daypart_boundaries_are_half_open() -> None:
    """11:55 → 上午, 12:00 → 下午, 18:00 → 晚上, 05:59 → 深夜.

    The four starts are far enough apart (≥ 2h) that each is its own episode,
    so the episode count per daypart is unambiguous.
    """
    day = date(2026, 11, 10)
    digest = _digest(
        [
            _window(_local(day, 11, 55), p=0.9),
            _window(_local(day, 12, 30), p=0.9),
            _window(_local(day, 18, 0), p=0.9),
            _window(_local(day, 5, 55), p=0.9),
        ],
        start_date=day,
        end_date=day,
    )
    counts = {stat.daypart: stat.episodes for stat in digest.distraction_by_daypart}
    assert counts == {"morning": 1, "afternoon": 1, "evening": 1, "night": 1}
    minutes = {stat.daypart: stat.distracted_minutes for stat in digest.distraction_by_daypart}
    assert minutes == {"morning": 5, "afternoon": 5, "evening": 5, "night": 5}


# ══════════════════════════════════════════════════════════════════════════════
# work_mix and distraction_by_context
# ══════════════════════════════════════════════════════════════════════════════


def test_work_mix_sorted_by_minutes_then_vocabulary_order() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base, context="writing"),
        _window(base + timedelta(minutes=5), context="coding"),
        _window(base + timedelta(minutes=10), context="coding"),
        _window(base + timedelta(minutes=15), context="reading"),
        _window(base + timedelta(minutes=20), context="reading"),
        _window(base + timedelta(minutes=25), context="coding"),
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert [item.context for item in digest.work_mix] == ["coding", "reading", "writing"]
    assert digest.work_mix[0].minutes == 15
    assert digest.work_mix[0].share == 0.5
    assert digest.work_mix[0].label == CONTEXT_LABELS_ZH["coding"]
    assert sum(item.minutes for item in digest.work_mix) == digest.totals.active_minutes


def test_work_mix_omits_zero_minute_contexts() -> None:
    day = date(2026, 11, 10)
    digest = _digest([_window(_local(day, 9))], start_date=day, end_date=day)
    assert [item.context for item in digest.work_mix] == ["coding"]


def test_distraction_by_context_uses_distracted_minutes_as_denominator() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base, p=0.9, context="entertainment"),
        _window(base + timedelta(minutes=5), p=0.9, context="entertainment"),
        _window(base + timedelta(minutes=10), p=0.1, context="coding"),
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert len(digest.distraction_by_context) == 1
    item = digest.distraction_by_context[0]
    assert item.context == "entertainment"
    assert item.minutes == 10
    assert item.share == 1.0
    assert digest.totals.distracted_share == 0.667


def test_unknown_task_context_is_reported_as_unidentified() -> None:
    day = date(2026, 11, 10)
    digest = _digest(
        [_window(_local(day, 9), context="unknown")], start_date=day, end_date=day
    )
    assert digest.work_mix[0].label == "未识别"


# ══════════════════════════════════════════════════════════════════════════════
# Totals and daily series
# ══════════════════════════════════════════════════════════════════════════════


def test_window_minutes_are_clipped_to_five() -> None:
    day = date(2026, 11, 10)
    digest = _digest([_window(_local(day, 9), minutes=30.0)], start_date=day, end_date=day)
    assert digest.totals.active_minutes == 5


def test_focused_and_distracted_minutes_partition_active_minutes() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base, p=0.9),
        _window(base + timedelta(minutes=5), p=0.1),
        _window(base + timedelta(minutes=10), p=0.5),
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert digest.totals.active_minutes == 15
    # p == 0.5 counts as distracted (threshold is >=).
    assert digest.totals.distracted_minutes == 10
    assert digest.totals.focused_minutes == 5


def test_daily_lists_every_day_of_the_span_with_zeros() -> None:
    start_date = date(2026, 11, 10)
    end_date = date(2026, 11, 13)
    digest = _digest(
        [_window(_local(date(2026, 11, 11), 9))],
        start_date=start_date,
        end_date=end_date,
    )
    assert [day.date for day in digest.daily] == [
        date(2026, 11, 10),
        date(2026, 11, 11),
        date(2026, 11, 12),
        date(2026, 11, 13),
    ]
    empty = digest.daily[0]
    assert empty.active_minutes == 0
    assert empty.distracted_minutes == 0
    assert empty.episodes == 0
    assert empty.top_context == "unknown"
    assert empty.top_context_label == "未识别"
    assert digest.totals.days_with_data == 1
    assert digest.period.days == 4


def test_daily_top_context_is_the_dominant_one() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base, context="reading"),
        _window(base + timedelta(minutes=5), context="coding"),
        _window(base + timedelta(minutes=10), context="coding"),
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert digest.daily[0].top_context == "coding"
    assert digest.daily[0].top_context_label == "编程"


def test_windows_outside_the_span_still_count_in_totals_but_not_in_daily() -> None:
    """A window that starts the local day before the span has no bucket."""
    day = date(2026, 11, 11)
    outside = _window(_local(date(2026, 11, 10), 23, 58), p=0.9)
    digest = _digest([outside], start_date=day, end_date=day)
    assert digest.totals.active_minutes == 5
    assert digest.daily[0].active_minutes == 0
    assert digest.totals.days_with_data == 0


def test_totals_are_zero_when_no_windows() -> None:
    day = date(2026, 11, 10)
    digest = _digest([], start_date=day, end_date=day)
    assert digest.totals.active_minutes == 0
    assert digest.totals.distracted_share == 0.0
    assert digest.episodes.avg_minutes == 0.0
    assert digest.episodes.longest_minutes == 0


# ══════════════════════════════════════════════════════════════════════════════
# data_state
# ══════════════════════════════════════════════════════════════════════════════


def test_data_state_no_model_when_model_unavailable() -> None:
    day = date(2026, 11, 10)
    digest = _digest([], start_date=day, end_date=day, model_available=False)
    assert digest.data_state == "no_model"
    assert digest.daily[0].active_minutes == 0


def test_data_state_insufficient_below_minimum_windows() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base + timedelta(minutes=5 * index))
        for index in range(MIN_ACTIVE_WINDOWS - 1)
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert digest.data_state == "insufficient"


def test_data_state_ready_at_minimum_windows() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base + timedelta(minutes=5 * index))
        for index in range(MIN_ACTIVE_WINDOWS)
    ]
    digest = _digest(windows, start_date=day, end_date=day)
    assert digest.data_state == "ready"


# ══════════════════════════════════════════════════════════════════════════════
# Interventions
# ══════════════════════════════════════════════════════════════════════════════


def test_intervention_counts_and_reason_ordering() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    facts = [
        InterventionFact(
            triggered_at_utc=base,
            reasons=("impulsivity", "task_aversion"),
            response="accepted",
        ),
        InterventionFact(triggered_at_utc=base, reasons=("impulsivity",), response="ignored"),
        InterventionFact(triggered_at_utc=base, reasons=("impulsivity",), response="dismissed"),
        InterventionFact(triggered_at_utc=base, reasons=(), response="pending"),
    ]
    digest = _digest([], start_date=day, end_date=day, interventions=facts)
    assert digest.interventions.total == 4
    assert digest.interventions.accepted == 1
    assert digest.interventions.ignored == 1
    assert digest.interventions.dismissed == 1
    assert digest.interventions.pending == 1
    assert [item.type for item in digest.interventions.by_reason] == [
        "impulsivity",
        "task_aversion",
    ]
    assert digest.interventions.by_reason[0].count == 3
    assert digest.interventions.by_reason[0].label == "冲动分心"


# ══════════════════════════════════════════════════════════════════════════════
# Evidence catalog
# ══════════════════════════════════════════════════════════════════════════════


def _rich_digest() -> PeriodDigest:
    """A digest with every catalog entry populated."""
    start_date = date(2026, 11, 10)
    end_date = date(2026, 11, 11)
    base = _local(start_date, 9)
    windows = [
        _window(base, p=0.9, context="entertainment"),
        _window(base + timedelta(minutes=5), p=0.9, context="entertainment"),
        _window(base + timedelta(minutes=10), p=0.1, context="coding"),
        _window(_local(end_date, 15), p=0.1, context="writing"),
    ]
    facts = [
        InterventionFact(
            triggered_at_utc=base,
            reasons=("impulsivity",),
            response="accepted",
        ),
        InterventionFact(triggered_at_utc=base, reasons=("impulsivity",), response="dismissed"),
    ]
    return _digest(
        windows, start_date=start_date, end_date=end_date, interventions=facts
    )


def test_evidence_catalog_ids_are_citation_safe() -> None:
    catalog = evidence_catalog(_rich_digest())
    assert catalog, "a populated digest must produce a catalog"
    for evidence_id in catalog:
        assert EVIDENCE_ID.match(evidence_id), evidence_id


def test_evidence_catalog_wording() -> None:
    digest = _rich_digest()
    catalog = evidence_catalog(digest)
    assert catalog["totals.active_minutes"] == "有效记录时长：20 分钟"
    assert catalog["totals.distracted_share"] == "分心时长占比：50%"
    assert catalog["episodes.count"] == "分心次数：1 次"
    assert catalog["episodes.longest_minutes"] == "最长一次分心：10 分钟"
    assert catalog["work_mix.coding"] == "编程：5 分钟（25%）"
    assert catalog["distraction_by_context.entertainment"] == "在娱乐上分心：10 分钟"
    assert catalog["daypart.morning"] == "上午：分心 1 次，共 10 分钟"
    assert catalog["daily.20261110"] == "11月10日：记录 15 分钟，分心 10 分钟，1 次"
    assert catalog["interventions.total"] == "收到提醒：2 次"
    assert catalog["interventions.accepted"] == "点了“好，试试看”：1 次"
    assert catalog["interventions.dismissed"] == "点了“不适合我”：1 次"
    assert catalog["reason.impulsivity"] == "提醒原因·冲动分心：2 次"


def test_evidence_catalog_omits_empty_days_and_quiet_dayparts() -> None:
    day = date(2026, 11, 10)
    digest = _digest([_window(_local(day, 9), p=0.1)], start_date=day, end_date=date(2026, 11, 11))
    catalog = evidence_catalog(digest)
    assert "daily.20261111" not in catalog
    assert "daily.20261110" in catalog
    assert not any(key.startswith("daypart.") for key in catalog)


def test_evidence_catalog_signal_direction_wording() -> None:
    from mindflow.domain.period_review import DistractionDriver

    day = date(2026, 11, 10)
    digest = build_period_digest(
        [],
        [],
        (
            DistractionDriver(
                feature="app_switch_count",
                label="应用切换次数",
                direction="higher",
                distracted_mean=9.2,
                focused_mean=3.1,
            ),
            DistractionDriver(
                feature="longest_segment_ratio",
                label="最长专注段占比",
                direction="lower",
                distracted_mean=0.2,
                focused_mean=0.7,
            ),
            DistractionDriver(
                feature="idle_ratio",
                label="空闲占比",
                direction="similar",
                distracted_mean=0.1,
                focused_mean=0.11,
            ),
        ),
        start_date=day,
        end_date=day,
        tz=SHANGHAI,
        model_version="m",
        is_base_model=False,
        model_available=True,
    )
    catalog = evidence_catalog(digest)
    assert catalog["signal.app_switch_count"] == "应用切换次数：分心时偏高（9.2 对 3.1）"
    assert catalog["signal.longest_segment_ratio"] == "最长专注段占比：分心时偏低（0.2 对 0.7）"
    assert catalog["signal.idle_ratio"] == "空闲占比：分心时差别不大（0.1 对 0.11）"


# ══════════════════════════════════════════════════════════════════════════════
# Template narrative
# ══════════════════════════════════════════════════════════════════════════════


def test_template_no_model_headline_and_copy() -> None:
    day = date(2026, 11, 10)
    narrative = template_narrative(
        _digest([], start_date=day, end_date=day, model_available=False)
    )
    assert narrative.source == "template"
    assert narrative.headline == "还没有可用的行为分类模型"
    assert [finding.text for finding in narrative.findings] == [
        "模型加载后才能按 5 分钟窗口判断专注和分心。"
    ]
    assert narrative.suggestions[0].text == "到“模型中心”确认基础模型已启用。"


def test_template_insufficient_headline_and_copy() -> None:
    day = date(2026, 11, 10)
    narrative = template_narrative(
        _digest([_window(_local(day, 9))], start_date=day, end_date=day)
    )
    assert narrative.headline == "记录还不够，暂时总结不出规律"
    assert narrative.findings[0].text == "这段时间只有 5 分钟有效记录，少于 1 小时。"
    assert narrative.findings[0].evidence[0].id == "totals.active_minutes"
    assert narrative.suggestions[0].text == "让 MindFlow 在后台多运行几天，再来看看。"


def _ready_digest(*, dismissed: int = 0) -> PeriodDigest:
    start_date = date(2026, 11, 10)
    base = _local(start_date, 9)
    windows = [
        _window(base + timedelta(minutes=5 * index), p=0.9 if index < 6 else 0.1)
        for index in range(MIN_ACTIVE_WINDOWS)
    ]
    facts = [
        InterventionFact(
            triggered_at_utc=base,
            reasons=("impulsivity",),
            response="dismissed" if index < dismissed else "accepted",
        )
        for index in range(dismissed)
    ]
    return _digest(windows, start_date=start_date, end_date=start_date, interventions=facts)


def test_template_ready_headline_reports_episodes_and_top_context() -> None:
    narrative = template_narrative(_ready_digest())
    assert narrative.headline == "最近1天分心 1 次，多发生在编程上"


def test_template_ready_without_distraction_says_so() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base + timedelta(minutes=5 * index), p=0.1)
        for index in range(MIN_ACTIVE_WINDOWS)
    ]
    narrative = template_narrative(_digest(windows, start_date=day, end_date=day))
    assert narrative.headline == "最近1天没有识别到明显的分心"


def test_template_ready_first_finding_reports_hours_and_share() -> None:
    narrative = template_narrative(_ready_digest())
    assert narrative.findings[0].text == (
        "有效记录 1 小时，模型判断专注 0.5 小时、分心 0.5 小时，分心约占 50%。"
    )
    assert [ref.id for ref in narrative.findings[0].evidence] == [
        "totals.active_minutes",
        "totals.focused_minutes",
        "totals.distracted_minutes",
        "totals.distracted_share",
    ]


def test_template_ready_mentions_base_model_caveat() -> None:
    narrative = template_narrative(_ready_digest())
    assert any("基础模型" in finding.text for finding in narrative.findings)


def test_template_ready_suggests_from_dismissed_feedback() -> None:
    narrative = template_narrative(_ready_digest(dismissed=2))
    assert any("不适合我" in suggestion.text for suggestion in narrative.suggestions)


def test_template_suggestions_are_capped_at_two() -> None:
    narrative = template_narrative(_ready_digest(dismissed=2))
    assert len(narrative.suggestions) <= 2


def test_template_falls_back_to_steady_pace_when_nothing_applies() -> None:
    day = date(2026, 11, 10)
    base = _local(day, 9)
    windows = [
        _window(base + timedelta(minutes=5 * index), p=0.1)
        for index in range(MIN_ACTIVE_WINDOWS)
    ]
    digest = _digest(
        windows, start_date=day, end_date=day, model_version="20260726_v2", is_base_model=False
    )
    narrative = template_narrative(digest)
    assert narrative.suggestions[0].text == "保持现在的节奏，有需要时再来回顾。"


def test_template_evidence_ids_exist_in_catalog() -> None:
    digest = _ready_digest()
    catalog = evidence_catalog(digest)
    for finding in template_narrative(digest).findings:
        for ref in finding.evidence:
            assert ref.id in catalog, ref.id


# ══════════════════════════════════════════════════════════════════════════════
# Model shape invariants
# ══════════════════════════════════════════════════════════════════════════════


def test_non_behavioral_features_are_excluded_by_name() -> None:
    assert {"hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "task_type_code"} == set(
        NON_BEHAVIORAL_FEATURES
    )


def test_digest_rejects_unknown_keys() -> None:
    digest = _rich_digest()
    with pytest.raises(ValidationError):
        PeriodDigest.model_validate({**digest.model_dump(), "window_titles": ["secret"]})


def test_digest_is_frozen() -> None:
    digest = _rich_digest()
    with pytest.raises(ValidationError):
        digest.totals.active_minutes = 0  # type: ignore[misc]


def test_every_nested_model_forbids_extra_keys() -> None:
    def walk(model_cls: type) -> None:
        config = getattr(model_cls, "model_config", {})
        assert config.get("extra") == "forbid", model_cls.__name__
        for field in getattr(model_cls, "model_fields", {}).values():
            annotation = field.annotation
            for candidate in getattr(annotation, "__args__", (annotation,)):
                if isinstance(candidate, type) and hasattr(candidate, "model_fields"):
                    walk(candidate)

    from mindflow.domain import period_review as module

    for name in (
        "PeriodRange",
        "ModelInfo",
        "PeriodTotals",
        "EpisodeSummary",
        "ContextMinutes",
        "DaypartStat",
        "DailyStat",
        "ModelSignal",
        "ReasonCount",
        "InterventionSummary",
        "PeriodDigest",
        "EvidenceRef",
        "NarrativeFinding",
        "NarrativeSuggestion",
        "NarrativePaper",
        "Narrative",
    ):
        walk(getattr(module, name))
