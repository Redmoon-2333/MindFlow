"""EvidenceBundle builder — the ML sensing layer's primary output.

Assembles an ``EvidenceBundle`` (§3 of 07-agent-upgrade-design.md) by:

  1. Querying raw activity events from ``ActivityRepository``.
  2. Computing features via ``domain/features`` (focus_score, switch_rate, …).
  3. Loading the user's personal baseline (``baseline_models`` table).
  4. Running ``DeviationDetector`` when baseline is available and sufficient.
  5. Querying recent intervention history from ``InterventionLogRepository``.
  6. Building a behavior summary via ``build_behavior_summary`` (summary.py).
  7. Detecting novelty with a lightweight heuristic (Phase A).

All DB access is async via injected repositories.  The builder itself is
stateless — a single instance is safe to share across requests.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.deviation import DeviationDetector
from mindflow.domain.events import ActivityEvent
from mindflow.domain.evidence import (
    EvidenceBundle,
    EvidenceItem,
    InterventionRecord,
    Severity,
)
from mindflow.domain.features import (
    MAX_ACCEPTABLE_SWITCHES_PER_HOUR,
    app_usage_ranking,
    focus_score,
    longest_focus_block_s,
    switch_rate_per_hour,
)
from mindflow.domain.procrastination import BehaviorSummary
from mindflow.infrastructure.llm.summary import build_behavior_summary
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.services.effectiveness_service import (
    EffectivenessReport,
    EffectivenessService,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Minimum days of baseline data required for reliable deviation detection
MIN_BASELINE_DAYS: int = 7

# Severity thresholds for focus_score (0–100, higher = better).
#   >= 70: info     — Normal / good focus.
#   [50, 70): mild  — Slightly low, noticeable but not urgent.
#   [30, 50): moderate — Clearly low, warrants attention.
#   < 30:   severe  — Critical deficit (extreme scatter / very high switching).
# Rationale: 70+ indicates the user spent most time in one app with few
# switches.  < 30 means the top-app ratio is very low OR switch frequency
# is extremely high.  The 30–50–70 bands match the focus-score range where
# the old codebase's "low focus" alerts started firing.
_FOCUS_SCORE_INFO: float = 70.0
_FOCUS_SCORE_MILD: float = 50.0
_FOCUS_SCORE_MODERATE: float = 30.0

# Severity thresholds for switch_rate_per_hour (higher = worse).
#   <= 15:          info     — Within normal range.
#   [15, 30]:       mild     — Elevated, approaching MAX_ACCEPTABLE.
#   (30, 45]:       moderate — Above MAX_ACCEPTABLE (30), high switching.
#   > 45:           severe   — Extreme task-switching (> 1.5x max acceptable).
# Rationale: MAX_ACCEPTABLE_SWITCHES_PER_HOUR = 30 is the codebase-wide
# threshold (domain/features.py:35).  > 45 means 1.5× that threshold.
_SWITCH_RATE_INFO: float = 15.0
_SWITCH_RATE_MILD: float = MAX_ACCEPTABLE_SWITCHES_PER_HOUR  # 30
_SWITCH_RATE_MODERATE: float = 45.0

# Severity thresholds for longest_focus_block_s (seconds, higher = better).
#   >= 1200 (20 min): info     — Good sustained focus.
#   [600, 1200):      mild     — Moderate blocks, some fragmentation.
#   [300, 600):       moderate — Short blocks (5–10 min), below impulsivity threshold.
#   < 300:            severe   — Very short (< 5 min — matches the rule-engine's
#                                impulsivity threshold in domain/procrastination.py:137).
# Rationale: 300 s is the impulsivity trigger in RuleEngine.  20+ minutes
# is a healthy sustained block for knowledge work.
_LONGEST_BLOCK_INFO: float = 1200.0
_LONGEST_BLOCK_MILD: float = 600.0
_LONGEST_BLOCK_MODERATE: float = 300.0

# Default confidence for feature-computation items (rule-based, high confidence)
_FEATURE_CONFIDENCE: float = 0.85

# How many days of intervention history to include
_INTERVENTION_LOOKBACK_DAYS: int = 7

# How many top apps to consider for novelty detection
_NOVELTY_TOP_N: int = 5


# ═══════════════════════════════════════════════════════════════════════════════
# baseline_models table (read-only reference)
# ═══════════════════════════════════════════════════════════════════════════════

baseline_models_metadata = sa.MetaData()

baseline_models = sa.Table(
    "baseline_models",
    baseline_models_metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("user_id", sa.Integer(), nullable=False),
    sa.Column("model_json", sa.Text(), nullable=False),
    sa.Column("training_events_count", sa.Integer(), nullable=False),
    sa.Column("created_at", sa.Text(), nullable=False),
    sa.Column("updated_at", sa.Text(), nullable=False),
)


# ═══════════════════════════════════════════════════════════════════════════════
# Severity helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _score_severity(value: float, info_th: float, mild_th: float, moderate_th: float, higher_is_better: bool) -> Severity:  # noqa: E501
    """Map a numeric metric to a severity level.

    Args:
        value: The observed value.
        info_th: Threshold for "info" (anything on the good side).
        mild_th: Threshold for "mild".
        moderate_th: Threshold for "moderate" (anything on the bad side is "severe").
        higher_is_better: If True, higher values are better (e.g. focus_score).
                          If False, lower values are better (e.g. switch_rate).

    Returns:
        One of "info", "mild", "moderate", "severe".
    """
    if higher_is_better:
        if value >= info_th:
            return "info"
        if value >= mild_th:
            return "mild"
        if value >= moderate_th:
            return "moderate"
        return "severe"
    else:
        if value <= info_th:
            return "info"
        if value <= mild_th:
            return "mild"
        if value <= moderate_th:
            return "moderate"
        return "severe"


def _focus_severity(score: float) -> Severity:
    """Severity for focus_score (higher = better)."""
    return _score_severity(
        score, _FOCUS_SCORE_INFO, _FOCUS_SCORE_MILD, _FOCUS_SCORE_MODERATE, higher_is_better=True
    )


def _switch_severity(rate: float) -> Severity:
    """Severity for switch_rate_per_hour (lower = better)."""
    return _score_severity(
        rate, _SWITCH_RATE_INFO, _SWITCH_RATE_MILD, _SWITCH_RATE_MODERATE, higher_is_better=False
    )


def _block_severity(duration_s: float) -> Severity:
    """Severity for longest_focus_block_s (higher = better)."""
    return _score_severity(
        duration_s,
        _LONGEST_BLOCK_INFO,
        _LONGEST_BLOCK_MILD,
        _LONGEST_BLOCK_MODERATE,
        higher_is_better=True,
    )


def _deviation_severity(deviation_score: str) -> Severity:
    """Map DeviationDetector severity to EvidenceItem severity.

    DeviationDetector returns "normal" | "mild" | "moderate" | "severe".
    "normal" is treated as "info" (no actionable deviation).
    """
    if deviation_score == "normal":
        return "info"
    # All others pass through directly
    return deviation_score  # type: ignore[return-value]


def _deviation_to_confidence(overall: float) -> float:
    """Map deviation z-score magnitude to confidence in [0, 1].

    Higher absolute z-scores → higher confidence that the deviation is real.
      z = 0.0   → 0.0 (no deviation)
      z = 1.5   → 0.5 (mild threshold)
      z = 4.0   → 0.95 (severe threshold, saturates)
    """
    abs_z = abs(overall)
    if abs_z >= 4.0:
        return 0.95
    if abs_z <= 0.0:
        return 0.0
    return round(abs_z / 4.0 * 0.95, 3)


def _response_to_effect_note(user_response: str | None) -> str:
    """Map a DB user_response value to a Chinese effect description."""
    mapping: dict[str | None, str] = {
        "accepted": "已接受并执行",
        "ignored": "用户忽略",
        "dismissed": "用户已关闭",
    }
    return mapping.get(user_response, "尚未回应")


# ═══════════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════════


class EvidenceBundleBuilder:
    """Assembles an ``EvidenceBundle`` from repositories and domain logic.

    Stateless — a single instance is safe to share.  All I/O is async
    via injected repositories.

    Args:
        activity_repo: Repository for activity event data.
        intervention_repo: Repository for intervention history.
        session_factory: SQLAlchemy session factory for baseline_models query.
        effectiveness_service: Optional effectiveness service for enriching
            intervention records with outcome data (G005 learning loop).
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
        session_factory: async_sessionmaker[AsyncSession],
        effectiveness_service: EffectivenessService | None = None,
    ) -> None:
        self._activity_repo = activity_repo
        self._intervention_repo = intervention_repo
        self._session_factory = session_factory
        self._effectiveness_service = effectiveness_service

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    async def build(
        self,
        user_id: int,
        window_start: datetime,
        window_end: datetime,
    ) -> EvidenceBundle:
        """Assemble a complete ``EvidenceBundle`` for the given time window.

        Args:
            user_id: The user to analyse.
            window_start: Start of the analysis window (timezone-aware UTC).
            window_end: End of the analysis window (timezone-aware UTC).

        Returns:
            A fully populated ``EvidenceBundle``.
        """
        # 1. Query raw activity events
        events = await self._activity_repo.query_range(user_id, window_start, window_end)
        # Note: events may be empty — all downstream functions handle this gracefully.

        # 2. Compute features and build evidence items
        items: list[EvidenceItem] = []
        items.extend(self._build_feature_items(events))

        # 3. Load baseline (single DB call, shared by deviation + novelty)
        baseline = await self._load_baseline(user_id)

        # 4. Run deviation detection
        items.extend(self._build_deviation_items(baseline, events, window_start))

        # 5. Build behavior summary
        behavior_summary = build_behavior_summary(events) if events else self._empty_summary()

        # 6. Query intervention history (last 7 days)
        interventions = await self._build_intervention_history(user_id, window_end)

        # 7. Detect novelty flags
        novelty_flags = self._detect_novelty(events, baseline)

        return EvidenceBundle(
            user_id=user_id,
            window=(window_start, window_end),
            items=tuple(items),
            behavior_summary=behavior_summary,
            intervention_history=tuple(interventions),
            novelty_flags=tuple(novelty_flags),
        )

    # ══════════════════════════════════════════════════════════════════════
    # Feature-based evidence items
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_feature_items(events: list[ActivityEvent]) -> list[EvidenceItem]:
        """Generate EvidenceItems from raw feature computations.

        Produces up to 4 items: focus_score, switch_rate, longest_block, top_apps.
        """
        items: list[EvidenceItem] = []

        if not events:
            items.append(
                EvidenceItem(
                    metric="focus_score",
                    value=0.0,
                    baseline=None,
                    severity="info",
                    confidence=_FEATURE_CONFIDENCE,
                    source="feature_computation",
                    human_readable="窗口内无活动数据",
                )
            )
            return items

        # --- focus_score ---
        score = focus_score(events)
        sev = _focus_severity(score)
        items.append(
            EvidenceItem(
                metric="focus_score",
                value=score,
                baseline=None,
                severity=sev,
                confidence=_FEATURE_CONFIDENCE,
                source="feature_computation",
                human_readable=_format_focus_readable(score, sev),
            )
        )

        # --- switch_rate_per_hour ---
        switch_rate = switch_rate_per_hour(events)
        sev = _switch_severity(switch_rate)
        items.append(
            EvidenceItem(
                metric="switch_rate",
                value=switch_rate,
                baseline=None,
                severity=sev,
                confidence=_FEATURE_CONFIDENCE,
                source="feature_computation",
                human_readable=_format_switch_readable(switch_rate, sev),
            )
        )

        # --- longest_focus_block_s ---
        longest_block = longest_focus_block_s(events)
        sev = _block_severity(longest_block)
        items.append(
            EvidenceItem(
                metric="longest_block",
                value=longest_block,
                baseline=None,
                severity=sev,
                confidence=_FEATURE_CONFIDENCE,
                source="feature_computation",
                human_readable=_format_block_readable(longest_block, sev),
            )
        )

        # --- top_apps (usage ranking) ---
        ranking = app_usage_ranking(events)
        if ranking:
            top_app_names = ", ".join(a.app_name for a in ranking[:3])
            items.append(
                EvidenceItem(
                    metric="top_apps",
                    value=top_app_names,
                    baseline=None,
                    severity="info",
                    confidence=_FEATURE_CONFIDENCE,
                    source="feature_computation",
                    human_readable=f"主要使用: {top_app_names}",
                )
            )

        return items

    # ══════════════════════════════════════════════════════════════════════
    # Baseline / deviation items
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_deviation_items(
        baseline: BaselineModel | None,
        events: list[ActivityEvent],
        window_start: datetime,
    ) -> list[EvidenceItem]:
        """Generate deviation-related EvidenceItems from the personal baseline.

        Returns a single behavior_deviation item, or an info-level item explaining
        that the baseline is still being built.
        """
        if baseline is None:
            return [
                EvidenceItem(
                    metric="behavior_deviation",
                    value=0.0,
                    baseline=None,
                    severity="info",
                    confidence=1.0,
                    source="welford_baseline",
                    human_readable="基线尚在建立中（尚无数据）",
                )
            ]

        if not baseline.has_sufficient_data():
            return [
                EvidenceItem(
                    metric="behavior_deviation",
                    value=0.0,
                    baseline=None,
                    severity="info",
                    confidence=1.0,
                    source="welford_baseline",
                    human_readable=(
                        f"基线尚在建立中（总计{baseline.total_samples()}个样本，"
                        f"建议至少{MIN_BASELINE_DAYS * 6}个，可启动训练补齐）"
                    ),
                )
            ]

        # Check per-bucket sufficiency for the current time window
        hour = window_start.hour
        dow = window_start.weekday()
        if not baseline.has_bucket_sufficient_data(hour, dow):
            return [
                EvidenceItem(
                    metric="behavior_deviation",
                    value=0.0,
                    baseline=None,
                    severity="info",
                    confidence=0.8,
                    source="welford_baseline",
                    human_readable=f"当前时段({hour}:00 周{dow})基线数据不足，暂无法计算偏差",
                )
            ]

        # Construct a pseudo feature row for the DeviationDetector
        pseudo_row = _build_pseudo_row(events, window_start)
        if pseudo_row is None:
            return [
                EvidenceItem(
                    metric="behavior_deviation",
                    value=0.0,
                    baseline=None,
                    severity="info",
                    confidence=1.0,
                    source="welford_baseline",
                    human_readable="数据不足以计算基线偏差",
                )
            ]

        detector = DeviationDetector(baseline)
        deviation = detector.score_window(pseudo_row)
        overall = deviation["overall_deviation"]
        sev = _deviation_severity(deviation["severity"])
        conf = _deviation_to_confidence(overall)

        if sev == "info":
            return [
                EvidenceItem(
                    metric="behavior_deviation",
                    value=overall,
                    baseline=0.0,
                    severity="info",
                    confidence=conf,
                    source="welford_baseline",
                    human_readable="行为模式与基线一致，无显著偏差",
                )
            ]

        # Build a richer human_readable for non-info deviations
        top_deviations = deviation.get("top_deviations", [])
        dev_detail = ""
        if top_deviations:
            top = top_deviations[0]
            dev_detail = f"，主要偏离: {top.get('feature', '未知')}(Z={top.get('z_score', 0):.1f})"

        return [
            EvidenceItem(
                metric="behavior_deviation",
                value=overall,
                baseline=0.0,
                severity=sev,
                confidence=conf,
                source="welford_baseline",
                human_readable=f"行为模式偏离基线 {overall:.1f} 个标准差（{sev}）{dev_detail}",
            )
        ]

    async def _load_baseline(self, user_id: int) -> BaselineModel | None:
        """Load the user's ``BaselineModel`` from the database.

        Returns None if no baseline has been built yet for this user.
        """
        # Latest baseline wins — the table has no user_id uniqueness, so a
        # retrained user may own several rows (review H1: unordered fetchone
        # could load a stale model).
        stmt = (
            sa.select(baseline_models.c.model_json)
            .where(baseline_models.c.user_id == user_id)
            .order_by(sa.desc(baseline_models.c.updated_at))
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            row = result.fetchone()

        if row is None:
            return None

        try:
            data: dict[str, Any] = json.loads(row.model_json)
        except (json.JSONDecodeError, TypeError):
            return None

        return BaselineModel.from_dict(data)

    # ══════════════════════════════════════════════════════════════════════
    # Intervention history
    # ══════════════════════════════════════════════════════════════════════

    async def _build_intervention_history(
        self,
        user_id: int,
        window_end: datetime,
    ) -> list[InterventionRecord]:
        """Query recent intervention logs and convert to ``InterventionRecord``.

        Covers the *INTERVENTION_LOOKBACK_DAYS* days leading up to *window_end*.
        When *effectiveness_service* is available, enriches the most recent 5
        records with outcome summaries (G005 learning loop).
        """
        lookback_start = window_end - timedelta(days=_INTERVENTION_LOOKBACK_DAYS)
        logs = await self._intervention_repo.query_range(user_id, lookback_start, window_end)

        # ── Pre-fetch effectiveness data for last 5 interventions ──────
        # Bounded at 5 to prevent N+1 across all logs (G005 learning loop).
        effect_data: dict[str, str] = {}
        if self._effectiveness_service is not None and logs:
            recent_logs = logs[-5:]
            for log_entry in recent_logs:
                lid: str = log_entry.get("id", "")
                if not lid:
                    continue
                try:
                    report = await self._effectiveness_service.compare_windows(lid)
                    if report.has_data:
                        summary = self._format_effectiveness_summary(report)
                        if summary:
                            effect_data[lid] = summary
                except Exception:
                    logger.debug("No effectiveness data for intervention {}", lid)

        records: list[InterventionRecord] = []
        for log in logs:
            triggered_at_str = log.get("triggered_at", "")
            try:
                triggered_at = datetime.fromisoformat(triggered_at_str)
            except (ValueError, TypeError):
                # Review M2: never drop history silently — experts receiving an
                # empty history would wrongly conclude no interventions happened.
                logger.warning(
                    "Dropped intervention log {}: bad triggered_at {!r}",
                    log.get("id"),
                    triggered_at_str,
                )
                continue

            initial_note = _response_to_effect_note(log.get("user_response"))
            enhanced = effect_data.get(log.get("id", ""), "")
            effect_note = f"{enhanced}；{initial_note}" if enhanced else initial_note

            records.append(
                InterventionRecord(
                    intervention_type=str(log.get("intervention_type", "unknown")),
                    triggered_at=triggered_at,
                    user_response=log.get("user_response"),
                    effect_note=effect_note,
                )
            )
        return records

    @staticmethod
    def _format_effectiveness_summary(report: EffectivenessReport) -> str:
        """Format an ``EffectivenessReport`` into a Chinese summary string.

        Produces text like "干预后专注 +18%" for use in effect_note.
        Returns empty string when no meaningful delta is present.
        """
        parts: list[str] = []
        focus_delta = report.deltas.get("focus_score", 0.0)
        switch_delta = report.deltas.get("switch_rate", 0.0)

        if abs(focus_delta) >= 1.0:
            sign = "+" if focus_delta > 0 else ""
            parts.append(f"专注{sign}{focus_delta:.0f}%")
        if abs(switch_delta) >= 0.5:
            if switch_delta < 0:
                parts.append(f"切换频率-{abs(switch_delta):.1f}次/时")
            else:
                parts.append(f"切换频率+{switch_delta:.1f}次/时")

        return "干预后：" + "，".join(parts) if parts else ""

    # ══════════════════════════════════════════════════════════════════════
    # Novelty detection
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _detect_novelty(
        events: list[ActivityEvent],
        baseline: BaselineModel | None,
    ) -> list[str]:
        """Detect novel app usage patterns (Phase A: lightweight heuristic).

        Flags apps in the current window's top 5 that have never appeared in
        the user's baseline top 10 across any bucket.

        Phase A simplification: only checks baseline-model top-apps.
        Cluster-based novelty detection is deferred to G005.

        Returns:
            A list of human-readable flag strings, or empty if none detected.
        """
        if not events or baseline is None:
            return []

        # Current window's top apps
        ranking = app_usage_ranking(events)
        if not ranking:
            return []
        current_top5 = {a.app_name for a in ranking[:_NOVELTY_TOP_N]}

        # Baseline top 10 across ALL buckets — collect from _top_apps
        baseline_apps: set[str] = set()
        for hour_bucket in baseline._top_apps.values():  # noqa: SLF001 — Phase A access to private attr
            for dow_bucket in hour_bucket.values():
                for app_name in dow_bucket:
                    baseline_apps.add(app_name)

        if not baseline_apps:
            return []

        novel = current_top5 - baseline_apps
        if not novel:
            return []

        return [f"新应用模式: {app}" for app in sorted(novel)]

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _empty_summary() -> BehaviorSummary:
        """Return a zeroed-out BehaviorSummary for empty event windows."""
        return BehaviorSummary(
            intended_task=None,
            duration_min=0.0,
            actual_focus_min=0.0,
            context_switches_per_hour=0.0,
            longest_focus_block_s=0.0,
            social_media_ratio=0.0,
            start_delay_min=0.0,
            keyword_flags=frozenset(),
            baseline_deviation=None,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _build_pseudo_row(events: list[ActivityEvent], window_start: datetime) -> dict[str, Any] | None:
    """Build a pseudo feature-row dict for ``DeviationDetector.score_window``.

    Extracts the features we can compute from the raw event list.  Features
    that require title analysis (title_code_ratio, title_doc_ratio, …) are
    omitted — ``DeviationDetector`` gracefully skips missing features.

    Returns None when there are too few non-idle events to compute anything
    meaningful.
    """
    non_idle = [e for e in events if not e.data.is_idle]
    if len(non_idle) < 2:
        return None

    app_durations: dict[str, float] = {}
    for ev in non_idle:
        app_durations[ev.data.process_name] = (
            app_durations.get(ev.data.process_name, 0.0) + ev.duration_s
        )

    unique_app_count = len(app_durations)
    max_app_duration = max(app_durations.values()) if app_durations else 0.0
    switch_frequency = switch_rate_per_hour(non_idle)

    # idle_ratio — cheap to compute and raises weighted coverage from 45% to
    # 55% of DeviationDetector's feature weights (review M1 partial fix).
    # Full classifier-backed productivity/entertainment/social/title ratios
    # depend on AppClassifier + pandas and belong in G005.
    total_dur = sum(ev.duration_s for ev in events)
    idle_dur = sum(ev.duration_s for ev in events if ev.data.is_idle)
    idle_ratio = idle_dur / total_dur if total_dur > 0 else 0.0

    return {
        "hour_of_day": window_start.hour,
        "day_of_week": window_start.weekday(),
        "switch_frequency": switch_frequency,
        "unique_app_count": float(unique_app_count),
        "max_app_duration": max_app_duration,
        "idle_ratio": idle_ratio,
        "window_start": window_start.isoformat(),
    }


def _format_focus_readable(score: float, severity: Severity) -> str:
    """Chinese human-readable for focus_score."""
    label = {
        "info": "正常",
        "mild": "偏低",
        "moderate": "明显偏低",
        "severe": "极低",
    }.get(severity, "未知")
    return f"专注度评分 {score:.1f}/100，{label}"


def _format_switch_readable(rate: float, severity: Severity) -> str:
    """Chinese human-readable for switch_rate."""
    label = {
        "info": "正常",
        "mild": "偏高",
        "moderate": "较高",
        "severe": "极高",
    }.get(severity, "未知")
    return f"应用切换频率 {rate:.1f} 次/小时，{label}"


def _format_block_readable(duration_s: float, severity: Severity) -> str:
    """Chinese human-readable for longest_focus_block."""
    minutes = duration_s / 60.0
    label = {
        "info": "良好",
        "mild": "偏短",
        "moderate": "较短",
        "severe": "极短",
    }.get(severity, "未知")
    return f"最长专注块 {minutes:.0f} 分钟，{label}"
