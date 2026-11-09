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

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.classifier import AppClassifier
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
    QUIET_IDLE_TOLERANCE_S,
    app_usage_ranking,
    count_confirmed_switches,
    focus_score,
    longest_focus_block_s,
    switch_rate_per_hour,
)
from mindflow.domain.procrastination import BehaviorSummary
from mindflow.domain.tmt import (
    TmtBreakdown,
    TmtInputs,
    compute_tmt,
    deviation_z_mean,
)
from mindflow.infrastructure.llm.summary import build_behavior_summary
from mindflow.infrastructure.repositories.activity import (
    SQLAlchemyActivityRepository,
)
from mindflow.infrastructure.repositories.baseline import BaselineRepository

# Backward-compat re-exports: the ``baseline_models`` table now lives in
# ``infrastructure/repositories/baseline.py`` (single source of truth).
# Existing importers (e.g. tests that create/insert rows) keep working.
from mindflow.infrastructure.repositories.baseline import (
    baseline_models as baseline_models,
)
from mindflow.infrastructure.repositories.intervention import (
    InterventionLogRepository,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.services.effectiveness_service import (
    EffectivenessReport,
    EffectivenessService,
)
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.time_utils import resolve_timezone

# Backward-compat alias: ``baseline_models.metadata`` is the MetaData the table
# was registered on. Kept so ``baseline_models_metadata.create_all`` importers
# (test fixtures) keep working after the table moved to the baseline repo.
baseline_models_metadata = baseline_models.metadata

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

# TMT proxies: the self-report lookback for expectancy, the "due soon" horizon
# for the value proxy, and how much confidence an info-level TMT row carries
# (behavioural proxy, not a measurement — deliberately below the
# feature-computation items).
_TMT_FEEDBACK_LOOKBACK_DAYS: int = 14
_TMT_DUE_SOON_DAYS: int = 7
_TMT_CONFIDENCE: float = 0.6

# How many top apps to consider for novelty detection
_NOVELTY_TOP_N: int = 5

# ML classifier focus probability thresholds (higher = better focus).
#   >= 0.70: info     — Model confident this is focus.
#   [0.50, 0.70): mild — Somewhat focus, borderline.
#   [0.30, 0.50): moderate — More likely distraction.
#   < 0.30:   severe  — Model confident this is distraction.
_ML_FOCUS_INFO: float = 0.70
_ML_FOCUS_MILD: float = 0.50
_ML_FOCUS_MODERATE: float = 0.30

# Human-readable labels for behavior clusters
_CLUSTER_READABLE: dict[str, str] = {
    "deep_focus": "深度专注",
    "shallow_work": "浅层工作",
    "browsing": "浏览浏览",
    "procrastination": "拖延状态",
    "idle": "空闲",
    "noise": "未分类",
}


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


def _ml_focus_severity(focus_proba: float) -> Severity:
    """Severity for ML classifier focus probability (higher = better focus)."""
    return _score_severity(
        focus_proba,
        _ML_FOCUS_INFO,
        _ML_FOCUS_MILD,
        _ML_FOCUS_MODERATE,
        higher_is_better=True,
    )


def _format_ml_focus_readable(focus_proba: float, severity: Severity) -> str:
    """Chinese human-readable for ML focus probability."""
    label = {
        "info": "专注",
        "mild": "一般",
        "moderate": "分心倾向",
        "severe": "明显分心",
    }.get(severity, "未知")
    return f"ML专注概率 {focus_proba:.0%}，{label}"


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
        session_factory: SQLAlchemy session factory (used to construct a
            default ``BaselineRepository`` when one is not injected).
        effectiveness_service: Optional effectiveness service for enriching
            intervention records with outcome data (G005 learning loop).
        baseline_repo: Repository for reading the user's personal baseline.
            Defaults to a ``BaselineRepository`` built from *session_factory*
            (kept optional so existing call sites need no change).
        prediction_service: Optional ML prediction service for ML evidence.
        telemetry_repo: Optional reader for persisted V2 feature windows. When
            injected, deviation detection scores the newest stored 5-minute
            window (the same vector the classifier sees); otherwise it derives
            a row from raw events. Also supplies self-reported focus feedback
            for the TMT expectancy proxy.
        task_repo: Optional task repository for the TMT value and delay
            proxies (open-task priority and nearest deadline). Absent → those
            proxies fall back to the behavioural/neutral values.
    """

    def __init__(
        self,
        activity_repo: SQLAlchemyActivityRepository,
        intervention_repo: InterventionLogRepository,
        session_factory: async_sessionmaker[AsyncSession],
        effectiveness_service: EffectivenessService | None = None,
        baseline_repo: BaselineRepository | None = None,
        prediction_service: FocusPredictionService | None = None,
        telemetry_repo: TelemetryRepository | None = None,
        task_repo: object | None = None,
    ) -> None:
        self._activity_repo = activity_repo
        self._intervention_repo = intervention_repo
        self._session_factory = session_factory
        self._effectiveness_service = effectiveness_service
        self._baseline_repo = baseline_repo or BaselineRepository(
            session_factory=session_factory
        )
        self._prediction_service = prediction_service
        self._telemetry_repo = telemetry_repo
        self._task_repo = task_repo

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
        # 1-3, 6. Independent DB reads — fetch concurrently. The events query,
        # baseline load, and intervention history have no data dependency on
        # each other (each opens its own session and uses only user_id / the
        # window bounds), so gather them instead of three sequential awaits.
        events, baseline, interventions = await asyncio.gather(
            self._activity_repo.query_overlapping_range(
                user_id, window_start, window_end
            ),
            self._load_baseline(user_id),
            self._build_intervention_history(user_id, window_end),
        )
        # Note: events may be empty — all downstream functions handle this gracefully.

        # 2. Compute features and build evidence items
        items: list[EvidenceItem] = []
        items.extend(self._build_feature_items(events))

        # 3. ML enrichment (v2 feature windows via FocusPredictionService)
        ml_items = await self._build_ml_items(events, window_start, window_end, user_id)
        items.extend(ml_items)

        # 4. Run deviation detection (personal baseline, 5-min v2 windows)
        deviation_items, deviation_summary = await self._build_deviation(
            baseline, events, window_start, user_id
        )
        items.extend(deviation_items)

        # 5. Build behavior summary
        behavior_summary = build_behavior_summary(events) if events else self._empty_summary()

        # 7. Detect novelty flags
        novelty_flags = self._detect_novelty(events, baseline)

        # 8. TMT decomposition from the same window (behavioural proxies)
        tmt_inputs = await self._build_tmt_inputs(
            events=events,
            deviation_summary=deviation_summary,
            window_end=window_end,
            user_id=user_id,
        )
        tmt = compute_tmt(tmt_inputs)
        items.extend(self._tmt_items(tmt))

        return EvidenceBundle(
            user_id=user_id,
            window=(window_start, window_end),
            items=tuple(items),
            behavior_summary=behavior_summary,
            intervention_history=tuple(interventions),
            novelty_flags=tuple(novelty_flags),
            events=tuple(events),
            deviation_summary=deviation_summary,
            tmt=tmt,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Feature-based evidence items
    # ══════════════════════════════════════════════════════════════════════

    def _build_feature_items(self, events: list[ActivityEvent]) -> list[EvidenceItem]:
        """Generate EvidenceItems from raw feature computations.

        Produces up to 4 rule-based items: focus_score, switch_rate,
        longest_block, top_apps.  When a ``ModelManager`` is available,
        ML-derived items (classifier prediction, cluster assignment) are
        appended alongside — they never replace the rule-based items.

        Degradation chain (mirrors LLM three-tier):
          1. ML models loaded  -> rule-based + ML enrichment
          2. ML models missing -> rule-only (current behaviour)
          3. ML inference fails -> log warning, rule-only fallback
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
        longest_block = longest_focus_block_s(
            events, idle_tolerance_s=QUIET_IDLE_TOLERANCE_S
        )
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

        # --- ML enrichment handled in build() via FocusPredictionService ---

        return items

    async def _build_ml_items(
        self,
        events: list[ActivityEvent],
        window_start: datetime,
        window_end: datetime,
        user_id: int = 1,
    ) -> list[EvidenceItem]:
        """Run ML inference via ``FocusPredictionService`` and return enrichment EvidenceItems.

        Uses v2 feature windows (privacy-preserving, 5-min) instead of the
        legacy v1 ``BehaviorFeatureExtractor`` (30-min windows with PII).

        Returns 0-4 items depending on prediction status. All exceptions are
        caught and logged — ML failure must never break the evidence pipeline.

        Degradation chain:
          1. Prediction ready + fresh + good coverage → full ML evidence
          2. Prediction ready but stale → ML evidence with staleness note
          3. No model / no data / schema mismatch → no ML items (rule-only fallback)
          4. Inference error → logged, no ML items
        """
        if self._prediction_service is None:
            return []

        try:
            prediction = await self._prediction_service.predict_range(
                user_id=user_id,
                start=window_start,
                end=window_end,
            )
        except Exception as exc:
            logger.debug("ML enrichment via FocusPredictionService failed: {}", exc)
            return []

        # Only enrich when we have a usable prediction
        if prediction.status == "no_model":
            logger.debug("ML enrichment skipped: no model loaded")
            return []

        if prediction.status == "no_data":
            logger.debug("ML enrichment skipped: no v2 feature windows in analysis range")
            return []

        if prediction.status == "schema_mismatch":
            logger.debug("ML enrichment skipped: feature schema mismatch")
            return []

        if prediction.status == "inference_error":
            logger.debug("ML enrichment skipped: {}", prediction.reason)
            return []

        # status is "ready" or "stale" — both produce evidence, but
        # "stale" items get a note in their human_readable field.
        is_stale = prediction.status == "stale"
        staleness_note = "（数据已过期）" if is_stale else ""

        items: list[EvidenceItem] = []

        # --- ml_focus_probability_mean ---
        if prediction.focus_probability is not None:
            focus_proba = prediction.focus_probability
            ml_focus_sev = _ml_focus_severity(focus_proba)
            confidence = round(
                focus_proba if focus_proba >= 0.5 else 1.0 - focus_proba, 3
            )
            items.append(
                EvidenceItem(
                    metric="ml_focus_probability",
                    value=round(focus_proba, 4),
                    baseline=None,
                    severity=ml_focus_sev,
                    confidence=confidence,
                    source="rf_classifier_v2",
                    human_readable=(
                        _format_ml_focus_readable(focus_proba, ml_focus_sev) + staleness_note
                    ),
                )
            )

        # --- ml_distracted_window_ratio ---
        if prediction.distracted_window_ratio is not None and prediction.window_count > 0:
            d_ratio = prediction.distracted_window_ratio
            # Confidence: higher when ratio is clearly high or low
            d_confidence = round(max(d_ratio, 1.0 - d_ratio), 3)
            items.append(
                EvidenceItem(
                    metric="ml_distracted_window_ratio",
                    value=round(d_ratio, 4),
                    baseline=None,
                    severity=_ml_focus_severity(1.0 - d_ratio),
                    confidence=d_confidence,
                    source="rf_classifier_v2",
                    human_readable=(
                        f"ML分心窗口占比 {d_ratio:.0%}{staleness_note}"
                    ),
                )
            )

        # --- ml_uncertainty_mean ---
        if prediction.uncertainty is not None:
            items.append(
                EvidenceItem(
                    metric="ml_uncertainty",
                    value=round(prediction.uncertainty, 4),
                    baseline=None,
                    severity="info",
                    confidence=max(1.0 - prediction.uncertainty, 0.1),
                    source="rf_classifier_v2",
                    human_readable=(
                        f"ML预测不确定度 {prediction.uncertainty:.2f}"
                        f"（{'高' if prediction.uncertainty > 0.5 else '低'}）{staleness_note}"
                    ),
                )
            )

        # --- ml_feature_coverage ---
        if prediction.window_count > 0:
            items.append(
                EvidenceItem(
                    metric="ml_feature_coverage",
                    value=round(prediction.coverage_ratio, 4),
                    baseline=None,
                    severity="info",
                    confidence=min(prediction.coverage_ratio + 0.2, 1.0),
                    source="rf_classifier_v2",
                    human_readable=(
                        f"ML特征覆盖率 {prediction.coverage_ratio:.0%}"
                        f"（{prediction.window_count}个窗口）{staleness_note}"
                    ),
                )
            )

        if not items:
            return []

        return items

    # ══════════════════════════════════════════════════════════════════════
    # Baseline / deviation items
    # ══════════════════════════════════════════════════════════════════════

    async def _build_deviation(
        self,
        baseline: BaselineModel | None,
        events: list[ActivityEvent],
        window_start: datetime,
        user_id: int,
    ) -> tuple[list[EvidenceItem], dict[str, Any] | None]:
        """Score the latest 5-minute V2 window against the personal baseline.

        Two inputs feed the score, in priority order:

        1. The newest persisted V2 feature window that starts inside the
           analysis window (``behavior_feature_windows``, 5-minute rows). This
           is the same vector the ML classifier sees, so the deviation and the
           prediction describe the same behaviour.
        2. A row derived from the raw activity events of the window, used only
           when no persisted window exists (e.g. an analysis running before the
           telemetry rollup caught up).

        The v1 detector this replaces compared a hand-rolled 30-minute row
        against 30-minute statistics; the feature vocabulary had since moved to
        V2 5-minute windows, so only ``idle_ratio`` matched and every score was
        capped near zero. Scoring the stored window keeps the bases identical.

        Returns ``(items, summary)``: exactly one ``behavior_deviation`` item
        (an info item when the baseline cannot support a score) and the
        contract's ``DeviationSummary`` dict, or None when no summary applies.
        """
        if baseline is None:
            return (
                [
                    EvidenceItem(
                        metric="behavior_deviation",
                        value=0.0,
                        baseline=None,
                        severity="info",
                        confidence=1.0,
                        source="welford_baseline",
                        human_readable="基线尚在建立中（尚无数据）",
                    )
                ],
                None,
            )

        if not baseline.has_sufficient_data():
            return (
                [
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
                ],
                None,
            )

        row = await self._latest_feature_row(user_id, window_start)
        if row is not None:
            # Bucket the stored window in the baseline's own business timezone;
            # using the naive UTC hour would compare a 08:00-local window
            # against the 00:00-local bucket.
            hour, dow = _local_bucket(window_start, baseline)
        else:
            row = _build_pseudo_row(events, window_start)
            hour, dow = window_start.hour, window_start.weekday()

        if row is None:
            return (
                [
                    EvidenceItem(
                        metric="behavior_deviation",
                        value=0.0,
                        baseline=None,
                        severity="info",
                        confidence=1.0,
                        source="welford_baseline",
                        human_readable="数据不足以计算基线偏差",
                    )
                ],
                None,
            )

        row = {**row, "hour_of_day": hour, "day_of_week": dow}
        detector = DeviationDetector(baseline)
        summary = detector.to_summary(row)
        if summary is None:
            # The window carries no feature the baseline has ever observed —
            # nothing honest to report, so this is the "insufficient" path.
            return (
                [
                    EvidenceItem(
                        metric="behavior_deviation",
                        value=0.0,
                        baseline=None,
                        severity="info",
                        confidence=0.8,
                        source="welford_baseline",
                        human_readable=f"当前时段({hour}:00 周{dow})基线数据不足，暂无法计算偏差",
                    )
                ],
                None,
            )

        overall = float(summary["overall_score"])
        sev = _deviation_severity(str(summary["severity"]))
        conf = _deviation_to_confidence(overall)

        if sev == "info":
            return (
                [
                    EvidenceItem(
                        metric="behavior_deviation",
                        value=overall,
                        baseline=0.0,
                        severity="info",
                        confidence=conf,
                        source="welford_baseline",
                        human_readable="行为模式与基线一致，无显著偏差",
                    )
                ],
                summary,
            )

        # Name the feature that actually drove the score, in Chinese.
        features = summary.get("features") or []
        dev_detail = ""
        if features:
            top = features[0]
            dev_detail = (
                f"，主要偏离: {top['label']}(稳健 z={top['robust_z']:.1f}"
                f"，当前 {top['value']:g} / 期望 {top['expected']:g})"
            )

        return (
            [
                EvidenceItem(
                    metric="behavior_deviation",
                    value=overall,
                    baseline=0.0,
                    severity=sev,
                    confidence=conf,
                    source="welford_baseline",
                    human_readable=(
                        f"行为模式偏离基线 {overall:.1f} 个稳健标准差（{sev}）{dev_detail}"
                    ),
                )
            ],
            summary,
        )

    async def _latest_feature_row(
        self,
        user_id: int,
        window_start: datetime,
    ) -> dict[str, Any] | None:
        """Newest persisted V2 feature window starting at or after *window_start*.

        One bounded query: the analysis window is widened to 24 hours so a
        panel run over a short slice still finds the most recent window.
        Returns None when the repository is unavailable or has no such row —
        both are normal (tests, pre-rollup analyses) and fall back to the
        event-derived row rather than failing the pipeline.

        ``hour_of_day`` / ``day_of_week`` are NOT derived here; the caller
        buckets in the baseline's timezone because only it knows which
        timezone the stored statistics were bucketed in.
        """
        if self._telemetry_repo is None:
            return None
        end = window_start + timedelta(hours=24)
        try:
            windows = await self._telemetry_repo.list_feature_windows_in_range(
                user_id, window_start, end
            )
        except Exception as exc:
            logger.warning(
                "Deviation baseline query failed (falling back to event row): {}", exc
            )
            return None
        if not windows:
            return None
        row = dict(windows[-1])
        row["features_json"] = row.get("features_json")
        # The repository leaves the feature vector in ``features_json``
        # (and/or the fNN columns); ``DeviationDetector`` reads either.
        return row

    # ══════════════════════════════════════════════════════════════════════
    # TMT (Temporal Motivation Theory) decomposition
    # ══════════════════════════════════════════════════════════════════════

    async def _build_tmt_inputs(
        self,
        *,
        events: list[ActivityEvent],
        deviation_summary: dict[str, Any] | None,
        window_end: datetime,
        user_id: int,
    ) -> TmtInputs:
        """Collect the behavioural proxies ``compute_tmt`` consumes.

        Every read is best-effort: a missing repository or a failed query
        leaves that signal ``None``, which ``compute_tmt`` turns into the
        documented neutral prior rather than a fabricated value.  The
        impulsiveness proxy reuses the deviation summary the bundle already
        carries instead of re-deriving switch statistics.
        """
        feedback = await self._load_focus_feedback(user_id, window_end)
        tasks = await self._load_open_tasks(user_id)
        value_task = _select_value_task(tasks, window_end)

        productive_share, _ = _productive_and_leisure_shares(events)

        return TmtInputs(
            feedback_scores=_recent_feedback_scores(feedback),
            task_priority=(
                float(value_task.priority) if value_task is not None else None
            ),
            task_count=len(tasks),
            productive_share=productive_share,
            deviation_z_mean=deviation_z_mean(deviation_summary),
            hours_to_deadline=_hours_to_deadline(tasks, window_end),
        )

    async def _load_focus_feedback(
        self, user_id: int, window_end: datetime
    ) -> list[dict[str, Any]]:
        """Self-reported focus feedback from the last 14 days, or [].

        Older rows are dropped because the expectancy proxy is meant to
        reflect the user's *recent* success expectation, not a lifetime mean.
        """
        if self._telemetry_repo is None:
            return []
        lister = getattr(self._telemetry_repo, "list_focus_feedback", None)
        if lister is None:
            return []
        try:
            rows = await lister(user_id)
        except Exception as exc:
            logger.debug(
                "Focus feedback unavailable for TMT expectancy: {}", _safe_error(exc)
            )
            return []
        if not isinstance(rows, list):
            return []
        cutoff = window_end - timedelta(days=_TMT_FEEDBACK_LOOKBACK_DAYS)
        return [
            row
            for row in rows
            if isinstance(row, dict) and _feedback_is_recent(row, cutoff)
        ]

    async def _load_open_tasks(self, user_id: int) -> list[Any]:
        """All open tasks (pending / in_progress), or [] when unavailable.

        The whole list is needed: the value proxy asks for the most important
        task *due within 7 days*, which is not necessarily in a top-N ranking
        ordered by deadline.
        """
        if self._task_repo is None:
            return []
        lister = getattr(self._task_repo, "list_tasks", None)
        if lister is None:
            return []
        try:
            rows = await lister(user_id)
        except Exception as exc:
            logger.debug("Task list unavailable for TMT proxies: {}", _safe_error(exc))
            return []
        if not isinstance(rows, list):
            return []
        return [
            task
            for task in rows
            if getattr(task, "status", None) in {"pending", "in_progress"}
        ]

    @staticmethod
    def _tmt_items(tmt: TmtBreakdown) -> list[EvidenceItem]:
        """One info-level EvidenceItem per TMT component (exactly four).

        These are always ``info`` so they never inflate the anomaly count; the
        compression layer keeps them only when they deviate from a baseline or
        match a recent intervention keyword, and the catalog carries every id
        regardless.  ``value`` is the component score so the critic can check
        an expert's citation against a real number.
        """
        components = tmt.components.as_mapping()
        return [
            EvidenceItem(
                metric=f"tmt_{key}",
                value=round(component.score, 4),
                baseline=None,
                severity="info",
                confidence=_TMT_CONFIDENCE,
                source=component.source,
                human_readable=(
                    f"TMT {component.label} {component.score:.2f}"
                    f"（{_tmt_level_zh(component.score)}；{component.source}）"
                ),
            )
            for key, component in components.items()
        ]

    async def _load_baseline(self, user_id: int) -> BaselineModel | None:
        """Load the user's ``BaselineModel`` via the baseline repository.

        Returns None if no baseline has been built yet for this user.
        Delegates fetching + JSON parsing to ``BaselineRepository`` so no raw
        SQL lives in the services layer (the "latest wins" ordering that
        guards against stale models — review H1 — is owned by the repo).
        """
        return await self._baseline_repo.get_latest(user_id)

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
        # Each compare_windows(lid) is independent, so gather the ≤5 calls
        # concurrently instead of awaiting them one at a time.
        effect_data: dict[str, str] = {}
        if self._effectiveness_service is not None and logs:
            recent_lids = [lid for log_entry in logs[-5:] if (lid := log_entry.get("id", ""))]
            if recent_lids:
                svc = self._effectiveness_service
                results = await asyncio.gather(
                    *(svc.compare_windows(lid) for lid in recent_lids),
                    return_exceptions=True,
                )
                for lid, report in zip(recent_lids, results, strict=True):
                    if isinstance(report, BaseException):
                        logger.debug("No effectiveness data for intervention {}", lid)
                        continue
                    if report.has_data:
                        summary = self._format_effectiveness_summary(report)
                        if summary:
                            effect_data[lid] = summary

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


def _local_bucket(window_start: datetime, baseline: BaselineModel) -> tuple[int, int]:
    """Local (hour, weekday) of *window_start* in the baseline's timezone.

    ``BaselineModel`` buckets its statistics in the user's business timezone;
    a detector that read ``window_start.hour`` from a UTC timestamp would
    compare an 08:00 local window against the 00:00 local bucket (an 8-hour
    offset for Asia/Shanghai). Naive timestamps are treated as UTC, matching
    ``domain/baseline.py``.
    """
    tz = resolve_timezone(baseline.timezone)
    if window_start.tzinfo is None:
        window_start = window_start.replace(tzinfo=UTC)
    local = window_start.astimezone(tz)
    return local.hour, local.weekday()


def _safe_error(exc: BaseException) -> str:
    """Short, credential-free description of an exception for debug logs."""
    return type(exc).__name__


def _tmt_level_zh(score: float) -> str:
    """Coarse Chinese qualifier for a TMT component score."""
    if score >= 0.66:
        return "偏高"
    if score >= 0.34:
        return "中等"
    return "偏低"


def _productive_and_leisure_shares(
    events: list[ActivityEvent],
) -> tuple[float | None, float | None]:
    """Active-time shares of productive vs entertainment/social categories.

    Uses the built-in :class:`AppClassifier` on the process name.  Idle time
    is excluded from the denominator: a user away from the keyboard tells us
    nothing about which apps they were working in.  Returns ``(None, None)``
    when there is no active time to divide by.
    """
    non_idle = [event for event in events if not event.data.is_idle]
    total = sum(event.duration_s for event in non_idle)
    if total <= 0.0:
        return None, None

    classifier = AppClassifier()
    productive_s = 0.0
    leisure_s = 0.0
    for event in non_idle:
        category = classifier.classify(
            event.data.process_name, getattr(event.data, "window_title", "") or ""
        )
        if category in {"code", "document", "browser_work"}:
            productive_s += event.duration_s
        elif category in {"entertainment", "social"}:
            leisure_s += event.duration_s
    return productive_s / total, leisure_s / total


def _parse_feedback_time(row: dict[str, Any]) -> datetime | None:
    """Parse a feedback row's ``created_at`` (ISO string or datetime)."""
    raw = row.get("created_at")
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _feedback_is_recent(row: dict[str, Any], cutoff: datetime) -> bool:
    """True when the feedback row is at or after *cutoff*.

    An unparseable timestamp is dropped: silently keeping it would let an
    arbitrarily old self-report drive the expectancy proxy.
    """
    created = _parse_feedback_time(row)
    return created is not None and created >= cutoff


def _recent_feedback_scores(rows: list[dict[str, Any]]) -> list[float]:
    """Self-reported focus scores (1-5) from the given feedback rows."""
    return [
        float(row["score"])
        for row in rows
        if isinstance(row.get("score"), (int, float))
        and not isinstance(row.get("score"), bool)
        and 1.0 <= float(row["score"]) <= 5.0
    ]


def _select_value_task(tasks: list[Any], reference: datetime) -> Any | None:
    """Most important open task due within 7 days, else most important overall.

    ``Task.priority`` is 1 (lowest) … 5 (highest), so "most important" is the
    highest number.  Ties break on the soonest deadline, then the stable id,
    so the choice is deterministic.
    """
    horizon = reference + timedelta(days=_TMT_DUE_SOON_DAYS)
    due_soon = [
        task
        for task in tasks
        if isinstance(task.deadline_utc, datetime) and task.deadline_utc <= horizon
    ]
    pool = due_soon or tasks
    if not pool:
        return None
    return min(
        pool,
        key=lambda task: (
            -float(task.priority),
            task.deadline_utc or datetime.max.replace(tzinfo=UTC),
            str(task.id),
        ),
    )


def _hours_to_deadline(tasks: list[Any], reference: datetime) -> float | None:
    """Hours from *reference* to the nearest open-task deadline.

    Overdue deadlines clamp to 0 in ``compute_tmt``; tasks without a deadline
    are ignored here (they carry no delay information).  Returns None when no
    open task has a deadline.
    """
    deadlines = [
        task.deadline_utc
        for task in tasks
        if isinstance(task.deadline_utc, datetime)
        and task.deadline_utc.tzinfo is not None
    ]
    if not deadlines:
        return None
    return (min(deadlines) - reference).total_seconds() / 3600.0


def _build_pseudo_row(events: list[ActivityEvent], window_start: datetime) -> dict[str, Any] | None:
    """Approximate a V2 feature row from raw events (fallback path only).

    Used when no persisted 5-minute feature window covers the analysis window
    (e.g. a panel run before the telemetry rollup caught up). It computes the
    subset of the V2 vocabulary that raw events can honestly support:
    confirmed app switches, the longest active segment's share of observed
    time, idle / active shares and the dominant app's share. Interaction-rate
    and browser features need collector buckets and browser heartbeats, which
    this path does not have — ``DeviationDetector`` skips any feature the
    baseline has no data for and renormalises the remaining weights, so a
    partial row degrades rather than lying.

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

    total_dur = sum(ev.duration_s for ev in events)
    idle_dur = sum(ev.duration_s for ev in events if ev.data.is_idle)
    active_dur = total_dur - idle_dur
    longest_segment = max(app_durations.values()) if app_durations else 0.0
    dominant_app = max(app_durations.values()) if app_durations else 0.0

    return {
        "hour_of_day": window_start.hour,
        "day_of_week": window_start.weekday(),
        "window_start": window_start.isoformat(),
        "app_switch_count": float(count_confirmed_switches(non_idle)),
        "longest_segment_ratio": longest_segment / total_dur if total_dur > 0 else 0.0,
        "idle_ratio": idle_dur / total_dur if total_dur > 0 else 0.0,
        "active_seconds_ratio": active_dur / total_dur if total_dur > 0 else 0.0,
        "top_app_ratio": dominant_app / active_dur if active_dur > 0 else 0.0,
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
