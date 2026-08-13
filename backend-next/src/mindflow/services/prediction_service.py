"""Unified ML prediction service bridging feature windows and online inference.

This is the single source of truth for online ML inference in MindFlow.
All consumers (Panel EvidenceBundle, Telemetry API, Chat evidence tools)
use this service so predictions are consistent across the application.

Key design:
  - Reads pre-computed v2 feature windows from the database (5-min windows,
    privacy-preserving, no PII in features).
  - One batch query, one ``predict_proba`` call — no N+1 against DB or model.
  - Strict schema validation: feature count, order, finite values all checked.
  - Returns a frozen ``FocusPrediction`` dataclass for every possible state
    (ready, no_model, no_data, stale, schema_mismatch, inference_error).
  - Never raises — all errors are captured as status values in the result.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from loguru import logger

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.domain.prediction import (
    MIN_COVERAGE_RATIO,
    STALE_THRESHOLD_S,
    FocusPrediction,
    FocusPredictionStatus,
)
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.train.models.manager import ModelManager

# How many seconds of feature windows to fetch per ``predict_latest`` call
_LATEST_LOOKBACK_S: float = 7200.0  # 2 hours

# Maximum windows per batch ``predict_proba`` call
_MAX_PREDICT_WINDOWS: int = 500


class FocusPredictionService:
    """Unified ML prediction service.

    Stateless — a single shared instance is safe for all consumers.
    The ``model_manager`` must be attached before predictions can be made;
    until then all calls return ``status=no_model``.

    Args:
        telemetry_repository: Repository for reading v2 feature windows.
        model_manager: Active model manager, or None (no_model state).
            Can be attached later via ``attach_model_manager()`` to
            support lazy loading after service construction.
    """

    def __init__(
        self,
        telemetry_repository: TelemetryRepository,
        model_manager: ModelManager | None = None,
    ) -> None:
        self._telemetry_repo = telemetry_repository
        self._model_manager = model_manager

    def attach_model_manager(self, model_manager: ModelManager) -> None:
        """Attach or replace the active model manager.

        Called during application startup when the model directory is found,
        or when a new model is hot-loaded at runtime.
        """
        self._model_manager = model_manager

    def detach_model_manager(self) -> None:
        """Detach the active manager so subsequent calls return ``no_model``."""
        self._model_manager = None

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    async def predict_latest(
        self,
        user_id: int = 1,
        now: datetime | None = None,
    ) -> FocusPrediction:
        """Predict focus for the most recent feature windows.

        Fetches the last 2 hours of v2 feature windows for *user_id*,
        runs batch ``predict_proba``, and returns an aggregated prediction.

        Args:
            user_id: The user to predict for.
            now: Override the "current time" (used in tests). Defaults to UTC now.

        Returns:
            A ``FocusPrediction`` describing the result.
        """
        now = now or datetime.now(UTC)

        # ── Guard: no model loaded ──────────────────────────────────────
        model_manager = self._model_manager
        if model_manager is None:
            return FocusPrediction(
                status="no_model",
                reason="未加载 ML 模型，请先训练",
            )

        # ── Guard: model classifier not fitted ──────────────────────────
        if not bool(getattr(model_manager.classifier, "_is_fitted", False)):
            return FocusPrediction(
                status="no_model",
                reason="分类器尚未训练完成",
                model_version=model_manager.current_version_tag,
            )

        # ── Fetch feature windows (range-bounded, never full history) ────
        lookback_start = now - timedelta(seconds=_LATEST_LOOKBACK_S)
        try:
            all_windows = await self._telemetry_repo.list_feature_windows_in_range(
                user_id=user_id,
                start=lookback_start,
                end=now,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
            )
        except Exception as exc:
            logger.warning("FocusPredictionService: DB query failed: {}", exc)
            return FocusPrediction(
                status="inference_error",
                reason=f"特征窗口查询失败：{exc}",
                model_version=model_manager.current_version_tag,
            )

        # Filter to lookback window (secondary guard; the query is already bounded)
        windows = [w for w in all_windows if _window_ends_after(w, lookback_start)]

        if not windows:
            return FocusPrediction(
                status="no_data",
                reason="在最近 2 小时内未找到 v2 特征窗口",
                model_version=model_manager.current_version_tag,
            )

        model_manager = self._model_manager
        if model_manager is None or not bool(
            getattr(model_manager.classifier, "_is_fitted", False)
        ):
            return FocusPrediction(status="no_model", reason="未加载 ML 模型")

        # Offload CPU-bound numpy/sklearn inference to a worker thread so the
        # event loop is not blocked for concurrent HTTP requests (audit report).
        return await asyncio.to_thread(
            self._predict_from_windows, windows, now, model_manager
        )

    async def predict_range(
        self,
        user_id: int,
        start: datetime,
        end: datetime,
        now: datetime | None = None,
    ) -> FocusPrediction:
        """Predict focus for a specific time range.

        Useful for EvidenceBundleBuilder to get ML evidence for a
        Panel analysis window.

        Args:
            user_id: The user to predict for.
            start: Start of the analysis window (UTC, timezone-aware).
            end: End of the analysis window (UTC, timezone-aware).
            now: Override "current time" for staleness checks.

        Returns:
            A ``FocusPrediction`` covering the specified range.
        """
        now = now or datetime.now(UTC)

        model_manager = self._model_manager
        if model_manager is None:
            return FocusPrediction(status="no_model", reason="未加载 ML 模型")

        if not bool(getattr(model_manager.classifier, "_is_fitted", False)):
            return FocusPrediction(
                status="no_model",
                reason="分类器尚未训练完成",
                model_version=model_manager.current_version_tag,
            )

        try:
            all_windows = await self._telemetry_repo.list_feature_windows_in_range(
                user_id=user_id,
                start=start,
                end=end,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
            )
        except Exception as exc:
            logger.warning("FocusPredictionService: range DB query failed: {}", exc)
            return FocusPrediction(
                status="inference_error",
                reason=f"特征窗口查询失败：{exc}",
                model_version=model_manager.current_version_tag,
            )

        # Filter to requested time range (secondary guard; the query is already bounded)
        windows = [
            w for w in all_windows
            if _window_overlaps_range(w, start, end)
        ]

        if not windows:
            return FocusPrediction(
                status="no_data",
                reason=f"在 {start.isoformat()} ~ {end.isoformat()} 范围内未找到 v2 特征窗口",
                model_version=model_manager.current_version_tag,
            )

        model_manager = self._model_manager
        if model_manager is None or not bool(
            getattr(model_manager.classifier, "_is_fitted", False)
        ):
            return FocusPrediction(status="no_model", reason="未加载 ML 模型")

        return await asyncio.to_thread(
            self._predict_from_windows, windows, now, model_manager
        )

    # ══════════════════════════════════════════════════════════════════════
    # Internal: predict from window rows
    # ══════════════════════════════════════════════════════════════════════

    def _predict_from_windows(
        self,
        windows: Sequence[dict[str, Any]],
        now: datetime,
        model_manager: ModelManager,
    ) -> FocusPrediction:
        """Run batch ML inference on a list of feature window rows.

        All degenerate states (schema mismatch, NaN features, inference
        errors) are captured as ``FocusPrediction`` status values — this
        method never raises.
        """
        model_version = model_manager.current_version_tag

        # ── Enforce window limit (before matrix construction) ───────────
        # Bounded first so parsed rows, matrix rows, and the reported
        # ``window_count`` always agree — the matrix is never built larger
        # than _MAX_PREDICT_WINDOWS. The repository returns windows ascending
        # by start, so keeping the tail preserves the most recent windows.
        if len(windows) > _MAX_PREDICT_WINDOWS:
            windows = windows[-_MAX_PREDICT_WINDOWS:]

        # ── Build feature matrix ────────────────────────────────────────
        parsed: list[dict[str, float]] = []
        newest_window_end: datetime | None = None
        newest_window_start: str | None = None
        for row in windows:
            try:
                features = json.loads(str(row.get("features_json", "{}")))
                if not isinstance(features, dict):
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            feature_row: dict[str, float] = {}
            for name in V2_FEATURE_NAMES:
                value = features.get(name)
                fv = _finite_float(value)
                feature_row[name] = fv
            parsed.append(feature_row)

            # Track newest window end for staleness
            w_end = features.get("window_end_utc") or row.get("window_end_utc")
            if w_end:
                try:
                    parsed_end = datetime.fromisoformat(str(w_end).replace("Z", "+00:00"))
                    if parsed_end.tzinfo is None:
                        parsed_end = parsed_end.replace(tzinfo=UTC)
                    if newest_window_end is None or parsed_end > newest_window_end:
                        newest_window_end = parsed_end
                        w_start = row.get("window_start_utc")
                        if w_start:
                            newest_window_start = str(w_start)
                except (ValueError, TypeError):
                    pass

        if not parsed:
            return FocusPrediction(
                status="no_data",
                reason="特征窗口解析后全部为空或无效",
                model_version=model_version,
            )

        window_count = len(parsed)

        # ── Build numpy matrix ──────────────────────────────────────────
        try:
            matrix = np.asarray(
                [[row[name] for name in V2_FEATURE_NAMES] for row in parsed],
                dtype=np.float64,
            )
        except (KeyError, ValueError, TypeError) as exc:
            return FocusPrediction(
                status="schema_mismatch",
                reason=f"特征矩阵构建失败（列名不匹配）：{exc}",
                model_version=model_version,
                window_count=window_count,
            )

        # ── Validate schema and feature names ──────────────────────────
        if matrix.ndim != 2 or matrix.shape[1] != len(V2_FEATURE_NAMES):
            return FocusPrediction(
                status="schema_mismatch",
                reason=f"特征矩阵形状异常：{matrix.shape}（期望 24 列）",
                model_version=model_version,
                window_count=window_count,
            )

        # ── Check model feature names ────────────────────────────────────
        model_feature_names = getattr(model_manager.classifier, "feature_names_", None)
        # Compare by content: the training pipeline stores feature_names_ as
        # ``list(V2_FEATURE_NAMES)`` while V2_FEATURE_NAMES itself is a tuple,
        # so a container-type comparison would reject every real model.
        if (
            model_feature_names is not None
            and isinstance(model_feature_names, (list, tuple))
            and tuple(model_feature_names) != tuple(V2_FEATURE_NAMES)
        ):
            return FocusPrediction(
                status="schema_mismatch",
                reason="模型特征名称与当前 V2_FEATURE_NAMES 不匹配",
                model_version=model_version,
                window_count=window_count,
            )

        # ── Validate no NaN / Inf ───────────────────────────────────────
        if not np.all(np.isfinite(matrix)):
            return FocusPrediction(
                status="inference_error",
                reason="特征矩阵包含 NaN 或 Infinite 值",
                model_version=model_version,
                window_count=window_count,
            )

        # ── Run prediction ──────────────────────────────────────────────
        try:
            probabilities = model_manager.classifier.predict_proba(matrix)
        except Exception as exc:
            logger.warning("FocusPredictionService: predict_proba failed: {}", exc)
            return FocusPrediction(
                status="inference_error",
                reason=f"模型推理失败：{exc}",
                model_version=model_version,
                window_count=window_count,
            )

        # ── Aggregate over windows (wrapped for "never raises") ────────
        # proba shape: (n_samples, 2) — columns [distraction, focus]
        try:
            focus_probas = probabilities[:, 1]
            mean_focus = float(np.mean(focus_probas))
            mean_uncertainty = float(np.mean(1.0 - np.abs(2.0 * focus_probas - 1.0)))
            distracted_ratio = float(np.mean(focus_probas < 0.5))
        except Exception as exc:
            return FocusPrediction(
                status="inference_error",
                reason=f"预测聚合失败：{exc}",
                model_version=model_version,
                window_count=window_count,
            )

        # ── Compute coverage ────────────────────────────────────────────
        # Expected windows based on 5-min window size and time span
        coverage_ratio = 1.0
        if newest_window_end is not None and window_count > 0:
            covered_seconds = (
                newest_window_end - now + timedelta(seconds=_LATEST_LOOKBACK_S)
            ).total_seconds()
            expected = max(1, int(covered_seconds / 300))
            coverage_ratio = min(1.0, window_count / expected)

        # ── Compute data age ────────────────────────────────────────────
        data_age_s: float | None = None
        if newest_window_end is not None:
            data_age_s = (now - newest_window_end).total_seconds()

        # ── Compute top factors ─────────────────────────────────────────
        top_factors: list[dict[str, float | str]] = []
        try:
            mean_vector = np.mean(matrix, axis=0)
            importances = model_manager.classifier.get_feature_importance()
            top_factors = [
                {
                    "feature": name,
                    "value": round(float(mean_vector[idx]), 6),
                    "importance": round(float(importances.get(name, 0.0)), 6),
                }
                for idx, name in enumerate(V2_FEATURE_NAMES)
            ]
            top_factors.sort(
                key=lambda f: float(f["importance"]) * max(abs(float(f["value"])), 0.01),
                reverse=True,
            )
        except Exception:
            logger.debug("Failed to compute top features, returning empty")
            top_factors = []

        # ── Determine status ────────────────────────────────────────────
        status: FocusPredictionStatus = "ready"
        reason = ""
        if data_age_s is not None and data_age_s > STALE_THRESHOLD_S:
            status = "stale"
            reason = (
                f"数据已过期（{data_age_s:.0f} 秒前最后的窗口，"
                f"阈值 {STALE_THRESHOLD_S:.0f} 秒）"
            )
        elif coverage_ratio < MIN_COVERAGE_RATIO:
            status = "stale"
            reason = f"数据覆盖率不足（{coverage_ratio:.1%}，阈值 {MIN_COVERAGE_RATIO:.0%}）"

        # Build top 3
        top_3 = top_factors[:3]

        return FocusPrediction(
            status=status,
            focus_probability=round(mean_focus, 6),
            uncertainty=round(mean_uncertainty, 6),
            distracted_window_ratio=round(distracted_ratio, 6),
            window_count=window_count,
            coverage_ratio=round(coverage_ratio, 4),
            data_age_s=round(data_age_s, 1) if data_age_s is not None else None,
            model_version=model_version,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            newest_window_start_utc=newest_window_start,
            top_factors=top_3,
            explanation_method="global_importance_times_observation",
            reason=reason,
        )

    async def check_health(self) -> dict[str, Any]:
        """Return a health-check summary of the prediction service.

        Used by the ``/health`` endpoint to expose ML subsystem status.

        Returns:
            A dict with active/candidate versions, model status, and
            recent inference status.
        """
        model_version = (
            self._model_manager.current_version_tag
            if self._model_manager is not None
            else None
        )
        is_ready = (
            self._model_manager is not None
            and bool(getattr(self._model_manager.classifier, "_is_fitted", False))
        )
        return {
            "status": "ready" if is_ready else "no_model",
            "model_version": model_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }


def _finite_float(value: object) -> float:
    """Convert a value to float, returning 0.0 for non-finite or non-numeric."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _parse_window_end(row: dict[str, Any]) -> datetime | None:
    """Parse a feature window's end time from its row dict."""
    w_end = row.get("window_end_utc")
    if not w_end:
        return None
    try:
        dt = datetime.fromisoformat(str(w_end).replace("Z", "+00:00"))
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    except (ValueError, TypeError):
        return None


def _window_ends_after(row: dict[str, Any], threshold: datetime) -> bool:
    """Return True if the feature window ends after *threshold*."""
    end = _parse_window_end(row)
    return end is not None and end > threshold


def _window_overlaps_range(row: dict[str, Any], start: datetime, end: datetime) -> bool:
    """Return True if the feature window overlaps [start, end)."""
    w_end = _parse_window_end(row)
    w_start = row.get("window_start_utc")
    if not w_start or not w_end:
        return False
    try:
        ws = datetime.fromisoformat(str(w_start).replace("Z", "+00:00"))
        we = (
            w_end
            if isinstance(w_end, datetime)
            else datetime.fromisoformat(str(w_end).replace("Z", "+00:00"))
        )
        if ws.tzinfo is None:
            ws = ws.replace(tzinfo=UTC)
        if we.tzinfo is None:
            we = we.replace(tzinfo=UTC)
        return ws < end and we > start
    except (ValueError, TypeError):
        return False
