"""Privacy-preserving telemetry orchestration."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.domain.baseline import BaselineModel
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.baseline import BaselineRepository
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.services.telemetry_features import (
    FEATURE_SCHEMA_VERSION,
    build_v2_feature_window,
)
from mindflow.time_utils import TimezoneLike, resolve_timezone, utc_today
from mindflow.train.v2 import V2_FEATURE_NAMES

_DEFAULTS: dict[str, Any] = {
    "input_telemetry_enabled": False,
    "browser_tracking_enabled": False,
    "interaction_retention_days": 7,
    "activity_retention_days": 30,
}

_PAIRING_CODE_TTL_S = 300

# Conditional baseline backfill horizon: at most this many business days of
# stored V2 windows feed a rebuild (matches the feature-window retention cut).
_BASELINE_BACKFILL_DAYS: Final = 180


@dataclass(frozen=True, slots=True)
class BaselineRebuildResult:
    """Outcome of one conditional baseline backfill run (Todo 9 seam).

    ``rebuilt`` is True only when the baseline row was actually replaced; an
    existing V2 baseline yields ``skipped_v2`` with ``rebuilt`` False so a
    caller can never mistake a no-op for a rebuild.
    """

    rebuilt: bool
    reason: Literal["missing", "schema_mismatch", "skipped_v2"]
    windows_loaded: int
    samples: int
    cutoff_utc: datetime


def _as_utc(value: Any) -> datetime:
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)


class TelemetryService:
    def __init__(
        self,
        repository: TelemetryRepository,
        preferences_repository: PreferencesRepository,
        data_dir: Path,
        activity_repository: SQLAlchemyActivityRepository | None = None,
        prediction_service: FocusPredictionService | None = None,
        baseline_repository: BaselineRepository | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._repository = repository
        self._preferences_repository = preferences_repository
        self._data_dir = data_dir
        self._activity_repository = activity_repository
        self._pairing_codes: dict[str, datetime] = {}
        self._input_watcher: Any = None
        self._model_manager: Any = None
        self._prediction_service = prediction_service
        # Baseline refresh during rollup is only active when both the
        # repository and a session factory are wired (the application wires
        # them; legacy/unit constructions leave the rollup window-only).
        self._baseline_repository = baseline_repository
        self._session_factory = session_factory

    def attach_input_watcher(self, watcher: Any) -> None:
        self._input_watcher = watcher

    def attach_model_manager(self, model_manager: Any) -> None:
        self._model_manager = model_manager

    async def get_preferences(self, user_id: int = 1) -> dict[str, Any]:
        preferences = await self._preferences_repository.get(user_id)
        telemetry = preferences.get("telemetry", {})
        return {**_DEFAULTS, **telemetry}

    async def patch_preferences(
        self,
        updates: dict[str, Any],
        user_id: int = 1,
    ) -> dict[str, Any]:
        current = await self._preferences_repository.get(user_id)
        telemetry = {**_DEFAULTS, **current.get("telemetry", {}), **updates}
        telemetry["interaction_retention_days"] = min(
            max(int(telemetry["interaction_retention_days"]), 1), 30
        )
        telemetry["activity_retention_days"] = min(
            max(int(telemetry["activity_retention_days"]), 7), 90
        )
        current["telemetry"] = telemetry
        await self._preferences_repository.set(user_id, current)
        if self._input_watcher is not None:
            if telemetry["input_telemetry_enabled"]:
                await self._input_watcher.start()
            else:
                await self._input_watcher.stop()
        return telemetry

    async def get_status(self, user_id: int = 1) -> dict[str, Any]:
        preferences = await self.get_preferences(user_id)
        status = await self._repository.get_status(user_id, utc_today())
        database_path = self._data_dir / "mindflow.db"
        watcher_status = (
            self._input_watcher.status if self._input_watcher is not None else "unavailable"
        )
        return {
            "preferences": preferences,
            "input_watcher_status": watcher_status,
            "database_size_bytes": database_path.stat().st_size
            if database_path.exists()
            else 0,
            **status,
        }

    def _cleanup_expired_pairing_codes(self, now: datetime) -> None:
        for code, expires_at in list(self._pairing_codes.items()):
            if expires_at <= now:
                del self._pairing_codes[code]

    async def create_pairing_code(self, user_id: int = 1) -> dict[str, Any]:
        now = datetime.now(UTC)
        self._cleanup_expired_pairing_codes(now)
        code = f"{secrets.randbelow(1_000_000):06d}"
        expires_at = now + timedelta(seconds=_PAIRING_CODE_TTL_S)
        self._pairing_codes[code] = expires_at
        await self.patch_preferences({"browser_tracking_enabled": True}, user_id)
        return {"code": code, "expires_at": expires_at.isoformat()}

    async def pair_browser(self, code: str, user_id: int = 1) -> str | None:
        expires_at = self._pairing_codes.pop(code, None)
        if expires_at is None or expires_at < datetime.now(UTC):
            return None
        token = secrets.token_urlsafe(32)
        await self._repository.save_browser_token(user_id, self._hash_token(token))
        return token

    async def verify_browser_token(self, token: str) -> bool:
        if not token:
            return False
        return await self._repository.verify_browser_token(self._hash_token(token))

    async def save_authenticated_browser_heartbeat(
        self,
        token: str,
        *,
        timestamp_utc: datetime,
        duration_s: float,
        browser_name: str,
        domain: str,
        audible: bool,
        incognito: bool,
        user_id: int = 1,
    ) -> dict[str, Any] | None:
        """Authenticate, touch token usage, and save heartbeat in one write transaction."""
        if not token:
            return None

        heartbeat: dict[str, Any] | None = None
        if incognito:
            response: dict[str, Any] = {"ignored": True, "reason": "incognito"}
        else:
            preferences = await self.get_preferences(user_id)
            if not preferences["browser_tracking_enabled"]:
                response = {"ignored": True, "reason": "disabled"}
            else:
                normalized_domain = self.normalize_domain(domain)
                if not normalized_domain:
                    response = {"ignored": True, "reason": "invalid_domain"}
                else:
                    normalized_browser = browser_name.lower()
                    heartbeat = {
                        "user_id": user_id,
                        "timestamp_utc": timestamp_utc,
                        "duration_s": min(max(duration_s, 1.0), 60.0),
                        "browser_name": normalized_browser,
                        "domain": normalized_domain,
                        "audible": audible,
                        "context_key": f"{normalized_browser}:{normalized_domain}",
                    }
                    response = {
                        "ignored": False,
                        "domain": normalized_domain,
                    }

        authorized, segment = await self._repository.save_authenticated_browser_heartbeat(
            self._hash_token(token),
            heartbeat=heartbeat,
        )
        if not authorized:
            return None
        if heartbeat is not None:
            response["segment"] = segment
        return response

    async def save_browser_heartbeat(
        self,
        *,
        timestamp_utc: datetime,
        duration_s: float,
        browser_name: str,
        domain: str,
        audible: bool,
        incognito: bool,
        user_id: int = 1,
    ) -> dict[str, Any]:
        if incognito:
            return {"ignored": True, "reason": "incognito"}
        preferences = await self.get_preferences(user_id)
        if not preferences["browser_tracking_enabled"]:
            return {"ignored": True, "reason": "disabled"}
        normalized_domain = self.normalize_domain(domain)
        if not normalized_domain:
            return {"ignored": True, "reason": "invalid_domain"}
        context_key = f"{browser_name.lower()}:{normalized_domain}"
        result = await self._repository.save_browser_heartbeat(
            user_id=user_id,
            timestamp_utc=timestamp_utc,
            duration_s=min(max(duration_s, 1.0), 60.0),
            browser_name=browser_name.lower(),
            domain=normalized_domain,
            audible=audible,
            context_key=context_key,
        )
        return {"ignored": False, "domain": normalized_domain, "segment": result}



    async def rollup_feature_windows(
        self,
        start: datetime,
        end: datetime,
        user_id: int = 1,
    ) -> int:
        if self._activity_repository is None:
            return 0

        events = await self._activity_repository.query_range(user_id, start, end)
        previous_event = await self._activity_repository.last_event_before(user_id, start)
        if (
            previous_event is not None
            and previous_event.timestamp_utc
            + timedelta(seconds=max(0.0, previous_event.duration_s))
            > start
        ):
            events.insert(0, previous_event)
        events.sort(key=lambda event: (event.timestamp_utc, event.id))

        buckets = await self._repository.list_interaction_buckets(user_id, start, end)
        buckets.sort(key=lambda bucket: str(bucket["window_start_utc"]))

        browser = await self._repository.list_browser_segments(user_id, start, end)
        previous_browser = await self._repository.last_browser_segment_before(user_id, start)
        if previous_browser is not None:
            previous_browser_start = _as_utc(previous_browser["timestamp"])
            if previous_browser_start + timedelta(
                seconds=max(0.0, float(previous_browser.get("duration_s", 0.0)))
            ) > start:
                browser.insert(0, previous_browser)
        browser_spans = sorted(
            (
                _as_utc(segment["timestamp"]),
                _as_utc(segment["timestamp"])
                + timedelta(seconds=max(0.0, float(segment.get("duration_s", 0.0)))),
                segment,
            )
            for segment in browser
        )

        rows: list[dict[str, Any]] = []
        event_index = 0
        bucket_index = 0
        browser_index = 0
        active_events: list[Any] = []
        active_browser: list[tuple[datetime, datetime, dict[str, Any]]] = []
        window_start = start.replace(
            minute=(start.minute // 5) * 5,
            second=0,
            microsecond=0,
        )

        while window_start < end:
            window_end = min(window_start + timedelta(minutes=5), end)

            active_events = [
                event
                for event in active_events
                if event.timestamp_utc
                + timedelta(seconds=max(0.0, event.duration_s))
                > window_start
            ]
            while event_index < len(events) and events[event_index].timestamp_utc < window_end:
                event = events[event_index]
                if event.timestamp_utc + timedelta(
                    seconds=max(0.0, event.duration_s)
                ) > window_start:
                    active_events.append(event)
                event_index += 1

            while (
                bucket_index < len(buckets)
                and _as_utc(buckets[bucket_index]["window_start_utc"]) < window_start
            ):
                bucket_index += 1
            window_buckets: list[dict[str, Any]] = []
            while (
                bucket_index < len(buckets)
                and _as_utc(buckets[bucket_index]["window_start_utc"]) < window_end
            ):
                window_buckets.append(buckets[bucket_index])
                bucket_index += 1

            active_browser = [
                span for span in active_browser if span[1] > window_start
            ]
            while (
                browser_index < len(browser_spans)
                and browser_spans[browser_index][0] < window_end
            ):
                span = browser_spans[browser_index]
                if span[1] > window_start:
                    active_browser.append(span)
                browser_index += 1
            window_browser = [span[2] for span in active_browser]

            if active_events or window_buckets or window_browser:
                features = build_v2_feature_window(
                    active_events,
                    window_buckets,
                    window_browser,
                    window_start,
                    window_end,
                )
                rows.append({
                    "user_id": user_id,
                    "window_start_utc": window_start,
                    "window_end_utc": window_end,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "features_json": json.dumps(features, ensure_ascii=False),
                    "label": None,
                })
            window_start = window_end

        if not rows:
            return 0

        if self._baseline_repository is not None and self._session_factory is not None:
            # One explicit transaction boundary: the window upsert and the
            # baseline refresh commit together. Only rows the upsert actually
            # inserted are folded into the baseline (Welford counts each window
            # once even when the same range is rolled repeatedly), and a
            # baseline failure rolls the windows back too — nothing is left
            # half-persisted, so a retry is safe and complete.
            async with self._session_factory() as session, session.begin():
                inserted = await self._repository.upsert_feature_windows(
                    rows,
                    session=session,
                )
                if inserted:
                    baseline = await self._baseline_repository.get_latest(
                        user_id,
                        session=session,
                    )
                    if baseline is None:
                        baseline = BaselineModel(user_id=user_id)
                    baseline.update(inserted)
                    await self._baseline_repository.upsert(baseline, session=session)
        else:
            await self._repository.upsert_feature_windows(rows)
        return len(rows)

    async def rebuild_baseline_if_needed(
        self,
        user_id: int = 1,
        *,
        timezone: TimezoneLike = "local",
        now_utc: datetime | None = None,
    ) -> BaselineRebuildResult:
        """Conditionally backfill the personal baseline from existing V2 windows.

        Startup seam — wired by Todo 12, deliberately never called from a
        request path. Loads at most the prior ``_BASELINE_BACKFILL_DAYS``
        business days of stored V2 windows with one bounded range query and
        atomically replaces the baseline row with a fresh model, but only when
        the row is missing or its stored ``feature_schema_version`` is not 2;
        an existing V2 baseline is left untouched (``skipped_v2``).

        The fresh model is built fully in memory before any write, then
        persisted with a single upsert inside one caller-owned transaction, so
        an interruption before that upsert leaves any prior baseline intact.
        A stored V1 payload is never upgraded in place — it is discarded and
        replaced only after the complete V2 rebuild succeeds.
        """
        if self._baseline_repository is None or self._session_factory is None:
            msg = (
                "rebuild_baseline_if_needed requires baseline_repository "
                "and session_factory wiring"
            )
            raise RuntimeError(msg)

        now = now_utc or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        cutoff = (
            now.astimezone(resolve_timezone(timezone))
            - timedelta(days=_BASELINE_BACKFILL_DAYS)
        ).astimezone(UTC)

        baseline = await self._baseline_repository.get_latest(user_id)
        if baseline is not None and baseline.FEATURE_SCHEMA_VERSION == FEATURE_SCHEMA_VERSION:
            return BaselineRebuildResult(
                rebuilt=False,
                reason="skipped_v2",
                windows_loaded=0,
                samples=0,
                cutoff_utc=cutoff,
            )
        reason = "missing" if baseline is None else "schema_mismatch"

        windows = await self._repository.list_feature_windows_in_range(
            user_id, cutoff, now
        )
        fresh = BaselineModel(user_id=user_id, timezone=timezone)
        fresh.update(windows)

        async with self._session_factory() as session, session.begin():
            await self._baseline_repository.upsert(fresh, session=session)

        return BaselineRebuildResult(
            rebuilt=True,
            reason=reason,
            windows_loaded=len(windows),
            samples=fresh.total_samples(),
            cutoff_utc=cutoff,
        )

    async def cleanup_retained_data(self, user_id: int = 1) -> int:
        preferences = await self.get_preferences(user_id)
        now = datetime.now(UTC)
        return await self._repository.cleanup_old_telemetry(
            interaction_cutoff=now
            - timedelta(days=preferences["interaction_retention_days"]),
            activity_cutoff=now
            - timedelta(days=preferences["activity_retention_days"]),
            feature_cutoff=now - timedelta(days=180),
        )

    async def predict_latest_focus(self, user_id: int = 1) -> dict[str, Any]:
        """Predict latest focus state via ``FocusPredictionService``.

        Returns a backward-compatible dict that adds new fields without
        removing any existing ones.
        """
        if self._prediction_service is not None:
            prediction = await self._prediction_service.predict_latest(user_id=user_id)
        elif self._model_manager is not None:
            # Legacy fallback: use direct model_manager path
            latest = await self._repository.latest_feature_window(
                user_id,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
            )
            if latest is None:
                return {
                    "mode": "ready",
                    "focus_probability": None,
                    "uncertainty": 1.0,
                    "top_factors": [],
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "reason": "no_feature_windows",
                    "status": "no_data",
                }
            try:
                features = json.loads(str(latest["features_json"]))
            except (KeyError, TypeError, json.JSONDecodeError):
                features = {}
            vector = np.asarray(
                [[float(features.get(name, 0.0)) for name in V2_FEATURE_NAMES]],
                dtype=np.float64,
            )
            probabilities = self._model_manager.classifier.predict_proba(vector)
            fp = min(max(float(probabilities[0][1]), 0.0), 1.0)
            importances = self._model_manager.classifier.get_feature_importance()
            ranked: list[dict[str, str | float]] = [
                {
                    "feature": name,
                    "value": round(float(vector[0][index]), 6),
                    "importance": round(float(importances.get(name, 0.0)), 6),
                }
                for index, name in enumerate(V2_FEATURE_NAMES)
            ]
            ranked.sort(
                key=lambda factor: float(factor["importance"])
                * max(abs(float(factor["value"])), 0.01),
                reverse=True,
            )
            return {
                "mode": "ready",
                "focus_probability": round(fp, 6),
                "uncertainty": round(1.0 - abs(2.0 * fp - 1.0), 6),
                "top_factors": ranked[:3],
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "window_start_utc": latest.get("window_start_utc"),
                "model_version": self._model_manager.current_version_tag,
                "status": "ready",
                "data_age_s": None,
                "coverage_ratio": 1.0,
                "explanation_method": "global_importance_times_observation",
                "reason": "",
            }
        else:
            return {
                "mode": "rule_engine_only",
                "focus_probability": None,
                "uncertainty": 1.0,
                "top_factors": [],
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "status": "no_model",
                "data_age_s": None,
                "coverage_ratio": 0.0,
                "explanation_method": "",
                "reason": "未加载 ML 模型",
            }

        # Convert FocusPrediction to the dict response format
        top_factors = [
            {
                "feature": f["feature"],
                "value": float(f["value"]),
                "importance": float(f["importance"]),
            }
            for f in prediction.top_factors
        ] if prediction.top_factors else []

        # Map status to mode string (backward compat)
        if prediction.status == "ready":
            mode = "ready"
        elif prediction.status == "no_model":
            mode = "rule_engine_only"
        elif prediction.status == "no_data" or prediction.status == "stale":
            mode = "ready"
        else:
            mode = "rule_engine_only"

        # Backward-compat: always provide uncertainty (0.0 is valid)
        uncertainty = prediction.uncertainty if prediction.uncertainty is not None else 1.0

        # Canonical boundary mapping: only ``ready`` may carry a numeric
        # probability. Every non-ready status (no_model, no_data, stale,
        # schema_mismatch, inference_error) must be present-and-null so the
        # API contract never leaks an ML value for an unavailable state.
        return {
            "mode": mode,
            "focus_probability": (
                prediction.focus_probability if prediction.status == "ready" else None
            ),
            "uncertainty": uncertainty,
            "top_factors": top_factors,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_version": prediction.model_version,
            "window_count": prediction.window_count,
            "window_start_utc": prediction.newest_window_start_utc,
            "status": prediction.status,
            "data_age_s": prediction.data_age_s,
            "coverage_ratio": prediction.coverage_ratio,
            "explanation_method": prediction.explanation_method,
            "reason": prediction.reason,
        }

    async def save_focus_feedback(
        self,
        session_id: str,
        label: Literal["focus", "distracted", "mixed"],
        score: int,
        task_type: str | None,
        user_id: int = 1,
    ) -> dict[str, Any]:
        return await self._repository.save_focus_feedback(
            user_id=user_id,
            session_id=session_id,
            label=label,
            score=score,
            task_type=task_type,
        )

    async def get_feedback_for_sessions(
        self, session_ids: list[str], user_id: int = 1
    ) -> dict[str, dict[str, Any]]:
        """Return feedback info keyed by session_id."""
        return await self._repository.get_feedback_by_session_ids(user_id, session_ids)

    async def save_intervention_check(
        self,
        user_id: int,
        checked_at: str,
        reason: str,
        source: str = "rule_engine",
        confidence: float | None = None,
        intervention_type: str | None = None,
        throttle_reason: str | None = None,
        ml_status: str | None = None,
    ) -> None:
        """Persist one auto-intervention audit row."""
        await self._repository.save_intervention_check(
            user_id=user_id,
            checked_at=checked_at,
            reason=reason,
            source=source,
            confidence=confidence,
            intervention_type=intervention_type,
            throttle_reason=throttle_reason,
            ml_status=ml_status,
        )

    async def clear_data(
        self,
        scope: Literal["interaction", "browser", "feedback", "all"],
        user_id: int = 1,
    ) -> int:
        return await self._repository.delete_scope(user_id, scope)

    @staticmethod
    def normalize_domain(value: str) -> str:
        raw = value.strip().lower()
        if not raw:
            return ""
        parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
        host = parsed.hostname or ""
        return host.removeprefix("www.")[:253]

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()
