"""Privacy-preserving telemetry orchestration."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np

from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.services.telemetry_features import (
    FEATURE_SCHEMA_VERSION,
    build_v2_feature_window,
)
from mindflow.time_utils import utc_today
from mindflow.train.v2 import V2_FEATURE_NAMES

_DEFAULTS: dict[str, Any] = {
    "input_telemetry_enabled": False,
    "browser_tracking_enabled": False,
    "interaction_retention_days": 7,
    "activity_retention_days": 30,
}


class TelemetryService:
    def __init__(
        self,
        repository: TelemetryRepository,
        preferences_repository: PreferencesRepository,
        data_dir: Path,
        activity_repository: SQLAlchemyActivityRepository | None = None,
    ) -> None:
        self._repository = repository
        self._preferences_repository = preferences_repository
        self._data_dir = data_dir
        self._activity_repository = activity_repository
        self._pairing_codes: dict[str, datetime] = {}
        self._input_watcher: Any = None
        self._model_manager: Any = None

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

    async def create_pairing_code(self, user_id: int = 1) -> dict[str, Any]:
        code = f"{secrets.randbelow(1_000_000):06d}"
        expires_at = datetime.now(UTC) + timedelta(minutes=5)
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
        buckets = await self._repository.list_interaction_buckets(user_id, start, end)
        browser = await self._repository.list_browser_segments(user_id, start, end)
        previous_browser = await self._repository.last_browser_segment_before(user_id, start)
        if previous_browser is not None:
            previous_browser_start = datetime.fromisoformat(previous_browser["timestamp"])
            if previous_browser_start.tzinfo is None:
                previous_browser_start = previous_browser_start.replace(tzinfo=UTC)
            if previous_browser_start + timedelta(
                seconds=max(0.0, float(previous_browser.get("duration_s", 0.0)))
            ) > start:
                browser.insert(0, previous_browser)
        count = 0
        window_start = start.replace(
            minute=(start.minute // 5) * 5,
            second=0,
            microsecond=0,
        )
        while window_start < end:
            window_end = min(window_start + timedelta(minutes=5), end)
            window_events = [
                event
                for event in events
                if event.timestamp_utc < window_end
                and event.timestamp_utc + timedelta(seconds=max(0.0, event.duration_s))
                > window_start
            ]
            window_buckets = [
                bucket
                for bucket in buckets
                if window_start
                <= datetime.fromisoformat(bucket["window_start_utc"])
                < window_end
            ]
            window_browser = []
            for segment in browser:
                segment_start = datetime.fromisoformat(segment["timestamp"])
                if segment_start.tzinfo is None:
                    segment_start = segment_start.replace(tzinfo=UTC)
                segment_end = segment_start + timedelta(
                    seconds=max(0.0, float(segment.get("duration_s", 0.0)))
                )
                if segment_start < window_end and segment_end > window_start:
                    window_browser.append(segment)
            if window_events or window_buckets or window_browser:
                features = build_v2_feature_window(
                    window_events,
                    window_buckets,
                    window_browser,
                    window_start,
                    window_end,
                )
                await self._repository.save_feature_window(
                    user_id=user_id,
                    window_start_utc=window_start,
                    window_end_utc=window_end,
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                    features_json=json.dumps(features, ensure_ascii=False),
                )
                count += 1
            window_start = window_end
        return count

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
        if self._model_manager is None:
            return {
                "mode": "rule_engine_only",
                "focus_probability": None,
                "uncertainty": 1.0,
                "top_factors": [],
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
            }
        windows = await self._repository.list_feature_windows(
            user_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
        )
        if not windows:
            return {
                "mode": "ready",
                "focus_probability": None,
                "uncertainty": 1.0,
                "top_factors": [],
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "reason": "no_feature_windows",
            }
        latest = windows[-1]
        try:
            features = json.loads(str(latest["features_json"]))
        except (KeyError, TypeError, json.JSONDecodeError):
            features = {}
        vector = np.asarray(
            [[float(features.get(name, 0.0)) for name in V2_FEATURE_NAMES]],
            dtype=np.float64,
        )
        probabilities = self._model_manager.classifier.predict_proba(vector)
        focus_probability = min(max(float(probabilities[0][1]), 0.0), 1.0)
        importances = self._model_manager.classifier.get_feature_importance()
        ranked = sorted(
            (
                {
                    "feature": name,
                    "value": round(float(vector[0][index]), 6),
                    "importance": round(float(importances.get(name, 0.0)), 6),
                }
                for index, name in enumerate(V2_FEATURE_NAMES)
            ),
            key=lambda factor: factor["importance"] * max(abs(factor["value"]), 0.01),
            reverse=True,
        )
        return {
            "mode": "ready",
            "focus_probability": round(focus_probability, 6),
            "uncertainty": round(1.0 - abs(2.0 * focus_probability - 1.0), 6),
            "top_factors": ranked[:3],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "window_start_utc": latest.get("window_start_utc"),
            "model_version": self._model_manager.current_version_tag,
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
