"""Privacy-preserving telemetry orchestration."""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal
from urllib.parse import urlsplit

import numpy as np
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from mindflow.config import get_settings
from mindflow.domain.app_classification import UserAppClassifier
from mindflow.domain.baseline import BaselineModel
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.domain.task_context import (
    TASK_CONTEXT_UNKNOWN,
    TaskContextMapper,
    summarize_task_context,
)
from mindflow.infrastructure.repositories.activity import SQLAlchemyActivityRepository
from mindflow.infrastructure.repositories.baseline import BaselineRepository
from mindflow.infrastructure.repositories.preferences import PreferencesRepository
from mindflow.infrastructure.repositories.telemetry import TelemetryRepository
from mindflow.ports import CollectorIntervalRecord
from mindflow.services.collector_interval_lifecycle import safe_error_text
from mindflow.services.prediction_service import FocusPredictionService
from mindflow.services.telemetry_features import (
    build_v2_feature_window,
    task_context_from_feature_window,
)
from mindflow.services.window_quality import build_window_quality
from mindflow.time_utils import TimezoneLike, resolve_timezone, utc_today

_DEFAULTS: dict[str, Any] = {
    "input_telemetry_enabled": False,
    "browser_tracking_enabled": False,
    "interaction_retention_days": 7,
    "activity_retention_days": 30,
}


def _effective_activity_retention_days(stored_telemetry: dict[str, Any]) -> int:
    """Resolve the single effective activity-retention value.

    The user preference ``activity_retention_days`` is authoritative; the
    environment ``event_retention_days`` is only the startup/default when the
    preference is missing.
    """
    if "activity_retention_days" in stored_telemetry:
        return int(stored_telemetry["activity_retention_days"])
    return int(get_settings().event_retention_days)

_PAIRING_CODE_TTL_S = 300

# Conditional baseline backfill horizon: at most this many business days of
# stored V2 windows feed a rebuild (matches the feature-window retention cut).
_BASELINE_BACKFILL_DAYS: Final = 180

# Feature-window rebuild horizon: how far back a schema upgrade may regenerate
# windows from raw activity events. The same 180-day cut as the feature-window
# retention (``cleanup_retained_data``), so a rebuild never asks for events the
# retention policy has already deleted.
_FEATURE_REBUILD_MAX_DAYS: Final = 180

# Rebuild chunk: one day of 5-minute windows per rollup call keeps memory flat
# (each chunk reloads only its own events) while staying a bounded query.
_FEATURE_REBUILD_CHUNK_DAYS: Final = 1

# Feature-schema version whose *labels* a rebuild may inherit. Only labels are
# read from it — never its features (a v4 rebuild is always regenerated from
# retained raw events), and its rows are never deleted by a backfill.
_FEATURE_REBUILD_LABEL_VERSION: Final = FEATURE_SCHEMA_VERSION - 1

# Trailing window a "recent" rollup covers when no watermark is known yet.
_RECENT_ROLLUP_WINDOW_HOURS: Final = 2
# One bucket of overlap on incremental rollups, so events that land just after a
# bucket boundary are still folded into it (optimisation plan 4.2).
_ROLLUP_OVERLAP_MINUTES: Final = 5


@dataclass(slots=True)
class _PairingRecord:
    """One pairing-code entry with expiry and failed-attempt tracking.

    Mutable (not frozen) so ``pair_browser`` can increment ``attempts``.
    """

    expires_at: datetime
    attempts: int = 0


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


@dataclass(frozen=True, slots=True)
class FeatureWindowRebuildResult:
    """Outcome of one feature-window rebuild (schema-upgrade seam).

    A schema bump leaves the previous version's windows in place — they stay
    readable by their own version and are never mixed into v4 training — so the
    upgrade path is an explicit, bounded regeneration from raw activity events:

        schema upgraded (v3 → v4) → ``rebuild_feature_windows()`` → v4 windows
        → stale-version windows retired within the rebuilt range

    Attributes:
        rebuilt: True only when at least one window was (re)written.
        reason: Why the call did what it did. ``current`` and ``no_windows``
            are no-ops from :meth:`TelemetryService.rebuild_feature_windows_
            if_needed`; ``empty_range`` means the requested range was empty;
            ``missing_raw_data`` means the range retains no activity events,
            interaction buckets or browser segments to rebuild from;
            ``preview`` means the windows were built but not persisted;
            ``rebuilt`` means windows were regenerated.
        windows_rolled: Windows built by this call (written only when applied).
        chunks: How many bounded rollup calls were made.
        legacy_purged: Windows of an older schema version deleted from the
            rebuilt range (0 when purging was disabled — the explicit backfill
            always keeps them).
        start_utc / end_utc: The half-open range covered by this call. Callers
            clamp their request to ``feature_rebuild_cutoff(now)`` first (the
            backfill CLI does), so these are the effective bounds that were
            actually rebuilt.
        applied: True when the caller asked for writes (``--apply``).
        windows_written: Rows actually persisted; 0 in preview mode.
        missing_raw_data: True when the range has no retained raw evidence, so
            nothing could be rebuilt from it. Reported instead of silently
            producing an empty range.
        labels_inherited: Rebuilt rows that took the previous-version label.
        labels_preserved: Rebuilt rows whose existing current-version label was
            kept (an existing v4 label always wins over an inherited v3 one).
    """

    rebuilt: bool
    reason: Literal[
        "rebuilt",
        "current",
        "no_windows",
        "empty_range",
        "missing_raw_data",
        "preview",
    ]
    windows_rolled: int
    chunks: int
    legacy_purged: int
    start_utc: datetime | None
    end_utc: datetime | None
    applied: bool = False
    windows_written: int = 0
    missing_raw_data: bool = False
    labels_inherited: int = 0
    labels_preserved: int = 0


@dataclass(frozen=True, slots=True)
class _PrefetchedRules:
    """Immutable rule list exposed through ``ClassificationRulesProtocol``.

    The rollup classifies every distinct (process, title) pair of a range; the
    classifier contract fetches rules per call, so the rules are read once and
    served from memory instead of issuing one query per pair.
    """

    rules: list[dict[str, Any]]

    async def get_all(self, user_id: int) -> list[dict[str, Any]]:
        return self.rules


class TelemetryClearResult(int):
    """Integer-compatible outcome for a telemetry deletion request.

    Existing callers can keep treating the result as the deleted-row count.
    ``partial`` and ``failures`` make post-database best-effort failures
    explicit instead of allowing the caller to mistake a half-complete wipe
    for full success.
    """

    failures: tuple[str, ...]
    partial: bool

    def __new__(
        cls,
        deleted: int,
        *,
        failures: tuple[str, ...] = (),
    ) -> TelemetryClearResult:
        result = int.__new__(cls, deleted)
        result.failures = failures
        result.partial = bool(failures)
        return result

    @property
    def deleted(self) -> int:
        return int(self)


def _as_utc(value: Any) -> datetime:
    timestamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return timestamp.replace(tzinfo=UTC) if timestamp.tzinfo is None else timestamp.astimezone(UTC)


def _decode_features(features_json: Any) -> dict[str, Any]:
    """Decode a stored ``features_json`` payload; malformed input reads empty."""
    if isinstance(features_json, dict):
        return features_json
    if not features_json:
        return {}
    try:
        parsed = json.loads(str(features_json))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def feature_rebuild_cutoff(now_utc: datetime) -> datetime:
    """Earliest instant a feature-window rebuild may read raw events from.

    ``_FEATURE_REBUILD_MAX_DAYS`` is the same 180-day cut the feature-window
    retention applies (``cleanup_retained_data``): raw events older than it are
    already deleted, so a rebuild asking for them would silently produce a blank
    range. Callers clamp their request to this instant and report the clamp
    explicitly — clamping lives here (not inside the rollup) so no caller can
    accidentally rebuild an unretained range by forgetting it.
    """
    return _as_utc(now_utc) - timedelta(days=_FEATURE_REBUILD_MAX_DAYS)


def _window_key(value: Any) -> str | None:
    """Normalise a stored ``window_start_utc`` to the UTC ISO form rows use.

    Window starts are persisted as UTC ISO text, and the upsert keys rows on
    that text, so label lookup must key on exactly the same normalised string
    (a stored non-UTC offset or a ``Z`` suffix would otherwise miss).
    """
    try:
        return _as_utc(value).isoformat()
    except (TypeError, ValueError):
        return None


def _label_by_window_start(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Index non-null labels by normalised window start; NULL labels are absent."""
    labels: dict[str, str] = {}
    for row in rows:
        label = row.get("label")
        key = _window_key(row.get("window_start_utc"))
        if label and key:
            labels[key] = str(label)
    return labels


class TelemetryService:
    def __init__(
        self,
        repository: TelemetryRepository,
        preferences_repository: PreferencesRepository,
        data_dir: Path,
        models_dir: Path | None = None,
        activity_repository: SQLAlchemyActivityRepository | None = None,
        prediction_service: FocusPredictionService | None = None,
        baseline_repository: BaselineRepository | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        interval_repository: Any | None = None,
        classification_rules_repository: Any | None = None,
        task_context_rules_repository: Any | None = None,
    ) -> None:
        self._repository = repository
        self._preferences_repository = preferences_repository
        self._data_dir = data_dir
        # Collector-interval audit repo (architecture plan B/3.3): lets the
        # rollup record coverage-gap failures next to the collector's own
        # interval rows so /health can explain missing feature windows.
        self._collector_interval_repository = interval_repository
        # Configured local ML artifact root; defaults to the same
        # data_dir-anchored "models" directory Settings resolves.
        self._models_dir = models_dir or (data_dir / "models")
        self._activity_repository = activity_repository
        self._pairing_codes: dict[str, _PairingRecord] = {}
        self._input_watcher: Any = None
        self._model_manager: Any = None
        self._prediction_service = prediction_service
        # Baseline refresh during rollup is only active when both the
        # repository and a session factory are wired (the application wires
        # them; legacy/unit constructions leave the rollup window-only).
        self._baseline_repository = baseline_repository
        self._session_factory = session_factory
        # Task-context resolution (feature schema v4). Both rule stores are
        # optional: without them the rollup still classifies through the
        # built-in AppClassifier heuristics, so an unwired/unit construction
        # produces real task-context columns rather than nothing.
        self._classification_rules_repository = classification_rules_repository
        self._task_context_rules_repository = task_context_rules_repository
        # Incremental-rollup watermark (optimisation plan 4.2). In-memory by
        # design: it exists to avoid re-computing windows that are already on
        # disk, and a restart simply re-rolls the full trailing window once —
        # which is exactly today's behaviour and is idempotent.
        self._last_successful_rollup: datetime | None = None

    @property
    def last_successful_rollup(self) -> datetime | None:
        """End of the last successfully rolled-up range, or None before the first."""
        return self._last_successful_rollup

    async def rollup_recent(
        self,
        now: datetime,
        *,
        window_hours: float = _RECENT_ROLLUP_WINDOW_HOURS,
        user_id: int = 1,
    ) -> int:
        """Roll up only what has not been rolled up yet (plan 4.2).

        The trailing ``window_hours`` window is the *upper bound*; when a
        previous rollup succeeded, the range starts just before the last
        successful bucket (``_ROLLUP_OVERLAP_MINUTES`` of overlap) so
        late-arriving events for the newest bucket are still folded in and the
        per-window upsert stays idempotent.

        The watermark only advances after a successful rollup: a failure leaves
        it untouched, so the next attempt re-covers the whole missing range
        instead of silently skipping it.
        """
        start = now - timedelta(hours=window_hours)
        if self._last_successful_rollup is not None:
            overlap = timedelta(minutes=_ROLLUP_OVERLAP_MINUTES)
            incremental = self._last_successful_rollup - overlap
            # Never start before the trailing window, and never after `now`.
            start = max(start, min(incremental, now))

        rolled = await self.rollup_feature_windows(start, now, user_id=user_id)
        # Never move the watermark backwards: a caller asking for an earlier
        # "now" (clock skew) must not reduce the coverage already achieved.
        if self._last_successful_rollup is None or now > self._last_successful_rollup:
            self._last_successful_rollup = now
        return rolled

    def attach_input_watcher(self, watcher: Any) -> None:
        self._input_watcher = watcher

    def attach_model_manager(self, model_manager: Any) -> None:
        self._model_manager = model_manager

    def detach_model_manager(self) -> None:
        """Detach all model references owned by the telemetry surface."""
        self._model_manager = None
        if self._prediction_service is not None:
            self._prediction_service.detach_model_manager()

    async def get_preferences(self, user_id: int = 1) -> dict[str, Any]:
        preferences = await self._preferences_repository.get(user_id)
        telemetry = preferences.get("telemetry", {})
        merged = {**_DEFAULTS, **telemetry}
        merged["activity_retention_days"] = _effective_activity_retention_days(
            telemetry
        )
        return merged

    async def patch_preferences(
        self,
        updates: dict[str, Any],
        user_id: int = 1,
    ) -> dict[str, Any]:
        current = await self._preferences_repository.get(user_id)
        stored_telemetry = current.get("telemetry", {})
        telemetry = {**_DEFAULTS, **stored_telemetry, **updates}
        telemetry["activity_retention_days"] = _effective_activity_retention_days(
            {**stored_telemetry, **updates}
        )
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

    _PAIRING_CODE_MAX_ATTEMPTS: int = 5
    """Failed pairing attempts allowed per code before it is invalidated.

    The pairing endpoint is unauthenticated (browser extensions hold no
    cookie), so a brute-force enumeration of the code space must be rate
    limited by the code itself. Combined with the enlarged code alphabet
    (audit report — pairing code hardening).
    """

    def _cleanup_expired_pairing_codes(self, now: datetime) -> None:
        for code, record in list(self._pairing_codes.items()):
            if record.expires_at <= now:
                del self._pairing_codes[code]

    async def create_pairing_code(self, user_id: int = 1) -> dict[str, Any]:
        now = datetime.now(UTC)
        self._cleanup_expired_pairing_codes(now)
        # 8-char, unambiguous alphabet (no 0/O/1/I): ~1.1e12 space, plus
        # per-code attempt cap below — the 6-digit space was brute-forceable
        # within the 300s TTL by a local process.
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        code = "".join(secrets.choice(alphabet) for _ in range(8))
        expires_at = now + timedelta(seconds=_PAIRING_CODE_TTL_S)
        self._pairing_codes[code] = _PairingRecord(
            expires_at=expires_at, attempts=0
        )
        await self.patch_preferences({"browser_tracking_enabled": True}, user_id)
        return {"code": code, "expires_at": expires_at.isoformat()}

    async def pair_browser(self, code: str, user_id: int = 1) -> str | None:
        record = self._pairing_codes.get(code)
        if record is None:
            return None
        now = datetime.now(UTC)
        if record.expires_at < now:
            del self._pairing_codes[code]
            return None
        # Failed attempts invalidate the code after the cap, so an attacker
        # cannot keep guessing the same code.
        record.attempts += 1
        if record.attempts > self._PAIRING_CODE_MAX_ATTEMPTS:
            del self._pairing_codes[code]
            return None
        # Successful pairing: consume the code.
        del self._pairing_codes[code]
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



    async def _collector_intervals_for_range(
        self, start: datetime, end: datetime, user_id: int
    ) -> list[CollectorIntervalRecord] | None:
        """Fetch audit evidence once; missing/failed history stays unknown."""
        interval_repo = self._collector_interval_repository
        if interval_repo is None:
            return None
        try:
            rows: list[CollectorIntervalRecord] = await interval_repo.list_overlapping_range(
                user_id, start, end,
            )
            return rows
        except Exception as exc:  # noqa: BLE001 — quality metadata is best-effort
            logger.debug("Collector interval lookup failed: {}", safe_error_text(exc))
            return None

    @staticmethod
    def _collector_state_for_window(
        start: datetime, end: datetime, intervals: list[CollectorIntervalRecord] | None,
    ) -> dict[str, dict[str, bool | None]]:
        enabled: dict[str, bool | None] = dict.fromkeys(("activity", "browser", "input"))
        available: dict[str, bool | None] = dict(enabled)
        for interval in intervals or []:
            if (interval.reason or "").startswith("coverage_gap:"):
                continue
            if (
                _as_utc(interval.started_at) < end
                and (interval.ended_at is None or _as_utc(interval.ended_at) > start)
            ):
                enabled["activity"] = True
                break
        # Run history does not record auxiliary switches or prove delivery.
        # Window payloads supply positive evidence in build_window_quality().
        return {"enabled": enabled, "available": available}

    async def _prefetched_rules(
        self, repository: Any | None, user_id: int
    ) -> list[dict[str, Any]]:
        """Read one rule store once; a failure degrades to "no user rules"."""
        if repository is None:
            return []
        try:
            rules: list[dict[str, Any]] = await repository.get_all(user_id)
        except Exception as exc:  # noqa: BLE001 — a rule read must not break a rollup
            logger.debug("Rule lookup failed: {}", safe_error_text(exc))
            return []
        return list(rules)

    async def _resolve_task_contexts(
        self,
        user_id: int,
        events: list[Any],
        browser: list[dict[str, Any]],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Resolve ``process -> task context`` and ``domain -> task context``.

        Reuses the existing machinery end to end: ``UserAppClassifier`` (user
        rules from ``app_classification_rules``, productive-learning heuristic,
        built-in ``AppClassifier``) produces an activity category, and
        ``TaskContextMapper`` (user rules from ``task_context_rules`` plus
        category defaults) produces the observed task context.  Neither the
        builder nor this method carries an app-name or domain-name list of its
        own.

        Resolution is per *process* / per *domain*: a process observed with
        several window titles keeps the context it spent the most seconds in
        (durations are the full event durations, not window-clipped — they only
        rank the process's own contexts, never enter a feature value).  Domains
        are resolved domain-rule first, then through the classifier's built-in
        domain hints, then through the browser process's category.
        """
        if not events and not browser:
            return {}, {}
        classifier = UserAppClassifier(
            rules_repo=_PrefetchedRules(
                await self._prefetched_rules(
                    self._classification_rules_repository, user_id
                )
            )
        )
        mapper = TaskContextMapper(
            await self._prefetched_rules(self._task_context_rules_repository, user_id)
        )

        process_weights: dict[str, dict[str, float]] = {}
        category_cache: dict[tuple[str, str], str] = {}
        for event in events:
            process = str(event.data.process_name or "")
            if not process:
                continue
            title = str(event.data.window_title or "")
            key = (process, title)
            context = category_cache.get(key)
            if context is None:
                category = await classifier.classify(process, title, user_id=user_id)
                context = mapper.context_for_category(category)
                category_cache[key] = context
            per_context = process_weights.setdefault(process, {})
            per_context[context] = per_context.get(context, 0.0) + max(
                0.0, float(event.duration_s)
            )
        process_contexts = {
            process: summarize_task_context(per_context).dominant
            for process, per_context in process_weights.items()
        }

        domain_contexts: dict[str, str] = {}
        domain_cache: dict[tuple[str, str], str] = {}
        for segment in browser:
            domain = str(segment.get("domain", ""))
            if not domain:
                continue
            browser_name = str(segment.get("browser_name", ""))
            key = (browser_name, domain)
            context = domain_cache.get(key)
            if context is None:
                context = mapper.context_for_domain(domain)
                if context is None:
                    domain_category = await classifier.classify(
                        domain, "", user_id=user_id
                    )
                    context = mapper.context_for_category(domain_category)
                    if context == TASK_CONTEXT_UNKNOWN and browser_name:
                        browser_category = await classifier.classify(
                            f"{browser_name}.exe", "", user_id=user_id
                        )
                        context = mapper.context_for_category(browser_category)
                domain_cache[key] = context
            domain_contexts.setdefault(domain, context)

        return process_contexts, domain_contexts

    async def _previous_task_context(
        self, user_id: int, start: datetime
    ) -> str | None:
        """Dominant task context of the window preceding a rollup range.

        The first window of a range has no in-memory predecessor but usually
        does have one on disk; reading it keeps ``task_context_transition``
        meaningful at range boundaries (scheduler runs, backfills) instead of
        resetting to "no transition" every time.
        """
        try:
            row = await self._repository.last_feature_window_before(
                user_id, start, FEATURE_SCHEMA_VERSION
            )
        except Exception as exc:  # noqa: BLE001 — transition is best-effort
            logger.debug("Previous window lookup failed: {}", safe_error_text(exc))
            return None
        if row is None:
            return None
        return task_context_from_feature_window(_decode_features(row.get("features_json")))

    async def _build_feature_window_rows(
        self,
        activity_repository: SQLAlchemyActivityRepository,
        start: datetime,
        end: datetime,
        user_id: int,
    ) -> list[dict[str, Any]]:
        """Build — never persist — the v4 windows covering [start, end).

        ``start`` must already be aligned to a 5-minute bucket and ``end`` is
        exclusive, exactly as :meth:`rollup_feature_windows` prepares them.
        Only windows with raw evidence (retained activity events, interaction
        buckets, or browser segments) produce a row, so an empty result tells a
        caller that the range has no retained raw data to rebuild from.

        Shared by the incremental rollup (which persists the rows) and the
        explicit v4 backfill (which resolves label inheritance first), so the
        two paths can never drift into different feature definitions.
        """
        events = await activity_repository.query_range(user_id, start, end)
        previous_event = await activity_repository.last_event_before(user_id, start)
        if (
            previous_event is not None
            and previous_event.timestamp_utc
            + timedelta(seconds=max(0.0, previous_event.duration_s))
            > start
        ):
            events.insert(0, previous_event)
        events.sort(key=lambda event: (event.timestamp_utc, event.id))

        buckets = await self._repository.list_interaction_buckets(
            user_id, start, end, overlapping=True,
        )
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

        # ── Task context (feature schema v4) ─────────────────────────────
        # Resolve process/domain → task context once per rollup through the
        # existing user-editable classification machinery, then hand the
        # resolved mapping to the (pure, sync) feature builder. The mapping is
        # data, never a hard-coded app list in the builder.
        process_contexts, domain_contexts = await self._resolve_task_contexts(
            user_id, events, browser
        )
        previous_task_context = await self._previous_task_context(user_id, start)

        rows: list[dict[str, Any]] = []
        event_index = 0
        bucket_index = 0
        browser_index = 0
        active_events: list[Any] = []
        active_buckets: list[dict[str, Any]] = []
        active_browser: list[tuple[datetime, datetime, dict[str, Any]]] = []

        # Reuse audit history across windows; historical auxiliary switches
        # are unknown unless the window contains positive delivery evidence.
        collector_intervals = await self._collector_intervals_for_range(start, end, user_id)
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

            active_buckets = [
                bucket for bucket in active_buckets
                if _as_utc(bucket["window_start_utc"])
                + timedelta(seconds=max(0.0, float(bucket.get("duration_s", 0))))
                > window_start
            ]
            window_buckets: list[dict[str, Any]] = []
            while (
                bucket_index < len(buckets)
                and _as_utc(buckets[bucket_index]["window_start_utc"]) < window_end
            ):
                bucket = buckets[bucket_index]
                active_buckets.append(bucket)
                # V3 aggregate counts stay attributed to the bucket's start;
                # quality coverage spans every overlapping window.
                if _as_utc(bucket["window_start_utc"]) >= window_start:
                    window_buckets.append(bucket)
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

            if active_events or active_buckets or window_browser:
                collector_state = self._collector_state_for_window(
                    window_start, window_end, collector_intervals,
                )
                features = build_v2_feature_window(
                    active_events,
                    window_buckets,
                    window_browser,
                    window_start,
                    window_end,
                    process_task_contexts=process_contexts,
                    domain_task_contexts=domain_contexts,
                    previous_task_context=previous_task_context,
                )
                # The next window compares against this one's dominant context.
                previous_task_context = task_context_from_feature_window(features)
                # Record what was actually observed. Without this record, a
                # window built while the browser collector was off looks
                # identical to one where the user simply did not browse (both
                # carry browser_ratio == 0), so the model would read an absence
                # of measurement as measured non-browsing.
                quality = build_window_quality(
                    window_start=window_start,
                    window_end=window_end,
                    events=active_events,
                    interaction_buckets=active_buckets,
                    browser_segments=window_browser,
                    enabled=collector_state.get("enabled"),
                    available=collector_state.get("available"),
                )
                rows.append({
                    "user_id": user_id,
                    "window_start_utc": window_start,
                    "window_end_utc": window_end,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "features_json": json.dumps(features, ensure_ascii=False),
                    "label": None,
                    "quality_json": json.dumps(quality.to_dict(), ensure_ascii=False),
                })
            window_start = window_end
        return rows

    async def rollup_feature_windows(
        self,
        start: datetime,
        end: datetime,
        user_id: int = 1,
    ) -> int:
        activity_repository = self._activity_repository
        if activity_repository is None:
            return 0
        if end <= start:
            return 0
        start = _as_utc(start)
        start = start.replace(
            minute=(start.minute // 5) * 5, second=0, microsecond=0,
        )
        end = _as_utc(end)

        rows = await self._build_feature_window_rows(
            activity_repository, start, end, user_id
        )
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

        # ── Coverage-gap detection (architecture plan B/3.3) ──────────
        # Expected 5-minute windows vs actually rolled rows. A gap > 20%
        # usually means the collector was down/sleeping; record it in the
        # collector_intervals audit trail so the UI can explain missing data.
        try:
            span_s = max(1.0, (end - start).total_seconds())
            expected = max(1, int(span_s / 300))
            if rows and len(rows) < expected * 0.8:
                interval_repo = getattr(
                    self, "_collector_interval_repository", None
                )
                if interval_repo is not None:
                    now_utc = datetime.now(UTC)
                    gap = await interval_repo.open(
                        user_id,
                        reason=(
                            "coverage_gap: "
                            f"{len(rows)}/{expected} windows rolled"
                        ),
                        failure=True,
                        now=now_utc,
                    )
                    await interval_repo.close(
                        gap.id, failure=True, reason=gap.reason, now=now_utc,
                    )
        except Exception as exc:
            logger.warning(
                "Coverage-gap detection failed: {}", safe_error_text(exc)
            )
        return len(rows)

    async def rebuild_feature_windows(
        self,
        start: datetime,
        end: datetime,
        user_id: int = 1,
        *,
        apply: bool = False,
    ) -> FeatureWindowRebuildResult:
        """Regenerate v4 windows for one bounded block from retained raw events.

        The explicit backfill seam (plan item 3). Window content is rebuilt
        through exactly the same code path as the incremental rollup — retained
        ``activity_events`` plus the interaction buckets and browser segments —
        so a rebuilt window is what the scheduler would have written had the
        schema been current at the time. ``features_json`` is never synthesised
        from the previous version's payload; only its *label* is inherited (a
        non-null existing v4 label always wins), and the previous version's rows
        are left in place as the evidence of what the model was trained on.

        ``apply=False`` is a true dry run: windows are built, counted and
        returned, and nothing is written. With ``apply=True`` the block is
        persisted inside one transaction, so a block either lands completely or
        not at all — re-running the same call after a failure is safe.

        Raises:
            RuntimeError: When the service has no activity repository wired (the
                rebuild cannot invent raw events).
        """
        activity_repository = self._activity_repository
        if activity_repository is None:
            msg = "rebuild_feature_windows requires activity_repository wiring"
            raise RuntimeError(msg)
        start = _as_utc(start)
        end = _as_utc(end)
        if end <= start:
            return FeatureWindowRebuildResult(
                rebuilt=False,
                reason="empty_range",
                windows_rolled=0,
                chunks=0,
                legacy_purged=0,
                start_utc=start,
                end_utc=end,
                applied=apply,
            )

        aligned_start = start.replace(
            minute=(start.minute // 5) * 5, second=0, microsecond=0,
        )
        rows = await self._build_feature_window_rows(
            activity_repository, aligned_start, end, user_id,
        )
        if not rows:
            return FeatureWindowRebuildResult(
                rebuilt=False,
                reason="missing_raw_data",
                windows_rolled=0,
                chunks=0,
                legacy_purged=0,
                start_utc=start,
                end_utc=end,
                applied=apply,
                missing_raw_data=True,
            )

        current_labels, legacy_labels = await self._rebuild_label_sources(
            user_id, aligned_start, end,
        )
        inherited = 0
        preserved = 0
        for row in rows:
            key = _window_key(row["window_start_utc"])
            existing = current_labels.get(key) if key else None
            if existing:
                # The user's own calibration on the current schema outranks any
                # inherited label: a rebuild never rewrites an existing v4 label.
                row["label"] = existing
                preserved += 1
                continue
            legacy = legacy_labels.get(key) if key else None
            if legacy:
                row["label"] = legacy
                inherited += 1

        written = 0
        if apply:
            await self._repository.upsert_feature_windows(rows)
            written = len(rows)

        return FeatureWindowRebuildResult(
            rebuilt=written > 0,
            reason="rebuilt" if apply else "preview",
            windows_rolled=len(rows),
            chunks=1,
            # Older-schema rows are deliberately kept by a backfill (they are
            # what the model was trained on); only label values cross versions.
            legacy_purged=0,
            start_utc=start,
            end_utc=end,
            applied=apply,
            windows_written=written,
            labels_inherited=inherited,
            labels_preserved=preserved,
        )

    async def _rebuild_label_sources(
        self, user_id: int, start: datetime, end: datetime,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Read the labels a rebuild may inherit, keyed by normalised start.

        Two bounded range queries, never a per-window lookup: current-version
        rows decide whether the target row already carries a label, and
        previous-version rows supply the label to inherit. NULL labels are
        absent from both maps — absence of a label is not evidence of one.
        """
        current_rows = await self._repository.list_feature_windows_in_range(
            user_id, start, end, FEATURE_SCHEMA_VERSION,
        )
        legacy_rows = await self._repository.list_feature_windows_in_range(
            user_id, start, end, _FEATURE_REBUILD_LABEL_VERSION,
        )
        return _label_by_window_start(current_rows), _label_by_window_start(legacy_rows)

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
        reason: Literal["missing", "schema_mismatch"] = (
            "missing" if baseline is None else "schema_mismatch"
        )

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
    ) -> TelemetryClearResult:
        """Delete the user's telemetry data for *scope*.

        ``all`` clears every in-scope behavioral table (raw events, input and
        browser telemetry, focus sessions/feedback, daily reports/analytics,
        intervention logs/checks/slot state, derived feature windows and
        baseline/model metadata), revokes browser pairing tokens (rows are
        kept with ``revoked_at`` set), and removes MindFlow-owned local
        training/model artifact files. Chat, backups, auth credentials, and
        user preferences are never touched. Token/artifact cleanup is
        best-effort; a returned ``TelemetryClearResult`` marks ``partial``
        and names failed steps while remaining usable as the deleted count.
        Narrower scopes keep their historical semantics.
        """
        deleted = await self._repository.delete_scope(user_id, scope)
        failures: list[str] = []
        if scope == "all":
            try:
                self._unload_runtime_models()
            except Exception as exc:
                failures.append("model_runtime")
                logger.warning("Telemetry wipe could not unload runtime models: {}", exc)

            try:
                await self._repository.revoke_browser_tokens(user_id)
            except Exception as exc:
                # Database deletion already committed; continue the wipe and
                # report the token failure instead of raising as if nothing ran.
                failures.append("browser_tokens")
                logger.warning("Telemetry wipe could not revoke browser tokens: {}", exc)

            try:
                self._delete_local_artifacts()
            except Exception as exc:
                # Artifact removal is independent of the database transaction.
                failures.append("model_artifacts")
                logger.warning("Telemetry wipe could not remove model artifacts: {}", exc)

        return TelemetryClearResult(deleted, failures=tuple(failures))

    def _unload_runtime_models(self) -> None:
        """Invalidate a real manager when available, then detach service refs."""
        manager = self._model_manager
        try:
            unload = getattr(manager, "unload", None) if manager is not None else None
            if callable(unload):
                unload()
        finally:
            # Fake managers used by tests may not implement ``unload``. They
            # are still supported because detaching does not depend on it.
            self.detach_model_manager()

    def _delete_local_artifacts(self) -> None:
        """Remove MindFlow-owned local training/model artifacts.

        All current artifacts (versioned model pickles and their HMAC
        signatures, ``manifest.json``, ``latest.json``,
        ``training_report.json``, and the signing key) live under
        ``models_dir/v2``; that directory is removed recursively. Backup
        files (``data_dir/backups``) and the auth ``token`` file are outside
        this root and never touched.
        """
        v2_dir = self._models_dir / "v2"
        if v2_dir.exists():
            shutil.rmtree(v2_dir)

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
