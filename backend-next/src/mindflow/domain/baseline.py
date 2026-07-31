"""Personal behavior baseline model — per-user, per-time-period statistics.

Learns what is "normal" for each user by tracking feature distributions
across time-of-day and day-of-week buckets. Updates incrementally using
Welford's online algorithm.

Consumes v2 feature windows: each row carries ``window_start_utc`` and a
flattened ``features_json`` (the 24-feature vocabulary from
``domain/feature_schema``). Bucketing into local (hour, weekday) and the
unique local dates used for ``total_days`` happen in the configured business
timezone. No external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.time_utils import TimezoneLike, resolve_timezone


def _timezone_key(value: TimezoneLike) -> str:
    """Stable serialization key for a timezone (IANA name, or tzname fallback)."""
    if not isinstance(value, str):
        key = getattr(value, "key", None)
        if key:
            return str(key)
        name = getattr(value, "tzname", None)
        if name is not None:
            return str(name(None)) or "UTC"
        return "UTC"
    return value


def _parse_datetime(value: Any) -> datetime:
    """Parse a window timestamp; naive datetimes are assumed UTC."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value
    s = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


class BaselineModel:
    """Per-user behavior baseline with time-aware bucket statistics.

    Persistable as JSON and reloadable. ``timezone`` is an explicit concern
    (an IANA name or ``tzinfo``) used to derive local (hour, weekday) buckets
    and unique local dates from UTC window starts.
    """

    FEATURE_COLS = list(V2_FEATURE_NAMES)
    GROUP_COLS = ["hour_of_day", "day_of_week"]
    FEATURE_SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

    def __init__(self, user_id: int, timezone: TimezoneLike = "local") -> None:
        self.user_id = user_id
        self.timezone: TimezoneLike = timezone
        self._tz: tzinfo = resolve_timezone(timezone)
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at
        self.total_days: int = 0
        # Local calendar dates seen across all updates — the exact source for
        # total_days. Kept as a set so disjoint batches never lose days.
        self._local_dates: set[str] = set()

        # stats[hour][dow][feature] = {"n": int, "mean": float, "M2": float}
        self._stats: dict[int, dict[int, dict[str, dict[str, float]]]] = {}
        # top_apps[hour][dow][app_name] = count
        self._top_apps: dict[int, dict[int, dict[str, int]]] = {}

        self._init_buckets()

    def _init_buckets(self) -> None:
        for hour in range(24):
            self._stats[hour] = {}
            self._top_apps[hour] = {}
            for dow in range(7):
                self._stats[hour][dow] = {}
                self._top_apps[hour][dow] = {}

    def _window_start_local(self, row: Mapping[str, Any]) -> datetime | None:
        """UTC window start bucketed into the configured business timezone.

        Returns None for a missing/malformed timestamp so the row is skipped
        without corrupting counts (same boundary policy as train/v2.py).
        """
        raw = row.get("window_start_utc") or row.get("window_start")
        if raw is None:
            return None
        try:
            dt = _parse_datetime(raw)
        except (ValueError, TypeError):
            return None
        return dt.astimezone(self._tz)

    @staticmethod
    def _features(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Parse the flattened ``features_json`` (string or dict) payload."""
        raw = row.get("features") or row.get("features_json")
        if raw is None:
            return {}
        if isinstance(raw, Mapping):
            return raw
        try:
            parsed = json.loads(str(raw))
        except (ValueError, TypeError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, Mapping) else None

    def update(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Incrementally update with new v2 feature windows (Welford's algorithm).

        Each row is a feature window with ``window_start_utc`` (or the legacy
        ``window_start`` alias) and flattened ``features_json`` carrying the 24
        v2 features. Rows with a malformed timestamp or unparseable JSON are
        skipped; nonnumeric feature values are skipped per feature. Neither
        path corrupts existing counts.

        Returns the number of windows processed.
        """
        if not rows:
            return 0

        processed = 0
        for row in rows:
            local = self._window_start_local(row)
            if local is None:
                continue
            hour = local.hour
            dow = local.weekday()
            features = self._features(row)
            if features is None:
                continue
            bucket = self._stats[hour][dow]

            for col in self.FEATURE_COLS:
                val = features.get(col)
                if val is None:
                    continue
                try:
                    val_f = float(val)
                except (ValueError, TypeError):
                    continue
                if col not in bucket:
                    bucket[col] = {"n": 0.0, "mean": 0.0, "M2": 0.0}
                prev = bucket[col]
                prev["n"] += 1.0
                delta = val_f - prev["mean"]
                prev["mean"] += delta / prev["n"]
                delta2 = val_f - prev["mean"]
                prev["M2"] += delta * delta2

            app = str(row.get("process_name", "unknown"))
            app_bucket = self._top_apps[hour][dow]
            app_bucket[app] = app_bucket.get(app, 0) + 1

            processed += 1

        if processed:
            # Track unique local dates for the total_days estimate. Only rows
            # with a valid timestamp contribute, so malformed rows can't inflate it.
            for row in rows:
                local = self._window_start_local(row)
                if local is not None:
                    self._local_dates.add(local.date().isoformat())
            self.total_days = max(self.total_days, len(self._local_dates))
            self.updated_at = datetime.now(UTC)

        return processed

    def get_stats(self, hour: int, dow: int) -> dict[str, dict[str, float]]:
        """Get mean/std/count for all features in a given bucket.

        Returns:
            {feature_name: {"n": int, "mean": float, "std": float}}
        """
        result: dict[str, dict[str, float]] = {}
        bucket = self._stats.get(hour, {}).get(dow, {})
        for col, s in bucket.items():
            n = int(s["n"])
            if n < 2:
                result[col] = {"n": float(n), "mean": 0.0, "std": 0.0}
            else:
                result[col] = {
                    "n": float(n),
                    "mean": round(s["mean"], 4),
                    "std": round(float(math.sqrt(s["M2"] / (n - 1))), 4),
                }
        return result

    def get_top_apps(self, hour: int, dow: int, limit: int = 5) -> list[dict[str, Any]]:
        """Get most common apps for a given bucket."""
        apps = self._top_apps.get(hour, {}).get(dow, {})
        sorted_apps = sorted(apps.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"app": a, "count": c} for a, c in sorted_apps]

    def has_sufficient_data(self, min_samples: int = 30) -> bool:
        """Check if baseline has enough total samples to be reliable."""
        total = self.total_samples()
        return total >= min_samples

    def total_samples(self) -> int:
        """Return the total number of training samples across all buckets."""
        total = 0
        for hour_bucket in self._stats.values():
            for dow_bucket in hour_bucket.values():
                for s in dow_bucket.values():
                    total += int(s.get("n", 0))
        return total

    def has_bucket_sufficient_data(self, hour: int, dow: int, min_samples: int = 2) -> bool:
        """Check if a specific (hour, dow) bucket has enough samples."""
        bucket = self._stats.get(hour, {}).get(dow, {})
        if not bucket:
            return False
        return all(int(s.get("n", 0)) >= min_samples for s in bucket.values())

    def overall_mean(self, feature: str) -> float | None:
        """Sample-weighted mean of *feature* across all (hour, dow) buckets.

        Uses each bucket's n and mean so that larger buckets carry more weight.
        Returns None when *feature* is not a valid FEATURE_COL or has no data.
        """
        if feature not in self.FEATURE_COLS:
            return None
        total_n = 0.0
        weighted_sum = 0.0
        for hour_bucket in self._stats.values():
            for dow_bucket in hour_bucket.values():
                s = dow_bucket.get(feature)
                if s is not None and s["n"] > 0:
                    weighted_sum += s["n"] * s["mean"]
                    total_n += s["n"]
        return round(weighted_sum / total_n, 4) if total_n > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "feature_schema_version": self.FEATURE_SCHEMA_VERSION,
            "timezone": _timezone_key(self.timezone),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_days": self.total_days,
            "local_dates": sorted(self._local_dates),
            "stats": {
                str(h): {
                    str(d): {
                        f: {
                            k: round(v, 6) if isinstance(v, float) else int(v) for k, v in s.items()
                        }
                        for f, s in dow_bucket.items()
                    }
                    for d, dow_bucket in hour_bucket.items()
                }
                for h, hour_bucket in self._stats.items()
            },
            "top_apps": {
                str(h): {str(d): apps for d, apps in top_bucket.items()}
                for h, top_bucket in self._top_apps.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineModel:
        model = cls(
            user_id=data["user_id"],
            timezone=data.get("timezone", "local"),
        )
        model.created_at = datetime.fromisoformat(data["created_at"])
        model.updated_at = datetime.fromisoformat(data["updated_at"])
        model.total_days = data.get("total_days", 0)
        # Restore the exact local-date set for V2 payloads; legacy V1 payloads
        # predate it and keep only the stored count (they get rebuilt anyway).
        model._local_dates = set(data.get("local_dates", ()))
        # Stored schema version is preserved verbatim (defaults to 1 for legacy
        # V1 payloads without the field) so a mismatch stays detectable — there
        # is deliberately no V1 compatibility mapper.
        model.FEATURE_SCHEMA_VERSION = int(data.get("feature_schema_version", 1))

        for h_str, hour_bucket in data.get("stats", {}).items():
            h = int(h_str)
            for d_str, dow_bucket in hour_bucket.items():
                d = int(d_str)
                model._stats[h][d] = {
                    f: {"n": float(s["n"]), "mean": float(s["mean"]), "M2": float(s["M2"])}
                    for f, s in dow_bucket.items()
                }

        for h_str, hour_bucket in data.get("top_apps", {}).items():
            h = int(h_str)
            for d_str, apps in hour_bucket.items():
                d = int(d_str)
                model._top_apps[h][d] = dict(apps)

        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BaselineModel:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
