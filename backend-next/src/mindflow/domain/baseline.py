"""Personal behavior baseline model — per-user, per-time-period statistics.

Learns what is "normal" for each user by tracking feature distributions
across time-of-day and day-of-week buckets. Updates incrementally using
Welford's online algorithm. No external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BaselineModel:
    """Per-user behavior baseline with time-aware statistics.

    For each (hour_of_day, day_of_week) bucket, tracks:
    - count, mean, variance for each numeric feature
    - top process names and their frequency
    - total observation count

    Persistable as JSON and reloadable.
    """

    FEATURE_COLS = [
        "unique_app_count",
        "switch_frequency",
        "productivity_ratio",
        "entertainment_ratio",
        "social_ratio",
        "max_app_duration",
        "idle_ratio",
        "title_code_ratio",
        "title_doc_ratio",
        "title_url_ratio",
        "title_meeting_ratio",
        "title_entertainment_ratio",
    ]
    GROUP_COLS = ["hour_of_day", "day_of_week"]

    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at
        self.total_days: int = 0

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

    def update(self, rows: list[Mapping[str, Any]]) -> int:
        """Incrementally update baseline with new feature windows.

        Uses Welford's online algorithm for mean and variance.

        Args:
            rows: List of feature dicts, each containing FEATURE_COLS
                  columns plus hour_of_day, day_of_week, and optionally
                  process_name.

        Returns:
            Number of windows processed.
        """
        if not rows:
            return 0

        processed = 0
        for row in rows:
            hour = int(row.get("hour_of_day", 12))
            dow = int(row.get("day_of_week", 0))
            bucket = self._stats[hour][dow]

            for col in self.FEATURE_COLS:
                val = row.get(col)
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

        # Track unique dates for total_days estimate
        unique_dates: set[str] = set()
        for row in rows:
            date_val = row.get("date") or row.get("window_start")
            if date_val is not None:
                if isinstance(date_val, str):
                    unique_dates.add(date_val[:10])
                else:
                    unique_dates.add(str(date_val)[:10])
        self.total_days = max(self.total_days, len(unique_dates))
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
        """Check if baseline has enough data overall to be reliable.

        Note: counts samples across ALL (hour, dow) buckets. A True here does
        not guarantee any specific bucket is well-populated — use
        has_bucket_sufficient_data() for per-bucket checks.
        """
        total = 0
        for hour_bucket in self._stats.values():
            for dow_bucket in hour_bucket.values():
                for s in dow_bucket.values():
                    total += int(s.get("n", 0))
        return total >= min_samples

    def has_bucket_sufficient_data(self, hour: int, dow: int, min_samples: int = 2) -> bool:
        """Check if a specific (hour, dow) bucket has enough samples.

        Complements has_sufficient_data(): deviation scoring for a given time
        window needs the matching bucket populated, not just the model overall.
        """
        bucket = self._stats.get(hour, {}).get(dow, {})
        if not bucket:
            return False
        return all(int(s.get("n", 0)) >= min_samples for s in bucket.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_days": self.total_days,
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
        model = cls(user_id=data["user_id"])
        model.created_at = datetime.fromisoformat(data["created_at"])
        model.updated_at = datetime.fromisoformat(data["updated_at"])
        model.total_days = data.get("total_days", 0)

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
