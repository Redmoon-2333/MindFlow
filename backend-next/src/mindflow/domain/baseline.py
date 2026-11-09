"""Personal behavior baseline model — per-user, per-time-period statistics.

Learns what is "normal" for each user by tracking feature distributions
across time-of-day and day-of-week buckets. Updates incrementally using
Welford's online algorithm.

Consumes v2 feature windows: each row carries ``window_start_utc`` and a
flattened ``features_json`` (the 24-feature vocabulary from
``domain/feature_schema``). Bucketing into local (hour, weekday) and the
unique local dates used for ``total_days`` happen in the configured business
timezone. No external dependencies beyond stdlib.

Robust statistics (empirical-Bayes baseline)
--------------------------------------------
Welford's mean/std are efficient but fragile: a handful of extreme windows
(a video binge, a day of debugging) inflate σ and hide real deviations.
Alongside the per-bucket mean/variance this model maintains one *global*
bounded ring buffer per feature of the most recent
``ROBUST_RESERVOIR_SIZE`` values, from which the median and the median
absolute deviation (MAD) are recomputed on demand. The ring is global rather
than per bucket on purpose: 168 buckets × 28 features of reservoirs would
dominate the persisted JSON, and each bucket already has exact Welford
statistics. The buffer is deterministic (a ring buffer, no random sampling),
so the same update sequence always yields the same robust statistics.

The two are combined by :meth:`BaselineModel.shrunk_expected`: the bucket's
exact mean/variance shrunk toward the global median/``MAD·1.4826``. When the
ring is still shorter than ``MIN_ROBUST_SAMPLES`` the global Welford mean/std
serves as the prior instead — a cold-start user still gets a usable baseline
from day one.

The scale estimate follows Rousseeuw & Croux (1993, JASA), *"Alternatives to
the Median Absolute Deviation"*: ``MAD · 1.4826`` is a consistent estimator
of σ under normality. Medians are used instead of means because they have a
breakdown point of 50% — half the reservoir can be arbitrarily corrupted
before the estimate moves.

Persistence compatibility
-------------------------
The robust reservoir is serialized under an optional ``robust`` key. Old
payloads (no ``robust`` key) load unchanged and simply have no robust state
until the next ``update``; new payloads written by this version are read by
the old ``from_dict`` (which ignores unknown keys) without error. The Welford
``stats`` block keeps its exact shape and meaning.
"""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.time_utils import TimezoneLike, resolve_timezone

#: How many most-recent values per feature feed the global median/MAD.
#: One ring per feature (not per bucket) keeps ``baseline_models`` JSON small;
#: 256 values is roughly a day of 5-minute windows, enough for a stable median.
ROBUST_RESERVOIR_SIZE: int = 256

#: Consistency constant turning the MAD into a σ estimate under normality
#: (Rousseeuw & Croux 1993, JASA; also Iglewicz & Hoaglin 1993 §3.2).
MAD_TO_SIGMA: float = 1.4826

#: Additive floor for any scale estimate, so a zero-dispersion bucket cannot
#: divide by 0.
MIN_SCALE: float = 1e-6

#: Relative floor: the scale may never be smaller than this fraction of the
#: expected value. Without it, a feature whose absolute spread is tiny but
#: whose mean is large (a ratio hovering at 0.95) would report enormous
#: z-scores for trivial movements.
SCALE_RELATIVE_FLOOR: float = 0.05

#: Minimum global ring-buffer size before its median/MAD is trusted. Below
#: this the global Welford mean/std takes over: the MAD of a handful of values
#: is itself too noisy to be a scale.
MIN_ROBUST_SAMPLES: int = 8

#: Pseudo-count for the empirical-Bayes shrinkage of a bucket estimate toward
#: the user's global estimate. ``k = 10`` observations of evidence are needed
#: before the bucket's own estimate gets half the weight (k / (n + k) = 0.5 at
#: n = 10), which is a deliberately conservative prior for a personal baseline
#: that starts with a handful of windows per time-of-day bucket.
DEFAULT_SHRINKAGE_K: int = 10


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
        # robust_global[feature] = bounded ring of recent values, across every
        # bucket. Deliberately NOT per bucket: 168 buckets × 28 features of
        # ring buffers would dominate the baseline_models JSON, and the bucket
        # already has exact Welford statistics.
        self._robust_global: dict[str, deque[float]] = {}

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
        skipped; nonnumeric and non-finite (NaN/Infinity) feature values are
        skipped per feature. Neither path corrupts existing counts.

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
                if not math.isfinite(val_f):
                    # NaN/Inf would poison Welford state and leak a non-finite
                    # overall mean into the analytics payload; skip per feature.
                    continue
                if col not in bucket:
                    bucket[col] = {"n": 0.0, "mean": 0.0, "M2": 0.0}
                prev = bucket[col]
                prev["n"] += 1.0
                delta = val_f - prev["mean"]
                prev["mean"] += delta / prev["n"]
                delta2 = val_f - prev["mean"]
                prev["M2"] += delta * delta2

                # Bounded ring for the global robust (median/MAD) statistics.
                # A ring buffer keeps the newest ROBUST_RESERVOIR_SIZE values;
                # dropping the oldest is what makes this O(1) per update.
                self._robust_global.setdefault(
                    col, deque(maxlen=ROBUST_RESERVOIR_SIZE)
                ).append(val_f)

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
                if s is not None and s["n"] > 0 and math.isfinite(s["mean"]):
                    weighted_sum += s["n"] * s["mean"]
                    total_n += s["n"]
        return round(weighted_sum / total_n, 4) if total_n > 0 else None

    # ══════════════════════════════════════════════════════════════════
    # Robust statistics (median / MAD)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        """Median of an already-materialised sequence (O(n log n), stdlib)."""
        ordered = sorted(values)
        n = len(ordered)
        if n == 0:
            return 0.0
        mid = n // 2
        if n % 2 == 1:
            return float(ordered[mid])
        return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0

    def global_robust_stats(self, feature: str) -> dict[str, float] | None:
        """Median and MAD-based scale of *feature* over the whole ring buffer.

        The ring holds the newest :data:`ROBUST_RESERVOIR_SIZE` observed values
        of the feature across every (hour, weekday) bucket, so the median has a
        50% breakdown point: half the buffer can be arbitrary before the
        estimate moves. The scale is ``1.4826 · MAD``, the σ-consistent form of
        Rousseeuw & Croux (1993).

        Returns ``None`` when fewer than :data:`MIN_ROBUST_SAMPLES` values have
        been seen — the MAD of a handful of values is itself too noisy to serve
        as a scale, and callers fall back to the global Welford statistics.
        """
        reservoir = self._robust_global.get(feature)
        if not reservoir or len(reservoir) < MIN_ROBUST_SAMPLES:
            return None
        values = list(reservoir)
        median = self._median(values)
        mad = self._median([abs(v - median) for v in values])
        return {
            "n": float(len(values)),
            "median": median,
            "mad": mad,
            "sigma": mad * MAD_TO_SIGMA,
        }

    def global_welford_stats(self, feature: str) -> dict[str, float] | None:
        """Pooled Welford mean/std of *feature* across every bucket.

        Used as the prior when the ring buffer is still too short for a
        trustworthy MAD (cold start). The pooled variance is
        ``Σ M2 / (N − 1)`` over all buckets — the exact sample variance of the
        union of the buckets, not an average of per-bucket variances.
        """
        if feature not in self.FEATURE_COLS:
            return None
        total_n = 0.0
        total_m2 = 0.0
        weighted_sum = 0.0
        for hour_bucket in self._stats.values():
            for dow_bucket in hour_bucket.values():
                s = dow_bucket.get(feature)
                if s is None or s["n"] <= 0 or not math.isfinite(s["mean"]):
                    continue
                total_n += s["n"]
                total_m2 += s["M2"]
                weighted_sum += s["n"] * s["mean"]
        if total_n <= 0.0:
            return None
        mean = weighted_sum / total_n
        std = math.sqrt(total_m2 / (total_n - 1.0)) if total_n > 1.0 else 0.0
        return {"n": total_n, "mean": mean, "std": std}

    def shrunk_expected(
        self,
        feature: str,
        hour: int,
        dow: int,
        k: int = DEFAULT_SHRINKAGE_K,
    ) -> dict[str, float] | None:
        """Empirical-Bayes expected value and scale for one (bucket, feature).

        The bucket's exact Welford statistics are shrunk toward the user's
        global robust statistics:

        ``μ̂ = (n_b · mean_b + k · median_g) / (n_b + k)`` and
        ``ŝ = sqrt((n_b · var_b + k · σ_g²) / (n_b + k))``.

        ``median_g`` and ``σ_g`` come from the global ring buffer (median,
        ``1.4826 · MAD``), so the prior itself is not dragged around by a few
        extreme windows.

        The scale is floored at ``1e-6 + 0.05·|μ̂|``: a feature that is nearly
        constant in absolute terms but large in magnitude (a ratio sitting at
        0.95) must not report a huge z-score for a trivial movement.

        A zero combined variance means the feature has never varied in either
        the bucket or the global ring — there is no dispersion to normalise
        against, and dividing by the floor would report a meaningless
        six-figure z for any nonzero value. ``None`` is returned instead so
        callers skip the feature, exactly as the previous MAD estimator did
        when the MAD was zero.

        Cold start (``n_b = 0``) reduces to the global estimate; if the ring is
        still too short it falls back to the pooled global Welford mean/std.
        Returns ``None`` only when the feature has never been observed.

        Returns:
            ``{"expected", "scale", "n_bucket", "n_global", "shrinkage"}``
            where ``shrinkage = k / (n_b + k)``.
        """
        if feature not in self.FEATURE_COLS:
            return None
        stats = self._stats.get(hour, {}).get(dow, {}).get(feature)
        n_b = float(stats["n"]) if stats is not None else 0.0
        mean_b = float(stats["mean"]) if stats is not None else 0.0
        var_b = (
            float(stats["M2"]) / (n_b - 1.0)
            if stats is not None and n_b > 1.0
            else 0.0
        )

        robust = self.global_robust_stats(feature)
        if robust is not None:
            prior_mean = robust["median"]
            prior_var = robust["sigma"] ** 2
            n_global = robust["n"]
        else:
            welford = self.global_welford_stats(feature)
            if welford is None and n_b <= 0.0:
                return None
            prior_mean = welford["mean"] if welford is not None else mean_b
            prior_var = welford["std"] ** 2 if welford is not None else var_b
            n_global = welford["n"] if welford is not None else 0.0

        if n_b <= 0.0:
            expected = prior_mean
            variance = prior_var
            shrinkage = 1.0
        else:
            denominator = n_b + float(k)
            expected = (n_b * mean_b + k * prior_mean) / denominator
            variance = (n_b * var_b + k * prior_var) / denominator
            shrinkage = float(k) / denominator

        if variance <= 0.0:
            # Never varied in the bucket or the global ring: no honest scale.
            return None
        scale = max(
            math.sqrt(variance),
            MIN_SCALE + SCALE_RELATIVE_FLOOR * abs(expected),
        )
        return {
            "expected": expected,
            "scale": scale,
            "n_bucket": n_b,
            "n_global": n_global,
            "shrinkage": shrinkage,
        }

    def robust_z(self, feature: str, value: float, hour: int, dow: int) -> float | None:
        """Robust z-score of *value* against the shrunk (hour, dow) expectation.

        ``z = (x − μ̂) / ŝ`` with ``μ̂``/``ŝ`` from :meth:`shrunk_expected`.
        The prior is a median/MAD estimate, so a single extreme day cannot
        inflate the denominator and mask the deviation (breakdown point 50%).

        Returns ``None`` when *feature* has no baseline data, or when *value*
        is not finite.
        """
        if not math.isfinite(value):
            return None
        estimate = self.shrunk_expected(feature, hour, dow)
        if estimate is None:
            return None
        return (value - estimate["expected"]) / estimate["scale"]

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
            # Additive, optional block: a reader that predates it ignores the
            # key, and a payload written before it exists simply has no robust
            # state (see the module docstring's compatibility note).
            "robust": {
                "reservoir_size": ROBUST_RESERVOIR_SIZE,
                "global": {f: list(values) for f, values in self._robust_global.items()},
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

        # Optional robust block. Absent in payloads written before the
        # empirical-Bayes upgrade: those load cleanly and simply start with an
        # empty reservoir, which ``shrunk_expected`` treats as "no data yet".
        robust = data.get("robust")
        if isinstance(robust, dict):
            reservoir_size = int(robust.get("reservoir_size", ROBUST_RESERVOIR_SIZE))
            for feature, values in dict(robust.get("global", {})).items():
                ring: deque[float] = deque(
                    (float(v) for v in values), maxlen=reservoir_size
                )
                if ring:
                    model._robust_global[str(feature)] = ring

        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BaselineModel:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
