"""Deviation detection — compare current behavior against personal baseline.

Computes multi-dimensional z-scores, flags anomalous time windows,
and ranks anomalies by severity for LLM context packing.
No external dependencies beyond stdlib.

Scoring model
-------------
Each 5-minute feature window is compared against the user's own baseline for
the same local (hour_of_day, day_of_week) bucket. The expected value and the
dispersion come from :meth:`BaselineModel.shrunk_expected`: an empirical-Bayes
(James–Stein) posterior mean that pulls a sparse bucket toward a prior, paired
with a median-absolute-deviation scale. Both replace the arithmetic mean /
standard deviation because they survive a handful of extreme days (breakdown
point 50%) that would otherwise inflate σ and mask exactly the deviations this
detector exists to find.

The prior and the pseudo-count are estimated, not configured (§8.25): a bucket
borrows its level from the same hour across all weekdays, and the shrinkage
strength is the moment estimate of σ²_w / τ² for that feature.

References:
    * Efron, B. & Morris, C. (1975). "Data Analysis Using Stein's Estimator
      and Its Generalizations." JASA 70(350), 311–319.
    * Rousseeuw, P. J. & Croux, C. (1993). "Alternatives to the Median
      Absolute Deviation." JASA 88(424), 1273–1283.

The per-feature scores are combined with fixed relative weights, renormalised
over whichever features are actually present in the window. A window that
carries only a subset of the vocabulary therefore still produces a comparable
average rather than being diluted toward zero by the missing columns.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.feature_labels import feature_label

#: How many features ``to_summary`` may return (contract §2 caps at 6).
MAX_SUMMARY_FEATURES: int = 6


class DeviationDetector:
    """Detect behavioral deviations from the personal baseline.

    For each 5-minute feature window, computes per-feature robust z-scores
    against the baseline for that (hour_of_day, day_of_week) bucket and
    combines them into a weighted overall deviation score.
    """

    #: Relative feature weights over the V2/V4 vocabulary, summing to 1.0.
    #: They are renormalised over the features that actually have baseline
    #: statistics in a given window, so a partially-populated row still yields
    #: a comparable average instead of being diluted toward zero by the
    #: missing columns. Switching and task continuity carry the most weight
    #: because they are what distinguishes focused work from fragmented
    #: switching; browser-derived ratios carry less because they are partly
    #: redundant with the interaction features. Three groups are deliberately
    #: absent: the cyclic time encodings (hour_sin, hour_cos, weekday_sin,
    #: weekday_cos) describe the context the bucket already conditions on; the
    #: task_* columns are sparse by construction (mostly ``unknown``) and would
    #: score their own missingness; and mouse/scroll intensity is dominated by
    #: hardware differences between users rather than by focus.
    FEATURE_WEIGHTS: dict[str, float] = {
        "app_switch_count": 0.18,
        "longest_segment_ratio": 0.14,
        "idle_ratio": 0.12,
        "domain_switch_count": 0.10,
        "input_active_ratio": 0.10,
        "keypress_rate_per_min": 0.08,
        "interaction_bursts_per_min": 0.06,
        "browser_ratio": 0.06,
        "audible_browser_ratio": 0.06,
        "top_app_ratio": 0.05,
        "interaction_interval_cv": 0.05,
    }

    #: Z-score thresholds for severity classification. Unchanged from the
    #: pre-robust detector so downstream severity gates keep their meaning.
    MILD_THRESHOLD = 1.5  # noticeable but common
    MODERATE_THRESHOLD = 2.5  # clearly unusual
    SEVERE_THRESHOLD = 4.0  # extreme outlier

    #: Optional override for the empirical-Bayes pseudo-count. ``None`` (the
    #: production default) lets :meth:`BaselineModel.shrunk_expected` estimate
    #: k from the data and use the hour-pooled prior (spec §8.25). Setting it
    #: pins the legacy fixed-k / global-robust behaviour, which is what the
    #: pre-§8.25 tests and A/B comparisons use.
    SHRINKAGE_K: float | None = None

    def __init__(self, baseline: BaselineModel) -> None:
        self.baseline = baseline

    # ── per-feature estimate ─────────────────────────────────────────────

    def _estimate(
        self, feature: str, hour: int, dow: int
    ) -> tuple[float, float] | None:
        """Expected value and dispersion for *feature*, or None when unknown.

        Delegates entirely to :meth:`BaselineModel.shrunk_expected`, which owns
        the empirical-Bayes arithmetic, the relative scale floor and the
        cold-start fallback. Returns ``None`` when the feature has never been
        observed, so the caller skips it rather than inventing a baseline.
        """
        shrunk = self.baseline.shrunk_expected(feature, hour, dow, k=self.SHRINKAGE_K)
        if shrunk is None:
            return None
        return shrunk["expected"], shrunk["scale"]

    @staticmethod
    def _feature_values(row: Mapping[str, Any]) -> Mapping[str, Any]:
        """Feature values of *row*, whether flattened or inside ``features_json``.

        Persisted windows carry their vector as JSON; callers that built a row
        in memory pass flattened keys. Accepting both keeps one scoring path
        for ``EvidenceBundle``, the API and tests instead of two conventions.
        Top-level keys win: an explicit caller value overrides the stored blob.
        """
        raw = row.get("features") or row.get("features_json")
        if raw is None:
            return row
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                return row
            raw = parsed
        if not isinstance(raw, Mapping):
            return row
        merged = dict(raw)
        merged.update({k: v for k, v in row.items() if k not in {"features", "features_json"}})
        return merged

    def _score_features(
        self, row: Mapping[str, Any], hour: int, dow: int
    ) -> list[dict[str, Any]]:
        """Score every weighted feature present and numeric in *row*.

        Returns one dict per scored feature with the raw value, expected
        value, robust z (clamped to ±10) and the feature's relative weight.
        """
        scored: list[dict[str, Any]] = []
        values = self._feature_values(row)
        for feature, weight in self.FEATURE_WEIGHTS.items():
            val_raw = values.get(feature)
            if val_raw is None:
                continue
            try:
                value = float(val_raw)
            except (ValueError, TypeError):
                continue
            estimate = self._estimate(feature, hour, dow)
            if estimate is None:
                # No baseline data for this feature in this bucket and no
                # global data either: the feature is skipped and its weight
                # leaves the denominator (renormalisation below).
                continue
            expected, scale = estimate
            z = max(min((value - expected) / scale, 10.0), -10.0)
            scored.append(
                {
                    "feature": feature,
                    "label": feature_label(feature),
                    "value": value,
                    "expected": expected,
                    "robust_z": round(z, 3),
                    "weight": weight,
                }
            )
        return scored

    @classmethod
    def severity_for(cls, overall: float) -> str:
        """Map an overall weighted deviation to a contract severity label."""
        if overall >= cls.SEVERE_THRESHOLD:
            return "severe"
        if overall >= cls.MODERATE_THRESHOLD:
            return "moderate"
        if overall >= cls.MILD_THRESHOLD:
            return "mild"
        return "normal"

    def score_window(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Score a single feature window against the personal baseline.

        Args:
            row: One V2 feature window as a dict. ``hour_of_day`` /
                ``day_of_week`` select the bucket; the remaining keys are read
                by :attr:`FEATURE_WEIGHTS`.

        Returns:
            dict with per-feature z-scores, the features that contributed, and
            the weighted overall deviation / severity.
        """
        hour = int(row.get("hour_of_day", 12))
        dow = int(row.get("day_of_week", 0))

        scored = self._score_features(row, hour, dow)
        z_scores = {item["feature"]: item["robust_z"] for item in scored}

        total_weight = sum(item["weight"] for item in scored)
        if total_weight > 0.0:
            weighted_sum = sum(item["weight"] * abs(item["robust_z"]) for item in scored)
            overall = round(weighted_sum / total_weight, 3)
        else:
            overall = 0.0

        severity = self.severity_for(overall)
        top_deviations = sorted(
            z_scores.items(), key=lambda x: abs(x[1]), reverse=True
        )[:3]

        return {
            "window_start": str(row.get("window_start", "")),
            "hour_of_day": hour,
            "day_of_week": dow,
            "overall_deviation": overall,
            "severity": severity,
            "z_scores": z_scores,
            "scored_features": scored,
            "top_deviations": [
                {"feature": f, "z_score": z, "direction": "up" if z > 0 else "down"}
                for f, z in top_deviations
                if abs(z) > 0.5
            ],
        }

    def to_summary(self, row: Mapping[str, Any]) -> dict[str, Any] | None:
        """Build the contract's ``DeviationSummary`` for one feature window.

        Shape (API contract §2)::

            {
              "overall_score": float,      # weighted mean |robust z|
              "severity": "normal" | "mild" | "moderate" | "severe",
              "method": "eb_shrinkage_mad",
              "bucket": {"hour_of_day", "day_of_week", "n_samples", "shrinkage"},
              "features": [{feature, label, value, expected, robust_z, weight}],
            }

        ``features`` is capped at :data:`MAX_SUMMARY_FEATURES` and sorted by
        ``|robust_z| · weight`` descending — the contribution to the overall
        score, not the bare z, so a high z on a low-weight feature does not
        outrank the driver of the score. Returns ``None`` when no feature in
        the row has baseline data (there is nothing honest to summarise).
        """
        hour = int(row.get("hour_of_day", 12))
        dow = int(row.get("day_of_week", 0))
        scored = self._score_features(row, hour, dow)
        if not scored:
            return None

        total_weight = sum(item["weight"] for item in scored)
        weighted_sum = sum(item["weight"] * abs(item["robust_z"]) for item in scored)
        overall = round(weighted_sum / total_weight, 3) if total_weight > 0.0 else 0.0

        # n_samples: how many windows the bucket actually holds. Taken from the
        # shrunk estimate itself so the reported shrinkage and the number that
        # produced it can never disagree — k is per-feature now, so the
        # shrinkage must come from the same estimate as n_bucket rather than
        # being recomputed from a class constant.
        n_samples = 0
        shrinkage = 1.0
        for item in scored:
            estimate = self.baseline.shrunk_expected(
                str(item["feature"]), hour, dow, k=self.SHRINKAGE_K
            )
            if estimate is not None and int(estimate["n_bucket"]) >= n_samples:
                n_samples = int(estimate["n_bucket"])
                shrinkage = float(estimate["shrinkage"])

        ranked = sorted(
            scored,
            key=lambda item: abs(item["robust_z"]) * item["weight"],
            reverse=True,
        )[:MAX_SUMMARY_FEATURES]

        return {
            "overall_score": overall,
            "severity": self.severity_for(overall),
            "method": "eb_shrinkage_mad",
            "bucket": {
                "hour_of_day": hour,
                "day_of_week": dow,
                "n_samples": n_samples,
                "shrinkage": round(shrinkage, 4),
            },
            "features": [
                {
                    "feature": item["feature"],
                    "label": item["label"],
                    "value": round(float(item["value"]), 4),
                    "expected": round(float(item["expected"]), 4),
                    "robust_z": item["robust_z"],
                    "weight": item["weight"],
                }
                for item in ranked
            ],
        }

    def analyze_dataframe(
        self,
        rows: Sequence[Mapping[str, Any]],
        window_titles: Sequence[str | list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Analyze all windows and return anomalies sorted by severity.

        Args:
            rows: V2 feature rows (one per 5-minute window).
            window_titles: Optional sequence of representative window titles
                per window, used to enrich anomaly descriptions.

        Returns:
            List of anomaly dicts, sorted by overall_deviation descending.
        """
        results: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            score = self.score_window(row)
            if score["severity"] != "normal":
                if window_titles is not None and idx < len(window_titles):
                    score["sample_titles"] = self._sample_titles(window_titles[idx])
                results.append(score)

        results.sort(key=lambda x: x["overall_deviation"], reverse=True)
        return results

    def _sample_titles(self, titles_value: Any) -> list[str]:
        """Extract representative titles from a window's title data."""
        if isinstance(titles_value, str):
            return [titles_value] if titles_value.strip() else []
        if isinstance(titles_value, (list, tuple)):
            return [str(t) for t in titles_value[:5] if str(t).strip()]
        return []

    def daily_summary(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Generate a daily summary with anomaly count and trend.

        Returns:
            dict with: total_windows, anomaly_count, anomaly_ratio,
            severity_counts, average_deviation, most_anomalous_hour
        """
        if not rows:
            return {
                "total_windows": 0,
                "anomaly_count": 0,
                "anomaly_ratio": 0.0,
                "severity_counts": {},
                "average_deviation": 0.0,
                "most_anomalous_hour": None,
            }

        severities: dict[str, int] = {}
        total_deviation = 0.0
        anomaly_hours: dict[int, float] = {}
        anomaly_count = 0

        for row in rows:
            score = self.score_window(row)
            total_deviation += score["overall_deviation"]
            sev = score["severity"]
            severities[sev] = severities.get(sev, 0) + 1
            if sev != "normal":
                anomaly_count += 1
                h = score["hour_of_day"]
                anomaly_hours[h] = anomaly_hours.get(h, 0.0) + score["overall_deviation"]

        n = len(rows)
        most_anomalous: int | None = (
            max(anomaly_hours, key=lambda h: anomaly_hours[h]) if anomaly_hours else None
        )

        return {
            "total_windows": n,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": round(anomaly_count / n, 3) if n > 0 else 0.0,
            "severity_counts": severities,
            "average_deviation": round(total_deviation / n, 3) if n > 0 else 0.0,
            "most_anomalous_hour": most_anomalous,
        }
