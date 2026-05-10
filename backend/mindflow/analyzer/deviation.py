"""Deviation detection — compare current behavior against personal baseline.

Computes multi-dimensional z-scores, flags anomalous time windows,
and ranks anomalies by severity for LLM context packing.
"""

import numpy as np
import pandas as pd

from mindflow.analyzer.baseline import BaselineModel


class DeviationDetector:
    """Detect behavioral deviations from personal baseline.

    For each 30-minute window, computes per-feature z-scores against
    the baseline for that (hour_of_day, day_of_week) bucket. Combines
    into a weighted overall deviation score.
    """

    # Weights for overall deviation score — behavior features matter more
    # than title features (which depend on collector having window titles)
    FEATURE_WEIGHTS: dict[str, float] = {
        "switch_frequency": 0.20,
        "unique_app_count": 0.15,
        "max_app_duration": 0.10,
        "idle_ratio": 0.10,
        "productivity_ratio": 0.05,
        "entertainment_ratio": 0.05,
        "social_ratio": 0.05,
        "title_code_ratio": 0.05,
        "title_doc_ratio": 0.05,
        "title_url_ratio": 0.05,
        "title_meeting_ratio": 0.05,
        "title_entertainment_ratio": 0.10,
    }

    # Z-score thresholds for severity classification
    MILD_THRESHOLD = 1.5   # noticeable but common
    MODERATE_THRESHOLD = 2.5  # clearly unusual
    SEVERE_THRESHOLD = 4.0   # extreme outlier

    def __init__(self, baseline: BaselineModel):
        self.baseline = baseline

    def score_window(self, row: pd.Series) -> dict:
        """Score a single feature window against baseline.

        Returns dict with per-feature z-scores and overall deviation.
        """
        hour = int(row.get("hour_of_day", 12))
        dow = int(row.get("day_of_week", 0))
        bucket_stats = self.baseline.get_stats(hour, dow)

        z_scores: dict[str, float] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for feature, weight in self.FEATURE_WEIGHTS.items():
            if feature not in row or pd.isna(row[feature]):
                continue
            val = float(row[feature])
            stats = bucket_stats.get(feature, {"n": 0, "mean": 0.0, "std": 0.0})
            if stats["n"] < 2 or stats["std"] == 0:
                z = 0.0
            else:
                z = (val - stats["mean"]) / max(stats["std"], 0.001)
            z = max(min(z, 10.0), -10.0)
            z_scores[feature] = round(z, 3)
            weighted_sum += weight * abs(z)
            total_weight += weight

        overall = round(weighted_sum / max(total_weight, 0.001), 3)

        if overall >= self.SEVERE_THRESHOLD:
            severity = "severe"
        elif overall >= self.MODERATE_THRESHOLD:
            severity = "moderate"
        elif overall >= self.MILD_THRESHOLD:
            severity = "mild"
        else:
            severity = "normal"

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
            "top_deviations": [
                {"feature": f, "z_score": z, "direction": "up" if z > 0 else "down"}
                for f, z in top_deviations if abs(z) > 0.5
            ],
        }

    def analyze_dataframe(
        self, features_df: pd.DataFrame, window_titles: pd.Series | None = None
    ) -> list[dict]:
        """Analyze all windows and return anomalies sorted by severity.

        Args:
            features_df: Feature DataFrame with one row per 30-min window.
            window_titles: Optional Series of representative window titles
                per window, used to enrich anomaly descriptions.

        Returns list of anomaly dicts, sorted by overall_deviation descending.
        """
        results: list[dict] = []

        for idx, (_, row) in enumerate(features_df.iterrows()):
            score = self.score_window(row)
            if score["severity"] != "normal":
                if window_titles is not None and idx < len(window_titles):
                    score["sample_titles"] = self._sample_titles(
                        window_titles.iloc[idx]
                    )
                results.append(score)

        results.sort(key=lambda x: x["overall_deviation"], reverse=True)
        return results

    def _sample_titles(self, titles_value) -> list[str]:
        """Extract representative titles from a window's title data."""
        if isinstance(titles_value, str):
            return [titles_value] if titles_value.strip() else []
        if isinstance(titles_value, (list, tuple)):
            return [str(t) for t in titles_value[:5] if str(t).strip()]
        return []

    def daily_summary(self, features_df: pd.DataFrame) -> dict:
        """Generate a daily summary with anomaly count and trend.

        Returns:
            dict with: total_windows, anomaly_count, anomaly_ratio,
            severity_counts, average_deviation, most_anomalous_hour
        """
        if features_df.empty:
            return {
                "total_windows": 0, "anomaly_count": 0,
                "anomaly_ratio": 0.0, "severity_counts": {},
                "average_deviation": 0.0, "most_anomalous_hour": None,
            }

        severities: dict[str, int] = {}
        total_deviation = 0.0
        anomaly_hours: dict[int, float] = {}
        anomaly_count = 0

        for _, row in features_df.iterrows():
            score = self.score_window(row)
            total_deviation += score["overall_deviation"]
            sev = score["severity"]
            severities[sev] = severities.get(sev, 0) + 1
            if sev != "normal":
                anomaly_count += 1
                h = score["hour_of_day"]
                anomaly_hours[h] = anomaly_hours.get(h, 0.0) + score["overall_deviation"]

        n = len(features_df)
        most_anomalous = max(anomaly_hours, key=anomaly_hours.get) if anomaly_hours else None

        return {
            "total_windows": n,
            "anomaly_count": anomaly_count,
            "anomaly_ratio": round(anomaly_count / n, 3) if n > 0 else 0.0,
            "severity_counts": severities,
            "average_deviation": round(total_deviation / n, 3) if n > 0 else 0.0,
            "most_anomalous_hour": most_anomalous,
        }
