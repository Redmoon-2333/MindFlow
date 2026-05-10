"""Package behavior analysis results into LLM-ready structured context.

Converts baseline statistics, deviation scores, and window titles into
a compact JSON payload (~500 tokens) optimized for LLM CBT attribution.
"""

import json
from datetime import date, datetime, timezone

from mindflow.analyzer.baseline import BaselineModel
from mindflow.analyzer.deviation import DeviationDetector


class LLMContextPacker:
    """Packages ML outputs into a structured JSON context for LLM consumption.

    The output is designed to be:
    - Compact (~500 tokens for typical daily report)
    - Objective (facts and statistics, no interpretations)
    - Evidence-rich (sample window titles for anomalies)
    """

    def pack_daily_report(
        self,
        baseline: BaselineModel,
        anomalies: list[dict],
        daily_summary: dict,
        focus_score: float,
        top_apps: list[dict],
        focus_trend: list[dict] | None = None,
        hmm_insights: dict | None = None,
    ) -> str:
        """Pack a full daily report for LLM.

        Returns a JSON string ready to be appended to the LLM prompt.
        """
        report_date = date.today().isoformat()

        payload = {
            "report_date": report_date,
            "user_profile": self._pack_baseline_summary(baseline),
            "today_summary": {
                "focus_score": focus_score,
                "top_applications": top_apps[:5],
                "total_active_windows": daily_summary.get("total_windows", 0),
                "anomaly_count": daily_summary.get("anomaly_count", 0),
                "anomaly_ratio": daily_summary.get("anomaly_ratio", 0),
                "most_anomalous_hour": daily_summary.get("most_anomalous_hour"),
            },
            "anomalies": self._pack_anomalies(anomalies[:5]),
            "focus_trend_7d": focus_trend[-7:] if focus_trend else [],
            "behavior_transitions": hmm_insights,
        }

        return json.dumps(payload, ensure_ascii=False, indent=2)

    def pack_intervention_context(
        self,
        baseline: BaselineModel,
        current_anomaly: dict,
        recent_anomalies: list[dict],
    ) -> str:
        """Pack context for a real-time intervention trigger.

        More compact than daily report — focused on the current anomaly.
        """
        payload = {
            "trigger_time": datetime.now(timezone.utc)
            .replace(tzinfo=None)
            .isoformat(),
            "current_anomaly": current_anomaly,
            "recent_pattern": [
                {
                    "time": a.get("window_start", ""),
                    "severity": a.get("severity", ""),
                    "overall_deviation": a.get("overall_deviation", 0),
                }
                for a in recent_anomalies[:3]
            ],
            "baseline_context": self._pack_baseline_summary(baseline),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _pack_baseline_summary(self, baseline: BaselineModel) -> dict:
        """Summarize baseline in 2-3 sentences worth of data."""
        if not baseline.has_sufficient_data(30):
            return {
                "status": "insufficient_data",
                "days_collected": baseline.total_days,
                "message": "Baseline still building. Not enough data for reliable comparison.",
            }

        # Pick a representative hour (e.g., 10am weekday) to show
        rep_hour = 10
        rep_dow = 1  # Tuesday
        rep_stats = baseline.get_stats(rep_hour, rep_dow)

        focus_features = {}
        for feat in ["switch_frequency", "unique_app_count", "max_app_duration"]:
            if feat in rep_stats and rep_stats[feat]["n"] >= 2:
                s = rep_stats[feat]
                focus_features[feat] = {
                    "typical": s["mean"],
                    "range": f"{max(0, s['mean'] - s['std']):.1f} - {s['mean'] + s['std']:.1f}",
                }

        return {
            "status": "ready",
            "days_collected": baseline.total_days,
            "representative_period": {
                "hour": rep_hour,
                "day_of_week": rep_dow,
                "typical_features": focus_features,
                "common_apps": baseline.get_top_apps(rep_hour, rep_dow, limit=3),
            },
        }

    def _pack_anomalies(self, anomalies: list[dict]) -> list[dict]:
        """Pack anomalies with key evidence, keeping it compact."""
        packed = []
        for a in anomalies:
            item = {
                "time": a.get("window_start", ""),
                "severity": a.get("severity", "mild"),
                "overall_deviation": a.get("overall_deviation", 0),
                "key_deviations": a.get("top_deviations", []),
            }
            if a.get("sample_titles"):
                item["sample_titles"] = a["sample_titles"][:3]
            packed.append(item)
        return packed
