"""Multi-signal weak supervision for focus/distraction labeling.

Replaces single-rule pseudo-labeling with consensus from multiple weak
supervision signals, each contributing a weighted vote. Labels come with
confidence scores so downstream models can weight training samples.
"""

import numpy as np
import pandas as pd


class LabelingSignal:
    """One weak supervision signal that votes focus(1) or distraction(0).

    Each signal returns (vote, confidence) where confidence ∈ [0, 1].
    """

    def __call__(self, row: pd.Series) -> tuple[int, float]:
        raise NotImplementedError


class ProductivitySignal(LabelingSignal):
    """High productivity_ratio + low switch_freq → focus."""

    def __call__(self, row: pd.Series) -> tuple[int, float]:
        pr = float(row.get("productivity_ratio", 0.5))
        sf = float(row.get("switch_frequency", 30))
        idle = float(row.get("idle_ratio", 0))

        if idle > 0.7:
            return 1, 0.1

        if pr > 0.8 and sf < 10:
            return 1, 0.9
        if pr > 0.6 and sf < 20:
            return 1, 0.7
        if pr < 0.2:
            return 0, 0.8
        if pr < 0.4 and sf > 30:
            return 0, 0.6
        return 1 if pr >= 0.5 else 0, 0.4


class SwitchFrequencySignal(LabelingSignal):
    """Rapid switching → distraction. Stable single-app → focus."""

    def __call__(self, row: pd.Series) -> tuple[int, float]:
        sf = float(row.get("switch_frequency", 30))
        unique = float(row.get("unique_app_count", 5))

        if sf < 5:
            return 1, 0.85
        if sf < 10 and unique <= 3:
            return 1, 0.7
        if sf > 40:
            return 0, 0.85
        if sf > 25:
            return 0, 0.65
        return 1 if sf < 15 else 0, 0.35


class TimeContextSignal(LabelingSignal):
    """Time-of-day prior: mornings tend to be more productive.

    9-12am and 2-5pm are focus-favorable windows on weekdays.
    """

    FOCUS_HOURS = {9, 10, 11, 14, 15, 16, 17}
    DISTRACTION_HOURS = {0, 1, 2, 22, 23}

    def __call__(self, row: pd.Series) -> tuple[int, float]:
        hour = int(row.get("hour_of_day", 12))
        dow = int(row.get("day_of_week", 0))

        if dow >= 5:
            return 1, 0.3

        if hour in self.FOCUS_HOURS:
            return 1, 0.5
        if hour in self.DISTRACTION_HOURS:
            return 0, 0.5
        return 1, 0.35


class ApplicationDiversitySignal(LabelingSignal):
    """Too many different apps in a window → likely distracted.

    Monitors the ratio of unique apps to total samples. If a user opens
    8+ different apps in one 30-min window with even distribution, it
    signals context-switching overload.
    """

    def __call__(self, row: pd.Series) -> tuple[int, float]:
        unique = float(row.get("unique_app_count", 3))
        max_dur = float(row.get("max_app_duration", 300))
        idle = float(row.get("idle_ratio", 0))

        if idle > 0.5:
            return 1, 0.2

        if unique <= 2:
            return 1, 0.75
        if unique <= 3 and max_dur > 600:
            return 1, 0.65
        if unique >= 6:
            return 0, 0.6
        if unique >= 4 and max_dur < 300:
            return 0, 0.55
        return 1 if unique <= 4 else 0, 0.3


class EntertainmentDominanceSignal(LabelingSignal):
    """Entertainment or social ratio dominates → strong distraction signal."""

    def __call__(self, row: pd.Series) -> tuple[int, float]:
        er = float(row.get("entertainment_ratio", 0))
        sr = float(row.get("social_ratio", 0))

        if er > 0.5 or sr > 0.5:
            return 0, 0.9
        if er > 0.3 or sr > 0.3:
            return 0, 0.7
        if er < 0.05 and sr < 0.05:
            return 1, 0.6
        return 1, 0.35


class ConsensusLabeler:
    """Aggregates multiple weak signals into a single label with confidence.

    Each signal votes independently. The consensus label is the
    weighted majority vote. Confidence = agreement level among signals.
    """

    def __init__(self, signals: list[LabelingSignal] | None = None):
        self.signals = signals or [
            ProductivitySignal(),
            SwitchFrequencySignal(),
            TimeContextSignal(),
            ApplicationDiversitySignal(),
            EntertainmentDominanceSignal(),
            TitleBasedSignal(),
        ]

    def label_single(self, row: pd.Series) -> tuple[int, float]:
        votes: list[tuple[int, float]] = [s(row) for s in self.signals]

        focus_weight = 0.0
        dist_weight = 0.0
        for vote, confidence in votes:
            if vote == 1:
                focus_weight += confidence
            else:
                dist_weight += confidence

        total_weight = focus_weight + dist_weight
        if total_weight == 0:
            return 1, 0.0

        label = 1 if focus_weight >= dist_weight else 0
        majority_weight = max(focus_weight, dist_weight)
        weighted_agreement = majority_weight / total_weight
        confidence = max(0.0, (weighted_agreement - 0.5) * 2.0)

        return label, round(confidence, 4)

    def label_dataframe(
        self, features_df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Label all rows in a feature DataFrame.

        Returns:
            labels: binary array (1=focus, 0=distraction)
            confidences: float array ∈ [0,1]
        """
        labels: list[int] = []
        confidences: list[float] = []

        for _, row in features_df.iterrows():
            label, conf = self.label_single(row)
            labels.append(label)
            confidences.append(conf)

        return np.array(labels, dtype=int), np.array(confidences, dtype=float)

    def label_with_confidence_split(
        self, features_df: pd.DataFrame, min_confidence: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Label and split into high-confidence and low-confidence sets.

        High-confidence samples → supervised training.
        Low-confidence samples → semi-supervised / self-training.
        """
        all_labels, all_confs = self.label_dataframe(features_df)

        high_mask = all_confs >= min_confidence
        low_mask = ~high_mask

        return (
            all_labels[high_mask],
            all_confs[high_mask],
            all_labels[low_mask],
            all_confs[low_mask],
        )

    def signal_breakdown(self, row: pd.Series) -> dict:
        """Return per-signal votes for inspection."""
        return {
            type(s).__name__: {"vote": v, "confidence": c}
            for s in self.signals
            for v, c in [s(row)]
        }


class TitleBasedSignal(LabelingSignal):
    """Focus signal from window title content — no app classification.

    Uses objective title features: code file extensions, document extensions,
    URL domains, meeting keywords, entertainment patterns.
    """

    def __call__(self, row: pd.Series) -> tuple[int, float]:
        code_r = float(row.get("title_code_ratio", 0))
        doc_r = float(row.get("title_doc_ratio", 0))
        url_r = float(row.get("title_url_ratio", 0))
        meeting_r = float(row.get("title_meeting_ratio", 0))
        entertain_r = float(row.get("title_entertainment_ratio", 0))

        has_title_data = (code_r + doc_r + url_r + meeting_r + entertain_r) > 0

        if not has_title_data:
            return 1, 0.1  # no title data → abstain with very low confidence

        focus_signals = code_r + doc_r + meeting_r * 0.5
        distract_signals = entertain_r

        if focus_signals > 0.5:
            return 1, 0.85
        if focus_signals > 0.2:
            return 1, 0.6
        if distract_signals > 0.3:
            return 0, 0.8
        if distract_signals > 0.1:
            return 0, 0.55
        if url_r > 0.5:
            return 1, 0.3  # browser but no clear signal
        return 1, 0.25
