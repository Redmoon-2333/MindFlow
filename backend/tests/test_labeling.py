import numpy as np
import pandas as pd

from mindflow.analyzer.labeling import (
    ConsensusLabeler,
    ProductivitySignal,
    SwitchFrequencySignal,
    TimeContextSignal,
    ApplicationDiversitySignal,
    EntertainmentDominanceSignal,
)


def test_productivity_signal_strong_focus():
    signal = ProductivitySignal()
    row = pd.Series({
        "productivity_ratio": 0.9,
        "switch_frequency": 5,
        "idle_ratio": 0.05,
    })
    vote, conf = signal(row)
    assert vote == 1
    assert conf > 0.7


def test_productivity_signal_clear_distraction():
    signal = ProductivitySignal()
    row = pd.Series({
        "productivity_ratio": 0.1,
        "switch_frequency": 35,
        "idle_ratio": 0.1,
    })
    vote, conf = signal(row)
    assert vote == 0
    assert conf > 0.5


def test_switch_frequency_signal_stable():
    signal = SwitchFrequencySignal()
    row = pd.Series({"switch_frequency": 3, "unique_app_count": 2})
    vote, conf = signal(row)
    assert vote == 1
    assert conf > 0.7


def test_switch_frequency_signal_rapid():
    signal = SwitchFrequencySignal()
    row = pd.Series({"switch_frequency": 50, "unique_app_count": 8})
    vote, conf = signal(row)
    assert vote == 0
    assert conf > 0.7


def test_time_context_morning():
    signal = TimeContextSignal()
    row = pd.Series({"hour_of_day": 10, "day_of_week": 2})
    vote, _ = signal(row)
    assert vote == 1


def test_time_context_weekend():
    signal = TimeContextSignal()
    row = pd.Series({"hour_of_day": 10, "day_of_week": 6})
    vote, conf = signal(row)
    assert conf <= 0.3


def test_app_diversity_high():
    signal = ApplicationDiversitySignal()
    row = pd.Series({
        "unique_app_count": 8,
        "max_app_duration": 200,
        "idle_ratio": 0.05,
    })
    vote, conf = signal(row)
    assert vote == 0
    assert conf > 0.5


def test_entertainment_dominance():
    signal = EntertainmentDominanceSignal()
    row = pd.Series({"entertainment_ratio": 0.6, "social_ratio": 0.05})
    vote, conf = signal(row)
    assert vote == 0
    assert conf > 0.8


def test_consensus_labeler_returns_label_and_confidence():
    labeler = ConsensusLabeler()
    row = pd.Series({
        "productivity_ratio": 0.85,
        "switch_frequency": 6,
        "unique_app_count": 2,
        "entertainment_ratio": 0.02,
        "social_ratio": 0.01,
        "idle_ratio": 0.03,
        "max_app_duration": 1200,
        "hour_of_day": 10,
        "day_of_week": 2,
    })
    label, conf = labeler.label_single(row)
    assert label in (0, 1)
    assert 0.0 <= conf <= 1.0


def test_consensus_labeler_high_agreement():
    """All signals agree on focus → high confidence."""
    labeler = ConsensusLabeler()
    row = pd.Series({
        "productivity_ratio": 0.9,
        "switch_frequency": 3,
        "unique_app_count": 1,
        "entertainment_ratio": 0.0,
        "social_ratio": 0.0,
        "idle_ratio": 0.01,
        "max_app_duration": 1800,
        "hour_of_day": 10,
        "day_of_week": 2,
    })
    label, conf = labeler.label_single(row)
    assert label == 1
    assert conf > 0.7


def test_consensus_labeler_label_dataframe():
    labeler = ConsensusLabeler()
    df = pd.DataFrame({
        "productivity_ratio": [0.85, 0.2, 0.5],
        "switch_frequency": [5, 40, 15],
        "unique_app_count": [2, 7, 4],
        "entertainment_ratio": [0.0, 0.5, 0.1],
        "social_ratio": [0.01, 0.3, 0.1],
        "idle_ratio": [0.02, 0.05, 0.1],
        "max_app_duration": [1500, 200, 500],
        "hour_of_day": [10, 23, 15],
        "day_of_week": [2, 3, 5],
    })
    labels, confs = labeler.label_dataframe(df)
    assert len(labels) == 3
    assert len(confs) == 3
    assert all(l in (0, 1) for l in labels)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_consensus_labeler_signal_breakdown():
    labeler = ConsensusLabeler()
    row = pd.Series({
        "productivity_ratio": 0.7,
        "switch_frequency": 10,
        "unique_app_count": 3,
        "entertainment_ratio": 0.1,
        "social_ratio": 0.1,
        "idle_ratio": 0.05,
        "max_app_duration": 800,
        "hour_of_day": 14,
        "day_of_week": 3,
    })
    breakdown = labeler.signal_breakdown(row)
    assert len(breakdown) == len(labeler.signals)
    for sig_name, result in breakdown.items():
        assert "vote" in result
        assert "confidence" in result
        assert result["vote"] in (0, 1)
        assert 0.0 <= result["confidence"] <= 1.0
