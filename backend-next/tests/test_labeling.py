"""Tests for multi-signal consensus labeling (pure stdlib, no pandas)."""

from __future__ import annotations

from mindflow.domain.labeling import (
    ApplicationDiversitySignal,
    ConsensusLabeler,
    EntertainmentDominanceSignal,
    ProductivitySignal,
    SwitchFrequencySignal,
    TimeContextSignal,
    TitleBasedSignal,
)


def _row(**overrides) -> dict[str, float]:
    """Create a minimal feature row with defaults."""
    defaults: dict[str, float] = {
        "productivity_ratio": 0.5,
        "switch_frequency": 15.0,
        "unique_app_count": 4.0,
        "max_app_duration": 600.0,
        "idle_ratio": 0.05,
        "entertainment_ratio": 0.2,
        "social_ratio": 0.1,
        "title_code_ratio": 0.0,
        "title_doc_ratio": 0.0,
        "title_url_ratio": 0.0,
        "title_meeting_ratio": 0.0,
        "title_entertainment_ratio": 0.0,
        "hour_of_day": 12.0,
        "day_of_week": 0.0,
    }
    defaults.update(overrides)
    return defaults


# ---- ProductivitySignal ----


class TestProductivitySignal:
    def test_strong_focus(self):
        signal = ProductivitySignal()
        vote, conf = signal(_row(productivity_ratio=0.9, switch_frequency=5.0, idle_ratio=0.05))
        assert vote == 1
        assert conf > 0.7

    def test_clear_distraction(self):
        signal = ProductivitySignal()
        vote, conf = signal(_row(productivity_ratio=0.1, switch_frequency=35.0, idle_ratio=0.1))
        assert vote == 0
        assert conf > 0.5

    def test_moderate_focus(self):
        signal = ProductivitySignal()
        vote, conf = signal(_row(productivity_ratio=0.7, switch_frequency=15.0, idle_ratio=0.05))
        assert vote == 1
        assert conf == 0.7

    def test_high_idle_abstains(self):
        signal = ProductivitySignal()
        vote, conf = signal(_row(idle_ratio=0.8, productivity_ratio=0.1))
        assert vote == 1
        assert conf == 0.1

    def test_low_productivity_borderline(self):
        signal = ProductivitySignal()
        vote, conf = signal(_row(productivity_ratio=0.3, switch_frequency=35.0))
        assert vote == 0
        assert conf == 0.6

    def test_ambiguous_middle(self):
        signal = ProductivitySignal()
        vote, conf = signal(_row(productivity_ratio=0.5, switch_frequency=25.0))
        assert vote == 1  # pr >= 0.5
        assert conf == 0.4


# ---- SwitchFrequencySignal ----


class TestSwitchFrequencySignal:
    def test_stable_focus(self):
        signal = SwitchFrequencySignal()
        vote, conf = signal(_row(switch_frequency=3.0, unique_app_count=2.0))
        assert vote == 1
        assert conf > 0.7

    def test_rapid_distraction(self):
        signal = SwitchFrequencySignal()
        vote, conf = signal(_row(switch_frequency=50.0, unique_app_count=8.0))
        assert vote == 0
        assert conf > 0.7

    def test_moderate_switch_focus(self):
        signal = SwitchFrequencySignal()
        vote, conf = signal(_row(switch_frequency=8.0, unique_app_count=2.0))
        assert vote == 1
        assert conf > 0.5

    def test_high_switch_distraction(self):
        signal = SwitchFrequencySignal()
        vote, conf = signal(_row(switch_frequency=30.0))
        assert vote == 0
        assert conf > 0.5

    def test_ambiguous(self):
        signal = SwitchFrequencySignal()
        vote, conf = signal(_row(switch_frequency=20.0))
        assert conf == 0.35


# ---- TimeContextSignal ----


class TestTimeContextSignal:
    def test_morning_weekday(self):
        signal = TimeContextSignal()
        vote, _ = signal(_row(hour_of_day=10.0, day_of_week=2.0))
        assert vote == 1

    def test_afternoon_weekday(self):
        signal = TimeContextSignal()
        vote, _ = signal(_row(hour_of_day=15.0, day_of_week=2.0))
        assert vote == 1

    def test_late_night(self):
        signal = TimeContextSignal()
        vote, _ = signal(_row(hour_of_day=1.0, day_of_week=2.0))
        assert vote == 0

    def test_weekend_low_confidence(self):
        signal = TimeContextSignal()
        _, conf = signal(_row(hour_of_day=10.0, day_of_week=6.0))
        assert conf <= 0.3

    def test_neutral_hour(self):
        signal = TimeContextSignal()
        vote, conf = signal(_row(hour_of_day=13.0, day_of_week=2.0))
        assert vote == 1
        assert conf == 0.35

    def test_weekend_saturday(self):
        signal = TimeContextSignal()
        vote, conf = signal(_row(hour_of_day=10.0, day_of_week=5.0))
        assert conf <= 0.3

    def test_weekday_evening_distraction(self):
        signal = TimeContextSignal()
        vote, conf = signal(_row(hour_of_day=23.0, day_of_week=0.0))
        assert vote == 0
        assert conf == 0.5


# ---- ApplicationDiversitySignal ----


class TestApplicationDiversitySignal:
    def test_low_diversity_focus(self):
        signal = ApplicationDiversitySignal()
        vote, conf = signal(_row(unique_app_count=2.0, max_app_duration=800.0, idle_ratio=0.05))
        assert vote == 1
        assert conf > 0.7

    def test_high_diversity_distraction(self):
        signal = ApplicationDiversitySignal()
        vote, conf = signal(_row(unique_app_count=8.0, max_app_duration=200.0, idle_ratio=0.05))
        assert vote == 0
        assert conf > 0.5

    def test_moderate_diverse_with_duration(self):
        signal = ApplicationDiversitySignal()
        vote, conf = signal(_row(unique_app_count=3.0, max_app_duration=900.0, idle_ratio=0.05))
        assert vote == 1
        assert conf > 0.5

    def test_high_idle_abstains(self):
        signal = ApplicationDiversitySignal()
        vote, conf = signal(_row(unique_app_count=8.0, idle_ratio=0.7))
        assert vote == 1
        assert conf == 0.2

    def test_medium_diverse_short_duration(self):
        signal = ApplicationDiversitySignal()
        vote, _ = signal(_row(unique_app_count=5.0, max_app_duration=200.0))
        assert vote == 0  # unique >= 4 and max_dur < 300

    def test_ambiguous_diversity(self):
        signal = ApplicationDiversitySignal()
        vote, conf = signal(_row(unique_app_count=4.0, max_app_duration=500.0))
        assert vote == 1  # unique <= 4
        assert conf == 0.3


# ---- EntertainmentDominanceSignal ----


class TestEntertainmentDominanceSignal:
    def test_high_entertainment(self):
        signal = EntertainmentDominanceSignal()
        vote, conf = signal(_row(entertainment_ratio=0.6, social_ratio=0.05))
        assert vote == 0
        assert conf > 0.8

    def test_high_social(self):
        signal = EntertainmentDominanceSignal()
        vote, conf = signal(_row(entertainment_ratio=0.05, social_ratio=0.6))
        assert vote == 0
        assert conf > 0.8

    def test_moderate_entertainment(self):
        signal = EntertainmentDominanceSignal()
        vote, conf = signal(_row(entertainment_ratio=0.4, social_ratio=0.05))
        assert vote == 0
        assert conf > 0.5

    def test_low_entertainment_focus(self):
        signal = EntertainmentDominanceSignal()
        vote, conf = signal(_row(entertainment_ratio=0.02, social_ratio=0.02))
        assert vote == 1
        assert conf > 0.5

    def test_borderline_entertainment(self):
        signal = EntertainmentDominanceSignal()
        vote, conf = signal(_row(entertainment_ratio=0.1, social_ratio=0.1))
        assert vote == 1
        assert conf == 0.35


# ---- TitleBasedSignal ----


class TestTitleBasedSignal:
    def test_high_code_focus(self):
        signal = TitleBasedSignal()
        vote, conf = signal(_row(title_code_ratio=0.6))
        assert vote == 1
        assert conf > 0.7

    def test_high_entertainment_distraction(self):
        signal = TitleBasedSignal()
        vote, conf = signal(_row(title_entertainment_ratio=0.4))
        assert vote == 0
        assert conf > 0.7

    def test_no_title_data(self):
        signal = TitleBasedSignal()
        vote, conf = signal(_row())
        assert vote == 1
        assert conf == 0.1

    def test_mixed_signals_code_dominant(self):
        signal = TitleBasedSignal()
        vote, conf = signal(_row(title_code_ratio=0.3, title_entertainment_ratio=0.2))
        assert vote == 1
        assert conf == 0.6

    def test_mixed_signals_entertainment_dominant(self):
        signal = TitleBasedSignal()
        vote, conf = signal(_row(title_code_ratio=0.1, title_entertainment_ratio=0.2))
        assert vote == 0  # distract_signals=0.2 > 0.1
        assert conf == 0.55

    def test_url_dominant_no_clear_signal(self):
        signal = TitleBasedSignal()
        vote, conf = signal(_row(title_url_ratio=0.6))
        assert vote == 1
        assert conf == 0.3

    def test_meeting_signal(self):
        signal = TitleBasedSignal()
        vote, conf = signal(_row(title_meeting_ratio=0.6, title_code_ratio=0.2))
        # focus_signals = 0.2 + 0 + 0.6*0.5 = 0.5 -> not > 0.5
        # So goes to next check: focus_signals > 0.2 -> True
        assert vote == 1
        assert conf == 0.6


# ---- ConsensusLabeler ----


class TestConsensusLabeler:
    def test_returns_label_and_confidence(self):
        labeler = ConsensusLabeler()
        label, conf = labeler.label_single(
            _row(
                productivity_ratio=0.85,
                switch_frequency=6.0,
                unique_app_count=2.0,
                entertainment_ratio=0.02,
                social_ratio=0.01,
                idle_ratio=0.03,
                max_app_duration=1200.0,
                hour_of_day=10.0,
                day_of_week=2.0,
            )
        )
        assert label in (0, 1)
        assert 0.0 <= conf <= 1.0

    def test_high_agreement_focus(self):
        """All signals agree on focus -> high confidence."""
        labeler = ConsensusLabeler()
        label, conf = labeler.label_single(
            _row(
                productivity_ratio=0.9,
                switch_frequency=3.0,
                unique_app_count=1.0,
                entertainment_ratio=0.0,
                social_ratio=0.0,
                idle_ratio=0.01,
                max_app_duration=1800.0,
                hour_of_day=10.0,
                day_of_week=2.0,
                title_code_ratio=0.7,
            )
        )
        assert label == 1
        assert conf > 0.7

    def test_label_dataframe(self):
        labeler = ConsensusLabeler()
        rows = [
            _row(
                productivity_ratio=0.85,
                switch_frequency=5.0,
                unique_app_count=2.0,
                entertainment_ratio=0.0,
                social_ratio=0.01,
                idle_ratio=0.02,
                max_app_duration=1500.0,
                hour_of_day=10.0,
                day_of_week=2.0,
            ),
            _row(
                productivity_ratio=0.2,
                switch_frequency=40.0,
                unique_app_count=7.0,
                entertainment_ratio=0.5,
                social_ratio=0.3,
                idle_ratio=0.05,
                max_app_duration=200.0,
                hour_of_day=23.0,
                day_of_week=3.0,
            ),
            _row(
                productivity_ratio=0.5,
                switch_frequency=15.0,
                unique_app_count=4.0,
                entertainment_ratio=0.1,
                social_ratio=0.1,
                idle_ratio=0.1,
                max_app_duration=500.0,
                hour_of_day=15.0,
                day_of_week=5.0,
            ),
        ]
        labels, confidences = labeler.label_dataframe(rows)
        assert len(labels) == 3
        assert len(confidences) == 3
        assert all(x in (0, 1) for x in labels)
        assert all(0.0 <= c <= 1.0 for c in confidences)

    def test_signal_breakdown(self):
        labeler = ConsensusLabeler()
        breakdown = labeler.signal_breakdown(
            _row(
                productivity_ratio=0.7,
                switch_frequency=10.0,
                unique_app_count=3.0,
                entertainment_ratio=0.1,
                social_ratio=0.1,
                idle_ratio=0.05,
                max_app_duration=800.0,
                hour_of_day=14.0,
                day_of_week=3.0,
            )
        )
        assert len(breakdown) == len(labeler.signals)
        for _, result in breakdown.items():
            assert "vote" in result
            assert "confidence" in result
            assert result["vote"] in (0.0, 1.0)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_rows(self):
        labeler = ConsensusLabeler()
        labels, confs = labeler.label_dataframe([])
        assert labels == []
        assert confs == []

    def test_confidence_split(self):
        labeler = ConsensusLabeler()
        rows = [
            _row(
                productivity_ratio=0.9,
                switch_frequency=3.0,
                unique_app_count=1.0,
                entertainment_ratio=0.0,
                social_ratio=0.0,
                idle_ratio=0.01,
                max_app_duration=1800.0,
                hour_of_day=10.0,
                day_of_week=2.0,
                title_code_ratio=0.7,
            ),
            _row(
                productivity_ratio=0.5,
                switch_frequency=15.0,
                unique_app_count=4.0,
                entertainment_ratio=0.1,
                social_ratio=0.1,
                idle_ratio=0.1,
                max_app_duration=500.0,
                hour_of_day=12.0,
                day_of_week=0.0,
            ),
        ]
        high_labels, high_confs, low_labels, low_confs = labeler.label_with_confidence_split(
            rows, min_confidence=0.5
        )
        assert len(high_labels) > 0 or len(low_labels) > 0
        assert len(high_labels) == len(high_confs)
        assert len(low_labels) == len(low_confs)

    def test_custom_signals(self):
        custom = [ProductivitySignal(), SwitchFrequencySignal()]
        labeler = ConsensusLabeler(signals=custom)
        assert len(labeler.signals) == 2
        label, conf = labeler.label_single(
            _row(
                productivity_ratio=0.9,
                switch_frequency=3.0,
            )
        )
        assert label == 1
        assert conf >= 0.0

    def test_zero_weight_not_crash(self):
        """When no signals vote (all abstain with 0 weight), should not crash."""
        labeler = ConsensusLabeler(signals=[])
        label, conf = labeler.label_single(_row())
        assert label == 1
        assert conf == 0.0

    def test_all_focus_signals(self):
        """When every signal votes focus."""
        labeler = ConsensusLabeler(
            signals=[
                ProductivitySignal(),
                SwitchFrequencySignal(),
            ]
        )
        row = _row(
            productivity_ratio=0.9,
            switch_frequency=3.0,
            unique_app_count=1.0,
            idle_ratio=0.01,
            max_app_duration=1800.0,
        )
        label, conf = labeler.label_single(row)
        assert label == 1
        assert conf > 0.0

    def test_title_based_in_default_signals(self):
        """TitleBasedSignal should be included by default."""
        labeler = ConsensusLabeler()
        signal_types = {type(s).__name__ for s in labeler.signals}
        assert "TitleBasedSignal" in signal_types
