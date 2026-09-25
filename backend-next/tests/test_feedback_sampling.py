"""Phase 3.3 regressions: active feedback sampling.

Labels are expensive (they interrupt the user) and not equally informative.  The
sampler asks only when the system is genuinely uncertain or internally
inconsistent, and it records *why* so the value of active sampling can be
measured per feedback rather than assumed.

Every "signal unavailable" case must stay silent: an unknown value may not
fabricate a request.
"""

from __future__ import annotations

import pytest

from mindflow.services.feedback_sampling import (
    DEFAULT_FEEDBACK_SAMPLING_CONFIG,
    TRIGGER_DATA_DRIFT,
    TRIGGER_MODEL_UNCERTAIN,
    TRIGGER_PANEL_DISAGREEMENT,
    TRIGGER_RULE_ML_CONFLICT,
    TRIGGER_TASK_CONTEXT_MISMATCH,
    FeedbackSamplingConfig,
    FeedbackSignals,
    decide_feedback_request,
    information_gain_proxy,
    signals_from_outputs,
    trigger_counts,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Silence by default
# ═══════════════════════════════════════════════════════════════════════════════


def test_confident_consistent_window_is_not_worth_asking_about() -> None:
    decision = decide_feedback_request(FeedbackSignals(
        model_probability=0.95,
        model_top_type="impulsivity",
        rule_top_type="impulsivity",
        rule_confidence=0.9,
        ml_confidence=0.95,
        panel_agreement=0.9,
        task_context_consistent=True,
        drift_psi=0.05,
    ))

    assert decision.should_request is False
    assert decision.reasons == ()
    assert decision.description


def test_no_signals_means_no_request() -> None:
    decision = decide_feedback_request(FeedbackSignals())
    assert decision.should_request is False
    assert decision.reasons == ()


# ═══════════════════════════════════════════════════════════════════════════════
# Individual triggers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("probability", [0.5, 0.35, 0.65, 0.4])
def test_probability_near_the_boundary_triggers_a_request(probability: float) -> None:
    decision = decide_feedback_request(FeedbackSignals(model_probability=probability))
    assert decision.should_request is True
    assert decision.reason_set == {TRIGGER_MODEL_UNCERTAIN}


@pytest.mark.parametrize("probability", [0.34, 0.66, 0.9, 0.05])
def test_confident_probabilities_do_not_trigger(probability: float) -> None:
    decision = decide_feedback_request(FeedbackSignals(model_probability=probability))
    assert decision.should_request is False


def test_rule_and_model_disagreeing_confidently_triggers_a_request() -> None:
    decision = decide_feedback_request(FeedbackSignals(
        model_probability=0.9,
        model_top_type="impulsivity",
        rule_top_type="task_aversion",
        rule_confidence=0.85,
        ml_confidence=0.9,
    ))
    assert decision.reason_set == {TRIGGER_RULE_ML_CONFLICT}


def test_type_mismatch_without_confidence_is_only_uncertainty() -> None:
    """A mismatch nobody is confident about is uncertainty, not a conflict."""
    decision = decide_feedback_request(FeedbackSignals(
        model_probability=0.5,
        model_top_type="impulsivity",
        rule_top_type="task_aversion",
    ))
    assert decision.reason_set == {TRIGGER_MODEL_UNCERTAIN}


def test_panel_disagreement_triggers_a_request() -> None:
    decision = decide_feedback_request(FeedbackSignals(
        model_probability=0.9, panel_agreement=0.2,
    ))
    assert decision.reason_set == {TRIGGER_PANEL_DISAGREEMENT}


def test_task_context_mismatch_triggers_a_request() -> None:
    decision = decide_feedback_request(FeedbackSignals(
        model_probability=0.9, task_context_consistent=False,
    ))
    assert decision.reason_set == {TRIGGER_TASK_CONTEXT_MISMATCH}


def test_drift_triggers_a_request() -> None:
    decision = decide_feedback_request(FeedbackSignals(
        model_probability=0.9, drift_psi=0.6,
    ))
    assert decision.reason_set == {TRIGGER_DATA_DRIFT}


# ═══════════════════════════════════════════════════════════════════════════════
# Combined reasons + configuration
# ═══════════════════════════════════════════════════════════════════════════════


def test_multiple_reasons_are_all_recorded() -> None:
    decision = decide_feedback_request(FeedbackSignals(
        model_probability=0.52,
        model_top_type="impulsivity",
        rule_top_type="task_aversion",
        rule_confidence=0.8,
        ml_confidence=0.8,
        panel_agreement=0.3,
        task_context_consistent=False,
        drift_psi=0.5,
    ))

    assert decision.should_request is True
    assert decision.reason_set == {
        TRIGGER_MODEL_UNCERTAIN,
        TRIGGER_RULE_ML_CONFLICT,
        TRIGGER_PANEL_DISAGREEMENT,
        TRIGGER_TASK_CONTEXT_MISMATCH,
        TRIGGER_DATA_DRIFT,
    }
    assert decision.description.startswith("触发主动标注：")


def test_max_reasons_caps_the_recorded_explanations() -> None:
    decision = decide_feedback_request(
        FeedbackSignals(
            model_probability=0.5,
            panel_agreement=0.1,
            task_context_consistent=False,
            drift_psi=0.9,
        ),
        FeedbackSamplingConfig(max_reasons=2),
    )
    assert len(decision.reasons) == 2


def test_thresholds_are_configurable_not_magic_numbers() -> None:
    strict = FeedbackSamplingConfig(uncertain_band_half_width=0.02)
    assert decide_feedback_request(
        FeedbackSignals(model_probability=0.45), strict,
    ).should_request is False
    assert decide_feedback_request(
        FeedbackSignals(model_probability=0.5), strict,
    ).should_request is True


def test_default_thresholds_match_the_documented_values() -> None:
    assert DEFAULT_FEEDBACK_SAMPLING_CONFIG.uncertain_band_half_width == 0.15
    assert DEFAULT_FEEDBACK_SAMPLING_CONFIG.rule_ml_min_confidence == 0.6
    assert DEFAULT_FEEDBACK_SAMPLING_CONFIG.panel_disagreement_threshold == 0.5
    assert DEFAULT_FEEDBACK_SAMPLING_CONFIG.drift_psi_threshold == 0.25


# ═══════════════════════════════════════════════════════════════════════════════
# Adapters + reporting
# ═══════════════════════════════════════════════════════════════════════════════


def test_signals_are_assembled_from_service_outputs() -> None:
    signals = signals_from_outputs(
        prediction={"focus_probability": 0.55, "top_type": "impulsivity",
                    "confidence": 0.62},
        rule_assessment={
            "procrastination_types": ["task_aversion"],
            "type_confidence": {"task_aversion": 0.77},
        },
        panel_agreement=0.4,
        task_context_consistent=False,
        drift_psi=0.3,
    )

    assert signals.model_probability == pytest.approx(0.55)
    assert signals.model_top_type == "impulsivity"
    assert signals.ml_confidence == pytest.approx(0.62)
    assert signals.rule_top_type == "task_aversion"
    assert signals.rule_confidence == pytest.approx(0.77)
    assert decide_feedback_request(signals).should_request is True


def test_partial_outputs_do_not_fabricate_signals() -> None:
    signals = signals_from_outputs(prediction={}, rule_assessment={})
    assert signals == FeedbackSignals()
    assert decide_feedback_request(signals).should_request is False


def test_information_gain_proxy_reports_targeting() -> None:
    decisions = [
        decide_feedback_request(FeedbackSignals(model_probability=0.5)),
        decide_feedback_request(FeedbackSignals(model_probability=0.5)),
        decide_feedback_request(FeedbackSignals(model_probability=0.99)),
        decide_feedback_request(FeedbackSignals(model_probability=0.99)),
    ]
    report = information_gain_proxy(decisions, baseline_uncertainty=0.25)

    assert report["request_rate"] == pytest.approx(0.5)
    assert report["triggered_uncertainty_rate"] == pytest.approx(1.0)
    assert report["uncertainty_lift"] == pytest.approx(4.0)


def test_information_gain_proxy_handles_no_requests() -> None:
    report = information_gain_proxy(
        [decide_feedback_request(FeedbackSignals(model_probability=0.99))],
        baseline_uncertainty=0.0,
    )
    assert report["request_rate"] == 0.0
    assert report["uncertainty_lift"] == 0.0


def test_trigger_counts_cover_every_trigger() -> None:
    decisions = [
        decide_feedback_request(FeedbackSignals(model_probability=0.5)),
        decide_feedback_request(FeedbackSignals(model_probability=0.95, drift_psi=0.9)),
    ]
    counts = trigger_counts(decisions)

    assert counts[TRIGGER_MODEL_UNCERTAIN] == 1
    assert counts[TRIGGER_DATA_DRIFT] == 1
    assert counts[TRIGGER_RULE_ML_CONFLICT] == 0
