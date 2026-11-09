"""Tests for ``services/intervention_policy.py`` — Thompson-sampling selection."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from mindflow.domain.procrastination import ProcrastinationType
from mindflow.services.intervention_policy import (
    ALLOWED_INTERVENTIONS,
    HALF_LIFE_DAYS,
    MIN_BUCKET_OBSERVATIONS,
    PRIMARY_PRIOR_ALPHA,
    PRIOR_ALPHA,
    PRIOR_BETA,
    ArmPosterior,
    candidate_types_for,
    context_bucket,
    posteriors_from_logs,
    rationale_payload,
    reward_of,
    rule_selection,
    selection_evidence,
    state_band,
    thompson_select,
    time_band,
)

_NOW = datetime(2026, 11, 9, 14, 0, tzinfo=UTC)
_BUCKET = "afternoon|distracted"


def _row(
    intervention_type: str,
    *,
    rating: str | None = None,
    response: str | None = None,
    response_source: str | None = None,
    age_days: float = 0.0,
    bucket: str | None = _BUCKET,
) -> dict[str, object]:
    context: dict[str, object] | None = None
    if bucket is not None:
        context = {"rationale": {"context_bucket": bucket}}
    return {
        "intervention_type": intervention_type,
        "user_response": response,
        "feedback_rating": rating,
        "response_source": response_source,
        "triggered_at": (_NOW - timedelta(days=age_days)).isoformat(),
        "context_json": context,
    }


class TestBuckets:
    @pytest.mark.parametrize(
        ("hour", "expected"),
        [
            (6, "morning"),
            (11, "morning"),
            (12, "afternoon"),
            (17, "afternoon"),
            (18, "evening"),
            (22, "evening"),
            (23, "night"),
            (4, "night"),
        ],
    )
    def test_time_band_boundaries(self, hour: int, expected: str) -> None:
        assert time_band(hour) == expected

    def test_state_band_never_invents_a_prediction(self) -> None:
        assert state_band(None) == "unknown"
        assert state_band(0.9) == "focused"
        assert state_band(0.1) == "distracted"

    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.60, "focused"),   # threshold inclusive
            (0.59, "uncertain"),
            (0.41, "uncertain"),
            (0.40, "distracted"),  # threshold inclusive
        ],
    )
    def test_uncertain_band_between_the_thresholds(
        self, probability: float, expected: str
    ) -> None:
        assert state_band(probability) == expected

    def test_context_bucket_composition(self) -> None:
        assert context_bucket(14, 0.2) == "afternoon|distracted"
        assert context_bucket(14, 0.5) == "afternoon|uncertain"
        assert context_bucket(23, None) == "night|unknown"


class TestRewardMapping:
    def test_helpful_or_accepted_is_a_full_success(self) -> None:
        assert reward_of(None, "helpful") == (1.0, 0.0)
        assert reward_of("accepted", None) == (1.0, 0.0)

    def test_annoying_or_dismissed_is_a_full_failure(self) -> None:
        assert reward_of(None, "annoying") == (0.0, 1.0)
        assert reward_of("dismissed", None) == (0.0, 1.0)
        assert reward_of("ignored", None) == (0.0, 1.0)

    def test_neutral_rating_is_half_each(self) -> None:
        assert reward_of(None, "neutral") == (0.5, 0.5)

    def test_auto_ignored_is_a_half_failure(self) -> None:
        assert reward_of("ignored", None, "auto") == (0.0, 0.5)
        # An explicit human "ignored" is a real refusal, not a timeout.
        assert reward_of("ignored", None, "human") == (0.0, 1.0)

    def test_rating_overrides_the_response(self) -> None:
        assert reward_of("dismissed", "helpful") == (1.0, 0.0)
        assert reward_of("accepted", "annoying") == (0.0, 1.0)

    def test_silence_carries_no_signal(self) -> None:
        assert reward_of(None, None) is None


class TestPosteriors:
    def test_cold_start_uses_documented_priors(self) -> None:
        arms = ALLOWED_INTERVENTIONS[ProcrastinationType.TASK_AVERSION]
        posteriors = posteriors_from_logs(
            [], candidates=arms, bucket=_BUCKET, now=_NOW
        )
        primary, secondary = arms
        assert posteriors[primary].alpha == PRIMARY_PRIOR_ALPHA
        assert posteriors[primary].beta == PRIOR_BETA
        assert posteriors[secondary].alpha == PRIOR_ALPHA
        assert posteriors[secondary].beta == PRIOR_BETA

    def test_counts_accumulate_for_the_matching_bucket(self) -> None:
        arms = ALLOWED_INTERVENTIONS[ProcrastinationType.TASK_AVERSION]
        rows = [
            _row("task_breakdown", rating="helpful"),
            _row("task_breakdown", rating="helpful"),
            _row("task_breakdown", rating="annoying"),
        ]
        posteriors = posteriors_from_logs(
            rows, candidates=arms, bucket=_BUCKET, now=_NOW
        )
        arm = posteriors["task_breakdown"]
        # Bucket has 3 observations (== MIN), so its own counts are used.
        assert arm.alpha == pytest.approx(PRIMARY_PRIOR_ALPHA + 2.0, abs=1e-3)
        assert arm.beta == pytest.approx(PRIOR_BETA + 1.0, abs=1e-3)

    def test_other_bucket_observations_do_not_count_directly(self) -> None:
        arms = ALLOWED_INTERVENTIONS[ProcrastinationType.TASK_AVERSION]
        rows = [
            _row("task_breakdown", rating="annoying", bucket="night|focused")
            for _ in range(10)
        ]
        posteriors = posteriors_from_logs(
            rows, candidates=arms, bucket=_BUCKET, now=_NOW
        )
        arm = posteriors["task_breakdown"]
        # Bucket itself has 0 observations → pooled counts × POOLING_WEIGHT.
        assert arm.beta == pytest.approx(PRIOR_BETA + 0.5 * 10.0, abs=1e-3)
        assert arm.alpha == pytest.approx(PRIMARY_PRIOR_ALPHA, abs=1e-3)

    def test_decay_halves_weight_after_one_half_life(self) -> None:
        arms = ALLOWED_INTERVENTIONS[ProcrastinationType.TASK_AVERSION]
        # Twice as many half-life-old observations weigh the same as fresh
        # ones — and both buckets clear MIN_BUCKET_OBSERVATIONS, so the
        # comparison is not confounded by pooling.
        fresh = posteriors_from_logs(
            [_row("task_breakdown", rating="helpful") for _ in range(6)],
            candidates=arms,
            bucket=_BUCKET,
            now=_NOW,
        )["task_breakdown"]
        aged = posteriors_from_logs(
            [
                _row("task_breakdown", rating="helpful", age_days=HALF_LIFE_DAYS)
                for _ in range(12)
            ],
            candidates=arms,
            bucket=_BUCKET,
            now=_NOW,
        )["task_breakdown"]
        assert fresh.alpha == pytest.approx(PRIMARY_PRIOR_ALPHA + 6.0, abs=1e-3)
        assert aged.alpha == pytest.approx(PRIMARY_PRIOR_ALPHA + 6.0, abs=1e-2)

    def test_thin_bucket_falls_back_to_pooled_counts(self) -> None:
        arms = ALLOWED_INTERVENTIONS[ProcrastinationType.TASK_AVERSION]
        rows = [
            _row("task_breakdown", rating="helpful"),  # in-bucket
            _row("task_breakdown", rating="helpful", bucket="morning|focused"),
        ]
        posteriors = posteriors_from_logs(
            rows, candidates=arms, bucket=_BUCKET, now=_NOW
        )
        # 1 in-bucket observation < MIN_BUCKET_OBSERVATIONS → pooled × 0.5
        # (2 observations total → 1.0 success).
        assert posteriors["task_breakdown"].alpha == pytest.approx(
            PRIMARY_PRIOR_ALPHA + 0.5 * 2.0, abs=1e-3
        )
        assert MIN_BUCKET_OBSERVATIONS == 3.0

    def test_posterior_mean_is_alpha_over_total(self) -> None:
        posterior = ArmPosterior("nudge", alpha=3.0, beta=1.0)
        assert posterior.posterior_mean == pytest.approx(0.75)


class TestSelection:
    def test_candidates_follow_the_type_order(self) -> None:
        types = (ProcrastinationType.IMPULSIVITY, ProcrastinationType.TASK_AVERSION)
        assert candidate_types_for(types) == (
            "environment_optimization",
            "nudge",
            "task_breakdown",
        )

    def test_unknown_type_yields_no_candidates(self) -> None:
        assert candidate_types_for(()) == ()
        with pytest.raises(ValueError):
            thompson_select(
                (), bucket=_BUCKET, posteriors={}, rng=random.Random(0)
            )

    def test_selection_is_reproducible_with_a_seed(self) -> None:
        types = (ProcrastinationType.TASK_AVERSION,)
        posteriors = {
            "task_breakdown": ArmPosterior("task_breakdown", 2.0, 2.0),
            "nudge": ArmPosterior("nudge", 2.0, 2.0),
        }
        first = thompson_select(
            types, bucket=_BUCKET, posteriors=posteriors, rng=random.Random(7)
        )
        second = thompson_select(
            types, bucket=_BUCKET, posteriors=posteriors, rng=random.Random(7)
        )
        assert first == second

    def test_selection_only_picks_allowed_candidates(self) -> None:
        types = (ProcrastinationType.IMPULSIVITY,)
        posteriors = {
            "environment_optimization": ArmPosterior(
                "environment_optimization", 1.0, 9.0
            ),
            "nudge": ArmPosterior("nudge", 9.0, 1.0),
        }
        selection = thompson_select(
            types, bucket=_BUCKET, posteriors=posteriors, rng=random.Random(1)
        )
        assert selection.intervention_type in ALLOWED_INTERVENTIONS[
            ProcrastinationType.IMPULSIVITY
        ]
        assert selection.intervention_type == "nudge"
        assert sum(1 for c in selection.candidates if c.chosen) == 1

    def test_rule_selection_bypasses_exploration(self) -> None:
        types = (
            ProcrastinationType.PERFECTIONISM,
            ProcrastinationType.TASK_AVERSION,
        )
        selection = rule_selection(types, bucket=_BUCKET)
        assert selection.selection == "rule"
        assert selection.intervention_type == "task_breakdown"
        assert selection.candidates[0].chosen

    def test_rationale_payload_matches_contract(self) -> None:
        types = (ProcrastinationType.DECISIONAL,)
        selection = thompson_select(
            types,
            bucket=_BUCKET,
            posteriors={
                "smart_prioritization": ArmPosterior("smart_prioritization", 3.0, 1.0),
                "nudge": ArmPosterior("nudge", 1.0, 3.0),
            },
            rng=random.Random(3),
        )
        payload = rationale_payload(selection, types)
        assert set(payload) == {
            "procrastination_type",
            "intervention_type",
            "selection",
            "context_bucket",
            "candidates",
            "evidence",
            "note",
        }
        assert payload["procrastination_type"] == "decisional"
        assert payload["selection"] == "thompson_sampling"
        assert set(payload["candidates"][0]) == {
            "intervention_type",
            "label",
            "alpha",
            "beta",
            "posterior_mean",
            "sample",
            "chosen",
        }
        assert 1 <= len(payload["evidence"]) <= 4

    def test_evidence_is_capped_at_four_lines(self) -> None:
        types = (ProcrastinationType.IMPULSIVITY, ProcrastinationType.TASK_AVERSION)
        selection = thompson_select(
            types, bucket=_BUCKET, posteriors={}, rng=random.Random(0)
        )
        assert len(selection_evidence(selection, types)) <= 4


class TestSimulation:
    """300 rounds against a fixed-rate simulated user (deterministic seed)."""

    def test_bandit_concentrates_on_the_best_arm(self) -> None:
        # Per-arm true helpful rates (orchestrator-specified 0.2 / 0.6).
        # ``nudge`` is best; the primary theory candidate is worse, so
        # success cannot come from the prior's head start.
        true_rate = {"task_breakdown": 0.2, "nudge": 0.6}
        types = (ProcrastinationType.TASK_AVERSION,)
        arms = ALLOWED_INTERVENTIONS[ProcrastinationType.TASK_AVERSION]
        rng = random.Random(20261109)
        rows: list[dict[str, object]] = []
        choices: list[str] = []

        for _ in range(300):
            posteriors = posteriors_from_logs(
                rows, candidates=arms, bucket=_BUCKET, now=_NOW
            )
            selection = thompson_select(
                types, bucket=_BUCKET, posteriors=posteriors, rng=rng
            )
            chosen = selection.intervention_type
            choices.append(chosen)
            helpful = rng.random() < true_rate[chosen]
            rows.append(
                _row(chosen, rating="helpful" if helpful else "annoying")
            )

        best_share = choices[-100:].count("nudge") / 100.0
        assert best_share >= 0.70, f"best arm chosen {best_share:.0%} in last 100"
