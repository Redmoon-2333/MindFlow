"""Midterm algorithm experiments (E1-E5) — offline, seeded, no network.

Every experiment exercises the **production** implementation of the algorithm
under test; nothing is re-implemented here:

  E1  empirical-Bayes personal baseline   -> ``domain/baseline.py``
  E2  Thompson-sampling intervention policy -> ``services/intervention_policy.py``
  E3  active learning (uncertainty sampling) -> ``services/feedback_sampling.py``
  E4  local explanation fidelity          -> ``train/local_explanation.py``
  E5  evidence-citation validity (mock replay) -> ``python -m mindflow.eval``

Artifacts go to ``data/experiments/<run>/`` (``data/`` is git-ignored):
``summary.json`` (every number) and ``report.md`` (Chinese summary table plus
the data-provenance/limitations section).

Usage::

    uv run python scripts/experiment_midterm_algorithms.py
    uv run python scripts/experiment_midterm_algorithms.py --quick
    uv run python scripts/experiment_midterm_algorithms.py --out data/experiments/manual
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mindflow.domain.baseline import BaselineModel
from mindflow.domain.feature_schema import V2_FEATURE_NAMES
from mindflow.domain.procrastination import ProcrastinationType
from mindflow.services.feedback_sampling import FeedbackSignals, decide_feedback_request
from mindflow.services.intervention_policy import (
    candidate_types_for,
    context_bucket,
    posteriors_from_logs,
    rule_selection,
    thompson_select,
)
from mindflow.time_utils import resolve_timezone
from mindflow.train import local_explanation
from mindflow.train.local_explanation import LocalExplainer
from mindflow.train.synthetic_v2 import _compute_daily_patterns, _extract_archetype_params
from mindflow.train.user_profiles import PROFILES
from mindflow.train.v2 import make_v2_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "experiments" / "20261110_midterm"
LIVE_SMOKE_SOURCE = PROJECT_ROOT / "data" / "experiments" / "20260925_051755" / "summary.json"

#: Business timezone used to bucket windows.  Fixed (not "local") so the
#: (hour, weekday) buckets do not depend on the machine running the script.
TIMEZONE = "Asia/Shanghai"

#: First day of the synthetic window series (same anchor as synthetic_v2).
START_DATE = date(2026, 7, 1)

#: Bootstrap settings shared by every confidence interval in this script.
BOOTSTRAP_RESAMPLES = 2000
CI_LEVEL = 95.0

#: Environment variables that must never reach a child process.
_SECRET_ENV_MARKERS = ("API_KEY", "CHATECNU", "DEEPSEEK")


@dataclass(frozen=True)
class Config:
    """Every knob of the five experiments, so ``--quick`` is one object."""

    e1_archetypes: int
    e1_seeds_per_archetype: int
    e1_days: int
    e1_train_days: int
    e1_sparse_threshold: int
    e1_observation_rate: float
    e1_observation_rate: float
    e2_rounds: int
    e2_seeds: int
    e3_archetypes: int
    e3_days: int
    e3_seeds: int
    e3_start_labels: int
    e3_batch: int
    e3_max_labels: int
    e3_eval_points: tuple[int, ...]
    e4_archetypes: int
    e4_train_days: int
    e4_windows: int
    e4_random_draws: int
    e5_scenario_ids: tuple[str, ...]
    quick: bool

    @classmethod
    def full(cls) -> Config:
        return cls(
            e1_archetypes=8,
            e1_seeds_per_archetype=3,
            e1_days=14,
            e1_train_days=7,
            e1_sparse_threshold=5,
            e1_observation_rate=0.3,
            e2_rounds=300,
            e2_seeds=200,
            e3_archetypes=10,
            e3_days=14,
            e3_seeds=30,
            e3_start_labels=10,
            e3_batch=5,
            e3_max_labels=100,
            e3_eval_points=(10, 25, 50, 100),
            e4_archetypes=4,
            e4_train_days=12,
            e4_windows=200,
            e4_random_draws=20,
            e5_scenario_ids=(),
            quick=False,
        )

    @classmethod
    def quick_mode(cls) -> Config:
        """Tiny parameters: same code paths, a fraction of the samples."""
        return cls(
            e1_archetypes=3,
            e1_seeds_per_archetype=2,
            e1_days=14,
            e1_train_days=7,
            e1_sparse_threshold=5,
            e1_observation_rate=0.3,
            e2_rounds=60,
            e2_seeds=10,
            e3_archetypes=2,
            e3_days=5,
            e3_seeds=1,
            e3_start_labels=10,
            e3_batch=5,
            e3_max_labels=20,
            e3_eval_points=(10, 20),
            e4_archetypes=2,
            e4_train_days=4,
            e4_windows=20,
            e4_random_draws=5,
            e5_scenario_ids=("IMP-001",),
            quick=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_windows(archetype_id: str, days: int, seed: int) -> list[dict[str, Any]]:
    """Reproducible v2 windows for one simulated user.

    ``generate_v2_synthetic_data`` seeds each archetype with
    ``seed + hash(profile_id)``, and ``hash()`` on ``str`` is randomised per
    process (PYTHONHASHSEED), so two invocations of the same command produce
    different windows.  This calls the module's own day simulator with an
    explicit ``random.Random`` instead: the production generator and its
    archetype parameters, but reproducible run to run.
    """
    params = _extract_archetype_params(PROFILES[archetype_id])
    rng = random.Random(seed)
    windows: list[dict[str, Any]] = []
    for offset in range(days):
        windows.extend(_compute_daily_patterns(params, START_DATE + timedelta(days=offset), rng))
    return windows


def _bucket_of(row: dict[str, Any], tz: Any) -> tuple[int, int]:
    """Local (hour, weekday) bucket of a window — the baseline's own rule."""
    moment = datetime.fromisoformat(str(row["window_start_utc"]))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    local = moment.astimezone(tz)
    return local.hour, local.weekday()


def _feature_matrix(rows: list[dict[str, Any]]) -> npt.NDArray[Any]:
    """Feature matrix in the canonical column order (as ``prepare_v2_training_data``)."""
    return np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in V2_FEATURE_NAMES] for row in rows],
        dtype=float,
    )


def _labels(rows: list[dict[str, Any]]) -> npt.NDArray[Any]:
    return np.asarray([int(row["label"]) for row in rows], dtype=int)


def _bootstrap_ci(values: list[float], seed: int) -> tuple[float, float, float]:
    """Mean and percentile 95% CI of *values* (bootstrap over the values)."""
    array = np.asarray([float(v) for v in values], dtype=float)
    if array.size == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(array.mean())
    if array.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [(100.0 - CI_LEVEL) / 2.0, 100.0 - (100.0 - CI_LEVEL) / 2.0])
    return (mean, float(low), float(high))


def _summary_stats(values: list[float], seed: int) -> dict[str, float]:
    mean, low, high = _bootstrap_ci(values, seed)
    return {
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "n": float(len(values)),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _ci_text(stats: dict[str, float], digits: int = 3) -> str:
    if "ci_low" not in stats:
        return "—"
    return f"[{_fmt(stats['ci_low'], digits)}, {_fmt(stats['ci_high'], digits)}]"


# ═══════════════════════════════════════════════════════════════════════════════
# E1 — empirical-Bayes personal baseline vs raw bucket mean
# ═══════════════════════════════════════════════════════════════════════════════

#: Baseline features compared in E1.  ``app_switch_count``, ``idle_ratio`` and
#: ``active_seconds_ratio`` are the canonical activity features the production
#: baseline is built on; the V2 names are exactly the ones ``BaselineModel``
#: tracks (``FEATURE_COLS = V2_FEATURE_NAMES``).
E1_FEATURES: tuple[str, ...] = ("app_switch_count", "idle_ratio", "active_seconds_ratio")


def _bucket_means(
    rows: list[dict[str, Any]], feature: str, tz: Any
) -> dict[tuple[int, int], list[float]]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        grouped.setdefault(_bucket_of(row, tz), []).append(float(row["features"][feature]))
    return grouped


#: The three E1 predictors.  ``eb_fixed_k10`` is the pre-8.25 implementation
#: (explicit ``k=10`` + global robust prior); ``eb_data_k`` is the new default
#: (``k=None``: method-of-moments k-hat + hour-pooled prior).
E1_ARMS: tuple[str, ...] = ("raw", "eb_fixed_k10", "eb_data_k")

#: Seeds of the development run — the set that exposed the flaw.  The
#: confirmation set below is disjoint from these and is the only one the
#: materials cite.
E1_DEV_SEED_BASE = 1000
#: Disjoint base for the confirmation seeds; far enough away that no seed
#: integer can coincide with a development seed.
E1_CONFIRM_SEED_BASE = 7000


def _e1_user(
    archetype_id: str, seed: int, config: Config, tz: Any
) -> dict[str, Any]:
    """Train on the first 7 days, score all three predictors on days 8-14."""
    windows = _generate_windows(archetype_id, config.e1_days, seed)
    # The generator emits all 288 windows of every day, so every (hour,
    # weekday) bucket would hold 12 training windows and "sparse" would be an
    # empty set.  A real collector only sees the windows the machine was on
    # for, so keep a random subset: at ``e1_observation_rate=0.3`` most buckets
    # fall below the sparse threshold, which is the regime the shrinkage prior
    # exists for.  The held-out days are subsampled the same way, keeping the
    # evaluation and the training sample comparable.
    rng = random.Random(seed * 31 + 7)
    windows = [row for row in windows if rng.random() < config.e1_observation_rate]
    days = sorted({str(row["window_start_utc"])[:10] for row in windows})
    split = days[config.e1_train_days]
    train = [row for row in windows if str(row["window_start_utc"])[:10] < split]
    held = [row for row in windows if str(row["window_start_utc"])[:10] >= split]

    baseline = BaselineModel(user_id=1, timezone=TIMEZONE)
    baseline.update(train)

    per_feature: dict[str, dict[str, Any]] = {}
    arm_sse: dict[str, float] = dict.fromkeys(E1_ARMS, 0.0)
    sparse_sse: dict[str, float] = dict.fromkeys(E1_ARMS, 0.0)
    sparse_buckets = 0
    scored_buckets = 0

    for feature in E1_FEATURES:
        train_buckets = _bucket_means(train, feature, tz)
        held_buckets = _bucket_means(held, feature, tz)
        global_mean = statistics.fmean(
            float(row["features"][feature]) for row in train
        )

        feature_sse: dict[str, float] = dict.fromkeys(E1_ARMS, 0.0)
        k_values: list[float] = []
        prior_kinds: Counter[str] = Counter()

        for bucket, values in held_buckets.items():
            target = statistics.fmean(values)
            train_values = train_buckets.get(bucket)
            # ``raw``: the training bucket mean, or the global training mean
            # when this bucket was never observed in training.
            predictions: dict[str, float] = {
                "raw": (
                    statistics.fmean(train_values) if train_values else global_mean
                )
            }
            fixed = baseline.shrunk_expected(feature, bucket[0], bucket[1], k=10)
            data_driven = baseline.shrunk_expected(feature, bucket[0], bucket[1])
            # ``None`` means the feature has never varied in the bucket or in
            # the prior; the global training mean is the only honest answer
            # left, and every arm then gets the same one.
            predictions["eb_fixed_k10"] = (
                fixed["expected"] if fixed is not None else global_mean
            )
            predictions["eb_data_k"] = (
                data_driven["expected"] if data_driven is not None else global_mean
            )
            if data_driven is not None:
                k_values.append(float(data_driven["k"]))
                prior_kinds[str(data_driven["prior"])] += 1

            for arm in E1_ARMS:
                squared = (predictions[arm] - target) ** 2
                feature_sse[arm] += squared
                arm_sse[arm] += squared
                if len(train_values or ()) < config.e1_sparse_threshold:
                    sparse_sse[arm] += squared

            if len(train_values or ()) < config.e1_sparse_threshold:
                sparse_buckets += 1

        count = len(held_buckets) or 1
        scored_buckets += count
        per_feature[feature] = {
            "mse": {arm: feature_sse[arm] / count for arm in E1_ARMS},
            # ``eb_data_k`` is the arm under test; the other two are baselines.
            "relative_reduction_vs_raw": {
                arm: (
                    (feature_sse["raw"] - feature_sse[arm]) / feature_sse["raw"]
                    if feature_sse["raw"] > 0
                    else 0.0
                )
                for arm in E1_ARMS
                if arm != "raw"
            },
            "mean_k": statistics.fmean(k_values) if k_values else None,
            "prior_kinds": dict(prior_kinds),
        }

    return {
        "user": f"{archetype_id}#{seed}",
        "train_windows": len(train),
        "held_windows": len(held),
        "held_buckets": len(_bucket_means(held, E1_FEATURES[0], tz)),
        "scored_buckets": scored_buckets,
        "mse": {arm: arm_sse[arm] / max(1, scored_buckets) for arm in E1_ARMS},
        "per_feature": per_feature,
        "sparse_buckets": sparse_buckets,
        # ``None`` when this user has no sparse held-out bucket at all; the
        # aggregate skips those users rather than inventing a zero.
        "sparse_relative_reduction": {
            arm: (
                (sparse_sse["raw"] - sparse_sse[arm]) / sparse_sse["raw"]
                if sparse_sse["raw"] > 0 and arm != "raw"
                else None
            )
            for arm in E1_ARMS
        },
    }


def _e1_arm_reduction(user: dict[str, Any], arm: str) -> float:
    """One user's pooled relative MSE reduction for *arm* against ``raw``."""
    raw = user["mse"]["raw"]
    return (raw - user["mse"][arm]) / raw if raw > 0 else 0.0


def _e1_seed_set(
    name: str, seeds: list[int], archetypes: list[str], config: Config, tz: Any
) -> dict[str, Any]:
    """Run E1 over one seed set and aggregate every arm."""
    users = [
        _e1_user(archetype, seed, config, tz)
        for archetype in archetypes
        for seed in seeds
    ]

    arms: dict[str, Any] = {
        arm: {"mean_mse": statistics.fmean(user["mse"][arm] for user in users)}
        for arm in E1_ARMS
    }
    for arm in E1_ARMS:
        if arm == "raw":
            continue
        values = [_e1_arm_reduction(user, arm) for user in users]
        arms[arm]["relative_mse_reduction"] = _summary_stats(values, seed=1)
        sparse_values = [
            user["sparse_relative_reduction"][arm]
            for user in users
            if user["sparse_relative_reduction"][arm] is not None
        ]
        arms[arm]["sparse_relative_mse_reduction"] = _summary_stats(
            sparse_values, seed=2
        )
        arms[arm]["users_with_sparse_buckets"] = len(sparse_values)

    per_feature: dict[str, Any] = {}
    for feature in E1_FEATURES:
        k_values = [
            user["per_feature"][feature]["mean_k"]
            for user in users
            if user["per_feature"][feature]["mean_k"] is not None
        ]
        per_feature[feature] = {
            "mean_k": statistics.fmean(k_values) if k_values else None,
            "reduction": {
                arm: _summary_stats(
                    [
                        user["per_feature"][feature]["relative_reduction_vs_raw"][arm]
                        for user in users
                    ],
                    seed=3,
                )
                for arm in ("eb_fixed_k10", "eb_data_k")
            },
        }

    return {
        "name": name,
        "seeds": seeds,
        "users": len(users),
        "arms": arms,
        "per_feature": per_feature,
        "sparse_bucket_share": (
            sum(user["sparse_buckets"] for user in users)
            / max(1, sum(user["scored_buckets"] for user in users))
        ),
        "per_user": users,
    }


def run_e1(config: Config) -> dict[str, Any]:
    """Three-arm E1 over a development and a disjoint confirmation seed set."""
    tz = resolve_timezone(TIMEZONE)
    archetypes = sorted(PROFILES)[: config.e1_archetypes]
    count = config.e1_seeds_per_archetype
    dev_seeds = [E1_DEV_SEED_BASE + index for index in range(count)]
    confirm_seeds = [E1_CONFIRM_SEED_BASE + index for index in range(count)]
    overlap = set(dev_seeds) & set(confirm_seeds)

    dev = _e1_seed_set("dev", dev_seeds, archetypes, config, tz)
    confirm = _e1_seed_set("confirm", confirm_seeds, archetypes, config, tz)

    return {
        "features": list(E1_FEATURES),
        "arms": list(E1_ARMS),
        "archetypes": archetypes,
        "observation_rate": config.e1_observation_rate,
        "sparse_threshold": config.e1_sparse_threshold,
        "shrinkage_k": {
            "eb_fixed_k10": "explicit k=10 + global robust prior (pre-8.25)",
            "eb_data_k": "k=None: method-of-moments k-hat + hour-pooled prior",
        },
        "seed_sets_disjoint": not overlap,
        "cited_seed_set": "confirm",
        "dev": dev,
        "confirm": confirm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# E2 — Thompson sampling vs rule selection vs uniform random
# ═══════════════════════════════════════════════════════════════════════════════

#: Procrastination type whose theory-gated candidate set the bandit ranks.
E2_PROCRASTINATION_TYPE = ProcrastinationType.TASK_AVERSION


def _e2_world(seed: int) -> tuple[list[str], dict[tuple[str, str], float], dict[str, str]]:
    """True acceptance probabilities for one simulated user.

    One probability per (intervention type, context bucket); the gap between
    the best and the second-best arm in a bucket is drawn from [0.10, 0.25],
    and which arm is best is itself random, so a fixed rule cannot win by
    construction.
    """
    rng = random.Random(seed)
    candidates = candidate_types_for([E2_PROCRASTINATION_TYPE])
    buckets = sorted({context_bucket(hour, None) for hour in range(24)})
    table: dict[tuple[str, str], float] = {}
    best_arm: dict[str, str] = {}
    for bucket in buckets:
        gap = rng.uniform(0.10, 0.25)
        best = rng.choice(list(candidates))
        top = rng.uniform(0.55, 0.85)
        best_arm[bucket] = best
        for arm in candidates:
            table[(arm, bucket)] = top if arm == best else max(0.05, top - gap)
    return buckets, table, best_arm


def _e2_rounds(
    seed: int, rounds: int, policy: str
) -> tuple[float, float, float]:
    """One simulated user under one policy: (regret, optimal rate, acceptance)."""
    rng = random.Random(seed * 7 + 1)
    buckets, truth, best_arm = _e2_world(seed)
    candidates = candidate_types_for([E2_PROCRASTINATION_TYPE])
    base_time = datetime(2026, 9, 1, tzinfo=UTC)
    rows: list[dict[str, Any]] = []

    cumulative_regret = 0.0
    accepted = 0
    tail = min(100, rounds)
    tail_hits = 0

    for step in range(rounds):
        hour = step % 24
        bucket = context_bucket(hour, None)
        now = base_time + timedelta(hours=step)

        if policy == "thompson":
            posteriors = posteriors_from_logs(
                rows, candidates=candidates, bucket=bucket, now=now
            )
            selection = thompson_select(
                [E2_PROCRASTINATION_TYPE], bucket=bucket, posteriors=posteriors, rng=rng
            )
            chosen = selection.intervention_type
        elif policy == "rule":
            chosen = rule_selection(
                [E2_PROCRASTINATION_TYPE], bucket=bucket
            ).intervention_type
        else:
            chosen = rng.choice(list(candidates))

        best = best_arm[bucket]
        cumulative_regret += truth[(best, bucket)] - truth[(chosen, bucket)]
        if step >= rounds - tail and chosen == best:
            tail_hits += 1
        success = rng.random() < truth[(chosen, bucket)]
        accepted += int(success)

        rows.append({
            "intervention_type": chosen,
            # The exact fields ``reward_of`` reads: an explicit rating wins
            # over the implicit response, and a response of ``accepted`` /
            # ``dismissed`` maps to (1, 0) / (0, 1).
            "user_response": "accepted" if success else "dismissed",
            "feedback_rating": None,
            "response_source": None,
            "triggered_at": now,
            "context_json": {"rationale": {"context_bucket": bucket}},
        })

    return (cumulative_regret, tail_hits / tail, accepted / rounds)


def run_e2(config: Config) -> dict[str, Any]:
    """Regret / optimal-arm rate / acceptance for the three policies."""
    policies = ("thompson", "rule", "uniform")
    seeds = [2000 + index for index in range(config.e2_seeds)]
    buckets, _, best_arm = _e2_world(seeds[0])
    candidates = [str(arm) for arm in candidate_types_for([E2_PROCRASTINATION_TYPE])]

    raw: dict[str, dict[str, list[float]]] = {
        policy: {"regret": [], "optimal_rate": [], "acceptance": []} for policy in policies
    }
    for seed in seeds:
        for policy in policies:
            regret, optimal_rate, acceptance = _e2_rounds(seed, config.e2_rounds, policy)
            raw[policy]["regret"].append(regret)
            raw[policy]["optimal_rate"].append(optimal_rate)
            raw[policy]["acceptance"].append(acceptance)

    stats = {
        policy: {
            metric: _summary_stats(values, seed=4 + index)
            for index, (metric, values) in enumerate(raw[policy].items())
        }
        for policy in policies
    }
    return {
        "rounds": config.e2_rounds,
        "seeds": len(seeds),
        "candidates": candidates,
        "context_buckets": buckets,
        "best_arm_by_bucket": best_arm,
        "throttling_simulated": False,
        "thompson": stats["thompson"],
        "rule": stats["rule"],
        "uniform": stats["uniform"],
        "seed_list": seeds,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# E3 — uncertainty sampling vs uniform random
# ═══════════════════════════════════════════════════════════════════════════════


def _e3_split(
    config: Config,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], dict[str, Any]]:
    """Date-based pool/test split: earliest 70% of days vs the rest."""
    archetypes = sorted(PROFILES)[: config.e3_archetypes]
    windows: list[dict[str, Any]] = []
    for index, archetype in enumerate(archetypes):
        windows.extend(_generate_windows(archetype, config.e3_days, 3000 + index))
    days = sorted({str(row["window_start_utc"])[:10] for row in windows})
    cut = days[int(len(days) * 0.7)]
    pool = [row for row in windows if str(row["window_start_utc"])[:10] < cut]
    test = [row for row in windows if str(row["window_start_utc"])[:10] >= cut]
    return (
        _feature_matrix(pool),
        _labels(pool),
        _feature_matrix(test),
        _labels(test),
        {"pool_days": [d for d in days if d < cut], "test_days": [d for d in days if d >= cut]},
    )


def _balanced_accuracy(model: Any, X_test: npt.NDArray[Any], y_test: npt.NDArray[Any]) -> float:
    from sklearn.metrics import balanced_accuracy_score

    probabilities = np.asarray(model.predict_proba(X_test), dtype=float)[:, 1]
    return float(balanced_accuracy_score(y_test, (probabilities >= 0.5).astype(int)))


def _e3_strategy_scores(
    strategy: str,
    model: Any,
    X_pool: npt.NDArray[Any],
    labelled: set[int],
    rng: np.random.Generator,
    batch: int,
) -> list[int]:
    """Indices to label next.

    ``uncertainty`` ranks the pool by the production criterion from
    ``services/feedback_sampling``: a window is worth labelling when
    ``decide_feedback_request`` says the model is uncertain about it, and
    within that band the closest to the 0.5 boundary comes first.
    ``information_gain_proxy`` is the aggregate report (request rate, lift)
    over a set of decisions, not a per-window score, so it cannot rank a pool.
    """
    remaining = np.asarray([index for index in range(len(X_pool)) if index not in labelled])
    if strategy == "random":
        return [int(index) for index in rng.choice(remaining, size=batch, replace=False)]

    probabilities = np.asarray(model.predict_proba(X_pool), dtype=float)[:, 1]
    flagged = [
        int(index)
        for index in remaining
        if decide_feedback_request(
            FeedbackSignals(model_probability=float(probabilities[index]))
        ).should_request
    ]
    ordered = sorted(
        flagged or [int(index) for index in remaining],
        key=lambda index: abs(float(probabilities[index]) - 0.5),
    )
    return ordered[:batch]


def _e3_seed_run(
    seed: int,
    config: Config,
    X_pool: npt.NDArray[Any],
    y_pool: npt.NDArray[Any],
    X_test: npt.NDArray[Any],
    y_test: npt.NDArray[Any],
    full_ba: float,
) -> dict[str, dict[str, Any]]:
    """Learning curve for both strategies under one seed."""
    names = list(V2_FEATURE_NAMES)
    out: dict[str, dict[str, Any]] = {}
    for strategy in ("uncertainty", "random"):
        rng = np.random.default_rng(seed)
        labelled: set[int] = set()
        # Ten random starting labels, both classes guaranteed.
        while len(labelled) < config.e3_start_labels:
            drawn = {int(i) for i in rng.choice(len(X_pool), config.e3_start_labels, replace=False)}
            if len({int(y_pool[i]) for i in drawn}) == 2:
                labelled = drawn

        curve: dict[int, float] = {}
        while True:
            indices = sorted(labelled)
            model = make_v2_classifier()
            model.fit(X_pool[indices], y_pool[indices], names)
            if len(labelled) in config.e3_eval_points:
                curve[len(labelled)] = _balanced_accuracy(model, X_test, y_test)
            if len(labelled) >= config.e3_max_labels:
                break
            batch = min(config.e3_batch, config.e3_max_labels - len(labelled))
            # The refit happens *after* the labels are chosen, so the batch is
            # picked from the model that exists now — no second fit needed.
            for index in _e3_strategy_scores(strategy, model, X_pool, labelled, rng, batch):
                labelled.add(index)

        target = 0.95 * full_ba
        reached = next(
            (size for size in sorted(curve) if curve[size] >= target),
            None,
        )
        out[strategy] = {
            "curve": {str(size): value for size, value in curve.items()},
            "labels_to_95pct_of_full": reached,
        }
    return out


def run_e3(config: Config) -> dict[str, Any]:
    """Test balanced accuracy at fixed label budgets, uncertainty vs random."""
    X_pool, y_pool, X_test, y_test, split = _e3_split(config)
    names = list(V2_FEATURE_NAMES)

    full_model = make_v2_classifier()
    full_model.fit(X_pool, y_pool, names)
    full_ba = _balanced_accuracy(full_model, X_test, y_test)

    seeds = [4000 + index for index in range(config.e3_seeds)]
    runs = [
        _e3_seed_run(seed, config, X_pool, y_pool, X_test, y_test, full_ba)
        for seed in seeds
    ]

    points = {str(point): {} for point in config.e3_eval_points}
    for point in config.e3_eval_points:
        for strategy in ("uncertainty", "random"):
            values = [
                run[strategy]["curve"][str(point)]
                for run in runs
                if str(point) in run[strategy]["curve"]
            ]
            points[str(point)][strategy] = _summary_stats(values, seed=9)

    reached = {
        strategy: [run[strategy]["labels_to_95pct_of_full"] for run in runs]
        for strategy in ("uncertainty", "random")
    }
    reached_summary = {
        strategy: {
            "reached_count": float(sum(1 for value in values if value is not None)),
            "mean_labels": (
                statistics.fmean([value for value in values if value is not None])
                if any(value is not None for value in values)
                else None
            ),
            "censored_at": float(config.e3_max_labels),
        }
        for strategy, values in reached.items()
    }

    return {
        "users": config.e3_archetypes,
        "pool_windows": int(len(y_pool)),
        "test_windows": int(len(y_test)),
        "split": split,
        "model": "mindflow.train.v2.make_v2_classifier (EnsembleClassifier, calibration=None)",
        "full_pool_balanced_accuracy": full_ba,
        "target_balanced_accuracy": 0.95 * full_ba,
        "seeds": len(seeds),
        "start_labels": config.e3_start_labels,
        "batch": config.e3_batch,
        "max_labels": config.e3_max_labels,
        "eval_points": list(config.e3_eval_points),
        "curves": points,
        "labels_to_95pct_of_full": reached_summary,
        "seed_list": seeds,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# E4 — local explanation fidelity
# ═══════════════════════════════════════════════════════════════════════════════


def run_e4(config: Config) -> dict[str, Any]:
    """Additivity error and deletion fidelity of ``LocalExplainer``."""
    archetypes = sorted(PROFILES)[: config.e4_archetypes]
    windows: list[dict[str, Any]] = []
    for index, archetype in enumerate(archetypes):
        windows.extend(_generate_windows(archetype, config.e4_train_days + 2, 5000 + index))
    days = sorted({str(row["window_start_utc"])[:10] for row in windows})
    split = days[config.e4_train_days]
    train = [row for row in windows if str(row["window_start_utc"])[:10] < split]
    test = [row for row in windows if str(row["window_start_utc"])[:10] >= split]

    names = list(V2_FEATURE_NAMES)
    X_train = _feature_matrix(train)
    y_train = _labels(train)
    X_test = _feature_matrix(test)

    rng = np.random.default_rng(6000)
    picks = rng.choice(len(X_test), size=min(config.e4_windows, len(X_test)), replace=False)
    X_eval = X_test[picks]

    model = make_v2_classifier()
    model.fit(X_train, y_train, names)
    # The reference value: the training-set column mean, i.e. the centre of
    # the background distribution TreeSHAP's expected value is taken over.
    # It is passed as ``baseline_values`` too, so the reported baseline and the
    # deletion target are the same number.
    reference = {
        name: float(value)
        for name, value in zip(names, X_train.mean(axis=0), strict=True)
    }
    baseline_prediction = np.asarray(model.predict_proba(X_eval), dtype=float)[:, 1]

    explainer = LocalExplainer(model)
    original_cap = local_explanation.MAX_CONTRIBUTIONS
    # Additivity is only meaningful over *all* features: the production cap of
    # 8 would compare a partial sum against the full output.
    local_explanation.MAX_CONTRIBUTIONS = len(names)
    try:
        explanations = []
        for index in range(len(X_eval)):
            row = X_eval[index : index + 1]
            explanations.append(
                explainer.explain(
                    row,
                    names,
                    focus_probas=baseline_prediction[index : index + 1],
                    baseline_values=reference,
                )
            )
    finally:
        local_explanation.MAX_CONTRIBUTIONS = original_cap

    methods = Counter(explanation.method for explanation in explanations)
    additivity = [
        abs(
            (explanation.base_value or 0.0)
            + sum(item.contribution for item in explanation.contributions)
            - float(baseline_prediction[index])
        )
        for index, explanation in enumerate(explanations)
        if explanation.method != "unavailable"
    ]
    output_spaces = Counter(explanation.output_space for explanation in explanations)

    reference_vector = X_train.mean(axis=0)

    def _deletion_delta(feature_indices: list[list[int]]) -> float:
        """Mean |Δ prediction| when each window's listed features are replaced.

        All replaced copies are predicted in one batch: the ensemble's
        ``predict_proba`` costs ~5 ms regardless of batch size, so one call per
        row would dominate the runtime for no extra information.
        """
        replaced = np.repeat(X_eval[:, None, :], 3, axis=1)
        for index, chosen in enumerate(feature_indices):
            for slot, feature_index in enumerate(chosen[:3]):
                replaced[index, slot, feature_index] = reference_vector[feature_index]
        predictions = np.asarray(
            model.predict_proba(replaced.reshape(-1, len(names))), dtype=float
        )[:, 1].reshape(len(X_eval), 3)
        deltas = np.abs(predictions - baseline_prediction[:, None])
        return float(deltas.mean()) if len(X_eval) else math.nan

    top3 = [
        [names.index(item.feature) for item in explanation.contributions[:3]]
        for explanation in explanations
    ]
    top_delta = _deletion_delta(top3)

    random_rng = np.random.default_rng(7000)
    random_deltas = [
        _deletion_delta(
            [
                [int(value) for value in random_rng.choice(len(names), 3, replace=False)]
                for _ in range(len(X_eval))
            ]
        )
        for _ in range(config.e4_random_draws)
    ]
    random_delta = float(statistics.fmean(random_deltas))

    return {
        "windows": len(X_eval),
        "train_windows": len(train),
        "model": "mindflow.train.v2.make_v2_classifier (EnsembleClassifier, calibration=None)",
        "methods": dict(methods),
        "output_spaces": dict(output_spaces),
        "reference": "training-set feature mean (TreeSHAP background centre)",
        "max_contribution_cap": len(names),
        "additivity_max_abs_error": float(max(additivity)) if additivity else None,
        "additivity_mean_abs_error": (
            float(statistics.fmean(additivity)) if additivity else None
        ),
        "deletion_top3_mean_abs_delta": top_delta,
        "deletion_random_mean_abs_delta": random_delta,
        "deletion_ratio_top_over_random": (
            top_delta / random_delta if random_delta > 0 else None
        ),
        "random_draws": config.e4_random_draws,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# E5 — evidence citation validity (mock replay) + recorded live smoke
# ═══════════════════════════════════════════════════════════════════════════════

#: Fallback numbers for the 2026-09-25 live smoke when its artifact is absent.
LIVE_SMOKE_FALLBACK = {
    "panel": {
        "requests": 9,
        "schema_attempts": 9,
        "schema_passes": 9,
        "citations_total": 7,
        "citations_valid": 7,
        "failure_rate": 0.0,
        "p95_latency_s": 26.7,
    },
    "attribution": {"requests": 1, "schema_attempts": 1, "schema_passes": 1},
}


def _clean_env() -> dict[str, str]:
    """Environment for the child harness: no credentials, UTF-8 stdout."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(marker in key.upper() for marker in _SECRET_ENV_MARKERS)
    }
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _latest_artifact(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def _harness_payload(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    scenarios = report.get("scenarios") or []
    return {
        "analyzer": report.get("analyzer_name"),
        "total": report.get("total"),
        "top1_accuracy": report.get("top1_accuracy"),
        "mean_jaccard": report.get("mean_jaccard"),
        "technique_accuracy": report.get("technique_accuracy"),
        "top1_hits": sum(1 for row in scenarios if row.get("top1_hit")),
        "predicted_source_counts": dict(
            Counter(str(row.get("predicted_source")) for row in scenarios)
        ),
    }


def run_e5(config: Config, out_dir: Path) -> dict[str, Any] | None:
    """Run the offline evaluation harness and record what it actually reports.

    Returns ``None`` in ``--quick`` mode: the harness is a full subprocess whose
    import cost alone dominates a quick run, and the smoke test only needs the
    shape of the summary, not a second set of accuracy numbers.  ``--quick`` is
    a plumbing check, so the real harness stays a full-mode-only concern.
    """
    if config.quick:
        return None
    harness_dir = out_dir / "eval_harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "mindflow.eval",
        "--mode",
        "both",
        "--output-dir",
        str(harness_dir),
    ]
    if config.e5_scenario_ids:
        command.extend(["--scenario-ids", ",".join(config.e5_scenario_ids)])

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=_clean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (harness_dir / "harness.log").write_text(
        (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
    )

    rule_artifact = _latest_artifact(harness_dir, "report_rule_engine_*.json")
    panel_artifact = _latest_artifact(harness_dir, "report_panel_mock_*.json")
    comparison_artifact = _latest_artifact(harness_dir, "comparison_*.json")

    def _load(path: Path | None) -> dict[str, Any] | None:
        if path is None or not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    comparison = _load(comparison_artifact)
    return {
        "command": " ".join(command[1:]),
        "returncode": completed.returncode,
        "artifact_dir": str(harness_dir),
        "artifacts": [
            str(path) for path in (rule_artifact, panel_artifact, comparison_artifact)
            if path is not None
        ],
        # The harness emits accuracy/Jaccard/technique metrics only. It has no
        # citation-validity, schema-validity or degradation counter, so those
        # stay null instead of being derived from something that is not them.
        "citation_validity": None,
        "schema_validity": None,
        "degradation_counts": None,
        "not_reported": [
            "citation_validity",
            "schema_validity",
            "degradation_counts",
        ],
        "note": (
            "离线 harness 只输出 Top-1 命中率、平均 Jaccard 与 Technique 匹配率，"
            "不输出引用合法率、结构化通过率或降级计数；此处如实记为 null，不做推算。"
        ),
        "rule_engine": _harness_payload(_load(rule_artifact)),
        "panel_mock": _harness_payload(_load(panel_artifact)),
        "comparison": (
            {
                "top1_delta": comparison.get("top1_delta"),
                "jaccard_delta": comparison.get("jaccard_delta"),
                "technique_delta": comparison.get("technique_delta"),
                "baseline_wins": comparison.get("baseline_wins"),
                "panel_wins": comparison.get("panel_wins"),
                "ties": comparison.get("ties"),
            }
            if comparison is not None
            else None
        ),
    }


def live_smoke_record() -> dict[str, Any]:
    """The 2026-09-25 live DeepSeek smoke numbers, read from its artifact."""
    if LIVE_SMOKE_SOURCE.is_file():
        payload = json.loads(LIVE_SMOKE_SOURCE.read_text(encoding="utf-8"))
        targets = payload.get("targets", {})
        panel = targets.get("panel", {})
        attribution = targets.get("attribution", {})
        citations = panel.get("citations", {})
        return {
            "source": str(LIVE_SMOKE_SOURCE),
            "run_id": payload.get("run_id"),
            "created_at": payload.get("created_at"),
            "mode": payload.get("mode"),
            "panel": {
                "requests": panel.get("requests"),
                "schema_attempts": panel.get("schema_attempts"),
                "schema_passes": panel.get("schema_passes"),
                "citations_total": citations.get("total"),
                "citations_valid": citations.get("valid"),
                "failure_rate": panel.get("failure_rate"),
                "p95_latency_s": (
                    round(float(panel["p95_latency_ms"]) / 1000.0, 1)
                    if panel.get("p95_latency_ms") is not None
                    else None
                ),
                "degraded": panel.get("degraded"),
                "degradation_path": panel.get("degradation_path"),
            },
            "attribution": {
                "requests": attribution.get("requests"),
                "schema_attempts": attribution.get("schema_attempts"),
                "schema_passes": attribution.get("schema_passes"),
            },
        }
    return {
        "source": "CLAUDE.md record",
        "run_id": "20260925_051755",
        "created_at": None,
        "mode": "live",
        **LIVE_SMOKE_FALLBACK,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def _verdict(positive: bool, yes: str, no: str) -> str:
    return yes if positive else no


def build_report(summary: dict[str, Any]) -> str:
    e1 = summary["e1"]
    e2 = summary["e2"]
    e3 = summary["e3"]
    e4 = summary["e4"]
    e5 = summary["e5"]
    meta = summary["meta"]

    lines: list[str] = [
        "# MindFlow 中期算法实验（E1–E5）",
        "",
        f"- 运行: `{meta['run_id']}`" + ("（--quick 小参数）" if meta["quick"] else ""),
        f"- 生成时间: {meta['created_at']}",
        f"- 后端根目录: `{meta['project_root']}`",
        "- 全部随机性均以固定种子驱动，种子列表见 `summary.json` 的 `meta.seeds`。",
        "",
        "## 总览",
        "",
        "| 实验 | 指标 | 新算法 | 对照 | 95% CI | 结论 |",
        "|---|---|---|---|---|---|",
    ]

    # Only the confirmation seed set is cited; the development set is shown in
    # its own section because it is the one that exposed the flaw.
    confirm = e1["confirm"]
    confirm_arms = confirm["arms"]
    raw_mse = confirm_arms["raw"]["mean_mse"]
    for arm in ("eb_fixed_k10", "eb_data_k"):
        stats = confirm_arms[arm]["relative_mse_reduction"]
        label = "旧实现 k=10" if arm == "eb_fixed_k10" else "新默认（k̂）"
        lines.append(
            f"| E1 稳健基线 | {label}：相对 MSE 下降（全体桶） | {_fmt(stats['mean'], 4)} | "
            f"0（原始桶均值 {_fmt(raw_mse, 4)}） | {_ci_text(stats, 4)} | "
            f"{_verdict(stats['mean'] > 0.0, '优于原始均值', '未优于原始均值')} |"
        )
        sparse_stats = confirm_arms[arm]["sparse_relative_mse_reduction"]
        sparse_verdict = (
            "优于原始均值"
            if sparse_stats["mean"] > 0.0
            else (
                "无稀疏桶"
                if confirm_arms[arm]["users_with_sparse_buckets"] == 0
                else "未优于原始均值"
            )
        )
        lines.append(
            f"| E1 稳健基线 | {label}：相对 MSE 下降（稀疏桶） | {_fmt(sparse_stats['mean'], 4)} | "
            f"0 | {_ci_text(sparse_stats, 4)} | {sparse_verdict} |"
        )

    thompson = e2["thompson"]
    rule = e2["rule"]
    uniform = e2["uniform"]
    lines.append(
        f"| E2 汤普森 | T={e2['rounds']} 累积遗憾 | {_fmt(thompson['regret']['mean'], 2)} | "
        f"规则 {_fmt(rule['regret']['mean'], 2)} / 随机 {_fmt(uniform['regret']['mean'], 2)} | "
        f"{_ci_text(thompson['regret'], 2)} | "
        f"{_verdict(thompson['regret']['mean'] < min(rule['regret']['mean'], uniform['regret']['mean']), '汤普森最低', '汤普森非最低')} |"
    )
    lines.append(
        f"| E2 汤普森 | 末 100 轮最优臂选择率 | {_fmt(thompson['optimal_rate']['mean'])} | "
        f"规则 {_fmt(rule['optimal_rate']['mean'])} / 随机 {_fmt(uniform['optimal_rate']['mean'])} | "
        f"{_ci_text(thompson['optimal_rate'])} | — |"
    )
    lines.append(
        f"| E2 汤普森 | 总体接受率 | {_fmt(thompson['acceptance']['mean'])} | "
        f"规则 {_fmt(rule['acceptance']['mean'])} / 随机 {_fmt(uniform['acceptance']['mean'])} | "
        f"{_ci_text(thompson['acceptance'])} | — |"
    )

    for point in e3["eval_points"]:
        key = str(point)
        uncertainty = e3["curves"][key]["uncertainty"]
        rand = e3["curves"][key]["random"]
        better = uncertainty["mean"] > rand["mean"]
        if e3["seeds"] < 2:
            verdict = "单种子，不可判定"
        else:
            verdict = _verdict(better, "不确定性采样更优", "未优于随机")
        lines.append(
            f"| E3 主动学习 | {point} 标签时测试 BA | {_fmt(uncertainty['mean'])} | "
            f"{_fmt(rand['mean'])} | {_ci_text(uncertainty)} | {verdict} |"
        )
    reached_u = e3["labels_to_95pct_of_full"]["uncertainty"]
    reached_r = e3["labels_to_95pct_of_full"]["random"]
    lines.append(
        f"| E3 主动学习 | 达到 0.95×全池 BA 所需标签 | "
        f"{_fmt(reached_u['mean_labels'], 1)}（{int(reached_u['reached_count'])}/{e3['seeds']} 个种子达到） | "
        f"{_fmt(reached_r['mean_labels'], 1)}（{int(reached_r['reached_count'])}/{e3['seeds']}） | — | — |"
    )

    lines.append(
        f"| E4 局部解释 | 可加性误差 max / mean | "
        f"{_fmt(e4['additivity_max_abs_error'], 6)} / {_fmt(e4['additivity_mean_abs_error'], 6)} | 0 | — | "
        f"{_verdict((e4['additivity_max_abs_error'] or 1.0) < 1e-3, '可加性成立', '可加性不成立')} |"
    )
    ratio = e4["deletion_ratio_top_over_random"]
    lines.append(
        f"| E4 局部解释 | 删除 3 特征平均 |Δp| | {_fmt(e4['deletion_top3_mean_abs_delta'], 4)} | "
        f"随机 3 特征 {_fmt(e4['deletion_random_mean_abs_delta'], 4)} | — | "
        f"top/random = {_fmt(ratio, 2)}，"
        f"{_verdict(bool(ratio and ratio > 1.0), '贡献排序有效', '贡献排序无效')} |"
    )

    harness = e5["harness"]
    live = e5["live_smoke_20260925"]
    rule_payload = (harness or {}).get("rule_engine") or {}
    panel_payload = (harness or {}).get("panel_mock") or {}
    if harness is None:
        lines.append(
            "| E5 引用合法率 | 离线 harness Top-1（规则 / mock 面板） | "
            f"未运行（{e5['harness_skipped']}） | — | — | — |"
        )
    else:
        lines.append(
            f"| E5 引用合法率 | 离线 harness Top-1（规则 / mock 面板） | "
            f"{_fmt(rule_payload.get('top1_accuracy'))} / {_fmt(panel_payload.get('top1_accuracy'))} | — | — | "
            "harness 不报告引用合法率 |"
        )
    lines.append(
        f"| E5 引用合法率 | 线上冒烟（2026-09-25）引用合法 | "
        f"{_fmt(live['panel'].get('citations_valid'))}/{_fmt(live['panel'].get('citations_total'))} | — | — | "
        f"schema {_fmt(live['panel'].get('schema_passes'))}/{_fmt(live['panel'].get('schema_attempts'))} |"
    )

    lines.extend(["", "## E1 稳健个人基线（三组对照）", ""])
    lines.append(
        "**汇报材料只引用 `confirm`（确认种子）的结果；`dev`（开发种子）是暴露 §8.25 缺陷的那一组，"
        "仅作对照，不作为结论依据。**"
    )
    lines.append(
        f"- 每个种子集 {e1['confirm']['users']} 个用户"
        f"（{len(e1['archetypes'])} 个学生原型 × {meta['config']['e1_seeds_per_archetype']} 个种子），"
        "每人 14 天，前 7 天训练、后 7 天留出。"
    )
    lines.append(
        f"- 两组种子完全不相交: {'是' if e1['seed_sets_disjoint'] else '否'}；"
        f"dev = {e1['dev']['seeds']}，confirm = {e1['confirm']['seeds']}。"
    )
    lines.append(f"- 特征: {', '.join(e1['features'])}（`BaselineModel.FEATURE_COLS` 的真实 V2 列名）。")
    lines.append(
        "- 三组臂: `raw` = 原始桶均值（空桶退回全局训练均值）；"
        "`eb_fixed_k10` = 显式 k=10 + 全局稳健先验（§8.25 之前的实现）；"
        "`eb_data_k` = 不传 k（新默认，矩估计 k̂ + 小时合并先验）。"
    )
    lines.append(
        f"- 稀疏桶占比: {e1['confirm']['sparse_bucket_share']:.1%}"
        f"（训练样本数 < {e1['sparse_threshold']} 的留出桶）。"
    )

    for set_name in ("confirm", "dev"):
        block = e1[set_name]
        title = "确认种子（材料引用这一组）" if set_name == "confirm" else "开发种子（暴露缺陷的那一组）"
        lines.extend([
            "",
            f"### {set_name} — {title}",
            "",
            "| 臂 | 平均 MSE | 相对 MSE 下降 | 95% CI | 稀疏桶相对下降 | 95% CI |",
            "|---|---|---|---|---|---|",
        ])
        for arm in e1["arms"]:
            arm_stats = block["arms"][arm]
            if arm == "raw":
                lines.append(
                    f"| raw | {_fmt(arm_stats['mean_mse'], 4)} | 0（基准） | — | 0（基准） | — |"
                )
                continue
            reduction = arm_stats["relative_mse_reduction"]
            sparse = arm_stats["sparse_relative_mse_reduction"]
            lines.append(
                f"| {arm} | {_fmt(arm_stats['mean_mse'], 4)} | {_fmt(reduction['mean'], 4)} | "
                f"{_ci_text(reduction, 4)} | {_fmt(sparse['mean'], 4)} | {_ci_text(sparse, 4)} |"
            )
        lines.extend([
            "",
            "| 特征 | 平均 k̂ | eb_fixed_k10 下降 | 95% CI | eb_data_k 下降 | 95% CI |",
            "|---|---|---|---|---|---|",
        ])
        for feature in e1["features"]:
            detail = block["per_feature"][feature]
            fixed = detail["reduction"]["eb_fixed_k10"]
            data_k = detail["reduction"]["eb_data_k"]
            lines.append(
                f"| {feature} | {_fmt(detail['mean_k'], 2)} | {_fmt(fixed['mean'], 4)} | "
                f"{_ci_text(fixed, 4)} | {_fmt(data_k['mean'], 4)} | {_ci_text(data_k, 4)} |"
            )

    confirm_data = e1["confirm"]["arms"]["eb_data_k"]["relative_mse_reduction"]
    if confirm_data["mean"] <= 0.0:
        lines.extend([
            "",
            "**结论（如实报告）**：在确认种子上，数据驱动 k̂ 的默认路径仍未优于原始桶均值。"
            "按 §8.25 第 6 条，结果不利时如实报告，不再修改算法。",
        ])
    else:
        lines.extend([
            "",
            "**结论**：在确认种子上，数据驱动 k̂ 的默认路径优于原始桶均值，"
            "且优于固定 k=10 的旧实现。",
        ])

    lines.extend(["", "## E2 Thompson 采样干预选择（模拟用户）", ""])
    lines.append(
        f"- {e2['seeds']} 个种子 × {e2['rounds']} 轮；候选干预类型 {', '.join(e2['candidates'])}"
        f"（`candidate_types_for(task_aversion)` 的真实返回值）。"
    )
    lines.append(
        f"- 上下文桶: {', '.join(e2['context_buckets'])}；最优/次优臂的真实接受概率差 ∈ [0.10, 0.25]，"
        "最优臂在每个桶内随机，规则策略无法靠构造取胜。"
    )
    lines.append("- 汤普森策略走生产路径：`posteriors_from_logs`（含 14 天半衰期衰减与部分池化）+ `thompson_select`。")
    lines.append("- 未模拟限流（throttling）；奖励为按 `reward_of` 约定编码的伯努利结果。")
    lines.extend([
        "",
        "| 策略 | 累积遗憾 | 95% CI | 末 100 轮最优臂率 | 总体接受率 |",
        "|---|---|---|---|---|",
    ])
    for label, stats in (("Thompson", thompson), ("规则映射", rule), ("均匀随机", uniform)):
        lines.append(
            f"| {label} | {_fmt(stats['regret']['mean'], 2)} | {_ci_text(stats['regret'], 2)} | "
            f"{_fmt(stats['optimal_rate']['mean'])} | {_fmt(stats['acceptance']['mean'])} |"
        )

    lines.extend(["", "## E3 主动学习（不确定性采样 vs 随机）", ""])
    lines.append(
        f"- {e3['users']} 个模拟用户，池 {e3['pool_windows']} 个窗口，测试 {e3['test_windows']} 个窗口；"
        f"按日期切分（池 = 最早的 70% 日期）。"
    )
    lines.append(f"- 模型: `{e3['model']}`。")
    lines.append(
        f"- 全池模型测试 BA = {_fmt(e3['full_pool_balanced_accuracy'], 4)}，"
        f"达标线 = 0.95 × 全池 = {_fmt(e3['target_balanced_accuracy'], 4)}。"
    )
    lines.append(
        f"- {e3['seeds']} 个种子；初始 {e3['start_labels']} 个随机标签（保证两类都有），"
        f"每批 {e3['batch']} 个，最多 {e3['max_labels']} 个。"
    )
    grid = [e3["start_labels"] + e3["batch"] * step for step in range(
        (e3["max_labels"] - e3["start_labels"]) // e3["batch"] + 1
    )]
    lines.append(
        f"- 评测点取在标签网格 {grid} 上；不在网格上的点不会出现，"
        "因为模型只在每批标注后重新训练。"
    )
    if e3["seeds"] < 2:
        lines.append(
            "- **仅 1 个种子（--quick），置信区间退化为一个点，不能作为结论依据。**"
        )
    if e3["seeds"] < 2:
        lines.append(
            "- **仅 1 个种子（--quick），置信区间退化为一个点，不能作为结论依据。**"
        )
    lines.append(
        "- 不确定性策略使用 `services/feedback_sampling.decide_feedback_request` 判定"
        "「模型不确定」，再按 |p−0.5| 升序取样；`information_gain_proxy` 是对一组决策的聚合报告"
        "（请求率/提升倍数），无法逐窗口排序，因此未用它作为选择器。"
    )
    lines.extend([
        "",
        "| 标签数 | 不确定性采样 BA | 95% CI | 随机 BA | 95% CI |",
        "|---|---|---|---|---|",
    ])
    for point in e3["eval_points"]:
        key = str(point)
        uncertainty = e3["curves"][key]["uncertainty"]
        rand = e3["curves"][key]["random"]
        lines.append(
            f"| {point} | {_fmt(uncertainty['mean'])} | {_ci_text(uncertainty)} | "
            f"{_fmt(rand['mean'])} | {_ci_text(rand)} |"
        )

    lines.extend(["", "## E4 局部解释忠实度", ""])
    lines.append(f"- 测试窗口: {e4['windows']}（训练 {e4['train_windows']} 个窗口）。")
    lines.append(f"- 模型: `{e4['model']}`；解释方法: {json.dumps(e4['methods'], ensure_ascii=False)}。")
    lines.append(
        f"- 可加性在 `{e4['output_spaces']}` 空间度量，且把 `MAX_CONTRIBUTIONS` 上限临时提升到 "
        f"{e4['max_contribution_cap']}（生产上限 8 只覆盖部分特征，无法与完整输出比较）。"
    )
    lines.append(f"- 删除实验的参考值: {e4['reference']}。")
    lines.extend([
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| 可加性误差 max | {_fmt(e4['additivity_max_abs_error'], 8)} |",
        f"| 可加性误差 mean | {_fmt(e4['additivity_mean_abs_error'], 8)} |",
        f"| 删除 top-3 特征平均 |Δp| | {_fmt(e4['deletion_top3_mean_abs_delta'], 4)} |",
        f"| 删除随机 3 特征平均 |Δp| | {_fmt(e4['deletion_random_mean_abs_delta'], 4)} |",
        f"| 比值（top / random） | {_fmt(e4['deletion_ratio_top_over_random'], 2)} |",
    ])

    lines.extend(["", "## E5 证据引用合法率", ""])
    if harness is None:
        lines.append(
            f"- 离线 harness: 本次未运行（{e5['harness_skipped']}）；"
            "全量运行会执行 `python -m mindflow.eval --mode both`（无 `--live`）。"
        )
    else:
        lines.append(f"- 离线 harness: `python {harness['command']}`（无 `--live`，LLM 环境变量已清除）。")
        lines.append(f"- 返回码 {harness['returncode']}；产物目录 `{harness['artifact_dir']}`。")
        lines.append(f"- {harness['note']}")
    if rule_payload:
        lines.append(
            f"- 规则引擎: Top-1 {_fmt(rule_payload.get('top1_accuracy'))}，"
            f"Jaccard {_fmt(rule_payload.get('mean_jaccard'))}，"
            f"Technique {_fmt(rule_payload.get('technique_accuracy'))}（{rule_payload.get('total')} 个场景）。"
        )
    if panel_payload:
        lines.append(
            f"- mock 面板: Top-1 {_fmt(panel_payload.get('top1_accuracy'))}，"
            f"Jaccard {_fmt(panel_payload.get('mean_jaccard'))}，"
            f"Technique {_fmt(panel_payload.get('technique_accuracy'))}（{panel_payload.get('total')} 个场景）。"
        )
    lines.append(
        f"- 线上冒烟记录（{live.get('source')}）: 面板 {live['panel'].get('requests')} 次请求，"
        f"schema {live['panel'].get('schema_passes')}/{live['panel'].get('schema_attempts')}，"
        f"引用 {live['panel'].get('citations_valid')}/{live['panel'].get('citations_total')} 合法，"
        f"失败率 {_fmt(live['panel'].get('failure_rate'))}，p95 {_fmt(live['panel'].get('p95_latency_s'), 1)} s；"
        f"attribution schema {live['attribution'].get('schema_passes')}/{live['attribution'].get('schema_attempts')}。"
    )

    lines.extend([
        "",
        "## 数据来源与局限",
        "",
        "1. **E1–E4 使用的是合成数据或模拟用户，不是真实用户效果。** "
        "E1/E3/E4 的窗口来自 `train/synthetic_v2.py` 的学生原型生成器，"
        "E2 的接受概率由本脚本设定。因此这里的数字衡量的是「算法在受控数据上的行为」，"
        "不能外推为真实用户的专注提升、真实用户的点击率或真实标注成本。",
        "2. **E1 的结果不支持「EB 收缩更好」，而这一结果由生成器的构造决定。** "
        "生成器把同一 (小时, 星期) 桶内的 `idle_ratio` / `active_seconds_ratio` 限制在窄带内"
        "（睡眠时段 idle 0.85–1.0、活跃时段 idle 0–0.08），桶内方差极小、桶间差异极大；"
        "原始桶均值几乎没有误差，而 EB 收缩把桶估计拉向全局中位数，"
        "于是在这两个比例特征上 MSE 差了一个量级以上。"
        "`app_switch_count` 是离散计数、桶内方差大，EB 才有小幅优势。"
        "真实数据里同一时段的作息波动远大于合成数据，这正是 EB 收缩要处理的情形，"
        "但本实验**无法**证明这一点——它只说明合成生成器不能代表真实波动，"
        "因此不能据此宣称稳健基线优于原始均值，也不能据此否定它。",
        "3. **E2 只模拟了选择策略本身。** 没有模拟限流、推送疲劳、用户流失，"
        "也没有模拟「干预无效」之外的负反馈；奖励是即时伯努利接受，而非长期行为改变。"
        "接受概率在每个 (干预类型, 上下文桶) 上固定，不随时间漂移，"
        "因此衰减机制的价值在本实验中体现不出来。",
        "4. **E3 的标签是合成真值，不是人工标注成本。** 主动学习省下的标签数"
        "（若体现为负值）不代表真实标注预算；合成数据的类别边界清晰，"
        "10 个随机标签即可达到较高 BA，学习曲线容易饱和，这对两种策略都是如此。",
        "5. **E4 的参考值是训练集均值，不是用户的个人基线。** 生产路径传入的是 "
        "`BaselineModel.shrunk_expected` 的个人期望；本实验改用训练集列均值"
        "（即 TreeSHAP 期望值的背景中心），使「报告的 baseline_value」与「删除时替换成的值」"
        "是同一个数。删除实验只替换 3 个特征，特征间存在相关性时删除操作本身会偏离数据流形，"
        "这是删除类忠实度指标的已知局限。",
        "6. **E5 的离线 harness 不产出引用合法率。** 它只报告 Top-1 命中率、Jaccard 与 "
        "Technique 匹配率；引用合法率/schema 通过率/降级计数只在线上冒烟脚本中统计，"
        "因此本节把离线部分记为 null，并单独引用 2026-09-25 的线上冒烟数字（9/9 schema、7/7 引用合法）。"
        "线上数字是单次运行的样本，不是统计意义上的合法率。",
        "7. **复现性说明。** `generate_v2_synthetic_data` 用 `hash(profile_id)` 播种，"
        "而 Python 的字符串哈希每个进程不同，同一命令两次运行会得到不同窗口。"
        "本脚本改为用显式 `random.Random(seed)` 调用同一模块的日模拟器，"
        "算法与参数完全一致，只是让结果可复现。",
        "",
    ])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def run(config: Config, out_dir: Path) -> dict[str, Any]:
    """Run all five experiments and write ``summary.json`` and ``report.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_midterm")

    e1 = run_e1(config)
    e2 = run_e2(config)
    e3 = run_e3(config)
    e4 = run_e4(config)
    harness = run_e5(config, out_dir)
    e5 = {
        "harness": harness,
        "harness_skipped": "quick mode" if harness is None else None,
        "live_smoke_20260925": live_smoke_record(),
    }

    summary: dict[str, Any] = {
        "meta": {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "quick": config.quick,
            "project_root": str(PROJECT_ROOT),
            "timezone": TIMEZONE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "ci_level": CI_LEVEL,
            "config": {
                "e1_archetypes": config.e1_archetypes,
                "e1_seeds_per_archetype": config.e1_seeds_per_archetype,
                "e2_rounds": config.e2_rounds,
                "e2_seeds": config.e2_seeds,
                "e3_archetypes": config.e3_archetypes,
                "e3_seeds": config.e3_seeds,
                "e3_eval_points": list(config.e3_eval_points),
                "e4_windows": config.e4_windows,
            },
            "seeds": {
                "e1_dev": e1["dev"]["seeds"],
                "e1_confirm": e1["confirm"]["seeds"],
                "e2": e2["seed_list"],
                "e3": e3["seed_list"],
                "e4": [6000, 7000],
                "e5": None,
            },
        },
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "e5": e5,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(build_report(summary), encoding="utf-8")
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, ProcrastinationType):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Tiny parameters (under 20 s) for smoke runs"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT),
        dest="out",
        help=f"Artifact directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)

    config = Config.quick_mode() if args.quick else Config.full()
    out_dir = Path(args.out)
    summary = run(config, out_dir)

    print(build_report(summary))
    print(f"Artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
