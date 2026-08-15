from datetime import UTC, datetime, timedelta

from mindflow.train.v2 import (
    V2_FEATURE_NAMES,
    evaluate_v2_candidates,
    evaluate_v2_quality_gate,
    prepare_v2_training_data,
)


def _feature_window(start: datetime, **overrides: float) -> dict[str, object]:
    features = {name: 0.0 for name in V2_FEATURE_NAMES}
    features.update({
        "longest_segment_ratio": 0.8,
        "input_active_ratio": 0.5,
        "top_app_ratio": 0.8,
    })
    features.update(overrides)
    return {
        "window_start_utc": start.isoformat(),
        "window_end_utc": (start + timedelta(minutes=5)).isoformat(),
        "feature_schema_version": 3,
        "features": features,
    }


def _feedback(
    session_id: str,
    start: datetime,
    score: int,
    label: str,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(minutes=30)).isoformat(),
        "score": score,
        "label": label,
        "task_type": "coding",
    }


def test_prepare_v2_training_data_prioritizes_explicit_feedback() -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = [
        _feature_window(start, idle_ratio=0.95),
        _feature_window(start + timedelta(hours=1), idle_ratio=0.05),
        _feature_window(start + timedelta(hours=2), idle_ratio=0.05),
    ]
    feedback = [
        _feedback("focus", start, 5, "focus"),
        _feedback("mixed", start + timedelta(hours=1), 3, "mixed"),
    ]

    data = prepare_v2_training_data(windows, feedback)

    assert data.labels.tolist() == [1, 1, 1]
    assert data.sample_weights.tolist() == [1.0, 0.3, 0.3]
    assert data.label_sources == ["explicit", "weak", "weak"]
    assert data.explicit_feedback_count == 1
    assert data.explicit_focus_count == 1
    assert data.explicit_distract_count == 0
    assert data.mixed_window_count == 0


def test_evaluate_v2_candidates_keeps_dates_out_of_training_folds() -> None:
    windows: list[dict[str, object]] = []
    feedback: list[dict[str, object]] = []
    base = datetime(2026, 7, 1, 9, tzinfo=UTC)
    for day_index in range(8):
        day_start = base + timedelta(days=day_index)
        focus_start = day_start
        distract_start = day_start + timedelta(hours=2)
        feedback.extend([
            _feedback(f"focus-{day_index}", focus_start, 5, "focus"),
            _feedback(f"distract-{day_index}", distract_start, 1, "distracted"),
        ])
        for offset in range(2):
            windows.append(
                _feature_window(
                    focus_start + timedelta(minutes=5 * offset),
                    idle_ratio=0.02,
                    longest_segment_ratio=0.95,
                    top_app_ratio=0.95,
                    interaction_bursts_per_min=1.2,
                )
            )
            windows.append(
                _feature_window(
                    distract_start + timedelta(minutes=5 * offset),
                    idle_ratio=0.55,
                    longest_segment_ratio=0.1,
                    top_app_ratio=0.25,
                    app_switch_count=8,
                    domain_switch_count=5,
                )
            )

    data = prepare_v2_training_data(windows, feedback)
    evaluation = evaluate_v2_candidates(data, random_state=7)

    assert evaluation["status"] == "evaluated"
    assert evaluation["candidate"]["balanced_accuracy"] >= 0.9
    assert evaluation["candidate"]["minority_f1"] >= 0.9
    assert evaluation["candidate"]["brier_score"] < evaluation["rule_baseline"]["brier_score"]
    assert len(evaluation["folds"]) >= 3
    for fold in evaluation["folds"]:
        assert set(fold["train_dates"]).isdisjoint(fold["test_dates"])


def test_v2_quality_gate_requires_explicit_feedback_and_stable_metrics() -> None:
    evaluation = {
        "status": "evaluated",
        "candidate": {
            "balanced_accuracy": 0.72,
            "minority_f1": 0.66,
            "brier_score": 0.14,
            "fold_balanced_accuracy_range": 0.08,
        },
            "rule_baseline": {"brier_score": 0.22},
            "folds": [{}, {}, {}],
            "fold_stability": {
                "passed": True,
                "min_balanced_accuracy": 0.68,
                "range": 0.08,
                "min_test_size": 8,
            },
        }

    passed = evaluate_v2_quality_gate(
        evaluation,
        explicit_feedback_count=30,
        explicit_focus_count=15,
        explicit_distract_count=15,
        distinct_feedback_days=7,
    )
    failed = evaluate_v2_quality_gate(
        evaluation,
        explicit_feedback_count=19,
        explicit_focus_count=15,
        explicit_distract_count=14,
        distinct_feedback_days=7,
    )

    assert passed["passed"] is True
    assert passed["mode"] == "ready"
    assert failed["passed"] is False
    assert failed["mode"] == "shadow"
    assert failed["checks"]["minimum_explicit_feedback"] is False
