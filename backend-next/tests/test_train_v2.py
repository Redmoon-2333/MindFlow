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


def test_window_labels_become_explicit_annotated_samples() -> None:
    """Option B: user-calibrated window labels become strong annotated samples."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = [
        _feature_window(start, idle_ratio=0.05),
        _feature_window(start + timedelta(hours=1), idle_ratio=0.05),
        _feature_window(start + timedelta(hours=2), idle_ratio=0.05),
    ]
    windows[0]["id"] = "w-focus"
    windows[1]["id"] = "w-distract"
    windows[2]["id"] = "w-unlabeled"
    window_labels = {"w-focus": 1, "w-distract": 0, "w-unlabeled": -1}

    data = prepare_v2_training_data(windows, [], window_labels=window_labels)

    # -1 (mixed) window label drops the window entirely.
    assert data.labels.tolist() == [1, 0]
    assert data.sample_weights.tolist() == [0.8, 0.8]
    assert data.label_sources == ["window_label", "window_label"]
    assert data.explicit_mask.tolist() == [True, True]
    assert data.window_label_count == 2
    # Quality-gate counts remain feedback-only.
    assert data.explicit_feedback_count == 0
    assert data.distinct_feedback_days == 0


def test_explicit_feedback_wins_over_window_label() -> None:
    """A window overlapping explicit feedback keeps the feedback label."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = [_feature_window(start, idle_ratio=0.05)]
    windows[0]["id"] = "w1"
    feedback = [_feedback("focus", start, 5, "focus")]
    window_labels = {"w1": 0}  # contradicted by explicit feedback

    data = prepare_v2_training_data(windows, feedback, window_labels=window_labels)

    assert data.labels.tolist() == [1]
    assert data.sample_weights.tolist() == [1.0]
    assert data.label_sources == ["explicit"]
    assert data.explicit_feedback_count == 1
    assert data.window_label_count == 0


def test_window_label_excluded_mixed_increments_mixed_count() -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = [_feature_window(start, idle_ratio=0.05)]
    windows[0]["id"] = "w1"
    data = prepare_v2_training_data(windows, [], window_labels={"w1": -1})
    assert len(data.features) == 0
    assert data.mixed_window_count == 1
    assert data.window_label_count == 0


def test_extract_window_labels_supports_strings_and_ints() -> None:
    from mindflow.train.pipeline import _extract_window_labels

    windows = [
        {"id": "a", "label": "focus"},
        {"id": "b", "label": "distracted"},
        {"id": "c", "label": 1},
        {"id": "d", "label": "mixed"},
        {"id": "e", "label": None},
        {"id": "f"},  # no label key
        {"id": "g", "label": "bogus"},
    ]
    labels = _extract_window_labels(windows)
    assert labels == {"a": 1, "b": 0, "c": 1, "d": -1}



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
