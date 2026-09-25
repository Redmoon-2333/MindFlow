"""Regression tests for label provenance and model publication safety.

These cover the defects found in the audit:

* auxiliary window labels must never enter the deployable evaluation mask;
* a feedback session only labels a window it actually covers;
* contradictory feedback on one window is excluded, not resolved arbitrarily;
* the scaler/calibrator split must not leak across the calibration holdout;
* a candidate that fails its trial load must not become active;
* the ``latest`` pointer and per-version manifests are written atomically.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.train.models.ensemble import EnsembleClassifier
from mindflow.train.models.manager import ModelManager, ModelPublicationError
from mindflow.train.v2 import (
    V2_FEATURE_NAMES,
    evaluate_v2_candidates,
    prepare_v2_training_data,
)

# ── Helpers ─────────────────────────────────────────────────────────────


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
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": features,
    }


def _feedback(
    session_id: str,
    start: datetime,
    score: int,
    label: str,
    *,
    minutes: int = 30,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "start_time": start.isoformat(),
        "end_time": (start + timedelta(minutes=minutes)).isoformat(),
        "score": score,
        "label": label,
        "task_type": "coding",
    }


# ── Label provenance ────────────────────────────────────────────────────


def test_auxiliary_labels_never_enter_evaluation_mask() -> None:
    """The eval mask is feedback-only; that is what makes the gate meaningful."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = []
    for i in range(4):
        window = _feature_window(start + timedelta(hours=i), idle_ratio=0.05)
        window["id"] = f"w{i}"
        windows.append(window)

    data = prepare_v2_training_data(
        windows, [], window_labels={"w0": 1, "w1": 0, "w2": 1, "w3": 0}
    )

    assert data.explicit_mask.sum() == 0, "auxiliary labels are not explicit feedback"
    assert data.window_label_mask is not None
    assert data.window_label_mask.sum() == 4
    assert data.train_mask is not None
    assert data.train_mask.sum() == 4, "they still supervise training"


def test_evaluation_ignores_auxiliary_only_samples() -> None:
    """A dataset with auxiliary labels but no feedback yields insufficient_data."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = []
    for day in range(6):
        for hour in range(4):
            window = _feature_window(
                start + timedelta(days=day, hours=hour), idle_ratio=0.05
            )
            window["id"] = f"w{day}-{hour}"
            windows.append(window)
    labels = {f"w{d}-{h}": (d % 2) for d in range(6) for h in range(4)}

    data = prepare_v2_training_data(windows, [], window_labels=labels)
    result = evaluate_v2_candidates(data)

    assert result["status"] == "insufficient_data"
    assert result["explicit_sample_count"] == 0


def test_partial_overlap_below_threshold_does_not_label() -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = [_feature_window(start, idle_ratio=0.05)]
    # Session starts 4:45 into the 5-minute window -> 15s coverage (5%).
    feedback = [_feedback("s", start + timedelta(minutes=4, seconds=45), 5, "focus")]

    data = prepare_v2_training_data(windows, feedback)

    assert data.matched_window_count == 0
    assert data.explicit_mask.sum() == 0


def test_mixed_feedback_does_not_fall_back_to_weaker_sources() -> None:
    """A `mixed` verdict yields no label — never a window label or heuristic."""
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    window = _feature_window(start, idle_ratio=0.05)
    window["id"] = "w1"
    feedback = [_feedback("s", start, 3, "mixed")]

    data = prepare_v2_training_data(
        [window], feedback, window_labels={"w1": 1}
    )

    assert len(data.features) == 0, "the window must be dropped, not relabelled"
    assert data.ambiguous_window_count == 1


def test_contradictory_feedback_is_excluded_not_arbitrated() -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows = [_feature_window(start, idle_ratio=0.05)]
    feedback = [
        _feedback("a", start, 5, "focus"),
        _feedback("b", start, 1, "distracted"),
    ]

    data = prepare_v2_training_data(windows, feedback)

    assert data.conflict_window_count == 1
    assert len(data.features) == 0
    assert data.explicit_feedback_count == 0


def test_real_feedback_beats_auxiliary_for_same_window() -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    window = _feature_window(start, idle_ratio=0.05)
    window["id"] = "w1"
    feedback = [_feedback("s", start, 5, "focus")]

    data = prepare_v2_training_data(
        [window], feedback, window_labels={"w1": 0}
    )

    assert data.labels.tolist() == [1]
    assert data.label_sources == ["explicit"]
    assert data.explicit_mask.tolist() == [True]
    assert data.window_label_count == 0


# ── Calibration isolation ───────────────────────────────────────────────


def _separable_dataset(
    n_days: int = 8, per_day: int = 12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    X, y, groups = [], [], []
    for day in range(n_days):
        for _ in range(per_day):
            label = int(rng.randint(0, 2))
            centre = 2.0 if label else -2.0
            X.append(rng.normal(centre, 0.5, size=4))
            y.append(label)
            groups.append(f"2026-07-{day + 1:02d}")
    return np.array(X), np.array(y), np.array(groups)


def test_scaler_is_fit_on_base_training_split_only() -> None:
    """Fitting the scaler on all rows leaks the holdout's statistics."""
    X, y, groups = _separable_dataset()
    clf = EnsembleClassifier(calibration="sigmoid")
    clf.fit(X, y, [f"f{i}" for i in range(X.shape[1])], groups=groups)

    # The fitted scaler must match a scaler fitted on the base-train split,
    # which is strictly smaller than the full dataset when calibration is on.
    train_ix, calib_ix = EnsembleClassifier._calibration_split(X, y, groups)
    assert len(calib_ix) > 0, "a holdout must exist for calibration"

    # Recover the split the classifier actually used by comparing means: the
    # scaler's mean must equal the mean of the base-train rows, not of all rows.
    scaler_mean = clf.scaler.mean_
    train_mean = X[train_ix].mean(axis=0)
    all_mean = X.mean(axis=0)
    assert np.allclose(scaler_mean, train_mean)
    assert not np.allclose(scaler_mean, all_mean)


def test_calibration_split_is_group_disjoint() -> None:
    """No capture day may sit on both sides of the calibration split."""
    X, y, groups = _separable_dataset()
    train_ix, calib_ix = EnsembleClassifier._calibration_split(X, y, groups)

    train_groups = set(groups[train_ix])
    calib_groups = set(groups[calib_ix])
    assert not (train_groups & calib_groups), "days must not leak across the split"
    assert len(train_groups) + len(calib_groups) == len(set(groups))


def test_grouped_split_keeps_both_classes_on_each_side() -> None:
    X, y, groups = _separable_dataset()
    train_ix, calib_ix = EnsembleClassifier._calibration_split(X, y, groups)
    assert len(np.unique(y[train_ix])) == 2
    assert len(np.unique(y[calib_ix])) == 2


def test_calibration_disabled_uses_all_rows_for_scaler() -> None:
    """Without a calibrator there is no holdout, so the scaler uses every row."""
    X, y, groups = _separable_dataset()
    clf = EnsembleClassifier(calibration=None)
    clf.fit(X, y, [f"f{i}" for i in range(X.shape[1])], groups=groups)
    assert np.allclose(clf.scaler.mean_, X.mean(axis=0))


# ── Publication safety ──────────────────────────────────────────────────


def _trained_manager(tmp_path: Path) -> ModelManager:
    np.random.seed(0)
    X = np.random.rand(40, len(V2_FEATURE_NAMES))
    y = np.array([i % 2 for i in range(40)])
    manager = ModelManager(models_dir=tmp_path / "v2", use_ensemble=False)
    manager.train_all(X, list(V2_FEATURE_NAMES), y)
    return manager


def test_per_version_manifest_is_written(tmp_path: Path) -> None:
    """Each run keeps its own manifest instead of overwriting one shared file."""
    manager = _trained_manager(tmp_path)
    first = manager.save_all(manifest={"tag_marker": "first"})
    tag_a = next(iter(first.values())).split("-", 1)[1].removesuffix(".pkl")

    manifests = list((tmp_path / "v2").glob("manifest-*.json"))
    assert any(tag_a in m.name for m in manifests)

    second = manager.save_all(manifest={"tag_marker": "second"})
    tag_b = next(iter(second.values())).split("-", 1)[1].removesuffix(".pkl")
    assert tag_a != tag_b
    # Both versions keep their own manifest record.
    names = {m.name for m in (tmp_path / "v2").glob("manifest-*.json")}
    assert f"manifest-{tag_a}.json" in names
    assert f"manifest-{tag_b}.json" in names


def test_latest_pointer_is_valid_json_and_matches_version(tmp_path: Path) -> None:
    manager = _trained_manager(tmp_path)
    saved = manager.save_all(activate=True, manifest={})
    latest = json.loads((tmp_path / "v2" / "latest.json").read_text(encoding="utf-8"))
    assert latest["classifier"] == saved["classifier"]
    assert latest["clustering"] == saved["clustering"]
    assert latest["hmm"] == saved["hmm"]


def test_activation_refuses_when_artifacts_fail_trial_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate that cannot be loaded must not replace the active version."""
    manager = _trained_manager(tmp_path)
    manager.save_all(activate=True, manifest={"generation": "baseline"})
    baseline = json.loads((tmp_path / "v2" / "latest.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(ModelManager, "_verify_loadable", lambda self, names: False)

    with pytest.raises(ModelPublicationError):
        manager.save_all(activate=True, manifest={"generation": "broken"})

    # The pointer still names the baseline version.
    after = json.loads((tmp_path / "v2" / "latest.json").read_text(encoding="utf-8"))
    assert after == baseline


def test_shadow_save_does_not_move_pointer(tmp_path: Path) -> None:
    """A shadow run writes artifacts but leaves the active version alone."""
    manager = _trained_manager(tmp_path)
    manager.save_all(activate=True, manifest={"generation": "active"})
    baseline = json.loads((tmp_path / "v2" / "latest.json").read_text(encoding="utf-8"))

    manager.save_all(activate=False, manifest={"generation": "shadow"})

    after = json.loads((tmp_path / "v2" / "latest.json").read_text(encoding="utf-8"))
    assert after == baseline


def test_atomic_write_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "pointer.json"
    ModelManager._atomic_write_text(target, '{"a": 1}')
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_saved_models_are_reloadable(tmp_path: Path) -> None:
    """The round trip through disk must reproduce a usable classifier."""
    manager = _trained_manager(tmp_path)
    manager.save_all(activate=True, manifest={})

    reloaded = ModelManager(models_dir=tmp_path / "v2", use_ensemble=False)
    assert reloaded.load_latest() is True
    proba = reloaded.classifier.predict_proba(
        np.random.rand(3, len(V2_FEATURE_NAMES))
    )
    assert proba.shape == (3, 2)
