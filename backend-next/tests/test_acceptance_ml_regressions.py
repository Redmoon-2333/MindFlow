"""Synthetic regressions for independent acceptance findings A01-A04 and A15."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.train import pipeline, v2
from mindflow.train.grouping import build_grouping_plan
from mindflow.train.models import ensemble
from mindflow.train.models.ensemble import EnsembleClassifier


@pytest.fixture(autouse=True)
def small_forests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ensemble, "_xgb_available", False)
    monkeypatch.setattr(
        EnsembleClassifier, "_RF_PARAMS",
        {"n_estimators": 5, "max_depth": 3, "random_state": 42, "n_jobs": 1},
    )


def _window(start: datetime, index: int, label: int) -> dict[str, Any]:
    return {
        "id": f"w{index}",
        "window_start_utc": start.isoformat(),
        "window_end_utc": (start + timedelta(minutes=5)).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": {
            "mouse_distance_per_min": float(index),
            "top_app_ratio": 0.95 if label else 0.1,
            "input_active_ratio": 0.8 if label else 0.05,
            "idle_ratio": 0.01 if label else 0.7,
            "app_switch_count": 0 if label else 12,
        },
    }


def _feedback(sid: str, start: datetime, end: datetime, label: int) -> dict[str, Any]:
    return {
        "session_id": sid, "start_time": start.isoformat(), "end_time": end.isoformat(),
        "label": "focus" if label else "distracted", "score": 5 if label else 1,
    }


def _dataset(
    counts: tuple[int, ...] = (4,) * 8, *, auxiliary: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows, feedback = [], []
    base = datetime(2026, 9, 1, 9, tzinfo=UTC)
    for day, count in enumerate(counts):
        for session in range(count):
            start = base + timedelta(days=day, minutes=session * 30)
            label = session % 2
            feedback.append(_feedback(
                f"s{day}-{session}", start, start + timedelta(minutes=15), label,
            ))
            for offset in range(3):
                windows.append(_window(
                    start + timedelta(minutes=offset * 5), len(windows), label,
                ))
    if auxiliary:
        for index in range(10):
            row = _window(base + timedelta(days=len(counts), minutes=index * 5),
                          len(windows), index % 2)
            row["label"] = index % 2
            windows.append(row)
    return windows, feedback


def _passing_evaluation() -> dict[str, Any]:
    return {
        "status": "evaluated",
        "candidate": {"balanced_accuracy": 0.9, "minority_f1": 0.9, "brier_score": 0.1},
        "rule_baseline": {"brier_score": 0.2},
        "fold_stability": {"passed": True},
        "calibration": {"method": "sigmoid", "status": "fitted"},
    }


@pytest.mark.parametrize("group_count", [3, 8])
def test_calibration_never_falls_back_to_rows(group_count: int) -> None:
    groups = np.repeat(np.arange(group_count), 12)
    y = (
        np.tile([0, 1], len(groups) // 2)
        if group_count == 3 else np.isin(groups, [0, 2]).astype(int)
    )
    features = np.zeros((len(y), 1))
    train, calibrate = EnsembleClassifier._calibration_split(features, y, groups)
    assert set(groups[train]).isdisjoint(groups[calibrate])
    assert len(np.unique(y[train])) == len(np.unique(y[calibrate])) == 2


@pytest.mark.parametrize("groups", [np.zeros(40), np.tile([0, 1], 20)])
def test_impossible_grouped_calibration_is_explicit(groups: np.ndarray) -> None:
    clf = EnsembleClassifier(calibration="sigmoid")
    with pytest.raises(ValueError, match="(?i)calibration"):
        clf.fit(np.zeros((40, 1)), np.tile([0, 1], 20), ["x"], groups=groups)
    assert not clf._is_fitted
    assert clf.calibrator is None


@pytest.mark.parametrize("stage", ["_calibration_split", "_fit_calibrator"])
def test_fit_does_not_silently_disable_requested_calibration(
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected calibration failure")

    clf = EnsembleClassifier(calibration="sigmoid")
    monkeypatch.setattr(clf, stage, fail)
    with pytest.raises(ValueError, match="(?i)calibration"):
        clf.fit(np.zeros((80, 1)), np.tile([0, 1], 40), ["x"])
    assert not clf._is_fitted


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_preserves_session_weight_under_duplicate_windows(
    monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    probabilities = []
    for copies in (1, 10):
        clf = EnsembleClassifier(calibration=method)
        y = np.concatenate([np.tile([0, 1], 10), np.ones(10), np.zeros(10 * copies)])
        weights = np.concatenate([np.ones(20), np.full(10, 0.1),
                                  np.full(10 * copies, 0.1 / copies)])
        monkeypatch.setattr(
            clf, "_calibration_split",
            lambda features, y, groups: (np.arange(20), np.arange(20, len(y))),
        )
        monkeypatch.setattr(
            clf, "_soft_vote_proba", lambda features: np.full((len(features), 2), 0.5),
        )
        clf.fit(np.zeros((len(y), 1)), y, ["x"], sample_weight=weights)
        probabilities.append(float(clf.predict_proba(np.zeros((1, 1)))[0, 1]))
    assert probabilities == pytest.approx([0.5, 0.5], abs=1e-3)


def test_evaluation_passes_merged_groups_and_recomputes_fold_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows, feedback = _dataset((2, 18, 2, 2), auxiliary=True)
    data = v2.prepare_v2_training_data(
        windows, feedback, window_labels=pipeline._extract_window_labels(windows),
    )
    # The merged contract is authoritative, even with auxiliary rows present.
    data.group_ids = ["merged" if d in data.dates[:6] else d for d in data.group_ids]
    data.sample_weights[:] = 999.0  # stale full-dataset weights must not be sliced
    calls: list[tuple[Any, Any, Any, Any, dict[str, Any]]] = []
    original = v2.fit_candidate
    marker = data.feature_names.index("mouse_distance_per_min")

    def capture(
        name: str, features: Any, y: Any, sample_weight: Any, *args: Any, **kwargs: Any,
    ) -> Any:
        calls.append((features.copy(), y.copy(), sample_weight.copy(), name, kwargs.copy()))
        return original(name, features, y, sample_weight, *args, **kwargs)

    monkeypatch.setattr(v2, "fit_candidate", capture)
    result = v2.evaluate_v2_candidates(data, calibration="sigmoid")

    # Every candidate is fitted once per evaluable fold of both schemes.
    assert len(calls) >= 3
    # Include folds that failed *after* fitting (e.g. calibration_unavailable):
    # their train_groups were still declared and still fed to the candidates.
    fold_train_sets = [
        set(fold["train_groups"])
        for scheme in (result["folds"], result["forward_chaining"]["folds"])
        for fold in scheme
        if fold.get("train_groups")
    ]
    assert fold_train_sets
    for features, _y, weights, _name, kwargs in calls:
        indices = features[:, marker].astype(int)
        assert kwargs["groups"].tolist() == [data.group_ids[i] for i in indices]
        assert kwargs["label_sources"].tolist() == [data.label_sources[i] for i in indices]
        assert kwargs["session_ids"].tolist() == [data.sample_feedback_ids[i] for i in indices]
        # No trained row may come from outside a declared fold's training
        # groups — that is what makes "leave future days out" true.
        trained = set(kwargs["groups"].tolist())
        assert any(trained <= declared for declared in fold_train_sets)
        # Session-balanced fold weights are recomputed, never the stale 999.0.
        assert weights.max() <= 1.0 + 1e-9
        explicit = data.explicit_mask[indices]
        sessions = np.asarray(data.sample_feedback_ids)[indices]
        for sid in set(sessions[explicit]):
            assert weights[sessions == sid].sum() == pytest.approx(1.0)
        assert weights[~explicit].sum() <= weights[explicit].sum() + 1e-9

    # No fold may leak a group (and therefore a session) across its boundary.
    for scheme in (result["folds"], result["forward_chaining"]["folds"]):
        for fold in scheme:
            assert set(fold["train_groups"]).isdisjoint(fold["test_groups"])
    # Calibration is attempted for every fold. When a fold cannot fit it the
    # failure is reported (never silently ignored), otherwise every fold
    # reports a fitted calibrator.
    assert result["calibration"]["method"] == "sigmoid"
    assert result["calibration"]["status"] in {"fitted", "unavailable"}
    if result["calibration"]["status"] == "unavailable":
        assert result["calibration"]["reason"]
        assert any(
            fold.get("calibration", {}).get("status") == "unavailable"
            for scheme in (result["folds"], result["forward_chaining"]["folds"])
            for fold in scheme
        )
    else:
        assert all(
            fold["calibration"]["status"] == "fitted"
            for scheme in (result["folds"], result["forward_chaining"]["folds"])
            for fold in scheme
            if "calibration" in fold
        )


@pytest.mark.parametrize("stage", ["_calibration_split", "_fit_calibrator"])
@pytest.mark.parametrize("partial", [False, True])
def test_evaluation_calibration_failure_is_reported_and_blocks_gate(
    monkeypatch: pytest.MonkeyPatch, stage: str, partial: bool,
) -> None:
    windows, feedback = _dataset()
    original = getattr(EnsembleClassifier, stage)
    calls = 0

    def fail(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if not partial or calls == 1:
            raise RuntimeError("injected calibration failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        EnsembleClassifier, stage, staticmethod(fail) if stage == "_calibration_split" else fail,
    )
    result = v2.evaluate_v2_candidates(
        v2.prepare_v2_training_data(windows, feedback), calibration="sigmoid",
    )
    assert result["status"] == "calibration_unavailable"
    assert result["calibration"]["status"] == "unavailable"
    failures = [f for f in result["folds"] if f["calibration"]["status"] == "unavailable"]
    assert len(failures) == (1 if partial else len(result["folds"]))
    assert bool(result["candidate"]) is partial
    assert not v2.evaluate_v2_quality_gate(
        result, explicit_feedback_count=32, explicit_focus_count=16,
        explicit_distract_count=16, distinct_feedback_days=8,
    )["passed"]


@pytest.mark.parametrize("status", [None, "unavailable"])
def test_quality_gate_fails_closed_without_successful_calibration(status: str | None) -> None:
    evaluation = _passing_evaluation()
    if status is None:
        del evaluation["calibration"]
    else:
        evaluation["calibration"]["status"] = status
    gate = v2.evaluate_v2_quality_gate(
        evaluation, explicit_feedback_count=32, explicit_focus_count=16,
        explicit_distract_count=16, distinct_feedback_days=8,
    )
    assert gate["passed"] is False
    assert gate["deployment_tier"] == "shadow"
    assert gate["checks"]["calibration_available"] is False


@pytest.mark.parametrize(
    ("counts", "failed_check"),
    [
        ({"distinct_feedback_days": 6}, "minimum_days"),
        ({"explicit_feedback_count": 19}, "minimum_explicit_feedback"),
        ({"explicit_focus_count": 4}, "minimum_class_feedback"),
        ({"explicit_distract_count": 4}, "minimum_class_feedback"),
    ],
)
def test_calibration_does_not_relax_feedback_gates(
    counts: dict[str, int], failed_check: str,
) -> None:
    gate_counts = {
        "explicit_feedback_count": 32, "explicit_focus_count": 16,
        "explicit_distract_count": 16, "distinct_feedback_days": 8,
    }
    gate_counts.update(counts)
    gate = v2.evaluate_v2_quality_gate(_passing_evaluation(), **gate_counts)
    assert not gate["passed"]
    assert gate["mode"] == "shadow"
    assert gate["checks"][failed_check] is False


def test_overlapping_session_date_links_do_not_depend_on_attribution_order() -> None:
    start = datetime(2026, 9, 1, 23, 55, tzinfo=UTC)
    windows = [_window(start, 0, 1), _window(start + timedelta(minutes=10), 1, 1)]
    short = _feedback("short", start, start + timedelta(minutes=5), 1)
    overnight = _feedback("overnight", start, start + timedelta(minutes=20), 1)
    first = v2.prepare_v2_training_data(windows, [short, overnight])
    second = v2.prepare_v2_training_data(windows, [overnight, short])
    for data in (first, second):
        assert data.session_dates["overnight"] == ["2026-09-01", "2026-09-02"]
        assert len(set(data.group_ids)) == 1
    assert first.group_ids == second.group_ids
    assert first.sample_feedback_ids == second.sample_feedback_ids


@pytest.mark.parametrize("check", ["groups", "report"])
def test_public_pipeline_uses_merged_groups_and_complete_version_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, check: str,
) -> None:
    windows, feedback = _dataset(auxiliary=True)
    midnight = datetime(2026, 9, 1, 23, 55, tzinfo=UTC)
    windows.extend([_window(midnight, len(windows), 1),
                    _window(midnight + timedelta(minutes=10), len(windows) + 1, 1)])
    feedback.append(_feedback("overnight", midnight, midnight + timedelta(minutes=20), 1))
    # Include excluded rows so every provenance field is checked nontrivially.
    for offset, score in ((1, 1), (2, 3)):
        start = datetime(2026, 9, 20, 9, tzinfo=UTC) + timedelta(hours=offset)
        windows.append(_window(start, len(windows), 1))
        feedback.append(_feedback(f"extra-{offset}", start, start + timedelta(minutes=5), 1))
        extra = _feedback(f"other-{offset}", start, start + timedelta(minutes=5), 0)
        extra["score"] = score
        feedback.append(extra)
    data = v2.prepare_v2_training_data(
        windows, feedback, window_labels=pipeline._extract_window_labels(windows),
    )
    calls = []
    original = pipeline.ModelManager.train_all

    def capture(self: Any, features: Any, names: Any, y: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(self, features, names, y, **kwargs)

    original_prepare = pipeline.prepare_v2_training_data

    def stale_weights(*args: Any, **kwargs: Any) -> Any:
        prepared = original_prepare(*args, **kwargs)
        prepared.sample_weights[:] = 999.0
        return prepared

    monkeypatch.setattr(pipeline.ModelManager, "train_all", capture)
    monkeypatch.setattr(pipeline, "prepare_v2_training_data", stale_weights)
    report = pipeline.run_training(
        source="db", data_dir=tmp_path / "data", models_dir=tmp_path / "models",
        feature_windows=windows, feedback_sessions=feedback, use_window_labels=True,
    )
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["sample_weight"], data.sample_weights[data.train_mask])
    assert calls[0]["label_sources"].tolist() == (
        np.asarray(data.label_sources)[data.train_mask].tolist()
    )
    assert calls[0]["session_ids"].tolist() == (
        np.asarray(data.sample_feedback_ids)[data.train_mask].tolist()
    )
    if check == "groups":
        assert calls[0]["groups"].tolist() == np.asarray(data.group_ids)[data.train_mask].tolist()
    assert report.version_tag is not None
    root = tmp_path / "models" / "v2"
    version = json.loads((root / f"training_report-{report.version_tag}.json").read_text("utf-8"))
    shared = json.loads((root / "training_report.json").read_text("utf-8"))
    assert version == shared == report.to_dict()
    assert version["label_source_counts"]["explicit"] > 0
    assert version["conflict_window_count"] == version["ambiguous_window_count"] == 1
    assert version["feature_schema_version"] == FEATURE_SCHEMA_VERSION


@pytest.mark.parametrize("stage", ["_calibration_split", "_fit_calibrator", "silent"])
def test_public_pipeline_controls_deployment_calibration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    windows, feedback = _dataset()
    monkeypatch.setattr(pipeline, "evaluate_v2_candidates", lambda *a, **k: _passing_evaluation())

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("injected deployment calibration failure")

    if stage == "silent":
        original_fit = EnsembleClassifier.fit

        def silently_unavailable(self: Any, *args: Any, **kwargs: Any) -> Any:
            fitted = original_fit(self, *args, **kwargs)
            self.calibrator = None
            return fitted

        monkeypatch.setattr(EnsembleClassifier, "fit", silently_unavailable)
    else:
        monkeypatch.setattr(EnsembleClassifier, stage, fail)
    root = tmp_path / "models" / "v2"
    root.mkdir(parents=True)
    active = root / "latest.json"
    active.write_text('{"classifier": "previous.pkl"}', encoding="utf-8")
    report = pipeline.run_training(
        source="db", data_dir=tmp_path / "data", models_dir=tmp_path / "models",
        feature_windows=windows, feedback_sessions=feedback,
    )
    assert report.activated is False
    assert report.model_mode == "shadow"
    assert report.quality_gate["passed"] is False
    assert report.quality_gate["checks"]["calibration_available"] is False
    assert report.classifier["calibration"]["status"] == "unavailable"
    assert ("not fitted" if stage == "silent" else "injected deployment") in (
        report.classifier["calibration"]["reason"]
    )
    assert report.saved_models == {}
    assert active.read_text("utf-8") == '{"classifier": "previous.pkl"}'
    assert json.loads((root / "training_report.json").read_text("utf-8")) == report.to_dict()


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("unchanged_side", ["base", "calibration"])
@pytest.mark.parametrize("auxiliary_side", ["base", "calibration", "both"])
def test_internal_fit_budgets_ignore_other_side_sessions(
    monkeypatch: pytest.MonkeyPatch, method: str, unchanged_side: str, auxiliary_side: str,
) -> None:
    captured: dict[str, np.ndarray] = {}
    calibrator_type = (
        ensemble.LogisticRegression if method == "sigmoid" else ensemble.IsotonicRegression
    )
    original_calibrator_fit = calibrator_type.fit

    def capture_calibrator(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured["calibration"] = np.asarray(kwargs["sample_weight"]).copy()
        return original_calibrator_fit(self, *args, **kwargs)

    monkeypatch.setattr(calibrator_type, "fit", capture_calibrator)
    # Discover the real deterministic group holdout, without mocking the split.
    template_groups = np.repeat(np.arange(4), 6)
    template_y = np.tile([0, 1], 12)
    _, template_cal = EnsembleClassifier._calibration_split(
        np.zeros((24, 1)), template_y, template_groups,
    )
    calibration_group = int(template_groups[template_cal[0]])
    base_group = next(group for group in range(4) if group != calibration_group)
    previous: np.ndarray | None = None

    for other_sessions in (2, 18):
        groups, labels, sources, session_ids = [], [], [], []
        changed_group = calibration_group if unchanged_side == "base" else base_group
        for group in range(4):
            session_count = other_sessions if group == changed_group else 2
            for session in range(session_count):
                for _ in range(3):
                    groups.append(group)
                    labels.append(session % 2)
                    sources.append("explicit")
                    session_ids.append(f"g{group}-s{session}")
            side = "calibration" if group == calibration_group else "base"
            if auxiliary_side in (side, "both"):
                for row in range(8):
                    groups.append(group)
                    labels.append(row % 2)
                    sources.append("window_label")
                    session_ids.append("")
        features = np.arange(len(labels), dtype=float).reshape(-1, 1)
        group_arr, y = np.asarray(groups), np.asarray(labels)
        source_arr, session_arr = np.asarray(sources), np.asarray(session_ids)
        global_plan = build_grouping_plan(
            dates=[str(group) for group in groups], labels=labels, sources=sources,
            window_dates_by_session={}, session_id_by_sample=session_ids,
        )
        train, calibrate = EnsembleClassifier._calibration_split(features, y, group_arr)
        assert set(group_arr[calibrate]) == {calibration_group}
        clf = EnsembleClassifier(calibration=method)
        original_rf_fit = clf.rf_model.fit

        def capture_rf(*args: Any, original_fit: Any = original_rf_fit, **kwargs: Any) -> Any:
            captured["base"] = np.asarray(kwargs["sample_weight"]).copy()
            return original_fit(*args, **kwargs)

        class XGBProbe:
            def fit(self, features: Any, labels: Any, sample_weight: Any) -> None:
                captured["xgb"] = np.asarray(sample_weight).copy()

            def predict_proba(self, features: Any) -> np.ndarray:
                return np.full((len(features), 2), 0.5)

        monkeypatch.setattr(clf.rf_model, "fit", capture_rf)
        clf._xgb_available = True
        clf.xgb_model = XGBProbe()
        clf.fit(
            features, y, ["x"], sample_weight=np.asarray(global_plan.weights),
            groups=group_arr, label_sources=source_arr, session_ids=session_arr,
        )
        np.testing.assert_array_equal(captured["xgb"], captured["base"])
        for side, indices in (("base", train), ("calibration", calibrate)):
            weights = captured[side]
            explicit = source_arr[indices] == "explicit"
            auxiliary = source_arr[indices] == "window_label"
            local_sessions = session_arr[indices]
            for sid in set(local_sessions[explicit]):
                assert weights[local_sessions == sid].sum() == pytest.approx(1.0)
            assert weights[auxiliary].sum() <= weights[explicit].sum() + 1e-9
        unchanged = captured[unchanged_side]
        if previous is not None:
            np.testing.assert_allclose(unchanged, previous)
        previous = unchanged.copy()


@pytest.mark.parametrize("method", [None, "sigmoid", "isotonic"])
def test_generic_weights_without_provenance_remain_exact(
    monkeypatch: pytest.MonkeyPatch, method: str | None,
) -> None:
    features = np.arange(48, dtype=float).reshape(-1, 1)
    labels, groups = np.tile([0, 1], 24), np.repeat(np.arange(4), 12)
    weights = np.linspace(0.25, 3.0, 48)
    clf = EnsembleClassifier(calibration=method)
    captured = {}
    original_rf_fit, original_calibrator_fit = clf.rf_model.fit, clf._fit_calibrator

    def capture_rf(*args: Any, **kwargs: Any) -> Any:
        captured["base"] = kwargs["sample_weight"].copy()
        return original_rf_fit(*args, **kwargs)

    def capture_calibrator(*args: Any, **kwargs: Any) -> Any:
        captured["calibration"] = kwargs["sample_weight"].copy()
        return original_calibrator_fit(*args, **kwargs)

    monkeypatch.setattr(clf.rf_model, "fit", capture_rf)
    monkeypatch.setattr(clf, "_fit_calibrator", capture_calibrator)
    clf.fit(features, labels, ["x"], sample_weight=weights, groups=groups)
    if method is None:
        np.testing.assert_array_equal(captured["base"], weights)
    else:
        train, calibrate = clf._calibration_split(features, labels, groups)
        np.testing.assert_array_equal(captured["base"], weights[train])
        np.testing.assert_array_equal(captured["calibration"], weights[calibrate])


def test_manager_filters_provenance_with_training_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows, feedback = _dataset(auxiliary=True)
    data = v2.prepare_v2_training_data(
        windows, feedback, window_labels=pipeline._extract_window_labels(windows),
    )
    manager = pipeline.ModelManager(models_dir=tmp_path, use_ensemble=True, calibration="sigmoid")
    assert isinstance(manager.classifier, EnsembleClassifier)
    captured = {}
    original_fit = manager.classifier.fit

    def capture(features: Any, labels: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return original_fit(features, labels, **kwargs)

    monkeypatch.setattr(manager.classifier, "fit", capture)
    weights = data.sample_weights.copy()
    weights[::5] = 0
    keep = weights >= 0.01
    manager.train_all(
        data.features, data.feature_names, data.labels, sample_weight=weights,
        min_confidence=0.01, groups=np.asarray(data.group_ids),
        label_sources=np.asarray(data.label_sources),
        session_ids=np.asarray(data.sample_feedback_ids),
    )
    np.testing.assert_array_equal(captured["sample_weight"], weights[keep])
    np.testing.assert_array_equal(captured["groups"], np.asarray(data.group_ids)[keep])
    np.testing.assert_array_equal(captured["label_sources"], np.asarray(data.label_sources)[keep])
    np.testing.assert_array_equal(
        captured["session_ids"], np.asarray(data.sample_feedback_ids)[keep],
    )


@pytest.mark.parametrize("auxiliary_side", ["base", "calibration"])
def test_internal_auxiliary_only_side_cannot_borrow_other_side_budget(
    monkeypatch: pytest.MonkeyPatch, auxiliary_side: str,
) -> None:
    features = np.arange(48, dtype=float).reshape(-1, 1)
    labels, groups = np.tile([0, 1], 24), np.repeat(np.arange(4), 12)
    train, calibrate = EnsembleClassifier._calibration_split(features, labels, groups)
    sources = np.full(48, "explicit", dtype=object)
    sessions = np.asarray([f"s{index // 2}" for index in range(48)], dtype=object)
    auxiliary = train if auxiliary_side == "base" else calibrate
    sources[auxiliary], sessions[auxiliary] = "window_label", ""
    clf = EnsembleClassifier(calibration="sigmoid")
    fitted = []
    monkeypatch.setattr(clf.rf_model, "fit", lambda *args, **kwargs: fitted.append(True))
    with pytest.raises(ensemble.CalibrationUnavailableError, match="local supervision weight"):
        clf.fit(
            features, labels, ["x"], sample_weight=np.ones(48), groups=groups,
            label_sources=sources, session_ids=sessions,
        )
    assert not fitted
    assert not clf._is_fitted


def test_auxiliary_only_uncalibrated_shadow_keeps_nominal_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, labels = np.arange(20, dtype=float).reshape(-1, 1), np.tile([0, 1], 10)
    clf = EnsembleClassifier(calibration=None)
    captured = []
    original_fit = clf.rf_model.fit

    def capture(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["sample_weight"].copy())
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(clf.rf_model, "fit", capture)
    clf.fit(
        features, labels, ["x"], sample_weight=np.full(20, 999.0),
        label_sources=np.full(20, "window_label"), session_ids=np.full(20, ""),
    )
    np.testing.assert_allclose(captured[0], np.full(20, 1 / 20))
