"""Publication guard regressions (plan item 1).

A run may move the active pointer only when the candidate the evaluation
selected and the classifier about to be saved are confirmed to be the same
model.  Everything else — no selection, an unconfirmable artifact, a mismatch,
or a silent XGBoost fallback during training — is saved as a shadow version with
an explicit reason, and the previously active version is left untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mindflow.train.candidates import (
    LOGISTIC_REGRESSION,
    RANDOM_FOREST,
    RF_XGB_SOFT_VOTING,
    RULE_ENGINE,
)
from mindflow.train.models.classifier import FocusClassifier
from mindflow.train.models.ensemble import EnsembleClassifier
from mindflow.train.pipeline import TrainingReport
from mindflow.train.publication import (
    REASON_MISMATCH,
    REASON_NOT_EVALUATED,
    REASON_RF_ONLY_ARTIFACT,
    REASON_UNCONFIRMED_ARTIFACT,
    REASON_XGB_FALLBACK,
    PublicationDecision,
    deployed_candidate_name,
    evaluate_publication,
    selected_candidate,
)


def _evaluation(selected: str | None) -> dict[str, Any]:
    return {"candidate_selection": {"selected": selected, "status": "selected"}}


# ═══════════════════════════════════════════════════════════════════════════════
# Candidate / artifact mapping
# ═══════════════════════════════════════════════════════════════════════════════


def test_ensemble_maps_to_the_soft_voting_candidate() -> None:
    assert deployed_candidate_name(EnsembleClassifier()) == RF_XGB_SOFT_VOTING


def test_rf_only_classifier_is_not_confirmable() -> None:
    """The RF-only fallback is *not* the evaluated `random_forest` candidate.

    `fit_candidate(RANDOM_FOREST)` adds `class_weight="balanced"` and is scored
    through the candidate pipeline; `FocusClassifier` is a bare RandomForest
    behind a scaler. Mapping one onto the other would authorise activating a
    model that was never evaluated, so the guard refuses to name it.
    """
    assert deployed_candidate_name(FocusClassifier()) is None


@pytest.mark.parametrize("artifact", [None, object(), "classifier"])
def test_unknown_artifact_is_unconfirmable(artifact: object) -> None:
    assert deployed_candidate_name(artifact) is None


def test_selected_candidate_reads_the_selection_block() -> None:
    assert selected_candidate(_evaluation(RANDOM_FOREST)) == RANDOM_FOREST
    assert selected_candidate(_evaluation(None)) is None
    assert selected_candidate({}) is None
    assert selected_candidate(None) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Decision table
# ═══════════════════════════════════════════════════════════════════════════════


def test_matching_candidate_and_artifact_allows_activation() -> None:
    decision = evaluate_publication(_evaluation(RF_XGB_SOFT_VOTING), EnsembleClassifier())

    assert decision.allowed is True
    assert decision.evaluation_candidate == RF_XGB_SOFT_VOTING
    assert decision.deployed_candidate == RF_XGB_SOFT_VOTING
    assert "一致" in decision.reason


def test_rf_only_selection_with_no_ensemble_is_still_blocked() -> None:
    """No evaluated candidate reproduces the RF-only artifact, ever."""
    decision = evaluate_publication(_evaluation(RANDOM_FOREST), FocusClassifier())
    assert decision.allowed is False


def test_rf_only_artifact_is_blocked_with_its_own_reason() -> None:
    decision = evaluate_publication(_evaluation(RANDOM_FOREST), FocusClassifier())

    assert decision.allowed is False
    assert decision.reason == REASON_RF_ONLY_ARTIFACT
    assert decision.deployed_candidate is None


@pytest.mark.parametrize("selected", [LOGISTIC_REGRESSION, RANDOM_FOREST, RULE_ENGINE])
def test_simpler_selection_blocks_ensemble_publication(selected: str) -> None:
    """The pipeline cannot publish a candidate it does not train."""
    decision = evaluate_publication(_evaluation(selected), EnsembleClassifier())

    assert decision.allowed is False
    assert decision.evaluation_candidate == selected
    assert decision.deployed_candidate == RF_XGB_SOFT_VOTING
    assert selected in decision.reason
    assert REASON_MISMATCH.format(selected=selected, deployed=RF_XGB_SOFT_VOTING) == decision.reason


def test_missing_selection_blocks_activation() -> None:
    decision = evaluate_publication(_evaluation(None), EnsembleClassifier())

    assert decision.allowed is False
    assert decision.reason == REASON_NOT_EVALUATED
    assert decision.evaluation_candidate is None


def test_unconfirmable_artifact_blocks_activation() -> None:
    decision = evaluate_publication(_evaluation(RF_XGB_SOFT_VOTING), object())

    assert decision.allowed is False
    assert decision.reason == REASON_UNCONFIRMED_ARTIFACT
    assert decision.deployed_candidate is None


def test_xgb_fallback_is_reported_as_a_fallback_not_a_plain_mismatch() -> None:
    """Asking for the ensemble and silently getting RF-only must block."""
    decision = evaluate_publication(
        _evaluation(RF_XGB_SOFT_VOTING), FocusClassifier(), xgb_fallback=True,
    )

    assert decision.allowed is False
    assert decision.reason == REASON_XGB_FALLBACK.format(selected=RF_XGB_SOFT_VOTING)
    assert "XGBoost" in decision.reason


def test_decision_serialises_for_report_and_manifest() -> None:
    payload = evaluate_publication(
        _evaluation(LOGISTIC_REGRESSION), EnsembleClassifier(),
    ).to_dict()

    assert set(payload) == {
        "allowed", "evaluation_candidate", "deployed_candidate", "reason",
    }
    assert payload["allowed"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# Training-report contract
# ═══════════════════════════════════════════════════════════════════════════════


def test_training_report_carries_the_publication_evidence() -> None:
    """Report and manifest expose candidate, artifact and the block reason."""
    report = TrainingReport(source="db")
    assert report.publication == {}
    assert report.evaluation_candidate is None
    assert report.deployed_classifier is None
    assert report.activation_blocked_reason is None

    report.publication = PublicationDecision(
        False, LOGISTIC_REGRESSION, RF_XGB_SOFT_VOTING, "mismatch",
    ).to_dict()
    report.evaluation_candidate = LOGISTIC_REGRESSION
    report.deployed_classifier = RF_XGB_SOFT_VOTING
    report.activation_blocked_reason = "mismatch"

    dumped = report.to_dict()
    assert dumped["publication"]["allowed"] is False
    assert dumped["evaluation_candidate"] == LOGISTIC_REGRESSION
    assert dumped["deployed_classifier"] == RF_XGB_SOFT_VOTING
    assert dumped["activation_blocked_reason"] == "mismatch"


class _StubManager:
    """Minimal manager double: the pipeline only reads the classifier + flags."""

    def __init__(self, classifier: object, *, requested: bool = True, used: bool = False):
        self.classifier = classifier
        self.was_ensemble_requested = requested
        self.use_ensemble = used


@pytest.mark.parametrize(
    ("selected", "classifier", "requested", "used", "expected_allowed"),
    [
        (RF_XGB_SOFT_VOTING, EnsembleClassifier(), True, True, True),
        (RANDOM_FOREST, FocusClassifier(), True, False, False),
        (RANDOM_FOREST, FocusClassifier(), False, False, False),
        (LOGISTIC_REGRESSION, EnsembleClassifier(), True, True, False),
        (RANDOM_FOREST, EnsembleClassifier(), True, True, False),
        (RF_XGB_SOFT_VOTING, FocusClassifier(), True, False, False),
        (None, EnsembleClassifier(), True, True, False),
    ],
)
def test_guard_table_against_real_classifier_types(
    selected: str | None,
    classifier: object,
    requested: bool,
    used: bool,
    expected_allowed: bool,
) -> None:
    manager = _StubManager(classifier, requested=requested, used=used)
    decision = evaluate_publication(
        _evaluation(selected),
        manager.classifier,
        xgb_fallback=manager.was_ensemble_requested and not manager.use_ensemble,
    )
    assert decision.allowed is expected_allowed


def test_pipeline_module_exposes_the_guard_wiring() -> None:
    """The pipeline must consult the guard before deciding to activate."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "mindflow" / "train" / "pipeline.py"
    ).read_text(encoding="utf-8")

    assert "evaluate_publication(" in source
    assert "and publication.allowed" in source
    assert '"publication": publication.to_dict()' in source


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end: a quality-gate pass that must NOT activate
# ═══════════════════════════════════════════════════════════════════════════════


def _separable_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Windows + explicit feedback that clear every quality gate.

    Deliberately perfectly separable: all five candidates reach the same
    accuracy, so the conservative selection rule keeps the *simplest* one while
    the pipeline trains the ensemble — the exact disagreement the guard exists
    to catch.
    """
    from datetime import UTC, datetime, timedelta

    from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION

    start = datetime(2026, 7, 1, 8, tzinfo=UTC)
    windows: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    index = 0
    for day in range(8):
        for slot in range(12):
            is_focus = slot % 2 == 0
            session_start = start + timedelta(days=day, hours=slot)
            windows.append({
                "window_start_utc": session_start.isoformat(),
                "window_end_utc": (session_start + timedelta(minutes=5)).isoformat(),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "features": {
                    "idle_ratio": 0.01 if is_focus else 0.7,
                    "longest_segment_ratio": 0.98 if is_focus else 0.05,
                    "top_app_ratio": 0.98 if is_focus else 0.1,
                    "input_active_ratio": 0.7 if is_focus else 0.05,
                    "app_switch_count": 0 if is_focus else 12,
                    "domain_switch_count": 0 if is_focus else 8,
                },
            })
            feedback.append({
                "session_id": f"session-{index}",
                "start_time": session_start.isoformat(),
                "end_time": (session_start + timedelta(minutes=30)).isoformat(),
                "label": "focus" if is_focus else "distracted",
                "score": 5 if is_focus else 1,
                "task_type": "coding",
            })
            index += 1
    return windows, feedback


def test_passing_gate_with_an_inconsistent_candidate_stays_shadow(tmp_path: Path) -> None:
    """The whole point of the guard: gate passes, candidate disagrees → shadow.

    The run still writes its artifacts (a later manual activation can adopt
    them), but the active pointer must not move and the reason must be recorded
    in both the report and the manifest.
    """
    import json as _json

    from mindflow.train.pipeline import run_training

    windows, feedback = _separable_dataset()
    models_dir = tmp_path / "models"
    report = run_training(
        source="db",
        data_dir=tmp_path / "data",
        models_dir=models_dir,
        feature_windows=windows,
        feedback_sessions=feedback,
        calibration=None,
        allow_activation=True,  # the caller DID permit activation
    )

    # The gate really passed — this block is the publication guard, not the gate.
    assert report.quality_gate["passed"] is True
    assert report.activation_allowed is True
    assert report.publication["allowed"] is False
    assert report.activated is False
    assert report.model_mode == "shadow"
    assert report.activation_blocked_reason
    assert report.evaluation_candidate != report.deployed_classifier

    v2_dir = models_dir / "v2"
    assert not (v2_dir / "latest.json").exists(), "active pointer must not move"
    assert report.version_tag
    assert (v2_dir / f"classifier-{report.version_tag}.pkl").exists()

    # Both the report and the manifest carry the verifiable evidence.
    persisted = _json.loads(
        (v2_dir / f"training_report-{report.version_tag}.json").read_text(encoding="utf-8")
    )
    assert persisted["publication"]["allowed"] is False
    assert persisted["evaluation_candidate"] == report.evaluation_candidate
    assert persisted["deployed_classifier"] == report.deployed_classifier
    assert persisted["activation_blocked_reason"] == report.activation_blocked_reason

    manifest = _json.loads((v2_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["publication"]["allowed"] is False
    assert manifest["activation_blocked_reason"] == report.activation_blocked_reason
