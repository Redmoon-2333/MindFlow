"""Publication guard — the published artifact must be the evaluated candidate.

The evaluation (`train/evaluation.py`, `train/candidates.py`) compares five
candidates and *selects* one by the stability rule.  The training pipeline, on
the other hand, trains exactly one artifact (`ModelManager.train_all` builds the
RF+XGB soft-voting ensemble, or an RF-only classifier when XGBoost is missing).

Those two can disagree: the selection rule deliberately prefers a *simpler*
model unless a complex one is stably better, so it may select
``logistic_regression`` while the pipeline would still save the ensemble.  Worse,
the ensemble silently degrades to RF-only when XGBoost is not installed.  Both
cases would activate an artifact that was never the one evaluated, which is
exactly the risk this guard removes:

    the active pointer may move only when the evaluated candidate and the
    artifact about to be saved are confirmed to be the same model.

Everything else is still saved (as a shadow version) with an explicit reason, so
a later manual activation or a future round that trains per-candidate artifacts
can adopt it.  This module is a pure decision function: it trains nothing, moves
nothing, and returns a decision the caller records in the training report and
the model manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mindflow.train.candidates import (
    LOGISTIC_REGRESSION,
    RF_XGB_SOFT_VOTING,
    RULE_ENGINE,
    XGBOOST,
)

#: Candidate name for the artifact `ModelManager` builds when the ensemble is
#: available: Random Forest + XGBoost soft voting. Its fold behaviour is
#: identical to deployment because `fit_candidate` fits the same
#: `EnsembleClassifier` with the same provenance arguments.
ENSEMBLE_CANDIDATE = RF_XGB_SOFT_VOTING

#: The XGBoost-missing artifact (`FocusClassifier`) is *not* the evaluated
#: `random_forest` candidate: the candidate adds `class_weight="balanced"` and is
#: compared through the candidate pipeline, while `FocusClassifier` is a bare
#: RandomForest behind a scaler. No evaluated candidate reproduces it, so it is
#: unconfirmable and may never be activated.
FALLBACK_CLASSIFIER_NAME = "FocusClassifier"

#: Candidates the current pipeline can actually publish (see above).
_PUBLISHABLE_CANDIDATES: frozenset[str] = frozenset({ENSEMBLE_CANDIDATE})

#: Human-readable reasons, kept stable so tests and manifests can assert them.
REASON_NOT_EVALUATED = "评估未给出选中候选，无法确认拟发布模型与评估一致"
REASON_UNCONFIRMED_ARTIFACT = "无法确认实际分类器类型，拒绝移动 active 指针"
REASON_MISMATCH = "评估选中候选 {selected} 与实际分类器 {deployed} 不一致"
REASON_XGB_FALLBACK = (
    "训练时 XGBoost 不可用，已回退为 RF-only 分类器；评估候选 {selected} 无法发布"
)
REASON_RF_ONLY_ARTIFACT = (
    "实际分类器为 RF-only（FocusClassifier），没有与之对应的已评估候选，禁止激活"
)
REASON_NOT_PUBLISHABLE = (
    "评估选中候选 {selected} 当前没有对应的可训练制品；本轮不扩展为发布所有候选"
)
REASON_ALLOWED = "评估候选与实际分类器一致（{candidate}）"


@dataclass(frozen=True)
class PublicationDecision:
    """Outcome of the evaluation-artifact consistency check.

    Attributes:
        allowed: True only when activation may move the active pointer.
        evaluation_candidate: Candidate selected by the evaluation, or None.
        deployed_candidate: Candidate name derived from the actual classifier.
        reason: Why activation is allowed or blocked (never empty).
    """

    allowed: bool
    evaluation_candidate: str | None
    deployed_candidate: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "evaluation_candidate": self.evaluation_candidate,
            "deployed_candidate": self.deployed_candidate,
            "reason": self.reason,
        }


def selected_candidate(evaluation: Mapping[str, Any] | None) -> str | None:
    """Return the candidate the evaluation selected, or None when unknown."""
    if not isinstance(evaluation, Mapping):
        return None
    selection = evaluation.get("candidate_selection")
    if not isinstance(selection, Mapping):
        return None
    selected = selection.get("selected")
    return str(selected) if selected else None


def deployed_candidate_name(classifier: object) -> str | None:
    """Map a classifier instance to the candidate name it implements.

    Only the ensemble maps to a candidate, because only the ensemble is fitted
    by the evaluation with the same configuration deployment uses. The RF-only
    fallback and anything unknown return None: "not confirmable" is the honest
    answer, and an unconfirmable artifact may not be activated.
    """
    if classifier is None:
        return None
    if type(classifier).__name__ == "EnsembleClassifier":
        return ENSEMBLE_CANDIDATE
    return None


def evaluate_publication(
    evaluation: Mapping[str, Any] | None,
    classifier: object,
    *,
    xgb_fallback: bool = False,
) -> PublicationDecision:
    """Decide whether the evaluated candidate and the artifact agree.

    Args:
        evaluation: The `evaluate_v2_candidates` report.
        classifier: The classifier instance about to be saved.
        xgb_fallback: True when the caller asked for the ensemble but training
            degraded to RF-only, so the block names the XGBoost fallback as the
            cause instead of reporting a bare mismatch.

    Returns:
        A :class:`PublicationDecision`; ``allowed`` is True only for an exact,
        confirmed match between the selected and deployed candidates.
    """
    selected = selected_candidate(evaluation)
    deployed = deployed_candidate_name(classifier)

    if selected is None:
        return PublicationDecision(False, None, deployed, REASON_NOT_EVALUATED)
    if deployed is None:
        reason = (
            REASON_XGB_FALLBACK.format(selected=selected)
            if xgb_fallback
            else (
                REASON_RF_ONLY_ARTIFACT
                if type(classifier).__name__ == FALLBACK_CLASSIFIER_NAME
                else REASON_UNCONFIRMED_ARTIFACT
            )
        )
        return PublicationDecision(False, selected, None, reason)
    if selected != deployed:
        reason = (
            REASON_XGB_FALLBACK.format(selected=selected)
            if xgb_fallback
            else REASON_MISMATCH.format(selected=selected, deployed=deployed)
        )
        return PublicationDecision(False, selected, deployed, reason)
    if selected not in _PUBLISHABLE_CANDIDATES:  # pragma: no cover - defensive
        return PublicationDecision(
            False, selected, deployed, REASON_NOT_PUBLISHABLE.format(selected=selected),
        )
    return PublicationDecision(
        True, selected, deployed, REASON_ALLOWED.format(candidate=selected),
    )


def candidate_is_publishable(candidate: str | None) -> bool:
    """True when the current pipeline can train an artifact for *candidate*."""
    return candidate in _PUBLISHABLE_CANDIDATES


__all__ = [
    "ENSEMBLE_CANDIDATE",
    "FALLBACK_CLASSIFIER_NAME",
    "LOGISTIC_REGRESSION",
    "REASON_ALLOWED",
    "REASON_MISMATCH",
    "REASON_NOT_EVALUATED",
    "REASON_NOT_PUBLISHABLE",
    "REASON_RF_ONLY_ARTIFACT",
    "REASON_UNCONFIRMED_ARTIFACT",
    "REASON_XGB_FALLBACK",
    "RULE_ENGINE",
    "XGBOOST",
    "PublicationDecision",
    "candidate_is_publishable",
    "deployed_candidate_name",
    "evaluate_publication",
    "selected_candidate",
]
