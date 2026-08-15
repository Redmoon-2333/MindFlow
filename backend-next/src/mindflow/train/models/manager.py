"""Central model management with versioned persistence.

F-phase: every saved ``.pkl`` is HMAC-signed (see ``train/serialization.py``)
and every load verifies the signature first. ``models_dir`` is a
``platformdirs`` user-writable directory — without this, a malicious local
process could drop a crafted pickle there and achieve code execution the
next time the app calls ``joblib.load``.
"""

from __future__ import annotations

import json
import re
import secrets
import warnings
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from loguru import logger
from sklearn.exceptions import InconsistentVersionWarning

from mindflow.train.models.classifier import FocusClassifier
from mindflow.train.models.clustering import BehaviorClustering
from mindflow.train.models.ensemble import _XGB_CLASS_MARKER, EnsembleClassifier
from mindflow.train.models.hmm import BehaviorHMM
from mindflow.train.models.types import TrainingSummary
from mindflow.train.serialization import (
    _load_or_create_signing_key,
    sign_model_file,
    verify_model_file,
)


class ModelSignatureError(Exception):
    """Raised when a model file's HMAC signature is missing or invalid.

    Deliberately not routed through ``mindflow.errors.MindFlowError``:
    ``train/`` is an offline CLI, not wired into the running app (see
    ``pyproject.toml`` — ``train`` module docstring), so there is no API
    boundary handler that needs to catch this by a shared root. Keeping it
    a plain ``Exception`` subclass avoids implying a relationship that
    does not exist.
    """



class ModelManager:
    """Central model management with versioned persistence.

    Solves the old backend's P1 technical debt: models are saved with a
    date-stamped filename (``{name}-{YYYYMMDD}.pkl``) and ``latest.json``
    tracks the current active version, enabling rollback by simply updating
    the pointer file.

    Directory layout::

        models/
        +- latest.json           # {"clustering": "clustering-20260717.pkl", ...}
        +- clustering-20260717.pkl
        +- classifier-20260717.pkl
        +- hmm-20260717.pkl
    """

    def __init__(
        self,
        models_dir: str | Path = Path("data/models"),
        use_ensemble: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.models_dir / "latest.json"
        self.clustering = BehaviorClustering()
        self.hmm = BehaviorHMM()

        self._use_ensemble: bool = False
        self.classifier: FocusClassifier | EnsembleClassifier = FocusClassifier()

        if use_ensemble:
            try:
                from xgboost import XGBClassifier  # noqa: F401 — probe availability

                self.classifier = EnsembleClassifier()
                self._use_ensemble = True
            except ImportError:
                logger.warning(
                    "use_ensemble=True but xgboost not installed; "
                    "falling back to RF-only FocusClassifier"
                )
                self.classifier = FocusClassifier()
                self._use_ensemble = False

    @property
    def _today_tag(self) -> str:
        return datetime.now(UTC).strftime("%Y%m%d")

    @property
    def _new_version_tag(self) -> str:
        """Timestamp + short random suffix so same-day runs never overwrite."""
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{secrets.token_hex(3)}"

    # ── Training ──────────────────────────────────────────────────────────

    def train_all(
        self,
        features: npt.NDArray[Any],
        feature_names: list[str],
        labels: npt.NDArray[Any],
        sample_weight: npt.NDArray[Any] | None = None,
        min_confidence: float = 0.0,
        use_explainer: bool = False,
    ) -> TrainingSummary:
        """Train all models and return summary.

        Args:
            features: Feature matrix ``(n_samples, n_features)``.
            feature_names: Names for each feature column.
            labels: Binary labels (1=focus, 0=distraction).
            sample_weight: Per-sample confidence weights.
            min_confidence: Filter samples below this confidence.

        Returns:
            ``TrainingSummary`` with clustering, classifier, hmm subsections.
        """
        summary_clustering: dict[str, Any] = {}
        summary_classifier: dict[str, Any] = {}
        summary_hmm: dict[str, Any] = {}

        # ── Clustering ──
        cluster_info = self.clustering.fit(features)
        summary_clustering = {
            "n_clusters": len([c for c in cluster_info if c.cluster_id != -1]),
            "noise_points": sum(1 for c in cluster_info if c.cluster_id == -1),
            "clusters": [
                {
                    "id": c.cluster_id,
                    "label": c.label,
                    "count": c.sample_count,
                    "avg_focus_score": c.avg_focus_score,
                }
                for c in cluster_info
            ],
        }

        # ── Classifier ──
        high_conf_mask = (
            np.ones(len(features), dtype=bool)
            if sample_weight is None
            else sample_weight >= min_confidence
        )
        X_high = features[high_conf_mask]
        y_high = labels[high_conf_mask]
        sw_high = (
            None
            if sample_weight is None
            else sample_weight[high_conf_mask]
        )
        low_conf_count = int((~high_conf_mask).sum())

        if len(np.unique(y_high)) >= 2 and len(X_high) >= 10:
            self.classifier.fit(
                X_high,
                y_high,
                feature_names,
                sample_weight=sw_high if sample_weight is not None else None,
            )
            summary_classifier = {
                "feature_importance": self.classifier.get_feature_importance(),
                "high_confidence_samples": int(len(X_high)),
                "filtered_low_confidence": low_conf_count,
                "n_samples": int(len(X_high)),
                "n_classes": int(len(np.unique(y_high))),
            }
        else:
            summary_classifier = {
                "error": "Not enough data for supervised training",
                "n_samples": len(X_high),
                "n_classes": int(len(np.unique(y_high))),
                "filtered_low_confidence": low_conf_count,
            }

        # ── HMM ──
        sequences = self._build_state_sequences()
        if sequences:
            self.hmm.fit(sequences)
            tm = self.hmm.get_transition_matrix()
            steady = self.hmm.get_steady_state()
            summary_hmm = {
                "transition_matrix": [
                    [round(float(v), 4) for v in row] for row in tm
                ],
                "steady_state": [round(float(v), 4) for v in steady],
                "state_names": list(self.hmm.state_names),
            }
        else:
            summary_hmm = {"error": "No valid state sequences for HMM training"}

        # ── Explanation ──
        explanation: dict[str, Any] = {}
        if use_explainer and self.classifier._is_fitted and len(X_high) > 0:
            try:
                from mindflow.train.explain import ModelExplainer

                expl = ModelExplainer(self.classifier, feature_names)
                explanation = expl.explain(X_high)
            except Exception:
                logger.warning(
                    "Model explanation failed, continuing without it"
                )

        return TrainingSummary(
            clustering=summary_clustering,
            classifier=summary_classifier,
            hmm=summary_hmm,
            explanation=explanation,
        )

    def _build_state_sequences(self) -> list[npt.NDArray[Any]]:
        """Build state ID sequences from clustering labels (single sequence)."""
        if self.clustering.labels_ is None or len(self.clustering.labels_) < 2:
            return []
        return [self.clustering.labels_.astype(int)]

    # ── Versioned persistence ─────────────────────────────────────────────

    def save_all(
        self,
        *,
        activate: bool = True,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Save all models with versioned filenames and a manifest.

        Returns:
            Dict mapping model names to their saved filenames.
        """
        tag = self._new_version_tag

        names: dict[str, str] = {
            "clustering": f"clustering-{tag}.pkl",
            "classifier": f"classifier-{tag}.pkl",
            "hmm": f"hmm-{tag}.pkl",
        }

        clustering_path = self.models_dir / names["clustering"]
        classifier_path = self.models_dir / names["classifier"]
        hmm_path = self.models_dir / names["hmm"]

        joblib.dump(self.clustering.to_dict(), str(clustering_path))
        joblib.dump(self.classifier.to_dict(), str(classifier_path))

        hmm_data: dict[str, Any] = {
            "transition_matrix": (
                self.hmm.transition_matrix.tolist()
                if self.hmm.transition_matrix is not None
                else None
            ),
            "state_names": list(self.hmm.state_names),
            "n_states": self.hmm.n_states,
            "is_fitted": self.hmm._is_fitted,
        }
        joblib.dump(hmm_data, str(hmm_path))

        # Sign every artifact so _load_versions can detect tampering/planted
        # pickles before ever calling joblib.load on them.
        signing_key = _load_or_create_signing_key(self.models_dir)
        for path in (clustering_path, classifier_path, hmm_path):
            sign_model_file(path, signing_key)

        manifest_data: dict[str, Any] = {
            "version": tag,
            "created_at": datetime.now(UTC).isoformat(),
            "files": names,
        }
        manifest_data.update(manifest or {})
        (self.models_dir / "manifest.json").write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if activate:
            self._write_latest(names)
            # Keep the model directory bounded: drop artifacts older than the
            # newest N versions while never touching the active version.
            self._prune_old_versions(
                keep=json.loads(
                    self.latest_path.read_text(encoding="utf-8")
                ) if self.latest_path.exists() else None
            )

        return names

    def _write_latest(self, names: dict[str, str]) -> None:
        """Write latest.json pointer file."""
        existing: dict[str, str] = {}
        if self.latest_path.exists():
            with suppress(json.JSONDecodeError, OSError):
                existing = json.loads(self.latest_path.read_text(encoding="utf-8"))
        existing.update(names)
        self.latest_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Version retention (audit report — model versions never cleaned) ──

    _MAX_KEPT_VERSIONS: int = 5
    """Number of most-recent model versions to keep after each save.

    Each training run writes 6 files (3 .pkl + 3 .pkl.hmac); without
    retention the models dir grows unboundedly (observed 666 .pkl files).
    The currently activated version is always retained even if older.
    """

    def _prune_old_versions(self, keep: dict[str, str] | None = None) -> None:
        """Delete old model artifacts beyond the retention window.

        Args:
            keep: Filenames to never delete (defaults to the current
                ``latest.json`` pointers — the active version).
        """
        protected = set((keep or {}).values())
        # Group artifacts by version tag parsed from filenames:
        #   classifier-20260806_095623_1e5fcd.pkl(.hmac)
        versions: dict[str, list[Path]] = {}
        for path in self.models_dir.glob("*.pkl*"):
            stem = path.name
            # strip .pkl / .pkl.hmac suffix
            base = stem.removesuffix(".hmac").removesuffix(".pkl")
            if "-" not in base:
                continue
            tag = base.split("-", 1)[1]
            versions.setdefault(tag, []).append(path)

        sorted_tags = sorted(versions.keys(), reverse=True)
        for tag in sorted_tags[self._MAX_KEPT_VERSIONS:]:
            for path in versions[tag]:
                if path.name in protected:
                    continue
                with suppress(OSError):
                    path.unlink()
                    logger.debug("Pruned old model artifact {}", path.name)

    def readiness_status(self) -> dict[str, Any]:
        reasons: list[str] = []
        if not bool(getattr(self.classifier, "_is_fitted", False)):
            reasons.append("classifier_not_fitted")
        if self.clustering.model is None:
            reasons.append("clustering_not_fitted")
        if not bool(getattr(self.hmm, "_is_fitted", False)):
            reasons.append("hmm_not_fitted")
        return {"ready": not reasons, "reasons": reasons}

    def unload(self) -> None:
        """Invalidate every loaded model while preserving object identity.

        Privacy wipes can leave the manager reachable through application
        state and other long-lived references. Resetting the current model
        objects in place ensures those references become unfitted too; merely
        assigning new objects here would leave the old fitted estimators usable.
        Persisted artifacts are intentionally untouched by this method.
        """
        self._reset_state_in_place(
            self.classifier,
            type(self.classifier)(),
        )
        self._reset_state_in_place(
            self.clustering,
            BehaviorClustering(method=self.clustering.method),
        )
        self._reset_state_in_place(
            self.hmm,
            BehaviorHMM(n_states=self.hmm.n_states),
        )

    @staticmethod
    def _reset_state_in_place(current: object, fresh: object) -> None:
        current_state = vars(current)
        current_state.clear()
        current_state.update(vars(fresh))

    def load_latest(self) -> bool:
        """Load the latest version of all models. Returns True if successful."""
        if not self.latest_path.exists():
            return False

        try:
            pointer: dict[str, str] = json.loads(
                self.latest_path.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            return False

        required = ["clustering", "classifier", "hmm"]
        if not all(k in pointer for k in required):
            return False

        return self._load_versions(pointer)

    def load_version(self, tag: str) -> bool:
        """Load a specific dated version of all models.

        Args:
            tag: Date string in ``YYYYMMDD`` format (e.g. ``"20260717"``).

        Returns:
            True if all three model files for that tag exist and load.
        """
        names = {
            "clustering": f"clustering-{tag}.pkl",
            "classifier": f"classifier-{tag}.pkl",
            "hmm": f"hmm-{tag}.pkl",
        }
        return self._load_versions(names)

    def list_versions(self) -> list[str]:
        """List all available model version tags from filenames.

        Returns:
            Sorted list of ``YYYYMMDD`` tags for which all three model files
            exist, newest first.
        """
        tags: set[str] = set()
        for f in self.models_dir.glob("clustering-*.pkl"):
            match = re.match(r"^clustering-(.+?)\.pkl$", f.name)
            if match:
                tags.add(match.group(1))

        valid: list[str] = []
        for tag in sorted(tags, reverse=True):
            required_exists = all(
                (self.models_dir / f"{name}-{tag}.pkl").exists()
                for name in ["clustering", "classifier", "hmm"]
            )
            if required_exists:
                valid.append(tag)

        return valid

    def rollback(self, tag: str) -> bool:
        """Rollback to a specific version and update latest.json.

        Args:
            tag: Date string in ``YYYYMMDD`` format.

        Returns:
            True if rollback succeeded.
        """
        if not self.load_version(tag):
            return False

        names = {
            "clustering": f"clustering-{tag}.pkl",
            "classifier": f"classifier-{tag}.pkl",
            "hmm": f"hmm-{tag}.pkl",
        }
        self._write_latest(names)
        return True

    def _load_versions(self, name_map: dict[str, str]) -> bool:
        """Load models from explicit filenames.

        Verifies each file's HMAC signature before calling ``joblib.load``.
        Refuses (raises ``ModelSignatureError``) rather than silently
        skipping when a signature is missing or mismatched: this is
        pre-1.0 (``train/`` is an offline CLI, not wired into the running
        app — see module docs), so there are no legacy unsigned installs
        to stay backward-compatible with, and a mismatch means tampering,
        not "just missing". Silently falling back to loading anyway would
        defeat the entire point of signing.
        """
        try:
            clustering_path = self.models_dir / name_map["clustering"]
            classifier_path = self.models_dir / name_map["classifier"]
            hmm_path = self.models_dir / name_map["hmm"]

            if not all(p.exists() for p in [clustering_path, classifier_path, hmm_path]):
                return False

            signing_key = _load_or_create_signing_key(self.models_dir)
            for path in (clustering_path, classifier_path, hmm_path):
                if not verify_model_file(path, signing_key):
                    logger.critical(
                        "Model file {} failed HMAC verification — refusing to load "
                        "(missing or tampered .hmac sibling)",
                        path,
                    )
                    raise ModelSignatureError(f"Signature verification failed for {path}")

            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                clustering_data: dict[str, Any] = joblib.load(str(clustering_path))
                classifier_data: dict[str, Any] = joblib.load(str(classifier_path))
                hmm_data: dict[str, Any] = joblib.load(str(hmm_path))

            self.clustering = BehaviorClustering.from_dict(clustering_data)
            if classifier_data.get("__class__") == _XGB_CLASS_MARKER:
                self.classifier = EnsembleClassifier.from_dict(classifier_data)
            else:
                self.classifier = FocusClassifier.from_dict(classifier_data)

            self.hmm = BehaviorHMM(n_states=int(hmm_data.get("n_states", 5)))
            tm = hmm_data.get("transition_matrix")
            self.hmm.transition_matrix = (
                np.array(tm) if tm is not None else None
            )
            self.hmm._is_fitted = bool(hmm_data.get("is_fitted", False))
            return True

        except InconsistentVersionWarning as exc:
            logger.error(
                "Model artifact rejected: sklearn version mismatch ({}). "
                "Re-train with the current environment to produce a "
                "compatible artifact "
                "(`uv run python -m mindflow.train --source db`).",
                exc,
            )
            return False
        except (FileNotFoundError, EOFError, KeyError, TypeError, ValueError):
            return False

    @property
    def current_version_tag(self) -> str | None:
        """Return the current version tag from latest.json, or None."""
        if not self.latest_path.exists():
            return None
        try:
            pointer: dict[str, str] = json.loads(
                self.latest_path.read_text(encoding="utf-8")
            )
            tag = pointer.get("clustering", "")
            if tag.startswith("clustering-") and tag.endswith(".pkl"):
                return tag[len("clustering-"): -len(".pkl")]
        except (json.JSONDecodeError, OSError) as exc:
            logger.opt(exception=True).warning(
                "Failed to parse latest.json for version tag: {}", exc
            )
        return None
