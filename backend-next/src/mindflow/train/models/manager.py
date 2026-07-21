"""Central model management with versioned persistence.

F-phase note: the ``joblib.load`` calls below are intentionally left exactly
as they were in the original ``train/models.py`` — model-signing / HMAC
verification is a separate (F-phase) concern and must NOT be added here.
"""

from __future__ import annotations

import json
from contextlib import suppress
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

from mindflow.train.models.classifier import FocusClassifier
from mindflow.train.models.clustering import BehaviorClustering
from mindflow.train.models.hmm import BehaviorHMM
from mindflow.train.models.types import TrainingSummary


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

    def __init__(self, models_dir: str | Path = Path("data/models")) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.models_dir / "latest.json"
        self.clustering = BehaviorClustering()
        self.classifier = FocusClassifier()
        self.hmm = BehaviorHMM()

    @property
    def _today_tag(self) -> str:
        return date.today().strftime("%Y%m%d")

    # ── Training ──────────────────────────────────────────────────────────

    def train_all(
        self,
        features: npt.NDArray[Any],
        feature_names: list[str],
        labels: npt.NDArray[Any],
        sample_weight: npt.NDArray[Any] | None = None,
        min_confidence: float = 0.0,
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
            svw = (
                sw_high
                if sw_high is not None
                else np.ones(len(X_high))
            )
            X_train, X_test, y_train, y_test, sw_train, _ = train_test_split(
                X_high,
                y_high,
                svw,
                test_size=0.2,
                random_state=42,
                stratify=y_high,
            )
            self.classifier.fit(
                X_train,
                y_train,
                feature_names,
                sample_weight=sw_train if sample_weight is not None else None,
            )
            eval_metrics = self.classifier.evaluate(X_test, y_test)
            summary_classifier = {
                "accuracy": eval_metrics.get("accuracy", 0.0),
                "precision": eval_metrics.get("precision", 0.0),
                "recall": eval_metrics.get("recall", 0.0),
                "f1": eval_metrics.get("f1", 0.0),
                "cv_mean": eval_metrics.get("cv_mean", 0.0),
                "cv_std": eval_metrics.get("cv_std", 0.0),
                "feature_importance": self.classifier.get_feature_importance(),
                "high_confidence_samples": len(X_high),
                "filtered_low_confidence": low_conf_count,
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

        return TrainingSummary(
            clustering=summary_clustering,
            classifier=summary_classifier,
            hmm=summary_hmm,
        )

    def _build_state_sequences(self) -> list[npt.NDArray[Any]]:
        """Build state ID sequences from clustering labels (single sequence)."""
        if self.clustering.labels_ is None or len(self.clustering.labels_) < 2:
            return []
        return [self.clustering.labels_.astype(int)]

    # ── Versioned persistence ─────────────────────────────────────────────

    def save_all(self) -> dict[str, str]:
        """Save all models with date-stamped filenames and update latest.json.

        Returns:
            Dict mapping model names to their saved filenames.
        """
        tag = self._today_tag

        names: dict[str, str] = {
            "clustering": f"clustering-{tag}.pkl",
            "classifier": f"classifier-{tag}.pkl",
            "hmm": f"hmm-{tag}.pkl",
        }

        joblib.dump(self.clustering.to_dict(), str(self.models_dir / names["clustering"]))
        joblib.dump(self.classifier.to_dict(), str(self.models_dir / names["classifier"]))

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
        joblib.dump(hmm_data, str(self.models_dir / names["hmm"]))

        # Write / update latest pointer
        self._write_latest(names)

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
        for f in self.models_dir.glob("*.pkl"):
            parts = f.stem.split("-")
            if len(parts) >= 2 and len(parts[-1]) == 8 and parts[-1].isdigit():
                tags.add(parts[-1])

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
        """Load models from explicit filenames."""
        try:
            clustering_path = self.models_dir / name_map["clustering"]
            classifier_path = self.models_dir / name_map["classifier"]
            hmm_path = self.models_dir / name_map["hmm"]

            if not all(p.exists() for p in [clustering_path, classifier_path, hmm_path]):
                return False

            self.clustering = BehaviorClustering.from_dict(
                joblib.load(str(clustering_path))
            )
            self.classifier = FocusClassifier.from_dict(
                joblib.load(str(classifier_path))
            )

            hmm_data: dict[str, Any] = joblib.load(str(hmm_path))
            self.hmm = BehaviorHMM(n_states=int(hmm_data.get("n_states", 5)))
            tm = hmm_data.get("transition_matrix")
            self.hmm.transition_matrix = (
                np.array(tm) if tm is not None else None
            )
            self.hmm._is_fitted = bool(hmm_data.get("is_fitted", False))
            return True

        except (FileNotFoundError, EOFError, KeyError, ValueError):
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
        except (json.JSONDecodeError, OSError):
            pass
        return None
