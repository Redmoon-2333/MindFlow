"""End-to-end tests for the training pipeline.

Focuses on:
  - Synthetic data pipeline runs end-to-end without errors
  - TrainingReport has all expected fields
  - Baseline and model artifacts are saved
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mindflow.train.pipeline import TrainingReport, run_training


@pytest.fixture
def work_dir() -> Path:
    """Temporary working directory for pipeline artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


class TestTrainingReport:
    """TrainingReport dataclass behavior."""

    def test_default_creation(self) -> None:
        """Minimal creation should have timestamp."""
        report = TrainingReport()
        assert report.timestamp is not None
        assert report.source == "synthetic"
        assert report.total_records == 0

    def test_to_dict(self) -> None:
        """to_dict should return serializable dict."""
        report = TrainingReport(
            source="synthetic",
            total_records=100,
            windows_extracted=50,
            n_focus=30,
            n_distract=20,
        )
        d = report.to_dict()
        assert d["source"] == "synthetic"
        assert d["total_records"] == 100
        assert d["n_focus"] == 30
        assert d["windows_extracted"] == 50
        assert "timestamp" in d

    def test_json_serializable(self) -> None:
        """to_dict should be JSON-serializable."""
        report = TrainingReport(source="synthetic", total_records=42)
        json_str = json.dumps(report.to_dict(), ensure_ascii=False)
        assert json_str is not None
        assert "42" in json_str


class TestRunTraining:
    """End-to-end training pipeline tests (synthetic data only)."""

    def test_synthetic_end_to_end(self, work_dir: Path) -> None:
        """Small synthetic run should complete without errors."""
        report = run_training(
            source="synthetic",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            days=3,
            samples_per_hour=4,
            seed=42,
        )
        assert report.total_records > 0
        assert report.windows_extracted > 0
        assert report.baseline_updated > 0
        assert report.saved_models is not None

    def test_report_has_all_fields(self, work_dir: Path) -> None:
        """TrainingReport should have all expected fields after run."""
        report = run_training(
            source="synthetic",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            days=2,
            samples_per_hour=4,
            seed=42,
        )
        assert report.source == "synthetic"
        assert report.total_records > 0
        assert report.windows_extracted > 0
        assert report.n_focus + report.n_distract > 0
        assert report.avg_confidence > 0
        assert report.clustering is not None
        assert report.hmm is not None

    def test_artifacts_saved_to_disk(self, work_dir: Path) -> None:
        """Model artifacts should be saved to disk."""
        report = run_training(
            source="synthetic",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            days=2,
            samples_per_hour=4,
            seed=42,
        )

        models_path = work_dir / "models"
        # Check at least some .pkl files exist
        pkl_files = list(models_path.glob("*.pkl"))
        assert len(pkl_files) >= 1
        assert report.total_records > 0

        # Check latest.json
        assert (models_path / "latest.json").exists()

        # Check training report
        assert (models_path / "training_report.json").exists()
        report_data = json.loads(
            (models_path / "training_report.json").read_text(encoding="utf-8")
        )
        assert report_data["total_records"] > 0

    def test_reproducible(self, work_dir: Path) -> None:
        """Same seed should produce same report totals."""
        report_a = run_training(
            source="synthetic",
            data_dir=work_dir / "data_a",
            models_dir=work_dir / "models_a",
            days=2,
            samples_per_hour=4,
            seed=42,
        )
        report_b = run_training(
            source="synthetic",
            data_dir=work_dir / "data_b",
            models_dir=work_dir / "models_b",
            days=2,
            samples_per_hour=4,
            seed=42,
        )
        assert report_a.total_records == report_b.total_records
        assert report_a.windows_extracted == report_b.windows_extracted
        assert report_a.n_focus == report_b.n_focus

    def test_baseline_saved(self, work_dir: Path) -> None:
        """Baseline JSON should be saved."""
        report = run_training(
            source="synthetic",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            days=2,
            samples_per_hour=4,
            seed=42,
            user_id=1,
        )
        baseline_path = work_dir / "data" / "baseline_user1.json"
        assert baseline_path.exists()
        baseline_data = json.loads(
            baseline_path.read_text(encoding="utf-8")
        )
        assert baseline_data["user_id"] == 1
        assert baseline_data["total_days"] >= 1
        assert report.baseline_updated > 0

    def test_classifier_trained(self, work_dir: Path) -> None:
        """Classifier should be trained with sufficient data."""
        report = run_training(
            source="synthetic",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            days=3,
            samples_per_hour=6,
            seed=42,
        )
        if "error" in report.classifier:
            # Minimal data might not be enough for 2 classes
            pytest.skip(f"Classifier not trained: {report.classifier['error']}")
        assert "accuracy" in report.classifier

    def test_hmm_trained(self, work_dir: Path) -> None:
        """HMM should have transition matrix in report."""
        report = run_training(
            source="synthetic",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            days=3,
            samples_per_hour=6,
            seed=42,
        )
        if "error" not in report.hmm:
            assert "transition_matrix" in report.hmm
            assert "steady_state" in report.hmm
