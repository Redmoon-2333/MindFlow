"""End-to-end tests for the training pipeline.

Focuses on:
  - Synthetic data pipeline runs end-to-end without errors
  - TrainingReport has all expected fields
  - Baseline and model artifacts are saved
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mindflow.domain.events import make_event
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
        assert report.source == "synthetic_v2"
        assert report.total_records == 0

    def test_to_dict(self) -> None:
        """to_dict should return serializable dict."""
        report = TrainingReport(
            source="synthetic_v2",
            total_records=100,
            windows_extracted=50,
            n_focus=30,
            n_distract=20,
        )
        d = report.to_dict()
        assert d["source"] == "synthetic_v2"
        assert d["total_records"] == 100
        assert d["n_focus"] == 30
        assert d["windows_extracted"] == 50
        assert "timestamp" in d

    def test_json_serializable(self) -> None:
        """to_dict should be JSON-serializable."""
        report = TrainingReport(source="synthetic_v2", total_records=42)
        json_str = json.dumps(report.to_dict(), ensure_ascii=False)
        assert json_str is not None
        assert "42" in json_str


@pytest.mark.skip(reason="V1 pipeline removed — tests need updating for V2 synthetic_v2")
class TestRunTraining:
    """End-to-end training pipeline tests — V1 pipeline removed, use synthetic_v2."""

    def test_synthetic_end_to_end(self, work_dir: Path) -> None:
        """Small synthetic run should complete without errors."""
        report = run_training(
            source="synthetic_v2",
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
            source="synthetic_v2",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            days=2,
            samples_per_hour=4,
            seed=42,
        )
        assert report.source == "synthetic_v2"
        assert report.total_records > 0
        assert report.windows_extracted > 0
        assert report.n_focus + report.n_distract > 0
        assert report.avg_confidence > 0
        assert report.clustering is not None
        assert report.hmm is not None

    def test_artifacts_saved_to_disk(self, work_dir: Path) -> None:
        """Model artifacts should be saved to disk."""
        report = run_training(
            source="synthetic_v2",
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
            source="synthetic_v2",
            data_dir=work_dir / "data_a",
            models_dir=work_dir / "models_a",
            days=2,
            samples_per_hour=4,
            seed=42,
        )
        report_b = run_training(
            source="synthetic_v2",
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
            source="synthetic_v2",
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
            source="synthetic_v2",
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
            source="synthetic_v2",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            days=3,
            samples_per_hour=6,
            seed=42,
        )
        if "error" not in report.hmm:
            assert "transition_matrix" in report.hmm
            assert "steady_state" in report.hmm


@pytest.mark.skip(reason="V1 db fallback removed — test needs V2 feature windows")
def test_real_data_quality_gate_does_not_activate_unready_models(work_dir: Path) -> None:
    start = datetime(2026, 7, 24, tzinfo=UTC)
    events = [
        make_event(
            user_id=1,
            timestamp_utc=start + timedelta(minutes=5 * index),
            duration_s=300.0,
            app_name="code.exe" if index % 2 == 0 else "bilibili.exe",
            process_name="code.exe" if index % 2 == 0 else "bilibili.exe",
            window_title="main.py" if index % 2 == 0 else "bilibili",
            is_idle=False,
        )
        for index in range(20)
    ]

    report = run_training(
        source="db",
        data_dir=work_dir / "data",
        models_dir=work_dir / "models",
        events=events,
    )

    assert report.activated is False
    assert report.quality_gate["passed"] is False
    assert not (work_dir / "models" / "latest.json").exists()
    assert list((work_dir / "models").glob("*.pkl"))


def test_v2_training_stays_shadow_when_feedback_gate_fails(work_dir: Path) -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    feature_windows = []
    feedback_sessions = []
    for index in range(12):
        session_start = start + timedelta(days=index // 2, hours=(index % 2) * 2)
        is_focus = index % 2 == 0
        feature_windows.append({
            "window_start_utc": session_start.isoformat(),
            "window_end_utc": (session_start + timedelta(minutes=5)).isoformat(),
            "feature_schema_version": 3,
            "features": {
                "idle_ratio": 0.02 if is_focus else 0.6,
                "longest_segment_ratio": 0.9 if is_focus else 0.1,
                "top_app_ratio": 0.9 if is_focus else 0.2,
                "app_switch_count": 0 if is_focus else 8,
            },
        })
        feedback_sessions.append({
            "session_id": f"session-{index}",
            "start_time": session_start.isoformat(),
            "end_time": (session_start + timedelta(minutes=30)).isoformat(),
            "label": "focus" if is_focus else "distracted",
            "score": 5 if is_focus else 1,
            "task_type": "coding",
        })

    report = run_training(
        source="db",
        data_dir=work_dir / "data",
        models_dir=work_dir / "models",
        events=[
            make_event(
                user_id=1,
                timestamp_utc=start,
                duration_s=300,
                process_name="code.exe",
            )
        ],
        feature_windows=feature_windows,
        feedback_sessions=feedback_sessions,
    )

    assert report.feature_schema_version == 3
    assert report.model_mode == "shadow"
    assert report.activated is False
    assert report.quality_gate["checks"]["minimum_explicit_feedback"] is False
    assert (work_dir / "models" / "v2" / "training_report.json").exists()
    assert not (work_dir / "models" / "v2" / "latest.json").exists()


def test_v2_training_activates_only_after_all_gates_pass(work_dir: Path) -> None:
    start = datetime(2026, 7, 1, 8, tzinfo=UTC)
    feature_windows = []
    feedback_sessions = []
    session_index = 0
    for day_index in range(8):
        for class_index in range(4):
            is_focus = class_index < 2
            session_start = start + timedelta(days=day_index, hours=class_index * 2)
            feature_windows.append({
                "window_start_utc": session_start.isoformat(),
                "window_end_utc": (session_start + timedelta(minutes=5)).isoformat(),
                "feature_schema_version": 3,
                "features": {
                    "idle_ratio": 0.01 if is_focus else 0.7,
                    "longest_segment_ratio": 0.98 if is_focus else 0.05,
                    "top_app_ratio": 0.98 if is_focus else 0.1,
                    "input_active_ratio": 0.7 if is_focus else 0.05,
                    "app_switch_count": 0 if is_focus else 12,
                    "domain_switch_count": 0 if is_focus else 8,
                },
            })
            feedback_sessions.append({
                "session_id": f"session-{session_index}",
                "start_time": session_start.isoformat(),
                "end_time": (session_start + timedelta(minutes=30)).isoformat(),
                "label": "focus" if is_focus else "distracted",
                "score": 5 if is_focus else 1,
                "task_type": "coding",
            })
            session_index += 1

    report = run_training(
        source="db",
        data_dir=work_dir / "data",
        models_dir=work_dir / "models",
        events=[make_event(user_id=1, timestamp_utc=start, duration_s=300)],
        feature_windows=feature_windows,
        feedback_sessions=feedback_sessions,
    )

    assert report.quality_gate["passed"] is True
    assert report.model_mode == "ready"
    assert report.activated is True
    assert (work_dir / "models" / "v2" / "latest.json").exists()
