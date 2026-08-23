"""End-to-end tests for the training pipeline.

Focuses on:
  - Synthetic data pipeline runs end-to-end without errors
  - TrainingReport has all expected fields
  - Baseline and model artifacts are saved
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mindflow.domain.events import make_event
from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.train import pipeline as pipeline_module
from mindflow.train.__main__ import main
from mindflow.train.pipeline import TrainingReport, run_training
from mindflow.train.user_profiles import list_archetype_ids


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
        assert report.feature_schema_version == FEATURE_SCHEMA_VERSION

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


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["mindflow-train", "--samples-per-hour", "6"], "--samples-per-hour"),
        (["mindflow-train", "--include-procrastination"], "--include-procrastination"),
    ],
)
def test_cli_rejects_removed_synthetic_controls(
    argv: list[str],
    flag: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        "mindflow.train.__main__.run_training",
        lambda **kwargs: TrainingReport(total_records=1),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize(
    ("num_users", "profiles", "expected"),
    [
        (2, None, list_archetype_ids()[:2]),
        (1, list_archetype_ids()[-2:], list_archetype_ids()[-2:]),
    ],
)
def test_synthetic_archetype_selection_respects_num_users_and_profiles(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_users: int,
    profiles: list[str] | None,
    expected: list[str],
) -> None:
    captured: dict[str, list[str] | None] = {}

    def fake_generate_v2_synthetic_data(
        *,
        archetype_ids: list[str] | None,
        days_per_archetype: int,
        seed: int,
        sample_explicit_ratio: float,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        captured["archetype_ids"] = archetype_ids
        return [], []

    monkeypatch.setattr(
        "mindflow.train.synthetic_v2.generate_v2_synthetic_data",
        fake_generate_v2_synthetic_data,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_v2_training",
        lambda **kwargs: TrainingReport(source=str(kwargs["source"])),
    )

    run_training(
        source="synthetic_v2",
        data_dir=work_dir / "data",
        models_dir=work_dir / "models",
        days=1,
        seed=42,
        num_users=num_users,
        user_profiles=profiles,
    )

    assert captured["archetype_ids"] == expected


@pytest.mark.parametrize(
    ("legacy_kwargs", "setting"),
    [
        ({"samples_per_hour": 6}, "samples_per_hour"),
        ({"include_procrastination": True}, "include_procrastination"),
    ],
)
def test_programmatic_legacy_synthetic_controls_warn_when_ignored(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_kwargs: dict[str, object],
    setting: str,
) -> None:
    monkeypatch.setattr(
        "mindflow.train.synthetic_v2.generate_v2_synthetic_data",
        lambda **kwargs: ([], []),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_run_v2_training",
        lambda **kwargs: TrainingReport(source=str(kwargs["source"])),
    )

    with pytest.warns(DeprecationWarning, match=setting):
        run_training(
            source="synthetic_v2",
            data_dir=work_dir / "data",
            models_dir=work_dir / "models",
            **legacy_kwargs,
        )


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
        calibration=None,  # toy data: skip post-hoc calibration
    )

    assert report.quality_gate["passed"] is True
    assert report.model_mode == "ready"
    assert report.activated is True
    assert (work_dir / "models" / "v2" / "latest.json").exists()
