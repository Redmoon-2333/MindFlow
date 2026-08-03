"""Regression tests for the reliability improvements.

Covers PanelGraph topology, moderator metadata round-trip, and the
training-report/manifest consistency contract.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mindflow.train.pipeline import run_training
from mindflow.train.v2 import V2_FEATURE_NAMES


class _FakeGateway:
    async def complete(self, system: str, user: str, model: str = "chat") -> str:
        return "{}"

    async def close(self) -> None:
        return None


def test_panel_graph_has_no_dead_fanout_and_schema_node() -> None:
    from mindflow.graph.panel_graph import PanelGraph

    graph = PanelGraph(_FakeGateway()).build()
    assert "attribution_call" not in graph.nodes
    assert "verdict_schema_validation" in graph.nodes
    assert "human_review_interrupt" in graph.nodes


def test_moderator_metadata_round_trip() -> None:
    from mindflow.agents.orchestrator import _parse_verdict

    raw = (
        '{"types":["impulsivity"],"confidence":{"impulsivity":0.7},'
        '"recommended_technique":"stimulus_control","rationale":"x",'
        '"dissent":[],"insufficient_data":true,"uncertainty":0.6,'
        '"evidence_gaps":["no_mouse_data"]}'
    )
    verdict = _parse_verdict(raw)
    assert verdict is not None
    assert verdict["types"] == ["impulsivity"]
    assert verdict["insufficient_data"] is True
    assert verdict["uncertainty"] == 0.6
    assert verdict["evidence_gaps"] == ["no_mouse_data"]


def _feature_window(start: datetime, *, is_focus: bool) -> dict[str, object]:
    features = {name: 0.0 for name in V2_FEATURE_NAMES}
    features.update({
        "idle_ratio": 0.02 if is_focus else 0.55,
        "longest_segment_ratio": 0.95 if is_focus else 0.1,
        "top_app_ratio": 0.95 if is_focus else 0.2,
        "input_active_ratio": 0.7 if is_focus else 0.05,
        "app_switch_count": 0 if is_focus else 8,
        "domain_switch_count": 0 if is_focus else 5,
    })
    return {
        "window_start_utc": start.isoformat(),
        "window_end_utc": (start + timedelta(minutes=5)).isoformat(),
        "feature_schema_version": 3,
        "features": features,
    }


def test_v2_training_report_matches_manifest(tmp_path: Path) -> None:
    start = datetime(2026, 7, 1, 9, tzinfo=UTC)
    windows: list[dict[str, object]] = []
    feedback: list[dict[str, object]] = []
    for day in range(6):
        for is_focus in (True, False):
            for offset in range(3):
                session_start = start + timedelta(
                    days=day, hours=2 if is_focus else 4, minutes=5 * offset
                )
                windows.append(_feature_window(session_start, is_focus=is_focus))
                feedback.append({
                    "session_id": f"session-{day}-{int(is_focus)}-{offset}",
                    "start_time": session_start.isoformat(),
                    "end_time": (session_start + timedelta(minutes=5)).isoformat(),
                    "label": "focus" if is_focus else "distracted",
                    "score": 5 if is_focus else 1,
                    "task_type": "coding",
                })

    report = run_training(
        source="db",
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        feature_windows=windows,
        feedback_sessions=feedback,
    )

    assert report.classifier["grouped_evaluation"]["status"] == "evaluated"
    assert report.version_tag is not None
    manifest_path = tmp_path / "models" / "v2" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == report.version_tag
    assert manifest["feature_schema_version"] == report.feature_schema_version
    assert manifest["explicit_feedback_count"] == 36
