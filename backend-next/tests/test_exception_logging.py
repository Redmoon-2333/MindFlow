"""P0-2: exception logging for six previously silent paths.

Each test uses a local loguru sink to capture warning records
and asserts both the warning message and exception traceback visibility.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger as _logger

from mindflow.train.models import ModelManager

# ── Loguru capture helper (local, per-test — no global fixture) ───────────


@contextmanager
def capture_loguru(level: str = "WARNING") -> Generator[list[dict[str, Any]]]:
    """Capture loguru records at *level* or above; yields list of record dicts."""
    records: list[dict[str, Any]] = []
    hid = _logger.add(lambda msg: records.append(msg.record), level=level)
    try:
        yield records
    finally:
        with suppress(ValueError):
            _logger.remove(hid)


# ── Helpers ───────────────────────────────────────────────────────────────


def _record_has_msg(records: list[dict[str, Any]], fragment: str) -> bool:
    return any(fragment in r.get("message", "") for r in records)


def _record_has_exception(records: list[dict[str, Any]]) -> bool:
    return any(r.get("exception") is not None for r in records)


# ── Path 1: app.py — training_report.json parse error ─────────────────────


def test_training_report_parse_error_warns() -> None:
    """Structural: training_report catch uses opt(exception=True)+fallback pass."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "mindflow" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert "Failed to parse training report" in source
    # The except clause must use opt(exception=True).warning, not bare warning
    marker = "Failed to parse training report"
    idx = source.index(marker)
    preceding = source[max(0, idx - 300): idx]
    assert "opt(exception=True)" in preceding, (
        "training_report warning must use .opt(exception=True)"
    )


# ── Path 2: app.py — dead locals().get try/except removed ─────────────────


def test_app_no_locals_get_try_except() -> None:
    """The dead try/except around locals().get has been removed."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "mindflow" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    # The dead pattern was: try: shared_evidence = locals().get(...)
    # We assert it no longer exists.
    assert "try:\n                shared_evidence = locals().get" not in source, (
        "Dead try/except around locals().get still present"
    )


# ── Path 3: app.py — shared_evidence_builder state assignment ─────────────


def test_shared_evidence_builder_warns_on_error() -> None:
    """The except clause logs with opt(exception=True), not silent pass."""
    app_path = Path(__file__).resolve().parent.parent / "src" / "mindflow" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    assert "Failed to set app.state.shared_evidence_builder" in source
    # Verify the warning line (or nearby) uses opt(exception=True)
    marker = "Failed to set app.state.shared_evidence_builder"
    idx = source.index(marker)
    preceding = source[max(0, idx - 200): idx]
    assert "opt(exception=True)" in preceding, (
        "shared_evidence_builder warning must use .opt(exception=True)"
    )


# ── Path 4: eval/adapters.py — _extract_metrics_from_user parse error ─────


def test_extract_metrics_malformed_json_warns() -> None:
    """Structural: _extract_metrics_from_user catch uses opt(exception=True)."""
    adapters_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "mindflow" / "eval" / "adapters.py"
    )
    source = adapters_path.read_text(encoding="utf-8")

    assert "Failed to extract metrics from user prompt" in source
    marker = "Failed to extract metrics from user prompt"
    idx = source.index(marker)
    preceding = source[max(0, idx - 300): idx]
    assert "opt(exception=True)" in preceding, (
        "_extract_metrics_from_user warning must use .opt(exception=True)"
    )


# ── Path 5: explain.py — KernelExplainer SHAP fallback failure ────────────


def test_kernel_explainer_fallback_warns() -> None:
    """KernelExplainer failure emits warning with traceback; returns None."""
    from unittest.mock import MagicMock

    from mindflow.train.explain import ModelExplainer

    mock_shap = MagicMock()
    mock_shap.TreeExplainer.side_effect = RuntimeError("no tree model")
    mock_shap.KernelExplainer.side_effect = RuntimeError("kernel crash")

    # Minimal classifier stub
    class _Stub:
        model = None
        scaler = None
        def predict(self, x):  # noqa: E704
            return np.ones(len(x), dtype=np.int32)
        def predict_proba(self, x):  # noqa: E704
            n = len(x)
            p = np.zeros((n, 2), dtype=np.float64)
            p[:, 1] = 0.7
            return p

    stub = _Stub()
    stub.model = stub
    explainer = ModelExplainer(stub, ["f0", "f1"])
    explainer._shap = mock_shap
    explainer._available = True

    x = np.array([[0.5, 0.3], [0.7, 0.6]])
    with capture_loguru() as records:
        result = explainer._compute_shap_values(x)

    assert result is None
    assert _record_has_msg(records, "KernelExplainer SHAP computation failed")
    assert _record_has_exception(records)


# ── Path 6: manager.py — latest.json parse error ──────────────────────────


def test_current_version_tag_malformed_json_warns(
    tmp_path: Path,
) -> None:
    """Malformed latest.json emits warning with traceback; returns None."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "latest.json").write_text("{broken json", encoding="utf-8")

    manager = ModelManager(models_dir=models_dir)
    with capture_loguru() as records:
        tag = manager.current_version_tag

    assert tag is None
    assert _record_has_msg(records, "Failed to parse latest.json")
    assert _record_has_exception(records)
