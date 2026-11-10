"""Smoke test for ``scripts/experiment_midterm_algorithms.py``.

Runs the whole midterm experiment suite in ``--quick`` mode and asserts the
deliverable contract: both artifacts exist, ``summary.json`` carries the five
experiment sections, and every number in it is finite.  The numerical
conclusions themselves are not asserted here — they are measurements, and a
test that pinned them would have to be rewritten every time the synthetic
generator changes.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experiment_midterm_algorithms.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "experiment_midterm_algorithms_under_test", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _walk_numbers(node: Any, path: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_walk_numbers(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk_numbers(value, f"{path}[{index}]"))
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        found.append((path, node))
    return found


def test_quick_run_writes_finite_summary(tmp_path: Path) -> None:
    script = _load_script()

    assert script.main(["--quick", "--out", str(tmp_path)]) == 0

    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"
    assert summary_path.is_file()
    assert report_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for section in ("e1", "e2", "e3", "e4", "e5"):
        assert section in summary, f"summary.json is missing {section}"

    numbers = _walk_numbers(summary)
    assert numbers, "summary.json carries no numbers at all"
    non_finite = [(path, value) for path, value in numbers if not math.isfinite(value)]
    assert not non_finite, f"non-finite numbers in summary.json: {non_finite[:5]}"

    report = report_path.read_text(encoding="utf-8")
    assert "数据来源与局限" in report
    assert "合成数据" in report
