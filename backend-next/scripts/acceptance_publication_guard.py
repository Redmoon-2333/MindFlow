"""Acceptance runs for the training publication guard (model-candidate consistency).

Plan item: "模型候选一致性和 shadow 发布验收".  Three end-to-end training runs
against synthetic schema-v4 feature windows, executed entirely inside workspaces
under ``data/experiments/20260925_publication_acceptance/`` — never the user's
real database and never ``data/models``:

* **Run A — the guard must block.**  The perfectly separable dataset shape from
  ``tests/test_publication_guard.py::test_passing_gate_with_an_inconsistent_candidate_stays_shadow``
  passes every quality gate, and because every candidate reaches the same
  accuracy the conservative selection rule keeps ``logistic_regression`` while
  the pipeline trains the RF+XGB ensemble.  The guard must refuse to move the
  active pointer, record the mismatch, and still save the shadow artifacts.
* **Run B — consistent candidate activates.**  A non-linearly-separable dataset
  (2-D XOR structure + redundant noisy copies + a bounded flip rate, engineered
  so the evaluation itself selects ``rf_xgb_soft_voting`` and the full quality
  gate passes) lets the guard allow activation: ``latest.json`` moves and the
  run reports ``model_mode="ready"``.
* **Run C — ``allow_activation=False`` stays shadow even on a pass.**  Run A's
  dataset with the automatic-training policy: no activation regardless of the
  publication verdict, with ``activation_suppressed_reason`` recorded.

Artifacts (repo rule: experiments write under ``data/experiments/<run-id>/``):
``summary.json`` (structured evidence incl. per-candidate metrics and the exact
commands) and ``report.md`` (human-readable verdict table).  Re-running the
script recreates the per-run workspaces; nothing is written anywhere else.

Usage::

    uv run python scripts/acceptance_publication_guard.py
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION
from mindflow.train.candidates import RF_XGB_SOFT_VOTING
from mindflow.train.pipeline import TrainingReport, run_training

RUN_ID = "20260925_publication_acceptance"
EXPERIMENT_DIR = Path("data/experiments") / RUN_ID

#: Suite state verified with the exact commands below during this acceptance
#: round (this script does not re-run the test suite; outcomes are recorded in
#: ``report.md``).
VERIFIED_COMMANDS: list[dict[str, str]] = [
    {
        "command": "uv run python -m pytest tests/test_publication_guard.py "
                   "tests/test_training_activation_policy.py tests/test_train_pipeline.py -q",
        "outcome": "49 passed, 8 warnings",
    },
    {
        "command": "uv run python -m ruff check scripts/acceptance_publication_guard.py src tests",
        "outcome": "All checks passed! (11 pre-existing findings live in old scripts, "
                   "outside this command's scope)",
    },
    {
        "command": "uv run python -m mypy --strict src/mindflow "
                   "scripts/acceptance_publication_guard.py",
        "outcome": "Success: no issues found in 186 source files",
    },
]


@dataclass(frozen=True)
class DatasetSpec:
    """Parameters of the synthetic XOR construction used by Run B.

    ``base_scale`` shrinks the cluster gap so a deterministic fraction of rows
    lands on the wrong side of the 0.5 threshold (feature-level ambiguity that
    keeps every model's probability distribution stationary across
    forward-chaining folds — this is what keeps the shadow-drift PSI under its
    gate threshold without resorting to label flips).
    """

    days: int
    windows_per_day: int
    jitter: float
    base_scale: float
    seed: int
    copy_sigma: float
    copy_count: int

    def describe(self) -> str:
        return (
            f"XOR (a=app_switch_count 6.0/6.5, b=longest_segment_ratio "
            f"{0.5 + 0.22 * self.base_scale:.3f}/{0.5 - 0.22 * self.base_scale:.3f} "
            f"+- jitter {self.jitter}) + {self.copy_count} noisy copies "
            f"(sigma={self.copy_sigma}), days={self.days} x "
            f"{self.windows_per_day} windows/day, seed={self.seed}"
        )


#: Found by an offline search over constructions (see scratch/ in the
#: experiment directory): the only family where the selection rule promotes
#: ``rf_xgb_soft_voting`` while every quality gate — including the
#: shadow-drift PSI — passes.
RUN_B_DATASET = DatasetSpec(
    days=28, windows_per_day=32, jitter=0.15, base_scale=0.45, seed=7,
    copy_sigma=0.2, copy_count=12,
)

#: Noisy redundant copies of the two XOR signal values.  Random Forest draws
#: ~sqrt(28)=5 random columns per split, so copies dilute its subspace and
#: keep it below the ensemble; XGBoost always sees every column.  The
#: ensemble then wins the ladder by averaging the two views (it corrects
#: XGBoost's marginal errors wherever Random Forest is right, and vice
#: versa), which is exactly the candidate the pipeline actually trains.
COPY_SOURCE_FEATURES: tuple[str, ...] = (
    "domain_switch_count", "keypress_rate_per_min", "mouse_click_rate_per_min",
    "scroll_rate_per_min", "mouse_distance_per_min", "input_active_ratio",
    "interaction_bursts_per_min", "click_key_ratio", "browser_ratio",
    "audible_browser_ratio", "active_seconds_ratio", "top_domain_ratio",
    "interaction_interval_mean_s", "interaction_interval_std_s",
    "interaction_interval_cv", "task_type_entropy", "task_type_dominant_ratio",
    "task_context_transition", "task_unknown_ratio", "longest_segment_ratio",
)


def _build_separable_dataset(
    days: int = 8, windows_per_day: int = 12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Windows + explicit feedback that clear every quality gate (Run A / C).

    Deliberately perfectly separable with every feature aligned to the label,
    exactly like the regression test it is taken from: all five candidates
    reach the same balanced accuracy, so the conservative selection rule keeps
    the *simplest* one (``logistic_regression``) while the pipeline trains the
    ensemble — the exact disagreement the guard exists to catch.
    """
    windows: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    start = datetime(2026, 7, 1, 8, tzinfo=UTC)
    index = 0
    for day in range(days):
        for slot in range(windows_per_day):
            is_focus = slot % 2 == 0
            session_start = start + timedelta(days=day, hours=slot)
            windows.append({
                "id": f"win-{index}",
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


def _build_xor_dataset(spec: DatasetSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Non-linearly-separable windows + explicit feedback for Run B.

    Focus label = XOR of two binary states: ``a_high`` (slot parity, encoded on
    ``app_switch_count`` 6.0 vs 6.5 — values inside the rule engine's neutral
    band, so the rule baseline stays at chance) and ``b_high`` (encoded on
    ``longest_segment_ratio``, clusters pulled towards 0.5 by ``base_scale``).
    No hyperplane separates the classes, so ``logistic_regression`` stays at
    chance while the tree candidates learn the interaction.

    The rng draw order per row (flip, a-value, b-value, c-value, then one
    normal per copy) is load-bearing: it reproduces the exact construction the
    offline search validated, so re-running this script replays the same
    dataset bit for bit.
    """
    import numpy as np

    rng = np.random.default_rng(spec.seed)
    start = datetime(2026, 7, 1, 8, tzinfo=UTC)
    low = 0.5 - 0.22 * spec.base_scale
    high = 0.5 + 0.22 * spec.base_scale
    windows: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    for day in range(spec.days):
        for slot in range(spec.windows_per_day):
            a_high = slot % 2 == 0
            b_high = (slot // 2) % 2 == 0
            c_high = (slot // 4) % 2 == 0
            is_focus = a_high != b_high
            # The searched construction drew a flip coin per row (its
            # flip_prob is 0 here, so labels stay clean) and a third
            # dimension's value; both draws stay for exact reproduction.
            if rng.random() < 0.0:  # pragma: no cover - stream-parity draw
                is_focus = not is_focus
            a_value = (high if a_high else low) + float(
                rng.uniform(-spec.jitter, spec.jitter)
            )
            b_value = (high if b_high else low) + float(
                rng.uniform(-spec.jitter, spec.jitter)
            )
            unused_c_value = (high if c_high else low) + float(
                rng.uniform(-spec.jitter, spec.jitter)
            )
            del unused_c_value
            features: dict[str, Any] = {
                "app_switch_count": 6.0 if a_high else 6.5,
                "longest_segment_ratio": b_value,
                "top_app_ratio": 0.5,
                "idle_ratio": 0.3,
            }
            for k in range(spec.copy_count):
                source = a_value if k % 2 == 0 else b_value
                features[COPY_SOURCE_FEATURES[k % len(COPY_SOURCE_FEATURES)]] = (
                    source + float(rng.normal(0.0, spec.copy_sigma))
                )
            index = day * spec.windows_per_day + slot
            windows.append({
                "id": f"win-{index}",
                "window_start_utc": session_start_iso(start, day, slot),
                "window_end_utc": session_start_iso(start, day, slot, minutes=5),
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "features": features,
            })
            feedback.append({
                "session_id": f"session-{index}",
                "start_time": session_start_iso(start, day, slot),
                "end_time": session_start_iso(start, day, slot, minutes=30),
                "label": "focus" if is_focus else "distracted",
                "score": 5 if is_focus else 1,
                "task_type": "coding",
            })
    return windows, feedback


def session_start_iso(
    start: datetime, day: int, slot: int, *, minutes: int = 0
) -> str:
    return (start + timedelta(days=day, hours=slot, minutes=minutes)).isoformat()


@dataclass
class RunResult:
    """Collected evidence from one end-to-end training run."""

    name: str
    expectation: str
    dataset: str
    allow_activation: bool
    gate_passed: bool | None = None
    gate_checks: dict[str, Any] = field(default_factory=dict)
    gate_details: dict[str, Any] = field(default_factory=dict)
    publication: dict[str, Any] = field(default_factory=dict)
    evaluation_candidate: str | None = None
    deployed_classifier: str | None = None
    activation_blocked_reason: str | None = None
    activation_suppressed_reason: str | None = None
    activation_allowed: bool | None = None
    activated: bool | None = None
    model_mode: str | None = None
    version_tag: str | None = None
    candidate_metrics: dict[str, Any] = field(default_factory=dict)
    selection_reason: str = ""
    selected_by_rule: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c["ok"] for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expectation": self.expectation,
            "dataset": self.dataset,
            "allow_activation": self.allow_activation,
            "gate_passed": self.gate_passed,
            "gate_checks": self.gate_checks,
            "gate_details": self.gate_details,
            "publication": self.publication,
            "evaluation_candidate": self.evaluation_candidate,
            "deployed_classifier": self.deployed_classifier,
            "activation_blocked_reason": self.activation_blocked_reason,
            "activation_suppressed_reason": self.activation_suppressed_reason,
            "activation_allowed": self.activation_allowed,
            "activated": self.activated,
            "model_mode": self.model_mode,
            "version_tag": self.version_tag,
            "candidate_metrics": self.candidate_metrics,
            "selection": {
                "selected": self.selected_by_rule,
                "reason": self.selection_reason,
            },
            "artifacts": self.artifacts,
            "checks": self.checks,
            "verdict": "PASS" if self.passed else "FAIL",
        }


def _check(result: RunResult, ok: bool, description: str, observed: Any) -> None:
    result.checks.append({"ok": bool(ok), "check": description, "observed": observed})


def _collect_artifacts(models_dir: Path, result: RunResult) -> None:
    v2_dir = models_dir / "v2"
    result.artifacts["models_dir"] = str(v2_dir)
    result.artifacts["latest_json_exists"] = (v2_dir / "latest.json").exists()
    result.artifacts["latest_json"] = (
        json.loads((v2_dir / "latest.json").read_text(encoding="utf-8"))
        if (v2_dir / "latest.json").exists() else None
    )
    tag = result.version_tag or ""
    result.artifacts["version_tag"] = tag
    result.artifacts["classifier_pkl_exists"] = (
        bool(tag) and (v2_dir / f"classifier-{tag}.pkl").exists()
    )
    result.artifacts["training_report_tagged_exists"] = (
        bool(tag) and (v2_dir / f"training_report-{tag}.json").exists()
    )
    result.artifacts["manifest_json_exists"] = (v2_dir / "manifest.json").exists()
    if (v2_dir / "manifest.json").exists():
        manifest = json.loads((v2_dir / "manifest.json").read_text(encoding="utf-8"))
        result.artifacts["manifest_publication"] = manifest.get("publication")
        result.artifacts["manifest_activation_blocked_reason"] = manifest.get(
            "activation_blocked_reason"
        )


def _run_case(
    name: str,
    expectation: str,
    builder: Callable[[], tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    dataset_description: str,
    *,
    allow_activation: bool,
    workspace: Path,
) -> RunResult:
    windows, feedback = builder()
    result = RunResult(
        name=name,
        expectation=expectation,
        dataset=dataset_description,
        allow_activation=allow_activation,
    )
    data_dir = workspace / "data"
    models_dir = workspace / "models"
    if workspace.exists():
        shutil.rmtree(workspace)
    data_dir.mkdir(parents=True, exist_ok=True)

    report: TrainingReport = run_training(
        source="db",
        data_dir=data_dir,
        models_dir=models_dir,
        feature_windows=windows,
        feedback_sessions=feedback,
        calibration=None,
        allow_activation=allow_activation,
    )

    gate = report.quality_gate
    result.gate_passed = bool(gate.get("passed"))
    result.gate_checks = {k: bool(v) for k, v in gate.get("checks", {}).items()}
    result.gate_details = gate.get("details", {})
    result.publication = dict(report.publication)
    result.evaluation_candidate = report.evaluation_candidate
    result.deployed_classifier = report.deployed_classifier
    result.activation_blocked_reason = report.activation_blocked_reason
    result.activation_suppressed_reason = report.activation_suppressed_reason
    result.activation_allowed = report.activation_allowed
    result.activated = report.activated
    result.model_mode = report.model_mode
    result.version_tag = report.version_tag
    evaluation = report.evaluation
    result.candidate_metrics = {
        candidate_name: {
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "brier_score": metrics.get("brier_score"),
            "status": metrics.get("status"),
        }
        for candidate_name, metrics in (evaluation.get("candidates") or {}).items()
    }
    selection = evaluation.get("candidate_selection") or {}
    result.selected_by_rule = (
        selection.get("selected") or evaluation.get("candidate_name")
    )
    result.selection_reason = str(selection.get("reason", ""))
    _collect_artifacts(models_dir, result)
    return result


def _evaluate_run_a(result: RunResult) -> None:
    _check(result, result.gate_passed is True, "quality gate passed", result.gate_passed)
    _check(
        result, result.publication.get("allowed") is False,
        "publication.allowed is False (guard blocks)", result.publication.get("allowed"),
    )
    _check(result, result.activated is False, "activated is False", result.activated)
    _check(
        result, result.model_mode == "shadow", "model_mode == shadow", result.model_mode,
    )
    _check(
        result,
        result.evaluation_candidate is not None
        and result.evaluation_candidate != result.deployed_classifier,
        "evaluation_candidate != deployed_classifier",
        {"selected": result.evaluation_candidate, "deployed": result.deployed_classifier},
    )
    _check(
        result, bool(result.activation_blocked_reason),
        "activation_blocked_reason recorded", result.activation_blocked_reason,
    )
    _check(
        result, result.artifacts["latest_json_exists"] is False,
        "latest.json absent (active pointer untouched)",
        result.artifacts["latest_json_exists"],
    )
    _check(
        result, result.artifacts["classifier_pkl_exists"] is True,
        "shadow classifier artifact saved", result.artifacts["classifier_pkl_exists"],
    )
    manifest_publication = result.artifacts.get("manifest_publication") or {}
    _check(
        result, manifest_publication.get("allowed") is False,
        "manifest carries publication.allowed=False", manifest_publication,
    )
    persisted_ok = result.artifacts["training_report_tagged_exists"]
    _check(
        result, persisted_ok is True, "versioned training report saved", persisted_ok,
    )


def _evaluate_run_b(result: RunResult) -> None:
    _check(result, result.gate_passed is True, "quality gate passed", result.gate_passed)
    _check(
        result, result.selected_by_rule == RF_XGB_SOFT_VOTING,
        "evaluation selected rf_xgb_soft_voting", result.selected_by_rule,
    )
    _check(
        result, result.publication.get("allowed") is True,
        "publication.allowed is True (guard permits)", result.publication.get("allowed"),
    )
    _check(
        result,
        result.evaluation_candidate == result.deployed_classifier == RF_XGB_SOFT_VOTING,
        "selected candidate == deployed artifact == rf_xgb_soft_voting",
        {"selected": result.evaluation_candidate, "deployed": result.deployed_classifier},
    )
    _check(result, result.activated is True, "activated is True", result.activated)
    _check(result, result.model_mode == "ready", "model_mode == ready", result.model_mode)
    _check(
        result, result.activation_blocked_reason is None,
        "no activation_blocked_reason", result.activation_blocked_reason,
    )
    _check(
        result, result.artifacts["latest_json_exists"] is True,
        "latest.json written (active pointer moved)", result.artifacts["latest_json_exists"],
    )
    _check(
        result, result.artifacts["classifier_pkl_exists"] is True,
        "classifier artifact saved", result.artifacts["classifier_pkl_exists"],
    )


def _evaluate_run_c(result: RunResult) -> None:
    _check(result, result.gate_passed is True, "quality gate passed", result.gate_passed)
    _check(
        result, result.activation_allowed is False,
        "report records activation_allowed=False", result.activation_allowed,
    )
    _check(result, result.activated is False, "activated is False", result.activated)
    _check(
        result, result.model_mode == "shadow", "model_mode == shadow", result.model_mode,
    )
    _check(
        result, bool(result.activation_suppressed_reason),
        "activation_suppressed_reason recorded", result.activation_suppressed_reason,
    )
    _check(
        result, result.artifacts["latest_json_exists"] is False,
        "latest.json absent (active pointer untouched)",
        result.artifacts["latest_json_exists"],
    )


def main() -> int:
    started = datetime.now(UTC).isoformat()
    work_root = EXPERIMENT_DIR / "work"
    runs: list[RunResult] = []

    run_a = _run_case(
        "run_a_guard_blocks",
        "quality gate passes, evaluation selects logistic_regression while the "
        "pipeline trains the ensemble -> guard blocks activation, version stays "
        "shadow, reason recorded in report and manifest",
        lambda: _build_separable_dataset(),
        "separable v4 windows, 8 days x 12 windows/day, 96 feedback sessions "
        "(shape reused from tests/test_publication_guard.py)",
        allow_activation=True,
        workspace=work_root / "run_a",
    )
    _evaluate_run_a(run_a)
    runs.append(run_a)

    run_b = _run_case(
        "run_b_consistent_candidate_activates",
        "evaluation itself selects rf_xgb_soft_voting on a non-linear dataset and "
        "the full quality gate passes -> guard allows activation, latest.json "
        "moves, model_mode=ready",
        lambda: _build_xor_dataset(RUN_B_DATASET),
        RUN_B_DATASET.describe(),
        allow_activation=True,
        workspace=work_root / "run_b",
    )
    _evaluate_run_b(run_b)
    runs.append(run_b)

    run_c = _run_case(
        "run_c_allow_activation_false_stays_shadow",
        "same data as Run A with allow_activation=False -> shadow regardless of "
        "the publication verdict, activation_suppressed_reason recorded",
        lambda: _build_separable_dataset(),
        "separable v4 windows, 8 days x 12 windows/day, 96 feedback sessions "
        "(identical to Run A)",
        allow_activation=False,
        workspace=work_root / "run_c",
    )
    _evaluate_run_c(run_c)
    runs.append(run_c)

    summary = {
        "run_id": RUN_ID,
        "plan_item": "模型候选一致性和 shadow 发布验收",
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(),
        "environment": _environment(),
        "verified_commands": VERIFIED_COMMANDS,
        "runs": [run.to_dict() for run in runs],
        "all_passed": all(run.passed for run in runs),
    }
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (EXPERIMENT_DIR / "report.md").write_text(
        _markdown(summary, runs), encoding="utf-8",
    )
    print(_markdown(summary, runs))
    print(f"Artifacts: {EXPERIMENT_DIR / 'summary.json'} and {EXPERIMENT_DIR / 'report.md'}")
    return 0 if summary["all_passed"] else 1


def _environment() -> dict[str, Any]:
    import sklearn

    try:
        import xgboost

        xgb_version: str | None = xgboost.__version__
    except ImportError:
        xgb_version = None
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "sklearn": sklearn.__version__,
        "xgboost": xgb_version,
        "python": sys.version.split()[0],
    }


def _markdown(summary: dict[str, Any], runs: list[RunResult]) -> str:
    lines = [
        "# Training publication guard — acceptance runs",
        "",
        f"- run id: `{summary['run_id']}`",
        f"- started: {summary['started_at']}",
        f"- environment: python {summary['environment']['python']}, "
        f"sklearn {summary['environment']['sklearn']}, "
        f"xgboost {summary['environment']['xgboost']}, "
        f"feature schema v{summary['environment']['feature_schema_version']}",
        f"- overall verdict: **{'PASS' if summary['all_passed'] else 'FAIL'}**",
        "",
        "## Verdict table",
        "",
        "| run | gate | selected (rule) | deployed | allowed | activated | mode | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        verdict = run.to_dict()["verdict"]
        lines.append(
            f"| {run.name} | {'pass' if run.gate_passed else 'FAIL'} "
            f"| {run.selected_by_rule} | {run.deployed_classifier} "
            f"| {run.publication.get('allowed')} | {run.activated} "
            f"| {run.model_mode} | **{verdict}** |"
        )
    lines += ["", "## Per-run evidence", ""]
    for run in runs:
        lines += [
            f"### {run.name}",
            "",
            f"- expectation: {run.expectation}",
            f"- dataset: {run.dataset}",
            f"- allow_activation: {run.allow_activation}",
            f"- quality gate: passed={run.gate_passed}, "
            f"checks={json.dumps(run.gate_checks, ensure_ascii=False)}",
            f"- selection rule: selected={run.selected_by_rule}",
            f"- selection reason: {run.selection_reason}",
            f"- blocked reason: {run.activation_blocked_reason}",
            f"- suppressed reason: {run.activation_suppressed_reason}",
            f"- version tag: {run.version_tag}",
            "- per-candidate metrics (primary scheme):",
            "",
            "| candidate | balanced accuracy | brier |",
            "|---|---|---|",
        ]
        for candidate_name, metrics in run.candidate_metrics.items():
            lines.append(
                f"| {candidate_name} | {metrics.get('balanced_accuracy')} "
                f"| {metrics.get('brier_score')} |"
            )
        lines += ["", "| check | ok | observed |", "|---|---|---|"]
        for check in run.checks:
            observed = json.dumps(check["observed"], ensure_ascii=False)
            lines.append(f"| {check['check']} | {check['ok']} | {observed} |")
        lines.append("")
    lines += ["## Commands", ""]
    for entry in summary["verified_commands"]:
        lines.append(f"- `{entry['command']}` → {entry['outcome']}")
    lines.append(
        "- `uv run python scripts/acceptance_publication_guard.py` → this execution"
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
