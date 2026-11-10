"""Reproducible synthetic pretraining, kept separate from personal quality gates.

A base release is a cold-start behavioural classifier, not a language model.
All labels in this module are simulated. Held-out synthetic users and future
windows measure the simulator's generalisation, not performance on real people.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mindflow.domain.feature_schema import FEATURE_SCHEMA_VERSION, V2_FEATURE_NAMES
from mindflow.train.candidates import rule_probabilities
from mindflow.train.evaluation import classification_metrics
from mindflow.train.models.clustering import BehaviorClustering
from mindflow.train.models.ensemble import EnsembleClassifier
from mindflow.train.models.manager import ModelManager

BASE_VERSION = "base_v4_20261111_r2"
SERIALIZATION_PACKAGES = ("scikit-learn", "numpy", "scipy", "joblib", "xgboost")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _matrix(rows: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
    return np.asarray(
        [[float(row["features"][name]) for name in V2_FEATURE_NAMES] for row in rows],
        dtype=np.float64,
    )


def _session_weights(rows: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
    """One simulated session has one unit of weight; rescale for tree fitting."""
    counts = Counter(str(row["session_id"]) for row in rows)
    weights = np.asarray([1.0 / counts[str(row["session_id"])] for row in rows])
    return np.asarray(weights / weights.mean(), dtype=np.float64)


def score_predictions(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray[Any, Any],
) -> dict[str, Any]:
    """Score fixed predictions, with class-specific denominators and abstention."""
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int_)
    predicted = np.asarray(probabilities >= 0.5, dtype=np.int_)
    result = classification_metrics(labels, predicted, probabilities)
    result.pop("calibration", None)
    result["macro_f1"] = round(float(f1_score(labels, predicted, average="macro")), 6)
    focus, distract = labels == 1, labels == 0
    result["normal_work_false_alert_rate"] = (
        round(float(np.mean(predicted[focus] == 0)), 6) if focus.any() else None
    )
    result["distraction_recall"] = (
        round(float(np.mean(predicted[distract] == 0)), 6) if distract.any() else None
    )
    result["focus_count"] = int(focus.sum())
    result["distracted_count"] = int(distract.sum())
    covered = (probabilities < 0.35) | (probabilities > 0.65)
    result["abstention"] = {
        "band": [0.35, 0.65],
        "coverage": round(float(covered.mean()), 6),
        "covered_count": int(covered.sum()),
        "accuracy_on_covered": (
            round(float(np.mean(predicted[covered] == labels[covered])), 6)
            if covered.any()
            else None
        ),
        "note": "Diagnostic operating point; automatic intervention has additional guards.",
    }
    return result


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    base_p: np.ndarray[Any, Any],
    rule_p: np.ndarray[Any, Any],
    *,
    seed: int,
    repeats: int = 500,
) -> dict[str, Any]:
    """Paired persona bootstrap: correlated windows are not independent people."""
    counts: dict[str, np.ndarray[Any, Any]] = defaultdict(lambda: np.zeros(6))
    for row, base, rule in zip(rows, base_p, rule_p, strict=True):
        truth = int(row["label"])
        values = counts[str(row["persona_id"])]
        values[truth] += 1
        values[2 + truth] += float((base >= 0.5) == truth)
        values[4 + truth] += float((rule >= 0.5) == truth)
    table = np.stack(list(counts.values()))
    rng = np.random.default_rng(seed)
    base_scores, deltas = [], []
    for _ in range(repeats):
        totals = table[rng.integers(0, len(table), len(table))].sum(axis=0)
        if np.any(totals[:2] == 0):
            continue
        base_score = float(np.mean(totals[2:4] / totals[:2]))
        base_scores.append(base_score)
        deltas.append(base_score - float(np.mean(totals[4:6] / totals[:2])))
    return {
        "unit": "synthetic_persona",
        "unit_count": len(counts),
        "resamples": repeats,
        "seed": seed,
        "confidence": 0.95,
        "base_balanced_accuracy_ci": np.quantile(base_scores, [0.025, 0.975]).tolist(),
        "base_minus_rule_balanced_accuracy_ci": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "interpretation": "Conditional on this simulator; not a real-user confidence interval.",
    }


def _export_corpus(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Gzip rows include provenance, so downstream users can reproduce all splits."""
    with gzip.open(run_dir / "synthetic_windows.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    metadata_fields = [
        "id",
        "persona_id",
        "role",
        "scenario",
        "session_id",
        "split",
        "label",
        "label_source",
        "window_start_utc",
        "window_end_utc",
        "local_hour",
        "timezone",
    ]
    with gzip.open(
        run_dir / "synthetic_features.csv.gz", "wt", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata_fields + list(V2_FEATURE_NAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: row.get(name) for name in metadata_fields},
                    **{name: row["features"][name] for name in V2_FEATURE_NAMES},
                }
            )


def _export_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    predictions: dict[str, np.ndarray[Any, Any]],
) -> None:
    names = ["id", "persona_id", "role", "scenario", "session_id", "split", "label"]
    with gzip.open(path, "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names + list(predictions))
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    **{name: row[name] for name in names},
                    **{
                        name: round(float(values[index]), 8) for name, values in predictions.items()
                    },
                }
            )


def train_base_model(
    run_dir: Path,
    *,
    seed: int = 20261110,
    personas_per_role: int = 6,
    days: int = 28,
    windows_per_day: int = 48,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate, audit, fit once, evaluate fixed holdouts, and optionally package.

    No personal database, API key or production-model pointer is read or written.
    A test set never chooses hyperparameters, calibration or a deployment model.
    """
    from mindflow.train.base_synthetic import audit_base_corpus, generate_base_corpus

    run_dir = run_dir.resolve()
    report_path = run_dir / "base_model_report.json"
    if report_path.exists():
        raise ValueError("Experiment already completed; choose a fresh run directory")
    if bundle_dir is not None and bundle_dir.exists():
        raise ValueError("Release directory already exists; choose a new release destination")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("[base] Generating multi-role synthetic telemetry", flush=True)
    corpus = generate_base_corpus(
        seed=seed,
        personas_per_role=personas_per_role,
        days=days,
        windows_per_day=windows_per_day,
    )
    audit = audit_base_corpus(corpus)
    _write_json(run_dir / "dataset_card.json", corpus.metadata)
    _write_json(run_dir / "data_audit.json", audit)
    if not audit.get("checks") or not all(audit["checks"].values()):
        raise ValueError(f"Synthetic corpus failed audit: {audit.get('checks')}")
    rows = corpus.windows
    _export_corpus(run_dir, rows)
    labelled = [row for row in rows if row["label"] in (0, 1)]
    train_rows = [row for row in labelled if row["split"] == "train"]
    if not train_rows:
        raise ValueError("No labelled training rows")
    # Global UTC cutoff, not a local calendar day: calibration cannot observe
    # future holdout dates even when personas live in different time zones.
    cutoff = max(str(row["window_end_utc"]) for row in train_rows)
    calibration_rows = [
        row
        for row in labelled
        if row["split"] == "calibration" and str(row["window_end_utc"]) <= cutoff
    ]
    if len(calibration_rows) < 20 or len({row["label"] for row in calibration_rows}) != 2:
        raise ValueError("Need both classes and at least 20 independent calibration rows")
    train_personas = {row["persona_id"] for row in train_rows}
    calibration_personas = {row["persona_id"] for row in calibration_rows}
    if train_personas & calibration_personas:
        raise ValueError("Training and calibration personas overlap")
    x_train, x_calibration = _matrix(train_rows), _matrix(calibration_rows)
    y_train = np.asarray([row["label"] for row in train_rows], dtype=np.int_)
    y_calibration = np.asarray([row["label"] for row in calibration_rows], dtype=np.int_)
    weights = _session_weights(train_rows)
    print(
        f"[base] Fitting {len(train_rows)} windows; calibration={len(calibration_rows)}", flush=True
    )
    manager = ModelManager(run_dir / "models" / "v2", use_ensemble=True)
    if not manager.use_ensemble or not isinstance(manager.classifier, EnsembleClassifier):
        raise ValueError("Base release requires the locked XGBoost extra")
    classifier = manager.classifier
    classifier.fit(x_train, y_train, list(V2_FEATURE_NAMES), sample_weight=weights)
    raw_calibration = classifier.predict_proba(x_calibration)[:, 1]
    calibrator = LogisticRegression(max_iter=1000, random_state=seed)
    calibrator.fit(
        raw_calibration.reshape(-1, 1),
        y_calibration,
        sample_weight=_session_weights(calibration_rows),
    )
    classifier.calibration, classifier.calibrator = "sigmoid", calibrator
    linear = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=seed))
    linear.fit(x_train, y_train, logisticregression__sample_weight=weights)
    # Cluster only training data, with scalable KMeans. HMM remains outside
    # production; no synthetic sequence is presented as a fitted temporal model.
    manager.clustering = BehaviorClustering(method="kmeans")
    manager.clustering.fit(x_train)
    evaluations: dict[str, Any] = {}
    session_output: list[dict[str, Any]] = []
    for split in ("test_persona", "test_future"):
        test = [row for row in labelled if row["split"] == split]
        if not test:
            raise ValueError(f"No held-out labelled rows for {split}")
        if split == "test_persona" and (
            {row["persona_id"] for row in test} & (train_personas | calibration_personas)
        ):
            raise ValueError("Unseen-persona holdout overlaps fitted personas")
        if split == "test_future":
            # Some timezone offsets make the first local future day overlap the
            # global fit horizon; exclude and report those boundary windows.
            test = [row for row in test if str(row["window_start_utc"]) > cutoff]
        x_test = _matrix(test)
        scaled = classifier.scaler.transform(x_test)
        context = x_test[:, list(V2_FEATURE_NAMES).index("task_type_code")]
        context_rule = np.where(
            np.isin(context, (1.0, 2.0, 3.0, 4.0)),
            0.8,
            np.where(np.isin(context, (5.0, 6.0)), 0.2, 0.5),
        )
        probabilities = {
            "rule_engine": rule_probabilities(x_test, list(V2_FEATURE_NAMES)),
            "task_context_rule": context_rule,
            "logistic_regression": linear.predict_proba(x_test)[:, 1],
            "random_forest": classifier.rf_model.predict_proba(scaled)[:, 1],
            "xgboost": classifier.xgb_model.predict_proba(scaled)[:, 1],
            "base_rf_xgb_calibrated": classifier.predict_proba(x_test)[:, 1],
        }
        score = {name: score_predictions(test, proba) for name, proba in probabilities.items()}
        by_scenario: dict[str, Any] = {}
        for scenario in sorted({str(row["scenario"]) for row in test}):
            mask = np.asarray([row["scenario"] == scenario for row in test])
            subset = [row for row, keep in zip(test, mask, strict=True) if keep]
            # Single-class slices report the denominators; BA is not interpreted
            # there. The false-alert and recall rates have explicit denominators.
            by_scenario[scenario] = {
                "sample_count": len(subset),
                "focus_count": sum(row["label"] == 1 for row in subset),
                "normal_work_false_alert_rate": (
                    float(
                        np.mean(
                            probabilities["base_rf_xgb_calibrated"][mask][
                                np.asarray([row["label"] == 1 for row in subset])
                            ]
                            < 0.5
                        )
                    )
                    if any(row["label"] == 1 for row in subset)
                    else None
                ),
                "distraction_recall": (
                    float(
                        np.mean(
                            probabilities["base_rf_xgb_calibrated"][mask][
                                np.asarray([row["label"] == 0 for row in subset])
                            ]
                            < 0.5
                        )
                    )
                    if any(row["label"] == 0 for row in subset)
                    else None
                ),
                "mean_focus_probability": float(
                    np.mean(probabilities["base_rf_xgb_calibrated"][mask])
                ),
            }
        by_role: dict[str, Any] = {}
        for role in sorted({str(row["role"]) for row in test}):
            mask = np.asarray([row["role"] == role for row in test])
            subset = [row for row, keep in zip(test, mask, strict=True) if keep]
            by_role[role] = score_predictions(subset, probabilities["base_rf_xgb_calibrated"][mask])
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(test):
            grouped[str(row["session_id"])].append(index)
        for session_id, indices in grouped.items():
            labels = {int(test[index]["label"]) for index in indices}
            if len(labels) != 1:
                continue
            session_output.append(
                {
                    "session_id": session_id,
                    "split": split,
                    "label": labels.pop(),
                    "persona_id": test[indices[0]]["persona_id"],
                    "base_probability": float(
                        np.mean(probabilities["base_rf_xgb_calibrated"][indices])
                    ),
                    "rule_probability": float(np.mean(probabilities["rule_engine"][indices])),
                }
            )
        evaluations[split] = {
            "models": score,
            "by_scenario": by_scenario,
            "by_role": by_role,
            "paired_persona_bootstrap": _cluster_bootstrap(
                test,
                probabilities["base_rf_xgb_calibrated"],
                probabilities["rule_engine"],
                seed=seed,
            ),
            "persona_count": len({row["persona_id"] for row in test}),
            "session_count": len(grouped),
        }
        _export_predictions(run_dir / f"predictions_{split}.csv.gz", test, probabilities)
        print(
            f"[base] {split}: {json.dumps(score['base_rf_xgb_calibrated'], ensure_ascii=False)}",
            flush=True,
        )
    _write_json(run_dir / "session_predictions.json", session_output)
    names = manager.save_all(
        activate=False,
        manifest={
            "source": "synthetic_base",
            "model_mode": "base",
            "synthetic_only": True,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "personal_quality_gate_passed": False,
        },
    )
    saved_tag = names["classifier"].removeprefix("classifier-").removesuffix(".pkl")
    if not manager.load_version(saved_tag):
        raise ValueError("Saved base model failed signed round-trip load")
    # Microbenchmark the actual reload, with warmed 1/24-window batches.
    latency: dict[str, Any] = {}
    for batch in (1, 24):
        sample = x_train[:batch]
        manager.classifier.predict_proba(sample)
        elapsed = []
        for _ in range(30):
            begin = time.perf_counter()
            manager.classifier.predict_proba(sample)
            elapsed.append((time.perf_counter() - begin) * 1000)
        latency[str(batch)] = {
            "repeats": 30,
            "p50_ms": float(np.median(elapsed)),
            "p95_ms": float(np.quantile(elapsed, 0.95)),
        }
    report: dict[str, Any] = {
        "version": BASE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "synthetic_base",
        "synthetic_only": True,
        "model_mode": "base",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(V2_FEATURE_NAMES),
        "real_user_count": 0,
        "real_feedback_sessions": 0,
        "personal_quality_gate_passed": False,
        "recipe": {
            "seed": seed,
            "personas_per_role": personas_per_role,
            "days": days,
            "windows_per_day": windows_per_day,
        },
        "data_audit": audit,
        "total_windows": len(rows),
        "train_windows": len(train_rows),
        "calibration_windows": len(calibration_rows),
        "excluded_from_supervision": len(rows) - len(labelled),
        "train_cutoff_utc": cutoff,
        "protocol": {
            "model_choice": "Fixed RF+XGBoost, calibrated on separate personas before test",
            "hyperparameter_search": False,
            "calibration": "Unseen personas, not later than the global training cutoff",
            "test_persona": "Persons never used by fitting or calibration",
            "test_future": "Seen training persons; windows strictly later than all fitted data",
            "test_reused_for_tuning": False,
            "unit_of_uncertainty": "synthetic_persona",
            "interpretation": "Simulator evaluation; real-user and product validation pending",
        },
        "evaluations": evaluations,
        "inference_latency": latency,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "packages": {name: importlib.metadata.version(name) for name in SERIALIZATION_PACKAGES},
        "saved_models": names,
        "training_seconds": time.perf_counter() - started,
    }
    _write_json(report_path, report)
    write_base_report(run_dir / "report.md", report)
    if bundle_dir is not None:
        release_base_bundle(bundle_dir, manager, report)
    return report


def write_base_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MindFlow 基础行为模型：可复现实验报告",
        "",
        "训练和评测已实际执行；数据与标签均为合成，真实参与者和反馈均为 0。",
        "指标描述合成场景下的分类能力；商业软件效果、真实人群收益另行评估。",
        "",
        f"- 版本：{report['version']}；schema v{report['feature_schema_version']}（28 维）。",
        f"- 总窗口：{report['total_windows']:,}；训练：{report['train_windows']:,}；"
        f"独立校准：{report['calibration_windows']:,}。",
        f"- 未纳入监督：{report['excluded_from_supervision']:,}（休息、离开、信息不足等）。",
        f"- 配置：{json.dumps(report['recipe'], ensure_ascii=False)}。",
        "",
        "## 固定测试集的算法对照",
        "",
        "BA 为 Balanced accuracy。误提醒代理率指专注窗口被判为分心的比例。",
        "实际通知还受置信度、工作态、节流等条件控制。",
        "",
        "| 测试划分 | 方法 | 窗口数 | BA | Macro F1 | Brier↓ | 误提醒代理率↓ | 分心召回↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, evaluation in report["evaluations"].items():
        for name, metrics in evaluation["models"].items():
            lines.append(
                f"| {split} | {name} | {metrics['sample_count']} | "
                f"{metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{metrics['brier_score']:.4f} | {metrics['normal_work_false_alert_rate']:.2%} | "
                f"{metrics['distraction_recall']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## 已知困难场景（完整保留测试结果）",
            "",
            "| 划分 | 场景 | 窗口数 | 指标 | 结果 |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for split, evaluation in report["evaluations"].items():
        for scenario, metric in (
            ("game_testing", "normal_work_false_alert_rate"),
            ("customer_support", "normal_work_false_alert_rate"),
            ("goal_drift", "distraction_recall"),
            ("off_topic_reading", "distraction_recall"),
        ):
            item = evaluation["by_scenario"].get(scenario)
            if item is not None and item[metric] is not None:
                lines.append(
                    f"| {split} | {scenario} | {item['sample_count']} | "
                    f"{metric} | {item[metric]:.2%} |"
                )
    lines.extend(
        [
            "",
            "相同软件及输入轨迹可能服务于不同目标；这类任务需要本人目标或纠正反馈。",
            "合成身份留出仍共享场景配方，未见模板、真实用户与实际通知效果尚待独立验证。",
            "类别规则只是透明启发式基线，并不代表 Rize、RescueTime 或其他产品。",
        ]
    )
    lines.extend(
        [
            "",
            "## 实验解释与复算",
            "",
            "训练、校准、未知用户测试按模拟身份隔离；未来日期测试严格晚于训练和校准的截止时刻。",
            "测试集没有参与超参或模型选择，不能将最佳对照算法事后换成已发布模型。",
            "每组结果、场景切片、按角色切片、按模拟身份重采样的区间与全部预测见同目录 JSON/CSV。",
            "模型只接收生产协议的 28 维特征，身份、情境名和软件名仅用于覆盖审计。",
            "软件名称仅用于构造不同输入活动与覆盖审计；类别映射会丢失任务意图，因此相同观测仍可能对应不同真实状态。",
            "计划休息与信息不足数据未伪装成分心或真实用户反馈。",
            "",
            "## 使用与后续个性化",
            "",
            "软件优先加载通过个人质量门的本地模型；否则使用标明来源的合成基础模型。",
            "现有 RF/XGBoost 采用个人反馈重训及独立校准进行个性化，不使用语言模型式的权重微调。",
            "个人训练沿用日期前向评估、唯一会话计数、发布一致性检查和 shadow-only 自动训练策略。",
            "正式对照试用将进一步测量正常任务误打扰、真实偏离召回、弃权覆盖、延迟及用户负担。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def release_base_bundle(
    bundle_dir: Path,
    manager: ModelManager,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Export only redistributable artifacts; never export a machine signing key."""
    bundle_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "classifier": manager.classifier.to_dict(),
        "clustering": manager.clustering.to_dict(),
        "hmm": {
            "transition_matrix": None,
            "state_names": list(manager.hmm.state_names),
            "n_states": manager.hmm.n_states,
            "is_fitted": False,
        },
    }
    files: dict[str, Any] = {}
    for kind, payload in artifacts.items():
        artifact = bundle_dir / f"{kind}-{BASE_VERSION}.pkl"
        joblib.dump(payload, artifact, compress=3)
        raw = artifact.read_bytes()
        files[kind] = {
            "name": artifact.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    manifest = {
        "version": BASE_VERSION,
        "source": "synthetic_base",
        "synthetic_only": True,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(V2_FEATURE_NAMES),
        "personal_quality_gate_passed": False,
        "packages": report["packages"],
        "recipe": report["recipe"],
        "files": files,
    }
    _write_json(bundle_dir / "bundle.json", manifest)
    write_base_report(bundle_dir / "model_card.md", report)
    print(
        "[base] Release manifest SHA256:",
        hashlib.sha256((bundle_dir / "bundle.json").read_bytes()).hexdigest(),
        flush=True,
    )
    return manifest
