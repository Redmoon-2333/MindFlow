"""CLI entry point for G006 evaluation harness.

Usage:
    # Run rule engine baseline against all 30 scenarios (default)
    python -m mindflow.eval

    # Run rule engine + mock panel (pipeline verification)
    python -m mindflow.eval --mode both

    # Run real DeepSeek panel (requires API key + --yes confirmation)
    python -m mindflow.eval --mode panel --live --yes

    # Run all three modes
    python -m mindflow.eval --mode both --live --yes

Cost estimate (--live):
    ~6-12 calls per scenario × 30 scenarios = 180-360 DeepSeek API calls.
    At ~¥1/1M tokens (deepseek-chat), ~¥0.5-2 total.
    Add --yes to skip confirmation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mindflow.eval.adapters import MockPanelGateway, panel_analyzer, rule_engine_analyzer
from mindflow.eval.runner import compare, run_eval
from mindflow.eval.scenarios import ALL_SCENARIOS, validate_all_scenarios


def _fmt_pct(v: float) -> str:
    """Format a float in [0,1] as percentage string."""
    return f"{v:.1%}"


def _print_summary_report(report: Any, label: str) -> None:
    """Print a formatted summary table for an EvalReport."""
    print()
    print(f"  {'=' * 50}")
    print(f"  {label}")
    print(f"  {'=' * 50}")
    print(f"  Top-1 命中率:    {_fmt_pct(report.top1_accuracy)} ({report.hits}/{report.total})")
    print(f"  平均 Jaccard:    {report.mean_jaccard:.3f}")
    print(f"  Technique 匹配:  {_fmt_pct(report.technique_accuracy)}")
    print()


def _print_comparison(cr: Any) -> None:
    """Print a formatted comparison table."""
    print()
    print(f"  {'=' * 50}")
    print(f"  对比: [{cr.baseline_name}] vs [{cr.panel_name}]")
    print(f"  {'=' * 50}")
    print(f"  Top-1 Δ:        {cr.top1_delta:+.1%}")
    print(f"  Jaccard Δ:      {cr.jaccard_delta:+.3f}")
    print(f"  Technique Δ:    {cr.technique_delta:+.1%}")
    print(f"  {cr.baseline_name} 胜: {cr.baseline_wins}")
    print(f"  {cr.panel_name} 胜: {cr.panel_wins}")
    print(f"  平局:           {cr.ties}")
    print()


def _print_detail(report: Any) -> None:
    """Print per-scenario detail."""
    print(f"  {'ID':<10} {'预期Top':<12} {'实际Top':<12} {'Top-1命中':<10} {'Jaccard':<8} {'Technique':<10}")
    print(f"  {'-' * 62}")
    for r in report.scenario_results:
        exp_top = r.expected_types[0].value if r.expected_types else "-"
        pred_top = r.predicted_types[0].value if r.predicted_types else "-"
        hit = "✓" if r.top1_hit else "✗"
        tech = "✓" if r.technique_match else "✗"
        print(f"  {r.scenario_id:<10} {exp_top:<12} {pred_top:<12} {hit:<10} {r.jaccard:<8.3f} {tech:<10}")
    print()


def _save_report(report: Any, path: Path) -> None:
    """Save an EvalReport or ComparisonReport as JSON."""
    data = {
        "type": "eval_report",
        "analyzer_name": getattr(report, "analyzer_name", "comparison"),
        "timestamp": datetime.now(UTC).isoformat(),
        "total": getattr(report, "total", None),
        "top1_accuracy": getattr(report, "top1_accuracy", None),
        "mean_jaccard": getattr(report, "mean_jaccard", None),
        "technique_accuracy": getattr(report, "technique_accuracy", None),
    }
    # Include per-scenario results
    if hasattr(report, "scenario_results"):
        data["scenarios"] = [
            {
                "scenario_id": r.scenario_id,
                "expected_types": [t.value for t in r.expected_types],
                "predicted_types": [t.value for t in r.predicted_types],
                "expected_technique": r.expected_technique.value if r.expected_technique else None,
                "predicted_technique": r.predicted_technique.value if r.predicted_technique else None,
                "top1_hit": r.top1_hit,
                "jaccard": r.jaccard,
                "technique_match": r.technique_match,
                "predicted_source": r.predicted_source,
            }
            for r in report.scenario_results
        ]

    # Include comparison data
    if hasattr(report, "top1_delta"):
        data["type"] = "comparison_report"
        data["baseline_name"] = getattr(report, "baseline_name", None)
        data["panel_name"] = getattr(report, "panel_name", None)
        data["top1_delta"] = report.top1_delta
        data["jaccard_delta"] = report.jaccard_delta
        data["technique_delta"] = report.technique_delta
        data["baseline_wins"] = report.baseline_wins
        data["panel_wins"] = report.panel_wins
        data["ties"] = report.ties

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [保存] {path}")


async def _run_rule_engine(output_dir: Path) -> Any:
    """Run rule engine against all scenarios."""
    print("\n  [规则引擎] 评估中...")
    report = await run_eval(
        rule_engine_analyzer,
        ALL_SCENARIOS,
        analyzer_name="rule_engine",
    )
    _print_summary_report(report, "规则引擎评估结果")
    _print_detail(report)
    _save_report(report, output_dir / f"report_rule_engine_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json")
    return report


async def _run_panel_mock(output_dir: Path) -> Any:
    """Run mock panel against all scenarios."""
    print("\n  [Mock Panel] 评估中...")
    gateway = MockPanelGateway()
    analyzer = panel_analyzer(gateway)
    report = await run_eval(analyzer, ALL_SCENARIOS, analyzer_name="panel_mock")
    _print_summary_report(report, "Mock Panel 评估结果")
    _print_detail(report)
    _save_report(report, output_dir / f"report_panel_mock_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json")
    return report


async def _run_panel_live(output_dir: Path) -> Any:
    """Run real DeepSeek panel against all scenarios."""
    from mindflow.agents.llm_gateway import DeepSeekGateway

    print("\n  [DeepSeek Panel] 评估中（30 场景，预计 180-360 次 API 调用）...")
    gateway = DeepSeekGateway()
    analyzer = panel_analyzer(gateway)
    report = await run_eval(analyzer, ALL_SCENARIOS, analyzer_name="panel_deepseek")
    _print_summary_report(report, "DeepSeek Panel 评估结果")
    _print_detail(report)
    _save_report(report, output_dir / f"report_panel_deepseek_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json")
    return report


async def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MindFlow G006 评估套件 — 规则引擎 vs 专家团对比评估",
    )
    parser.add_argument(
        "--mode",
        choices=["rule", "panel", "both"],
        default="rule",
        help="评估模式: rule=仅规则引擎, panel=仅专家团, both=两者+对比",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="使用真实 DeepSeek API（默认使用 MockGateway 管线验证）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过实时 API 调用的成本确认",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        dest="output_dir",
        help="报告输出目录（默认: data/eval_reports）",
    )

    args = parser.parse_args()

    # Validate scenarios first
    print()
    print("=" * 60)
    print("  MindFlow G006 评估套件")
    print("=" * 60)

    issues = validate_all_scenarios()
    if issues:
        print(f"\n  [警告] 场景验证发现 {len(issues)} 个问题:")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"\n  [通过] {len(ALL_SCENARIOS)} 个场景验证无误")

    # Resolve output directory
    cwd = Path.cwd().resolve()
    project_root = cwd
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists():
            project_root = parent
            break
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "eval_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Live mode confirmation
    if args.live and not args.yes:
        print()
        print("  [成本提示] 使用 DeepSeek --live 模式:")
        print("    - 30 场景 × ~6-12 次调用/场景 = 180-360 次 API 调用")
        print("    - deepseek-chat 约 ¥1/1M tokens")
        print("    - 预估总成本: ¥0.5-2")
        print()
        confirm = input("  确认继续? [y/N] ").strip().lower()
        if confirm != "y":
            print("  已取消。使用 --yes 跳过确认。")
            sys.exit(0)

    # Run evaluations

    rule_report = None
    panel_mock_report = None
    panel_live_report = None

    if args.mode in ("rule", "both"):
        rule_report = await _run_rule_engine(output_dir)

    if args.mode in ("panel", "both"):
        if args.live:
            panel_live_report = await _run_panel_live(output_dir)
        else:
            panel_mock_report = await _run_panel_mock(output_dir)

    # Comparison
    if args.mode == "both" and rule_report is not None:
        panel_report = panel_live_report or panel_mock_report
        if panel_report is not None:
            cr = compare(
                rule_report,
                panel_report,
                baseline_name="rule_engine",
                panel_name="panel_deepseek" if args.live else "panel_mock",
            )
            _print_comparison(cr)
            _save_report(cr, output_dir / f"comparison_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json")

    print()
    print("  Done.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
